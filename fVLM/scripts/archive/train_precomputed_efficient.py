#!/usr/bin/env python3
"""
Efficient Joint Training on Precomputed 202K Videos.

Uses precomputed VAE latents for efficient training with:
- Resumable checkpoints (1-2 hour chunks)
- wandb logging with nice plots
- ETA tracking
- Multi-fine iterations (coarse → fine₁ → fine₂)

Training modes:
- reconstruction_only: Uses all 202K videos (no captions needed)
- joint: Uses videos with captions in metadata (for caption+reconstruction)

Success metric: loss_fine < loss_coarse (by 5-15%+)
"""

import sys
import os
import time
import json
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from transformers import AutoTokenizer
import numpy as np
import random
import argparse

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Training
    "max_hours": 1.5,  # 1.5 hour chunks (resumable)
    "batch_size": 4,
    "grad_accum": 4,  # Effective batch = 16
    "num_frames": 16,  # Use 16 of 24 available frames
    "learning_rate": 3e-5,
    "lambda_recon": 0.5,
    "lambda_coarse": 1.0,
    "warmup_steps": 200,
    "max_grad_norm": 1.0,

    # Multi-fine iterations
    "fine_iterations": 2,  # coarse → fine₁ → fine₂

    # Logging
    "log_interval": 50,
    "save_interval": 500,
    "plot_interval": 200,

    # Data (v2 format: includes both frames and latents)
    "data_dir": "/mnt/d/projects/fVLM/data/precomputed_24f_v2",
    "num_workers": 4,
    "pin_memory": True,

    # Training mode
    "mode": "reconstruction_only",  # or "joint" if captions available

    # Checkpointing
    "output_dir": "outputs/precomputed_efficient",
    "resume_checkpoint": None,  # Set to path to resume

    # wandb
    "wandb_project": "foveated-vlm-efficient",
    "wandb_run_name": None,  # Auto-generated if None
}


class PrecomputedDataset(Dataset):
    """Dataset loading precomputed VAE latents and frames."""

    def __init__(self, data_dir: str, num_frames: int = 16, mode: str = "reconstruction_only"):
        self.data_dir = Path(data_dir)
        self.latent_dir = self.data_dir / "latents"
        self.num_frames = num_frames
        self.mode = mode

        # Find all .pt files
        self.files = sorted(self.latent_dir.glob("*.pt"), key=lambda x: int(x.stem))
        print(f"Found {len(self.files)} precomputed videos")

        # Load metadata for captions (if available)
        self.captions = {}
        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                for sample in meta.get("samples", []):
                    self.captions[sample["video_id"]] = sample.get("caption", "")
            print(f"Loaded {len(self.captions)} captions from metadata")

        # Filter for joint mode (need captions)
        if mode == "joint":
            self.files = [f for f in self.files if f.stem in self.captions]
            print(f"Joint mode: {len(self.files)} videos with captions")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pt_path = self.files[idx]
        video_id = pt_path.stem

        # Load precomputed data (v2 format: dict with frames and latents)
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        # Handle both v1 (tensor only) and v2 (dict) formats
        if isinstance(data, dict):
            # v2 format: frames: [T, 3, 256, 256], latents: [T, 4, 32, 32]
            frames = data["frames"].float()  # Convert from bfloat16
            latents = data["latents"]
        else:
            # v1 format: latents only (skip these files)
            raise ValueError(f"v1 format file (latents only): {pt_path}. Need v2 format with frames.")

        # Sample frames if we have more than needed
        T = frames.shape[0]
        if T > self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            frames = frames[indices]
            latents = latents[indices]
        elif T < self.num_frames:
            # Pad with last frame
            pad_frames = frames[-1:].repeat(self.num_frames - T, 1, 1, 1)
            pad_latents = latents[-1:].repeat(self.num_frames - T, 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)
            latents = torch.cat([latents, pad_latents], dim=0)

        # Get caption
        caption = self.captions.get(video_id, "")

        return {
            "frames": frames,
            "latents": latents,
            "caption": caption,
            "video_id": video_id,
        }


def collate_fn(batch):
    """Collate batch with captions."""
    frames = torch.stack([b["frames"] for b in batch])
    latents = torch.stack([b["latents"] for b in batch])
    captions = [b["caption"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    return {
        "frames": frames,
        "latents": latents,
        "captions": captions,
        "video_ids": video_ids,
    }


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization."""
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch,
                    metrics, config, output_dir, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latest
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")

    # Save periodic
    torch.save(checkpoint, output_dir / f"checkpoint_step{step}.pt")

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / "checkpoint_best.pt")

    print(f"Saved checkpoint at step {step}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["step"], checkpoint["epoch"], checkpoint.get("metrics", {})


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def train(config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if HAS_WANDB:
        run_name = config["wandb_run_name"] or f"efficient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config["wandb_project"],
            name=run_name,
            config=config,
            resume="allow",
        )
        print(f"wandb initialized: {run_name}")

    # Create dataset and dataloader
    dataset = PrecomputedDataset(
        data_dir=config["data_dir"],
        num_frames=config["num_frames"],
        mode=config["mode"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} videos")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    print("Loading model...")
    model = FoveatedVideoModel(
        deep_query=True,
        freeze_dino=True,
        lambda_coarse=config["lambda_coarse"],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    # Estimate total steps
    steps_per_epoch = len(dataloader) // config["grad_accum"]
    estimated_steps_per_hour = steps_per_epoch * 2  # Rough estimate
    total_steps = int(config["max_hours"] * estimated_steps_per_hour)

    # Scheduler
    scheduler = get_lr_scheduler(optimizer, config["warmup_steps"], total_steps)

    # Mixed precision
    scaler = GradScaler()

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    best_ratio = 0.0
    metrics_history = {"loss_fine": [], "loss_coarse": [], "ratio": []}

    if config["resume_checkpoint"]:
        resume_path = Path(config["resume_checkpoint"])
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            start_step, start_epoch, loaded_metrics = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler
            )
            if loaded_metrics:
                metrics_history = loaded_metrics.get("history", metrics_history)
                best_ratio = loaded_metrics.get("best_ratio", 0.0)
            print(f"Resumed at step {start_step}, epoch {start_epoch}")
    else:
        # Check for existing checkpoint
        latest_ckpt = output_dir / "checkpoint_latest.pt"
        if latest_ckpt.exists():
            print(f"Found existing checkpoint at {latest_ckpt}")
            start_step, start_epoch, loaded_metrics = load_checkpoint(
                latest_ckpt, model, optimizer, scheduler, scaler
            )
            if loaded_metrics:
                metrics_history = loaded_metrics.get("history", metrics_history)
                best_ratio = loaded_metrics.get("best_ratio", 0.0)
            print(f"Resumed at step {start_step}, epoch {start_epoch}")

    # Training state
    model.train()
    global_step = start_step
    epoch = start_epoch
    start_time = time.time()
    max_runtime = config["max_hours"] * 3600

    # Metrics tracking
    running_loss_fine = 0.0
    running_loss_coarse = 0.0
    running_count = 0

    # Stats for ETA
    step_times = []

    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"Mode: {config['mode']}")
    print(f"Max runtime: {config['max_hours']} hours")
    print(f"Batch size: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"Frames per video: {config['num_frames']}")
    print(f"Fine iterations: {config['fine_iterations']}")
    print(f"{'='*60}\n")

    optimizer.zero_grad()
    accum_step = 0

    while True:
        epoch += 1

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            # Check time limit
            elapsed = time.time() - start_time
            if elapsed >= max_runtime:
                print(f"\nTime limit reached ({config['max_hours']}h)")
                break

            # Move to device
            frames = batch["frames"].to(device)  # [B, T, 3, H, W]
            latents = batch["latents"].to(device)  # [B, T, 4, 32, 32]

            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get empty text embeds for reconstruction-only
                B = frames.shape[0]
                text_embeds = model.get_empty_text_embeds(B).to(device)

                # Forward with multi-fine
                loss, loss_fine, loss_coarse = model(
                    text_embeds, frames, latents,
                    fine_iterations=config["fine_iterations"],
                )

                # Scale loss for gradient accumulation
                loss = loss / config["grad_accum"]

            # Backward
            scaler.scale(loss).backward()

            accum_step += 1

            # Update metrics
            running_loss_fine += loss_fine.item()
            running_loss_coarse += loss_coarse.item()
            running_count += 1

            # Optimizer step
            if accum_step >= config["grad_accum"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_step = 0
                global_step += 1

                step_time = time.time() - step_start
                step_times.append(step_time)
                if len(step_times) > 100:
                    step_times.pop(0)

                # Logging
                if global_step % config["log_interval"] == 0:
                    avg_fine = running_loss_fine / running_count
                    avg_coarse = running_loss_coarse / running_count
                    ratio = avg_coarse / avg_fine if avg_fine > 0 else 1.0

                    # ETA calculation
                    avg_step_time = sum(step_times) / len(step_times)
                    remaining_time = max_runtime - elapsed
                    remaining_steps = int(remaining_time / avg_step_time) if avg_step_time > 0 else 0

                    # Progress
                    progress_pct = (elapsed / max_runtime) * 100

                    print(f"Step {global_step:6d} | "
                          f"Fine: {avg_fine:.4f} | Coarse: {avg_coarse:.4f} | "
                          f"Ratio: {ratio:.3f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Progress: {progress_pct:.1f}% | ETA: {format_time(remaining_time)}")

                    # wandb logging
                    if HAS_WANDB:
                        wandb.log({
                            "loss_fine": avg_fine,
                            "loss_coarse": avg_coarse,
                            "ratio_coarse_fine": ratio,
                            "improvement_pct": (ratio - 1.0) * 100,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": global_step,
                            "progress_pct": progress_pct,
                            "samples_per_sec": config["batch_size"] * config["grad_accum"] / avg_step_time,
                        })

                    # Track history
                    metrics_history["loss_fine"].append(avg_fine)
                    metrics_history["loss_coarse"].append(avg_coarse)
                    metrics_history["ratio"].append(ratio)

                    # Reset running metrics
                    running_loss_fine = 0.0
                    running_loss_coarse = 0.0
                    running_count = 0

                    # Check for best model
                    if ratio > best_ratio:
                        best_ratio = ratio
                        is_best = True
                    else:
                        is_best = False

                # Save checkpoint
                if global_step % config["save_interval"] == 0:
                    metrics = {
                        "history": metrics_history,
                        "best_ratio": best_ratio,
                    }
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, epoch, metrics, config, output_dir,
                        is_best=(global_step % config["log_interval"] == 0 and ratio > best_ratio)
                    )

        # Check time limit after epoch
        elapsed = time.time() - start_time
        if elapsed >= max_runtime:
            break

    # Final save
    print("\nTraining complete. Saving final checkpoint...")
    metrics = {
        "history": metrics_history,
        "best_ratio": best_ratio,
    }
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        global_step, epoch, metrics, config, output_dir
    )

    # Final summary
    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps: {global_step}")
    print(f"Total epochs: {epoch}")
    print(f"Runtime: {elapsed_hours:.2f} hours")
    print(f"Best ratio: {best_ratio:.3f}")

    if metrics_history["ratio"]:
        final_ratio = metrics_history["ratio"][-1]
        print(f"Final ratio: {final_ratio:.3f}")
        if final_ratio > 1.0:
            improvement = (final_ratio - 1.0) * 100
            print(f"✓ loss_coarse > loss_fine by {improvement:.1f}% - HYPOTHESIS VALIDATED!")
        else:
            print(f"✗ loss_fine >= loss_coarse - needs more training or tuning")

    print(f"{'='*60}")

    if HAS_WANDB:
        wandb.finish()

    return global_step, best_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_hours", type=float, default=CONFIG["max_hours"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--num_frames", type=int, default=CONFIG["num_frames"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--mode", type=str, default=CONFIG["mode"],
                        choices=["reconstruction_only", "joint"])
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    # Update config
    CONFIG["max_hours"] = args.max_hours
    CONFIG["batch_size"] = args.batch_size
    CONFIG["num_frames"] = args.num_frames
    CONFIG["resume_checkpoint"] = args.resume
    CONFIG["mode"] = args.mode
    CONFIG["wandb_run_name"] = args.wandb_name

    train(CONFIG)
