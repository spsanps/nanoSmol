#!/usr/bin/env python3
"""
Fast prediction-only training for SmolVLM2 video model.

Optimizations:
- Prediction task only (no captioning) - faster
- Consolidated data loading from multiple directories
- Larger batch size
- Efficient VAE decoding
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.smolvlm_video import SmolVLMVideo

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class LatentDataset(Dataset):
    """Dataset loading precomputed VAE latents."""

    def __init__(self, latent_dir: str, num_frames: int = 8):
        self.latent_dir = Path(latent_dir)
        self.num_frames = num_frames

        # Find all latent files
        self.latent_files = sorted(list(self.latent_dir.glob("*.pt")))
        print(f"  Found {len(self.latent_files)} latents in {latent_dir}")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = self.latent_files[idx]
        latents = torch.load(latent_path, weights_only=True)  # (T, 4, H, W)

        # Sample context frames + target
        total_frames = latents.shape[0]
        if total_frames >= self.num_frames + 1:
            max_start = total_frames - self.num_frames - 1
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            context = latents[start:start + self.num_frames]
            target = latents[start + self.num_frames]
        else:
            # Pad if needed
            context = latents[:self.num_frames]
            if len(context) < self.num_frames:
                pad = self.num_frames - len(context)
                context = F.pad(context, (0, 0, 0, 0, 0, 0, 0, pad))
            target = latents[-1]

        return {"context": context, "target": target}


def collate_fn(batch):
    return {
        "context": torch.stack([b["context"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"FAST PREDICTION-ONLY TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print(f"\n=== Loading Model ===")
    model = SmolVLMVideo(
        model_name=args.model_name,
        freeze_vision=True,  # Freeze vision for speed
        freeze_connector=False,
        gradient_checkpointing=True,
    )
    model = model.to(device)

    # Move VAE to device once
    model.vae.to(device, dtype=torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.1f}M")
    print(f"Trainable: {trainable_params/1e6:.1f}M")

    # Load datasets from multiple directories
    print(f"\n=== Loading Datasets ===")
    datasets = []
    for data_dir in args.data_dirs:
        latent_dir = Path(data_dir)
        if latent_dir.exists():
            ds = LatentDataset(str(latent_dir), num_frames=args.num_frames)
            if len(ds) > 0:
                datasets.append(ds)

    if not datasets:
        print("ERROR: No data found!")
        return

    combined_dataset = ConcatDataset(datasets)
    total_samples = len(combined_dataset)
    print(f"\nTotal samples: {total_samples}")

    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # LR scheduler
    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Wandb
    if HAS_WANDB and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # Training info
    samples_per_epoch = total_samples
    total_sample_passes = samples_per_epoch * args.epochs

    print(f"\n=== Training Config ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Total sample passes: {total_sample_passes:,}")
    print(f"Learning rate: {args.lr}")

    # Estimate time
    estimated_speed = 10  # samples/sec estimate
    estimated_time = total_sample_passes / estimated_speed / 3600
    print(f"Estimated time: {estimated_time:.1f} hours")

    print(f"\n=== Starting Training ===")

    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    # Metrics tracking
    losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()

            # Move to device
            context = batch["context"].to(device, dtype=torch.bfloat16)
            target = batch["target"].to(device, dtype=torch.bfloat16)

            B, T, C, H, W = context.shape

            # Decode latents to pixels
            with torch.no_grad():
                flat_latents = context.view(B * T, C, H, W).float()
                pixel_values = model.vae.decode(flat_latents).sample
                pixel_values = pixel_values.view(B, T, 3, H * 8, W * 8)
                pixel_values = pixel_values.to(torch.bfloat16)

            # Forward - prediction only
            pred_out = model.forward_predict(pixel_values, target)
            loss = pred_out["loss"]

            # Backward
            loss.backward()

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            epoch_samples += B
            losses.append(loss.item())

            # Logging
            batch_time = time.time() - batch_start
            samples_per_sec = B / batch_time

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                remaining = (total_steps - global_step) * (elapsed / global_step)
                mem = torch.cuda.max_memory_allocated() / 1e9
                lr = scheduler.get_last_lr()[0]
                avg_loss = sum(losses[-50:]) / len(losses[-50:])

                log_dict = {
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "lr": lr,
                    "samples_per_sec": samples_per_sec,
                    "memory_gb": mem,
                    "step": global_step,
                }

                if HAS_WANDB and args.wandb:
                    wandb.log(log_dict)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "spd": f"{samples_per_sec:.1f}/s",
                    "mem": f"{mem:.1f}GB",
                    "eta": f"{remaining/3600:.1f}h",
                })

            # Save checkpoint
            if global_step % args.save_every == 0:
                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, ckpt_path)
                print(f"\n  Saved checkpoint: {ckpt_path}")

            # Check time limit
            if args.max_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= args.max_hours:
                    print(f"\n  Time limit reached ({args.max_hours}h). Stopping training.")
                    break

        # Break out of epoch loop if time limit reached
        if args.max_hours is not None:
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= args.max_hours:
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_epoch_loss:.4f}")
        print(f"  Samples: {epoch_samples:,}")
        print(f"  Time: {elapsed/3600:.2f}h elapsed")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = checkpoint_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "loss": avg_epoch_loss,
            }, best_path)
            print(f"  New best! Saved to {best_path}")

    # Final save
    total_time = time.time() - start_time
    final_path = checkpoint_dir / "final.pt"
    torch.save({
        "epochs": args.epochs,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "final_loss": avg_epoch_loss,
        "training_time": total_time,
    }, final_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total steps: {global_step}")
    print(f"Total samples: {global_step * args.batch_size:,}")
    print(f"Final loss: {avg_epoch_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Throughput: {global_step * args.batch_size / total_time:.1f} samples/sec")
    print(f"Model saved: {final_path}")

    # Save training log
    log_path = output_dir / "training_summary.md"
    with open(log_path, "w") as f:
        f.write(f"# Training Summary\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Config\n")
        f.write(f"- Model: {args.model_name}\n")
        f.write(f"- Samples: {total_samples:,}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.lr}\n\n")
        f.write(f"## Results\n")
        f.write(f"- Training time: {total_time/3600:.2f} hours\n")
        f.write(f"- Final loss: {avg_epoch_loss:.4f}\n")
        f.write(f"- Best loss: {best_loss:.4f}\n")
        f.write(f"- Throughput: {global_step * args.batch_size / total_time:.1f} samples/sec\n")

    print(f"Summary saved: {log_path}")

    if HAS_WANDB and args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    # Data - multiple directories (all precomputed VAE latents)
    parser.add_argument("--data_dirs", nargs="+", default=[
        "data/frames_latents",
        "data/latents",
        "data/latents_phase2",
        "data/webvid/latents",
        "data/webvid_large/latents",
        "data/webvid_test/latents",
    ])
    parser.add_argument("--num_frames", type=int, default=8)

    # Model
    parser.add_argument("--model_name", type=str,
                        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_hours", type=float, default=None, help="Stop training after this many hours")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/smolvlm_fast")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="smolvlm-video")
    parser.add_argument("--run_name", type=str, default=None)

    # Workers
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"fast_{args.epochs}ep_b{args.batch_size}"

    train(args)


if __name__ == "__main__":
    main()
