#!/usr/bin/env python3
"""
Experiment runner for prediction head alternatives.

Tests different prediction heads on VAE latent prediction task.
Uses local precomputed data for speed.
"""

import sys
import time
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# Import alternative heads
from direct_mlp_head import DirectMLPHead
from cross_attention_head import CrossAttentionHead

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_frames_from_dir(frame_dir, device, num_frames=8):
    """Load frames from precomputed directory."""
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if len(frame_paths) < num_frames:
        return None

    # Sample evenly
    indices = np.linspace(0, len(frame_paths) - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        img = Image.open(frame_paths[idx]).convert('RGB')
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
        frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(frame)

    frames = torch.stack(frames)  # [T, C, H, W]
    return frames


def normalize_frames(frames):
    """ImageNet normalize frames."""
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


@torch.no_grad()
def compute_vae_latents(vae, frames, device):
    """Compute VAE latents. frames: [B, T, C, H, W] in [0,1]."""
    B, T, C, H, W = frames.shape
    frames_vae = frames * 2 - 1
    frames_flat = frames_vae.reshape(B * T, C, H, W).to(device)
    latents_flat = vae.encode(frames_flat).latent_dist.sample()
    latents_flat = latents_flat * vae.config.scaling_factor
    latents = latents_flat.reshape(B, T, 4, H // 8, W // 8)
    return latents


def run_experiment(
    head_type: str = "film",  # "film", "mlp", "cross_attention"
    max_minutes: float = 30.0,
    batch_size: int = 2,
    grad_accum: int = 4,
    num_frames: int = 8,
    learning_rate: float = 3e-5,
    warmup_steps: int = 50,
    log_interval: int = 25,
    data_dir: str = "/mnt/d/projects/fVLM/data/precomputed/frames",
    output_dir: str = None,
    use_wandb: bool = True,
):
    """Run prediction head experiment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = f"outputs/pred_head_{head_type}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PREDICTION HEAD EXPERIMENT: {head_type.upper()}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max time: {max_minutes} minutes")
    print(f"Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"Data: {data_dir}")
    print("=" * 70)

    # Load model
    print("\nLoading model...", flush=True)
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    # Replace prediction head based on experiment type
    if head_type == "mlp":
        print("Replacing with DirectMLPHead...", flush=True)
        model.pred_head = DirectMLPHead(h_dim=576, latent_channels=4).to(device)
    elif head_type == "cross_attention":
        print("Replacing with CrossAttentionHead...", flush=True)
        model.pred_head = CrossAttentionHead(h_dim=576, latent_channels=4).to(device)
    elif head_type == "film":
        print("Using default FiLM head...", flush=True)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    pred_head_params = sum(p.numel() for p in model.pred_head.parameters())
    print(f"Prediction head parameters: {pred_head_params / 1e6:.2f}M")

    model.train()

    # Load VAE
    print("Loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda')

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load data
    print(f"\nLoading data from {data_dir}...", flush=True)
    data_dir = Path(data_dir)
    all_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    random.shuffle(all_dirs)
    print(f"Found {len(all_dirs)} videos")

    # Wandb
    if use_wandb and HAS_WANDB:
        wandb.init(
            project="foveated-vlm-pred-head",
            name=f"{head_type}_{datetime.now().strftime('%m%d_%H%M')}",
            config={
                "head_type": head_type,
                "max_minutes": max_minutes,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "num_frames": num_frames,
                "learning_rate": learning_rate,
                "pred_head_params": pred_head_params,
            }
        )

    # Training loop
    step = 0
    data_idx = 0
    accum_fine = 0.0
    accum_coarse = 0.0
    accum_count = 0
    ratios = []
    start_time = time.time()
    max_seconds = max_minutes * 60

    print(f"\nStarting training (max {max_minutes} minutes)...", flush=True)
    pbar = tqdm(desc="Training")
    optimizer.zero_grad()

    while True:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"\nTime limit reached ({max_minutes} min)")
            break

        # Load batch
        batch_frames = []
        while len(batch_frames) < batch_size:
            if data_idx >= len(all_dirs):
                random.shuffle(all_dirs)
                data_idx = 0

            frames = load_frames_from_dir(all_dirs[data_idx], device, num_frames)
            data_idx += 1
            if frames is not None:
                batch_frames.append(frames)

        # Prepare batch
        frames_raw = torch.stack(batch_frames).to(device)  # [B, T, C, H, W]
        frames_norm = normalize_frames(frames_raw)

        # Compute VAE latents
        vae_latents = compute_vae_latents(vae, frames_raw, device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Get empty text embeds (reconstruction only)
            text_embeds = model.get_empty_text_embeds(frames_norm.shape[0])

            # Forward pass
            _, loss_fine, loss_coarse = model(text_embeds, frames_norm, vae_latents)

            loss = loss_fine / grad_accum

        scaler.scale(loss).backward()

        # Accumulate
        accum_fine += loss_fine.item()
        accum_coarse += loss_coarse.item()
        accum_count += 1

        if accum_count >= grad_accum:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            pbar.update(1)

            # Compute metrics
            avg_fine = accum_fine / grad_accum
            avg_coarse = accum_coarse / grad_accum
            ratio = avg_coarse / max(avg_fine, 1e-8)
            ratios.append(ratio)

            # Log
            if step % log_interval == 0:
                recent_ratio = np.mean(ratios[-50:]) if len(ratios) >= 50 else np.mean(ratios)
                elapsed_min = elapsed / 60
                remaining_min = max_minutes - elapsed_min

                log_dict = {
                    "loss_fine": avg_fine,
                    "loss_coarse": avg_coarse,
                    "ratio": ratio,
                    "ratio_avg50": recent_ratio,
                    "lr": scheduler.get_last_lr()[0],
                    "step": step,
                    "elapsed_min": elapsed_min,
                }
                if use_wandb and HAS_WANDB:
                    wandb.log(log_dict)

                pbar.set_postfix({
                    "fine": f"{avg_fine:.4f}",
                    "coarse": f"{avg_coarse:.4f}",
                    "ratio": f"{ratio:.3f}",
                    "avg": f"{recent_ratio:.3f}",
                    "rem": f"{remaining_min:.1f}m",
                })

            # Reset accumulators
            accum_fine = 0.0
            accum_coarse = 0.0
            accum_count = 0

    pbar.close()

    # Final statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    final_ratio = np.mean(ratios[-100:]) if len(ratios) >= 100 else np.mean(ratios)
    print(f"Head type: {head_type}")
    print(f"Total steps: {step}")
    print(f"Final ratio (avg last 100): {final_ratio:.4f}")
    print(f"Min ratio: {min(ratios):.4f}")
    print(f"Max ratio: {max(ratios):.4f}")
    print(f"Ratio > 1.0: {sum(1 for r in ratios if r > 1.0) / len(ratios) * 100:.1f}%")

    # Save results
    results = {
        "head_type": head_type,
        "total_steps": step,
        "final_ratio": float(final_ratio),
        "min_ratio": float(min(ratios)),
        "max_ratio": float(max(ratios)),
        "ratio_above_1": float(sum(1 for r in ratios if r > 1.0) / len(ratios)),
        "all_ratios": [float(r) for r in ratios],
        "pred_head_params": pred_head_params,
    }

    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    if use_wandb and HAS_WANDB:
        wandb.finish()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=str, default="film",
                        choices=["film", "mlp", "cross_attention"])
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/d/projects/fVLM/data/precomputed/frames")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    results = run_experiment(
        head_type=args.head,
        max_minutes=args.minutes,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )
