#!/usr/bin/env python3
"""
Fast multi-task training for SmolVLM2-based video model.

Tasks:
1. Caption: video → text
2. Predict: video → next frame latents

Optimizations:
- Larger batch size (low memory footprint)
- Gradient accumulation
- Mixed precision (bfloat16)
- Gradient checkpointing
- Efficient data loading
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.smolvlm_video import SmolVLMVideo

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WebVidDataset(Dataset):
    """WebVid dataset for multi-task training."""

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 8,
        frame_size: int = 512,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size

        # Load captions
        caption_file = self.data_dir / "captions.json"
        if caption_file.exists():
            with open(caption_file) as f:
                self.captions = json.load(f)
        else:
            self.captions = {}

        # Find all video latent files
        latent_dir = self.data_dir / "latents"
        if latent_dir.exists():
            self.video_ids = sorted([
                f.stem for f in latent_dir.glob("*.pt")
            ])
        else:
            self.video_ids = []

        print(f"Found {len(self.video_ids)} videos with latents")

        # Find frame directories
        frame_dir = self.data_dir / "frames"
        if frame_dir.exists():
            self.frame_dirs = {
                d.name: d for d in frame_dir.iterdir() if d.is_dir()
            }
        else:
            self.frame_dirs = {}

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        # Load latents
        latent_path = self.data_dir / "latents" / f"{video_id}.pt"
        latents = torch.load(latent_path, weights_only=True)  # (T, 4, 64, 64)

        # Sample frames
        total_frames = latents.shape[0]
        if total_frames >= self.num_frames + 1:
            # Random start for context + 1 for target
            max_start = total_frames - self.num_frames - 1
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            context_latents = latents[start:start + self.num_frames]
            target_latent = latents[start + self.num_frames]
        else:
            # Pad if too few frames
            context_latents = latents[:self.num_frames]
            if len(context_latents) < self.num_frames:
                pad = self.num_frames - len(context_latents)
                context_latents = F.pad(context_latents, (0, 0, 0, 0, 0, 0, 0, pad))
            target_latent = latents[-1]

        # Get caption
        caption = self.captions.get(video_id, "A video.")

        # Load frames if available (for VLM input)
        if video_id in self.frame_dirs:
            frame_dir = self.frame_dirs[video_id]
            frame_files = sorted(frame_dir.glob("*.jpg"))[:self.num_frames]
            # Would load and process frames here
            # For now, we'll generate from latents
            pixel_values = None
        else:
            pixel_values = None

        return {
            "video_id": video_id,
            "context_latents": context_latents,  # (num_frames, 4, 64, 64)
            "target_latent": target_latent,      # (4, 64, 64)
            "caption": caption,
        }


def collate_fn(batch):
    """Collate batch of samples."""
    return {
        "video_ids": [b["video_id"] for b in batch],
        "context_latents": torch.stack([b["context_latents"] for b in batch]),
        "target_latents": torch.stack([b["target_latent"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\n=== Loading Model ===")
    model = SmolVLMVideo(
        model_name=args.model_name,
        freeze_vision=args.freeze_vision,
        freeze_connector=args.freeze_connector,
        gradient_checkpointing=True,
    )
    model = model.to(device)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.1f}M")
    print(f"Trainable: {trainable/1e6:.1f}M")

    # Create dataset
    print("\n=== Loading Dataset ===")
    dataset = WebVidDataset(
        data_dir=args.data_dir,
        num_frames=args.num_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + torch.cos(torch.tensor((step - warmup_steps) / (total_steps - warmup_steps) * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output directory
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

    # Training loop
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_caption_loss = 0
        epoch_predict_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            context_latents = batch["context_latents"].to(device, dtype=torch.bfloat16)
            target_latents = batch["target_latents"].to(device, dtype=torch.bfloat16)
            captions = batch["captions"]

            # === Task 1: Prediction (latent → latent) ===
            # Use VAE to decode context latents to pixel values for VLM
            with torch.no_grad():
                # Flatten batch and frames
                B, T, C, H, W = context_latents.shape
                flat_latents = context_latents.view(B * T, C, H, W)

                # VAE needs float32 and to be on same device
                model.vae.to(device, dtype=torch.float32)
                pixel_values = model.vae.decode(flat_latents.float()).sample

                # H, W are latent sizes (32), multiply by 8 to get pixel size (256)
                pixel_values = pixel_values.view(B, T, 3, H * 8, W * 8)
                pixel_values = pixel_values.to(torch.bfloat16)

            # Prediction forward
            pred_out = model.forward_predict(pixel_values, target_latents)
            predict_loss = pred_out["loss"]

            # === Task 2: Captioning ===
            # Process captions
            inputs = model.processor(
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Caption forward (use pixel_values from above)
            caption_out = model.forward_caption(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            caption_loss = caption_out["loss"]

            # Combined loss
            loss = args.lambda_predict * predict_loss + args.lambda_caption * caption_loss
            loss = loss / args.grad_accum

            # Backward
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    mem = torch.cuda.max_memory_allocated() / 1e9

                    log_dict = {
                        "loss": loss.item() * args.grad_accum,
                        "predict_loss": predict_loss.item(),
                        "caption_loss": caption_loss.item(),
                        "lr": lr,
                        "memory_gb": mem,
                        "step": global_step,
                    }

                    if HAS_WANDB and args.wandb:
                        wandb.log(log_dict)

                    pbar.set_postfix({
                        "loss": f"{loss.item() * args.grad_accum:.4f}",
                        "pred": f"{predict_loss.item():.4f}",
                        "cap": f"{caption_loss.item():.4f}",
                        "mem": f"{mem:.1f}GB",
                    })

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item() * args.grad_accum,
                    }, ckpt_path)
                    print(f"\nSaved checkpoint to {ckpt_path}")

            epoch_loss += loss.item() * args.grad_accum
            epoch_caption_loss += caption_loss.item()
            epoch_predict_loss += predict_loss.item()
            num_batches += 1

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_caption = epoch_caption_loss / num_batches
        avg_predict = epoch_predict_loss / num_batches

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Caption Loss: {avg_caption:.4f}")
        print(f"  Avg Predict Loss: {avg_predict:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, best_path)
            print(f"  New best model saved!")

    # Save final model
    final_path = checkpoint_dir / "final.pt"
    torch.save({
        "epoch": args.epochs,
        "step": global_step,
        "model_state_dict": model.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    if HAS_WANDB and args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLM Video Model")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/webvid",
                        help="Path to WebVid data directory")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of context frames")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                        help="SmolVLM2 model name")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Freeze vision encoder")
    parser.add_argument("--freeze_connector", action="store_true",
                        help="Freeze vision-language connector")

    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # Loss weights
    parser.add_argument("--lambda_predict", type=float, default=1.0,
                        help="Weight for prediction loss")
    parser.add_argument("--lambda_caption", type=float, default=0.1,
                        help="Weight for caption loss")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/smolvlm",
                        help="Output directory")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps")

    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="smolvlm-video",
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name")

    # Workers
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"smolvlm_b{args.batch_size}x{args.grad_accum}_lr{args.lr}"

    train(args)


if __name__ == "__main__":
    main()
