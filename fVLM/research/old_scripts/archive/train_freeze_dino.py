#!/usr/bin/env python3
"""
Long training run with freeze_dino=True (the winning configuration).

Based on ablation results:
- freeze_dino=True achieved ratio=0.9966 (best) and sim=0.43 (57% differentiation)
- This run trains for 10K+ steps to validate the trend continues

Usage:
    python scripts/train_freeze_dino.py [--steps 10000] [--batch_size 2]
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LocalVideoDataset(Dataset):
    """Load precomputed frames and latents from local storage."""

    def __init__(self, frames_dir: str, latents_dir: str):
        self.frames_dir = Path(frames_dir)
        self.latents_dir = Path(latents_dir)

        # Find videos with both frames and latents
        frame_ids = {f.stem for f in self.frames_dir.glob("*.pt")}
        latent_ids = {f.stem for f in self.latents_dir.glob("*.pt")}
        self.video_ids = sorted(frame_ids & latent_ids)

        print(f"Found {len(self.video_ids)} videos with both frames and latents")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = torch.load(self.frames_dir / f"{video_id}.pt")
        latents = torch.load(self.latents_dir / f"{video_id}.pt")
        return frames, latents


class FoveatedEncoderFrozen(nn.Module):
    """Vision encoder with frozen DINO backbone."""

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-small",
        query_dim: int = 384,
        output_dim: int = 384,
    ):
        super().__init__()

        # Load pretrained DINOv2 and FREEZE it
        self.dino = AutoModel.from_pretrained(dino_model_name)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        self.dino_dim = self.dino.config.hidden_size

        # Only these projections are trainable
        self.query_input_proj = nn.Linear(query_dim, self.dino_dim, bias=False)
        self.query_output_proj = nn.Linear(self.dino_dim, output_dim)

        self.register_buffer("_dummy", torch.zeros(1))

    @property
    def device(self):
        return self._dummy.device

    def encode_patches(self, images: torch.Tensor):
        """Encode images to patch tokens (frozen)."""
        with torch.no_grad():
            outputs = self.dino(images, output_hidden_states=True)
            patch_features = outputs.last_hidden_state
        return patch_features, {'patch_features': patch_features}

    def query_attend(self, query: torch.Tensor, cache: dict) -> torch.Tensor:
        """Use query to attend over cached patch features."""
        q_embed = self.query_input_proj(query)
        patch_features = cache['patch_features']

        q_embed = q_embed.unsqueeze(1)
        attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (self.dino_dim ** 0.5), dim=-1)

        z = torch.bmm(attn_weights, patch_features)
        z = z.squeeze(1)
        z = self.query_output_proj(z)

        return z


class FoveatedVideoModelFrozen(nn.Module):
    """Foveated VLM with frozen DINO backbone."""

    def __init__(
        self,
        llm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_name: str = "facebook/dinov2-small",
    ):
        super().__init__()

        # Frozen vision encoder
        self.encoder = FoveatedEncoderFrozen(
            dino_model_name=dino_name,
            query_dim=384,
            output_dim=384,
        )

        # LLM (trainable)
        from transformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_dim = self.llm.config.hidden_size

        # Query vectors (trainable) - std=1.0 as per bug fix
        self.q_static = nn.Parameter(torch.randn(1, 384))
        self.q_init = nn.Parameter(torch.randn(1, 384))

        # Projections (trainable)
        self.visual_proj = nn.Linear(384, self.llm_dim)
        self.query_proj = nn.Linear(self.llm_dim, 384)

        # Prediction head (trainable)
        self.pred_head = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, 4 * 32 * 32),
        )

        self.register_buffer("_dummy", torch.zeros(1))

    @property
    def device(self):
        return self._dummy.device

    def forward(self, frames: torch.Tensor, target_latents: torch.Tensor):
        """
        Forward pass with two-pass architecture.

        Args:
            frames: [B, T, 3, H, W] video frames
            target_latents: [B, T, 4, 32, 32] VAE latents

        Returns:
            loss: Combined loss
            loss_fine: Pass 2 loss (with dynamic queries)
            loss_coarse: Pass 1 loss (with static query)
        """
        B, T = frames.shape[:2]
        device = frames.device

        losses_fine = []
        losses_coarse = []

        # Initialize LLM hidden state
        hidden = torch.zeros(B, 1, self.llm_dim, device=device)

        for t in range(T - 1):
            frame_t = frames[:, t]
            target_t = target_latents[:, t + 1]

            # Encode patches (frozen DINO)
            _, cache = self.encoder.encode_patches(frame_t)

            # Pass 1: Static query (coarse)
            q_coarse = self.q_static.expand(B, -1)
            z_coarse = self.encoder.query_attend(q_coarse, cache)
            z_coarse_proj = self.visual_proj(z_coarse).unsqueeze(1)

            # Pass 2: Dynamic query (fine)
            q_init = self.q_init.expand(B, -1)
            z_init = self.encoder.query_attend(q_init, cache)
            z_init_proj = self.visual_proj(z_init).unsqueeze(1)

            # LLM processes visual token
            llm_input = hidden + z_init_proj
            llm_out = self.llm(inputs_embeds=llm_input, output_hidden_states=True)
            llm_hidden = llm_out.hidden_states[-1][:, -1:, :]

            # Generate dynamic query from LLM
            q_dynamic = self.query_proj(llm_hidden.squeeze(1))
            z_fine = self.encoder.query_attend(q_dynamic, cache)
            z_fine_proj = self.visual_proj(z_fine).unsqueeze(1)

            # Predictions
            pred_fine = self.pred_head(llm_hidden.squeeze(1))
            pred_fine = pred_fine.view(B, 4, 32, 32)

            pred_coarse = self.pred_head(z_coarse_proj.squeeze(1))
            pred_coarse = pred_coarse.view(B, 4, 32, 32)

            # Losses
            loss_fine = F.mse_loss(pred_fine, target_t)
            loss_coarse = F.mse_loss(pred_coarse, target_t)

            losses_fine.append(loss_fine)
            losses_coarse.append(loss_coarse)

            # Update hidden state
            hidden = llm_hidden + z_fine_proj

        # Average losses
        avg_loss_fine = torch.stack(losses_fine).mean()
        avg_loss_coarse = torch.stack(losses_coarse).mean()

        # Combined loss with auxiliary
        loss = avg_loss_fine + 0.5 * avg_loss_coarse

        return loss, avg_loss_fine, avg_loss_coarse

    def compute_feature_similarity(self, frames: torch.Tensor) -> float:
        """Compute cosine similarity between z_fine and z_coarse."""
        B, T = frames.shape[:2]

        with torch.no_grad():
            sims = []
            for t in range(T):
                frame_t = frames[:, t]
                _, cache = self.encoder.encode_patches(frame_t)

                # Static query
                q_coarse = self.q_static.expand(B, -1)
                z_coarse = self.encoder.query_attend(q_coarse, cache)

                # Dynamic query (use q_init for simplicity)
                q_fine = self.q_init.expand(B, -1)
                z_fine = self.encoder.query_attend(q_fine, cache)

                sim = F.cosine_similarity(z_fine, z_coarse, dim=-1).mean()
                sims.append(sim.item())

            return sum(sims) / len(sims)


def train(args):
    """Run long training with frozen DINO."""

    print("=" * 70)
    print("LONG TRAINING RUN: freeze_dino=True")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    output_dir = Path("outputs/freeze_dino_long")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = LocalVideoDataset("data/frames", "data/latents")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    print("\nLoading model with frozen DINO...")
    model = FoveatedVideoModelFrozen().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total params: {total_params/1e6:.1f}M")
    print(f"Trainable: {trainable_params/1e6:.1f}M")
    print(f"Frozen (DINO): {frozen_params/1e6:.1f}M")

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )

    scaler = GradScaler()

    # Training state
    step = 0
    metrics = {
        'loss_fine': [],
        'loss_coarse': [],
        'ratio': [],
        'feature_sim': [],
        'lr': [],
    }

    print(f"\nStarting training for {args.steps} steps...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 70)

    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        for frames, latents in dataloader:
            if step >= args.steps:
                break

            frames = frames.to(device)
            latents = latents.to(device)

            optimizer.zero_grad()

            with autocast():
                loss, loss_fine, loss_coarse = model(frames, latents)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Compute metrics
            ratio = (loss_coarse / loss_fine).item()

            metrics['loss_fine'].append(loss_fine.item())
            metrics['loss_coarse'].append(loss_coarse.item())
            metrics['ratio'].append(ratio)
            metrics['lr'].append(scheduler.get_last_lr()[0])

            # Compute feature similarity periodically
            if step % 100 == 0:
                sim = model.compute_feature_similarity(frames)
                metrics['feature_sim'].append((step, sim))
                pbar.set_postfix({
                    'fine': f'{loss_fine.item():.3f}',
                    'coarse': f'{loss_coarse.item():.3f}',
                    'ratio': f'{ratio:.4f}',
                    'sim': f'{sim:.3f}',
                })
            else:
                pbar.set_postfix({
                    'fine': f'{loss_fine.item():.3f}',
                    'coarse': f'{loss_coarse.item():.3f}',
                    'ratio': f'{ratio:.4f}',
                })

            step += 1
            pbar.update(1)

            # Save checkpoint periodically
            if step % 1000 == 0:
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }
                torch.save(checkpoint, output_dir / f"checkpoint_{step}.pt")

                # Also save metrics
                with open(output_dir / "metrics.json", 'w') as f:
                    json.dump({
                        'step': step,
                        'avg_loss_fine': sum(metrics['loss_fine'][-100:]) / 100,
                        'avg_loss_coarse': sum(metrics['loss_coarse'][-100:]) / 100,
                        'avg_ratio': sum(metrics['ratio'][-100:]) / 100,
                        'feature_sim_history': metrics['feature_sim'],
                    }, f, indent=2)

                print(f"\n[Step {step}] Saved checkpoint")

    pbar.close()

    # Final metrics
    final_loss_fine = sum(metrics['loss_fine'][-100:]) / min(100, len(metrics['loss_fine']))
    final_loss_coarse = sum(metrics['loss_coarse'][-100:]) / min(100, len(metrics['loss_coarse']))
    final_ratio = sum(metrics['ratio'][-100:]) / min(100, len(metrics['ratio']))
    final_sim = metrics['feature_sim'][-1][1] if metrics['feature_sim'] else 0

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Loss (fine):   {final_loss_fine:.4f}")
    print(f"Final Loss (coarse): {final_loss_coarse:.4f}")
    print(f"Final Ratio:         {final_ratio:.4f}")
    print(f"Final Feature Sim:   {final_sim:.4f}")
    print(f"\nCheckpoints saved to: {output_dir}")

    # Save final results
    results = {
        'config': {
            'freeze_dino': True,
            'steps': args.steps,
            'batch_size': args.batch_size,
            'lr': args.lr,
        },
        'final_metrics': {
            'loss_fine': final_loss_fine,
            'loss_coarse': final_loss_coarse,
            'ratio': final_ratio,
            'feature_sim': final_sim,
        },
        'full_metrics': {
            'loss_fine': metrics['loss_fine'],
            'loss_coarse': metrics['loss_coarse'],
            'ratio': metrics['ratio'],
            'feature_sim': metrics['feature_sim'],
        },
    }

    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train(args)
