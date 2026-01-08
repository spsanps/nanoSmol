#!/usr/bin/env python3
"""
Quick experiment: Multi-pass query refinement using local precomputed data.

Tests the multi-pass hypothesis without streaming overhead.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import argparse
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import FoveatedEncoder
from src.model.prediction import PredictionHead
from transformers import AutoModelForCausalLM

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class LocalVideoDataset(Dataset):
    """Load precomputed frames and latents from local storage."""

    def __init__(self, frames_dir: str, latents_dir: str, num_frames: int = 8):
        self.frames_dir = Path(frames_dir)
        self.latents_dir = Path(latents_dir)
        self.num_frames = num_frames

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

        # Handle variable length - sample or pad to num_frames
        T = frames.shape[0]
        if T > self.num_frames:
            # Random sample
            start = random.randint(0, T - self.num_frames)
            frames = frames[start:start + self.num_frames]
            latents = latents[start:start + self.num_frames]
        elif T < self.num_frames:
            # Pad by repeating last frame
            pad = self.num_frames - T
            frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)
            latents = torch.cat([latents, latents[-1:].repeat(pad, 1, 1, 1)], dim=0)

        return frames, latents


class MultiPassModel(nn.Module):
    """Multi-pass foveated model for quick experiments."""

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
        num_refine_passes: int = 1,
        freeze_dino: bool = True,
    ):
        super().__init__()

        self.num_refine_passes = num_refine_passes
        self.dino_dim = dino_dim
        self.llm_dim = llm_dim

        # Vision encoder
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_model,
            query_dim=query_dim,
            output_dim=dino_dim,
            deep_query=True,
        )

        if freeze_dino:
            for param in self.encoder.dino.parameters():
                param.requires_grad = False
            self.encoder.dino.eval()

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm.config.use_cache = False

        # Projections
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.llm_to_query = nn.Linear(llm_dim, query_dim)
        self.visual_scale = 0.14

        # Prediction head
        self.pred_head = PredictionHead(h_dim=llm_dim, latent_channels=4)

        # Queries
        self.q_static = nn.Parameter(torch.randn(1, query_dim))
        self.q_init = nn.Parameter(torch.randn(1, query_dim))
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Tokens
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.refine_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
            for _ in range(num_refine_passes)
        ])
        self.no_text_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

    def forward(self, frames: torch.Tensor, latents: torch.Tensor):
        B, T = frames.shape[:2]
        device = frames.device

        # Encode frames
        frames_flat = frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Build caches
        all_caches = []
        if 'kv_cache' in cache_flat:
            num_layers = len(cache_flat['kv_cache'])
            for t in range(T):
                frame_kv_cache = []
                for layer_idx in range(num_layers):
                    layer_cache = cache_flat['kv_cache'][layer_idx]
                    K_all = layer_cache['K'].reshape(B, T, N, D)
                    V_all = layer_cache['V'].reshape(B, T, N, D)
                    frame_kv_cache.append({
                        'K': K_all[:, t], 'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': frame_kv_cache})
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Targets
        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
        prev_latents = torch.cat([z_vae_init, latents[:, :-1]], dim=1)
        target_latents = latents

        # Pass 1: Coarse
        q_static = self.q_static.expand(B, -1)
        z_coarse = torch.stack([self.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)
        z_coarse = self.dino_to_llm(z_coarse)
        z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale

        no_text = self.no_text_token.expand(B, -1, -1)
        coarse_token = self.coarse_token.expand(B, -1, -1)
        seq_coarse = torch.cat([no_text, coarse_token, z_coarse], dim=1)

        h_coarse = self.llm.model(inputs_embeds=seq_coarse).last_hidden_state
        pred_coarse = self.pred_head(h_coarse[:, 1:1+T], prev_latents)
        loss_coarse = F.mse_loss(pred_coarse, target_latents)

        queries = self.llm_to_query(h_coarse[:, 2:])

        # Refinement passes
        losses_refine = []
        current_queries = queries

        for pass_idx in range(self.num_refine_passes):
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, current_queries[:, :-1]], dim=1)

            z_refine = torch.stack([self.encoder.query_attend(shifted_q[:, t], all_caches[t]) for t in range(T)], dim=1)
            z_refine = self.dino_to_llm(z_refine)
            z_refine = z_refine / (z_refine.std() + 1e-6) * self.visual_scale

            refine_token = self.refine_tokens[pass_idx].expand(B, -1, -1)
            seq_refine = torch.cat([no_text, refine_token, z_refine], dim=1)

            h_refine = self.llm.model(inputs_embeds=seq_refine).last_hidden_state
            pred_refine = self.pred_head(h_refine[:, 1:1+T], prev_latents)
            loss_refine = F.mse_loss(pred_refine, target_latents)
            losses_refine.append(loss_refine)

            if pass_idx < self.num_refine_passes - 1:
                current_queries = self.llm_to_query(h_refine[:, 2:])

        loss_fine = losses_refine[-1]
        loss = loss_fine + 0.5 * loss_coarse

        return loss, loss_fine, loss_coarse, losses_refine


def run_experiment(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Experiment: {config['name']}")
    print(f"  num_refine_passes: {config['num_refine_passes']}")
    print(f"  num_frames: {config['num_frames']}")
    print(f"  steps: {config['steps']}")
    print(f"{'='*70}")

    if HAS_WANDB and config.get('use_wandb', True):
        wandb.init(
            project="foveated-vlm-multipass",
            name=config['name'],
            config=config,
            reinit=True,
        )

    model = MultiPassModel(
        num_refine_passes=config['num_refine_passes'],
        freeze_dino=True,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    dataset = LocalVideoDataset("data/frames", "data/latents", num_frames=config['num_frames'])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get('lr', 3e-5),
        weight_decay=0.01,
    )

    scaler = GradScaler()
    model.train()

    step = 0
    running_fine = 0
    running_coarse = 0

    print(f"\nTraining...")

    while step < config['steps']:
        for frames, latents in dataloader:
            if step >= config['steps']:
                break

            frames = frames.to(device)
            latents = latents.to(device)

            optimizer.zero_grad()

            with autocast():
                loss, loss_fine, loss_coarse, losses_refine = model(frames, latents)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_fine = 0.95 * running_fine + 0.05 * loss_fine.item() if step > 0 else loss_fine.item()
            running_coarse = 0.95 * running_coarse + 0.05 * loss_coarse.item() if step > 0 else loss_coarse.item()
            ratio = running_coarse / (running_fine + 1e-8)

            if HAS_WANDB and config.get('use_wandb', True):
                log_dict = {
                    'loss': loss.item(),
                    'loss_fine': loss_fine.item(),
                    'loss_coarse': loss_coarse.item(),
                    'ratio': ratio,
                }
                for i, l in enumerate(losses_refine):
                    log_dict[f'loss_refine_{i}'] = l.item()
                wandb.log(log_dict, step=step)

            if step % 100 == 0:
                refine_str = ", ".join([f"r{i}={l.item():.3f}" for i, l in enumerate(losses_refine)])
                print(f"[{step:5d}] fine={running_fine:.4f}, coarse={running_coarse:.4f}, ratio={ratio:.4f}, {refine_str}")

            step += 1

    final_ratio = running_coarse / (running_fine + 1e-8)
    print(f"\n{'='*70}")
    print(f"FINAL: {config['name']}")
    print(f"  loss_fine: {running_fine:.4f}")
    print(f"  loss_coarse: {running_coarse:.4f}")
    print(f"  ratio: {final_ratio:.4f}")
    print(f"{'='*70}")

    if HAS_WANDB and config.get('use_wandb', True):
        wandb.log({'final_fine': running_fine, 'final_coarse': running_coarse, 'final_ratio': final_ratio})
        wandb.finish()

    return {'name': config['name'], 'final_fine': running_fine, 'final_coarse': running_coarse, 'final_ratio': final_ratio}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1500, help='Steps per experiment')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    use_wandb = not args.no_wandb and HAS_WANDB

    experiments = [
        {'name': 'A_1pass_8frames', 'num_refine_passes': 1, 'num_frames': 8, 'steps': args.steps, 'use_wandb': use_wandb},
        {'name': 'B_2pass_8frames', 'num_refine_passes': 2, 'num_frames': 8, 'steps': args.steps, 'use_wandb': use_wandb},
        {'name': 'C_3pass_8frames', 'num_refine_passes': 3, 'num_frames': 8, 'steps': args.steps, 'use_wandb': use_wandb},
        {'name': 'D_1pass_16frames', 'num_refine_passes': 1, 'num_frames': 16, 'steps': args.steps, 'use_wandb': use_wandb},
        {'name': 'E_3pass_16frames', 'num_refine_passes': 3, 'num_frames': 16, 'steps': args.steps, 'use_wandb': use_wandb},
    ]

    results = []
    for config in experiments:
        result = run_experiment(config)
        results.append(result)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        status = "✓" if r['final_ratio'] > 1.01 else "✗"
        print(f"{status} {r['name']:25s} ratio={r['final_ratio']:.4f} (fine={r['final_fine']:.4f}, coarse={r['final_coarse']:.4f})")
    print("="*70)


if __name__ == "__main__":
    main()
