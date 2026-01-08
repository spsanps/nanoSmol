#!/usr/bin/env python3
"""
Experiment: Query diversity fixes.

Tests:
1. Baseline (LLM-generated queries)
2. Fixed q_init (bypass LLM, use q_init for all fine passes)
3. Query diversity loss (push queries away from q_static)
4. Orthogonal queries (project out q_static component)
5. Random queries (sanity check - should be different)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
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
    def __init__(self, frames_dir: str, latents_dir: str, num_frames: int = 8):
        self.frames_dir = Path(frames_dir)
        self.latents_dir = Path(latents_dir)
        self.num_frames = num_frames
        frame_ids = {f.stem for f in self.frames_dir.glob("*.pt")}
        latent_ids = {f.stem for f in self.latents_dir.glob("*.pt")}
        self.video_ids = sorted(frame_ids & latent_ids)
        print(f"Found {len(self.video_ids)} videos")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = torch.load(self.frames_dir / f"{video_id}.pt")
        latents = torch.load(self.latents_dir / f"{video_id}.pt")
        T = frames.shape[0]
        if T > self.num_frames:
            start = random.randint(0, T - self.num_frames)
            frames = frames[start:start + self.num_frames]
            latents = latents[start:start + self.num_frames]
        elif T < self.num_frames:
            pad = self.num_frames - T
            frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)
            latents = torch.cat([latents, latents[-1:].repeat(pad, 1, 1, 1)], dim=0)
        return frames, latents


class QueryDiversityModel(nn.Module):
    """Model with configurable query strategies."""

    def __init__(
        self,
        query_mode: str = "llm",  # "llm", "fixed", "diversity", "orthogonal", "random"
        diversity_weight: float = 0.1,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
    ):
        super().__init__()

        self.query_mode = query_mode
        self.diversity_weight = diversity_weight
        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.query_dim = query_dim

        # Vision encoder
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_model,
            query_dim=query_dim,
            output_dim=dino_dim,
            deep_query=True,
        )

        # Freeze DINO
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

        # Queries - make q_init orthogonal to q_static by design
        self.q_static = nn.Parameter(torch.randn(1, query_dim))

        # Initialize q_init to be orthogonal to q_static
        q_init_raw = torch.randn(1, query_dim)
        q_static_norm = self.q_static.data / self.q_static.data.norm()
        q_init_raw = q_init_raw - (q_init_raw @ q_static_norm.T) * q_static_norm
        q_init_raw = q_init_raw / q_init_raw.norm() * (query_dim ** 0.5)  # Scale to std~1
        self.q_init = nn.Parameter(q_init_raw)

        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Tokens
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.fine_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
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
                        'K': K_all[:, t],
                        'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({
                    'patch_features': patch_features[:, t],
                    'kv_cache': frame_kv_cache,
                })
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Targets
        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
        prev_latents = torch.cat([z_vae_init, latents[:, :-1]], dim=1)
        target_latents = latents

        # Pass 1: Coarse (static query)
        q_static = self.q_static.expand(B, -1)
        z_coarse = torch.stack([self.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)
        z_coarse_proj = self.dino_to_llm(z_coarse)
        z_coarse_norm = z_coarse_proj / (z_coarse_proj.std() + 1e-6) * self.visual_scale

        no_text = self.no_text_token.expand(B, -1, -1)
        coarse_token = self.coarse_token.expand(B, -1, -1)
        seq_coarse = torch.cat([no_text, coarse_token, z_coarse_norm], dim=1)

        h_coarse = self.llm.model(inputs_embeds=seq_coarse).last_hidden_state
        h_coarse_pred = h_coarse[:, 1:1+T]
        pred_coarse = self.pred_head(h_coarse_pred, prev_latents)
        loss_coarse = F.mse_loss(pred_coarse, target_latents)

        # Generate queries based on mode
        loss_diversity = torch.tensor(0.0, device=device)

        if self.query_mode == "llm":
            # Standard: LLM generates queries
            queries = self.llm_to_query(h_coarse[:, 2:])
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        elif self.query_mode == "fixed":
            # Fixed: Use q_init for ALL frames (bypass LLM)
            shifted_q = self.q_init.expand(B, T, -1)

        elif self.query_mode == "diversity":
            # Diversity loss: Push queries away from q_static
            queries = self.llm_to_query(h_coarse[:, 2:])
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

            # Penalize similarity to q_static
            q_static_expanded = self.q_static.expand(B, T, -1)
            similarity = F.cosine_similarity(shifted_q, q_static_expanded, dim=-1).mean()
            loss_diversity = self.diversity_weight * similarity.abs()

        elif self.query_mode == "orthogonal":
            # Orthogonal: Project out q_static component from queries
            queries = self.llm_to_query(h_coarse[:, 2:])
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

            # Project out q_static component (Gram-Schmidt)
            q_static_norm = F.normalize(self.q_static, dim=-1)  # [1, D]
            projection = (shifted_q @ q_static_norm.T) * q_static_norm  # [B, T, D]
            shifted_q = shifted_q - projection

        elif self.query_mode == "random":
            # Random: Use random queries (sanity check)
            shifted_q = torch.randn(B, T, self.query_dim, device=device)

        else:
            raise ValueError(f"Unknown query_mode: {self.query_mode}")

        # Pass 2: Fine (dynamic queries)
        z_fine = torch.stack([self.encoder.query_attend(shifted_q[:, t], all_caches[t]) for t in range(T)], dim=1)
        z_fine_proj = self.dino_to_llm(z_fine)
        z_fine_norm = z_fine_proj / (z_fine_proj.std() + 1e-6) * self.visual_scale

        fine_token = self.fine_token.expand(B, -1, -1)
        seq_fine = torch.cat([no_text, fine_token, z_fine_norm], dim=1)

        h_fine = self.llm.model(inputs_embeds=seq_fine).last_hidden_state
        h_fine_pred = h_fine[:, 1:1+T]
        pred_fine = self.pred_head(h_fine_pred, prev_latents)
        loss_fine = F.mse_loss(pred_fine, target_latents)

        # Combined loss
        loss = loss_fine + 0.5 * loss_coarse + loss_diversity

        # Compute query similarity for logging
        with torch.no_grad():
            q_sim = F.cosine_similarity(shifted_q, self.q_static.expand(B, T, -1), dim=-1).mean()
            z_sim = F.cosine_similarity(z_coarse.reshape(-1, z_coarse.shape[-1]),
                                        z_fine.reshape(-1, z_fine.shape[-1]), dim=-1).mean()

        return loss, loss_fine, loss_coarse, loss_diversity, q_sim, z_sim


def run_experiment(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Experiment: {config['name']}")
    print(f"  query_mode: {config['query_mode']}")
    print(f"  steps: {config['steps']}")
    print(f"{'='*70}")

    if HAS_WANDB and config.get('use_wandb', True):
        wandb.init(
            project="foveated-vlm-query-diversity",
            name=config['name'],
            config=config,
            reinit=True,
        )

    model = QueryDiversityModel(
        query_mode=config['query_mode'],
        diversity_weight=config.get('diversity_weight', 0.1),
    ).to(device)

    # Verify q_init is orthogonal to q_static
    with torch.no_grad():
        ortho_check = F.cosine_similarity(model.q_static, model.q_init, dim=-1).item()
        print(f"q_static vs q_init similarity: {ortho_check:.4f} (should be ~0)")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    dataset = LocalVideoDataset("data/frames", "data/latents", num_frames=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
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
                loss, loss_fine, loss_coarse, loss_div, q_sim, z_sim = model(frames, latents)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_fine = 0.95 * running_fine + 0.05 * loss_fine.item() if step > 0 else loss_fine.item()
            running_coarse = 0.95 * running_coarse + 0.05 * loss_coarse.item() if step > 0 else loss_coarse.item()
            ratio = running_coarse / (running_fine + 1e-8)

            if HAS_WANDB and config.get('use_wandb', True):
                wandb.log({
                    'loss': loss.item(),
                    'loss_fine': loss_fine.item(),
                    'loss_coarse': loss_coarse.item(),
                    'loss_diversity': loss_div.item(),
                    'q_similarity': q_sim.item(),
                    'z_similarity': z_sim.item(),
                    'ratio': ratio,
                }, step=step)

            if step % 50 == 0:
                print(f"[{step:5d}] fine={running_fine:.4f}, coarse={running_coarse:.4f}, "
                      f"ratio={ratio:.4f}, q_sim={q_sim.item():.3f}, z_sim={z_sim.item():.3f}")

            step += 1

    final_ratio = running_coarse / (running_fine + 1e-8)
    print(f"\n{'='*70}")
    print(f"FINAL: {config['name']}")
    print(f"  loss_fine: {running_fine:.4f}")
    print(f"  loss_coarse: {running_coarse:.4f}")
    print(f"  ratio: {final_ratio:.4f}")
    print(f"{'='*70}")

    if HAS_WANDB and config.get('use_wandb', True):
        wandb.log({
            'final_fine': running_fine,
            'final_coarse': running_coarse,
            'final_ratio': final_ratio,
        })
        wandb.finish()

    return {
        'name': config['name'],
        'final_fine': running_fine,
        'final_coarse': running_coarse,
        'final_ratio': final_ratio,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500, help='Steps per experiment')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    use_wandb = not args.no_wandb and HAS_WANDB

    experiments = [
        # Baseline
        {
            'name': '1_baseline_llm',
            'query_mode': 'llm',
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Fixed q_init (bypass LLM)
        {
            'name': '2_fixed_qinit',
            'query_mode': 'fixed',
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Diversity loss
        {
            'name': '3_diversity_loss',
            'query_mode': 'diversity',
            'diversity_weight': 0.1,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Orthogonal projection
        {
            'name': '4_orthogonal',
            'query_mode': 'orthogonal',
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Random queries (sanity check)
        {
            'name': '5_random',
            'query_mode': 'random',
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
    ]

    results = []
    for config in experiments:
        result = run_experiment(config)
        results.append(result)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Experiment':<25} {'Ratio':>8} {'Fine':>8} {'Coarse':>8}")
    print("-"*70)
    for r in results:
        status = "✓" if r['final_ratio'] > 1.02 else "✗"
        print(f"{status} {r['name']:<23} {r['final_ratio']:>8.4f} {r['final_fine']:>8.4f} {r['final_coarse']:>8.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
