#!/usr/bin/env python3
"""
Mini-experiments: Testing attention fixes to create fine/coarse gap.

Tests:
1. Baseline (current behavior)
2. Temperature = 0.1 (sharper attention)
3. Contrastive loss (push z_fine away from z_coarse)
4. Top-k hard attention (k=16)
5. Combined: temp=0.1 + contrastive
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
    """Load precomputed frames and latents from local storage."""

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


class AttentionFixModel(nn.Module):
    """Model with configurable attention fixes."""

    def __init__(
        self,
        temperature: float = 1.0,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
        use_topk: bool = False,
        topk_k: int = 16,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
    ):
        super().__init__()

        self.temperature = temperature
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.use_topk = use_topk
        self.topk_k = topk_k
        self.dino_dim = dino_dim
        self.llm_dim = llm_dim

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

        # Queries
        self.q_static = nn.Parameter(torch.randn(1, query_dim))
        self.q_init = nn.Parameter(torch.randn(1, query_dim))
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Tokens
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.refine_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.no_text_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

    def query_attend_with_fixes(self, query, cache):
        """Query attention with temperature and top-k fixes."""
        patch_features = cache['patch_features']  # [B, N, D]
        B, N, D = patch_features.shape

        # Project query
        q_proj = self.encoder.query_input_proj(query)  # [B, D]

        # Compute attention scores
        scores = torch.einsum('bd,bnd->bn', q_proj, patch_features)  # [B, N]

        # Apply temperature
        scores = scores / self.temperature

        # Softmax
        attn = F.softmax(scores, dim=-1)  # [B, N]

        # Optional top-k
        if self.use_topk:
            topk = torch.topk(attn, k=min(self.topk_k, N), dim=-1)
            attn_hard = torch.zeros_like(attn)
            attn_hard.scatter_(-1, topk.indices, topk.values)
            attn = attn_hard / (attn_hard.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum
        output = torch.einsum('bn,bnd->bd', attn, patch_features)  # [B, D]

        # Output projection
        output = self.encoder.query_output_proj(output)

        return output

    def forward(self, frames: torch.Tensor, latents: torch.Tensor):
        B, T = frames.shape[:2]
        device = frames.device

        # Encode frames
        frames_flat = frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Build simple caches (just patch features for our custom attention)
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Targets
        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
        prev_latents = torch.cat([z_vae_init, latents[:, :-1]], dim=1)
        target_latents = latents

        # Pass 1: Coarse (static query)
        q_static = self.q_static.expand(B, -1)
        z_coarse_list = [self.query_attend_with_fixes(q_static, all_caches[t]) for t in range(T)]
        z_coarse = torch.stack(z_coarse_list, dim=1)
        z_coarse_raw = z_coarse.clone()  # Save for contrastive
        z_coarse = self.dino_to_llm(z_coarse)
        z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale

        no_text = self.no_text_token.expand(B, -1, -1)
        coarse_token = self.coarse_token.expand(B, -1, -1)
        seq_coarse = torch.cat([no_text, coarse_token, z_coarse], dim=1)

        h_coarse = self.llm.model(inputs_embeds=seq_coarse).last_hidden_state
        pred_coarse = self.pred_head(h_coarse[:, 1:1+T], prev_latents)
        loss_coarse = F.mse_loss(pred_coarse, target_latents)

        queries = self.llm_to_query(h_coarse[:, 2:])

        # Pass 2: Fine (dynamic query)
        q_init = self.q_init.expand(B, -1).unsqueeze(1)
        shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        z_fine_list = [self.query_attend_with_fixes(shifted_q[:, t], all_caches[t]) for t in range(T)]
        z_fine = torch.stack(z_fine_list, dim=1)
        z_fine_raw = z_fine.clone()  # Save for contrastive
        z_fine = self.dino_to_llm(z_fine)
        z_fine = z_fine / (z_fine.std() + 1e-6) * self.visual_scale

        refine_token = self.refine_token.expand(B, -1, -1)
        seq_fine = torch.cat([no_text, refine_token, z_fine], dim=1)

        h_fine = self.llm.model(inputs_embeds=seq_fine).last_hidden_state
        pred_fine = self.pred_head(h_fine[:, 1:1+T], prev_latents)
        loss_fine = F.mse_loss(pred_fine, target_latents)

        # Combined loss
        loss = loss_fine + 0.5 * loss_coarse

        # Contrastive loss (push z_fine away from z_coarse)
        loss_contrastive = torch.tensor(0.0, device=device)
        if self.use_contrastive:
            # Cosine similarity between fine and coarse features
            z_fine_norm = F.normalize(z_fine_raw, dim=-1)
            z_coarse_norm = F.normalize(z_coarse_raw, dim=-1)
            similarity = (z_fine_norm * z_coarse_norm).sum(dim=-1).mean()
            loss_contrastive = self.contrastive_weight * similarity
            loss = loss + loss_contrastive

        # Compute feature similarity for logging
        with torch.no_grad():
            z_sim = F.cosine_similarity(z_fine_raw, z_coarse_raw, dim=-1).mean()

        return loss, loss_fine, loss_coarse, loss_contrastive, z_sim


def run_experiment(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Experiment: {config['name']}")
    print(f"  temperature: {config.get('temperature', 1.0)}")
    print(f"  contrastive: {config.get('use_contrastive', False)}")
    print(f"  topk: {config.get('use_topk', False)}")
    print(f"  steps: {config['steps']}")
    print(f"{'='*70}")

    if HAS_WANDB and config.get('use_wandb', True):
        wandb.init(
            project="foveated-vlm-attention-fixes",
            name=config['name'],
            config=config,
            reinit=True,
        )

    model = AttentionFixModel(
        temperature=config.get('temperature', 1.0),
        use_contrastive=config.get('use_contrastive', False),
        contrastive_weight=config.get('contrastive_weight', 0.1),
        use_topk=config.get('use_topk', False),
        topk_k=config.get('topk_k', 16),
    ).to(device)

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
    running_sim = 0

    print(f"\nTraining...")

    while step < config['steps']:
        for frames, latents in dataloader:
            if step >= config['steps']:
                break

            frames = frames.to(device)
            latents = latents.to(device)

            optimizer.zero_grad()

            with autocast():
                loss, loss_fine, loss_coarse, loss_contrast, z_sim = model(frames, latents)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_fine = 0.95 * running_fine + 0.05 * loss_fine.item() if step > 0 else loss_fine.item()
            running_coarse = 0.95 * running_coarse + 0.05 * loss_coarse.item() if step > 0 else loss_coarse.item()
            running_sim = 0.95 * running_sim + 0.05 * z_sim.item() if step > 0 else z_sim.item()
            ratio = running_coarse / (running_fine + 1e-8)

            if HAS_WANDB and config.get('use_wandb', True):
                wandb.log({
                    'loss': loss.item(),
                    'loss_fine': loss_fine.item(),
                    'loss_coarse': loss_coarse.item(),
                    'loss_contrast': loss_contrast.item(),
                    'z_similarity': z_sim.item(),
                    'ratio': ratio,
                }, step=step)

            if step % 50 == 0:
                print(f"[{step:5d}] fine={running_fine:.4f}, coarse={running_coarse:.4f}, "
                      f"ratio={ratio:.4f}, z_sim={running_sim:.4f}")

            step += 1

    final_ratio = running_coarse / (running_fine + 1e-8)
    print(f"\n{'='*70}")
    print(f"FINAL: {config['name']}")
    print(f"  loss_fine: {running_fine:.4f}")
    print(f"  loss_coarse: {running_coarse:.4f}")
    print(f"  ratio: {final_ratio:.4f}")
    print(f"  z_similarity: {running_sim:.4f}")
    print(f"{'='*70}")

    if HAS_WANDB and config.get('use_wandb', True):
        wandb.log({
            'final_fine': running_fine,
            'final_coarse': running_coarse,
            'final_ratio': final_ratio,
            'final_z_sim': running_sim,
        })
        wandb.finish()

    return {
        'name': config['name'],
        'final_fine': running_fine,
        'final_coarse': running_coarse,
        'final_ratio': final_ratio,
        'final_z_sim': running_sim,
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
            'name': '1_baseline',
            'temperature': 1.0,
            'use_contrastive': False,
            'use_topk': False,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Temperature only
        {
            'name': '2_temp_0.1',
            'temperature': 0.1,
            'use_contrastive': False,
            'use_topk': False,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Contrastive only
        {
            'name': '3_contrastive',
            'temperature': 1.0,
            'use_contrastive': True,
            'contrastive_weight': 0.1,
            'use_topk': False,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Top-k only
        {
            'name': '4_topk_16',
            'temperature': 1.0,
            'use_contrastive': False,
            'use_topk': True,
            'topk_k': 16,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Combined: temp + contrastive
        {
            'name': '5_temp_contrastive',
            'temperature': 0.1,
            'use_contrastive': True,
            'contrastive_weight': 0.1,
            'use_topk': False,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        # Combined: temp + topk
        {
            'name': '6_temp_topk',
            'temperature': 0.1,
            'use_contrastive': False,
            'use_topk': True,
            'topk_k': 16,
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
    print(f"{'Experiment':<25} {'Ratio':>8} {'Fine':>8} {'Coarse':>8} {'Z_Sim':>8}")
    print("-"*70)
    for r in results:
        status = "✓" if r['final_ratio'] > 1.02 else "✗"
        print(f"{status} {r['name']:<23} {r['final_ratio']:>8.4f} {r['final_fine']:>8.4f} "
              f"{r['final_coarse']:>8.4f} {r['final_z_sim']:>8.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
