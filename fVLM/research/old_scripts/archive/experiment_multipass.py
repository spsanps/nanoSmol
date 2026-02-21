#!/usr/bin/env python3
"""
Experiment: Multi-pass query refinement + longer sequences

Hypothesis 1: Single refinement pass isn't enough - need iterative refinement
Hypothesis 2: Longer sequences provide more temporal context for dynamic queries

This script tests:
- A: Baseline (1 pass, 8 frames)
- B: Multi-pass (3 refinement passes, 8 frames)
- C: Longer sequences (1 pass, 16 frames)
- D: Combined (3 passes, 16 frames)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import FoveatedEncoder
from src.model.prediction import PredictionHead
from src.data.streaming_dataset import StreamingWebVidDataset
from transformers import AutoModelForCausalLM
from diffusers import AutoencoderKL

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed, logging disabled")


class MultiPassFoveatedModel(nn.Module):
    """
    Foveated VLM with multi-pass query refinement.

    Key difference from original:
    - After Pass 2, we can do additional refinement passes
    - Each pass uses the previous hidden state to generate a better query
    """

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
        num_refine_passes: int = 1,  # Number of refinement passes after coarse
        freeze_dino: bool = True,
    ):
        super().__init__()

        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.query_dim = query_dim
        self.num_refine_passes = num_refine_passes

        # Vision encoder with deep query
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_model,
            query_dim=query_dim,
            output_dim=dino_dim,
            deep_query=True,
        )

        # Freeze DINO (ablation winner)
        if freeze_dino:
            for param in self.encoder.dino.parameters():
                param.requires_grad = False
            self.encoder.dino.eval()

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm.config.use_cache = False

        # Projections
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.visual_scale = 0.14
        self.llm_to_query = nn.Linear(llm_dim, query_dim)

        # Prediction head
        self.pred_head = PredictionHead(h_dim=llm_dim, latent_channels=4)

        # Learned queries (std=1.0)
        self.q_static = nn.Parameter(torch.randn(1, query_dim))
        self.q_init = nn.Parameter(torch.randn(1, query_dim))
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Pass tokens - one for coarse, one for each refinement pass
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.refine_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
            for _ in range(num_refine_passes)
        ])

        self.no_text_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

    def forward(self, raw_frames: torch.Tensor, vae_latents: torch.Tensor):
        """
        Multi-pass forward.

        Returns:
            loss: Combined loss
            loss_fine: Final refinement pass loss
            loss_coarse: Coarse pass loss
            losses_refine: List of intermediate refinement losses
        """
        B, T = raw_frames.shape[:2]
        device = raw_frames.device

        # === Encode all frames ===
        frames_flat = raw_frames.reshape(B * T, 3, 256, 256)
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

        # Targets and conditioning
        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
        prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
        target_latents = vae_latents

        # === Pass 1: Coarse (static query) ===
        q_static = self.q_static.expand(B, -1)
        z_coarse_list = [self.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
        z_coarse = torch.stack(z_coarse_list, dim=1)
        z_coarse = self.dino_to_llm(z_coarse)
        z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale

        # LLM forward for coarse
        no_text = self.no_text_token.expand(B, -1, -1)
        coarse_token = self.coarse_token.expand(B, -1, -1)
        seq_coarse = torch.cat([no_text, coarse_token, z_coarse], dim=1)

        outputs_coarse = self.llm.model(inputs_embeds=seq_coarse)
        h_coarse = outputs_coarse.last_hidden_state

        # Coarse prediction and loss
        h_coarse_pred = h_coarse[:, 1:1+T]  # After no_text token
        pred_coarse = self.pred_head(h_coarse_pred, prev_latents)
        loss_coarse = F.mse_loss(pred_coarse, target_latents)

        # Get initial queries from coarse pass
        queries = self.llm_to_query(h_coarse[:, 2:])  # After no_text and coarse_token

        # === Refinement passes ===
        losses_refine = []
        current_queries = queries

        for pass_idx in range(self.num_refine_passes):
            # Shift queries (q_t predicts frame t+1)
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, current_queries[:, :-1]], dim=1)

            # Attend with refined queries
            z_refine_list = [self.encoder.query_attend(shifted_q[:, t], all_caches[t]) for t in range(T)]
            z_refine = torch.stack(z_refine_list, dim=1)
            z_refine = self.dino_to_llm(z_refine)
            z_refine = z_refine / (z_refine.std() + 1e-6) * self.visual_scale

            # LLM forward for this refinement pass
            refine_token = self.refine_tokens[pass_idx].expand(B, -1, -1)
            seq_refine = torch.cat([no_text, refine_token, z_refine], dim=1)

            outputs_refine = self.llm.model(inputs_embeds=seq_refine)
            h_refine = outputs_refine.last_hidden_state

            # Prediction and loss for this pass
            h_refine_pred = h_refine[:, 1:1+T]
            pred_refine = self.pred_head(h_refine_pred, prev_latents)
            loss_refine = F.mse_loss(pred_refine, target_latents)
            losses_refine.append(loss_refine)

            # Generate queries for next pass (if not last)
            if pass_idx < self.num_refine_passes - 1:
                current_queries = self.llm_to_query(h_refine[:, 2:])

        # Final loss is the last refinement pass
        loss_fine = losses_refine[-1]

        # Combined loss: fine + coarse + intermediate passes (with decreasing weight)
        loss = loss_fine + 0.5 * loss_coarse
        for i, l in enumerate(losses_refine[:-1]):
            weight = 0.3 * (0.5 ** i)  # Decreasing weight for earlier passes
            loss = loss + weight * l

        return loss, loss_fine, loss_coarse, losses_refine


def run_experiment(config: dict):
    """Run a single experiment configuration."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Experiment: {config['name']}")
    print(f"  num_refine_passes: {config['num_refine_passes']}")
    print(f"  num_frames: {config['num_frames']}")
    print(f"  steps: {config['steps']}")
    print(f"{'='*70}")

    # Initialize wandb
    if HAS_WANDB and config.get('use_wandb', True):
        wandb.init(
            project="foveated-vlm-experiments",
            name=config['name'],
            config=config,
            reinit=True,
        )

    # Create model
    model = MultiPassFoveatedModel(
        num_refine_passes=config['num_refine_passes'],
        freeze_dino=True,
    ).to(device)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    # Load VAE for dataset
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16,
    ).to(device)
    vae.eval()

    # Dataset
    dataset = StreamingWebVidDataset(
        vae=vae,
        num_frames=config['num_frames'],
        frame_size=256,
        device=str(device),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get('lr', 3e-5),
        weight_decay=0.01,
    )

    scaler = GradScaler()

    # Training loop
    model.train()
    step = 0
    running_fine = 0
    running_coarse = 0

    print(f"\nStarting training...")

    for sample in dataset:
        if step >= config['steps']:
            break

        if sample is None:
            continue

        frames = sample['frames'].unsqueeze(0).to(device)
        latents = sample['latents'].unsqueeze(0).to(device)

        optimizer.zero_grad()

        with autocast():
            loss, loss_fine, loss_coarse, losses_refine = model(frames, latents)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        running_fine = 0.99 * running_fine + 0.01 * loss_fine.item() if step > 0 else loss_fine.item()
        running_coarse = 0.99 * running_coarse + 0.01 * loss_coarse.item() if step > 0 else loss_coarse.item()
        ratio = running_coarse / (running_fine + 1e-8)

        # Log to wandb
        if HAS_WANDB and config.get('use_wandb', True):
            log_dict = {
                'loss': loss.item(),
                'loss_fine': loss_fine.item(),
                'loss_coarse': loss_coarse.item(),
                'ratio': ratio,
                'running_fine': running_fine,
                'running_coarse': running_coarse,
            }
            for i, l in enumerate(losses_refine):
                log_dict[f'loss_refine_{i}'] = l.item()
            wandb.log(log_dict, step=step)

        if step % 50 == 0:
            refine_str = ", ".join([f"r{i}={l.item():.3f}" for i, l in enumerate(losses_refine)])
            print(f"[{step:5d}] fine={running_fine:.3f}, coarse={running_coarse:.3f}, "
                  f"ratio={ratio:.4f}, {refine_str}")

        step += 1

    # Final results
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000, help='Steps per experiment')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'baseline', 'multipass', 'longer', 'combined'],
                       help='Which experiment to run')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    args = parser.parse_args()

    use_wandb = not args.no_wandb and HAS_WANDB

    experiments = {
        'baseline': {
            'name': 'A_baseline_1pass_8frames',
            'num_refine_passes': 1,
            'num_frames': 8,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        'multipass': {
            'name': 'B_multipass_3pass_8frames',
            'num_refine_passes': 3,
            'num_frames': 8,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        'longer': {
            'name': 'C_longer_1pass_16frames',
            'num_refine_passes': 1,
            'num_frames': 16,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
        'combined': {
            'name': 'D_combined_3pass_16frames',
            'num_refine_passes': 3,
            'num_frames': 16,
            'steps': args.steps,
            'use_wandb': use_wandb,
        },
    }

    results = []

    if args.experiment == 'all':
        for name, config in experiments.items():
            result = run_experiment(config)
            results.append(result)
    else:
        result = run_experiment(experiments[args.experiment])
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for r in results:
        print(f"{r['name']:40s} ratio={r['final_ratio']:.4f} (fine={r['final_fine']:.3f}, coarse={r['final_coarse']:.3f})")
    print("="*70)


if __name__ == "__main__":
    main()
