#!/usr/bin/env python3
"""
Diagnostic script to understand why ratio stays at 1.0.

Checks:
1. Are queries different per frame?
2. Is deep query mode working vs shallow?
3. Is prev_latents a shortcut?
4. Are features different before normalization?
5. What does attention actually look like?
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import FoveatedEncoder
from src.model.prediction import PredictionHead
from src.model.foveated_vlm import FoveatedVideoModel
from transformers import AutoModelForCausalLM


class LocalVideoDataset(Dataset):
    def __init__(self, frames_dir: str, latents_dir: str, num_frames: int = 8):
        self.frames_dir = Path(frames_dir)
        self.latents_dir = Path(latents_dir)
        self.num_frames = num_frames
        frame_ids = {f.stem for f in self.frames_dir.glob("*.pt")}
        latent_ids = {f.stem for f in self.latents_dir.glob("*.pt")}
        self.video_ids = sorted(frame_ids & latent_ids)

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


def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("="*70)

    # Load a batch
    dataset = LocalVideoDataset("data/frames", "data/latents", num_frames=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    frames, latents = next(iter(dataloader))
    frames = frames.to(device)
    latents = latents.to(device)
    B, T = frames.shape[:2]
    print(f"Batch: {B} videos, {T} frames each")
    print("="*70)

    # Load the REAL model
    print("\n1. Loading FoveatedVideoModel...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)
    model.eval()

    print(f"   deep_query: {model.encoder.deep_query}")
    print(f"   q_static std: {model.q_static.std().item():.4f}")
    print(f"   q_init std: {model.q_init.std().item():.4f}")

    # =========================================================================
    print("\n" + "="*70)
    print("2. DIAGNOSTIC: Encode frames and check cache structure")
    print("="*70)

    with torch.no_grad():
        # Encode all frames
        frames_flat = frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = model.encoder.encode_patches(frames_flat)

        # Check cache structure
        print(f"   Cache keys: {cache_flat.keys()}")
        if 'kv_cache' in cache_flat:
            print(f"   KV cache present: Yes, {len(cache_flat['kv_cache'])} layers")
            print(f"   Deep query mode is ACTIVE")
        else:
            print(f"   KV cache present: No")
            print(f"   WARNING: Deep query mode NOT active!")

        # Reshape for per-frame access
        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)
        print(f"   Patch features shape: {patch_features.shape}")

        # Build per-frame caches
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
            for t in range(T):
                all_caches.append({'patch_features': patch_features[:, t]})

    # =========================================================================
    print("\n" + "="*70)
    print("3. DIAGNOSTIC: Coarse pass - extract features and queries")
    print("="*70)

    with torch.no_grad():
        # Get text embeddings (empty for self-supervised)
        text_embeds = model.get_empty_text_embeds(B)
        N_text = text_embeds.shape[1]

        # Pass 1: Coarse with static query
        q_static = model.q_static.expand(B, -1)

        z_coarse_list = []
        for t in range(T):
            z_t = model.encoder.query_attend(q_static, all_caches[t])
            z_coarse_list.append(z_t)
        z_coarse_raw = torch.stack(z_coarse_list, dim=1)  # Before projection

        z_coarse = model.dino_to_llm(z_coarse_raw)
        z_coarse_norm = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

        print(f"   z_coarse_raw shape: {z_coarse_raw.shape}")
        print(f"   z_coarse_raw mean: {z_coarse_raw.mean().item():.4f}, std: {z_coarse_raw.std().item():.4f}")

        # LLM forward for coarse pass
        coarse_token = model.coarse_token.expand(B, -1, -1)
        seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse_norm], dim=1)

        h_pass1 = model.llm.model(inputs_embeds=seq_pass1).last_hidden_state

        # Extract queries
        h_for_queries = h_pass1[:, N_text + 1:]  # After text and coarse token
        queries = model.llm_to_query(h_for_queries)

        print(f"\n   Queries shape: {queries.shape}")
        print(f"   Queries mean: {queries.mean().item():.4f}, std: {queries.std().item():.4f}")

    # =========================================================================
    print("\n" + "="*70)
    print("4. DIAGNOSTIC: Query similarity across frames")
    print("="*70)

    with torch.no_grad():
        print("   Cosine similarity between consecutive frame queries:")
        for t in range(T-1):
            sim = F.cosine_similarity(queries[:, t], queries[:, t+1], dim=-1).mean()
            print(f"      Frame {t} vs {t+1}: {sim.item():.4f}")

        print("\n   Query similarity to learned queries:")
        for t in range(min(4, T)):  # First 4 frames
            sim_static = F.cosine_similarity(queries[:, t], q_static, dim=-1).mean()
            sim_init = F.cosine_similarity(queries[:, t], model.q_init.expand(B, -1), dim=-1).mean()
            print(f"      Frame {t}: vs q_static={sim_static.item():.4f}, vs q_init={sim_init.item():.4f}")

        # Are queries diverse or all the same?
        query_variance = queries.var(dim=1).mean()
        print(f"\n   Query variance across frames: {query_variance.item():.6f}")

    # =========================================================================
    print("\n" + "="*70)
    print("5. DIAGNOSTIC: Fine pass - extract features with dynamic queries")
    print("="*70)

    with torch.no_grad():
        # Shift queries
        q_init = model.q_init.expand(B, -1).unsqueeze(1)
        shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        z_fine_list = []
        for t in range(T):
            z_t = model.encoder.query_attend(shifted_q[:, t], all_caches[t])
            z_fine_list.append(z_t)
        z_fine_raw = torch.stack(z_fine_list, dim=1)

        z_fine = model.dino_to_llm(z_fine_raw)
        z_fine_norm = z_fine / (z_fine.std() + 1e-6) * model.visual_scale

        print(f"   z_fine_raw shape: {z_fine_raw.shape}")
        print(f"   z_fine_raw mean: {z_fine_raw.mean().item():.4f}, std: {z_fine_raw.std().item():.4f}")

    # =========================================================================
    print("\n" + "="*70)
    print("6. DIAGNOSTIC: Feature similarity (z_coarse vs z_fine)")
    print("="*70)

    with torch.no_grad():
        print("   Per-frame cosine similarity (BEFORE projection):")
        for t in range(T):
            sim = F.cosine_similarity(z_coarse_raw[:, t], z_fine_raw[:, t], dim=-1).mean()
            print(f"      Frame {t}: {sim.item():.4f}")

        # Overall
        overall_sim_raw = F.cosine_similarity(
            z_coarse_raw.reshape(-1, z_coarse_raw.shape[-1]),
            z_fine_raw.reshape(-1, z_fine_raw.shape[-1]),
            dim=-1
        ).mean()
        print(f"\n   Overall similarity (raw): {overall_sim_raw.item():.4f}")

        print("\n   Per-frame cosine similarity (AFTER projection & norm):")
        for t in range(T):
            sim = F.cosine_similarity(z_coarse_norm[:, t], z_fine_norm[:, t], dim=-1).mean()
            print(f"      Frame {t}: {sim.item():.4f}")

        overall_sim_norm = F.cosine_similarity(
            z_coarse_norm.reshape(-1, z_coarse_norm.shape[-1]),
            z_fine_norm.reshape(-1, z_fine_norm.shape[-1]),
            dim=-1
        ).mean()
        print(f"\n   Overall similarity (normalized): {overall_sim_norm.item():.4f}")

    # =========================================================================
    print("\n" + "="*70)
    print("7. DIAGNOSTIC: LLM hidden states (h_coarse vs h_fine)")
    print("="*70)

    with torch.no_grad():
        # h_coarse
        h_coarse_for_pred = h_pass1[:, N_text:N_text + T]

        # h_fine
        fine_token = model.fine_token.expand(B, -1, -1)
        seq_pass2 = torch.cat([text_embeds, fine_token, z_fine_norm], dim=1)
        h_pass2 = model.llm.model(inputs_embeds=seq_pass2).last_hidden_state
        h_fine_for_pred = h_pass2[:, N_text:N_text + T]

        print(f"   h_coarse shape: {h_coarse_for_pred.shape}")
        print(f"   h_fine shape: {h_fine_for_pred.shape}")

        print("\n   Per-frame h similarity:")
        for t in range(T):
            sim = F.cosine_similarity(h_coarse_for_pred[:, t], h_fine_for_pred[:, t], dim=-1).mean()
            print(f"      Frame {t}: {sim.item():.4f}")

        overall_h_sim = F.cosine_similarity(
            h_coarse_for_pred.reshape(-1, h_coarse_for_pred.shape[-1]),
            h_fine_for_pred.reshape(-1, h_fine_for_pred.shape[-1]),
            dim=-1
        ).mean()
        print(f"\n   Overall h similarity: {overall_h_sim.item():.4f}")

    # =========================================================================
    print("\n" + "="*70)
    print("8. DIAGNOSTIC: Prediction head ablation")
    print("="*70)

    with torch.no_grad():
        # Setup
        z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
        prev_latents = torch.cat([z_vae_init, latents[:, :-1]], dim=1)
        target_latents = latents

        # Normal predictions
        pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)
        pred_fine = model.pred_head(h_fine_for_pred, prev_latents)

        loss_coarse = F.mse_loss(pred_coarse, target_latents)
        loss_fine = F.mse_loss(pred_fine, target_latents)

        print(f"   Loss coarse: {loss_coarse.item():.4f}")
        print(f"   Loss fine: {loss_fine.item():.4f}")
        print(f"   Ratio: {(loss_coarse / loss_fine).item():.4f}")

        # Ablation: zero prev_latents
        pred_zero_prev = model.pred_head(h_fine_for_pred, torch.zeros_like(prev_latents))
        loss_zero_prev = F.mse_loss(pred_zero_prev, target_latents)

        # Ablation: random h
        pred_rand_h = model.pred_head(torch.randn_like(h_fine_for_pred), prev_latents)
        loss_rand_h = F.mse_loss(pred_rand_h, target_latents)

        print(f"\n   Zero prev_latents loss: {loss_zero_prev.item():.4f}")
        print(f"   Random h loss: {loss_rand_h.item():.4f}")

        h_contribution = (loss_zero_prev - loss_fine) / loss_zero_prev * 100 if loss_zero_prev > 0 else 0
        prev_contribution = (loss_rand_h - loss_fine) / loss_rand_h * 100 if loss_rand_h > 0 else 0
        print(f"\n   h reduces loss by: {h_contribution:.1f}%")
        print(f"   prev_latents reduces loss by: {prev_contribution:.1f}%")

    # =========================================================================
    print("\n" + "="*70)
    print("9. DIAGNOSTIC: Deep query verification")
    print("="*70)

    with torch.no_grad():
        # Compare outputs with very different queries
        q1 = torch.randn(B, model.q_static.shape[-1], device=device) * 5  # Large magnitude
        q2 = -q1  # Opposite direction

        z1 = model.encoder.query_attend(q1, all_caches[0])
        z2 = model.encoder.query_attend(q2, all_caches[0])

        sim_opposite = F.cosine_similarity(z1, z2, dim=-1).mean()
        print(f"   Opposite queries (q vs -q) produce features with similarity: {sim_opposite.item():.4f}")

        # Orthogonal queries
        q3 = torch.randn(B, model.q_static.shape[-1], device=device)
        q3 = q3 - (q3 * q1).sum(dim=-1, keepdim=True) / (q1 * q1).sum(dim=-1, keepdim=True) * q1  # Gram-Schmidt

        z3 = model.encoder.query_attend(q3, all_caches[0])
        sim_ortho = F.cosine_similarity(z1, z3, dim=-1).mean()
        print(f"   Orthogonal queries produce features with similarity: {sim_ortho.item():.4f}")

    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"   Queries vary across frames: variance = {query_variance.item():.6f}")
    print(f"   z_coarse vs z_fine similarity (raw): {overall_sim_raw.item():.4f}")
    print(f"   z_coarse vs z_fine similarity (norm): {overall_sim_norm.item():.4f}")
    print(f"   h_coarse vs h_fine similarity: {overall_h_sim.item():.4f}")
    print(f"   Loss ratio (coarse/fine): {(loss_coarse / loss_fine).item():.4f}")
    print("="*70)


if __name__ == "__main__":
    run_diagnostics()
