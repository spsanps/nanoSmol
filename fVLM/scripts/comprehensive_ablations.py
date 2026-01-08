#!/usr/bin/env python3
"""
Comprehensive Ablation Experiments for Foveated VLM

Tests 7 configurations to determine best path forward:
1. A_baseline: Current model (temp=1.0)
2. B_temp_0.1: Sharper attention
3. C_contrastive: Push z_fine away from z_coarse (λ=0.1)
4. D_freeze_dino: Preserve pretrained feature diversity
5. E_hard_attn: Top-k=32 hard attention
6. F_entropy_reg: Entropy margin loss
7. G_combined: Best fixes together

Each experiment runs for 1500 steps using local precomputed data.
GPU memory kept under 20GB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
import json
import copy
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Local Dataset - Uses precomputed frames and latents
# ============================================================================

class LocalVideoDataset(Dataset):
    """Dataset using precomputed frames and latents."""

    def __init__(self, frames_dir: str, latents_dir: str):
        self.frames_dir = Path(frames_dir)
        self.latents_dir = Path(latents_dir)

        # Get matching files
        frame_files = set(f.stem for f in self.frames_dir.glob("*.pt"))
        latent_files = set(f.stem for f in self.latents_dir.glob("*.pt"))
        self.video_ids = sorted(frame_files & latent_files)

        print(f"Found {len(self.video_ids)} videos with both frames and latents")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        frames = torch.load(self.frames_dir / f"{video_id}.pt", weights_only=True)
        latents = torch.load(self.latents_dir / f"{video_id}.pt", weights_only=True)

        return {'frames': frames, 'latents': latents, 'video_id': video_id}


# ============================================================================
# Model Variants - Modified versions of FoveatedEncoder for ablations
# ============================================================================

class FoveatedEncoderWithAblations(nn.Module):
    """
    Vision encoder with configurable ablation options.
    """

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-small",
        query_dim: int = 384,
        output_dim: int = 384,
        deep_query: bool = True,
        temperature: float = 1.0,  # NEW: attention temperature
        hard_attn_k: int = 0,  # NEW: top-k hard attention (0 = soft)
    ):
        super().__init__()

        from transformers import AutoModel

        self.deep_query = deep_query
        self.temperature = temperature
        self.hard_attn_k = hard_attn_k

        # Load pretrained DINOv2
        self.dino = AutoModel.from_pretrained(dino_model_name)
        self.dino_dim = self.dino.config.hidden_size
        self.num_heads = self.dino.config.num_attention_heads
        self.head_dim = self.dino_dim // self.num_heads

        # Projections
        self.query_input_proj = nn.Linear(query_dim, self.dino_dim, bias=False)
        self.query_output_proj = nn.Linear(self.dino_dim, output_dim)

        self.register_buffer("_dummy", torch.zeros(1))

    @property
    def device(self):
        return self._dummy.device

    def encode_patches(self, images):
        """Encode images to patch tokens with KV caching for deep mode."""
        if self.deep_query:
            return self._encode_patches_deep(images)
        else:
            return self._encode_patches_shallow(images)

    def _encode_patches_shallow(self, images):
        outputs = self.dino(images, output_hidden_states=True)
        patch_features = outputs.last_hidden_state
        cache = {'patch_features': patch_features}
        return patch_features, cache

    def _encode_patches_deep(self, images):
        embeddings = self.dino.embeddings(images)
        hidden_states = embeddings
        kv_cache = []

        for layer in self.dino.encoder.layer:
            normed = layer.norm1(hidden_states)
            attn = layer.attention.attention
            K = attn.key(normed)
            V = attn.value(normed)
            kv_cache.append({'K': K, 'V': V, 'layer': layer})
            hidden_states = layer(hidden_states)

        patch_features = self.dino.layernorm(hidden_states)
        cache = {'patch_features': patch_features, 'kv_cache': kv_cache}
        return patch_features, cache

    def query_attend(self, query, cache):
        """Use query to attend over cached features with configurable temperature/hard attention."""
        if self.deep_query:
            return self._query_attend_deep(query, cache)
        else:
            return self._query_attend_shallow(query, cache)

    def _query_attend_shallow(self, query, cache):
        q_embed = self.query_input_proj(query)
        patch_features = cache['patch_features']

        q_embed = q_embed.unsqueeze(1)
        attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))

        # Apply temperature scaling
        attn_scores = attn_scores / (self.dino_dim ** 0.5) / self.temperature

        # Apply hard attention if specified
        if self.hard_attn_k > 0:
            # Keep only top-k attention scores
            k = min(self.hard_attn_k, attn_scores.shape[-1])
            topk_vals, topk_idx = attn_scores.topk(k, dim=-1)
            mask = torch.zeros_like(attn_scores).fill_(float('-inf'))
            mask.scatter_(-1, topk_idx, 0)
            attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        z = torch.bmm(attn_weights, patch_features)
        z = z.squeeze(1)
        z = self.query_output_proj(z)
        return z

    def _query_attend_deep(self, query, cache):
        kv_cache = cache['kv_cache']
        B = query.shape[0]

        q_hidden = self.query_input_proj(query)
        q_hidden = q_hidden.unsqueeze(1)

        for layer_cache in kv_cache:
            K = layer_cache['K']
            V = layer_cache['V']
            layer = layer_cache['layer']
            attn = layer.attention.attention

            q_normed = layer.norm1(q_hidden)
            Q = attn.query(q_normed)

            def reshape_for_heads(x):
                b, n, _ = x.shape
                return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

            Q = reshape_for_heads(Q)
            K_heads = reshape_for_heads(K)
            V_heads = reshape_for_heads(V)

            # Apply temperature scaling
            scale = math.sqrt(self.head_dim) * self.temperature
            attn_scores = torch.matmul(Q, K_heads.transpose(-2, -1)) / scale

            # Apply hard attention if specified
            if self.hard_attn_k > 0:
                k = min(self.hard_attn_k, attn_scores.shape[-1])
                topk_vals, topk_idx = attn_scores.topk(k, dim=-1)
                mask = torch.zeros_like(attn_scores).fill_(float('-inf'))
                mask.scatter_(-1, topk_idx, 0)
                attn_scores = attn_scores + mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V_heads)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(B, 1, self.dino_dim)
            attn_output = layer.attention.output.dense(attn_output)
            attn_output = layer.attention.output.dropout(attn_output)

            if hasattr(layer, 'layer_scale1'):
                attn_output = layer.layer_scale1(attn_output)

            q_hidden = q_hidden + attn_output

            q_normed2 = layer.norm2(q_hidden)
            ffn_output = layer.mlp(q_normed2)

            if hasattr(layer, 'layer_scale2'):
                ffn_output = layer.layer_scale2(ffn_output)

            q_hidden = q_hidden + ffn_output

        q_hidden = self.dino.layernorm(q_hidden)
        z = q_hidden.squeeze(1)
        z = self.query_output_proj(z)
        return z


# ============================================================================
# Model with Ablation Support
# ============================================================================

class FoveatedVideoModelAblation(nn.Module):
    """
    Foveated VLM with configurable ablation options.
    """

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
        lambda_coarse: float = 1.0,
        deep_query: bool = True,
        # Ablation options
        temperature: float = 1.0,
        lambda_contrastive: float = 0.0,
        freeze_dino: bool = False,
        hard_attn_k: int = 0,
        entropy_margin: float = 0.0,  # For entropy regularization
    ):
        super().__init__()

        from transformers import AutoModelForCausalLM
        from src.model.prediction import PredictionHead

        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.query_dim = query_dim
        self.lambda_coarse = lambda_coarse
        self.lambda_contrastive = lambda_contrastive
        self.entropy_margin = entropy_margin

        # Vision encoder with ablation support
        self.encoder = FoveatedEncoderWithAblations(
            dino_model_name=dino_model,
            query_dim=query_dim,
            output_dim=dino_dim,
            deep_query=deep_query,
            temperature=temperature,
            hard_attn_k=hard_attn_k,
        )

        # Freeze DINO if requested
        if freeze_dino:
            for param in self.encoder.dino.parameters():
                param.requires_grad = False

        # Core LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm.config.use_cache = False

        # Projections
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.visual_scale = 0.14
        self.llm_to_query = nn.Linear(llm_dim, query_dim)

        # Prediction head
        self.pred_head = PredictionHead(h_dim=llm_dim, latent_channels=4)

        # Learned parameters
        self.q_static = nn.Parameter(torch.randn(1, query_dim))
        self.q_init = nn.Parameter(torch.randn(1, query_dim))
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Mode tokens
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.fine_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.no_text_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

    def get_empty_text_embeds(self, batch_size: int):
        return self.no_text_token.expand(batch_size, -1, -1)

    def compute_attention_entropy(self, query, cache):
        """Compute entropy of attention distribution for a query."""
        q_embed = self.encoder.query_input_proj(query)
        patch_features = cache['patch_features']

        q_embed = q_embed.unsqueeze(1)
        attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
        attn_scores = attn_scores / (self.dino_dim ** 0.5) / self.encoder.temperature
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Entropy: -sum(p * log(p))
        entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean()
        return entropy, attn_weights

    def forward(self, text_embeds, raw_frames, vae_latents):
        """Forward pass with ablation losses."""
        B, T = raw_frames.shape[:2]
        N_text = text_embeds.shape[1]

        # Encode all frames
        frames_flat = raw_frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Create per-frame caches
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

        # === Pass 1: Coarse ===
        q_static = self.q_static.expand(B, -1)

        z_coarse_list = []
        for t in range(T):
            z_t = self.encoder.query_attend(q_static, all_caches[t])
            z_coarse_list.append(z_t)
        z_coarse = torch.stack(z_coarse_list, dim=1)
        z_coarse_raw = z_coarse.clone()  # For contrastive loss
        z_coarse = self.dino_to_llm(z_coarse)
        z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale

        coarse_token = self.coarse_token.expand(B, -1, -1)
        seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)

        outputs_pass1 = self.llm.model(inputs_embeds=seq_pass1)
        h_pass1 = outputs_pass1.last_hidden_state

        h_for_queries = h_pass1[:, N_text + 1:]
        queries = self.llm_to_query(h_for_queries)

        # Auxiliary loss on Pass 1
        h_coarse_for_pred = h_pass1[:, N_text:N_text + T]

        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
        prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
        target_latents = vae_latents

        pred_coarse = self.pred_head(h_coarse_for_pred, prev_latents)
        loss_coarse = F.mse_loss(pred_coarse, target_latents)

        # === Pass 2: Fine ===
        q_init = self.q_init.expand(B, -1).unsqueeze(1)
        shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        z_focused_list = []
        for t in range(T):
            z_t = self.encoder.query_attend(shifted_q[:, t], all_caches[t])
            z_focused_list.append(z_t)
        z_focused = torch.stack(z_focused_list, dim=1)
        z_focused_raw = z_focused.clone()  # For contrastive loss
        z_focused = self.dino_to_llm(z_focused)
        z_focused = z_focused / (z_focused.std() + 1e-6) * self.visual_scale

        fine_token = self.fine_token.expand(B, -1, -1)
        seq_pass2 = torch.cat([text_embeds, fine_token, z_focused], dim=1)

        outputs_pass2 = self.llm.model(inputs_embeds=seq_pass2)
        h_pass2 = outputs_pass2.last_hidden_state

        h_fine_for_pred = h_pass2[:, N_text:N_text + T]
        pred_fine = self.pred_head(h_fine_for_pred, prev_latents)
        loss_fine = F.mse_loss(pred_fine, target_latents)

        # === Auxiliary Losses ===
        loss_extra = torch.tensor(0.0, device=raw_frames.device)

        # Contrastive loss: push z_fine away from z_coarse
        if self.lambda_contrastive > 0:
            z_coarse_flat = z_coarse_raw.reshape(-1, z_coarse_raw.shape[-1])
            z_fine_flat = z_focused_raw.reshape(-1, z_focused_raw.shape[-1])
            cos_sim = F.cosine_similarity(z_coarse_flat, z_fine_flat, dim=-1).mean()
            loss_extra = loss_extra + self.lambda_contrastive * cos_sim

        # Entropy regularization: fine should have lower entropy
        if self.entropy_margin > 0:
            # Use first frame for entropy comparison
            entropy_coarse, _ = self.compute_attention_entropy(q_static, all_caches[0])
            entropy_fine, _ = self.compute_attention_entropy(shifted_q[:, 0], all_caches[0])
            # Penalize if fine entropy > coarse entropy - margin
            entropy_loss = F.relu(entropy_fine - entropy_coarse + self.entropy_margin)
            loss_extra = loss_extra + 0.1 * entropy_loss

        # Combined loss
        loss = loss_fine + self.lambda_coarse * loss_coarse + loss_extra

        return loss, loss_fine, loss_coarse


# ============================================================================
# Experiment Runner
# ============================================================================

def compute_feature_similarity(model, frames, device):
    """Compute cosine similarity between z_coarse and z_fine features."""
    B, T = frames.shape[:2]

    with torch.no_grad():
        frames_flat = frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = model.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

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
                        'K': K_all[:, t], 'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': frame_kv_cache})
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Get coarse and fine features
        q_static = model.q_static.expand(B, -1)
        z_coarse = torch.stack([model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)

        q_init = model.q_init.expand(B, -1)
        z_fine = torch.stack([model.encoder.query_attend(q_init, all_caches[t]) for t in range(T)], dim=1)

        # Compute similarity
        z_coarse_flat = z_coarse.reshape(-1, z_coarse.shape[-1]).float()
        z_fine_flat = z_fine.reshape(-1, z_fine.shape[-1]).float()
        cos_sim = F.cosine_similarity(z_coarse_flat, z_fine_flat, dim=-1).mean().item()

    return cos_sim


def run_experiment(config, dataset, num_steps=1500, device='cuda'):
    """Run a single ablation experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'='*70}")
    print(f"Config: {config}")

    # Create model
    model = FoveatedVideoModelAblation(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        dino_dim=384,
        llm_dim=576,
        query_dim=128,
        deep_query=True,
        temperature=config.get('temperature', 1.0),
        lambda_contrastive=config.get('lambda_contrastive', 0.0),
        freeze_dino=config.get('freeze_dino', False),
        hard_attn_k=config.get('hard_attn_k', 0),
        entropy_margin=config.get('entropy_margin', 0.0),
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_params:.1f}M, Trainable: {trainable_params:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=0.01
    )
    scaler = GradScaler()

    # DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

    # Training loop
    model.train()
    results = {
        'loss_fine': [],
        'loss_coarse': [],
        'ratio': [],
        'feature_sim': [],
        'steps': [],
    }

    data_iter = iter(loader)
    pbar = tqdm(range(num_steps), desc=config['name'])

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        frames = batch['frames'].to(device)
        latents = batch['latents'].to(device)
        B, T = frames.shape[:2]

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_embeds = model.get_empty_text_embeds(B).to(device)
            loss, loss_fine, loss_coarse = model(text_embeds, frames, latents)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        lf = loss_fine.item()
        lc = loss_coarse.item()
        ratio = lc / (lf + 1e-8)

        results['loss_fine'].append(lf)
        results['loss_coarse'].append(lc)
        results['ratio'].append(ratio)
        results['steps'].append(step)

        # Compute feature similarity every 100 steps
        if step % 100 == 0:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                feat_sim = compute_feature_similarity(model, frames, device)
            results['feature_sim'].append((step, feat_sim))
            pbar.set_postfix({
                'fine': f'{lf:.3f}',
                'coarse': f'{lc:.3f}',
                'ratio': f'{ratio:.3f}',
                'sim': f'{feat_sim:.3f}'
            })
        else:
            pbar.set_postfix({
                'fine': f'{lf:.3f}',
                'coarse': f'{lc:.3f}',
                'ratio': f'{ratio:.3f}'
            })

    # Final metrics (last 200 steps)
    final_slice = slice(-200, None)
    summary = {
        'name': config['name'],
        'config': config,
        'avg_loss_fine': float(np.mean(results['loss_fine'][final_slice])),
        'avg_loss_coarse': float(np.mean(results['loss_coarse'][final_slice])),
        'avg_ratio': float(np.mean(results['ratio'][final_slice])),
        'final_feature_sim': results['feature_sim'][-1][1] if results['feature_sim'] else None,
        'loss_fine_curve': [float(x) for x in results['loss_fine'][::10]],  # Every 10 steps
        'ratio_curve': [float(x) for x in results['ratio'][::10]],
        'feature_sim_curve': results['feature_sim'],
    }

    print(f"\nFinal Results ({config['name']}):")
    print(f"  Loss Fine:      {summary['avg_loss_fine']:.4f}")
    print(f"  Loss Coarse:    {summary['avg_loss_coarse']:.4f}")
    print(f"  Ratio:          {summary['avg_ratio']:.4f}")
    print(f"  Feature Sim:    {summary['final_feature_sim']:.4f}")

    # Clear GPU memory
    del model, optimizer
    torch.cuda.empty_cache()

    return summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path('outputs/comprehensive_ablations')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("COMPREHENSIVE ABLATION EXPERIMENTS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Check GPU memory
    if torch.cuda.is_available():
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        print(f"GPU Memory: {mem_free:.1f}GB free / {mem_total:.1f}GB total")

    # Load dataset
    frames_dir = "data/frames"
    latents_dir = "data/latents"
    dataset = LocalVideoDataset(frames_dir, latents_dir)

    # Define experiments
    experiments = [
        {
            'name': 'A_baseline',
            'temperature': 1.0,
            'lambda_contrastive': 0.0,
            'freeze_dino': False,
            'hard_attn_k': 0,
            'entropy_margin': 0.0,
        },
        {
            'name': 'B_temp_0.1',
            'temperature': 0.1,
            'lambda_contrastive': 0.0,
            'freeze_dino': False,
            'hard_attn_k': 0,
            'entropy_margin': 0.0,
        },
        {
            'name': 'C_contrastive',
            'temperature': 1.0,
            'lambda_contrastive': 0.1,
            'freeze_dino': False,
            'hard_attn_k': 0,
            'entropy_margin': 0.0,
        },
        {
            'name': 'D_freeze_dino',
            'temperature': 1.0,
            'lambda_contrastive': 0.0,
            'freeze_dino': True,
            'hard_attn_k': 0,
            'entropy_margin': 0.0,
        },
        {
            'name': 'E_hard_attn_32',
            'temperature': 1.0,
            'lambda_contrastive': 0.0,
            'freeze_dino': False,
            'hard_attn_k': 32,
            'entropy_margin': 0.0,
        },
        {
            'name': 'F_entropy_reg',
            'temperature': 1.0,
            'lambda_contrastive': 0.0,
            'freeze_dino': False,
            'hard_attn_k': 0,
            'entropy_margin': 0.5,
        },
        {
            'name': 'G_combined_best',
            'temperature': 0.1,
            'lambda_contrastive': 0.1,
            'freeze_dino': True,
            'hard_attn_k': 32,
            'entropy_margin': 0.5,
        },
    ]

    # Run experiments
    all_results = []
    for config in experiments:
        result = run_experiment(config, dataset, num_steps=1500, device=device)
        all_results.append(result)

        # Save intermediate results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Generate report
    generate_report(all_results, output_dir)


def generate_report(all_results, output_dir):
    """Generate markdown report."""
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Experiment':<20} {'Fine':<10} {'Coarse':<10} {'Ratio':<10} {'Feat Sim':<10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['name']:<20} {r['avg_loss_fine']:<10.4f} {r['avg_loss_coarse']:<10.4f} "
              f"{r['avg_ratio']:<10.4f} {r['final_feature_sim'] or 0:<10.4f}")

    # Best configs
    best_ratio = max(all_results, key=lambda x: x['avg_ratio'])
    best_diversity = min(all_results, key=lambda x: x['final_feature_sim'] if x['final_feature_sim'] is not None else 1.0)

    print(f"\nBest by Ratio:      {best_ratio['name']} (ratio={best_ratio['avg_ratio']:.4f})")
    print(f"Best by Diversity:  {best_diversity['name']} (sim={best_diversity['final_feature_sim']:.4f})")

    # Generate markdown
    report = f"""# Comprehensive Ablation Experiment Results

**Date:** {datetime.now().isoformat()}
**Steps per experiment:** 1500
**Dataset:** Local precomputed (813 videos)

## Summary Table

| Experiment | Loss Fine | Loss Coarse | Ratio | Feature Sim |
|------------|-----------|-------------|-------|-------------|
"""
    for r in all_results:
        report += f"| {r['name']} | {r['avg_loss_fine']:.4f} | {r['avg_loss_coarse']:.4f} | {r['avg_ratio']:.4f} | {r['final_feature_sim'] or 0:.4f} |\n"

    report += f"""
## Best Configurations

- **Best by Ratio:** {best_ratio['name']} (ratio={best_ratio['avg_ratio']:.4f})
- **Best by Feature Diversity:** {best_diversity['name']} (sim={best_diversity['final_feature_sim']:.4f})

## Interpretation

- **Ratio > 1.0** means loss_coarse > loss_fine → fine attention WORKS!
- **Lower Feature Sim** means z_fine and z_coarse are more different (better)

## Configurations Tested

1. **A_baseline**: Current model (temp=1.0, no contrastive, train DINO)
2. **B_temp_0.1**: Sharper attention temperature
3. **C_contrastive**: Push features apart (λ=0.1)
4. **D_freeze_dino**: Preserve pretrained feature diversity
5. **E_hard_attn_32**: Top-32 hard attention (sparse selection)
6. **F_entropy_reg**: Entropy margin loss (fine < coarse)
7. **G_combined_best**: All fixes together

## Key Findings

"""

    # Add interpretation
    if best_ratio['avg_ratio'] > 1.05:
        report += f"- ✅ **{best_ratio['name']}** achieved ratio > 1.05, validating that fine attention beats coarse!\n"
    else:
        report += "- ⚠️ No configuration achieved ratio > 1.05. The core hypothesis may need rethinking.\n"

    if best_diversity['final_feature_sim'] is not None and best_diversity['final_feature_sim'] < 0.5:
        report += f"- ✅ **{best_diversity['name']}** achieved feature similarity < 0.5, showing good differentiation\n"

    # Check temperature effect
    baseline = next(r for r in all_results if r['name'] == 'A_baseline')
    temp_exp = next((r for r in all_results if r['name'] == 'B_temp_0.1'), None)
    if temp_exp:
        temp_improvement = (temp_exp['avg_ratio'] - baseline['avg_ratio']) / baseline['avg_ratio'] * 100
        report += f"- Temperature 0.1 vs 1.0: {temp_improvement:+.1f}% ratio change\n"

    # Check contrastive effect
    contrast_exp = next((r for r in all_results if r['name'] == 'C_contrastive'), None)
    if contrast_exp and baseline:
        sim_reduction = baseline['final_feature_sim'] - contrast_exp['final_feature_sim'] if baseline['final_feature_sim'] else 0
        report += f"- Contrastive loss reduced feature similarity by {sim_reduction:.2f}\n"

    report += """
## Recommendations

Based on these experiments:
"""

    if best_ratio['avg_ratio'] > 1.02:
        report += f"1. **Use {best_ratio['name']} configuration** - it shows the best fine vs coarse separation\n"

    if best_diversity['final_feature_sim'] and best_diversity['final_feature_sim'] < 0.5:
        report += f"2. **{best_diversity['name']}** provides best feature diversity\n"

    report += """
## Next Steps

1. Run best configuration for longer (10K+ steps) to see if trends continue
2. If ratio still ~1.0, consider task-level changes (predict optical flow, delta latents)
3. Evaluate on more dynamic videos (Something-Something v2)

---
*Generated by comprehensive_ablations.py*
"""

    with open(output_dir / 'ABLATION_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_dir / 'ABLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
