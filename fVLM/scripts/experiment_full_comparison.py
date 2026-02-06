#!/usr/bin/env python3
"""
Full Comparison: Foveated vs Baseline VLM

Experiments:
  1. Foveated 8F + Recon (caption + reconstruction loss)
  2. Foveated 8F - Recon (caption only, ablation)
  3. Baseline 8F (caption only)
  4. Foveated 64F + Recon
  5. Foveated 64F - Recon
  6. Baseline 64F

Evaluation uses TRUE AUTOREGRESSIVE inference for foveated models.
"""

import sys
import os
import gc
import json
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer
import numpy as np
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel
from src.model.baseline_vlm import BaselineVLM

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Paths
DATA_8F = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
DATA_64F = Path("/mnt/d/projects/fVLM/data/webvid_64f_5k/shards")  # New 5K dataset
OUTPUT_BASE = Path("/mnt/d/projects/fVLM/outputs/full_comparison_v2")  # Fresh output dir

# Configs - CRITICAL: max_steps must be < 1 epoch
# 8F: ~924 samples / 16 batch = 58 steps/epoch -> use 50 steps
# 64F: ~5000 samples / 16 batch = 312 steps/epoch -> use 300 steps
CONFIGS = {
    '8f': {
        'data_dir': DATA_8F,
        'num_frames': 8,
        'batch_size': 4,
        'grad_accum': 4,
        'frame_size': 256,
        'max_steps': 50,  # < 1 epoch (58 steps/epoch)
        'checkpoints': [25, 50],
    },
    '64f': {
        'data_dir': DATA_64F,
        'num_frames': 64,
        'batch_size': 2,
        'grad_accum': 8,
        'frame_size': 224,
        'max_steps': 280,  # < 1 epoch (4500 samples / 16 batch = 281 steps/epoch)
        'checkpoints': [100, 280],
    },
}

LEARNING_RATE = 3e-5
WARMUP_STEPS = 50
LAMBDA_RECON = 0.5


# ============================================================================
# DATASETS
# ============================================================================

class ShardedDataset8F(torch.utils.data.IterableDataset):
    """8-frame sharded dataset with VAE latents."""

    def __init__(self, shard_dir, num_frames=8, train=True):
        self.shard_dir = Path(shard_dir)
        self.num_frames = num_frames
        all_shards = sorted(self.shard_dir.glob("shard_*.pt"))
        # Simple split: last 10% for val
        n_val = max(1, len(all_shards) // 10)
        if train:
            self.shard_files = all_shards[:-n_val]
        else:
            self.shard_files = all_shards[-n_val:]
        print(f"  {'Train' if train else 'Val'}: {len(self.shard_files)} shards")

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location='cpu', weights_only=False)
                samples = shard.get('samples', shard)
                if isinstance(samples, dict):
                    samples = [samples]
                np.random.shuffle(samples)
                for sample in samples:
                    yield self._process(sample)
            except Exception as e:
                print(f"Error loading {shard_path}: {e}")
                continue

    def _process(self, data):
        frames = data['frames']   # [T, 3, 256, 256] uint8
        latents = data['latents'] # [T, 4, 32, 32]
        caption = data['caption']

        T = frames.shape[0]
        if T > self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            frames = frames[indices]
            latents = latents[indices]

        frames = frames.float() / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'latents': latents.float(), 'caption': caption}


class ShardedDataset64F(torch.utils.data.IterableDataset):
    """64-frame sharded dataset with VAE latents."""

    def __init__(self, shard_dir, num_frames=64, train=True):
        self.shard_dir = Path(shard_dir)
        self.num_frames = num_frames
        all_shards = sorted(self.shard_dir.glob("shard_*.pt"))
        n_val = max(1, len(all_shards) // 10)
        if train:
            self.shard_files = all_shards[:-n_val]
        else:
            self.shard_files = all_shards[-n_val:]
        print(f"  {'Train' if train else 'Val'}: {len(self.shard_files)} shards")

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location='cpu', weights_only=False)
                samples = shard.get('samples', shard)
                if isinstance(samples, dict):
                    samples = [samples]
                np.random.shuffle(samples)
                for sample in samples:
                    yield self._process(sample)
            except Exception as e:
                print(f"Error loading {shard_path}: {e}")
                continue

    def _process(self, data):
        frames = data['frames']   # [T, 3, H, W] uint8
        latents = data['latents'] # [T, 4, H/8, W/8]
        caption = data['caption']

        T = frames.shape[0]
        if T > self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            frames = frames[indices]
            latents = latents[indices]
        elif T < self.num_frames:
            # Pad with last frame
            pad = self.num_frames - T
            frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)
            latents = torch.cat([latents, latents[-1:].expand(pad, -1, -1, -1)], dim=0)

        frames = frames.float() / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'latents': latents.float(), 'caption': caption}


# ============================================================================
# FORWARD PASSES
# ============================================================================

def forward_foveated_joint(model, frames, caption_ids, caption_mask, vae_latents, tokenizer):
    """Foveated forward with caption + reconstruction loss."""
    B, T = frames.shape[:2]
    device = frames.device

    # Encode frames
    frames_flat = frames.reshape(B * T, 3, frames.shape[-2], frames.shape[-1])
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Build per-frame caches
    all_caches = []
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        K_all = [cache_flat['kv_cache'][l]['K'].reshape(B, T, N, D) for l in range(num_layers)]
        V_all = [cache_flat['kv_cache'][l]['V'].reshape(B, T, N, D) for l in range(num_layers)]
        layers = [cache_flat['kv_cache'][l]['layer'] for l in range(num_layers)]

        for t in range(T):
            frame_kv = [{'K': K_all[l][:, t], 'V': V_all[l][:, t], 'layer': layers[l]}
                       for l in range(num_layers)]
            all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': frame_kv})
    else:
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    # Text embeddings
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]
    text_embeds = model.get_empty_text_embeds(B)
    N_text = text_embeds.shape[1]

    # Prev latents for reconstruction
    # Handle different latent sizes (32x32 for 256px, 28x28 for 224px)
    latent_h, latent_w = vae_latents.shape[-2], vae_latents.shape[-1]
    z_vae_init = model.z_vae_init  # [1, 4, 32, 32]
    if z_vae_init.shape[-1] != latent_w:
        z_vae_init = F.interpolate(z_vae_init, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
    z_vae_init = z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)

    # Tokens
    coarse_token = model.coarse_token.expand(B, -1, -1)
    fine_token = model.fine_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)

    # Coarse pass for query generation
    q_static = model.q_static.expand(B, -1)
    z_coarse = torch.stack([model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # Generate queries
    seq_query = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_query = model.llm.model(inputs_embeds=seq_query)
    queries = model.llm_to_query(outputs_query.last_hidden_state[:, 2:])

    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    # Fine pass
    z_fine = torch.stack([model.encoder.query_attend(shifted_q[:, t], all_caches[t]) for t in range(T)], dim=1)
    z_fine_llm = model.dino_to_llm(z_fine)
    z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

    # Caption loss
    seq_cap = torch.cat([fine_token, z_fine_llm, caption_embeds], dim=1)
    outputs_cap = model.llm.model(inputs_embeds=seq_cap)
    logits_cap = model.llm.lm_head(outputs_cap.last_hidden_state)
    caption_logits = logits_cap[:, 1+T:-1, :]
    loss_caption = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.size(-1)),
        caption_targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )

    # Reconstruction loss
    seq_rec = torch.cat([text_embeds, fine_token, z_fine_llm], dim=1)
    outputs_rec = model.llm.model(inputs_embeds=seq_rec)
    h_for_pred = outputs_rec.last_hidden_state[:, N_text:N_text + T]
    pred_latents = model.pred_head(h_for_pred, prev_latents)
    loss_recon = F.mse_loss(pred_latents, vae_latents)

    return loss_caption, loss_recon


def forward_foveated_caption_only(model, frames, caption_ids, caption_mask, tokenizer):
    """Foveated forward with caption loss only (no reconstruction)."""
    return model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)


def forward_baseline(model, frames, caption_ids, tokenizer):
    """Baseline forward with caption loss only."""
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:].clone()
    caption_targets[caption_targets == tokenizer.pad_token_id] = -100
    loss, _ = model(frames, caption_embeds, caption_targets)
    return loss


# ============================================================================
# TRAINING
# ============================================================================

def train_experiment(exp_name, model_type, frame_config, use_recon, device):
    """Train a single experiment."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {exp_name}")
    print(f"  Model: {model_type}, Frames: {frame_config}, Recon: {use_recon}")
    print(f"{'='*70}")

    config = CONFIGS[frame_config]
    output_dir = OUTPUT_BASE / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    if model_type == 'foveated':
        model = FoveatedVideoModel(
            dino_model='facebook/dinov2-small',
            llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
            dino_dim=384, llm_dim=576, query_dim=384,
            deep_query=True, freeze_dino=False,
        ).to(device)
    else:
        model = BaselineVLM(
            dino_model='facebook/dinov2-small',
            llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
            pixel_shuffle_scale=4,
            freeze_dino=False, freeze_llm=False,
        ).to(device)

    # Dataset
    if frame_config == '8f':
        dataset = ShardedDataset8F(config['data_dir'], config['num_frames'], train=True)
    else:
        dataset = ShardedDataset64F(config['data_dir'], config['num_frames'], train=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = GradScaler('cuda')

    # Training
    model.train()
    step = 0
    accum_loss = 0
    accum_recon = 0
    accum_count = 0
    metrics_history = []

    pbar = tqdm(total=config['max_steps'], desc=exp_name)
    data_iter = iter(dataset)

    while step < config['max_steps']:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            sample = next(data_iter)

        frames = sample['frames'].unsqueeze(0).to(device)
        latents = sample['latents'].unsqueeze(0).to(device) if use_recon else None
        caption = sample['caption']

        # Resize for baseline if needed
        if model_type == 'baseline' and frames.shape[-1] != 224:
            frames = F.interpolate(frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1]),
                                   size=(224, 224), mode='bilinear', align_corners=False)
            frames = frames.reshape(1, -1, 3, 224, 224)

        # Tokenize
        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)
        caption_mask = enc['attention_mask'].to(device)

        # Forward
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            if model_type == 'foveated':
                if use_recon:
                    loss_cap, loss_rec = forward_foveated_joint(
                        model, frames, caption_ids, caption_mask, latents, tokenizer)
                    loss = loss_cap + LAMBDA_RECON * loss_rec
                    accum_recon += loss_rec.item()
                else:
                    loss = forward_foveated_caption_only(
                        model, frames, caption_ids, caption_mask, tokenizer)
                    loss_rec = torch.tensor(0.0)
            else:
                loss = forward_baseline(model, frames, caption_ids, tokenizer)
                loss_rec = torch.tensor(0.0)

        loss_scaled = loss / config['grad_accum']
        scaler.scale(loss_scaled).backward()

        accum_loss += loss.item()
        accum_count += 1

        if accum_count >= config['grad_accum']:
            # LR warmup
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1
            avg_loss = accum_loss / config['grad_accum']
            avg_recon = accum_recon / config['grad_accum'] if use_recon else 0

            metrics_history.append({
                'step': step,
                'loss': avg_loss,
                'loss_recon': avg_recon,
            })

            accum_loss = 0
            accum_recon = 0
            accum_count = 0

            pbar.update(1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Checkpoint
            if step in config['checkpoints']:
                ckpt_path = output_dir / f'step_{step:06d}.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'metrics': metrics_history,
                }, ckpt_path)
                print(f"\n  Saved: {ckpt_path}")

    pbar.close()

    # Save final metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_history, f)

    # Cleanup
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_experiment(exp_name, model_type, frame_config, device):
    """Evaluate checkpoints using TRUE AUTOREGRESSIVE inference for foveated."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {exp_name}")
    print(f"{'='*70}")

    config = CONFIGS[frame_config]
    output_dir = OUTPUT_BASE / exp_name

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # Validation dataset
    if frame_config == '8f':
        val_dataset = ShardedDataset8F(config['data_dir'], config['num_frames'], train=False)
    else:
        val_dataset = ShardedDataset64F(config['data_dir'], config['num_frames'], train=False)

    results = []

    for step in config['checkpoints']:
        ckpt_path = output_dir / f'step_{step:06d}.pt'
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        # Load model
        if model_type == 'foveated':
            model = FoveatedVideoModel(
                dino_model='facebook/dinov2-small',
                llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
                dino_dim=384, llm_dim=576, query_dim=384,
                deep_query=True, freeze_dino=False,
            ).to(device)
        else:
            model = BaselineVLM(
                dino_model='facebook/dinov2-small',
                llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
                pixel_shuffle_scale=4,
                freeze_dino=False, freeze_llm=False,
            ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Evaluate
        losses = []
        n_samples = 0
        max_samples = 50  # Limit for speed

        with torch.no_grad():
            for sample in val_dataset:
                if n_samples >= max_samples:
                    break

                frames = sample['frames'].unsqueeze(0).to(device)
                caption = sample['caption']

                if model_type == 'baseline' and frames.shape[-1] != 224:
                    frames = F.interpolate(frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1]),
                                          size=(224, 224), mode='bilinear', align_corners=False)
                    frames = frames.reshape(1, -1, 3, 224, 224)

                enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                               max_length=64, truncation=True)
                caption_ids = enc['input_ids'].to(device)
                caption_mask = enc['attention_mask'].to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    if model_type == 'foveated':
                        # TRUE AUTOREGRESSIVE INFERENCE
                        loss = model.forward_autoregressive_captioning(
                            frames, caption_ids, caption_mask)
                    else:
                        caption_embeds = model.llm.model.embed_tokens(caption_ids)
                        caption_targets = caption_ids[:, 1:].clone()
                        caption_targets[caption_targets == tokenizer.pad_token_id] = -100
                        loss, _ = model(frames, caption_embeds, caption_targets)

                losses.append(loss.item())
                n_samples += 1

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        ppl = np.exp(mean_loss)

        results.append({
            'experiment': exp_name,
            'step': step,
            'loss_mean': mean_loss,
            'loss_std': std_loss,
            'perplexity': ppl,
            'n_samples': n_samples,
        })

        print(f"  Step {step}: Loss={mean_loss:.4f} (+/- {std_loss:.4f}), PPL={ppl:.2f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def generate_plots(all_results, output_dir):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Organize results
    by_exp = {}
    for r in all_results:
        exp = r['experiment']
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(r)

    # Plot 1: Caption Loss Comparison (8F)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'foveated_8f_recon': 'blue',
        'foveated_8f_norecon': 'cyan',
        'baseline_8f': 'red',
        'foveated_64f_recon': 'darkblue',
        'foveated_64f_norecon': 'lightblue',
        'baseline_64f': 'darkred',
    }
    labels = {
        'foveated_8f_recon': 'Foveated+Recon (1 tok)',
        'foveated_8f_norecon': 'Foveated (1 tok)',
        'baseline_8f': 'Baseline (16 tok)',
        'foveated_64f_recon': 'Foveated+Recon (1 tok)',
        'foveated_64f_norecon': 'Foveated (1 tok)',
        'baseline_64f': 'Baseline (16 tok)',
    }

    # 8-frame comparison
    ax = axes[0]
    for exp in ['foveated_8f_recon', 'foveated_8f_norecon', 'baseline_8f']:
        if exp in by_exp:
            data = sorted(by_exp[exp], key=lambda x: x['step'])
            steps = [d['step'] for d in data]
            losses = [d['loss_mean'] for d in data]
            ax.plot(steps, losses, 'o-', color=colors[exp], label=labels[exp], linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Caption Loss (CE)')
    ax.set_title('8-Frame Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 64-frame comparison
    ax = axes[1]
    for exp in ['foveated_64f_recon', 'foveated_64f_norecon', 'baseline_64f']:
        if exp in by_exp:
            data = sorted(by_exp[exp], key=lambda x: x['step'])
            steps = [d['step'] for d in data]
            losses = [d['loss_mean'] for d in data]
            ax.plot(steps, losses, 'o-', color=colors[exp], label=labels[exp], linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Caption Loss (CE)')
    ax.set_title('64-Frame Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'caption_loss_comparison.png', dpi=150)
    plt.savefig(plots_dir / 'caption_loss_comparison.pdf')
    plt.close()

    # Plot 2: Reconstruction Ablation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 8F ablation
    ax = axes[0]
    for exp in ['foveated_8f_recon', 'foveated_8f_norecon']:
        if exp in by_exp:
            data = sorted(by_exp[exp], key=lambda x: x['step'])
            steps = [d['step'] for d in data]
            losses = [d['loss_mean'] for d in data]
            ax.plot(steps, losses, 'o-', color=colors[exp], label=labels[exp], linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Caption Loss (CE)')
    ax.set_title('8-Frame: Reconstruction Ablation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 64F ablation
    ax = axes[1]
    for exp in ['foveated_64f_recon', 'foveated_64f_norecon']:
        if exp in by_exp:
            data = sorted(by_exp[exp], key=lambda x: x['step'])
            steps = [d['step'] for d in data]
            losses = [d['loss_mean'] for d in data]
            ax.plot(steps, losses, 'o-', color=colors[exp], label=labels[exp], linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Caption Loss (CE)')
    ax.set_title('64-Frame: Reconstruction Ablation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'reconstruction_ablation.png', dpi=150)
    plt.savefig(plots_dir / 'reconstruction_ablation.pdf')
    plt.close()

    # Plot 3: Token Efficiency Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get final losses (step 300)
    final_losses = {}
    tokens = {
        'foveated_8f_recon': 8, 'foveated_8f_norecon': 8, 'baseline_8f': 128,
        'foveated_64f_recon': 64, 'foveated_64f_norecon': 64, 'baseline_64f': 1024,
    }

    for exp, data in by_exp.items():
        step_300 = [d for d in data if d['step'] == 300]
        if step_300:
            final_losses[exp] = step_300[0]['loss_mean']

    x_labels = []
    losses_plot = []
    colors_plot = []

    order = ['foveated_8f_recon', 'foveated_8f_norecon', 'baseline_8f',
             'foveated_64f_recon', 'foveated_64f_norecon', 'baseline_64f']

    for exp in order:
        if exp in final_losses:
            x_labels.append(f"{labels[exp]}\n({tokens[exp]} tok)")
            losses_plot.append(final_losses[exp])
            colors_plot.append(colors[exp])

    x = np.arange(len(x_labels))
    bars = ax.bar(x, losses_plot, color=colors_plot, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha='right')
    ax.set_ylabel('Caption Loss (CE)')
    ax.set_title('Final Caption Loss @ Step 300')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, loss in zip(bars, losses_plot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{loss:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(plots_dir / 'token_efficiency.png', dpi=150)
    plt.savefig(plots_dir / 'token_efficiency.pdf')
    plt.close()

    # Plot 4: Scaling Analysis (8F vs 64F)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = [8, 64]

    # Foveated with recon
    if 'foveated_8f_recon' in final_losses and 'foveated_64f_recon' in final_losses:
        y = [final_losses['foveated_8f_recon'], final_losses['foveated_64f_recon']]
        ax.plot(x, y, 'o-', color='blue', label='Foveated+Recon', linewidth=2, markersize=10)

    # Foveated without recon
    if 'foveated_8f_norecon' in final_losses and 'foveated_64f_norecon' in final_losses:
        y = [final_losses['foveated_8f_norecon'], final_losses['foveated_64f_norecon']]
        ax.plot(x, y, 's--', color='cyan', label='Foveated', linewidth=2, markersize=10)

    # Baseline
    if 'baseline_8f' in final_losses and 'baseline_64f' in final_losses:
        y = [final_losses['baseline_8f'], final_losses['baseline_64f']]
        ax.plot(x, y, '^-', color='red', label='Baseline', linewidth=2, markersize=10)

    ax.set_xlabel('Number of Frames')
    ax.set_ylabel('Caption Loss (CE)')
    ax.set_title('Scaling: 8 vs 64 Frames')
    ax.set_xticks([8, 64])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'scaling_analysis.png', dpi=150)
    plt.savefig(plots_dir / 'scaling_analysis.pdf')
    plt.close()

    print(f"\nPlots saved to {plots_dir}/")


def generate_results_table(all_results, output_dir):
    """Generate markdown results table."""
    # Sort by experiment and step
    sorted_results = sorted(all_results, key=lambda x: (x['experiment'], x['step']))

    lines = [
        "# Full Comparison Results",
        "",
        "## Summary Table",
        "",
        "| Experiment | Step | Loss | Std | PPL | Tokens |",
        "|------------|------|------|-----|-----|--------|",
    ]

    tokens_map = {
        'foveated_8f_recon': 8, 'foveated_8f_norecon': 8, 'baseline_8f': 128,
        'foveated_64f_recon': 64, 'foveated_64f_norecon': 64, 'baseline_64f': 1024,
    }

    for r in sorted_results:
        tokens = tokens_map.get(r['experiment'], '?')
        lines.append(f"| {r['experiment']} | {r['step']} | {r['loss_mean']:.4f} | {r['loss_std']:.4f} | {r['perplexity']:.2f} | {tokens} |")

    lines.extend([
        "",
        "## Key Findings",
        "",
        "### Reconstruction Ablation",
        "",
    ])

    # Compute differences
    by_exp = {}
    for r in all_results:
        if r['step'] == 300:
            by_exp[r['experiment']] = r

    if 'foveated_8f_recon' in by_exp and 'foveated_8f_norecon' in by_exp:
        diff = by_exp['foveated_8f_norecon']['loss_mean'] - by_exp['foveated_8f_recon']['loss_mean']
        pct = diff / by_exp['foveated_8f_norecon']['loss_mean'] * 100
        lines.append(f"- 8-frame: Reconstruction {'helps' if diff > 0 else 'hurts'} by {abs(diff):.4f} ({abs(pct):.1f}%)")

    if 'foveated_64f_recon' in by_exp and 'foveated_64f_norecon' in by_exp:
        diff = by_exp['foveated_64f_norecon']['loss_mean'] - by_exp['foveated_64f_recon']['loss_mean']
        pct = diff / by_exp['foveated_64f_norecon']['loss_mean'] * 100
        lines.append(f"- 64-frame: Reconstruction {'helps' if diff > 0 else 'hurts'} by {abs(diff):.4f} ({abs(pct):.1f}%)")

    lines.extend([
        "",
        "### Foveated vs Baseline",
        "",
    ])

    if 'foveated_8f_recon' in by_exp and 'baseline_8f' in by_exp:
        diff = by_exp['baseline_8f']['loss_mean'] - by_exp['foveated_8f_recon']['loss_mean']
        pct = diff / by_exp['baseline_8f']['loss_mean'] * 100
        winner = "Foveated" if diff > 0 else "Baseline"
        lines.append(f"- 8-frame: {winner} wins by {abs(diff):.4f} ({abs(pct):.1f}%) - Foveated uses 16x fewer tokens")

    if 'foveated_64f_recon' in by_exp and 'baseline_64f' in by_exp:
        diff = by_exp['baseline_64f']['loss_mean'] - by_exp['foveated_64f_recon']['loss_mean']
        pct = diff / by_exp['baseline_64f']['loss_mean'] * 100
        winner = "Foveated" if diff > 0 else "Baseline"
        lines.append(f"- 64-frame: {winner} wins by {abs(diff):.4f} ({abs(pct):.1f}%) - Foveated uses 16x fewer tokens")

    lines.extend([
        "",
        "---",
        f"*Generated: {datetime.now().isoformat()}*",
    ])

    with open(output_dir / 'results.md', 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nResults saved to {output_dir}/results.md")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Run all (train + eval + plot)')
    parser.add_argument('--exp', type=str, default=None,
                       help='Run specific experiment (e.g., foveated_8f_recon)')
    args = parser.parse_args()

    if args.all:
        args.train = args.eval = args.plot = True

    if not any([args.train, args.eval, args.plot]):
        args.all = args.train = args.eval = args.plot = True

    device = torch.device('cuda')
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Define experiments
    experiments = [
        ('foveated_8f_recon', 'foveated', '8f', True),
        ('foveated_8f_norecon', 'foveated', '8f', False),
        ('baseline_8f', 'baseline', '8f', False),
        ('foveated_64f_recon', 'foveated', '64f', True),
        ('foveated_64f_norecon', 'foveated', '64f', False),
        ('baseline_64f', 'baseline', '64f', False),
    ]

    # Filter by --exp if specified
    if args.exp:
        experiments = [e for e in experiments if e[0] == args.exp]
        if not experiments:
            print(f"Unknown experiment: {args.exp}")
            return

    # Training
    if args.train:
        print("\n" + "="*70)
        print("TRAINING PHASE")
        print("="*70)

        for exp_name, model_type, frame_config, use_recon in experiments:
            train_experiment(exp_name, model_type, frame_config, use_recon, device)

    # Evaluation
    all_results = []
    if args.eval:
        print("\n" + "="*70)
        print("EVALUATION PHASE")
        print("="*70)

        for exp_name, model_type, frame_config, use_recon in experiments:
            results = evaluate_experiment(exp_name, model_type, frame_config, device)
            all_results.extend(results)

        # Save all results
        with open(OUTPUT_BASE / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Load results if only plotting
    if args.plot and not args.eval:
        results_file = OUTPUT_BASE / 'all_results.json'
        if results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)

    # Plotting
    if args.plot and all_results:
        print("\n" + "="*70)
        print("PLOTTING PHASE")
        print("="*70)

        generate_plots(all_results, OUTPUT_BASE)
        generate_results_table(all_results, OUTPUT_BASE)


if __name__ == '__main__':
    main()
