#!/usr/bin/env python3
"""
Comprehensive Scaling Law Study: Foveated vs Baseline VLM

Runs training + inline evaluation across multiple model sizes and frame counts.
Produces dense scaling curves (20 eval points per run) with both training-approximation
and true-inference loss for foveated models.

Key engineering:
  - Auto-profiles optimal micro-batch size per run (maximizes GPU utilization)
  - Uses DataLoader with proper batching (not sample-by-sample!)
  - Monitors GPU util/memory during first steps, warns if underutilized
  - Consistent EB=16 across all runs

Model configs: S-S (160M), M-S (385M), S-B (220M), B-L (1.8B, stretch)
Frame counts: 8F and 64F
Architectures: foveated (caption only) and baseline

Usage:
    python run_comprehensive_scaling.py                    # Run all P1
    python run_comprehensive_scaling.py --priority P2      # Include B-L
    python run_comprehensive_scaling.py --plot-only         # Just regenerate plots
    python run_comprehensive_scaling.py --run SS_8f_fov     # Single run
"""

import sys
import os
import gc
import json
import time
import argparse
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.foveated_vlm import FoveatedVideoModel
from src.model.baseline_vlm import BaselineVLM

# ============================================================================
# CONSTANTS
# ============================================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Data paths
DATA_8F = Path("/mnt/d/projects/fVLM/data/webvid_8f_5k/shards")
DATA_64F = Path("/mnt/d/projects/fVLM/data/webvid_64f_5k/shards")
OUTPUT_BASE = Path("/mnt/d/projects/fVLM/outputs/scaling_comprehensive")

LEARNING_RATE = 3e-5
WARMUP_STEPS = 50
TARGET_EB = 16  # Effective batch size (consistent for all runs)
MAX_STEPS = 280  # < 1 epoch for both 8F and 64F (4500 samples / 16 EB = 281)
EVAL_EVERY = 14  # 280 / 14 = 20 eval points
MAX_VAL_SAMPLES = 50  # Fast eval, enough for reliable estimates
PROFILE_HEADROOM_GB = 2.0  # Minimum GPU memory headroom

# ============================================================================
# MODEL CONFIGS
# ============================================================================

MODEL_CONFIGS = {
    'S-S': {
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_model': 'facebook/dinov2-small',
        'llm_dim': 576,
        'dino_dim': 384,
        'llm_params': 135e6,
        'priority': 'P1',
    },
    'M-S': {
        'llm_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'dino_model': 'facebook/dinov2-small',
        'llm_dim': 960,
        'dino_dim': 384,
        'llm_params': 360e6,
        'priority': 'P1',
    },
    'S-B': {
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_model': 'facebook/dinov2-base',
        'llm_dim': 576,
        'dino_dim': 768,
        'llm_params': 135e6,
        'priority': 'P1',
    },
    'B-L': {
        'llm_model': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'dino_model': 'facebook/dinov2-base',
        'llm_dim': 2048,
        'dino_dim': 768,
        'llm_params': 1.7e9,
        'priority': 'P2',
    },
}

FRAME_CONFIGS = {
    '8f': {
        'data_dir': DATA_8F,
        'num_frames': 8,
        'frame_size': 224,
    },
    '64f': {
        'data_dir': DATA_64F,
        'num_frames': 64,
        'frame_size': 224,
    },
}

# DINO FLOPs per frame (forward only)
DINO_FLOPS = {
    'facebook/dinov2-small': 5.5e9,
    'facebook/dinov2-base': 21.6e9,
}

# ============================================================================
# GPU MONITORING
# ============================================================================

def get_gpu_stats():
    """Get GPU utilization and memory from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        parts = result.stdout.strip().split(', ')
        return {'util': int(parts[0]), 'mem_used': int(parts[1]), 'mem_total': int(parts[2])}
    except Exception:
        return None


# ============================================================================
# FLOPS CALCULATOR
# ============================================================================

def compute_flops_per_sample(model_cfg, frame_cfg, arch_type):
    """Compute training FLOPs per sample (forward + backward = 3x forward)."""
    T = frame_cfg['num_frames']
    L_text = 64  # max text tokens

    dino_flops = DINO_FLOPS[model_cfg['dino_model']] * T
    llm_flops_per_token = 6 * model_cfg['llm_params']  # 6N rule

    if arch_type == 'foveated':
        # 2 LLM passes: query generation + captioning
        seq_len = 1 + T + L_text
        llm_flops = 2 * seq_len * llm_flops_per_token
    else:
        # 1 LLM pass but longer sequence (16 tokens/frame)
        seq_len = 1 + T * 16 + L_text
        llm_flops = seq_len * llm_flops_per_token

    forward_flops = dino_flops + llm_flops
    # Training = 3x forward (forward + backward ≈ 2x forward + optimizer)
    return 3 * forward_flops


# ============================================================================
# DATASET + DATALOADER
# ============================================================================

class ShardedDataset(torch.utils.data.IterableDataset):
    """Unified sharded dataset for both 8F and 64F data."""

    def __init__(self, shard_dir, num_frames=8, frame_size=224, train=True):
        self.shard_dir = Path(shard_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        all_shards = sorted(self.shard_dir.glob("shard_*.pt"))
        n_val = max(1, len(all_shards) // 10)
        if train:
            self.shard_files = all_shards[:-n_val]
        else:
            self.shard_files = all_shards[-n_val:]

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location='cpu', weights_only=False)
                samples = shard.get('samples', [shard] if isinstance(shard, dict) else shard)
                if isinstance(samples, dict):
                    samples = [samples]
                np.random.shuffle(samples)
                for sample in samples:
                    yield self._process(sample)
            except Exception:
                continue

    def _process(self, data):
        frames = data['frames']
        caption = data['caption']

        T = frames.shape[0]
        if T > self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            frames = frames[indices]
        elif T < self.num_frames:
            pad = self.num_frames - T
            frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)

        frames = frames.float() / 255.0
        if frames.shape[-1] != self.frame_size:
            frames = F.interpolate(frames, size=(self.frame_size, self.frame_size),
                                   mode='bilinear', align_corners=False)
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'caption': caption}


def make_collate_fn(tokenizer, max_length=64):
    """Create a collate function that batches frames and tokenizes captions."""
    def collate_fn(samples):
        frames = torch.stack([s['frames'] for s in samples])  # [B, T, C, H, W]
        captions = [s['caption'] for s in samples]
        enc = tokenizer(captions, return_tensors='pt', padding='max_length',
                       max_length=max_length, truncation=True)
        return {
            'frames': frames,
            'caption_ids': enc['input_ids'],
            'caption_mask': enc['attention_mask'],
        }
    return collate_fn


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(model_cfg, arch_type, device, use_grad_ckpt=False):
    """Create model from config."""
    if arch_type == 'foveated':
        model = FoveatedVideoModel(
            dino_model=model_cfg['dino_model'],
            llm_model=model_cfg['llm_model'],
            dino_dim=model_cfg['dino_dim'],
            llm_dim=model_cfg['llm_dim'],
            query_dim=model_cfg['dino_dim'],
            deep_query=True,
            freeze_dino=False,
        )
    else:
        model = BaselineVLM(
            dino_model=model_cfg['dino_model'],
            llm_model=model_cfg['llm_model'],
            pixel_shuffle_scale=4,
            freeze_dino=False,
            freeze_llm=False,
        )

    model = model.to(device)

    if use_grad_ckpt:
        if hasattr(model, 'llm'):
            model.llm.gradient_checkpointing_enable()

    return model


# ============================================================================
# BATCH SIZE PROFILING
# ============================================================================

def profile_optimal_batch(model_cfg, arch_type, frame_cfg, device, target_eb=TARGET_EB):
    """Profile to find the largest micro-batch that fits with headroom.

    Creates a temporary model, tests batch sizes from large to small,
    picks the largest that fits with PROFILE_HEADROOM_GB free.
    Then cleans up the temporary model entirely.
    """
    print("  Profiling optimal batch size...")
    use_grad_ckpt = (model_cfg.get('priority') == 'P2' and frame_cfg['num_frames'] == 64)
    model = create_model(model_cfg, arch_type, device, use_grad_ckpt=use_grad_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    tokenizer.pad_token = tokenizer.eos_token
    T = frame_cfg['num_frames']

    best = (1, target_eb)

    for micro_batch in [8, 4, 2, 1]:
        if target_eb % micro_batch != 0:
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Dummy forward + backward (simulates one micro-batch)
            frames = torch.randn(micro_batch, T, 3, 224, 224, device=device)
            enc = tokenizer(['a test caption for profiling'] * micro_batch,
                          return_tensors='pt', padding='max_length',
                          max_length=64, truncation=True)
            caption_ids = enc['input_ids'].to(device)
            caption_mask = enc['attention_mask'].to(device)

            model.train()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                if arch_type == 'foveated':
                    loss = model.forward_captioning(
                        frames, caption_ids, caption_mask, use_fine=True)
                else:
                    caption_embeds = model.llm.model.embed_tokens(caption_ids)
                    caption_targets = caption_ids[:, 1:].clone()
                    caption_targets[caption_targets == tokenizer.pad_token_id] = -100
                    loss, _ = model(frames, caption_embeds, caption_targets)

            loss.backward()
            model.zero_grad(set_to_none=True)

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            headroom = total_mem - peak_mem

            del frames, caption_ids, caption_mask, loss
            torch.cuda.empty_cache()

            grad_accum = target_eb // micro_batch
            if headroom >= PROFILE_HEADROOM_GB:
                print(f"    batch={micro_batch}: peak={peak_mem:.1f}GB, "
                      f"headroom={headroom:.1f}GB -> OK")
                best = (micro_batch, grad_accum)
                break
            else:
                print(f"    batch={micro_batch}: peak={peak_mem:.1f}GB, "
                      f"headroom={headroom:.1f}GB -> too tight")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                print(f"    batch={micro_batch}: OOM")
            else:
                print(f"    batch={micro_batch}: error ({e})")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    # Clean up profiling model completely
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    micro_batch, grad_accum = best
    print(f"  -> Selected: micro_batch={micro_batch}, grad_accum={grad_accum}, "
          f"EB={micro_batch * grad_accum}")
    return best


# ============================================================================
# FORWARD PASSES
# ============================================================================

def forward_foveated(model, frames, caption_ids, caption_mask):
    """Foveated caption-only forward (training approximation)."""
    return model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)


def forward_baseline(model, frames, caption_ids, tokenizer):
    """Baseline caption forward."""
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:].clone()
    caption_targets[caption_targets == tokenizer.pad_token_id] = -100
    loss, _ = model(frames, caption_embeds, caption_targets)
    return loss


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(model, arch_type, val_samples, tokenizer, device, max_samples=MAX_VAL_SAMPLES):
    """Evaluate model, returning both train-approx and inference loss for foveated."""
    model.eval()
    losses_train = []
    losses_infer = []

    for i, sample in enumerate(val_samples):
        if i >= max_samples:
            break

        frames = sample['frames'].unsqueeze(0).to(device)
        caption = sample['caption']

        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)
        caption_mask = enc['attention_mask'].to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            if arch_type == 'foveated':
                # Training approximation loss
                loss_train = model.forward_captioning(
                    frames, caption_ids, caption_mask, use_fine=True)
                losses_train.append(loss_train.item())

                # True inference loss
                loss_infer = model.forward_autoregressive_captioning(
                    frames, caption_ids, caption_mask)
                losses_infer.append(loss_infer.item())
            else:
                caption_embeds = model.llm.model.embed_tokens(caption_ids)
                caption_targets = caption_ids[:, 1:].clone()
                caption_targets[caption_targets == tokenizer.pad_token_id] = -100
                loss, _ = model(frames, caption_embeds, caption_targets)
                losses_train.append(loss.item())
                losses_infer.append(loss.item())  # Same for baseline

    model.train()

    return {
        'loss_train': float(np.mean(losses_train)),
        'loss_train_std': float(np.std(losses_train)),
        'loss_infer': float(np.mean(losses_infer)),
        'loss_infer_std': float(np.std(losses_infer)),
        'ppl_train': float(np.exp(np.mean(losses_train))),
        'ppl_infer': float(np.exp(np.mean(losses_infer))),
        'n_samples': len(losses_train),
    }


# ============================================================================
# TRAINING + INLINE EVAL
# ============================================================================

def train_and_evaluate(run_name, model_cfg, frame_cfg, arch_type, device, results_file):
    """Train a model with profiled batch size and evaluate inline."""
    print(f"\n{'='*70}")
    print(f"RUN: {run_name}")
    print(f"  Model: {model_cfg['llm_model'].split('/')[-1]} + {model_cfg['dino_model'].split('/')[-1]}")
    print(f"  Arch: {arch_type}, Frames: {frame_cfg['num_frames']}")
    print(f"{'='*70}")

    # Step 1: Profile optimal batch size
    micro_batch, grad_accum = profile_optimal_batch(
        model_cfg, arch_type, frame_cfg, device)
    effective_batch = micro_batch * grad_accum
    assert effective_batch == TARGET_EB, f"EB mismatch: {effective_batch} != {TARGET_EB}"

    print(f"  Config: micro_batch={micro_batch}, grad_accum={grad_accum}, EB={effective_batch}")

    # Step 2: Create fresh model for training
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    tokenizer.pad_token = tokenizer.eos_token

    use_grad_ckpt = (model_cfg.get('priority') == 'P2' and frame_cfg['num_frames'] == 64)
    model = create_model(model_cfg, arch_type, device, use_grad_ckpt=use_grad_ckpt)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {param_count:.1f}M")
    print(f"  GPU Memory (model loaded): {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Step 3: Create DataLoader with proper batching
    train_dataset = ShardedDataset(
        frame_cfg['data_dir'], frame_cfg['num_frames'], frame_cfg['frame_size'], train=True)
    val_dataset = ShardedDataset(
        frame_cfg['data_dir'], frame_cfg['num_frames'], frame_cfg['frame_size'], train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=micro_batch,
        collate_fn=make_collate_fn(tokenizer),
        pin_memory=True, num_workers=0,
    )

    # Pre-load validation samples
    val_samples = []
    for s in val_dataset:
        val_samples.append(s)
        if len(val_samples) >= MAX_VAL_SAMPLES:
            break
    print(f"  Val samples: {len(val_samples)}")

    # FLOPs
    flops_per_sample = compute_flops_per_sample(model_cfg, frame_cfg, arch_type)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = GradScaler('cuda')

    # Training loop
    model.train()
    step = 0
    micro_step = 0
    accum_loss = 0.0
    results = []
    gpu_utils = []

    pbar = tqdm(total=MAX_STEPS, desc=run_name)
    eval_steps = set(range(EVAL_EVERY, MAX_STEPS + 1, EVAL_EVERY))
    start_time = time.time()

    for batch in train_loader:
        if step >= MAX_STEPS:
            break

        frames = batch['frames'].to(device, non_blocking=True)
        caption_ids = batch['caption_ids'].to(device, non_blocking=True)
        caption_mask = batch['caption_mask'].to(device, non_blocking=True)

        # Forward
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            if arch_type == 'foveated':
                loss = forward_foveated(model, frames, caption_ids, caption_mask)
            else:
                loss = forward_baseline(model, frames, caption_ids, tokenizer)

        loss_scaled = loss / grad_accum
        scaler.scale(loss_scaled).backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step % grad_accum == 0:
            # LR warmup
            step += 1
            if step <= WARMUP_STEPS:
                lr = LEARNING_RATE * step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss = accum_loss / grad_accum
            accum_loss = 0.0

            pbar.update(1)
            pbar.set_postfix({'loss': f'{train_loss:.4f}'})

            # GPU monitoring for first 5 optimizer steps
            if step <= 5:
                stats = get_gpu_stats()
                if stats:
                    gpu_utils.append(stats['util'])
                if step == 5 and gpu_utils:
                    avg_util = np.mean(gpu_utils)
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    tqdm.write(
                        f"  GPU Monitor (steps 1-5): util={avg_util:.0f}%, "
                        f"peak_mem={peak_mem:.1f}GB, "
                        f"micro_batch={micro_batch}")
                    if avg_util < 50:
                        tqdm.write(
                            f"  WARNING: Low GPU utilization ({avg_util:.0f}%). "
                            f"Consider increasing micro_batch or investigating data loading.")

            # Inline evaluation
            if step in eval_steps:
                eval_result = evaluate_model(
                    model, arch_type, val_samples, tokenizer, device)

                total_flops = flops_per_sample * effective_batch * step

                result = {
                    'run': run_name,
                    'model_config': run_name.split('_')[0],
                    'arch_type': arch_type,
                    'frames': frame_cfg['num_frames'],
                    'step': step,
                    'train_loss_running': train_loss,
                    'loss_train': eval_result['loss_train'],
                    'loss_train_std': eval_result['loss_train_std'],
                    'loss_infer': eval_result['loss_infer'],
                    'loss_infer_std': eval_result['loss_infer_std'],
                    'ppl_train': eval_result['ppl_train'],
                    'ppl_infer': eval_result['ppl_infer'],
                    'n_val_samples': eval_result['n_samples'],
                    'flops_per_sample': flops_per_sample,
                    'total_flops': total_flops,
                    'effective_batch': effective_batch,
                    'micro_batch': micro_batch,
                    'grad_accum': grad_accum,
                    'visual_tokens': 1 if arch_type == 'foveated' else 16,
                    'llm_model': model_cfg['llm_model'],
                    'dino_model': model_cfg['dino_model'],
                    'param_count_M': param_count,
                    'timestamp': datetime.now().isoformat(),
                }
                results.append(result)

                # Save incrementally
                _save_results_incremental(results_file, result)

                gap = abs(eval_result['loss_train'] - eval_result['loss_infer'])
                gap_pct = gap / eval_result['loss_train'] * 100
                tqdm.write(
                    f"  Step {step}: "
                    f"train={eval_result['loss_train']:.4f} "
                    f"infer={eval_result['loss_infer']:.4f} "
                    f"gap={gap_pct:.2f}% "
                    f"FLOPs={total_flops/1e15:.3f}P"
                )

    pbar.close()
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed/60:.1f} min")

    # Cleanup
    del model, optimizer, scaler, val_samples, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    return results


def _save_results_incremental(results_file, new_result):
    """Append a single result to the JSON file."""
    results_file = Path(results_file)
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(new_result)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)


# ============================================================================
# PLOTTING
# ============================================================================

def generate_all_plots(results_file, output_dir):
    """Generate all 6 plot types from results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(results_file) as f:
        all_results = json.load(f)

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme: model config determines color, arch determines linestyle
    config_colors = {
        'S-S': '#1f77b4',  # blue
        'M-S': '#ff7f0e',  # orange
        'S-B': '#2ca02c',  # green
        'B-L': '#d62728',  # red
    }
    arch_styles = {
        'foveated': {'ls': '-', 'marker': 'o', 'ms': 5},
        'baseline': {'ls': '--', 'marker': 's', 'ms': 5},
    }

    # Group data by key
    def group_by(data, *keys):
        groups = {}
        for d in data:
            key = tuple(d[k] for k in keys)
            groups.setdefault(key, []).append(d)
        for k in groups:
            groups[k] = sorted(groups[k], key=lambda x: x['step'])
        return groups

    # =========================================================================
    # Plot 1: Loss vs Training FLOPs (2 panels: 8F, 64F)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, n_frames in enumerate([8, 64]):
        ax = axes[ax_idx]
        frame_data = [d for d in all_results if d['frames'] == n_frames]
        groups = group_by(frame_data, 'model_config', 'arch_type')

        for (cfg, arch), points in groups.items():
            color = config_colors.get(cfg, 'gray')
            style = arch_styles.get(arch, {'ls': '-', 'marker': 'o', 'ms': 5})
            flops = np.array([p['total_flops'] for p in points]) / 1e15
            losses_t = [p['loss_train'] for p in points]
            losses_i = [p['loss_infer'] for p in points]

            # Training approximation (solid/dashed per arch)
            ax.plot(flops, losses_t, color=color, linestyle=style['ls'],
                   marker=style['marker'], markersize=style['ms'],
                   label=f'{cfg} {arch}', linewidth=1.5)

            # True inference (dotted, only for foveated)
            if arch == 'foveated':
                ax.plot(flops, losses_i, color=color, linestyle=':',
                       marker=style['marker'], markersize=3, alpha=0.6,
                       label=f'{cfg} fov (inference)', linewidth=1)

        ax.set_xlabel('Training FLOPs (PFLOPs)', fontsize=11)
        ax.set_ylabel('Caption Loss (CE)', fontsize=11)
        ax.set_title(f'{n_frames}-Frame: Loss vs Compute', fontsize=13)
        ax.set_xscale('log')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_vs_flops.png', dpi=150)
    plt.savefig(plots_dir / 'loss_vs_flops.pdf')
    plt.close()

    # =========================================================================
    # Plot 2: ISO-Loss Curves
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, n_frames in enumerate([8, 64]):
        ax = axes[ax_idx]
        frame_data = [d for d in all_results if d['frames'] == n_frames]

        # Find reasonable target losses
        all_losses = [d['loss_infer'] for d in frame_data]
        if not all_losses:
            continue
        loss_min, loss_max = min(all_losses), max(all_losses)
        target_losses = np.linspace(loss_min + 0.1, loss_max - 0.2, 5)

        groups = group_by(frame_data, 'model_config', 'arch_type')
        bar_data = {}

        for (cfg, arch), points in groups.items():
            flops = np.array([p['total_flops'] for p in points])
            losses = np.array([p['loss_infer'] for p in points])

            # Interpolate to find FLOPs at target losses
            interp_flops = []
            for tl in target_losses:
                if losses.min() <= tl <= losses.max():
                    idx = np.searchsorted(-losses, -tl)
                    if 0 < idx < len(losses):
                        l1, l2 = losses[idx-1], losses[idx]
                        f1, f2 = flops[idx-1], flops[idx]
                        t = (tl - l1) / (l2 - l1) if l2 != l1 else 0.5
                        interp_flops.append(np.exp(np.log(f1) * (1-t) + np.log(f2) * t))
                    else:
                        interp_flops.append(np.nan)
                else:
                    interp_flops.append(np.nan)
            bar_data[(cfg, arch)] = np.array(interp_flops)

        # Plot grouped bars
        x = np.arange(len(target_losses))
        n_groups = len(bar_data)
        width = 0.8 / max(n_groups, 1)

        for i, ((cfg, arch), flops_arr) in enumerate(bar_data.items()):
            offset = (i - n_groups/2 + 0.5) * width
            color = config_colors.get(cfg, 'gray')
            alpha = 0.9 if arch == 'foveated' else 0.5
            hatch = '' if arch == 'foveated' else '//'
            valid = ~np.isnan(flops_arr)
            vals = flops_arr.copy() / 1e15
            vals[~valid] = 0
            ax.bar(x[valid] + offset, vals[valid], width, label=f'{cfg} {arch}',
                   color=color, alpha=alpha, hatch=hatch)

        ax.set_xlabel('Target Loss', fontsize=11)
        ax.set_ylabel('FLOPs Required (PFLOPs)', fontsize=11)
        ax.set_title(f'{n_frames}-Frame: ISO-Loss Curves', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l:.2f}' for l in target_losses])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / 'iso_loss_curves.png', dpi=150)
    plt.savefig(plots_dir / 'iso_loss_curves.pdf')
    plt.close()

    # =========================================================================
    # Plot 3: ISO-FLOP Curves
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, n_frames in enumerate([8, 64]):
        ax = axes[ax_idx]
        frame_data = [d for d in all_results if d['frames'] == n_frames]
        groups = group_by(frame_data, 'model_config', 'arch_type')

        # Find common FLOP budgets
        all_flops = sorted(set(d['total_flops'] for d in frame_data))
        if len(all_flops) < 2:
            continue
        target_flops = np.logspace(np.log10(min(all_flops)), np.log10(max(all_flops)), 8)

        for (cfg, arch), points in groups.items():
            color = config_colors.get(cfg, 'gray')
            style = arch_styles.get(arch, {'ls': '-', 'marker': 'o', 'ms': 5})
            flops = np.array([p['total_flops'] for p in points])
            losses = np.array([p['loss_infer'] for p in points])

            # Interpolate losses at target FLOPs
            interp_losses = np.interp(target_flops, flops, losses,
                                      left=np.nan, right=np.nan)
            valid = ~np.isnan(interp_losses)
            ax.plot(target_flops[valid] / 1e15, interp_losses[valid],
                   color=color, linestyle=style['ls'],
                   marker=style['marker'], markersize=style['ms'],
                   label=f'{cfg} {arch}', linewidth=1.5)

        ax.set_xlabel('Compute Budget (PFLOPs)', fontsize=11)
        ax.set_ylabel('Best Loss Achievable', fontsize=11)
        ax.set_title(f'{n_frames}-Frame: ISO-FLOP Curves', fontsize=13)
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'iso_flop_curves.png', dpi=150)
    plt.savefig(plots_dir / 'iso_flop_curves.pdf')
    plt.close()

    # =========================================================================
    # Plot 4: Perplexity vs FLOPs (log-log)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, n_frames in enumerate([8, 64]):
        ax = axes[ax_idx]
        frame_data = [d for d in all_results if d['frames'] == n_frames]
        groups = group_by(frame_data, 'model_config', 'arch_type')

        for (cfg, arch), points in groups.items():
            color = config_colors.get(cfg, 'gray')
            style = arch_styles.get(arch, {'ls': '-', 'marker': 'o', 'ms': 5})
            flops = np.array([p['total_flops'] for p in points]) / 1e15
            ppl = [p['ppl_infer'] for p in points]

            ax.plot(flops, ppl, color=color, linestyle=style['ls'],
                   marker=style['marker'], markersize=style['ms'],
                   label=f'{cfg} {arch}', linewidth=1.5)

        ax.set_xlabel('Training FLOPs (PFLOPs)', fontsize=11)
        ax.set_ylabel('Perplexity', fontsize=11)
        ax.set_title(f'{n_frames}-Frame: Perplexity vs Compute', fontsize=13)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'perplexity_vs_flops.png', dpi=150)
    plt.savefig(plots_dir / 'perplexity_vs_flops.pdf')
    plt.close()

    # =========================================================================
    # Plot 5: Model Size Comparison (final step bars)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, n_frames in enumerate([8, 64]):
        ax = axes[ax_idx]
        frame_data = [d for d in all_results if d['frames'] == n_frames]

        # Get final step per run
        groups = group_by(frame_data, 'model_config', 'arch_type')
        final_data = {}
        for key, points in groups.items():
            final_data[key] = points[-1]  # Last eval point

        configs = sorted(set(k[0] for k in final_data.keys()))
        x = np.arange(len(configs))
        width = 0.35

        fov_losses = []
        base_losses = []
        for cfg in configs:
            fov = final_data.get((cfg, 'foveated'))
            base = final_data.get((cfg, 'baseline'))
            fov_losses.append(fov['loss_infer'] if fov else 0)
            base_losses.append(base['loss_infer'] if base else 0)

        ax.bar(x - width/2, fov_losses, width,
               label='Foveated (1 tok/frame)', color='#1f77b4')
        ax.bar(x + width/2, base_losses, width,
               label='Baseline (16 tok/frame)', color='#ff7f0e')

        # Add value labels and delta
        for i, (fl, bl) in enumerate(zip(fov_losses, base_losses)):
            if fl > 0:
                ax.text(i - width/2, fl + 0.02, f'{fl:.3f}', ha='center', fontsize=8)
            if bl > 0:
                ax.text(i + width/2, bl + 0.02, f'{bl:.3f}', ha='center', fontsize=8)
            if fl > 0 and bl > 0:
                delta_pct = (fl - bl) / bl * 100
                color = 'red' if delta_pct > 0 else 'green'
                ax.annotate(f'{delta_pct:+.1f}%', xy=(i, max(fl, bl) + 0.08),
                           ha='center', fontsize=9, color=color, fontweight='bold')

        ax.set_xlabel('Model Configuration', fontsize=11)
        ax.set_ylabel('Caption Loss (CE)', fontsize=11)
        ax.set_title(f'{n_frames}-Frame: Model Size Comparison', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / 'model_size_comparison.png', dpi=150)
    plt.savefig(plots_dir / 'model_size_comparison.pdf')
    plt.close()

    # =========================================================================
    # Plot 6: Frame Scaling (8F vs 64F)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, arch in enumerate(['foveated', 'baseline']):
        ax = axes[ax_idx]
        arch_data = [d for d in all_results if d['arch_type'] == arch]
        groups = group_by(arch_data, 'model_config', 'frames')

        configs = sorted(set(d['model_config'] for d in arch_data))

        for cfg in configs:
            color = config_colors.get(cfg, 'gray')
            frame_counts = []
            final_losses = []

            for n_frames in [8, 64]:
                key = (cfg, n_frames)
                if key in groups:
                    points = groups[key]
                    final_losses.append(points[-1]['loss_infer'])
                    frame_counts.append(n_frames)

            if len(frame_counts) == 2:
                ax.plot(frame_counts, final_losses, 'o-', color=color,
                       label=cfg, linewidth=2, markersize=10)

                # Annotate improvement
                delta = final_losses[1] - final_losses[0]
                delta_pct = delta / final_losses[0] * 100
                mid_x = np.mean(frame_counts)
                mid_y = np.mean(final_losses)
                ax.annotate(f'{delta_pct:+.1f}%', xy=(mid_x, mid_y),
                           fontsize=10, ha='center', color=color)

        ax.set_xlabel('Number of Frames', fontsize=11)
        ax.set_ylabel('Caption Loss (CE)', fontsize=11)
        ax.set_title(f'{arch.capitalize()}: 8F vs 64F Scaling', fontsize=13)
        ax.set_xticks([8, 64])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'frame_scaling.png', dpi=150)
    plt.savefig(plots_dir / 'frame_scaling.pdf')
    plt.close()

    print(f"\nAll plots saved to {plots_dir}/")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Run':<20} {'Step':>5} {'Train Loss':>11} {'Infer Loss':>11} "
          f"{'Gap%':>6} {'FLOPs(P)':>9} {'PPL':>7} {'uBatch':>7}")
    print("-"*100)

    groups = group_by(all_results, 'run')
    for run_name in sorted(groups.keys()):
        points = groups[run_name]
        final = points[-1]
        gap = abs(final['loss_train'] - final['loss_infer']) / final['loss_train'] * 100
        ub = final.get('micro_batch', '?')
        print(f"{run_name:<20} {final['step']:>5} {final['loss_train']:>11.4f} "
              f"{final['loss_infer']:>11.4f} {gap:>5.2f}% "
              f"{final['total_flops']/1e15:>9.3f} {final['ppl_infer']:>7.1f} "
              f"{ub:>7}")


# ============================================================================
# MAIN
# ============================================================================

def build_run_list(priority='P1'):
    """Build list of (run_name, model_config_key, frame_config_key, arch_type)."""
    runs = []
    for mcfg_name, mcfg in MODEL_CONFIGS.items():
        if priority == 'P1' and mcfg['priority'] == 'P2':
            continue
        for fcfg_name in ['8f', '64f']:
            for arch in ['foveated', 'baseline']:
                run_name = f"{mcfg_name}_{fcfg_name}_{arch[:3]}"
                runs.append((run_name, mcfg_name, fcfg_name, arch))
    return runs


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Scaling Law Study')
    parser.add_argument('--priority', type=str, default='P1', choices=['P1', 'P2'],
                       help='P1=S-S/M-S/S-B only, P2=include B-L')
    parser.add_argument('--plot-only', action='store_true', help='Only regenerate plots')
    parser.add_argument('--run', type=str, default=None,
                       help='Run a specific experiment (e.g. SS_8f_fov)')
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_BASE / 'results.json'

    if args.plot_only:
        if results_file.exists():
            generate_all_plots(results_file, OUTPUT_BASE)
        else:
            print(f"No results file found: {results_file}")
        return

    device = torch.device('cuda')
    runs = build_run_list(args.priority)

    # Filter to single run if specified
    if args.run:
        runs = [r for r in runs if r[0] == args.run]
        if not runs:
            print(f"Unknown run: {args.run}")
            print(f"Available: {[r[0] for r in build_run_list('P2')]}")
            return

    # Check for stale results from old (buggy) runs and clear them
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)
        # Old results don't have 'micro_batch' field - they used broken batching
        if existing and 'micro_batch' not in existing[0]:
            backup = results_file.with_suffix('.json.bak')
            print(f"  Old results detected (no micro_batch field). "
                  f"Backing up to {backup.name} and starting fresh.")
            import shutil
            shutil.copy2(results_file, backup)
            results_file.unlink()
            existing = []

    # Skip already completed runs (only valid results with 20 eval points)
    existing_results = []
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
    # A run is complete if it has all 20 eval points (step 280)
    completed_runs = set()
    from collections import Counter
    run_counts = Counter(d['run'] for d in existing_results)
    for run_name, count in run_counts.items():
        max_step = max(d['step'] for d in existing_results if d['run'] == run_name)
        if max_step >= MAX_STEPS:
            completed_runs.add(run_name)

    print("="*70)
    print("COMPREHENSIVE SCALING LAW STUDY (v2 - proper batching)")
    print("="*70)
    print(f"Priority: {args.priority}")
    print(f"Total runs: {len(runs)}")
    print(f"Already completed: {len(completed_runs & set(r[0] for r in runs))}")
    print(f"Target EB: {TARGET_EB}")
    print(f"Results: {results_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    for run_name, mcfg_name, fcfg_name, arch_type in runs:
        if run_name in completed_runs:
            print(f"\nSkipping {run_name} (already completed)")
            continue

        # Clean partial results for this run (if restarting)
        if results_file.exists():
            with open(results_file) as f:
                all_data = json.load(f)
            cleaned = [d for d in all_data if d['run'] != run_name]
            if len(cleaned) != len(all_data):
                print(f"\n  Removing {len(all_data) - len(cleaned)} partial results for {run_name}")
                with open(results_file, 'w') as f:
                    json.dump(cleaned, f, indent=2)

        model_cfg = MODEL_CONFIGS[mcfg_name]
        frame_cfg = FRAME_CONFIGS[fcfg_name]

        try:
            train_and_evaluate(
                run_name, model_cfg, frame_cfg, arch_type,
                device, results_file
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM on {run_name}! Skipping...")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n  ERROR on {run_name}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # Generate plots
    if results_file.exists():
        generate_all_plots(results_file, OUTPUT_BASE)

    print(f"\n{'='*70}")
    print(f"STUDY COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
