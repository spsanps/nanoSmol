#!/usr/bin/env python3
"""
Evaluate 64-frame models and generate scaling comparison.
"""

import os
import sys
import gc
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model.foveated_vlm import FoveatedVideoModel
from model.baseline_vlm import BaselineVLM

# Paths
DATA_DIR = Path('/mnt/d/projects/fVLM/data/webvid_64frames/shards')
OUTPUT_BASE = Path('/mnt/d/projects/fVLM/outputs/scaling_64frames')
SPLIT_FILE = Path('/mnt/d/projects/fVLM/data/webvid_64frames/data_split.json')
RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Config
NUM_FRAMES = 64
MAX_VAL_SAMPLES = 100
CHECKPOINT_STEPS = [100, 300, 1000, 3000]

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Shard64Dataset(torch.utils.data.IterableDataset):
    """Dataset for 64-frame sharded data."""

    def __init__(self, shard_dir, shard_list, num_frames=64):
        self.shard_dir = Path(shard_dir)
        self.shard_files = [self.shard_dir / s for s in shard_list
                          if (self.shard_dir / s).exists()]
        self.num_frames = num_frames

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location='cpu', weights_only=False)
                for sample in shard['samples']:
                    yield self._process(sample)
                del shard
            except Exception as e:
                continue

    def _process(self, data):
        frames = data['frames']
        caption = data['caption']

        T = frames.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T-1, self.num_frames).long()
        else:
            indices = list(range(T)) + [T-1] * (self.num_frames - T)
            indices = torch.tensor(indices)

        frames = frames[indices].float()
        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'caption': caption}


def load_foveated(checkpoint_path, device):
    """Load foveated model."""
    model = FoveatedVideoModel(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        dino_dim=384,
        llm_dim=576,
        query_dim=384,
        deep_query=True,
        freeze_dino=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    return model


def load_baseline(checkpoint_path, device):
    """Load baseline model."""
    model = BaselineVLM(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        freeze_dino=False,
        freeze_llm=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    return model


def evaluate_foveated(model, samples, tokenizer, device):
    """Evaluate foveated model."""
    losses = []
    for sample in tqdm(samples, desc="Evaluating foveated"):
        frames = sample['frames'].unsqueeze(0).to(device)
        caption = sample['caption']

        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)
        caption_mask = enc['attention_mask'].to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)

        losses.append(loss.item())

    return np.array(losses)


def evaluate_baseline(model, samples, tokenizer, device):
    """Evaluate baseline model."""
    losses = []
    for sample in tqdm(samples, desc="Evaluating baseline"):
        frames = sample['frames'].unsqueeze(0).to(device)
        caption = sample['caption']

        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)

        caption_embeds = model.llm.get_input_embeddings()(caption_ids)
        caption_targets = caption_ids[:, 1:].clone()
        caption_targets[caption_targets == tokenizer.pad_token_id] = -100

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _ = model(frames, caption_embeds, caption_targets)

        losses.append(loss.item())

    return np.array(losses)


def main():
    device = torch.device('cuda')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load split
    if not SPLIT_FILE.exists():
        print(f"Split file not found: {SPLIT_FILE}")
        return

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    val_shards = split['val_shards']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # Load validation samples
    print(f"Loading validation data from {len(val_shards)} shards...")
    val_dataset = Shard64Dataset(DATA_DIR, val_shards, num_frames=NUM_FRAMES)
    samples = []
    for s in val_dataset:
        samples.append(s)
        if len(samples) >= MAX_VAL_SAMPLES:
            break
    print(f"Loaded {len(samples)} validation samples")

    all_results = []

    # FLOP estimates for 64-frame
    # Foveated: 1 + 64 + 64 = 129 tokens
    # Baseline: 1 + 64*16 + 64 = 1089 tokens
    # DINO: 6 * 22M * 256 * 64 = 2.16e12 FLOPs
    # LLM: 6 * 135M * seq_len
    DINO_FLOPS = 6 * 22e6 * 256 * 64
    FOV_LLM_FLOPS = 6 * 135e6 * 129
    BASE_LLM_FLOPS = 6 * 135e6 * 1089
    FLOPS_FOVEATED = DINO_FLOPS + FOV_LLM_FLOPS  # ~2.27e12
    FLOPS_BASELINE = DINO_FLOPS + BASE_LLM_FLOPS  # ~3.04e12

    print(f"\nFLOPs per sample:")
    print(f"  Foveated: {FLOPS_FOVEATED/1e9:.1f} GFLOPs")
    print(f"  Baseline: {FLOPS_BASELINE/1e9:.1f} GFLOPs")
    print(f"  Ratio: {FLOPS_BASELINE/FLOPS_FOVEATED:.2f}x")

    # Evaluate foveated
    fov_dir = OUTPUT_BASE / 'foveated_64f'
    for step in CHECKPOINT_STEPS:
        ckpt_path = fov_dir / f'step_{step:06d}.pt'
        if not ckpt_path.exists():
            print(f"Skipping foveated step {step} (not found)")
            continue

        print(f"\nEvaluating foveated @ step {step}...")
        gc.collect()
        torch.cuda.empty_cache()

        model = load_foveated(ckpt_path, device)
        losses = evaluate_foveated(model, samples, tokenizer, device)
        del model

        result = {
            'config': '64-frame',
            'model_type': 'foveated',
            'step': step,
            'loss_mean': float(np.mean(losses)),
            'loss_std': float(np.std(losses)),
            'loss_se': float(np.std(losses) / np.sqrt(len(losses))),
            'perplexity': float(np.exp(np.mean(losses))),
            'n_samples': len(losses),
            'flops_per_sample': FLOPS_FOVEATED,
            'total_flops': FLOPS_FOVEATED * step * 16,  # Effective BS=16
            'visual_tokens': 64,
            'num_frames': 64,
        }
        all_results.append(result)
        print(f"  Loss: {result['loss_mean']:.4f} +/- {result['loss_se']:.4f}")

    # Evaluate baseline
    base_dir = OUTPUT_BASE / 'baseline_64f'
    for step in CHECKPOINT_STEPS:
        ckpt_path = base_dir / f'step_{step:06d}.pt'
        if not ckpt_path.exists():
            print(f"Skipping baseline step {step} (not found)")
            continue

        print(f"\nEvaluating baseline @ step {step}...")
        gc.collect()
        torch.cuda.empty_cache()

        model = load_baseline(ckpt_path, device)
        losses = evaluate_baseline(model, samples, tokenizer, device)
        del model

        result = {
            'config': '64-frame',
            'model_type': 'baseline',
            'step': step,
            'loss_mean': float(np.mean(losses)),
            'loss_std': float(np.std(losses)),
            'loss_se': float(np.std(losses) / np.sqrt(len(losses))),
            'perplexity': float(np.exp(np.mean(losses))),
            'n_samples': len(losses),
            'flops_per_sample': FLOPS_BASELINE,
            'total_flops': FLOPS_BASELINE * step * 16,
            'visual_tokens': 64 * 16,  # 1024 tokens
            'num_frames': 64,
        }
        all_results.append(result)
        print(f"  Loss: {result['loss_mean']:.4f} +/- {result['loss_se']:.4f}")

    # Save results
    results_file = RESULTS_DIR / 'scaling_64frames.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate comparison plot
    if all_results:
        plot_comparison(all_results, RESULTS_DIR)

    # Print summary
    print("\n" + "="*70)
    print("64-FRAME SCALING RESULTS")
    print("="*70)
    print(f"{'Type':<12} {'Step':<6} {'Loss':<10} {'PPL':<10} {'FLOPs (T)':<12}")
    print("-"*70)
    for r in sorted(all_results, key=lambda x: (x['model_type'], x['step'])):
        print(f"{r['model_type']:<12} {r['step']:<6} {r['loss_mean']:.4f}     "
              f"{r['perplexity']:.2f}      {r['total_flops']/1e12:.2f}")


def plot_comparison(results, output_dir):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs FLOPs
    ax = axes[0]
    for model_type in ['foveated', 'baseline']:
        subset = [r for r in results if r['model_type'] == model_type]
        if not subset:
            continue
        subset = sorted(subset, key=lambda x: x['total_flops'])
        flops = [r['total_flops'] / 1e12 for r in subset]
        losses = [r['loss_mean'] for r in subset]
        loss_se = [r['loss_se'] for r in subset]

        color = 'C0' if model_type == 'foveated' else 'C1'
        marker = 'o' if model_type == 'foveated' else 's'
        tokens = 64 if model_type == 'foveated' else 1024

        ax.errorbar(flops, losses, yerr=loss_se, label=f'{model_type} ({tokens} tokens)',
                   color=color, marker=marker, capsize=3, linewidth=2, markersize=8)

    ax.set_xlabel('Training FLOPs (TFLOPs)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('64-Frame Models: Loss vs Compute', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Bar comparison at final step
    ax = axes[1]
    final = [r for r in results if r['step'] == 3000]
    if len(final) >= 2:
        fov = next((r for r in final if r['model_type'] == 'foveated'), None)
        base = next((r for r in final if r['model_type'] == 'baseline'), None)

        if fov and base:
            x = np.arange(2)
            width = 0.6

            losses = [fov['loss_mean'], base['loss_mean']]
            errors = [fov['loss_se'], base['loss_se']]
            flops_ratio = base['total_flops'] / fov['total_flops']

            bars = ax.bar(x, losses, width, yerr=errors, capsize=5,
                         color=['C0', 'C1'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Foveated\n(64 tokens)\n{fov["total_flops"]/1e12:.1f} TFLOPs',
                               f'Baseline\n(1024 tokens)\n{base["total_flops"]/1e12:.1f} TFLOPs'])
            ax.set_ylabel('Validation Loss @ 3000 steps', fontsize=12)
            ax.set_title(f'64-Frame Efficiency: Baseline uses {flops_ratio:.1f}x more FLOPs', fontsize=14)

            # Add delta annotation
            delta_pct = (fov['loss_mean'] - base['loss_mean']) / base['loss_mean'] * 100
            ax.annotate(f'{delta_pct:+.1f}%', xy=(0.5, max(losses) + 0.1),
                       ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_64frames.png', dpi=150)
    plt.savefig(output_dir / 'comparison_64frames.pdf')
    print(f"Saved plot: {output_dir / 'comparison_64frames.png'}")
    plt.close()


if __name__ == '__main__':
    main()
