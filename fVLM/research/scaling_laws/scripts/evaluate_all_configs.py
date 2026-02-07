#!/usr/bin/env python3
"""
Evaluate all scaling law configurations (S-S, M-S, etc.)
Generates combined scaling data for analysis.
"""

import os
import sys
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model.foveated_vlm import FoveatedVideoModel

# Paths
SHARD_DIR = Path('/mnt/d/projects/fVLM/data/frames_latents_sharded')
SPLIT_FILE = PROJECT_ROOT / 'configs' / 'data_split.json'
OUTPUT_DIR = Path(__file__).parent.parent / 'data'

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

MAX_VAL_SAMPLES = 300
CHECKPOINT_STEPS = [100, 300, 1000, 3000]

# Model configurations
CONFIGS = {
    'S-S': {
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_model': 'facebook/dinov2-small',
        'llm_dim': 576,
        'dino_dim': 384,
        # FLOPs: 6 * LLM_params * seq_len + DINO_flops
        # Foveated seq: 1 + 8*1 + 64 = 73 tokens
        # Baseline seq: 1 + 8*16 + 64 = 193 tokens
        'flops_foveated': 329.5e9,  # GFLOPs per sample (corrected)
        'flops_baseline': 426.7e9,  # +29.5% more FLOPs
        'output_base': Path('/mnt/d/projects/fVLM/outputs/scaling_study'),
        'foveated_dir': 'foveated_opt',
        'baseline_dir': 'baseline',
    },
    'M-S': {
        'llm_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'dino_model': 'facebook/dinov2-small',
        'llm_dim': 960,
        'dino_dim': 384,
        # Foveated seq: 73 tokens, Baseline seq: 193 tokens
        'flops_foveated': 428.0e9,  # GFLOPs per sample (corrected)
        'flops_baseline': 687.2e9,  # +60.6% more FLOPs
        'output_base': Path('/mnt/d/projects/fVLM/outputs/scaling_multisize'),
        'foveated_dir': 'foveated_M-S',
        'baseline_dir': 'baseline_M-S',
    },
}

# Dataset
from torch.utils.data import IterableDataset

class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir, shard_list, num_frames=8, frame_size=224):
        self.shard_dir = Path(shard_dir)
        self.shard_files = [self.shard_dir / s for s in shard_list if (self.shard_dir / s).exists()]
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location='cpu', weights_only=False)
                for sample in shard['samples']:
                    yield self._process(sample)
                del shard
            except:
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
        if frames.shape[-1] != self.frame_size:
            frames = F.interpolate(frames, size=(self.frame_size, self.frame_size), mode='bilinear', align_corners=False)
        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        return {'frames': frames, 'caption': caption}


def load_foveated_model(checkpoint_path, cfg, device):
    """Load foveated model from checkpoint."""
    model = FoveatedVideoModel(
        dino_model=cfg['dino_model'],
        llm_model=cfg['llm_model'],
        dino_dim=cfg['dino_dim'],
        llm_dim=cfg['llm_dim'],
        query_dim=cfg['dino_dim'],
        deep_query=True,
        freeze_dino=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_baseline_model(checkpoint_path, cfg, device):
    """Load baseline VLM model from checkpoint."""
    from model.baseline_vlm import BaselineVLM

    model = BaselineVLM(
        dino_model=cfg['dino_model'],
        llm_model=cfg['llm_model'],
        freeze_dino=False,
        freeze_llm=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate_foveated(model, samples, tokenizer, device):
    """Evaluate foveated model on samples."""
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
    """Evaluate baseline model on samples."""
    losses = []

    for sample in tqdm(samples, desc="Evaluating baseline"):
        frames = sample['frames'].unsqueeze(0).to(device)
        caption = sample['caption']

        # Tokenize caption
        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)

        # Get caption embeddings
        caption_embeds = model.llm.get_input_embeddings()(caption_ids)  # [1, L, llm_dim]

        # Caption targets are the shifted caption ids (for next-token prediction)
        # Baseline model does: seq = [visual_token, visual_features, caption_embeds]
        #                      caption_logits = logits[:, visual_len:-1, :]
        # If caption_embeds has L tokens, caption_logits has L-1 tokens
        # These logits at position i predict caption[i+1]
        # So targets should be caption[1:L] which has L-1 tokens
        caption_targets = caption_ids[:, 1:].clone()  # [1, L-1] - tokens 1 to L-1
        # Mask padding tokens
        caption_targets[caption_targets == tokenizer.pad_token_id] = -100
        # Keep all caption embeds - the model handles the slicing
        # caption_embeds stays as [1, L, dim]

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _ = model(frames, caption_embeds, caption_targets)

        losses.append(loss.item())

    return np.array(losses)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, choices=['S-S', 'M-S', 'all'], default='all')
    parser.add_argument('--model-type', type=str, choices=['foveated', 'baseline', 'both'], default='both')
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    val_shards = split['val_shards']

    # Load tokenizer (same for all)
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    configs_to_eval = [args.config] if args.config != 'all' else list(CONFIGS.keys())

    all_results = []

    for config_name in configs_to_eval:
        cfg = CONFIGS[config_name]
        print(f"\n{'='*70}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*70}")

        # Load validation samples
        val_dataset = ShardedDataset(SHARD_DIR, val_shards, frame_size=224)
        samples = []
        for s in val_dataset:
            samples.append(s)
            if len(samples) >= MAX_VAL_SAMPLES:
                break
        print(f"Loaded {len(samples)} validation samples")

        # Evaluate foveated
        if args.model_type in ['foveated', 'both']:
            fov_dir = cfg['output_base'] / cfg['foveated_dir']
            for step in CHECKPOINT_STEPS:
                ckpt_path = fov_dir / f'step_{step:06d}.pt'
                if not ckpt_path.exists():
                    print(f"  Skipping foveated step {step} (not found)")
                    continue

                print(f"\n  Evaluating foveated @ step {step}...")
                gc.collect()
                torch.cuda.empty_cache()

                model = load_foveated_model(ckpt_path, cfg, device)
                losses = evaluate_foveated(model, samples, tokenizer, device)
                del model

                result = {
                    'config': config_name,
                    'model_type': 'foveated',
                    'step': step,
                    'loss_mean': float(np.mean(losses)),
                    'loss_std': float(np.std(losses)),
                    'loss_se': float(np.std(losses) / np.sqrt(len(losses))),
                    'perplexity': float(np.exp(np.mean(losses))),
                    'n_samples': len(losses),
                    'flops_per_sample': cfg['flops_foveated'],
                    'total_flops': cfg['flops_foveated'] * step * 48,  # Assume BS=48 effective
                    'visual_tokens': 1,
                }
                all_results.append(result)
                print(f"    Loss: {result['loss_mean']:.4f} +/- {result['loss_se']:.4f}, PPL: {result['perplexity']:.2f}")

        # Evaluate baseline
        if args.model_type in ['baseline', 'both']:
            base_dir = cfg['output_base'] / cfg['baseline_dir']
            for step in CHECKPOINT_STEPS:
                ckpt_path = base_dir / f'step_{step:06d}.pt'
                if not ckpt_path.exists():
                    print(f"  Skipping baseline step {step} (not found)")
                    continue

                print(f"\n  Evaluating baseline @ step {step}...")
                gc.collect()
                torch.cuda.empty_cache()

                model = load_baseline_model(ckpt_path, cfg, device)
                losses = evaluate_baseline(model, samples, tokenizer, device)
                del model

                result = {
                    'config': config_name,
                    'model_type': 'baseline',
                    'step': step,
                    'loss_mean': float(np.mean(losses)),
                    'loss_std': float(np.std(losses)),
                    'loss_se': float(np.std(losses) / np.sqrt(len(losses))),
                    'perplexity': float(np.exp(np.mean(losses))),
                    'n_samples': len(losses),
                    'flops_per_sample': cfg['flops_baseline'],
                    'total_flops': cfg['flops_baseline'] * step * 48,
                    'visual_tokens': 16,
                }
                all_results.append(result)
                print(f"    Loss: {result['loss_mean']:.4f} +/- {result['loss_se']:.4f}, PPL: {result['perplexity']:.2f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f'scaling_data_combined.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Config':<8} {'Type':<10} {'Step':<6} {'Loss':<10} {'PPL':<10} {'FLOPs (T)':<12}")
    print("-"*80)
    for r in all_results:
        print(f"{r['config']:<8} {r['model_type']:<10} {r['step']:<6} {r['loss_mean']:.4f}     {r['perplexity']:.2f}      {r['total_flops']/1e12:.2f}")


if __name__ == '__main__':
    main()
