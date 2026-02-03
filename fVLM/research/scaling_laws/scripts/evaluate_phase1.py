#!/usr/bin/env python3
"""Evaluate Phase 1 checkpoints (S-S config)."""

import os
import sys
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Config
SHARD_DIR = Path('/mnt/d/projects/fVLM/data/frames_latents_sharded')
SPLIT_FILE = PROJECT_ROOT / 'configs' / 'data_split.json'
OUTPUT_BASE = Path('/mnt/d/projects/fVLM/outputs/scaling_study')
DATA_DIR = Path(__file__).parent.parent / 'data'

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

MAX_VAL_SAMPLES = 300  # Reduced for faster evaluation

CONFIGS = {
    'foveated_opt': {'model_type': 'foveated', 'frame_size': 224, 'fine_iterations': 1, 'flops': 144.7e9, 'tokens': 1},
    'baseline': {'model_type': 'baseline', 'frame_size': 224, 'fine_iterations': 0, 'flops': 151.3e9, 'tokens': 16},
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


def main():
    print("=" * 70)
    print("PHASE 1 EVALUATION (S-S Config)")
    print("=" * 70)

    device = torch.device('cuda')
    gc.collect()
    torch.cuda.empty_cache()

    # Load split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    val_shards = split['val_shards']

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    all_results = []

    for exp_name, cfg in CONFIGS.items():
        print(f'\n=== EVALUATING: {exp_name} ===')

        # Load val samples
        val_dataset = ShardedDataset(SHARD_DIR, val_shards)
        samples = []
        for s in val_dataset:
            samples.append(s)
            if len(samples) >= MAX_VAL_SAMPLES:
                break
        print(f'Loaded {len(samples)} val samples')

        # Create model
        if cfg['model_type'] == 'foveated':
            from src.model.foveated_vlm import FoveatedVideoModel
            model = FoveatedVideoModel(
                dino_model='facebook/dinov2-small',
                llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
                deep_query=True, freeze_dino=False,
            ).to(device)
        else:
            from src.model.baseline_vlm import BaselineVLM
            model = BaselineVLM(
                dino_model='facebook/dinov2-small',
                llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
                pixel_shuffle_scale=4,
            ).to(device)

        exp_dir = OUTPUT_BASE / exp_name
        for ckpt_path in sorted(exp_dir.glob('step_*.pt')):
            step = int(ckpt_path.stem.split('_')[1])
            print(f'  Step {step}...', end=' ', flush=True)

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            losses = []
            with torch.no_grad():
                for s in tqdm(samples, desc=f"Eval {step}", leave=False):
                    frames = s['frames'].unsqueeze(0).to(device)
                    caption = s['caption']

                    tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
                    caption_ids = tokens['input_ids'].to(device)
                    caption_mask = torch.ones_like(caption_ids)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        if cfg['model_type'] == 'foveated':
                            loss = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)
                        else:
                            caption_embeds = model.llm.model.embed_tokens(caption_ids)
                            caption_targets = caption_ids[:, 1:].clone()
                            caption_targets[caption_targets == tokenizer.pad_token_id] = -100
                            loss, _ = model.forward(frames, caption_embeds, caption_targets)

                    losses.append(loss.item())

            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            se_loss = std_loss / np.sqrt(len(losses))
            ppl = np.exp(mean_loss)
            total_flops = step * 16 * cfg['flops'] * 3

            print(f'loss={mean_loss:.4f}, ppl={ppl:.1f}')

            all_results.append({
                'experiment': exp_name,
                'model_type': cfg['model_type'],
                'model_config': 'S-S',
                'step': step,
                'val_loss': float(mean_loss),
                'val_loss_std': float(std_loss),
                'val_loss_se': float(se_loss),
                'perplexity': float(ppl),
                'n_samples': len(losses),
                'flops_per_sample': cfg['flops'],
                'total_training_flops': total_flops,
                'visual_tokens_per_frame': cfg['tokens'],
            })

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / 'scaling_data_S-S.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    json_path = DATA_DIR / 'scaling_data_S-S.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nSaved: {csv_path}')
    print(f'Saved: {json_path}')

    # Print summary
    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    print(f"{'Experiment':<20} {'Step':>6} {'Loss':>8} {'PPL':>8} {'FLOPs(PF)':>12}")
    print('-' * 60)
    for r in sorted(all_results, key=lambda x: (x['experiment'], x['step'])):
        print(f"{r['experiment']:<20} {r['step']:>6} {r['val_loss']:>8.4f} {r['perplexity']:>8.1f} {r['total_training_flops']/1e15:>12.4f}")


if __name__ == "__main__":
    main()
