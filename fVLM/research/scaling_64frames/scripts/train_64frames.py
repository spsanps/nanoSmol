#!/usr/bin/env python3
"""
Train foveated and baseline models on 64-frame data.
Produces scaling law checkpoints at steps 100, 300, 1000, 3000.
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
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model.foveated_vlm import FoveatedVideoModel
from model.baseline_vlm import BaselineVLM

# Paths
DATA_DIR = Path('/mnt/d/projects/fVLM/data/webvid_64frames/shards')
OUTPUT_BASE = Path('/mnt/d/projects/fVLM/outputs/scaling_64frames')
SPLIT_FILE = Path('/mnt/d/projects/fVLM/data/webvid_64frames/data_split.json')

# Training config
NUM_FRAMES = 64
FRAME_SIZE = 224
BATCH_SIZE = 2  # Smaller due to 64 frames
GRAD_ACCUM = 8  # Effective batch = 16
LEARNING_RATE = 3e-5
CHECKPOINT_STEPS = [100, 300, 1000, 3000]
MAX_STEPS = 3000
WARMUP_STEPS = 100

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
                print(f"Error loading {shard_path}: {e}")
                continue

    def _process(self, data):
        frames = data['frames']  # [T, 3, H, W] uint8
        caption = data['caption']

        T = frames.shape[0]
        # Sample exactly num_frames
        if T >= self.num_frames:
            indices = torch.linspace(0, T-1, self.num_frames).long()
        else:
            # Repeat last frame if needed
            indices = list(range(T)) + [T-1] * (self.num_frames - T)
            indices = torch.tensor(indices)

        frames = frames[indices].float()

        # Normalize
        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'caption': caption}


def create_split(shard_dir, output_file, val_ratio=0.1, seed=42):
    """Create train/val split from shards."""
    shards = sorted([f.name for f in Path(shard_dir).glob('*.pt')])

    np.random.seed(seed)
    np.random.shuffle(shards)

    n_val = max(1, int(len(shards) * val_ratio))
    val_shards = shards[:n_val]
    train_shards = shards[n_val:]

    split = {
        'seed': seed,
        'val_shards': val_shards,
        'train_shards': train_shards,
        'val_count': n_val,
        'train_count': len(train_shards),
    }

    with open(output_file, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"Created split: {len(train_shards)} train, {len(val_shards)} val shards")
    return split


def train_foveated(train_shards, output_dir, device):
    """Train foveated model on 64-frame data."""
    print("\n" + "="*70)
    print("TRAINING FOVEATED MODEL (64 frames)")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = FoveatedVideoModel(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        dino_dim=384,
        llm_dim=576,
        query_dim=384,
        deep_query=True,
        freeze_dino=False,
    ).to(device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = Shard64Dataset(DATA_DIR, train_shards, num_frames=NUM_FRAMES)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    step = 0
    accum_loss = 0
    accum_steps = 0

    pbar = tqdm(total=MAX_STEPS, desc="Training foveated")

    data_iter = iter(dataset)

    while step < MAX_STEPS:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            sample = next(data_iter)

        frames = sample['frames'].unsqueeze(0).to(device)  # [1, 64, 3, H, W]
        caption = sample['caption']

        # Tokenize
        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)
        caption_mask = enc['attention_mask'].to(device)

        # Forward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)

        loss = loss / GRAD_ACCUM
        loss.backward()

        accum_loss += loss.item() * GRAD_ACCUM
        accum_steps += 1

        if accum_steps >= GRAD_ACCUM:
            # Warmup LR
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            avg_loss = accum_loss / GRAD_ACCUM
            accum_loss = 0
            accum_steps = 0

            pbar.update(1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Checkpoint
            if step in CHECKPOINT_STEPS:
                ckpt_path = output_dir / f'step_{step:06d}.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
                print(f"\n  Saved checkpoint: {ckpt_path}")

    pbar.close()
    print("Foveated training complete!")

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def train_baseline(train_shards, output_dir, device):
    """Train baseline model on 64-frame data."""
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL (64 frames)")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = BaselineVLM(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        freeze_dino=False,
        freeze_llm=False,
    ).to(device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = Shard64Dataset(DATA_DIR, train_shards, num_frames=NUM_FRAMES)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    step = 0
    accum_loss = 0
    accum_steps = 0

    pbar = tqdm(total=MAX_STEPS, desc="Training baseline")

    data_iter = iter(dataset)

    while step < MAX_STEPS:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            sample = next(data_iter)

        frames = sample['frames'].unsqueeze(0).to(device)  # [1, 64, 3, H, W]
        caption = sample['caption']

        # Tokenize
        enc = tokenizer(caption, return_tensors='pt', padding='max_length',
                       max_length=64, truncation=True)
        caption_ids = enc['input_ids'].to(device)

        # Get embeddings and targets
        caption_embeds = model.llm.get_input_embeddings()(caption_ids)
        caption_targets = caption_ids[:, 1:].clone()
        caption_targets[caption_targets == tokenizer.pad_token_id] = -100

        # Forward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _ = model(frames, caption_embeds, caption_targets)

        loss = loss / GRAD_ACCUM
        loss.backward()

        accum_loss += loss.item() * GRAD_ACCUM
        accum_steps += 1

        if accum_steps >= GRAD_ACCUM:
            # Warmup LR
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            avg_loss = accum_loss / GRAD_ACCUM
            accum_loss = 0
            accum_steps = 0

            pbar.update(1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Checkpoint
            if step in CHECKPOINT_STEPS:
                ckpt_path = output_dir / f'step_{step:06d}.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
                print(f"\n  Saved checkpoint: {ckpt_path}")

    pbar.close()
    print("Baseline training complete!")

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['foveated', 'baseline', 'both'],
                       default='both')
    args = parser.parse_args()

    device = torch.device('cuda')

    # Create split if needed
    if not SPLIT_FILE.exists():
        if not DATA_DIR.exists():
            print(f"Data directory not found: {DATA_DIR}")
            print("Please run download_long_videos.py first")
            return
        create_split(DATA_DIR, SPLIT_FILE)

    # Load split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_shards = split['train_shards']

    print(f"Training with {len(train_shards)} shards")

    # Train
    if args.model in ['foveated', 'both']:
        train_foveated(train_shards, OUTPUT_BASE / 'foveated_64f', device)

    if args.model in ['baseline', 'both']:
        train_baseline(train_shards, OUTPUT_BASE / 'baseline_64f', device)

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
