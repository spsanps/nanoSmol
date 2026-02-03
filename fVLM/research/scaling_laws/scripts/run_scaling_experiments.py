#!/usr/bin/env python3
"""
Run scaling law experiments for foveated vs baseline VLM.

Trains multiple configurations at multiple step counts to gather
data points for scaling law analysis.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENTS = [
    # Foveated Optimized (224px, 1 fine iter)
    {"name": "foveated_opt", "frame_size": 224, "fine_iterations": 1, "model_type": "foveated"},
    # Baseline (224px, 16 tok/frame)
    {"name": "baseline", "frame_size": 224, "fine_iterations": 0, "model_type": "baseline"},
    # Foveated Original (256px, 2 fine iter) - for comparison
    {"name": "foveated_orig", "frame_size": 256, "fine_iterations": 2, "model_type": "foveated"},
]

STEP_COUNTS = [100, 300, 1000, 3000]

COMMON_CONFIG = {
    "batch_size": 16,
    "learning_rate": 3e-5,
    "warmup_steps": 50,
    "num_frames": 8,
    "lambda_recon": 0.5,
    "deep_query": True,
    "freeze_dino": False,
}

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
SPLIT_FILE = PROJECT_ROOT / "configs" / "data_split.json"
OUTPUT_BASE = Path("/mnt/d/projects/fVLM/outputs/scaling_study")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================================
# Dataset
# ============================================================================

class ShardedVideoDataset(IterableDataset):
    def __init__(self, shard_dir, shard_whitelist, num_frames=8, frame_size=224):
        self.shard_dir = Path(shard_dir)
        self.shard_files = sorted([
            f for f in self.shard_dir.glob("*.pt")
            if f.name in shard_whitelist
        ])
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                for sample in shard['samples']:
                    yield self._process(sample)
                del shard
            except Exception as e:
                print(f"Skipping {shard_path.name}: {e}")

    def _process(self, data):
        frames = data['frames']
        latents = data['latents']
        caption = data['caption']

        T = frames.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            indices = torch.arange(T)
            pad = self.num_frames - T
            indices = torch.cat([indices, indices[-1:].expand(pad)])

        frames = frames[indices].float()
        latents = latents[indices].float()

        # Resize if needed
        if frames.shape[-1] != self.frame_size:
            frames = F.interpolate(
                frames, size=(self.frame_size, self.frame_size),
                mode='bilinear', align_corners=False
            )

        # Normalize
        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {
            'frames': frames,
            'latents': latents,
            'caption': caption,
        }


def collate_fn(batch):
    return {
        'frames': torch.stack([b['frames'] for b in batch]),
        'latents': torch.stack([b['latents'] for b in batch]),
        'captions': [b['caption'] for b in batch],
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_foveated(model, tokenizer, dataloader, config, device, max_steps, checkpoint_steps):
    """Train foveated model and save checkpoints at specified steps."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    checkpoints = {}
    step = 0
    model.train()

    start_time = time.time()
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        frames = batch['frames'].to(device)
        latents = batch['latents'].to(device)
        captions = batch['captions']

        # Tokenize captions
        tokens = tokenizer(
            captions, padding=True, truncation=True,
            max_length=64, return_tensors='pt'
        ).to(device)

        # Forward pass
        loss, metrics = model.forward_joint_multifine(
            frames, latents, tokens['input_ids'],
            fine_iterations=config['fine_iterations'],
            lambda_recon=config['lambda_recon'],
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Learning rate warmup
        if step <= config['warmup_steps']:
            lr_scale = step / config['warmup_steps']
            for pg in optimizer.param_groups:
                pg['lr'] = config['learning_rate'] * lr_scale

        # Save checkpoint at specified steps
        if step in checkpoint_steps:
            elapsed = time.time() - start_time
            checkpoints[step] = {
                'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'step': step,
                'loss': loss.item(),
                'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()},
                'elapsed_time': elapsed,
            }
            print(f"  Step {step}: loss={loss.item():.4f}, time={elapsed:.1f}s")

        if step % 100 == 0:
            print(f"  Step {step}/{max_steps}: loss={loss.item():.4f}")

    return checkpoints


def train_baseline(model, tokenizer, dataloader, config, device, max_steps, checkpoint_steps):
    """Train baseline model and save checkpoints at specified steps."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    checkpoints = {}
    step = 0
    model.train()

    start_time = time.time()
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        frames = batch['frames'].to(device)
        captions = batch['captions']

        # Tokenize captions
        tokens = tokenizer(
            captions, padding=True, truncation=True,
            max_length=64, return_tensors='pt'
        ).to(device)

        # Forward pass
        loss = model.forward_caption(frames, tokens['input_ids'])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Learning rate warmup
        if step <= config['warmup_steps']:
            lr_scale = step / config['warmup_steps']
            for pg in optimizer.param_groups:
                pg['lr'] = config['learning_rate'] * lr_scale

        # Save checkpoint at specified steps
        if step in checkpoint_steps:
            elapsed = time.time() - start_time
            checkpoints[step] = {
                'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'step': step,
                'loss': loss.item(),
                'elapsed_time': elapsed,
            }
            print(f"  Step {step}: loss={loss.item():.4f}, time={elapsed:.1f}s")

        if step % 100 == 0:
            print(f"  Step {step}/{max_steps}: loss={loss.item():.4f}")

    return checkpoints


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("SCALING LAW EXPERIMENTS")
    print("=" * 80)

    device = torch.device("cuda")

    # Load data split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_shards = set(split['train_shards'])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []

    for exp_config in EXPERIMENTS:
        exp_name = exp_config['name']
        frame_size = exp_config['frame_size']
        fine_iterations = exp_config['fine_iterations']
        model_type = exp_config['model_type']

        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"  Frame size: {frame_size}, Fine iterations: {fine_iterations}")
        print(f"{'='*80}")

        # Create dataset
        dataset = ShardedVideoDataset(
            SHARD_DIR, train_shards,
            num_frames=COMMON_CONFIG['num_frames'],
            frame_size=frame_size
        )
        dataloader = DataLoader(
            dataset, batch_size=COMMON_CONFIG['batch_size'],
            collate_fn=collate_fn, num_workers=2
        )

        # Create model
        if model_type == "foveated":
            from src.model.foveated_vlm import FoveatedVideoModel
            model = FoveatedVideoModel(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                deep_query=COMMON_CONFIG['deep_query'],
                freeze_dino=COMMON_CONFIG['freeze_dino'],
            ).to(device)
        else:
            from src.model.baseline_vlm import BaselineVLM
            model = BaselineVLM(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                pixel_shuffle_scale=4,
            ).to(device)

        # Train and collect checkpoints
        max_steps = max(STEP_COUNTS)
        config = {**COMMON_CONFIG, 'fine_iterations': fine_iterations}

        if model_type == "foveated":
            checkpoints = train_foveated(
                model, tokenizer, dataloader, config, device,
                max_steps, set(STEP_COUNTS)
            )
        else:
            checkpoints = train_baseline(
                model, tokenizer, dataloader, config, device,
                max_steps, set(STEP_COUNTS)
            )

        # Save checkpoints
        exp_dir = OUTPUT_BASE / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for step, ckpt_data in checkpoints.items():
            ckpt_path = exp_dir / f"step_{step:06d}.pt"
            torch.save({
                'model_state_dict': ckpt_data['state_dict'],
                'step': step,
                'config': {**exp_config, **COMMON_CONFIG},
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            results.append({
                'experiment': exp_name,
                'model_type': model_type,
                'frame_size': frame_size,
                'fine_iterations': fine_iterations,
                'step': step,
                'train_loss': ckpt_data['loss'],
                'elapsed_time': ckpt_data['elapsed_time'],
            })

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save results summary
    results_path = OUTPUT_BASE / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
