#!/usr/bin/env python3
"""
Multi-size scaling law study.

Varies both model size (LLM + vision encoder) and training compute
to derive Chinchilla-style scaling laws.
"""

import os
import sys
import json
import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

STEP_COUNTS = [100, 300, 1000, 3000]

# Model configurations: (llm_model, dino_model, batch_size, name)
MODEL_CONFIGS = {
    # Phase 1: Small LLM + Small Vision (current baseline)
    "S-S": {
        "llm_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dino_model": "facebook/dinov2-small",
        "llm_dim": 576,
        "dino_dim": 384,
        "batch_size": 16,
        "total_params_M": 160,
    },
    # Phase 2: Medium LLM + Small Vision
    "M-S": {
        "llm_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "dino_model": "facebook/dinov2-small",
        "llm_dim": 960,
        "dino_dim": 384,
        "batch_size": 8,
        "total_params_M": 385,
    },
    # Phase 3: Small LLM + Base Vision
    "S-B": {
        "llm_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dino_model": "facebook/dinov2-base",
        "llm_dim": 576,
        "dino_dim": 768,
        "batch_size": 12,
        "total_params_M": 220,
    },
}

# Architecture types to test
ARCH_TYPES = ["foveated", "baseline"]

COMMON = {
    "learning_rate": 3e-5,
    "warmup_steps": 50,
    "num_frames": 8,
    "frame_size": 224,
}

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
SPLIT_FILE = PROJECT_ROOT / "configs" / "data_split.json"
OUTPUT_BASE = Path("/mnt/d/projects/fVLM/outputs/scaling_multisize")
RESEARCH_DIR = Path(__file__).parent.parent

MAX_VAL_SAMPLES = 1000

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================================
# Utilities
# ============================================================================

def clear_gpu():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory_gb():
    """Get current GPU memory usage in GB."""
    return torch.cuda.memory_allocated() / 1e9


# ============================================================================
# Dataset
# ============================================================================

class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir, shard_list, num_frames, frame_size):
        self.shard_dir = Path(shard_dir)
        self.shard_files = [self.shard_dir / s for s in shard_list if (self.shard_dir / s).exists()]
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                for sample in shard['samples']:
                    yield self._process(sample)
                del shard
            except Exception:
                continue

    def _process(self, data):
        frames = data['frames']
        latents = data['latents']
        caption = data['caption']

        T = frames.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            indices = list(range(T)) + [T-1] * (self.num_frames - T)
            indices = torch.tensor(indices)

        frames = frames[indices].float()
        latents = latents[indices].float()

        if frames.shape[-1] != self.frame_size:
            frames = F.interpolate(frames, size=(self.frame_size, self.frame_size), mode='bilinear', align_corners=False)

        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'latents': latents, 'caption': caption}


def collate_fn(batch):
    return {
        'frames': torch.stack([b['frames'] for b in batch]),
        'latents': torch.stack([b['latents'] for b in batch]),
        'captions': [b['caption'] for b in batch],
    }


# ============================================================================
# Model Creation
# ============================================================================

def create_model(arch_type, model_cfg, device):
    """Create model based on architecture type and config."""
    if arch_type == "foveated":
        from src.model.foveated_vlm import FoveatedVideoModel
        model = FoveatedVideoModel(
            dino_model=model_cfg['dino_model'],
            llm_model=model_cfg['llm_model'],
            dino_dim=model_cfg['dino_dim'],
            llm_dim=model_cfg['llm_dim'],
            deep_query=True,
            freeze_dino=False,
        )
    else:
        from src.model.baseline_vlm import BaselineVLM
        model = BaselineVLM(
            dino_model=model_cfg['dino_model'],
            llm_model=model_cfg['llm_model'],
            pixel_shuffle_scale=4,
        )

    return model.to(device)


# ============================================================================
# Training
# ============================================================================

def train_model(model, arch_type, tokenizer, dataloader, batch_size, device, max_steps, checkpoint_steps, exp_name):
    """Train model and save checkpoints."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON['learning_rate'])
    scaler = torch.amp.GradScaler('cuda')

    exp_dir = OUTPUT_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_saved = []
    step = 0
    model.train()
    start_time = time.time()
    data_iter = iter(dataloader)

    pbar = tqdm(total=max_steps, desc=f"Training {exp_name}")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        frames = batch['frames'].to(device)
        captions = batch['captions']

        tokens = tokenizer(captions, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        caption_ids = tokens['input_ids']
        caption_mask = tokens['attention_mask']

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if arch_type == "foveated":
                loss = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)
            else:
                caption_embeds = model.llm.model.embed_tokens(caption_ids)
                caption_targets = caption_ids[:, 1:].clone()
                caption_targets[caption_targets == tokenizer.pad_token_id] = -100
                loss, _ = model.forward(frames, caption_embeds, caption_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # Warmup
        if step <= COMMON['warmup_steps']:
            for pg in optimizer.param_groups:
                pg['lr'] = COMMON['learning_rate'] * (step / COMMON['warmup_steps'])

        # Save checkpoint
        if step in checkpoint_steps:
            ckpt_path = exp_dir / f"step_{step:06d}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'loss': loss.item(),
                'elapsed': time.time() - start_time,
            }, ckpt_path)
            checkpoints_saved.append(step)
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'saved': step})

        pbar.update(1)

    pbar.close()
    return checkpoints_saved


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_checkpoint(model, arch_type, tokenizer, samples, device):
    """Evaluate a single checkpoint."""
    model.eval()
    losses = []

    for s in tqdm(samples, desc="Evaluating", leave=False):
        frames = s['frames'].unsqueeze(0).to(device)
        caption = s['caption']

        tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
        caption_ids = tokens['input_ids'].to(device)
        caption_targets = caption_ids[:, 1:]

        B, T = 1, frames.shape[1]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if arch_type == "foveated":
                # Use forward_captioning for evaluation
                loss = model.forward_captioning(
                    frames, caption_ids,
                    torch.ones_like(caption_ids),
                    use_fine=True
                )
                losses.append(loss.item())
            else:
                # Baseline evaluation
                visual_features = model.encode_frames(frames)
                num_tok = visual_features.shape[2]
                visual_features = visual_features.reshape(B, T * num_tok, model.llm_dim)

                caption_embeds = model.llm.model.embed_tokens(caption_ids)
                visual_token = model.visual_token.expand(B, -1, -1)
                seq = torch.cat([visual_token, visual_features, caption_embeds], dim=1)

                outputs = model.llm.model(inputs_embeds=seq)
                logits = model.llm.lm_head(outputs.last_hidden_state)
                visual_len = 1 + T * num_tok
                caption_logits = logits[:, visual_len:-1, :]

                loss = F.cross_entropy(
                    caption_logits.reshape(-1, caption_logits.size(-1)),
                    caption_targets.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
                losses.append(loss.item())

    return np.array(losses)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='S-S', choices=list(MODEL_CONFIGS.keys()),
                        help='Model size configuration')
    parser.add_argument('--arch', type=str, default='both', choices=['foveated', 'baseline', 'both'],
                        help='Architecture type to train')
    parser.add_argument('--eval-only', action='store_true', help='Skip training, only evaluate')
    args = parser.parse_args()

    print("=" * 80)
    print(f"MULTI-SIZE SCALING LAW STUDY")
    print(f"Config: {args.config}, Arch: {args.arch}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    clear_gpu()
    print(f"GPU memory at start: {get_gpu_memory_gb():.2f} GB")

    # Load data split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_shards = split['train_shards']
    val_shards = split['val_shards']

    model_cfg = MODEL_CONFIGS[args.config]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    archs = [args.arch] if args.arch != 'both' else ARCH_TYPES
    all_results = []
    max_steps = max(STEP_COUNTS)

    for arch_type in archs:
        exp_name = f"{arch_type}_{args.config}"

        if not args.eval_only:
            print(f"\n{'='*60}")
            print(f"TRAINING: {exp_name}")
            print(f"{'='*60}")

            # Create dataset
            dataset = ShardedDataset(
                SHARD_DIR, train_shards,
                COMMON['num_frames'], COMMON['frame_size']
            )
            dataloader = DataLoader(
                dataset, batch_size=model_cfg['batch_size'],
                collate_fn=collate_fn, num_workers=2
            )

            # Create and train model
            model = create_model(arch_type, model_cfg, device)
            print(f"Model created. GPU memory: {get_gpu_memory_gb():.2f} GB")

            train_model(
                model, arch_type, tokenizer, dataloader,
                model_cfg['batch_size'], device, max_steps,
                set(STEP_COUNTS), exp_name
            )

            del model, dataloader, dataset
            clear_gpu()
            print(f"GPU memory after training: {get_gpu_memory_gb():.2f} GB")

        # Evaluation
        print(f"\n{'='*60}")
        print(f"EVALUATING: {exp_name}")
        print(f"{'='*60}")

        # Load val samples
        val_dataset = ShardedDataset(SHARD_DIR, val_shards, COMMON['num_frames'], COMMON['frame_size'])
        samples = []
        for s in val_dataset:
            samples.append(s)
            if len(samples) >= MAX_VAL_SAMPLES:
                break
        print(f"Loaded {len(samples)} val samples")

        # Create model for evaluation
        model = create_model(arch_type, model_cfg, device)

        exp_dir = OUTPUT_BASE / exp_name
        for ckpt_path in sorted(exp_dir.glob("step_*.pt")):
            step = int(ckpt_path.stem.split('_')[1])
            print(f"  Step {step}...", end=" ", flush=True)

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            losses = evaluate_checkpoint(model, arch_type, tokenizer, samples, device)

            mean_loss = losses.mean()
            std_loss = losses.std()
            se_loss = std_loss / np.sqrt(len(losses))
            ppl = np.exp(mean_loss)

            # Rough FLOPs estimate based on model size
            flops_per_sample = model_cfg['total_params_M'] * 1e6 * 6 * 1000  # Approximate
            total_flops = step * model_cfg['batch_size'] * flops_per_sample * 3

            print(f"loss={mean_loss:.4f}, ppl={ppl:.1f}")

            all_results.append({
                'experiment': exp_name,
                'arch_type': arch_type,
                'model_config': args.config,
                'llm_model': model_cfg['llm_model'],
                'dino_model': model_cfg['dino_model'],
                'total_params_M': model_cfg['total_params_M'],
                'batch_size': model_cfg['batch_size'],
                'step': step,
                'val_loss': float(mean_loss),
                'val_loss_std': float(std_loss),
                'val_loss_se': float(se_loss),
                'perplexity': float(ppl),
                'n_samples': len(losses),
                'total_training_flops': total_flops,
            })

        del model, samples
        clear_gpu()

    # Save results
    data_dir = RESEARCH_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"scaling_data_{args.config}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Experiment':<25} {'Step':>6} {'Loss':>8} {'PPL':>8}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: (x['experiment'], x['step'])):
        print(f"{r['experiment']:<25} {r['step']:>6} {r['val_loss']:>8.4f} {r['perplexity']:>8.1f}")

    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
