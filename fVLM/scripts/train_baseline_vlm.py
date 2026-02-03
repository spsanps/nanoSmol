#!/usr/bin/env python3
"""
Baseline VLM training: DINOv2 + PixelShuffle + SmolLM2

Uses 16 tokens per frame (vs foveated's 1 token) for fair comparison.
Same training recipe as foveated: end-to-end, everything trainable.

Output: /mnt/d/projects/fVLM/outputs/baseline_vlm/
"""

import os
import sys
import time
import random
import json
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import AutoTokenizer
import numpy as np
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.model.baseline_vlm import BaselineVLM

# ImageNet normalization (same as foveated)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "shard_dir": "/mnt/d/projects/fVLM/data/frames_latents_sharded",
    "split_file": str(PROJECT_ROOT / "configs" / "data_split.json"),
    "num_frames": 8,
    "frame_size": 224,  # Resize to 224x224 for clean 16x16 patch grid

    "batch_size": 16,  # Same as foveated
    "grad_accum": 1,   # No accumulation needed, fits in ~11 GB
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "grad_clip": 1.0,

    "pixel_shuffle_scale": 4,  # 16x16 -> 4x4 = 16 tokens/frame
    "freeze_dino": False,
    "freeze_llm": False,

    "log_interval": 50,
    "save_interval": 2000,
    "output_dir": "/mnt/d/projects/fVLM/outputs/baseline_vlm",
    "max_checkpoints": 3,
}


# ============================================================================
# DATASET
# ============================================================================

class ShardedVideoDataset(torch.utils.data.IterableDataset):
    """Sharded dataset with frame resizing for baseline model."""

    def __init__(self, shard_dir: str, num_frames: int = 8, frame_size: int = 224,
                 shard_whitelist: list = None):
        self.shard_dir = Path(shard_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size

        all_shards = sorted(self.shard_dir.glob("shard_*.pt"))
        if shard_whitelist is not None:
            whitelist_set = set(shard_whitelist)
            self.shard_files = [s for s in all_shards if s.name in whitelist_set]
            print(f"  Shard whitelist: {len(self.shard_files)}/{len(all_shards)} shards")
        else:
            self.shard_files = all_shards

        self.num_shards = len(self.shard_files)
        self.samples_per_shard = 200
        print(f"  Total shards: {self.num_shards}")
        print(f"  Approx samples: {self.num_shards * self.samples_per_shard}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard_indices = list(range(self.num_shards))
        random.shuffle(shard_indices)

        if worker_info is not None:
            per_worker = len(shard_indices) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(shard_indices)
            shard_indices = shard_indices[start:end]

        for shard_idx in shard_indices:
            try:
                shard = torch.load(
                    self.shard_files[shard_idx],
                    map_location="cpu", weights_only=False
                )
                samples = shard['samples']
                del shard
            except Exception as e:
                print(f"  Skipping corrupt shard {self.shard_files[shard_idx].name}: {e}")
                continue

            indices = list(range(len(samples)))
            random.shuffle(indices)
            for i in indices:
                yield self._process(samples[i])
            del samples

    def _process(self, data):
        frames = data['frames']  # [24, 3, 256, 256] uint8
        caption = data['caption']

        T_orig = frames.shape[0]
        if T_orig > self.num_frames:
            indices = np.linspace(0, T_orig - 1, self.num_frames, dtype=int)
            frames = frames[indices]

        # Resize to 224x224 for clean DINOv2 patch grid
        # frames: [T, 3, 256, 256] -> [T, 3, 224, 224]
        frames_float = frames.float()
        frames_resized = F.interpolate(
            frames_float,
            size=(self.frame_size, self.frame_size),
            mode='bilinear',
            align_corners=False
        )

        # Normalize with ImageNet stats
        frames_norm = frames_resized / 255.0
        frames_norm = (frames_norm - IMAGENET_MEAN) / IMAGENET_STD

        return {
            'frames': frames_norm,  # [T, 3, 224, 224] normalized
            'caption': caption,
        }


class CollateFn:
    def __init__(self, tokenizer, max_caption_len=64):
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len

    def __call__(self, batch):
        frames = torch.stack([b['frames'] for b in batch])  # [B, T, 3, 224, 224]
        captions = [b['caption'] for b in batch]

        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_caption_len,
            return_tensors='pt'
        )

        return {
            'frames': frames,
            'caption_ids': tokens['input_ids'],
            'caption_mask': tokens['attention_mask'],
        }


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, step, epoch, metrics,
                    save_dir, max_keep=3):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_dir / 'latest.pt')
    if step % 2000 == 0 and step > 0:
        torch.save(checkpoint, save_dir / f'step_{step:06d}.pt')

    all_ckpts = sorted(save_dir.glob('step_*.pt'), key=lambda p: p.stat().st_mtime)
    if len(all_ckpts) > max_keep:
        for old in all_ckpts[:-max_keep]:
            old.unlink()

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved: step {step}", flush=True)


def load_checkpoint(model, optimizer, scheduler, save_dir, device):
    save_dir = Path(save_dir)
    ckpt_path = save_dir / 'latest.pt'
    if not ckpt_path.exists():
        return 0, 0

    print(f"  Resuming from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    print(f"  Resumed at step {step}, epoch {epoch}")
    return step, epoch


# ============================================================================
# MAIN
# ============================================================================

def run_training():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    start_time = time.time()
    device = torch.device("cuda")

    # Load data split
    with open(CONFIG["split_file"]) as f:
        split = json.load(f)
    train_shards = split["train_shards"]
    approx_samples = len(train_shards) * 200
    effective_bs = CONFIG["batch_size"] * CONFIG["grad_accum"]
    max_steps = approx_samples // effective_bs

    # Allow env var overrides
    if os.environ.get("MAX_STEPS"):
        max_steps = int(os.environ["MAX_STEPS"])
    if os.environ.get("OUTPUT_DIR"):
        CONFIG["output_dir"] = os.environ["OUTPUT_DIR"]

    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE VLM - DINOv2 + PixelShuffle + SmolLM2")
    print("=" * 80)
    print(f"Train shards: {len(train_shards)} (~{approx_samples} samples)")
    print(f"Val shards: {split['val_count']} (EXCLUDED)")
    print(f"Batch: {CONFIG['batch_size']} x {CONFIG['grad_accum']} = {effective_bs}")
    print(f"Max steps: {max_steps}")
    print(f"Tokens per frame: 16 (vs foveated's 1)")
    print(f"Frame size: {CONFIG['frame_size']}x{CONFIG['frame_size']}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**CONFIG, 'max_steps': max_steps, 'train_shards_count': len(train_shards)}, f, indent=2)

    # Model
    print("\n[1/4] Loading model...")
    model = BaselineVLM(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        pixel_shuffle_scale=CONFIG["pixel_shuffle_scale"],
        freeze_dino=CONFIG["freeze_dino"],
        freeze_llm=CONFIG["freeze_llm"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / max(CONFIG["warmup_steps"], 1)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    print("[2/4] Checking for checkpoint...")
    start_step, start_epoch = load_checkpoint(
        model, optimizer, scheduler, ckpt_dir, device
    )

    # Dataset
    print("[3/4] Loading dataset...")
    dataset = ShardedVideoDataset(
        shard_dir=CONFIG["shard_dir"],
        num_frames=CONFIG["num_frames"],
        frame_size=CONFIG["frame_size"],
        shard_whitelist=train_shards,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=CollateFn(tokenizer),
    )

    # Training
    print(f"[4/4] Starting training (max {max_steps} steps)...")
    print(f"{'='*80}\n")

    step = start_step
    epoch = start_epoch
    micro_step = 0
    accum_loss = 0.0
    losses = deque(maxlen=100)
    step_times = deque(maxlen=50)

    model.train()
    pbar = tqdm(total=max_steps, initial=start_step, desc="Training")
    optimizer.zero_grad()
    last_step_time = time.time()

    try:
        while step < max_steps:
            epoch += 1
            if epoch > 1:
                print(f"\n[SINGLE EPOCH] Epoch 1 complete at step {step}. Stopping.")
                break

            for batch in dataloader:
                if step >= max_steps:
                    break

                frames = batch['frames'].to(device)  # [B, T, 3, 224, 224]
                caption_ids = batch['caption_ids'].to(device)
                caption_mask = batch['caption_mask'].to(device)

                # Get caption embeddings
                caption_embeds = model.llm.model.embed_tokens(caption_ids)

                # Targets: shifted caption ids
                caption_targets = caption_ids[:, 1:].clone()
                caption_targets[caption_mask[:, 1:] == 0] = -100

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss, _ = model(frames, caption_embeds, caption_targets)
                    loss = loss / CONFIG["grad_accum"]

                loss.backward()
                accum_loss += loss.item()
                micro_step += 1

                if micro_step % CONFIG["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    batch_loss = accum_loss
                    losses.append(batch_loss)
                    accum_loss = 0.0

                    now = time.time()
                    step_times.append(now - last_step_time)
                    last_step_time = now

                    step += 1
                    pbar.update(1)

                    if step % CONFIG["log_interval"] == 0:
                        avg_loss = np.mean(list(losses))
                        avg_time = np.mean(list(step_times))
                        remaining = max_steps - step
                        eta = datetime.now() + timedelta(seconds=remaining * avg_time)
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9

                        log_msg = (
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Step {step:5d}/{max_steps} ep{epoch} | "
                            f"loss: {avg_loss:.4f} ppl: {np.exp(avg_loss):.1f} | "
                            f"{avg_time:.1f}s/step | {mem_gb:.1f}GB | "
                            f"ETA: {eta.strftime('%H:%M')}"
                        )
                        tqdm.write(log_msg)
                        with open(output_dir / 'train.log', 'a') as lf:
                            lf.write(log_msg + '\n')

                    if step % CONFIG["save_interval"] == 0:
                        metrics = {
                            'loss': float(np.mean(list(losses))),
                            'ppl': float(np.exp(np.mean(list(losses)))),
                        }
                        save_checkpoint(model, optimizer, scheduler,
                                        step, epoch, metrics, ckpt_dir,
                                        max_keep=CONFIG["max_checkpoints"])

    except (KeyboardInterrupt, StopIteration):
        print("\n[STOPPED]")

    finally:
        pbar.close()
        elapsed = (time.time() - start_time) / 3600
        avg_loss = float(np.mean(list(losses))) if losses else 0.0

        metrics = {
            'loss': avg_loss,
            'ppl': float(np.exp(avg_loss)) if avg_loss > 0 else 0.0,
            'elapsed_hours': elapsed,
            'total_steps': step,
        }
        save_checkpoint(model, optimizer, scheduler,
                        step, epoch, metrics, ckpt_dir,
                        max_keep=CONFIG["max_checkpoints"])

        summary = {
            'experiment': 'baseline_vlm',
            'tokens_per_frame': 16,
            'elapsed_hours': elapsed,
            'total_steps': step,
            'epochs': epoch,
            'final_loss': avg_loss,
            'final_ppl': float(np.exp(avg_loss)) if avg_loss > 0 else 0.0,
            'train_shards': len(train_shards),
            'config': CONFIG,
        }
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"COMPLETE: {step} steps in {elapsed:.2f}h")
        print(f"Final loss: {avg_loss:.4f}, PPL: {np.exp(avg_loss):.1f}")
        print(f"Checkpoint: {ckpt_dir / 'latest.pt'}")
        print(f"{'='*80}")


if __name__ == "__main__":
    run_training()
