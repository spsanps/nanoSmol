#!/usr/bin/env python3
"""
Single-epoch SmolVLM2 caption fine-tuning on train split.

Fine-tunes SmolVLM2-256M-Video-Instruct on the same training data as the
foveated model for fair comparison. Caption-only loss (no reconstruction).

Output: /mnt/d/projects/fVLM/outputs/smolvlm_singleepoch/
"""

import sys
import time
import json
import random
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
from collections import deque

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "shard_dir": "/mnt/d/projects/fVLM/data/frames_latents_sharded",
    "split_file": str(PROJECT_ROOT / "configs" / "data_split.json"),
    "num_frames": 8,

    "model_name": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",

    "batch_size": 2,
    "grad_accum": 8,     # effective batch = 16 (same as foveated)
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "grad_clip": 1.0,

    "log_interval": 50,
    "save_interval": 2000,
    "output_dir": "/mnt/d/projects/fVLM/outputs/smolvlm_singleepoch",
    "max_checkpoints": 3,
}


# ============================================================================
# DATASET
# ============================================================================

class ShardedCaptionDataset(torch.utils.data.IterableDataset):
    """Sharded dataset that yields raw frames + captions for SmolVLM.

    Unlike the foveated dataset, this returns raw uint8 frames
    (SmolVLM's processor handles normalization).
    """

    def __init__(self, shard_dir: str, num_frames: int = 8,
                 shard_whitelist: list = None):
        self.shard_dir = Path(shard_dir)
        self.num_frames = num_frames

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

        # Convert to list of PIL images for SmolVLM processor
        pil_frames = []
        for t in range(frames.shape[0]):
            # [3, 256, 256] uint8 -> PIL Image
            frame_np = frames[t].permute(1, 2, 0).numpy()  # [256, 256, 3]
            pil_frames.append(Image.fromarray(frame_np))

        return {
            'pil_frames': pil_frames,
            'caption': caption,
        }


def collate_fn(batch):
    """Simple collation - processor handles tokenization per batch."""
    return {
        'pil_frames': [b['pil_frames'] for b in batch],
        'captions': [b['caption'] for b in batch],
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
    if step % 2000 == 0:
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
# BATCH PROCESSING
# ============================================================================

def prepare_caption_batch(processor, pil_frames_batch, captions, device):
    """Prepare a batch for caption training with proper label masking.

    Returns input_ids, attention_mask, pixel_values, and labels where
    only caption tokens (assistant response) contribute to loss.
    """
    B = len(captions)

    # Build chat messages for each sample
    all_input_ids = []
    all_attention_masks = []
    all_pixel_values = []
    all_labels = []

    for i in range(B):
        frames = pil_frames_batch[i]  # list of PIL images

        # Build chat template: user asks, assistant provides caption
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video:"},
                ] + [{"type": "image"} for _ in frames],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": captions[i]},
                ],
            },
        ]

        # Process with the SmolVLM processor
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(
            text=text,
            images=frames,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels: mask everything except the assistant's caption tokens
        # Find where the assistant response starts
        labels = input_ids.clone()

        # Tokenize just the caption to find its token IDs
        caption_tokens = processor.tokenizer(
            captions[i], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(0)

        # Mask all tokens before the caption with -100
        # The caption tokens appear at the end of the sequence
        caption_len = len(caption_tokens)
        if caption_len < len(labels):
            labels[:-caption_len] = -100

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

        if "pixel_values" in inputs:
            all_pixel_values.append(inputs["pixel_values"].squeeze(0))

    # Pad to max length in batch
    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_id = processor.tokenizer.pad_token_id or 0

    padded_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    padded_attention = torch.zeros(B, max_len, dtype=torch.long)
    padded_labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i in range(B):
        seq_len = all_input_ids[i].shape[0]
        padded_input_ids[i, :seq_len] = all_input_ids[i]
        padded_attention[i, :seq_len] = all_attention_masks[i]
        padded_labels[i, :seq_len] = all_labels[i]

    result = {
        "input_ids": padded_input_ids.to(device),
        "attention_mask": padded_attention.to(device),
        "labels": padded_labels.to(device),
    }

    if all_pixel_values:
        # Stack pixel values - they should all have the same shape
        result["pixel_values"] = torch.stack(all_pixel_values).to(device, dtype=torch.bfloat16)

    return result


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

    # Allow env var overrides for partial-epoch runs
    import os
    if os.environ.get("MAX_STEPS"):
        max_steps = int(os.environ["MAX_STEPS"])
    if os.environ.get("OUTPUT_DIR"):
        CONFIG["output_dir"] = os.environ["OUTPUT_DIR"]

    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SmolVLM2 - SINGLE EPOCH CAPTION FINE-TUNING (with data split)")
    print("=" * 80)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Train shards: {len(train_shards)} (~{approx_samples} samples)")
    print(f"Val shards: {split['val_count']} (EXCLUDED)")
    print(f"Batch: {CONFIG['batch_size']} x {CONFIG['grad_accum']} = {effective_bs}")
    print(f"Max steps: {max_steps} (single epoch)")
    print(f"Output: {output_dir}")
    print("=" * 80)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**CONFIG, 'max_steps': max_steps, 'train_shards_count': len(train_shards)}, f, indent=2)

    # Load model
    print("\n[1/4] Loading SmolVLM2...")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        CONFIG["model_name"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
    )
    # Disable image splitting: our 256x256 frames don't benefit from tiling,
    # and 8 frames with splitting exceeds the 8192 token limit
    processor.image_processor.do_image_splitting = False

    # Enable gradient checkpointing
    if hasattr(model.model, 'text_model'):
        model.model.text_model.gradient_checkpointing_enable()

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

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
    dataset = ShardedCaptionDataset(
        shard_dir=CONFIG["shard_dir"],
        num_frames=CONFIG["num_frames"],
        shard_whitelist=train_shards,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        pin_memory=False,  # PIL images can't be pinned
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Training
    print(f"[4/4] Starting training (max {max_steps} optimizer steps)...")
    print(f"{'='*80}\n")

    step = start_step
    epoch = start_epoch
    micro_step = 0
    accum_loss = 0.0
    losses = deque(maxlen=100)
    step_times = deque(maxlen=50)

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

                # Prepare batch with label masking
                try:
                    prepared = prepare_caption_batch(
                        processor, batch['pil_frames'], batch['captions'], device
                    )
                except Exception as e:
                    print(f"  Skipping batch: {e}")
                    continue

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=prepared["input_ids"],
                        attention_mask=prepared["attention_mask"],
                        pixel_values=prepared.get("pixel_values"),
                        labels=prepared["labels"],
                    )
                    loss = outputs.loss / CONFIG["grad_accum"]

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
                            f"Step {step:5d}/{max_steps} | "
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
            'experiment': 'smolvlm_singleepoch',
            'model': CONFIG['model_name'],
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
