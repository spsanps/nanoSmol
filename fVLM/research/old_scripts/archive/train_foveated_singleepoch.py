#!/usr/bin/env python3
"""
Single-epoch foveated VLM training on train split only.

Uses the same architecture and forward pass as train_joint_multifine_precomputed.py
but with data split enforcement and single-epoch stopping.

Output: /mnt/d/projects/fVLM/outputs/foveated_singleepoch/
"""

import sys
import os
import time
import random
import json
import shutil
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import AutoTokenizer
import numpy as np
from collections import deque

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "shard_dir": "/mnt/d/projects/fVLM/data/frames_latents_sharded",
    "split_file": str(PROJECT_ROOT / "configs" / "data_split.json"),
    "num_frames": 8,

    "batch_size": 16,
    "grad_accum": 1,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "grad_clip": 1.0,

    "lambda_recon": 0.5,
    "lambda_coarse": 0.0,
    "fine_iterations": 2,

    "deep_query": True,
    "freeze_dino": False,

    "log_interval": 50,
    "save_interval": 2000,
    "output_dir": "/mnt/d/projects/fVLM/outputs/foveated_singleepoch",
    "max_checkpoints": 3,
}


# ============================================================================
# DATASET (with split filtering)
# ============================================================================

class ShardedVideoDataset(torch.utils.data.IterableDataset):
    """Sharded dataset that respects train/val split."""

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
        frames = data['frames']
        latents = data['latents']
        caption = data['caption']

        T_orig = frames.shape[0]
        if T_orig > self.num_frames:
            indices = np.linspace(0, T_orig - 1, self.num_frames, dtype=int)
            frames = frames[indices]
            latents = latents[indices]

        frames = frames.float() / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {
            'frames': frames,
            'latents': latents.float(),
            'caption': caption,
        }


class CollateFn:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        frames = torch.stack([b['frames'] for b in batch])
        latents = torch.stack([b['latents'] for b in batch])
        captions = [b['caption'] for b in batch]

        tokens = self.tokenizer(
            captions, padding=True, truncation=True,
            max_length=64, return_tensors='pt'
        )
        return {
            'frames': frames,
            'latents': latents,
            'caption_ids': tokens['input_ids'],
            'caption_mask': tokens['attention_mask'],
        }


# ============================================================================
# FORWARD PASS (imported logic from train_joint_multifine_precomputed.py)
# ============================================================================

def forward_multifine_joint(model, frames, caption_ids, caption_mask, vae_latents,
                            tokenizer, num_iterations=2, compute_coarse_loss=True):
    B, T = frames.shape[:2]
    device = frames.device

    frames_flat = frames.reshape(B * T, 3, frames.shape[-2], frames.shape[-1])
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = []
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        K_all_layers, V_all_layers, layers_ref = [], [], []
        for layer_idx in range(num_layers):
            lc = cache_flat['kv_cache'][layer_idx]
            K_all_layers.append(lc['K'].reshape(B, T, N, D))
            V_all_layers.append(lc['V'].reshape(B, T, N, D))
            layers_ref.append(lc['layer'])

        for t in range(T):
            frame_kv_cache = []
            for layer_idx in range(num_layers):
                frame_kv_cache.append({
                    'K': K_all_layers[layer_idx][:, t],
                    'V': V_all_layers[layer_idx][:, t],
                    'layer': layers_ref[layer_idx],
                })
            all_caches.append({
                'patch_features': patch_features[:, t],
                'kv_cache': frame_kv_cache,
            })
    else:
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]
    text_embeds = model.get_empty_text_embeds(B)
    N_text = text_embeds.shape[1]

    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)

    coarse_token = model.coarse_token.expand(B, -1, -1)
    fine_token = model.fine_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)

    q_static = model.q_static.expand(B, -1)
    z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    loss_cap_coarse = torch.tensor(0.0, device=device)
    loss_rec_coarse = torch.tensor(0.0, device=device)

    if compute_coarse_loss:
        with torch.no_grad():
            seq_cap_coarse = torch.cat([coarse_token, z_coarse_llm, caption_embeds], dim=1)
            outputs_cap_coarse = model.llm.model(inputs_embeds=seq_cap_coarse)
            logits_cap_coarse = model.llm.lm_head(outputs_cap_coarse.last_hidden_state)
            caption_logits_coarse = logits_cap_coarse[:, 1+T:-1, :]
            loss_cap_coarse = F.cross_entropy(
                caption_logits_coarse.reshape(-1, caption_logits_coarse.size(-1)),
                caption_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )

            seq_rec_coarse = torch.cat([text_embeds, coarse_token, z_coarse_llm], dim=1)
            outputs_rec_coarse = model.llm.model(inputs_embeds=seq_rec_coarse)
            h_coarse_for_pred = outputs_rec_coarse.last_hidden_state[:, N_text:N_text + T]
            pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)
            loss_rec_coarse = F.mse_loss(pred_coarse, vae_latents)

    seq_query0 = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_query0 = model.llm.model(inputs_embeds=seq_query0)
    queries = model.llm_to_query(outputs_query0.last_hidden_state[:, 2:])

    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    current_queries = torch.cat([q_init, queries[:, :-1]], dim=1)

    cap_losses, rec_losses = [], []

    for iteration in range(num_iterations):
        z_fine_list = [model.encoder.query_attend(current_queries[:, t], all_caches[t]) for t in range(T)]
        z_fine = torch.stack(z_fine_list, dim=1)
        z_fine_llm = model.dino_to_llm(z_fine)
        z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

        seq_cap_fine = torch.cat([fine_token, z_fine_llm, caption_embeds], dim=1)
        outputs_cap_fine = model.llm.model(inputs_embeds=seq_cap_fine)
        logits_cap_fine = model.llm.lm_head(outputs_cap_fine.last_hidden_state)
        caption_logits_fine = logits_cap_fine[:, 1+T:-1, :]
        loss_cap_iter = F.cross_entropy(
            caption_logits_fine.reshape(-1, caption_logits_fine.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        cap_losses.append(loss_cap_iter)

        seq_rec_fine = torch.cat([text_embeds, fine_token, z_fine_llm], dim=1)
        outputs_rec_fine = model.llm.model(inputs_embeds=seq_rec_fine)
        h_fine_for_pred = outputs_rec_fine.last_hidden_state[:, N_text:N_text + T]
        pred_fine = model.pred_head(h_fine_for_pred, prev_latents)
        loss_rec_iter = F.mse_loss(pred_fine, vae_latents)
        rec_losses.append(loss_rec_iter)

        if iteration < num_iterations - 1:
            seq_query = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
            outputs_query = model.llm.model(inputs_embeds=seq_query)
            next_queries = model.llm_to_query(outputs_query.last_hidden_state[:, 2:])
            current_queries = torch.cat([q_init, next_queries[:, :-1]], dim=1)

    return loss_cap_coarse, cap_losses, loss_rec_coarse, rec_losses


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(model, optimizer, scaler, scheduler, step, epoch, metrics,
                    save_dir, max_keep=3):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
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


def load_checkpoint(model, optimizer, scaler, scheduler, save_dir, device):
    save_dir = Path(save_dir)
    ckpt_path = save_dir / 'latest.pt'
    if not ckpt_path.exists():
        return 0, 0

    print(f"  Resuming from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
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
    max_steps = approx_samples // CONFIG["batch_size"]

    # Allow env var overrides for partial-epoch runs
    if os.environ.get("MAX_STEPS"):
        max_steps = int(os.environ["MAX_STEPS"])
    if os.environ.get("OUTPUT_DIR"):
        CONFIG["output_dir"] = os.environ["OUTPUT_DIR"]

    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_fine_iters = CONFIG["fine_iterations"]

    print("=" * 80)
    print("FOVEATED VLM - SINGLE EPOCH TRAINING (with data split)")
    print("=" * 80)
    print(f"Train shards: {len(train_shards)} (~{approx_samples} samples)")
    print(f"Val shards: {split['val_count']} (EXCLUDED)")
    print(f"Max steps: {max_steps} (single epoch at BS={CONFIG['batch_size']})")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Save config for reproducibility
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**CONFIG, 'max_steps': max_steps, 'train_shards_count': len(train_shards)}, f, indent=2)

    # Model
    print("\n[1/4] Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=CONFIG["deep_query"],
        freeze_dino=CONFIG["freeze_dino"],
    ).to(device)

    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scaler = GradScaler('cuda')

    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / max(CONFIG["warmup_steps"], 1)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    print("[2/4] Checking for checkpoint...")
    start_step, start_epoch = load_checkpoint(
        model, optimizer, scaler, scheduler, ckpt_dir, device
    )

    # Dataset (TRAIN SPLIT ONLY)
    print("[3/4] Loading dataset...")
    dataset = ShardedVideoDataset(
        shard_dir=CONFIG["shard_dir"],
        num_frames=CONFIG["num_frames"],
        shard_whitelist=train_shards,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        collate_fn=CollateFn(),
    )

    # Training
    print(f"[4/4] Starting training (max {max_steps} steps)...")
    print(f"{'='*80}\n")

    step = start_step
    epoch = start_epoch
    accum_count = 0
    accum_coarse_count = 0
    accum_cap_coarse = 0.0
    accum_cap_iters = [0.0] * num_fine_iters
    accum_rec_coarse = 0.0
    accum_rec_iters = [0.0] * num_fine_iters

    cap_ratios = deque(maxlen=100)
    rec_ratios = deque(maxlen=100)
    step_times = deque(maxlen=50)

    pbar = tqdm(total=max_steps, initial=start_step, desc="Training")
    optimizer.zero_grad()
    last_step_time = time.time()

    try:
        while step < max_steps:
            epoch += 1
            if epoch > 1:
                # Single epoch: stop after first pass
                print(f"\n[SINGLE EPOCH] Epoch 1 complete at step {step}. Stopping.")
                break

            for batch in dataloader:
                if step >= max_steps:
                    break

                frames = batch['frames'].to(device, non_blocking=True)
                latents = batch['latents'].to(device, non_blocking=True)
                caption_ids = batch['caption_ids'].to(device, non_blocking=True)
                caption_mask = batch['caption_mask'].to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    do_coarse = (step % 10 == 0) or (step < start_step + 50)
                    cap_coarse, cap_iter_losses, rec_coarse, rec_iter_losses = \
                        forward_multifine_joint(
                            model, frames, caption_ids, caption_mask,
                            latents, tokenizer, num_fine_iters,
                            compute_coarse_loss=do_coarse
                        )

                    loss_cap = cap_iter_losses[-1]
                    loss_rec = rec_iter_losses[-1]
                    loss = loss_cap + CONFIG["lambda_recon"] * loss_rec

                scaler.scale(loss).backward()

                for i, l in enumerate(cap_iter_losses):
                    accum_cap_iters[i] += l.item()
                for i, l in enumerate(rec_iter_losses):
                    accum_rec_iters[i] += l.item()
                if do_coarse:
                    accum_cap_coarse += cap_coarse.item()
                    accum_rec_coarse += rec_coarse.item()
                    accum_coarse_count += 1
                accum_count += 1

                # Step (no grad accum for this config)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                avg_cap_iters = [x / accum_count for x in accum_cap_iters]
                avg_rec_iters = [x / accum_count for x in accum_rec_iters]

                if accum_coarse_count > 0:
                    avg_cap_coarse = accum_cap_coarse / accum_coarse_count
                    avg_rec_coarse = accum_rec_coarse / accum_coarse_count
                    cap_ratio = avg_cap_coarse / (avg_cap_iters[-1] + 1e-8)
                    rec_ratio = avg_rec_coarse / (avg_rec_iters[-1] + 1e-8)
                    cap_ratios.append(cap_ratio)
                    rec_ratios.append(rec_ratio)

                now = time.time()
                step_times.append(now - last_step_time)
                last_step_time = now

                step += 1
                pbar.update(1)

                if step % CONFIG["log_interval"] == 0:
                    avg_step_time = np.mean(list(step_times))
                    remaining = max_steps - step
                    eta = datetime.now() + timedelta(seconds=remaining * avg_step_time)
                    avg_cap_r = np.mean(list(cap_ratios)) if cap_ratios else 1.0
                    avg_rec_r = np.mean(list(rec_ratios)) if rec_ratios else 1.0
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9

                    cap_prog = " -> ".join([f"{l:.3f}" for l in avg_cap_iters])
                    log_msg = (
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Step {step:5d}/{max_steps} ep{epoch} | "
                        f"cap: [{cap_prog}] r={avg_cap_r:.3f} | "
                        f"rec: {avg_rec_iters[-1]:.3f} r={avg_rec_r:.3f} | "
                        f"{avg_step_time:.1f}s/step | {mem_gb:.1f}GB | "
                        f"ETA: {eta.strftime('%H:%M')}"
                    )
                    tqdm.write(log_msg)
                    with open(output_dir / 'train.log', 'a') as lf:
                        lf.write(log_msg + '\n')

                if step % CONFIG["save_interval"] == 0:
                    metrics = {
                        'cap_ratio': float(np.mean(list(cap_ratios))) if cap_ratios else 1.0,
                        'rec_ratio': float(np.mean(list(rec_ratios))) if rec_ratios else 1.0,
                        'cap_loss_fine': avg_cap_iters[-1],
                        'rec_loss_fine': avg_rec_iters[-1],
                    }
                    save_checkpoint(model, optimizer, scaler, scheduler,
                                    step, epoch, metrics, ckpt_dir,
                                    max_keep=CONFIG["max_checkpoints"])

                # Reset accumulators
                accum_cap_coarse = 0.0
                accum_cap_iters = [0.0] * num_fine_iters
                accum_rec_coarse = 0.0
                accum_rec_iters = [0.0] * num_fine_iters
                accum_count = 0
                accum_coarse_count = 0

    except (KeyboardInterrupt, StopIteration):
        print("\n[STOPPED]")

    finally:
        pbar.close()
        elapsed = (time.time() - start_time) / 3600
        final_cap_ratio = float(np.mean(list(cap_ratios))) if cap_ratios else 1.0
        final_rec_ratio = float(np.mean(list(rec_ratios))) if rec_ratios else 1.0

        metrics = {
            'cap_ratio': final_cap_ratio,
            'rec_ratio': final_rec_ratio,
            'elapsed_hours': elapsed,
            'total_steps': step,
        }
        save_checkpoint(model, optimizer, scaler, scheduler,
                        step, epoch, metrics, ckpt_dir,
                        max_keep=CONFIG["max_checkpoints"])

        summary = {
            'experiment': 'foveated_singleepoch',
            'elapsed_hours': elapsed,
            'total_steps': step,
            'epochs': epoch,
            'final_cap_ratio': final_cap_ratio,
            'final_rec_ratio': final_rec_ratio,
            'train_shards': len(train_shards),
            'config': CONFIG,
        }
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"COMPLETE: {step} steps in {elapsed:.2f}h")
        print(f"Cap ratio: {final_cap_ratio:.4f}, Rec ratio: {final_rec_ratio:.4f}")
        print(f"Checkpoint: {ckpt_dir / 'latest.pt'}")
        print(f"{'='*80}")


if __name__ == "__main__":
    run_training()
