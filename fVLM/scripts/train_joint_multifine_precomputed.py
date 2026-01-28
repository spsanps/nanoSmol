#!/usr/bin/env python3
"""
Joint Reconstruction + Captioning with Multi-Fine Iterations.
Uses PRECOMPUTED frames + VAE latents (no downloading, no VAE during training).

Key differences from train_joint_multifine_8h.py:
  - Loads precomputed frames + latents from disk (fast!)
  - freeze_dino=False → trainable DINO (the whole point of precomputing frames)
  - deep_query=True → queries propagate through all 12 DINO layers
  - Fully resumable with optimizer/scaler state
  - wandb logging
  - Space-aware checkpoint management

Architecture per batch:
  frames → DINO (trainable) → KV cache
  coarse (q_static) → z° → LLM → queries₁
  fine₁ (queries₁)  → z₁ → LLM → queries₂
  fine₂ (queries₂)  → z₂ (final)

Loss = loss_caption_fine₂ + λ * loss_recon_fine₂
"""

import sys
import os
import time
import random
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import AutoTokenizer
import numpy as np
import json
import shutil
from collections import deque

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    "data_dir": "/mnt/d/projects/fVLM/data/frames_latents_100k/features",
    "num_frames": 8,  # Subsample from 24 precomputed frames (8 = sweet spot)

    # Training (BS=16 = sweet spot: 17.2GB VRAM, 1.07s/step, no grad accum)
    "batch_size": 16,
    "grad_accum": 1,   # No accumulation - full batch fits in VRAM
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_steps": 100000,
    "max_hours": 0,  # 0 = unlimited (run until stopped)
    "grad_clip": 1.0,

    # Loss weights
    "lambda_recon": 0.5,
    "lambda_coarse": 0.0,  # Auxiliary coarse loss (0 = train on final fine only)

    # Multi-fine iterations
    "fine_iterations": 2,  # coarse -> fine1 -> fine2

    # Model
    "deep_query": True,
    "freeze_dino": False,  # Trainable DINO! (why we precomputed frames)

    # Logging
    "log_interval": 50,
    "save_interval": 2000,

    # Checkpointing
    "output_dir": "outputs/joint_multifine_precomputed",
    "resume_checkpoint": None,  # Set to path or "latest" to auto-detect
    "max_checkpoints": 5,  # Keep only N most recent + milestones
}


# ============================================================================
# DATASET
# ============================================================================

class PrecomputedVideoDataset(Dataset):
    """Loads precomputed frames + VAE latents from disk."""

    def __init__(self, data_dir: str, num_frames: int = 16):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames

        # List all .pt files
        self.files = sorted(self.data_dir.glob("*.pt"))
        print(f"Found {len(self.files)} precomputed videos")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        frames = data['frames']   # [24, 3, 256, 256] uint8
        latents = data['latents'] # [24, 4, 32, 32] bfloat16
        caption = data['caption']

        T_orig = frames.shape[0]

        # Subsample frames if needed
        if T_orig > self.num_frames:
            indices = np.linspace(0, T_orig - 1, self.num_frames, dtype=int)
            frames = frames[indices]
            latents = latents[indices]

        # Normalize frames: uint8 -> float32 -> ImageNet norm
        frames = frames.float() / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {
            'frames': frames,           # [T, 3, 256, 256] float32
            'latents': latents.float(),  # [T, 4, 32, 32] float32
            'caption': caption,
        }


# ============================================================================
# FORWARD PASS (Multi-Fine Joint)
# ============================================================================

def forward_multifine_joint(model, frames, caption_ids, caption_mask, vae_latents,
                            tokenizer, num_iterations=2, compute_coarse_loss=True):
    """
    Multi-fine iteration forward pass for BOTH captioning and reconstruction.

    Architecture:
      coarse (q_static) -> z_coarse -> LLM -> queries_1
      fine_1 (queries_1) -> z_fine_1 -> LLM -> queries_2
      fine_2 (queries_2) -> z_fine_2 (final)

    Args:
        compute_coarse_loss: If False, skip coarse loss (saves 2 LLM calls).
            Set to True periodically for ratio tracking.
    """
    B, T = frames.shape[:2]
    device = frames.device

    # Encode all frames with DINO (trainable!)
    frames_flat = frames.reshape(B * T, 3, frames.shape[-2], frames.shape[-1])
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Build per-frame caches for query attention
    all_caches = []
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        # Reshape once, then index per frame
        K_all_layers = []
        V_all_layers = []
        layers_ref = []
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

    # Prepare text embeddings
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]
    text_embeds = model.get_empty_text_embeds(B)
    N_text = text_embeds.shape[1]

    # prev_latents for reconstruction conditioning
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)

    # Mode tokens
    coarse_token = model.coarse_token.expand(B, -1, -1)
    fine_token = model.fine_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)

    # === Pass 0: Coarse features + query generation ===
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # Coarse losses (only when requested - saves 2 LLM forward passes)
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

    # Generate first queries from coarse
    seq_query0 = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_query0 = model.llm.model(inputs_embeds=seq_query0)
    queries = model.llm_to_query(outputs_query0.last_hidden_state[:, 2:])

    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    current_queries = torch.cat([q_init, queries[:, :-1]], dim=1)

    # === Fine iterations ===
    cap_losses = []
    rec_losses = []

    for iteration in range(num_iterations):
        z_fine_list = [model.encoder.query_attend(current_queries[:, t], all_caches[t]) for t in range(T)]
        z_fine = torch.stack(z_fine_list, dim=1)
        z_fine_llm = model.dino_to_llm(z_fine)
        z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

        # Caption loss
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

        # Reconstruction loss
        seq_rec_fine = torch.cat([text_embeds, fine_token, z_fine_llm], dim=1)
        outputs_rec_fine = model.llm.model(inputs_embeds=seq_rec_fine)
        h_fine_for_pred = outputs_rec_fine.last_hidden_state[:, N_text:N_text + T]
        pred_fine = model.pred_head(h_fine_for_pred, prev_latents)
        loss_rec_iter = F.mse_loss(pred_fine, vae_latents)
        rec_losses.append(loss_rec_iter)

        # Generate queries for next iteration
        if iteration < num_iterations - 1:
            seq_query = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
            outputs_query = model.llm.model(inputs_embeds=seq_query)
            next_queries = model.llm_to_query(outputs_query.last_hidden_state[:, 2:])
            current_queries = torch.cat([q_init, next_queries[:, :-1]], dim=1)

    return loss_cap_coarse, cap_losses, loss_rec_coarse, rec_losses


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(model, optimizer, scaler, scheduler, step, epoch, metrics,
                    save_dir, max_keep=5):
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

    # Always save latest
    torch.save(checkpoint, save_dir / 'latest.pt')

    # Save milestone checkpoints
    if step % 10000 == 0 or step == CONFIG['save_interval']:
        torch.save(checkpoint, save_dir / f'step_{step:06d}.pt')

    # Clean old checkpoints (keep milestones + latest N)
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
    metrics = ckpt.get('metrics', {})
    print(f"  Resumed at step {step}, epoch {epoch}")
    if metrics:
        print(f"  Last metrics: cap_ratio={metrics.get('cap_ratio', '?'):.4f}, "
              f"rec_ratio={metrics.get('rec_ratio', '?'):.4f}")
    return step, epoch


# ============================================================================
# MAIN TRAINING
# ============================================================================

def run_training():
    # ---- Performance tuning ----
    torch.set_float32_matmul_precision('medium')  # TF32 for matmuls
    torch.backends.cudnn.benchmark = True  # Optimize conv algorithms
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    start_time = time.time()
    device = torch.device("cuda")

    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_fine_iters = CONFIG["fine_iterations"]

    # Disk space check
    data_disk = shutil.disk_usage(CONFIG["data_dir"])
    out_disk = shutil.disk_usage(str(output_dir))
    print(f"Data disk: {data_disk.free / 1e9:.0f} GB free")
    print(f"Output disk: {out_disk.free / 1e9:.0f} GB free")

    print("=" * 80)
    print("JOINT MULTI-FINE TRAINING (PRECOMPUTED DATA)")
    print("=" * 80)
    print(f"Data: {CONFIG['data_dir']}")
    print(f"deep_query={CONFIG['deep_query']}, freeze_dino={CONFIG['freeze_dino']}")
    print(f"Fine iterations: {num_fine_iters} (coarse -> fine1 -> fine2)")
    print(f"Batch: {CONFIG['batch_size']} x {CONFIG['grad_accum']} = "
          f"{CONFIG['batch_size'] * CONFIG['grad_accum']}")
    print(f"Frames: {CONFIG['num_frames']}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # wandb
    if HAS_WANDB:
        wandb.init(
            project="foveated-vlm-joint",
            name=f"multifine_precomp_{datetime.now().strftime('%m%d_%H%M')}",
            config=CONFIG,
            resume="allow",
        )

    # ---- Load model ----
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
    print(f"  Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # Note: torch.compile skipped - dynamic shapes in multi-pass forward
    # cause recompilation overhead that negates gains

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Optimizer ----
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

    # ---- Resume ----
    print("[2/4] Checking for checkpoint...")
    start_step, start_epoch = load_checkpoint(
        model, optimizer, scaler, scheduler, ckpt_dir, device
    )

    # ---- Dataset ----
    print("[3/4] Loading dataset...")
    dataset = PrecomputedVideoDataset(
        data_dir=CONFIG["data_dir"],
        num_frames=CONFIG["num_frames"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=CollateFn(),
    )

    print(f"  Dataset: {len(dataset)} videos")
    print(f"  Batches/epoch: {len(dataset) // CONFIG['batch_size']}")

    # ---- Training ----
    print("[4/4] Starting training...")
    print(f"{'='*80}\n")

    step = start_step
    epoch = start_epoch
    accum_count = 0
    accum_cap_coarse = 0.0
    accum_cap_iters = [0.0] * num_fine_iters
    accum_rec_coarse = 0.0
    accum_rec_iters = [0.0] * num_fine_iters

    cap_ratios = deque(maxlen=100)
    rec_ratios = deque(maxlen=100)
    step_times = deque(maxlen=50)

    pbar = tqdm(total=CONFIG["max_steps"], initial=start_step, desc="Training")
    optimizer.zero_grad()
    last_step_time = time.time()

    try:
        while step < CONFIG["max_steps"]:
            epoch += 1

            for batch in dataloader:
                # Time limit check
                if CONFIG["max_hours"] > 0:
                    elapsed = (time.time() - start_time) / 3600
                    if elapsed >= CONFIG["max_hours"]:
                        print(f"\n[TIME LIMIT] {CONFIG['max_hours']}h reached")
                        raise StopIteration

                if step >= CONFIG["max_steps"]:
                    break

                # Move to GPU
                frames = batch['frames'].to(device)
                latents = batch['latents'].to(device)
                caption_ids = batch['caption_ids'].to(device)
                caption_mask = batch['caption_mask'].to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    cap_coarse, cap_iter_losses, rec_coarse, rec_iter_losses = \
                        forward_multifine_joint(
                            model, frames, caption_ids, caption_mask,
                            latents, tokenizer, num_fine_iters,
                            compute_coarse_loss=True
                        )

                    # Loss on final fine iteration
                    loss_cap = cap_iter_losses[-1]
                    loss_rec = rec_iter_losses[-1]
                    loss = (loss_cap + CONFIG["lambda_recon"] * loss_rec)

                    # Optional: add coarse auxiliary loss
                    if CONFIG["lambda_coarse"] > 0:
                        loss = loss + CONFIG["lambda_coarse"] * (cap_coarse + rec_coarse)

                    loss = loss / CONFIG["grad_accum"]

                scaler.scale(loss).backward()

                # Accumulate metrics
                accum_cap_coarse += cap_coarse.item()
                for i, l in enumerate(cap_iter_losses):
                    accum_cap_iters[i] += l.item()
                accum_rec_coarse += rec_coarse.item()
                for i, l in enumerate(rec_iter_losses):
                    accum_rec_iters[i] += l.item()
                accum_count += 1

                if accum_count >= CONFIG["grad_accum"]:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CONFIG["grad_clip"]
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Compute averages
                    avg_cap_coarse = accum_cap_coarse / accum_count
                    avg_cap_iters = [x / accum_count for x in accum_cap_iters]
                    avg_rec_coarse = accum_rec_coarse / accum_count
                    avg_rec_iters = [x / accum_count for x in accum_rec_iters]

                    cap_ratio = avg_cap_coarse / (avg_cap_iters[-1] + 1e-8)
                    rec_ratio = avg_rec_coarse / (avg_rec_iters[-1] + 1e-8)
                    cap_ratios.append(cap_ratio)
                    rec_ratios.append(rec_ratio)

                    now = time.time()
                    step_times.append(now - last_step_time)
                    last_step_time = now

                    step += 1
                    pbar.update(1)

                    # Logging
                    if step % CONFIG["log_interval"] == 0:
                        avg_step_time = np.mean(list(step_times))
                        remaining_steps = CONFIG["max_steps"] - step
                        eta_seconds = remaining_steps * avg_step_time
                        if CONFIG["max_hours"] > 0:
                            time_left = CONFIG["max_hours"] * 3600 - (time.time() - start_time)
                            eta_seconds = min(eta_seconds, time_left)
                        eta = datetime.now() + timedelta(seconds=max(0, eta_seconds))

                        cap_prog = " -> ".join([f"{l:.3f}" for l in avg_cap_iters])
                        rec_prog = " -> ".join([f"{l:.3f}" for l in avg_rec_iters])

                        avg_cap_r = np.mean(list(cap_ratios))
                        avg_rec_r = np.mean(list(rec_ratios))
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9

                        log_msg = (
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Step {step:6d} ep{epoch} | "
                            f"cap: {avg_cap_coarse:.3f} -> [{cap_prog}] "
                            f"r={avg_cap_r:.3f} | "
                            f"rec: {avg_rec_coarse:.3f} -> [{rec_prog}] "
                            f"r={avg_rec_r:.3f} | "
                            f"{avg_step_time:.1f}s/step | "
                            f"VRAM: {mem_gb:.1f}GB | "
                            f"ETA: {eta.strftime('%H:%M')}"
                        )
                        tqdm.write(log_msg)

                        if HAS_WANDB:
                            log_dict = {
                                "caption/loss_coarse": avg_cap_coarse,
                                "caption/loss_fine": avg_cap_iters[-1],
                                "caption/ratio": cap_ratio,
                                "caption/ratio_avg": avg_cap_r,
                                "recon/loss_coarse": avg_rec_coarse,
                                "recon/loss_fine": avg_rec_iters[-1],
                                "recon/ratio": rec_ratio,
                                "recon/ratio_avg": avg_rec_r,
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                                "train/vram_gb": mem_gb,
                                "train/step_time": avg_step_time,
                            }
                            for i, l in enumerate(avg_cap_iters):
                                log_dict[f"caption/iter_{i+1}"] = l
                            for i, l in enumerate(avg_rec_iters):
                                log_dict[f"recon/iter_{i+1}"] = l
                            wandb.log(log_dict, step=step)

                    # Checkpoint
                    if step % CONFIG["save_interval"] == 0:
                        metrics = {
                            'cap_ratio': float(np.mean(list(cap_ratios))),
                            'rec_ratio': float(np.mean(list(rec_ratios))),
                            'cap_loss_fine': avg_cap_iters[-1],
                            'rec_loss_fine': avg_rec_iters[-1],
                            'elapsed_hours': (time.time() - start_time) / 3600,
                        }
                        save_checkpoint(
                            model, optimizer, scaler, scheduler,
                            step, epoch, metrics, ckpt_dir,
                            max_keep=CONFIG["max_checkpoints"]
                        )

                    # Reset accumulators
                    accum_cap_coarse = 0.0
                    accum_cap_iters = [0.0] * num_fine_iters
                    accum_rec_coarse = 0.0
                    accum_rec_iters = [0.0] * num_fine_iters
                    accum_count = 0

    except (KeyboardInterrupt, StopIteration):
        print("\n[STOPPED]")

    finally:
        pbar.close()

        elapsed_hours = (time.time() - start_time) / 3600
        final_cap_ratio = float(np.mean(list(cap_ratios))) if cap_ratios else 1.0
        final_rec_ratio = float(np.mean(list(rec_ratios))) if rec_ratios else 1.0

        # Final checkpoint
        metrics = {
            'cap_ratio': final_cap_ratio,
            'rec_ratio': final_rec_ratio,
            'elapsed_hours': elapsed_hours,
            'total_steps': step,
        }
        save_checkpoint(
            model, optimizer, scaler, scheduler,
            step, epoch, metrics, ckpt_dir,
            max_keep=CONFIG["max_checkpoints"]
        )

        # Summary
        summary = {
            'experiment': 'joint_multifine_precomputed',
            'fine_iterations': num_fine_iters,
            'deep_query': CONFIG['deep_query'],
            'freeze_dino': CONFIG['freeze_dino'],
            'elapsed_hours': elapsed_hours,
            'total_steps': step,
            'total_epochs': epoch,
            'final_cap_ratio': final_cap_ratio,
            'final_rec_ratio': final_rec_ratio,
            'config': CONFIG,
        }
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {elapsed_hours:.2f} hours")
        print(f"Steps: {step}, Epochs: {epoch}")
        print(f"Final Caption Ratio: {final_cap_ratio:.4f}")
        print(f"Final Recon Ratio:   {final_rec_ratio:.4f}")
        print(f"{'='*80}")

        if HAS_WANDB:
            wandb.log({
                "final/cap_ratio": final_cap_ratio,
                "final/rec_ratio": final_rec_ratio,
                "final/steps": step,
                "final/hours": elapsed_hours,
            })
            wandb.finish()


class CollateFn:
    """Collate with cached tokenizer (avoid reloading each batch)."""

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


if __name__ == "__main__":
    run_training()
