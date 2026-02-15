#!/usr/bin/env python
"""
Unified training script for foveated VLM (all stages).

Usage:
    # Single GPU
    python release/train.py --config release/configs/stage1_webvid.yaml

    # Multi-GPU (2xA100)
    torchrun --nproc_per_node=2 release/train.py --config release/configs/stage1_webvid.yaml

    # Dry run (verify config, dataloaders, shapes)
    python release/train.py --config release/configs/stage1_webvid.yaml --dry-run
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

# TF32 for free speedup on Ampere+ GPUs (RTX 3090, A100, RTX 5090, etc.)
torch.set_float32_matmul_precision("high")

# Ensure release/ is importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from release.model import FoveatedVLM
from release.model.multi_token_vlm import MultiTokenVLM
from release.data.webdataset_loader import make_dataloader
from release.data.text_interleave import InterleavedDataLoader
from release.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    reduce_mean,
)
from release.utils.checkpoint import save_checkpoint, load_latest_checkpoint
from release.utils.lr_schedule import get_cosine_schedule_with_warmup
from release.utils.logging_utils import TrainingLogger
from release.utils.attention_viz import compute_attention_entropy, save_attention_maps


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="fVLM training")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse config, build model & dataloader, print shapes, exit")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------- #
# Build components
# --------------------------------------------------------------------------- #

def build_model(cfg: dict, device: torch.device):
    if cfg["model"].get("multi_token", False):
        model = MultiTokenVLM(
            llm_name=cfg["model"]["llm"],
            dino_name=cfg["model"]["dino"],
            tokens_per_frame=cfg["model"].get("tokens_per_frame", 16),
            visual_scale=cfg["model"].get("visual_scale", 0.14),
        )
        if is_main_process():
            print(f"  Model: MultiTokenVLM ({cfg['model'].get('tokens_per_frame', 16)} tokens/frame)")
    else:
        model = FoveatedVLM(
            llm_name=cfg["model"]["llm"],
            dino_name=cfg["model"]["dino"],
            query_dim=cfg["model"].get("query_dim", 384),
            visual_scale=cfg["model"].get("visual_scale", 0.14),
            lambda_coarse=cfg["model"].get("lambda_coarse", 0.0),
        )

    # Initialise from a previous-stage checkpoint (Stage 2 loads Stage 1, etc.)
    init_from = cfg["model"].get("init_from")
    if init_from and os.path.exists(init_from):
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():
            print(f"  Loaded weights from {init_from}")

    if cfg["model"].get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    model = model.to(device)

    # ---- Freeze parameters based on config ----
    dino_module = getattr(model, 'encoder', None)
    if dino_module is not None:
        dino_module = getattr(dino_module, 'dino', None)
    if dino_module is None:
        dino_module = getattr(model, 'dino', None)

    if cfg["model"].get("freeze_dino", False) and dino_module is not None:
        for p in dino_module.parameters():
            p.requires_grad = False
        if is_main_process():
            print("  Frozen: DINO encoder")

    if cfg["model"].get("freeze_llm", False):
        for p in model.llm.parameters():
            p.requires_grad = False
        if is_main_process():
            print("  Frozen: LLM backbone")

    return model


def _get_tokenizer(cfg: dict):
    """Lazy-load the tokenizer for on-the-fly tokenization of raw captions."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["llm"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_train_loader(cfg: dict, epoch: int = 0):
    """Build the training dataloader (vision + optional text interleave)."""
    stage = cfg.get("stage", 1)
    tokenizer = _get_tokenizer(cfg)

    vision_loader = make_dataloader(
        shard_pattern=cfg["data"]["train_shards"],
        batch_size=cfg["training"]["batch_size"],
        max_frames=cfg["data"].get("max_frames", 64),
        shuffle=True,
        seed=cfg["training"].get("seed", 42),
        epoch=epoch,
        num_workers=cfg["data"].get("num_workers", 4),
        tokenizer=tokenizer,
        stage=stage,
        replicate_image_frames=cfg["data"].get("replicate_image_frames", 1),
    )

    text_ratio = cfg["data"].get("text_ratio", 0.0)
    if text_ratio > 0 and cfg["data"].get("text_shards"):
        text_loader = make_dataloader(
            shard_pattern=cfg["data"]["text_shards"],
            batch_size=cfg["training"]["batch_size"],
            max_frames=1,
            shuffle=True,
            seed=cfg["training"].get("seed", 42),
            epoch=epoch,
            num_workers=max(1, cfg["data"].get("num_workers", 4) // 2),
            tokenizer=tokenizer,
            stage=stage,
        )
        return InterleavedDataLoader(
            vision_loader=vision_loader,
            text_loader=text_loader,
            text_ratio=text_ratio,
            seed=cfg["training"].get("seed", 42) + epoch,
        )

    return vision_loader


def build_val_loader(cfg: dict):
    val_shards = cfg["data"].get("val_shards")
    if not val_shards:
        return None
    stage = cfg.get("stage", 1)
    tokenizer = _get_tokenizer(cfg)
    return make_dataloader(
        shard_pattern=val_shards,
        batch_size=cfg["training"]["batch_size"],
        max_frames=cfg["data"].get("max_frames", 64),
        shuffle=False,
        num_workers=max(1, cfg["data"].get("num_workers", 4) // 2),
        tokenizer=tokenizer,
        stage=stage,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype, use_amp, cfg,
             save_attn_dir=None, step=0):
    """Run validation and return dict of average losses + attention entropy."""
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    is_foveated = hasattr(raw_model, "encoder")

    total_loss = 0.0
    total_fine = 0.0
    total_coarse = 0.0
    total_entropy = 0.0
    entropy_count = 0
    count = 0
    max_samples = cfg.get("eval", {}).get("max_samples", 1000)
    eval_mode = "coarse_only" if cfg["model"].get("coarse_only", False) else "coarse_fine"
    attn_samples_saved = 0
    max_attn_saves = 10  # save attention maps for first 10 eval batches

    for batch in val_loader:
        if count >= max_samples:
            break

        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
                frame_mask=batch.get("frame_mask"),
                mode=eval_mode,
            )

        bs = batch["frames"].shape[0]
        total_loss += outputs["loss"].item() * bs
        total_fine += outputs.get("fine_loss", outputs["loss"]).item() * bs
        total_coarse += outputs.get("coarse_loss", torch.tensor(0.0)).item() * bs
        count += bs

        # Attention entropy (foveated model only, sample periodically)
        if is_foveated and entropy_count < 50:
            try:
                frames = batch["frames"]
                B, T = frames.shape[:2]
                per_frame_caches = raw_model._encode_all_frames(frames)
                q_static = raw_model.q_static.expand(B, -1)
                # Compute entropy on first frame
                _, attn_w = raw_model.encoder.query_attend(
                    q_static, per_frame_caches[0], return_attention=True,
                )
                total_entropy += compute_attention_entropy(attn_w) * bs
                entropy_count += bs

                # Save attention maps for a few samples
                if save_attn_dir and attn_samples_saved < max_attn_saves:
                    for t in range(min(T, 4)):
                        if t > 0:
                            _, attn_w = raw_model.encoder.query_attend(
                                q_static, per_frame_caches[t], return_attention=True,
                            )
                        save_attention_maps(
                            attn_w, save_attn_dir, step,
                            sample_idx=0, frame_idx=t,
                            prefix=f"attn_s{attn_samples_saved:03d}",
                        )
                    attn_samples_saved += 1
            except Exception:
                pass  # don't break eval if attention extraction fails

    avg_loss = reduce_mean(torch.tensor(total_loss / max(count, 1), device=device)).item()
    avg_fine = reduce_mean(torch.tensor(total_fine / max(count, 1), device=device)).item()
    avg_coarse = reduce_mean(torch.tensor(total_coarse / max(count, 1), device=device)).item()
    avg_entropy = total_entropy / max(entropy_count, 1) if entropy_count > 0 else 0.0

    return {
        "val_loss": avg_loss,
        "val_fine_loss": avg_fine,
        "val_coarse_loss": avg_coarse,
        "attention_entropy": avg_entropy,
    }


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(cfg: dict, args):
    rank, world_size, device = setup_distributed()

    if is_main_process():
        print(f"=== fVLM Training: Stage {cfg['stage']} ===")
        print(f"  World size:  {world_size}")
        print(f"  Device:      {device}")
        print(f"  Dtype:       {cfg['training'].get('dtype', 'float32')}")

    # ---- Model ----
    model = build_model(cfg, device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    raw_model = model.module if hasattr(model, "module") else model

    # ---- Optimizer (differential LR) ----
    param_groups = raw_model.get_param_groups(
        lr_backbone=cfg["training"].get("lr_dino", 1e-5),
        lr_connector=cfg["training"].get("lr_connector", 1e-4),
    )
    # Override LLM LR if specified separately from DINO
    llm_lr = cfg["training"].get("lr_llm")
    if llm_lr is not None:
        for g in param_groups:
            if g.get("name") == "llm":
                g["lr"] = llm_lr

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    # ---- Schedule ----
    grad_accum = cfg["training"].get("grad_accum", 1)
    effective_batch = cfg["training"]["batch_size"] * grad_accum * world_size
    total_steps = cfg["training"]["total_samples"] // effective_batch
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.05))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Mixed precision ----
    dtype_str = cfg["training"].get("dtype", "float32")
    use_amp = dtype_str in ("bfloat16", "float16")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype_str, torch.float32
    )
    # GradScaler only needed for float16 (bfloat16 doesn't need it)
    scaler = torch.amp.GradScaler(enabled=(dtype_str == "float16"))

    # ---- Resume ----
    start_step = 0
    data_position = 0
    ckpt_dir = cfg["checkpoint"]["save_dir"]

    if cfg["checkpoint"].get("resume") == "auto":
        resume_info = load_latest_checkpoint(
            ckpt_dir, model, optimizer, scaler, scheduler,
            map_location=str(device),
        )
        if resume_info:
            start_step = resume_info["step"]
            data_position = resume_info["data_position"]

    # ---- Logger ----
    logger = TrainingLogger(
        project=cfg.get("wandb", {}).get("project", "foveated-vlm"),
        config=cfg,
        enabled=is_main_process(),
    )

    # ---- torch.compile ----
    if cfg["training"].get("compile", False) and hasattr(torch, "compile"):
        if is_main_process():
            print("  Compiling encoder with torch.compile (dynamic=True) ...")
        raw_model.encoder = torch.compile(raw_model.encoder, dynamic=True)

    # ---- Val loader ----
    val_loader = build_val_loader(cfg)

    # ---- Dry run ----
    if args.dry_run:
        if is_main_process():
            loader = build_train_loader(cfg, epoch=0)
            batch = next(iter(loader))
            print(f"\n  Dry run OK:")
            for k, v in batch.items():
                shape = v.shape if hasattr(v, "shape") else type(v).__name__
                print(f"    {k:20s} {shape}")
            print(f"    total_steps      = {total_steps}")
            print(f"    warmup_steps     = {warmup_steps}")
            print(f"    effective_batch  = {effective_batch}")
            n_params = sum(p.numel() for p in raw_model.parameters())
            n_train  = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            print(f"    total_params     = {n_params:,}")
            print(f"    trainable_params = {n_train:,}")
        cleanup_distributed()
        return

    if is_main_process():
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"  Parameters:  {n_params:,}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup:      {warmup_steps}")
        print(f"  Eff. batch:  {effective_batch}")
        print(f"  Starting at: step={start_step}, samples={data_position}")
        print()

    # ---- Train ----
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)
    save_every = cfg["checkpoint"].get("save_every_steps", 1000)
    eval_every = cfg.get("eval", {}).get("every_steps", 500)
    log_every = 10
    train_mode = "coarse_only" if cfg["model"].get("coarse_only", False) else "coarse_fine"
    if is_main_process() and train_mode != "coarse_fine":
        print(f"  Train mode: {train_mode}")

    global_step = start_step
    samples_seen = data_position
    epoch = data_position // max(cfg["training"]["total_samples"], 1)
    micro_step = 0

    model.train()
    optimizer.zero_grad()

    t0 = time.time()

    while global_step < total_steps:
        train_loader = build_train_loader(cfg, epoch=epoch)

        for batch in train_loader:
            if global_step >= total_steps:
                break

            # Move to device
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Gradient accumulation: skip DDP sync on non-final micro-steps
            is_accum = ((micro_step + 1) % grad_accum != 0)
            sync_ctx = model.no_sync() if (world_size > 1 and is_accum) else nullcontext()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        frames=batch["frames"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        loss_mask=batch["loss_mask"],
                        frame_mask=batch.get("frame_mask"),
                        mode=train_mode,
                    )
                    loss = outputs["loss"] / grad_accum

                scaler.scale(loss).backward()

            samples_seen += batch["frames"].shape[0] * world_size
            micro_step += 1

            # Skip optimizer step if still accumulating
            if is_accum:
                continue

            # ---- Optimizer step ----
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # ---- Logging ----
            if is_main_process() and global_step % log_every == 0:
                elapsed = time.time() - t0
                samples_per_sec = samples_seen / max(elapsed, 1e-6)
                lr_groups = {g.get("name", "default"): g["lr"]
                             for g in optimizer.param_groups}
                logger.log_step(
                    step=global_step,
                    loss=outputs["loss"].item(),
                    fine_loss=outputs["fine_loss"].item(),
                    coarse_loss=outputs["coarse_loss"].item(),
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    samples_seen=samples_seen,
                    samples_per_sec=samples_per_sec,
                    lr_groups=lr_groups,
                )

            # ---- Evaluation ----
            if val_loader is not None and global_step % eval_every == 0:
                attn_dir = os.path.join(ckpt_dir, "attention_maps") if is_main_process() else None
                val_result = evaluate(
                    model, val_loader, device, amp_dtype, use_amp, cfg,
                    save_attn_dir=attn_dir, step=global_step,
                )
                if is_main_process():
                    logger.log_eval(
                        step=global_step,
                        val_loss=val_result["val_loss"],
                        val_fine_loss=val_result["val_fine_loss"],
                        val_coarse_loss=val_result["val_coarse_loss"],
                        attention_entropy=val_result["attention_entropy"],
                    )
                model.train()

            # ---- Checkpoint ----
            if global_step % save_every == 0:
                metric = None
                if val_loader is not None and global_step % eval_every != 0:
                    val_result = evaluate(model, val_loader, device, amp_dtype, use_amp, cfg)
                    metric = val_result["val_loss"]
                    model.train()
                elif val_loader is not None:
                    metric = val_result["val_loss"]  # reuse from eval above
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    step=global_step,
                    data_position=samples_seen,
                    save_dir=ckpt_dir,
                    metric_value=metric,
                    config=cfg,
                )

        epoch += 1

    # ---- Final checkpoint ----
    save_checkpoint(
        model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
        step=global_step, data_position=samples_seen, save_dir=ckpt_dir,
        config=cfg,
    )

    if is_main_process():
        elapsed = time.time() - t0
        logger.save_run_summary(final_loss=outputs["loss"].item(), total_samples=samples_seen)
        logger.finish()
        print(f"\n  Training complete: {global_step} steps, "
              f"{samples_seen:,} samples, {elapsed/3600:.1f}h")

    cleanup_distributed()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    train(cfg, args)
