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
import gc
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
torch.backends.cudnn.benchmark = True       # auto-tune conv algorithms (DINO patch embed)
torch.backends.cudnn.allow_tf32 = True      # TF32 for cuDNN convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # redundant with set_float32_matmul_precision but explicit

# nanochat: expandable segments for GPU memory allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure release/ is importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from release.model import FoveatedVLM
from release.model.multi_token_vlm import MultiTokenVLM
from release.data.webdataset_loader import make_dataloader, create_dpo_webdataset
from release.data.collate import collate_dpo
from release.data.text_interleave import InterleavedDataLoader
from release.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    reduce_mean,
)
from release.utils.checkpoint import save_checkpoint, load_latest_checkpoint
from release.utils.lr_schedule import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_converging_schedule
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
            deep_query=cfg["model"].get("deep_query", True),
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

    # channels_last for DINO conv layers (patch embedding) — better tensor core util
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'dino'):
        model.encoder.dino = model.encoder.dino.to(memory_format=torch.channels_last)

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
        min_frames=cfg["data"].get("min_frames", 0),
        shuffle=True,
        seed=cfg["training"].get("seed", 42),
        epoch=epoch,
        num_workers=cfg["data"].get("num_workers", 12),
        prefetch_factor=cfg["data"].get("prefetch_factor", 8),
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
            num_workers=max(1, cfg["data"].get("num_workers", 12) // 2),
            prefetch_factor=cfg["data"].get("prefetch_factor", 8),
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


def build_dpo_train_loader(cfg: dict, epoch: int = 0):
    """Build the training dataloader for DPO (preference) data."""
    tokenizer = _get_tokenizer(cfg)

    dataset = create_dpo_webdataset(
        shard_pattern=cfg["data"]["train_shards"],
        tokenizer=tokenizer,
        max_frames=cfg["data"].get("max_frames", 64),
        shuffle=True,
        seed=cfg["training"].get("seed", 42),
        epoch=epoch,
        num_workers=cfg["data"].get("num_workers", 2),
        replicate_image_frames=cfg["data"].get("replicate_image_frames", 1),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
        collate_fn=collate_dpo,
        pin_memory=True,
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
        persistent_workers=cfg["data"].get("num_workers", 2) > 0,
    )
    return loader


def build_reference_model(cfg: dict, device: torch.device):
    """
    Build a frozen reference model for DPO training.

    The reference model is a copy of the policy model loaded from the same
    init_from checkpoint (the Stage 2 best). All parameters are frozen
    and the model is set to eval mode.
    """
    ref_model = build_model(cfg, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> dict:
    """
    Compute the DPO loss and reward accuracy.

    DPO loss = -log_sigmoid(β * ((π_chosen - π_ref_chosen) - (π_rejected - π_ref_rejected)))

    Parameters
    ----------
    policy_chosen_logps    : [B]  log-probs from policy on chosen
    policy_rejected_logps  : [B]  log-probs from policy on rejected
    ref_chosen_logps       : [B]  log-probs from reference on chosen
    ref_rejected_logps     : [B]  log-probs from reference on rejected
    beta                   : float  DPO temperature

    Returns
    -------
    dict with keys:
      loss             : scalar DPO loss
      reward_accuracy  : float, fraction where chosen is preferred
      chosen_reward    : [B]   implicit reward for chosen
      rejected_reward  : [B]   implicit reward for rejected
    """
    # Implicit rewards: β * (log π_policy - log π_ref)
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss: -log σ(r_chosen - r_rejected)
    logits = chosen_reward - rejected_reward
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # Reward accuracy: fraction where chosen is preferred over rejected
    reward_accuracy = (logits > 0).float().mean().item()

    return {
        "loss": loss,
        "reward_accuracy": reward_accuracy,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
    }


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
        num_workers=0,  # load in main process — eval is small, avoids RAM spike
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
                kv_cache, _, mask_flat = raw_model._encode_all_frames(frames)
                q_static = raw_model.q_static.expand(B, -1)
                # Compute entropy on first frame
                frame0_kv = raw_model._extract_frame_kv(kv_cache, mask_flat, B, T, 0)
                _, attn_w = raw_model.encoder.query_attend(
                    q_static, frame0_kv, return_attention=True,
                )
                total_entropy += compute_attention_entropy(attn_w) * bs
                entropy_count += bs

                # Save attention maps for a few samples
                if save_attn_dir and attn_samples_saved < max_attn_saves:
                    for t in range(min(T, 4)):
                        if t > 0:
                            frame_kv = raw_model._extract_frame_kv(kv_cache, mask_flat, B, T, t)
                            _, attn_w = raw_model.encoder.query_attend(
                                q_static, frame_kv, return_attention=True,
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
# Throughput: maximize batch size to fill GPU memory
# --------------------------------------------------------------------------- #

def _maximize_batch_size(cfg: dict, device: torch.device):
    """
    Increase batch_size and decrease grad_accum to keep the same effective
    batch while processing more samples per forward pass.  Larger micro-batches
    improve GPU utilization by giving the GPU more parallel work.

    The effective batch (batch_size * grad_accum * world_size) stays constant
    so learning dynamics are unchanged.
    """
    bs = cfg["training"]["batch_size"]
    ga = cfg["training"]["grad_accum"]
    effective = bs * ga

    # Determine max batch size based on available VRAM
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    else:
        return

    # Conservative VRAM targets per model size (leave headroom for spikes)
    llm_path = cfg["model"].get("llm", "")
    if "1.7B" in llm_path or "1.7b" in llm_path:
        max_bs = 8   # 1.7B needs gradient checkpointing, limited VRAM
    elif "360M" in llm_path or "360m" in llm_path:
        max_bs = min(effective, 16)  # 360M: ~6 GB model+optim, fits bs=16
    else:
        # 135M or smaller: model is tiny but video frames dominate VRAM.
        # DINO processes ALL frames in the batch at once; with bucketed padding
        # a batch of 32 × 64 padded frames = 2048 images → OOM on 32GB.
        # bs=16 is a safe 2× improvement over bs=8.
        max_bs = min(effective, 16)

    if max_bs <= bs:
        return  # already at or above target

    new_ga = max(1, effective // max_bs)
    new_bs = effective // new_ga  # adjust to keep effective exact

    if new_bs > bs:
        if is_main_process():
            print(f"  [THROUGHPUT] Batch size: {bs}×{ga} → {new_bs}×{new_ga} "
                  f"(effective={new_bs * new_ga}, was {effective})")
        cfg["training"]["batch_size"] = new_bs
        cfg["training"]["grad_accum"] = new_ga


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(cfg: dict, args):
    rank, world_size, device = setup_distributed()

    # ---- Throughput overrides ----
    # KEEP IT SIMPLE: only safe code-level opts (TF32, cuDNN, channels_last).
    # DO NOT override batch_size, num_workers, or prefetch_factor here.
    # bs=32/16 + high workers caused repeated system OOM crashes.
    # C1-C3 ran stable for hours at bs=8, workers=2, 43-44 samp/s.

    # ---- DPO mode detection ----
    is_dpo = cfg.get("loss", {}).get("type") == "dpo"
    dpo_beta = cfg.get("loss", {}).get("beta", 0.1)

    if is_main_process():
        print(f"=== fVLM Training: Stage {cfg['stage']} ===")
        print(f"  World size:  {world_size}")
        print(f"  Device:      {device}")
        print(f"  Dtype:       {cfg['training'].get('dtype', 'float32')}")
        if is_dpo:
            print(f"  Loss type:   DPO (beta={dpo_beta})")

    # ---- Model ----
    model = build_model(cfg, device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    raw_model = model.module if hasattr(model, "module") else model

    # ---- Reference model for DPO (frozen copy from same init checkpoint) ----
    ref_model = None
    if is_dpo:
        if is_main_process():
            print("  Loading reference model (frozen) ...")
        ref_model = build_reference_model(cfg, device)
        if is_main_process():
            ref_params = sum(p.numel() for p in ref_model.parameters())
            print(f"  Reference model: {ref_params:,} params (all frozen)")

    # ---- Gradient checkpointing (nanochat: trade compute for memory) ----
    if cfg["model"].get("gradient_checkpointing", False):
        if hasattr(raw_model, "enable_gradient_checkpointing"):
            # llm_only=True when compile_encoder is set (DINO can be compiled
            # only if it doesn't use gradient checkpointing)
            llm_only = cfg["training"].get("compile_encoder", False)
            raw_model.enable_gradient_checkpointing(llm_only=llm_only)
            if is_main_process():
                mode = "LLM only" if llm_only else "LLM + DINO"
                print(f"  Gradient checkpointing: {mode}")

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
        fused=True,  # nanochat: fused kernel eliminates Python overhead
    )

    # ---- Schedule ----
    grad_accum = cfg["training"].get("grad_accum", 1)
    effective_batch = cfg["training"]["batch_size"] * grad_accum * world_size
    total_steps = cfg["training"]["total_samples"] // effective_batch
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.05))

    schedule_type = cfg["training"].get("schedule", "cosine")
    if schedule_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    elif schedule_type == "converging":
        # Stage 1: connector 100:1 → 1:1 convergence with backbone
        target_lr = cfg["training"].get("target_lr", 3e-5)
        scheduler = get_converging_schedule(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            target_lr=target_lr,
        )
        if is_main_process():
            print(f"  Schedule: converging to target_lr={target_lr} (100:1 → 1:1)")
    else:
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
        compile_mode = cfg["training"].get("compile_mode", "reduce-overhead")
        if is_main_process():
            print(f"  Compiling model with torch.compile ({compile_mode}) ...")
        # Compile individual components to avoid graph breaks at boundaries
        # DINO encoder: fixed 224x224 inputs → dynamic=False for better optimization
        raw_model.encoder = torch.compile(raw_model.encoder, mode=compile_mode, dynamic=False)
        # LLM: variable sequence length → dynamic=True
        raw_model.llm = torch.compile(raw_model.llm, mode=compile_mode, dynamic=True)
        raw_model.dino_to_llm = torch.compile(raw_model.dino_to_llm, mode=compile_mode)
        raw_model.llm_to_query = torch.compile(raw_model.llm_to_query, mode=compile_mode)
    elif cfg["training"].get("compile_encoder", False) and hasattr(torch, "compile"):
        # Selective compile: DINO encoder only. Safe with gradient checkpointing
        # because DINO doesn't use grad_ckpt when llm_only=True.
        # DINO has fixed 224×224 inputs → dynamic=False for better optimization.
        compile_mode = cfg["training"].get("compile_mode", "reduce-overhead")
        if is_main_process():
            print(f"  Compiling DINO encoder only with torch.compile ({compile_mode}) ...")
        raw_model.encoder = torch.compile(raw_model.encoder, mode=compile_mode, dynamic=False)

    # ---- Val loader ----
    val_loader = build_val_loader(cfg)

    # ---- Dry run ----
    if args.dry_run:
        if is_main_process():
            if is_dpo:
                loader = build_dpo_train_loader(cfg, epoch=0)
            else:
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
    optimizer.zero_grad(set_to_none=True)  # nanochat: faster than setting to zero

    # nanochat: disable GC during training — saves ~500ms per collection
    gc.collect()
    gc.disable()

    t0 = time.time()

    # Track DPO metrics across micro-steps for logging
    dpo_reward_acc_accum = 0.0
    dpo_chosen_reward_accum = 0.0
    dpo_rejected_reward_accum = 0.0
    dpo_micro_count = 0

    while global_step < total_steps:
        if is_dpo:
            train_loader = build_dpo_train_loader(cfg, epoch=epoch)
        else:
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
                try:
                    if is_dpo:
                        # ---- DPO forward pass ----
                        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                            # Policy model forward
                            policy_out = raw_model.forward_dpo(
                                frames=batch["frames"],
                                chosen_input_ids=batch["chosen_input_ids"],
                                chosen_attention_mask=batch["chosen_attention_mask"],
                                chosen_loss_mask=batch["chosen_loss_mask"],
                                rejected_input_ids=batch["rejected_input_ids"],
                                rejected_attention_mask=batch["rejected_attention_mask"],
                                rejected_loss_mask=batch["rejected_loss_mask"],
                                frame_mask=batch.get("frame_mask"),
                            )

                            # Reference model forward (frozen, no grad)
                            with torch.no_grad():
                                ref_out = ref_model.forward_dpo(
                                    frames=batch["frames"],
                                    chosen_input_ids=batch["chosen_input_ids"],
                                    chosen_attention_mask=batch["chosen_attention_mask"],
                                    chosen_loss_mask=batch["chosen_loss_mask"],
                                    rejected_input_ids=batch["rejected_input_ids"],
                                    rejected_attention_mask=batch["rejected_attention_mask"],
                                    rejected_loss_mask=batch["rejected_loss_mask"],
                                    frame_mask=batch.get("frame_mask"),
                                )

                            # Compute DPO loss
                            dpo_result = compute_dpo_loss(
                                policy_chosen_logps=policy_out["chosen_logps"],
                                policy_rejected_logps=policy_out["rejected_logps"],
                                ref_chosen_logps=ref_out["chosen_logps"],
                                ref_rejected_logps=ref_out["rejected_logps"],
                                beta=dpo_beta,
                            )
                            loss = dpo_result["loss"] / grad_accum

                            # Store outputs for logging (mimic SFT outputs dict)
                            outputs = {
                                "loss": dpo_result["loss"],
                                "fine_loss": dpo_result["loss"],  # alias for logger
                                "coarse_loss": torch.tensor(0.0, device=device),
                                "reward_accuracy": dpo_result["reward_accuracy"],
                                "chosen_reward": dpo_result["chosen_reward"].mean().item(),
                                "rejected_reward": dpo_result["rejected_reward"].mean().item(),
                            }

                        # Accumulate DPO metrics for logging at optimizer step
                        dpo_reward_acc_accum += dpo_result["reward_accuracy"]
                        dpo_chosen_reward_accum += dpo_result["chosen_reward"].mean().item()
                        dpo_rejected_reward_accum += dpo_result["rejected_reward"].mean().item()
                        dpo_micro_count += 1

                        scaler.scale(loss).backward()
                    else:
                        # ---- Standard SFT forward pass ----
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
                except torch.cuda.OutOfMemoryError:
                    # Rare: batch with too many real frames. Skip and continue.
                    if is_main_process():
                        n_real = batch.get("frame_mask", batch["frames"]).sum().item()
                        print(f"  [OOM] Skipping batch at step {global_step} "
                              f"(n_real={n_real}). Clearing cache.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0  # reset accumulation
                    dpo_reward_acc_accum = 0.0
                    dpo_chosen_reward_accum = 0.0
                    dpo_rejected_reward_accum = 0.0
                    dpo_micro_count = 0
                    continue

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
            optimizer.zero_grad(set_to_none=True)  # nanochat: faster
            global_step += 1

            # ---- Logging ----
            if is_main_process() and global_step % log_every == 0:
                elapsed = time.time() - t0
                samples_per_sec = samples_seen / max(elapsed, 1e-6)
                lr_groups = {g.get("name", "default"): g["lr"]
                             for g in optimizer.param_groups}

                if is_dpo and dpo_micro_count > 0:
                    # DPO-specific logging
                    avg_reward_acc = dpo_reward_acc_accum / dpo_micro_count
                    avg_chosen_reward = dpo_chosen_reward_accum / dpo_micro_count
                    avg_rejected_reward = dpo_rejected_reward_accum / dpo_micro_count
                    reward_margin = avg_chosen_reward - avg_rejected_reward

                    print(
                        f"  step {global_step:6d} | dpo_loss {outputs['loss'].item():.4f} | "
                        f"rew_acc {avg_reward_acc:.3f} | margin {reward_margin:.3f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"gnorm {(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm):.2f} | "
                        f"{samples_per_sec:.0f} samp/s",
                        flush=True,
                    )

                    # Log to wandb/CSV via logger (use fine_loss slot for DPO loss)
                    logger.log_step(
                        step=global_step,
                        loss=outputs["loss"].item(),
                        fine_loss=outputs["loss"].item(),
                        coarse_loss=0.0,
                        lr=scheduler.get_last_lr()[0],
                        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        samples_seen=samples_seen,
                        samples_per_sec=samples_per_sec,
                        lr_groups=lr_groups,
                    )

                    # Log DPO-specific metrics to wandb
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                "dpo/reward_accuracy": avg_reward_acc,
                                "dpo/chosen_reward": avg_chosen_reward,
                                "dpo/rejected_reward": avg_rejected_reward,
                                "dpo/reward_margin": reward_margin,
                            }, step=global_step)
                    except Exception:
                        pass

                    # Reset DPO accumulators
                    dpo_reward_acc_accum = 0.0
                    dpo_chosen_reward_accum = 0.0
                    dpo_rejected_reward_accum = 0.0
                    dpo_micro_count = 0
                else:
                    # Standard SFT logging
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
                else:
                    # No val_loader — use train loss as metric (pretraining style)
                    metric = outputs["loss"].item() if isinstance(outputs["loss"], torch.Tensor) else outputs["loss"]
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
        final_loss = outputs["loss"].item() if isinstance(outputs["loss"], torch.Tensor) else outputs["loss"]
        logger.save_run_summary(final_loss=final_loss, total_samples=samples_seen)
        logger.finish()
        print(f"\n  Training complete: {global_step} steps, "
              f"{samples_seen:,} samples, {elapsed/3600:.1f}h")

    cleanup_distributed()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    train(cfg, args)
