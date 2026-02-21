#!/usr/bin/env python
"""
Unified evaluation script for foveated VLM.

Supports three evaluation modes:
  1. coarse_only      -- fast, single static-query pass
  2. coarse_fine      -- two-pass (parallel approx, matches training)
  3. autoregressive   -- true sequential inference with KV cache

Usage:
    python release/evaluate.py \
        --config release/configs/stage1_webvid.yaml \
        --checkpoint /workspace/checkpoints/stage1/best.pt \
        --mode coarse_fine \
        --max-samples 5000
"""

import argparse
import os
import sys
import time

import torch
import yaml

# Needed for CLI invocation: `python release/evaluate.py` sets __file__'s parent as the
# working dir, but `release.*` imports require the repo root on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from release.model import FoveatedVLM
from release.data.webdataset_loader import make_dataloader
from release.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    reduce_mean,
)


def parse_args():
    p = argparse.ArgumentParser(description="fVLM evaluation")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt path")
    p.add_argument("--mode", default="coarse_fine",
                   choices=["coarse_only", "coarse_fine", "autoregressive"],
                   help="Evaluation forward mode")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--split", default="val", choices=["val", "train"],
                   help="Which data split to evaluate on")
    return p.parse_args()


def load_model(cfg: dict, checkpoint_path: str, device: torch.device):
    model = FoveatedVLM(
        llm_name=cfg["model"]["llm"],
        dino_name=cfg["model"]["dino"],
        query_dim=cfg["model"].get("query_dim", 384),
        visual_scale=cfg["model"].get("visual_scale", 0.14),
        lambda_coarse=cfg["model"].get("lambda_coarse", 0.0),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    step = ckpt.get("step", "?")
    if is_main_process():
        print(f"  Loaded checkpoint: step {step}")

    return model.to(device)


@torch.no_grad()
def run_eval(model, loader, device, mode, max_samples, amp_dtype, use_amp):
    model.eval()
    total_loss = 0.0
    count = 0
    t0 = time.time()

    for batch in loader:
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
                mode=mode,
            )

        bs = batch["frames"].shape[0]
        total_loss += outputs["loss"].item() * bs
        count += bs

        if is_main_process() and count % 500 == 0:
            print(f"    ... {count}/{max_samples} samples", flush=True)

    elapsed = time.time() - t0
    avg_loss = total_loss / max(count, 1)

    avg_loss_t = torch.tensor(avg_loss, device=device)
    avg_loss_t = reduce_mean(avg_loss_t)

    return avg_loss_t.item(), count, elapsed


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, world_size, device = setup_distributed()

    if is_main_process():
        print(f"=== fVLM Evaluation: Stage {cfg['stage']} ===")
        print(f"  Mode:       {args.mode}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Max samples: {args.max_samples}")

    model = load_model(cfg, args.checkpoint, device)

    # Load tokenizer for on-the-fly tokenization of samples without pre-tokenized data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["llm"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    shard_key = "val_shards" if args.split == "val" else "train_shards"
    loader = make_dataloader(
        shard_pattern=cfg["data"][shard_key],
        batch_size=cfg["training"]["batch_size"],
        max_frames=cfg["data"].get("max_frames", 64),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        tokenizer=tokenizer,
        stage=cfg.get("stage", 1),
    )

    dtype_str = cfg["training"].get("dtype", "float32")
    use_amp = dtype_str in ("bfloat16", "float16")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype_str, torch.float32
    )

    avg_loss, count, elapsed = run_eval(
        model, loader, device, args.mode, args.max_samples, amp_dtype, use_amp,
    )

    if is_main_process():
        print(f"\n  Results ({args.mode}):")
        print(f"    avg_loss   = {avg_loss:.4f}")
        print(f"    samples    = {count}")
        print(f"    elapsed    = {elapsed:.1f}s")
        print(f"    throughput = {count/max(elapsed,1e-6):.1f} samples/s")

    cleanup_distributed()


if __name__ == "__main__":
    main()
