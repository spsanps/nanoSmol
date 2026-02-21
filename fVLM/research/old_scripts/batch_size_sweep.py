#!/usr/bin/env python
"""Sweep batch sizes with real data to find optimal throughput."""

import os
import sys
import time
import gc

import torch
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TMPDIR", "/workspace/tmp")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from release.model import FoveatedVLM
from release.train import build_train_loader


def test_batch_size(cfg, bs, max_steps=30, device=torch.device("cuda")):
    """Test a specific batch size, return avg samp/s or None if OOM."""
    cfg_copy = {**cfg}
    cfg_copy["training"] = {**cfg["training"], "batch_size": bs}

    model = FoveatedVLM(
        llm_name=cfg["model"]["llm"],
        dino_name=cfg["model"]["dino"],
        query_dim=cfg["model"].get("query_dim", 384),
        visual_scale=cfg["model"].get("visual_scale", 0.14),
        lambda_coarse=cfg["model"].get("lambda_coarse", 0.0),
        deep_query=cfg["model"].get("deep_query", True),
    ).to(device)
    model.enable_gradient_checkpointing()
    model.encoder.dino = model.encoder.dino.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, fused=True)
    amp_dtype = torch.bfloat16

    loader = build_train_loader(cfg_copy, epoch=0)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    gc.collect()
    gc.disable()

    total_samples = 0
    oom_count = 0
    step = 0
    skip_first = 3  # skip warmup steps

    t_start = None

    for batch in loader:
        if step >= max_steps + skip_first:
            break

        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(
                    frames=batch["frames"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"],
                    frame_mask=batch.get("frame_mask"),
                    mode="coarse_fine",
                )
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            step += 1
            if step > skip_first:
                if t_start is None:
                    t_start = time.time()
                total_samples += batch["frames"].shape[0]

        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if oom_count > 3:
                break

    gc.enable()
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    del model, optimizer, loader
    torch.cuda.empty_cache()
    gc.collect()

    if t_start is None or total_samples == 0:
        return None, oom_count, peak_mem

    elapsed = time.time() - t_start
    sps = total_samples / elapsed
    return sps, oom_count, peak_mem


def main():
    config_path = "release/configs/final/stage1_1.7B.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("Batch Size Sweep (real data, optimized coarse pass)")
    print("=" * 60)
    print(f"{'BS':>4}  {'samp/s':>8}  {'OOMs':>5}  {'PeakGB':>7}")
    print("-" * 40)

    for bs in [32, 48, 64]:
        sps, ooms, peak = test_batch_size(cfg, bs, max_steps=25)
        if sps is not None:
            print(f"  {bs:3d}   {sps:7.1f}   {ooms:4d}   {peak:6.1f}")
        else:
            print(f"  {bs:3d}   FAILED    {ooms:4d}   {peak:6.1f}")

    print("-" * 40)
    print("Done")


if __name__ == "__main__":
    main()
