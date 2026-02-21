#!/usr/bin/env python
"""Quick 50-step training test with the optimized model to measure real-data throughput."""

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
from release.train import build_train_loader, _get_tokenizer


def main():
    device = torch.device("cuda")
    config_path = "release/configs/final/stage1_1.7B.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("50-step Real Data Training Test (optimized coarse pass)")
    print("=" * 70)

    # Build model
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Batch size: {cfg['training']['batch_size']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.01, fused=True,
    )

    amp_dtype = torch.bfloat16
    max_grad_norm = 1.0

    # Build real data loader
    loader = build_train_loader(cfg, epoch=0)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    gc.collect()
    gc.disable()

    # GPU monitoring
    print(f"\n{'Step':>6} {'Loss':>8} {'GNorm':>8} {'ms/step':>10} {'samp/s':>8} {'T':>4} {'n_real':>8} {'MemGB':>7}")
    print("-" * 70)

    total_samples = 0
    t_start = time.time()
    step = 0
    max_steps = 50

    for batch in loader:
        if step >= max_steps:
            break

        # Move to device
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        bs = batch["frames"].shape[0]
        T = batch["frames"].shape[1]
        n_real = batch.get("frame_mask", batch["frames"][:, :, 0, 0, 0]).sum().item() if "frame_mask" in batch else bs * T

        t0 = time.time()

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
                loss = outputs["loss"]

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            t1 = time.time()
            step_ms = (t1 - t0) * 1000
            total_samples += bs
            step += 1

            mem_gb = torch.cuda.max_memory_allocated() / 1e9

            if step % 5 == 0 or step <= 5:
                sps = total_samples / (time.time() - t_start)
                gnorm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                print(f"  {step:4d}   {loss.item():7.4f}   {gnorm:7.2f}   {step_ms:9.1f}   {sps:7.1f}   {T:3d}   {n_real:7.0f}   {mem_gb:6.1f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] step {step}, T={T}, n_real={n_real}")
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            continue

    elapsed = time.time() - t_start
    avg_sps = total_samples / elapsed

    print("-" * 70)
    print(f"\n  Total: {step} steps, {total_samples} samples in {elapsed:.1f}s")
    print(f"  Average throughput: {avg_sps:.1f} samp/s")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    gc.enable()


if __name__ == "__main__":
    main()
