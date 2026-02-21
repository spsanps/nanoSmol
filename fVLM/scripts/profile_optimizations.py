#!/usr/bin/env python3
"""
Profile Codex-recommended optimizations for fVLM training.

Tests each optimization independently and in combination:
1. Baseline (current config: grad_ckpt=True, compile=default)
2. No gradient checkpointing
3. Liger Kernel FusedLinearCrossEntropyLoss
4. max-autotune compile mode
5. fullgraph=True for encoder compile
6. Increased batch size (max_total_frames)
7. Combined: best of above
"""

import os, sys, time, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from release.model import FoveatedVLM

# ── Config ──────────────────────────────────────────────────────────
MODEL_LLM = "/workspace/models/SmolLM2-135M-Instruct"  # Use 135M for fast profiling
MODEL_DINO = "/workspace/models/dinov2-small"
DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP_STEPS = 3
MEASURE_STEPS = 10
NUM_FRAMES = 8
SEQ_LEN = 128

# Try to import Liger Kernel
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
    print("WARNING: liger-kernel not installed, skipping FLCE tests")


def create_model(grad_ckpt=False, use_fused_ce=False):
    """Create fresh model instance."""
    model = FoveatedVLM(
        llm_name=MODEL_LLM,
        dino_name=MODEL_DINO,
        query_dim=384,
        visual_scale=0.14,
        deep_query=True,
        use_fused_ce=use_fused_ce,
    )
    model = model.to(DEVICE).to(DTYPE)
    if grad_ckpt:
        model.enable_gradient_checkpointing(use_reentrant=False)
    model.train()
    return model


def create_optimizer(model):
    """Simple optimizer for profiling."""
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


def create_batch(batch_size, num_frames=NUM_FRAMES, seq_len=SEQ_LEN):
    """Create synthetic training batch."""
    frames = torch.randn(batch_size, num_frames, 3, 224, 224, device=DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, 49152, (batch_size, seq_len), device=DEVICE)
    attention_mask = torch.ones(batch_size, seq_len, device=DEVICE, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, device=DEVICE, dtype=torch.float32)
    return frames, input_ids, attention_mask, loss_mask


def profile_config(name, model, optimizer, batch_size,
                   compile_model=False, compile_mode="default", fullgraph_encoder=False):
    """Profile a single configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  batch_size={batch_size}, compile={compile_model}, mode={compile_mode}")
    print(f"  fused_ce={model.use_fused_ce}, fullgraph_encoder={fullgraph_encoder}")
    print(f"{'='*60}")

    # Apply compile if requested
    if compile_model:
        try:
            model.encoder = torch.compile(
                model.encoder, mode=compile_mode, dynamic=False,
                fullgraph=fullgraph_encoder,
            )
            model.llm = torch.compile(model.llm, mode=compile_mode, dynamic=True)
            model.dino_to_llm = torch.compile(model.dino_to_llm, mode=compile_mode)
            model.llm_to_query = torch.compile(model.llm_to_query, mode=compile_mode)
        except Exception as e:
            print(f"  COMPILE FAILED: {e}")

    # Warmup
    print(f"  Warming up ({WARMUP_STEPS} steps)...")
    for step in range(WARMUP_STEPS):
        frames, input_ids, attn_mask, loss_mask = create_batch(batch_size)
        optimizer.zero_grad()

        result = model(frames=frames, input_ids=input_ids,
                      attention_mask=attn_mask, loss_mask=loss_mask, mode="coarse_fine")

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure
    print(f"  Measuring ({MEASURE_STEPS} steps)...")
    torch.cuda.synchronize()
    t0 = time.time()

    total_samples = 0
    for step in range(MEASURE_STEPS):
        frames, input_ids, attn_mask, loss_mask = create_batch(batch_size)
        optimizer.zero_grad()

        result = model(frames=frames, input_ids=input_ids,
                      attention_mask=attn_mask, loss_mask=loss_mask, mode="coarse_fine")

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_samples += batch_size

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    throughput = total_samples / elapsed
    ms_per_step = elapsed / MEASURE_STEPS * 1000

    print(f"  RESULTS:")
    print(f"    Throughput: {throughput:.1f} samp/s")
    print(f"    Time/step:  {ms_per_step:.0f} ms")
    print(f"    Peak VRAM:  {peak_mem_mb:.0f} MB ({peak_mem_mb/1024:.1f} GB)")
    print(f"    Loss:       {result['loss'].item():.4f}")

    return {
        "name": name,
        "batch_size": batch_size,
        "throughput_sps": round(throughput, 1),
        "ms_per_step": round(ms_per_step, 0),
        "peak_vram_mb": round(peak_mem_mb, 0),
        "peak_vram_gb": round(peak_mem_mb / 1024, 2),
        "loss": round(result["loss"].item(), 4),
    }


def cleanup():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    results = []

    # ── Test 1: Baseline (grad_ckpt=True, no compile) ──
    print("\n\n" + "="*70)
    print("TEST 1: Baseline (grad_ckpt=True, no compile)")
    print("="*70)
    model = create_model(grad_ckpt=True)
    opt = create_optimizer(model)
    r = profile_config("baseline_grad_ckpt", model, opt, batch_size=8)
    results.append(r)
    del model, opt; cleanup()

    # ── Test 2: No gradient checkpointing ──
    print("\n\n" + "="*70)
    print("TEST 2: No gradient checkpointing")
    print("="*70)
    model = create_model(grad_ckpt=False)
    opt = create_optimizer(model)
    r = profile_config("no_grad_ckpt", model, opt, batch_size=8)
    results.append(r)
    del model, opt; cleanup()

    # ── Test 3: Liger Kernel FLCE ──
    if HAS_LIGER:
        print("\n\n" + "="*70)
        print("TEST 3: Liger Kernel FLCE (no grad ckpt)")
        print("="*70)
        model = create_model(grad_ckpt=False, use_fused_ce=True)
        opt = create_optimizer(model)
        r = profile_config("liger_flce", model, opt, batch_size=8)
        results.append(r)
        del model, opt; cleanup()

    # ── Test 4: torch.compile default (no grad ckpt) ──
    print("\n\n" + "="*70)
    print("TEST 4: torch.compile default (no grad ckpt)")
    print("="*70)
    model = create_model(grad_ckpt=False)
    opt = create_optimizer(model)
    r = profile_config("compile_default", model, opt, batch_size=8,
                       compile_model=True, compile_mode="default")
    results.append(r)
    del model, opt; cleanup()

    # ── Test 5: torch.compile max-autotune (no grad ckpt) ──
    print("\n\n" + "="*70)
    print("TEST 5: torch.compile max-autotune (no grad ckpt)")
    print("="*70)
    model = create_model(grad_ckpt=False)
    opt = create_optimizer(model)
    r = profile_config("compile_max_autotune", model, opt, batch_size=8,
                       compile_model=True, compile_mode="max-autotune")
    results.append(r)
    del model, opt; cleanup()

    # ── Test 6: Increased batch size (no grad ckpt) ──
    print("\n\n" + "="*70)
    print("TEST 6: batch_size=16 (no grad ckpt)")
    print("="*70)
    model = create_model(grad_ckpt=False)
    opt = create_optimizer(model)
    r = profile_config("no_ckpt_bs16", model, opt, batch_size=16)
    results.append(r)
    del model, opt; cleanup()

    # ── Test 7: batch_size=32 (no grad ckpt) ──
    print("\n\n" + "="*70)
    print("TEST 7: batch_size=32 (no grad ckpt)")
    print("="*70)
    model = create_model(grad_ckpt=False)
    opt = create_optimizer(model)
    try:
        r = profile_config("no_ckpt_bs32", model, opt, batch_size=32)
        results.append(r)
    except torch.cuda.OutOfMemoryError:
        print("  OOM at batch_size=32")
        results.append({"name": "no_ckpt_bs32", "error": "OOM"})
    del model, opt; cleanup()

    # ── Test 8: Combined best (Liger + no ckpt + compile + larger batch) ──
    if HAS_LIGER:
        print("\n\n" + "="*70)
        print("TEST 8: Combined (Liger + no ckpt + compile default + bs=16)")
        print("="*70)
        model = create_model(grad_ckpt=False, use_fused_ce=True)
        opt = create_optimizer(model)
        r = profile_config("combined_best_bs16", model, opt, batch_size=16,
                           compile_model=True, compile_mode="default")
        results.append(r)
        del model, opt; cleanup()

        print("\n\n" + "="*70)
        print("TEST 9: Combined (Liger + no ckpt + compile default + bs=32)")
        print("="*70)
        model = create_model(grad_ckpt=False, use_fused_ce=True)
        opt = create_optimizer(model)
        try:
            r = profile_config("combined_best_bs32", model, opt, batch_size=32,
                               compile_model=True, compile_mode="default")
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print("  OOM at batch_size=32 with combined optimizations")
            results.append({"name": "combined_best_bs32", "error": "OOM"})
        del model, opt; cleanup()

    # ── Summary ──
    print("\n\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Throughput':>12} {'ms/step':>10} {'Peak VRAM':>12}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'OOM':>12}")
        else:
            print(f"{r['name']:<30} {r['throughput_sps']:>10.1f}/s {r['ms_per_step']:>8.0f}ms {r['peak_vram_gb']:>10.1f}GB")

    # Save results
    out_path = "/workspace/optimization_profile_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
