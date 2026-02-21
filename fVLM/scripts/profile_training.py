#!/usr/bin/env python
"""
Profile fVLM 1.7B training step to identify bottlenecks.

Breaks down time into:
  1. Data loading (CPU→GPU transfer)
  2. DINO patch encoding
  3. Coarse query attend (static query → all frames)
  4. Coarse LLM forward (backbone only, for dynamic queries)
  5. Fine query attend (shifted queries → all frames)
  6. Fine LLM forward (backbone + lm_head → logits)
  7. Loss computation
  8. Backward pass
  9. Optimizer step

Uses CUDA events for precise GPU timing.
"""

import os
import sys
import time
import gc

import torch
import torch.nn.functional as F
import yaml

# Ensure release/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TMPDIR", "/workspace/tmp")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def cuda_timer():
    """Context manager for precise GPU timing using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def profile_forward_backward(model, batch, device, amp_dtype, n_warmup=3, n_measure=10):
    """Profile individual components of the forward+backward pass."""
    raw = model
    B, T = batch["frames"].shape[:2]
    S = batch["input_ids"].shape[1]

    results = {}

    # Helper to time a function
    def time_fn(name, fn, n_warmup=n_warmup, n_measure=n_measure):
        # Warmup
        for _ in range(n_warmup):
            fn()
            torch.cuda.synchronize()

        times = []
        for _ in range(n_measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg = sum(times) / len(times)
        results[name] = {"avg_ms": avg, "min_ms": min(times), "max_ms": max(times)}
        return avg

    frames = batch["frames"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]
    frame_mask = batch.get("frame_mask")

    # ---- Profile DINO encoding ----
    def dino_encode():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return raw._encode_all_frames(frames, frame_mask)

    time_fn("1_dino_encode", dino_encode)

    # Get cached results for subsequent profiling
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        kv_cache, patch_features, mask_flat = raw._encode_all_frames(frames, frame_mask)

    # ---- Profile coarse query attend ----
    q_static = raw.q_static.expand(B, -1)

    def coarse_query():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return raw._query_all_frames(q_static, kv_cache, B, T, mask_flat, patch_features)

    time_fn("2_coarse_query_attend", coarse_query)

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        z_coarse = raw._query_all_frames(q_static, kv_cache, B, T, mask_flat, patch_features)
        z_coarse_llm = raw._project_visual(z_coarse)

    # ---- Profile visual projection ----
    def project_visual():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return raw._project_visual(z_coarse)

    time_fn("3_project_visual", project_visual)

    # ---- Profile text embedding ----
    def text_embed():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return raw._embed_text(input_ids)

    time_fn("4_text_embed", text_embed)

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        text_embeds = raw._embed_text(input_ids)

    # ---- Profile coarse LLM forward ----
    def coarse_llm():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            seq_coarse = torch.cat([z_coarse_llm, text_embeds], dim=1)
            return raw.llm.model(inputs_embeds=seq_coarse)

    time_fn("5_coarse_llm_backbone", coarse_llm)

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        seq_coarse = torch.cat([z_coarse_llm, text_embeds], dim=1)
        out_coarse = raw.llm.model(inputs_embeds=seq_coarse)
        h_coarse = out_coarse.last_hidden_state

    # ---- Profile query generation ----
    def gen_queries():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            h_vis = h_coarse[:, :T, :]
            queries = raw.llm_to_query(h_vis)
            q_init = raw.q_init.expand(B, 1, -1)
            return torch.cat([q_init, queries[:, :-1]], dim=1)

    time_fn("6_query_generation", gen_queries)

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        h_vis = h_coarse[:, :T, :]
        queries = raw.llm_to_query(h_vis)
        q_init = raw.q_init.expand(B, 1, -1)
        shifted_queries = torch.cat([q_init, queries[:, :-1]], dim=1)

    # ---- Profile fine query attend ----
    def fine_query():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return raw._query_all_frames_batched(shifted_queries, kv_cache, B, T, mask_flat, patch_features)

    time_fn("7_fine_query_attend", fine_query)

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        z_fine = raw._query_all_frames_batched(shifted_queries, kv_cache, B, T, mask_flat, patch_features)
        z_fine_llm = raw._project_visual(z_fine)

    # ---- Profile fine LLM forward ----
    def fine_llm():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            seq_fine = torch.cat([z_fine_llm, text_embeds], dim=1)
            out_fine = raw.llm.model(inputs_embeds=seq_fine)
            return raw.llm.lm_head(out_fine.last_hidden_state)

    time_fn("8_fine_llm_fwd_+_lmhead", fine_llm)

    # ---- Profile full forward+backward ----
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)

    def full_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(
                frames=frames,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                frame_mask=frame_mask,
                mode="coarse_fine",
            )
            loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    time_fn("9_full_train_step", full_step, n_warmup=2, n_measure=5)

    # ---- Profile just backward ----
    def fwd_bwd():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(
                frames=frames,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                frame_mask=frame_mask,
                mode="coarse_fine",
            )
            loss = outputs["loss"]
        loss.backward()

    time_fn("9a_fwd_+_bwd_only", fwd_bwd, n_warmup=2, n_measure=5)

    return results


def main():
    device = torch.device("cuda")

    # Load config
    config_path = "/workspace/workdir/nanoSmol/fVLM/release/configs/final/stage1_1.7B.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("fVLM 1.7B Training Profiler")
    print("=" * 70)

    # Build model
    from release.model import FoveatedVLM
    model = FoveatedVLM(
        llm_name=cfg["model"]["llm"],
        dino_name=cfg["model"]["dino"],
        query_dim=cfg["model"].get("query_dim", 384),
        visual_scale=cfg["model"].get("visual_scale", 0.14),
        lambda_coarse=cfg["model"].get("lambda_coarse", 0.0),
        deep_query=cfg["model"].get("deep_query", True),
    ).to(device)

    if cfg["model"].get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    # channels_last for DINO conv
    model.encoder.dino = model.encoder.dino.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print(f"Gradient checkpointing: {cfg['model'].get('gradient_checkpointing', False)}")

    # Create synthetic batch (faster than loading real data)
    bs = cfg["training"]["batch_size"]
    T = 8   # typical image batch (replicated to 8 frames)
    S = 256  # typical text length

    print(f"\nSynthetic batch: B={bs}, T={T}, S={S}")

    batch = {
        "frames": torch.randn(bs, T, 3, 224, 224, device=device),
        "input_ids": torch.randint(0, 49152, (bs, S), device=device),
        "attention_mask": torch.ones(bs, S, device=device),
        "loss_mask": torch.ones(bs, S, device=device),
    }

    amp_dtype = torch.bfloat16

    # Profile
    print("\nProfiling (3 warmup, 10 measure each)...")
    print("-" * 70)

    results = profile_forward_backward(model, batch, device, amp_dtype)

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Component':<35} {'Avg (ms)':>10} {'Min':>10} {'Max':>10} {'%':>7}")
    print("-" * 70)

    full_step_ms = results.get("9_full_train_step", {}).get("avg_ms", 1)

    for name, stats in sorted(results.items()):
        pct = stats["avg_ms"] / full_step_ms * 100 if "9_" not in name else 100.0
        print(f"  {name:<33} {stats['avg_ms']:>10.1f} {stats['min_ms']:>10.1f} {stats['max_ms']:>10.1f} {pct:>6.1f}%")

    # Compute derived metrics
    print("\n" + "=" * 70)
    print("Derived Metrics")
    print("-" * 70)

    fwd_only = sum(
        results[k]["avg_ms"] for k in results
        if k.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_"))
    )
    full = results["9_full_train_step"]["avg_ms"]
    fwd_bwd = results["9a_fwd_+_bwd_only"]["avg_ms"]
    bwd_only = fwd_bwd - fwd_only
    optim_only = full - fwd_bwd

    print(f"  Forward pass (sum of components): {fwd_only:.1f} ms")
    print(f"  Backward pass (estimated):        {bwd_only:.1f} ms")
    print(f"  Optimizer step (estimated):        {optim_only:.1f} ms")
    print(f"  Throughput:                        {bs * 1000 / full:.1f} samp/s")
    print(f"  DINO encoding % of forward:       {results['1_dino_encode']['avg_ms']/fwd_only*100:.1f}%")

    coarse_llm_pct = results["5_coarse_llm_backbone"]["avg_ms"] / fwd_only * 100
    fine_llm_pct = results["8_fine_llm_fwd_+_lmhead"]["avg_ms"] / fwd_only * 100
    print(f"  Coarse LLM % of forward:          {coarse_llm_pct:.1f}%")
    print(f"  Fine LLM % of forward:            {fine_llm_pct:.1f}%")
    print(f"  LLM total (coarse+fine):          {coarse_llm_pct + fine_llm_pct:.1f}%")

    # Memory report
    print(f"\n  GPU memory allocated:             {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"  GPU memory reserved:              {torch.cuda.max_memory_reserved()/1e9:.1f} GB")

    # Now profile with different T values
    print("\n" + "=" * 70)
    print("Frame count sweep (T=1,4,8,16,32,64)")
    print("-" * 70)

    for T_test in [1, 4, 8, 16, 32]:
        torch.cuda.reset_peak_memory_stats()
        batch_test = {
            "frames": torch.randn(bs, T_test, 3, 224, 224, device=device),
            "input_ids": torch.randint(0, 49152, (bs, S), device=device),
            "attention_mask": torch.ones(bs, S, device=device),
            "loss_mask": torch.ones(bs, S, device=device),
        }

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)

        # Warmup
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(frames=batch_test["frames"], input_ids=batch_test["input_ids"],
                           attention_mask=batch_test["attention_mask"], loss_mask=batch_test["loss_mask"],
                           mode="coarse_fine")
            out["loss"].backward()
            optimizer.step()
            torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            optimizer.zero_grad(set_to_none=True)
            start.record()
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(frames=batch_test["frames"], input_ids=batch_test["input_ids"],
                           attention_mask=batch_test["attention_mask"], loss_mask=batch_test["loss_mask"],
                           mode="coarse_fine")
            out["loss"].backward()
            optimizer.step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated() / 1e9
        sps = bs * 1000 / avg
        print(f"  T={T_test:3d}: {avg:8.1f} ms/step, {sps:6.1f} samp/s, {peak:5.1f} GB peak")

        del batch_test, optimizer
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Profile complete.")


if __name__ == "__main__":
    main()
