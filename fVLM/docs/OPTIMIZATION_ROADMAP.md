# Optimization Roadmap — Codex Consultation Results

**Date**: 2026-02-21
**Current baseline**: 27 samp/s on A100 80GB (135M model, bf16)
**Current VRAM**: 36.5 GB used / 80 GB total (43.5 GB wasted)
**Target**: 95%+ GPU utilization, 40-55 samp/s

## All Recommendations (Priority Order)

### 1. Remove Gradient Checkpointing
- **Impact**: +20-30% throughput
- **Effort**: Config change (`gradient_checkpointing: false`)
- **Risk**: Low
- **Rationale**: Grad checkpointing recomputes activations during backward pass (20-30% compute overhead). Currently using only 36.5GB of 80GB — removing it costs ~15GB memory, pushing to ~48-52GB. Still well within 80GB budget.
- **Status**: DONE in configs (stage1_1.7B.yaml, stage2_1.7B.yaml). Stage3 DPO keeps it (reference model doubles memory).

### 2. Increase Batch Size / max_total_frames
- **Impact**: +10-20% throughput
- **Effort**: Config change
- **Risk**: Medium (OOM possible, needs profiling)
- **Rationale**: After removing grad checkpointing, ~28-32GB headroom remains. Increase `max_total_frames` from 512 to 768-1024 and `max_batch_size` from 64 to 96-128 to fill GPU.
- **Status**: DONE in configs (768 frames, 96 max_batch).
- **TO TEST**: Profile to find actual OOM boundary.

### 3. Liger Kernel FusedLinearCrossEntropyLoss
- **Impact**: +5-15% throughput, >4x memory reduction on CE loss
- **Effort**: Medium (code change in foveated_vlm.py)
- **Risk**: Low
- **Rationale**: Current `_ce_loss()` does `lm_head(h_text)` creating `[B, S, 49152]` logits tensor, then `cross_entropy()` consumes it. Liger FLCE fuses lm_head + CE in one Triton kernel — **never materializes the logits tensor**. For 1.7B with vocab 49152, this saves ~2GB per forward pass.
- **Integration**:
  ```python
  # pip install liger-kernel
  from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
  # In forward: skip lm_head, pass hidden states + weight directly
  loss = LigerFusedLinearCrossEntropyLoss(ignore_index=pad_id)(
      hidden_states.reshape(-1, dim), lm_head.weight, labels.reshape(-1)
  )
  ```
- **Status**: DONE — integrated as `use_fused_ce` flag in foveated_vlm.py and configs.
- **TO TEST**: Verify numerical equivalence with standard CE. Profile memory savings.

### 4. torch.compile max-autotune Mode
- **Impact**: +3-8% throughput
- **Effort**: Config change (`compile_mode: max-autotune`)
- **Risk**: Low (longer first-iteration compile time)
- **Rationale**: `max-autotune` benchmarks multiple kernel implementations and picks the fastest. One-time compilation cost amortized over many training steps.
- **Status**: NOT YET APPLIED — configs currently use `default` mode.
- **TO TEST**: Compare `default` vs `max-autotune` vs `reduce-overhead` in profiling.

### 5. fullgraph=True for Encoder Compile
- **Impact**: +2-5% throughput
- **Effort**: Code change (1 line in train.py)
- **Risk**: Low (may fail if graph breaks exist)
- **Rationale**: DINO encoder has fixed-shape inputs (224x224) and no Python-level branching. `fullgraph=True` allows the compiler to optimize the entire graph without fallbacks.
- **Code**:
  ```python
  raw_model.encoder = torch.compile(
      raw_model.encoder, mode="reduce-overhead", dynamic=False, fullgraph=True
  )
  ```
- **Status**: NOT YET APPLIED.
- **TO TEST**: Check if encoder compiles with fullgraph=True (no graph breaks).

### 6. Shape Padding in Compile Options
- **Impact**: +1-3% throughput
- **Effort**: Code change (1 line)
- **Risk**: Low
- **Rationale**: Pads tensors to better alignment boundaries for Tensor Core utilization (A100 prefers dimensions divisible by 8/16).
- **Code**:
  ```python
  raw_model.llm = torch.compile(
      raw_model.llm, mode="max-autotune", dynamic=True,
      options={"shape_padding": True}
  )
  ```
- **Status**: NOT YET APPLIED.
- **TO TEST**: Profile with and without.

### 7. CUDA Stream Overlap for DINO Encoding
- **Impact**: +5-10% throughput
- **Effort**: High (architecture change)
- **Risk**: Medium
- **Rationale**: DINO encoder is frozen and produces KV caches. For the NEXT batch, DINO can run on a separate CUDA stream overlapping with the LLM backward pass of the CURRENT batch.
- **Caveat**: Codex noted "as batch size increases, pipelining benefit diminishes... at 85-90% GPU util, overhead may cause degradation." Deprioritized.
- **Status**: NOT APPLIED — high effort, uncertain benefit.
- **TO TEST**: Only after items 1-5 are profiled.

### 8. Sequence Packing
- **Impact**: +10-20% throughput
- **Effort**: Very High (major refactor)
- **Risk**: Medium
- **Rationale**: Current padding wastes compute. Packing multiple samples into one sequence with block-diagonal attention masks eliminates padding waste.
- **Challenges for fVLM**:
  - Two LLM passes (coarse + fine) need separate packing
  - Variable visual tokens (1-64) AND variable text (50-200)
  - Flash Attention 2.x `cu_seqlens` required for efficient masking
- **Status**: NOT APPLIED — too complex for now.
- **Note**: Dynamic batching already captures most visual-side packing benefit.

## Memory Budget Summary

| Component | Current (GB) | After Opts 1-3 (GB) |
|-----------|-------------|---------------------|
| Model params (bf16) | ~3.4 | ~3.4 |
| Optimizer states (fp32 AdamW) | ~13.6 | ~13.6 |
| Activations | ~12 (grad ckpt) | ~25 (no ckpt) |
| Logit tensor | ~2 | ~0.1 (FLCE) |
| DINO KV cache | ~5 | ~5 |
| Headroom/fragmentation | ~0.5 | ~2 |
| **Total** | **~36.5** | **~49** |
| **Remaining** | **~43.5** | **~31** |

With ~31GB headroom after removing checkpointing + FLCE, batch sizes can roughly double.

## Profiling Plan

Script: `scripts/profile_optimizations.py`

Tests to run (sequentially, 135M model, synthetic data):

| Test | Config |
|------|--------|
| 1. Baseline | grad_ckpt=True, no compile, bs=8 |
| 2. No grad ckpt | grad_ckpt=False, no compile, bs=8 |
| 3. Liger FLCE | grad_ckpt=False, use_fused_ce=True, bs=8 |
| 4. Compile default | grad_ckpt=False, compile=default, bs=8 |
| 5. Compile max-autotune | grad_ckpt=False, compile=max-autotune, bs=8 |
| 6. Larger batch (bs=16) | grad_ckpt=False, bs=16 |
| 7. Larger batch (bs=32) | grad_ckpt=False, bs=32 |
| 8. Combined (bs=16) | Liger + no ckpt + compile + bs=16 |
| 9. Combined (bs=32) | Liger + no ckpt + compile + bs=32 |

Each test: 3 warmup + 10 measurement steps. Reports throughput (samp/s), ms/step, peak VRAM.

## Implementation Status

| # | Optimization | Code Done | Config Done | Profiled |
|---|-------------|-----------|-------------|----------|
| 1 | Remove grad ckpt | N/A | YES | NO |
| 2 | Increase batch/frames | N/A | YES (768/96) | NO |
| 3 | Liger FLCE | YES | YES | NO |
| 4 | max-autotune compile | NO | NO | NO |
| 5 | fullgraph encoder | NO | NO | NO |
| 6 | Shape padding | NO | NO | NO |
| 7 | CUDA stream overlap | NO | NO | NO |
| 8 | Sequence packing | NO | NO | NO |

## Sources (from Codex report)

- [Liger Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [torchtune + Liger Kernel (PyTorch Blog)](https://pytorch.org/blog/peak-performance-minimized-memory/)
- [State of torch.compile (August 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [NVIDIA NeMo Sequence Packing](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/sequence_packing.html)
- [Gradient Checkpointing Tradeoffs](https://buildai.substack.com/p/gradient-checkpointing-memory-compute)
- [CUDA Streams Pipelining](https://chaimrand.medium.com/pipelining-ai-ml-training-workloads-with-cuda-streams-bf5746449409)
