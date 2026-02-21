# Throughput Optimizations for fVLM Training

All optimizations applied to maximize GPU utilization during training.
Measured on RTX 5090 32GB with PyTorch 2.10.0 + CUDA 12.8.

## Summary Table

| Optimization | Category | Impact | Where |
|---|---|---|---|
| TF32 matmul precision | GPU compute | ~10-15% faster matmul | `train.py` line 29 |
| cuDNN benchmark mode | GPU compute | Auto-tunes DINO conv | `train.py` line 30 |
| cuDNN TF32 allow | GPU compute | TF32 for convolutions | `train.py` line 31 |
| SDPA attention | GPU compute | Fused attention kernel | `encoder.py` line 216 |
| torch.compile (encoder) | GPU compute | Eliminates Python overhead | `train.py` line 387 |
| torch.compile (LLM) | GPU compute | Eliminates Python overhead | `train.py` line 388 |
| torch.compile (projections) | GPU compute | Eliminates Python overhead | `train.py` lines 389-390 |
| Fused AdamW | Optimizer | Eliminates Python loop | `train.py` line 337 |
| BF16 autocast | Mixed precision | Half-precision compute | `train.py` line 361 |
| set_to_none=True | Memory | Faster zero_grad | `train.py` lines 434, 499 |
| GC disabled during training | CPU | Saves ~500ms/collection | `train.py` lines 436-438 |
| 12 DataLoader workers | Data pipeline | Saturates 16 vCPU | `train.py` line 301 override |
| prefetch_factor=8 | Data pipeline | Keeps GPU fed | `train.py` line 302 override |
| pin_memory=True | Data transfer | DMA CPU→GPU | `webdataset_loader.py` line 389 |
| persistent_workers=True | Data pipeline | No worker restart per epoch | `webdataset_loader.py` line 424 |
| non_blocking transfers | Data transfer | Async CPU→GPU | `train.py` line 451 |
| torchvision decode_jpeg | Data pipeline | 27% faster vs PIL | `webdataset_loader.py` line 53 |
| expandable_segments | Memory | Reduces fragmentation | `train.py` line 34 |
| no_sync() for grad_accum | DDP | Skips AllReduce mid-accum | `train.py` line 457 |
| Batched DINO encode | Architecture | Single DINO pass for all frames | `foveated_vlm.py` |
| Maximize batch size | GPU utilization | 20-40% | `train.py` `_maximize_batch_size()` |
| Bucketed padding | Shapes | 10-20% | `collate.py` frame/text buckets |
| channels_last (DINO) | Memory layout | 5-10% | `train.py` + `encoder.py` |
| Length-sorted batching | Padding waste | 5-15% | `webdataset_loader.py` sort buffer |

## Details

### 1. TF32 Precision (Ampere+ GPUs)
```python
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
```
TF32 uses 19-bit mantissa (vs FP32's 23-bit) on tensor cores. ~10-15% speedup
on matmul-heavy workloads with negligible precision loss. Critical for both
DINO (12 attention layers) and LLM forward/backward.

### 2. cuDNN Benchmark
Auto-tunes convolution algorithms for the given input sizes. DINO's patch
embedding is a single conv2d (14x14 kernel, stride 14), so the tuning cost
is amortized quickly across thousands of steps.

### 3. SDPA (Scaled Dot-Product Attention)
Both DINO query-attend and LLM self-attention use
`F.scaled_dot_product_attention()` which dispatches to:
- FlashAttention-2 on Ampere+ (A100, RTX 5090)
- Memory-efficient attention as fallback
No intermediate attention matrix allocation → O(1) memory per attention layer.

### 4. torch.compile
Compiles encoder, LLM, and projection layers separately (avoids graph breaks
at component boundaries). Uses `dynamic=True` for variable-length inputs.

**Important caveat:** Benchmarked on RTX 5090 with 135M model — compile
HURTS throughput (44.1 samp/s vs 46.5 without). The 135M model is too small
for compile overhead to pay off with variable-length video inputs. Dynamic
shapes cause frequent recompilation. **Only enable for 360M+ models** where
Python overhead is proportionally larger relative to compute.

### 5. Data Pipeline Optimization
```
num_workers=12    (up from 8)  — uses 12/16 available CPU cores
prefetch_factor=8 (up from 4)  — queues 8*12=96 batches ahead of GPU
pin_memory=True               — enables DMA for CPU→GPU transfer
persistent_workers=True       — avoids worker process restart overhead
```
Combined with torchvision's `decode_jpeg` (bypasses PIL/numpy), the data
pipeline keeps the GPU continuously fed.

### 6. Batched DINO Encoding
All frames in a batch are concatenated to [B*T, 3, 224, 224] and passed
through DINO in a single forward call. K/V are cached at every layer. This
is ~2.8x faster than per-frame encoding (measured in commit 5588aed).

## GPU Utilization Profile

Before optimizations (Phase 1a):
- 135M model: ~67 samp/s (frozen DINO), ~47 samp/s (unfrozen)
- GPU utilization: 70-85%, avg ~78%

After optimizations (Phase 1b):
- Expected: 85-95% GPU utilization with torch.compile + more workers
- 360M model expected higher utilization (more compute per step)

## Recommendations for Other GPUs

### A100 80GB (2x, DDP)
- num_workers: 8 per GPU (avoid CPU contention with DDP)
- batch_size: scale up 2-4x (more VRAM)
- torch.compile: use `mode="reduce-overhead"` for static shapes
- Enable gradient checkpointing for 1.7B model

### H100 / RTX 5090
- All optimizations above apply
- FP8 matmul available on H100 (torch.float8_e4m3fn) — not yet tested
- torch.compile with `mode="max-autotune"` for best kernel selection

### Multi-GPU Scaling
- no_sync() already implemented for grad_accum micro-steps
- WebDataset handles shard splitting across workers automatically
- Sync points: grad_norm clipping, optimizer step, eval
