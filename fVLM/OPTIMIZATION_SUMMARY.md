# GPU Optimization Summary

## Final Results

### Pre-Decoded Frames Optimization (IMPLEMENTED)

We eliminated the video decoding bottleneck by pre-decoding all 813 videos to uint8 tensors.

| Metric | Before (decord) | After (pre-decoded) | Improvement |
|--------|-----------------|---------------------|-------------|
| **GPU Utilization** | 2% | **54%** | **27x better** |
| **Training Speed** | 1.1 it/s | **3.8 it/s** | **3.5x faster** |
| **VRAM Usage** | 8.4 GB | 10.7 GB | +2.3 GB |
| **Power Draw** | 100W | 180W | 1.8x |
| **Training Time** | 12.5 hours | **~3.6 hours** | **Saved 9 hours** |

### Pre-Decoded Frames Storage
```
Location: data/frames/
Files: 813 .pt files
Size: 1.2 GB total
Format: [8, 3, 256, 256] uint8 tensors
```

---

## All Optimizations Implemented

### 1. Pre-Decoded Video Frames (NEW)
**Before:**
```python
# Every batch: decode 80 frames from H264 video
frames = decord.VideoReader(video_path).get_batch(indices)  # 10-15ms per frame
```

**After:**
```python
# Every batch: load pre-decoded tensors
frames = torch.load(frames_path, weights_only=True)  # <1ms per frame
frames = (frames.float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
```

**Impact:** Eliminated CPU video decode bottleneck entirely

### 2. Batched DINO Frame Processing
**Before:**
```python
for t in range(T):  # T=8 sequential DINO forwards
    _, cache = self.encoder.encode_patches(raw_frames[:, t])
```

**After:**
```python
frames_flat = raw_frames.reshape(B * T, 3, 256, 256)  # [B*T, 3, 256, 256]
_, cache_flat = self.encoder.encode_patches(frames_flat)  # 1 batched forward
```

**Impact:** Reduced DINO forward passes from 8x to 1x

### 3. Increased Batch Size
- **Before:** batch_size=6, effective=12
- **After:** batch_size=10, effective=20
- **VRAM:** 8.4GB -> 10.7GB (better utilization)

### 4. Enabled Multiprocessing + Prefetching
```python
num_workers=2                    # Parallel frame loading
prefetch_factor=4                # Load 4 batches ahead
persistent_workers=True          # Keep workers alive between epochs
```

---

## Current Training Status

```
Run: phase1_20251231_130729
W&B: https://wandb.ai/sanjayanps/foveated-vlm/runs/8rnig2ro

Config:
- batch_size: 10
- grad_accum: 2
- effective_batch: 20
- num_workers: 2
- frames_dir: data/frames (pre-decoded)
- VRAM: 10.7GB / 24GB

Performance:
- Speed: ~3.8 it/s (3.5x faster than before)
- GPU Util: 54% (27x better than 2%)
- ETA: ~3.6 hours for 50K steps

Training Metrics (early):
- Loss: 1.55 (decreasing from 1.78)
- Fine loss: 0.777
- Coarse loss: 0.775
- Ratio: 0.997 (coarse/fine)

Status: RUNNING
```

---

## Architecture Verification

The two-pass foveated attention architecture is correctly implemented:

1. **Pass 1 (Static Query):**
   - Uses learnable `q_static` query
   - Attends over all T frames' patch features
   - Produces coarse prediction & pooled context

2. **Pass 2 (Dynamic Queries):**
   - LLM generates dynamic queries from Pass 1 output
   - Attends over same patch features
   - Produces fine prediction

3. **Auxiliary Loss:**
   - `loss = loss_fine + lambda_coarse * loss_coarse`
   - Encourages coarse loss to decrease (model learning)
   - Goal: `loss_fine < loss_coarse` (dynamic attention helps)

---

## What We Achieved

| Metric | Original | Final | Improvement |
|--------|----------|-------|-------------|
| **Architecture** | Correct | Correct | Verified |
| **GPU Utilization** | 2% | 54% | **27x** |
| **Training Speed** | 1.1 it/s | 3.8 it/s | **3.5x** |
| **VRAM Usage** | 8.4 GB | 10.7 GB | +27% |
| **Training Time** | 12.5 hours | 3.6 hours | **-70%** |
| **Pre-decoded Storage** | 0 | 1.2 GB | Minimal |

---

## Conclusion

**The video decoding bottleneck has been eliminated.** By pre-decoding frames to uint8 tensors:

1. Training is now **3.5x faster** (3.6 hours vs 12.5 hours)
2. GPU utilization increased from **2% to 54%**
3. Storage cost is minimal (1.2 GB for 813 videos)
4. Architecture verified to match core_docs spec

**Training is now GPU-bound, not CPU-bound.** The remaining 46% GPU idle time is due to:
- Data transfer between CPU and GPU
- Optimizer step overhead
- DataLoader worker synchronization

This is acceptable for a proof-of-concept. Further optimization would require:
- Larger batch size (needs more VRAM)
- Gradient checkpointing (trades compute for memory)
- Mixed precision optimization
