# Efficient Training with Precomputed DINO Features

## Overview

This document describes the efficient training pipeline for the Foveated VLM using precomputed DINO features. This approach skips the DINO encoder during training, achieving ~200 samples/sec throughput.

## Data Preparation

### Precomputed Data Format

Each video is saved as a `.pt` file containing:
```python
{
    "dino_features": tensor[T, N, D],  # [24, 325, 384] DINO patch features
    "latents": tensor[T, 4, 32, 32],   # VAE latents (prediction target)
    "caption": str                      # Video caption
}
```

- **T**: Number of frames (24)
- **N**: Number of DINO patches + CLS token (325 = 18×18 + 1 for 256×256 images)
- **D**: DINO embedding dimension (384 for ViT-S)

### Precompute Script

```bash
python fVLM/scripts/precompute_dino.py
```

**Configuration:**
- Duration filter: 20-60s videos
- Frames: 24 per video
- Target: 100K videos
- Storage: ~6 MB/video = ~600 GB total
- Time: ~5-6 hours at ~5.5 videos/sec

## Training

### Quick Start

```bash
python fVLM/scripts/train_dino_efficient.py --max_hours 2.0
```

### Key Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 64 | Optimal for RTX 4090 |
| grad_accum | 1 | Effective batch = 64 |
| num_frames | 16 | Sample from 24 available |
| learning_rate | 1e-4 | Higher for large batch |
| fine_iterations | 2 | coarse → fine₁ → fine₂ |
| max_hours | 2.0 | Resumable chunks |

### Throughput Benchmarks (RTX 4090)

| Batch Size | Samples/sec | VRAM |
|------------|-------------|------|
| 8 | 31 | 3.9 GB |
| 16 | 70 | 4.1 GB |
| 32 | 128 | 7.5 GB |
| 48 | 171 | 10.8 GB |
| 64 | 205 | 14.1 GB |
| 80 | 215 | 17.4 GB |

### Training Time Estimates

- **1 epoch** (100K samples): ~8 minutes at batch=64
- **2 hours** of training: ~15 epochs, ~22,000 steps
- **Full convergence**: TBD (monitor ratio metric)

## Architecture

### EfficientFoveatedModel

Simplified model that uses precomputed DINO features:

```
Input: dino_features [B, T, N, D]
       latents [B, T, 4, 32, 32]

Pass 1 (Coarse):
  - Static query q_static attends to all patches
  - LLM processes coarse features → predicts queries
  - Prediction head predicts next-frame latents

Pass 2+ (Fine, repeated for fine_iterations):
  - Dynamic queries from previous pass
  - Autoregressive: q_t depends on LLM output from t-1
  - Final prediction head output → loss_fine
```

### Key Differences from Full Model

1. **No DINO encoder** - uses precomputed patch features
2. **Shallow attention only** - single cross-attention per frame
3. **No deep query injection** - would require cached K,V per layer
4. **No DINO fine-tuning** - features are frozen

## Success Metric

**Primary goal:** `loss_fine < loss_coarse`

- Ratio > 1.0 means fine pass is better (hypothesis validated)
- 5-15% improvement = PoC successful
- >15% improvement = very promising

## Monitoring

### wandb Dashboard

View at: https://wandb.ai/sanjayanps/foveated-vlm-efficient

Key metrics:
- `loss_fine`: Fine pass reconstruction loss
- `loss_coarse`: Coarse pass reconstruction loss
- `ratio`: coarse/fine (want > 1.0)
- `improvement_pct`: (ratio - 1) × 100

### Checkpoints

Saved to `outputs/dino_efficient/`:
- `checkpoint_latest.pt`: Most recent
- `checkpoint_step{N}.pt`: Periodic saves
- Auto-resumes from latest on restart

## Resuming Training

Training automatically resumes from checkpoint:

```bash
# Continues from outputs/dino_efficient/checkpoint_latest.pt
python fVLM/scripts/train_dino_efficient.py --max_hours 2.0

# Or specify checkpoint explicitly
python fVLM/scripts/train_dino_efficient.py --resume path/to/checkpoint.pt
```

## Troubleshooting

### Output not appearing
Use unbuffered Python:
```bash
PYTHONUNBUFFERED=1 python fVLM/scripts/train_dino_efficient.py
```

### DataLoader hangs on WSL
Set `num_workers=0` in config (already done).

### OOM errors
Reduce batch_size. Batch 48 uses 10.8 GB, leaving headroom.

## Files

| File | Purpose |
|------|---------|
| `scripts/precompute_dino.py` | Precompute DINO features + VAE latents |
| `scripts/train_dino_efficient.py` | Efficient training script |
| `src/model/prediction.py` | Prediction head (FiLM conditioning) |
| `outputs/dino_efficient/` | Checkpoints and logs |

## Next Steps After Training

1. **Evaluate ratio**: Check if loss_fine < loss_coarse consistently
2. **Visualize attention**: See if queries focus on different regions
3. **Add captioning**: Extend to joint caption+reconstruction if ratio validates
4. **Scale up**: Train longer or with more data if promising
