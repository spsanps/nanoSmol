# Efficient JOINT Training with Precomputed DINO Features

## Overview

This document describes the efficient JOINT training pipeline for the Foveated VLM using precomputed DINO features. This approach skips the DINO encoder during training while preserving proper joint training with caption + reconstruction loss.

## CRITICAL: Why Joint Training is Essential

**Reconstruction-only training does NOT work.** Here's why:

### The Problem with Reconstruction-Only

When training with only reconstruction loss (predicting next-frame VAE latents):
- The model learns to compress visual features
- There's no signal guiding WHERE to attend
- The coarse and fine paths produce nearly identical results (ratio ≈ 1.0)
- No semantic understanding develops

### How Joint Training Fixes This

Joint training adds **autoregressive caption generation** after video tokens:

```
Sequence: [mode_token, z_1, z_2, ..., z_T, caption_1, caption_2, ...]
                ↑                            ↑
         visual features              predict these autoregressively
```

The caption loss teaches the model:
1. **WHAT to look for**: Semantically important content
2. **WHERE to attend**: Regions relevant for describing the video
3. **Temporal reasoning**: How frame content relates to language

### The Combined Objective

```python
loss = loss_caption + λ_recon * loss_reconstruction
```

- **Caption loss (primary)**: Cross-entropy on generated text
- **Reconstruction loss (secondary)**: MSE on VAE latents
- λ_recon = 0.1 (caption loss dominates by design)

### Success Metric

**cap_fine < cap_coarse** (fine path produces better captions)

This validates that autoregressive queries (informed by previous frames) produce better features than static queries.

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
PYTHONUNBUFFERED=1 python fVLM/scripts/train_dino_efficient.py --max_hours 2.0
```

### Key Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 32 | Joint training needs more VRAM |
| grad_accum | 2 | Effective batch = 64 |
| num_frames | 16 | Sample from 24 available |
| learning_rate | 1e-4 | Higher for large batch |
| fine_iterations | 2 | coarse → fine₁ → fine₂ |
| max_hours | 2.0 | Resumable chunks |
| lambda_recon | 0.1 | Caption loss is primary |
| max_caption_tokens | 64 | Caption truncation length |

### Throughput Benchmarks (RTX 4090)

**Reconstruction-only (deprecated):**
| Batch Size | Samples/sec | VRAM |
|------------|-------------|------|
| 64 | 205 | 14.1 GB |

**Joint training (current):**
| Batch Size | Samples/sec | VRAM |
|------------|-------------|------|
| 32 | ~100 | ~16 GB |

Joint training is slower due to caption embedding and longer sequences, but provides essential semantic signal.

### Training Time Estimates

- **1 epoch** (100K samples): ~16 minutes at batch=32
- **2 hours** of training: ~7 epochs, ~10,000 steps
- **Full convergence**: TBD (monitor cap_ratio metric)

## Architecture

### EfficientFoveatedModel (Joint Training)

Uses precomputed DINO features with caption + reconstruction loss:

```
Input: dino_features [B, T, N, D]
       latents [B, T, 4, 32, 32]
       caption_ids [B, L]

Pass 1 (Coarse):
  - Static query q_static attends to all patches
  - Build sequence: [coarse_token, z_1...z_T, caption_embeds]
  - LLM processes → caption loss + reconstruction loss

Pass 2+ (Fine, repeated for fine_iterations):
  - Dynamic queries from previous pass (autoregressive)
  - Build sequence: [fine_token, z_fine_1...z_fine_T, caption_embeds]
  - LLM processes → caption loss + reconstruction loss
```

### Key Differences from Full Model

1. **No DINO encoder** - uses precomputed patch features
2. **Shallow attention only** - single cross-attention per frame
3. **No deep query injection** - would require cached K,V per layer
4. **No DINO fine-tuning** - features are frozen
5. **Joint loss** - caption (primary) + reconstruction (secondary)

## Success Metric

**Primary goal:** `cap_fine < cap_coarse`

- Caption ratio > 1.0 means fine pass produces better captions
- This validates that autoregressive queries help
- 5-15% improvement = PoC successful
- >15% improvement = very promising

**Why caption loss, not reconstruction?**
- Reconstruction measures visual compression quality
- Caption loss measures semantic understanding
- The foveated hypothesis is about WHAT to attend to, which is semantic

## Monitoring

### wandb Dashboard

View at: https://wandb.ai/sanjayanps/foveated-vlm-efficient

Key metrics:
- `cap_fine`: Fine pass caption loss (cross-entropy)
- `cap_coarse`: Coarse pass caption loss
- `cap_ratio`: coarse/fine (want > 1.0)
- `cap_improvement_pct`: (ratio - 1) × 100
- `rec_fine`: Fine pass reconstruction loss (MSE)
- `rec_coarse`: Coarse pass reconstruction loss

**Primary metric: `cap_ratio`** - validates foveated attention hypothesis

### Checkpoints

Saved to `outputs/dino_efficient_joint/`:
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
| `scripts/train_dino_efficient.py` | **Joint** training script (caption + reconstruction) |
| `src/model/prediction.py` | Prediction head (FiLM conditioning) |
| `outputs/dino_efficient_joint/` | Checkpoints and logs |

**Note:** The training script was updated from reconstruction-only to joint training because reconstruction-only provides no semantic signal (ratio stays at ~1.0).

## Next Steps After Training

1. **Evaluate ratio**: Check if loss_fine < loss_coarse consistently
2. **Visualize attention**: See if queries focus on different regions
3. **Add captioning**: Extend to joint caption+reconstruction if ratio validates
4. **Scale up**: Train longer or with more data if promising
