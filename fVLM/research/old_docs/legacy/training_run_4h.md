# 4-Hour SmolVLM2 Training Run

**Date**: 2026-01-02
**Model**: SmolVLM2-256M-Video-Instruct with VAE decoder head
**Task**: Next-frame latent prediction

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | HuggingFaceTB/SmolVLM2-256M-Video-Instruct |
| Total params | 345.5M |
| Trainable params | 175.4M |
| Batch size | 8 |
| Context frames | 4 |
| Learning rate | 1e-4 |
| Max hours | 4 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |

## Dataset

| Source | Samples |
|--------|---------|
| data/frames_latents | 813 |
| data/latents | 813 |
| data/latents_phase2 | 817 |
| data/webvid/latents | 150 |
| data/webvid_large/latents | 6,431 |
| data/webvid_test/latents | 50 |
| **Total** | **9,074** |

## Training Metrics

- **Throughput**: ~13.5 samples/sec
- **Memory**: 9.5GB peak (RTX 4090, 24GB)
- **Initial loss**: 0.97
- **Loss after 4 min**: 0.77 (20% reduction)

## Estimated 4-Hour Results

- Steps: ~24,000 batches
- Samples processed: ~192,000
- Epochs: ~21 passes through dataset
- Expected final loss: ~0.3-0.5 (based on learning curve)

## Wandb Tracking

Project: smolvlm-video
Run: smolvlm_4h_fast
URL: https://wandb.ai/sanjayanps/smolvlm-video/runs/epjb5tk0

## Architecture Notes

The SmolVLM2 model processes:
1. Context frames (4 x 256x256) decoded from VAE latents
2. Through SigLIP vision encoder
3. Through language model (576 hidden dim, 30 layers)
4. VAE decoder head predicts next frame latent (4x32x32)

Loss: MSE between predicted and target VAE latents

## Comparison to SmolVLM2 Training

| Metric | Our Run | SmolVLM2 (HuggingFace) |
|--------|---------|------------------------|
| Training samples | ~192K (4h) | 3.3M |
| Model size | 345M | 256M-2B range |
| Epochs | ~21 | Not specified |
| Training objective | Latent prediction | Video QA |

While our sample count is ~1.7% of SmolVLM2's full training, we're doing multiple passes through our data with a focused prediction objective.
