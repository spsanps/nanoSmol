# DINO Patch Prediction Research

**Status:** Exploratory / On Hold
**Date:** 2026-01-19

## Summary

Explored replacing VAE latent prediction with DINO patch prediction as the training target.

## Motivation

Current system predicts VAE latents (4×32×32 = 4,096 values). The hypothesis was that predicting DINO patches directly might:
1. Better align with the foveated attention mechanism (patches are spatial)
2. Remove VAE as a dependency
3. Allow caching DINO patches for faster training

## Key Findings

### Experiment 1: Patch Similarity Analysis (100 videos)

| Gap | Time | Cosine Similarity | MSE |
|-----|------|-------------------|-----|
| 1 frame | 0.125s | 0.933 | 0.81 |
| 8 frames | 1.0s | 0.829 | 2.06 |
| 12 frames | 1.5s | 0.804 | 2.38 |

**Finding:** Patches are highly correlated at short gaps, diverge at longer gaps.

### Experiment 2: Prediction Baselines (gap=8)

| Model | MSE | vs Copy |
|-------|-----|---------|
| Copy baseline | 2.16 | - |
| Linear | 3.55 | -64% worse |
| MLP | 4.02 | -86% worse |

**Finding:** Naive prediction from single frame is hard. Models diverge from targets.

### Experiment 3: Gap Comparison

| Gap | Time | Copy MSE | Model MSE | Improvement |
|-----|------|----------|-----------|-------------|
| 2 | 0.25s | 1.32 | 1.32 | -0.3% |
| 4 | 0.5s | 1.79 | 1.68 | +5.9% |
| 8 | 1.0s | 2.26 | 2.03 | +9.9% |
| 12 | 1.5s | 2.59 | 2.29 | +11.8% |

**Finding:** Improvement increases with temporal gap. At 1.5s, model beats copy by 12%.

## Technical Notes

- DINOv2-small with 256×256 input produces 325 tokens (1 CLS + 18×18 patches), not 257
- Patch dimension: 384
- Storage estimate: ~400GB for 100K videos (feasible)

## Blocking Issue Identified

During discussion, identified a more fundamental issue with the **FiLM prediction head**:
- Current FiLM broadcasts SAME gamma/beta across all spatial positions
- LLM has no per-region control over prediction
- This affects BOTH current VAE prediction AND proposed patch prediction

**Decision:** Before changing to DINO patches, first experiment with non-FiLM prediction heads on existing VAE setup. This isolates the architectural choice from the target representation choice.

## Next Steps

1. ~~Continue with patch prediction~~ ON HOLD
2. First: Experiment with alternative prediction heads (no FiLM) for VAE latents
3. Then: Revisit patch prediction with better prediction architecture

## Files

- `01b_patch_analysis_local.py` - Patch similarity analysis
- `02_prediction_baseline.py` - Prediction baselines
- `03_gap_comparison.py` - Gap comparison
- `results/` - Experiment outputs and plots
