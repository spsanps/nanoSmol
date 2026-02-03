# Scaling Law Study: Foveated VLM vs Baseline VLM

## Overview

Systematic comparison of foveated and baseline architectures across:
- Multiple training durations (100, 300, 1K, 3K, 10K steps)
- Compute budgets (FLOPs)
- Token efficiency

## Research Questions

1. **Scaling behavior**: How does loss scale with compute for each architecture?
2. **Crossover point**: At what compute budget does foveated beat baseline?
3. **Token efficiency**: How does quality/token scale?
4. **Empirical laws**: Can we derive Chinchilla-style scaling laws?

## Experiment Matrix

| Architecture | Steps | Frame Size | Fine Iters | Visual Tok/Frame |
|--------------|-------|------------|------------|------------------|
| Foveated-Opt | 100, 300, 1K, 3K | 224 | 1 | 1 |
| Baseline | 100, 300, 1K, 3K | 224 | N/A | 16 |
| Foveated-Orig | 100, 300, 1K, 3K | 256 | 2 | 1 |

## Metrics Collected

Per checkpoint:
- Validation loss (caption cross-entropy)
- Perplexity
- Training FLOPs (cumulative)
- Inference FLOPs (per sample)
- Visual tokens per video
- Wall-clock training time

## Expected Outputs

1. `data/scaling_data.csv` - Raw metrics for all experiments
2. `plots/loss_vs_flops.png` - Scaling curves
3. `plots/loss_vs_steps.png` - Training curves
4. `plots/iso_loss_curves.png` - Compute required for same quality
5. `plots/efficiency_ratio.png` - Quality per FLOP over training
6. `results/scaling_laws.json` - Fitted parameters
7. `results/analysis_report.md` - Summary and conclusions

## Running the Study

```bash
# 1. Run all training experiments
python scripts/run_scaling_experiments.py

# 2. Evaluate all checkpoints
python scripts/evaluate_all_checkpoints.py

# 3. Generate analysis and plots
python scripts/analyze_scaling_laws.py
```
