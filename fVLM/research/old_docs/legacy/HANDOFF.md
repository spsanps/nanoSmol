# Handoff Note — Foveated VLM Scaling Experiments

**Date:** 2026-02-07
**Previous session:** Comprehensive scaling law experiments + parameter-corrected analysis

---

## Project Summary

Novel video VLM that processes frames with **1 token/frame** (foveated attention) vs baseline's 16 tokens/frame. The LLM generates queries that control WHERE to look in each frame via cross-attention with DINOv2 features.

**Core question:** How much quality do you lose for 16x fewer visual tokens?

**Answer:** 1.6-2.4% (parameter-corrected), shrinking with LLM scale. At 1.7B, likely ~1%.

---

## What Was Accomplished

### 1. Comprehensive Scaling Experiments (10/12 P1 runs complete)
- Script: `fVLM/research/scaling_laws/scripts/run_comprehensive_scaling.py`
- 3 model configs (S-S, M-S, S-B) x 2 frame counts (8F, 64F) x 2 architectures (foveated, baseline)
- 280 steps x EB=16 = 4,480 samples (< 1 epoch on 4,500 train)
- Eval every 14 steps = 20 data points per run
- Results: `/mnt/d/projects/fVLM/outputs/scaling_comprehensive/results.json` (205 data points)

### 2. Run Status

| Run | Status | Notes |
|-----|--------|-------|
| S-S_8f_fov | COMPLETE (20 pts) | |
| S-S_8f_bas | COMPLETE (20 pts) | |
| S-S_64f_fov | COMPLETE (20 pts) | |
| S-S_64f_bas | COMPLETE (20 pts) | |
| M-S_8f_fov | COMPLETE (20 pts) | |
| M-S_8f_bas | COMPLETE (20 pts) | |
| M-S_64f_fov | COMPLETE (20 pts) | |
| M-S_64f_bas | COMPLETE (20 pts) | |
| S-B_8f_fov | COMPLETE (20 pts) | |
| S-B_8f_bas | COMPLETE (20 pts) | |
| S-B_64f_fov | PARTIAL (5 pts, step 70) | CUDA error at 23.3/24.5GB |
| S-B_64f_bas | NOT RUN | Never started (S-B_64f_fov failed first) |

### 3. Scaling Law Plots (9 types)
- Script: `fVLM/research/scaling_laws/scripts/plot_scaling_fits.py`
- Output: `/mnt/d/projects/fVLM/outputs/scaling_comprehensive/scaling_analysis/`
- Plots: loss_vs_flops_fitted, training_curves, model_size_scaling, efficiency_frontier, smolvlm_extrapolation, token_efficiency, perplexity_vs_flops, iso_flop_curves, chinchilla_ratio

### 4. Parameter-Corrected Analysis (final analysis done)
Baseline has MORE params than foveated in every config pair (PixelShuffle proj vs query attn). Bidirectional interpolation corrects for this:

| Config | Fov Params | Bas Params | Raw Gap | True Architectural Gap |
|--------|-----------|-----------|---------|----------------------|
| S-S 64F | 159M | 172M | +2.81% | **+2.44%** |
| M-S 64F | 386M | 411M | +2.37% | **+1.60%** |

Method: Interpolate on each architecture's OWN loss-vs-params curve. Both directions agree (within 0.1%).
Gap shrinks with LLM scale. At 1.7B, likely ~1% or less.

---

## Key Results

1. **Foveated 1.6-2.4% worse** (param-corrected) but 1.1-3x fewer FLOPs, 16x fewer tokens
2. **LLM size >> DINO size**: 7% gain from LLM scaling vs <1% from bigger DINO
3. **Train/inference gap < 0.6%**: Parallel training approximation validated
4. **Massively data-starved**: D/N ~ 1e-5 vs Chinchilla optimal ~20. Power law fits unreliable
5. **64F worse than 8F** at this data scale (models can't learn temporal patterns from 5K samples)

---

## Pending Work (by priority)

1. **S-B_64f runs** — Failed CUDA OOM at 23.3GB. Could retry with gradient checkpointing or skip (low value — bigger DINO doesn't help).
2. **P2 experiments (B-L / 1.7B)** — 4 runs with SmolLM2-1.7B. Would validate "gap shrinks with scale" prediction. May need gradient checkpointing for 64F.
3. **Dataset mix recommendation** — User asked for "best fit for best loss with video dataset mix close to the full SmolVLM run". SmolVLM trained on ~4.4M samples; our experiments use only 5K. Analysis incomplete.

---

## File Locations

### Scripts
- `fVLM/research/scaling_laws/scripts/run_comprehensive_scaling.py` — Main experiment runner
- `fVLM/research/scaling_laws/scripts/plot_scaling_fits.py` — Analysis/plotting (9 plot types)
- `fVLM/research/scaling_laws/scripts/plot_scaling_laws.py` — Older plotting script
- `fVLM/research/scaling_laws/scripts/evaluate_all_configs.py` — Config evaluation
- `fVLM/scripts/precompute_8f.py` — 8F data precomputation

### Data (D drive — NOT in repo, may not transfer)
- `/mnt/d/projects/fVLM/data/webvid_8f_5k/` — 500 shards x 10 = 5000 samples, 8 frames 224x224
- `/mnt/d/projects/fVLM/data/webvid_64f_5k/` — 500 shards x 10 = 5000 samples, 64 frames 224x224
- `/mnt/d/projects/fVLM/data/frames_latents_sharded/` — 513 shards x 200 = 102,600 samples, 24 frames 256x256 (legacy)

### Outputs (D drive — NOT in repo, may not transfer)
- `/mnt/d/projects/fVLM/outputs/scaling_comprehensive/results.json` — 205 data points
- `/mnt/d/projects/fVLM/outputs/scaling_comprehensive/scaling_analysis/` — 9 PNG+PDF plots
- `/mnt/d/projects/fVLM/outputs/full_comparison_v2/` — Previous experiment (7.3GB, has checkpoints)

### Docs (in repo)
- `fVLM/docs/KNOWLEDGE.md` — Comprehensive knowledge base (2100+ lines), has all experiment results
- `fVLM/CLAUDE.md` — Project guide, architecture notes, debugging checklists

---

## Known Issues / Gotchas

1. **`total_mem` bug in run_comprehensive_scaling.py line 322**: Uses `.total_mem` which should be `.total_memory`. Verify before re-running.
2. **S-B_64f hits VRAM ceiling**: 23.3GB peak on 24.5GB card. Need gradient checkpointing or smaller micro-batch.
3. **Matplotlib rcParam**: `savefig.bbox_inches` is NOT a valid rcParam. Use `plt.savefig(..., bbox_inches='tight')` instead.
4. **Never train > 1 epoch**: `steps * batch_size < total_train_samples`. Always verify dataset size first.
5. **.gitignore blocks *.pdf** — PDFs are gitignored (redundant with PNGs).
6. **Batching bug (fixed)**: Original training loop did `unsqueeze(0)` always = batch 1. Now uses proper DataLoader + collate.

---

## Model Configs Reference

| Config | LLM | DINO | Fov Params | Bas Params |
|--------|-----|------|-----------|-----------|
| S-S | SmolLM2-135M | dinov2-small | 159M | 172M |
| M-S | SmolLM2-360M | dinov2-small | 386M | 411M |
| S-B | SmolLM2-135M | dinov2-base | 224M | 251M |
| B-L (untested) | SmolLM2-1.7B | dinov2-base | ~1.8B | ~1.9B |

---

## Quick Reference

```bash
# CWD for all scripts
cd /mnt/c/Users/sanps/Desktop/Projects/dino/nanoSmolLM/fVLM

# Re-run plotting
python research/scaling_laws/scripts/plot_scaling_fits.py

# Re-run specific experiment (edit CONFIGS_TO_RUN in script)
python research/scaling_laws/scripts/run_comprehensive_scaling.py

# Precompute 8F data (if data lost)
python scripts/precompute_8f.py

# Check results
python -c "import json; d=json.load(open('/mnt/d/projects/fVLM/outputs/scaling_comprehensive/results.json')); print(len(d), 'data points')"
```
