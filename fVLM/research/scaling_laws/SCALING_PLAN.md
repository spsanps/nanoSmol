# Scaling Law Experiment Plan

## Goal
Create Chinchilla-style scaling laws for foveated vs baseline VLM by varying:
1. **Model size** (LLM + vision encoder parameters)
2. **Training compute** (steps × batch_size × FLOPs)

## Available Model Sizes

### LLM Options (SmolLM2 family)
| Model | Params | Hidden Dim | Layers |
|-------|--------|------------|--------|
| SmolLM2-135M | 135M | 576 | 30 |
| SmolLM2-360M | 360M | 960 | 32 |
| SmolLM2-1.7B | 1.7B | 2048 | 24 |

### Vision Encoder Options (DINOv2)
| Model | Params | Hidden Dim | Patch Size |
|-------|--------|------------|------------|
| dinov2-small | 22M | 384 | 14 |
| dinov2-base | 86M | 768 | 14 |
| dinov2-large | 300M | 1024 | 14 |

## Experiment Configurations

### Config Matrix (9 configurations)
| ID | LLM | Vision | Total Params | Est. FLOPs/sample |
|----|-----|--------|--------------|-------------------|
| S-S | 135M | small | ~160M | 145 GFLOPs |
| S-B | 135M | base | ~220M | 180 GFLOPs |
| S-L | 135M | large | ~430M | 280 GFLOPs |
| M-S | 360M | small | ~385M | 290 GFLOPs |
| M-B | 360M | base | ~450M | 360 GFLOPs |
| M-L | 360M | large | ~660M | 520 GFLOPs |
| L-S | 1.7B | small | ~1.72B | 1.2 TFLOPs |
| L-B | 1.7B | base | ~1.79B | 1.4 TFLOPs |
| L-L | 1.7B | large | ~2.0B | 1.8 TFLOPs |

### VRAM Constraints (RTX 4090 = 24GB)
- S-S, S-B, S-L: OK with BS=16
- M-S, M-B: OK with BS=8-12
- M-L: OK with BS=4-8
- L-*: Requires BS=1-2 + gradient accumulation

## Prioritized Experiment Plan

### Phase 1: Core Comparison (Current + Extended)
**Goal:** Establish baseline scaling with current model size

| Experiment | Config | Steps | Purpose |
|------------|--------|-------|---------|
| foveated_S-S | 135M + small | 100, 300, 1K, 3K | Current foveated |
| baseline_S-S | 135M + small | 100, 300, 1K, 3K | Current baseline |

### Phase 2: LLM Size Scaling
**Goal:** How does larger LLM affect foveated advantage?

| Experiment | Config | Steps | Purpose |
|------------|--------|-------|---------|
| foveated_M-S | 360M + small | 100, 300, 1K, 3K | Medium LLM |
| baseline_M-S | 360M + small | 100, 300, 1K, 3K | Medium LLM |

### Phase 3: Vision Encoder Scaling
**Goal:** Does foveated benefit more from larger vision encoders?

| Experiment | Config | Steps | Purpose |
|------------|--------|-------|---------|
| foveated_S-B | 135M + base | 100, 300, 1K, 3K | Larger vision |
| baseline_S-B | 135M + base | 100, 300, 1K, 3K | Larger vision |

### Phase 4: Full Matrix (if time permits)
- All 9 configurations at multiple step counts
- Enables fitting full scaling law: L(N, D) = A/N^α + B/D^β + C

## Expected Plots

1. **Loss vs FLOPs** (per config): Shows efficiency
2. **Loss vs Model Size** (at fixed FLOPs): Shows capacity scaling
3. **Iso-loss curves**: FLOPs required for same quality
4. **Efficiency ratio**: foveated_loss / baseline_loss vs compute
5. **Optimal model size**: Given compute budget, which config is best?

## Empirical Laws to Derive

1. **Scaling exponent**: L ∝ C^(-α) where C = compute
2. **Crossover point**: At what compute does foveated beat baseline?
3. **Efficiency multiplier**: How much more efficient is foveated?
4. **Optimal allocation**: Given budget, how to split model vs data?

## Implementation

Script: `run_scaling_study_multisize.py`
- Iterates through config matrix
- Handles different batch sizes per config
- Saves checkpoints and metrics
- Generates unified scaling plots

## Timeline Estimate

| Phase | Configs | Est. Time |
|-------|---------|-----------|
| Phase 1 | 2 | 1-2 hours (done) |
| Phase 2 | 2 | 2-3 hours |
| Phase 3 | 2 | 2-3 hours |
| Evaluation | All | 1-2 hours |
| Analysis | - | 30 min |

Total: ~8-10 hours for phases 1-3
