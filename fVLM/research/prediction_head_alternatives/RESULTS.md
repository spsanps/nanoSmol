# Prediction Head Alternatives: Experiment Results

**Date:** 2026-01-19
**Duration:** ~30 minutes each experiment

---

## Summary Table

| Head Type | Steps | Final Ratio | Ratio > 1.0 | Parameters | Notes |
|-----------|-------|-------------|-------------|------------|-------|
| **MLP** | 815 | **1.020** | 94.2% | 22.17M | Slight improvement |
| **Cross-Attention** | 834 | **1.000** | 65.6% | 1.33M | No improvement |
| FiLM (baseline)* | - | ~1.17 | ~100% | 0.66M | From joint training |

*FiLM baseline is from previous joint recon+caption experiments, not directly comparable.

---

## Key Findings

### 1. Cross-Attention Head Failed to Show Improvement

**Result:** Ratio = 1.00007 (essentially no difference between fine and coarse)

**Hypothesis was:** If LLM can spatially attend over previous latent, fine queries should extract more useful spatial information than coarse queries.

**What happened:** The attention mechanism produces essentially identical predictions for fine and coarse paths. This suggests:
- The spatial attention over VAE latent space is NOT helpful for this task
- VAE latents may be too compressed/abstract for meaningful spatial attention
- The 32x32 spatial resolution may be too coarse for attention to matter

### 2. MLP Head Shows Modest Improvement

**Result:** Ratio = 1.020 (2% improvement, fine beats coarse 94% of the time)

**Surprising:** A simple MLP without any spatial structure outperforms cross-attention.

**Interpretation:**
- The MLP forces the LLM hidden state `h` to encode more predictive information
- No spatial shortcuts means `h` must contain everything needed for prediction
- The 2% ratio suggests fine queries do encode slightly more useful information

### 3. Both Underperform FiLM in Joint Training

**Critical context:** These experiments used reconstruction-only training on local data. Previous experiments showed:
- FiLM with reconstruction-only: ratio ≈ 1.0 (no improvement)
- FiLM with joint training (recon + caption): ratio ≈ 1.17 (17% improvement)

**Conclusion:** The prediction head architecture alone cannot solve the fine/coarse gap. The improvement comes from the **training signal** (captioning teaches WHERE to look), not from the prediction head design.

---

## Detailed Analysis

### MLP Head (Experiment 1)

```
Architecture:
  h (576) + flatten(z_prev) (4096) = 4672
  → Linear(4672, 2048) + GELU + LayerNorm
  → Linear(2048, 2048) + GELU + LayerNorm
  → Linear(2048, 4096)
  → reshape to (4, 32, 32)

Parameters: 22.17M (much larger than FiLM's 0.66M)
```

**Observations:**
- Ratio starts near 1.0, increases to ~1.02-1.06 in middle, then fluctuates
- High variance in ratio (0.98 to 1.06)
- The large parameter count means the model can memorize, but still shows fine > coarse

**Why it works (slightly):**
- Forces all information through `h`, no spatial bypass
- If fine queries extract better features, this difference propagates to prediction
- But 2% is marginal - not a strong signal

### Cross-Attention Head (Experiment 2)

```
Architecture:
  z_prev → Conv encoder → 1024 spatial tokens (32x32)
  h → query generation per position
  Cross-attention: Q(h, pos) attends to K,V(z_prev tokens)
  Output → Conv decoder → (4, 32, 32)

Parameters: 1.33M (smaller but more structured)
```

**Observations:**
- Ratio stays extremely close to 1.0 throughout (max 1.002, min 0.997)
- The attention mechanism equalizes fine and coarse
- No learning signal to differentiate the paths

**Why it failed:**
1. **VAE latents are too compressed:** 32x32 spatial tokens from VAE don't have enough local structure for attention to matter
2. **Position is predetermined:** The output position is fixed, so attention just learns to aggregate all input positions similarly
3. **No semantic signal:** Without captioning, there's no gradient pushing "attend here vs there"

---

## Implications

### For Prediction Head Design

1. **Spatial attention on VAE latents is not helpful** for reconstruction
2. **Simple architectures work as well as complex ones** when training signal is weak
3. **The training objective (captioning) is more important** than prediction head design

### For Future Work

1. **Don't invest more in prediction head alternatives** until we have a better training signal
2. **Cross-attention might work with DINO patches** (higher spatial resolution, more semantic features)
3. **Joint training remains the key** - prediction head improvements are marginal without semantic learning

---

## Comparison Chart

```
Ratio Performance:

MLP:              |=====|           ratio = 1.020 (2% better)
Cross-Attention:  |=|               ratio = 1.000 (no diff)
FiLM (recon):     |=|               ratio = 1.000 (baseline)
FiLM (joint):     |==========|      ratio = 1.170 (17% better)
                  0        1.2
```

**Takeaway:** The prediction head matters less than we thought. The training objective (joint captioning) provides the gradient signal that makes fine queries better than coarse.

---

## Recommendations

1. **Keep current FiLM head** - it's simple and works with joint training
2. **Don't pursue cross-attention** on VAE latents - no benefit
3. **Revisit spatial heads if/when** we move to DINO patch prediction
4. **Focus on training objectives** rather than prediction architecture

---

## wandb Links

- MLP: https://wandb.ai/sanjayanps/foveated-vlm-pred-head/runs/mceb04gv
- Cross-Attention: https://wandb.ai/sanjayanps/foveated-vlm-pred-head/runs/nkz7qt4u
