# Comprehensive Ablation Analysis: Foveated VLM

**Date:** 2026-01-06
**Experiments:** 7 ablations × 1500 steps each
**GPU:** RTX 4090, batch_size=2

## Executive Summary

After running 7 comprehensive ablation experiments (10,500 total training steps), we found:

| Rank | Experiment | Ratio | Feature Sim | Verdict |
|------|-----------|-------|-------------|---------|
| 🥇 | **D_freeze_dino** | **0.9966** | 0.43 | **WINNER: Best ratio + good differentiation** |
| 🥈 | B_temp_0.1 | 0.9987 | 0.93 | Moderate improvement |
| 🥈 | G_combined_best | 0.9987 | -0.53 | Same as temp, more complexity |
| 4 | C_contrastive | 1.0000 | -0.9993 | Max differentiation, no prediction gain |
| 5 | F_entropy_reg | 1.0000 | 0.98 | No impact |
| 6 | A_baseline | 1.0001 | 0.99 | Features collapsed (problem) |
| 7 | E_hard_attn_32 | 1.0001 | 0.98 | No impact |

**Key Finding:** Freezing DINO is the single most effective intervention, achieving:
- 0.34% improvement in loss ratio (loss_fine < loss_coarse)
- 57% feature differentiation (sim = 0.43 vs baseline 0.99)

---

## Detailed Analysis with Intuitions

### 1. A_baseline: The Problem
```
Config: temp=1.0, contrastive=0, freeze_dino=False
Result: ratio=1.0001, sim=0.9913
```

**What we observed:** Features from different queries are 99.13% similar.

**Why this happens:** When DINO is fine-tuned end-to-end, the gradient signal from the reconstruction loss causes all features to converge toward a "universal" representation that minimizes MSE. The query mechanism has no incentive to extract different information because both passes are optimizing the same objective.

**Intuition:** Imagine training two people to describe a painting. If you grade them on the same metric, they'll converge to similar descriptions. The queries become decorrelated from the attention mechanism.

---

### 2. B_temp_0.1: Sharper Attention
```
Config: temp=0.1 (10x sharper), contrastive=0, freeze_dino=False
Result: ratio=0.9987, sim=0.9266
```

**What we observed:**
- Feature similarity dropped from 0.99 → 0.93 (7% improvement)
- Ratio improved from 1.0001 → 0.9987 (0.14% improvement)

**Why this works (partially):** Lower temperature makes softmax attention "peakier", concentrating on fewer patches. Different queries now attend to more distinct patch sets. However, DINO still adapts to minimize reconstruction loss, causing features to re-converge over training.

**Intuition:** It's like giving two people magnifying glasses instead of binoculars. They focus on smaller areas, but if they're both trying to describe the same scene, they'll still end up similar.

---

### 3. C_contrastive: Maximum Differentiation, No Prediction Gain
```
Config: temp=1.0, contrastive=0.1, freeze_dino=False
Result: ratio=1.0000, sim=-0.9993
```

**What we observed:**
- Feature similarity went from +0.99 → **-0.9993** (opposite directions!)
- BUT ratio stayed at exactly 1.0000 (no improvement in prediction)

**Why this happens:** The contrastive loss explicitly pushes z_fine and z_coarse to be orthogonal/opposite. The model achieves this trivially by projecting to opposite hemispheres. However, this doesn't guarantee z_fine contains BETTER information - just DIFFERENT information.

**Intuition:** If I ask two people to say opposite things about a painting, one might say "it's blue" and the other "it's not blue." They're maximally different but neither is more informative about what's actually in the painting.

**Key insight:** Feature differentiation ≠ prediction improvement. The model found a shortcut.

---

### 4. D_freeze_dino: THE WINNER
```
Config: temp=1.0, contrastive=0, freeze_dino=True
Result: ratio=0.9966, sim=0.4300
```

**What we observed:**
- Feature similarity = 0.43 (57% differentiation)
- Ratio = 0.9966 (0.34% improvement - BEST!)
- Loss values were also lower (0.4978 vs baseline 0.5235)

**Why this works:** When DINO is frozen:
1. **Preserved diversity:** DINO's pretrained features remain rich and spatially diverse (it was trained on massive image datasets with contrastive objectives)
2. **Query specialization:** The query mechanism learns to extract genuinely different information from the frozen feature space
3. **No feature collapse:** DINO can't adapt to "help" the model cheat by making all features similar

**Intuition:** Imagine you have a pre-trained expert (DINO) who gives rich descriptions of different image regions. By freezing them, you force your query mechanism to actually learn WHERE to look, rather than changing what the expert says. The expert's unchanging diversity provides the foundation for meaningful query differentiation.

**Why ratio < 1.0 (loss_fine < loss_coarse):** With frozen, diverse features, the fine query (which can use text context) genuinely extracts more relevant information than the coarse query (which is static).

---

### 5. E_hard_attn_32: No Impact
```
Config: temp=1.0, hard_attn_k=32, freeze_dino=False
Result: ratio=1.0001, sim=0.9771
```

**What we observed:** Minimal improvement (1.4% less similar).

**Why this doesn't work:** Hard attention (top-32 patches) still allows significant overlap. With 256 total patches, selecting 32 leaves room for >50% overlap between queries. More importantly, DINO still adapts to make even the "different" patches produce similar features.

**Intuition:** Picking the top 32 vs top 32 patches from similar attention patterns gives similar results. The sparsity doesn't address the root cause (feature collapse during training).

---

### 6. F_entropy_reg: No Impact
```
Config: entropy_margin=0.5, freeze_dino=False
Result: ratio=1.0000, sim=0.9841
```

**What we observed:** Almost no change from baseline.

**Why this doesn't work:** The entropy regularizer encourages fine attention to have lower entropy (more focused) than coarse attention. However, this doesn't prevent feature collapse - the model can have focused attention on patches that still produce similar features.

**Intuition:** Being more focused doesn't help if what you're focused on has been homogenized.

---

### 7. G_combined_best: Diminishing Returns
```
Config: temp=0.1, contrastive=0.1, freeze_dino=True, hard_attn_k=32
Result: ratio=0.9987, sim=-0.5251
```

**What we observed:**
- Ratio = 0.9987 (same as temp_0.1 alone, WORSE than freeze_dino alone)
- Feature sim = -0.53 (negative due to contrastive)

**Why combining hurts:** The contrastive loss creates an artificial pressure toward opposite features, which conflicts with the natural differentiation from frozen DINO. The model spends capacity achieving negative similarity rather than improving predictions.

**Intuition:** Adding multiple interventions can interfere. Frozen DINO alone provides clean, natural diversity. Adding contrastive loss creates artificial opposition that doesn't translate to better predictions.

---

## Recommendations

### Immediate Next Steps

1. **Use freeze_dino=True as the new default**
   - Achieves best ratio (0.9966) and good differentiation (0.43)
   - Simplest intervention with clearest benefit

2. **Run extended training with frozen DINO**
   - Train for 10K+ steps to see if ratio continues to improve
   - Current 1500 steps may not be enough for full effect

3. **Combine freeze_dino + temp=0.1**
   - Try without contrastive loss (which hurts)
   - May achieve even better results

### Architectural Insights

1. **Pretrained encoder diversity is crucial**
   - Don't fine-tune the vision encoder when learning query-based attention
   - The encoder's pretrained features provide the "raw material" for differentiation

2. **Feature differentiation ≠ better predictions**
   - Contrastive loss shows you can have perfect differentiation (sim=-1.0) with no prediction improvement
   - The goal is USEFUL differentiation, not just any differentiation

3. **The core hypothesis needs refinement**
   - Current: "loss_fine < loss_coarse" validates foveated attention
   - Refined: "loss_fine < loss_coarse WHEN encoder preserves pretrained diversity"

---

## Summary Table

| Ablation | What It Does | Result | Intuition |
|----------|-------------|--------|-----------|
| Baseline | Train everything | ❌ sim=0.99 | Features collapse |
| Temp 0.1 | Sharper attention | 〰️ Modest help | Focuses attention but features still collapse |
| Contrastive | Push features apart | ❌ Cheats | Achieves opposite features but not better predictions |
| **Freeze DINO** | **Keep encoder frozen** | ✅ **Best** | **Preserves diversity, forces query learning** |
| Hard attention | Sparse patch selection | ❌ No effect | Sparsity doesn't prevent collapse |
| Entropy reg | Focus fine more | ❌ No effect | Focus without diversity is useless |
| Combined | All together | 〰️ Conflicts | Interventions interfere with each other |

---

*Analysis by Claude, 2026-01-06*
