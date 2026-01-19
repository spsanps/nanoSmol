# Proposal: Non-FiLM Prediction Head Experiments

**Date:** 2026-01-19
**Status:** PROPOSAL - Awaiting approval

---

## Problem Statement

The current FiLM prediction head has a fundamental limitation:

```python
# Current implementation (src/model/prediction.py)
gamma = gamma.view(-1, 256, 1, 1)  # Shape: [B*T, 256, 1, 1]
beta = beta.view(-1, 256, 1, 1)    # Shape: [B*T, 256, 1, 1]
feat = gamma * feat + beta          # Broadcasts to [B*T, 256, 32, 32]
```

**The Issue:** `gamma` and `beta` are broadcast across ALL 32x32 spatial positions. The LLM's hidden state `h` (576-dim) is compressed to 256 pairs of scalars that apply identically everywhere.

**Impact:** The LLM cannot express:
- "Focus on the top-left corner"
- "This region will change more"
- "Predict different motion in different areas"

This limits both:
1. Current VAE latent prediction
2. Proposed DINO patch prediction (which is why we paused that research)

---

## Proposed Experiments

### Experiment 1: Direct MLP Prediction (Baseline)

**Hypothesis:** Replace FiLM with a simple MLP that predicts the next latent directly from concatenated inputs.

**Architecture:**
```
Input: concat(h, flatten(z_prev))  # [B*T, 576 + 4*32*32] = [B*T, 4672]
   ↓
MLP layers (576 → 1024 → 1024 → 4*32*32)
   ↓
Output: reshape to [B*T, 4, 32, 32]
```

**Rationale:**
- Baseline to understand what we lose by removing spatial structure
- Forces LLM hidden state to encode ALL prediction information

**Expected Outcome:** Worse than FiLM due to losing spatial structure of prev_latent

---

### Experiment 2: Cross-Attention Prediction

**Hypothesis:** Use the LLM hidden state as query to attend over the previous latent's spatial positions.

**Architecture:**
```
Query: Linear(h)                    # [B*T, 256]
Keys:  Conv(z_prev) → flatten      # [B*T, 1024, 256] (32x32 positions)
Values: Conv(z_prev) → flatten     # [B*T, 1024, 256]

Attention: softmax(Q @ K^T / sqrt(d)) @ V
   ↓
Output: reshape + decode to [B*T, 4, 32, 32]
```

**Rationale:**
- LLM can now "choose" which regions of previous frame to attend to
- Different h values will attend to different spatial positions
- Still uses previous latent structure but with selective attention

**Expected Outcome:** Should outperform FiLM if spatial selectivity matters

---

### Experiment 3: Spatially-Varying FiLM (Position-Conditional)

**Hypothesis:** Generate different gamma/beta for different spatial positions.

**Architecture:**
```
Position embeddings: [1024, 64] (32x32 positions, 64-dim each)
h_expanded: h repeated for each position → [B*T, 1024, 576]
Combined: concat(h_expanded, pos_embed) → [B*T, 1024, 640]
   ↓
MLP per position → gamma, beta per position → [B*T, 1024, 256], [B*T, 1024, 256]
   ↓
Reshape to [B*T, 256, 32, 32] and apply
```

**Rationale:**
- Minimal change from current FiLM
- Adds spatial awareness while keeping the modulation paradigm
- Position embeddings let LLM learn position-dependent modulation

**Expected Outcome:** Moderate improvement - more flexible than current FiLM

---

### Experiment 4: Spatial Transformer (Most Flexible)

**Hypothesis:** Use a small transformer decoder where LLM hidden state modulates spatial tokens.

**Architecture:**
```
z_prev → Conv → spatial tokens [B*T, 1024, 256]  (32x32 positions)
h → Linear → conditioning [B*T, 256]

Transformer decoder:
  - Self-attention over spatial tokens
  - Cross-attention with h as additional context
  - 2-3 layers

Output: decode spatial tokens → [B*T, 4, 32, 32]
```

**Rationale:**
- Maximum flexibility for LLM to influence prediction
- Spatial tokens can interact with each other AND with LLM context
- Similar to how diffusion models condition on text

**Expected Outcome:** Best performance but highest compute cost

---

## Experiment Protocol

### Common Setup
- **Target:** VAE latent prediction (same as current)
- **Data:** WebVid streaming (same as current)
- **Model:** FoveatedVLM with modified prediction head only
- **Steps:** 1000 steps per experiment (sufficient for comparison)
- **Batch:** 2 × 4 = 8 effective (same as previous experiments)
- **Metric:** fine/coarse ratio (same as current)

### Comparison Baseline
- Current FiLM head performance from joint training
- Expected ratio: ~1.17 for reconstruction

### Success Criteria
- Any non-FiLM head achieving ratio > 1.17 = improvement
- Cross-attention or spatial transformer achieving ratio > 1.25 = significant

---

## Implementation Order

1. **Experiment 2 (Cross-Attention)** - Most promising, moderate complexity
2. **Experiment 3 (Spatially-Varying FiLM)** - Minimal change from current
3. **Experiment 4 (Spatial Transformer)** - If cross-attention shows promise
4. **Experiment 1 (Direct MLP)** - Only if needed as ablation baseline

---

## Questions Before Proceeding

1. **Scope:** Should we test on reconstruction only, or also captioning?
   - Recommendation: Start with reconstruction since that's where FiLM limitation was identified

2. **Compute budget:** How many experiments can we run?
   - Recommendation: Start with Experiment 2, then decide based on results

3. **Checkpoint:** Should we start from scratch or fine-tune from joint_multifine checkpoint?
   - Recommendation: Start from scratch to isolate prediction head effect

---

## Expected Timeline

- Experiment 2: ~6-8 hours (includes implementation + 1000 steps training)
- Analysis: ~2 hours
- Decision point: Proceed with remaining experiments or pivot

---

## Files to Create

```
research/prediction_head_alternatives/
├── PROPOSAL.md          # This document
├── cross_attention_head.py  # Experiment 2 implementation
├── spatial_film_head.py     # Experiment 3 implementation
├── spatial_transformer_head.py  # Experiment 4 implementation
├── experiment_runner.py     # Training script for all variants
└── results/
    └── comparison.md        # Results summary
```

---

**Awaiting approval to proceed.**
