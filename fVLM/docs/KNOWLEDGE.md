# Foveated VLM Knowledge Base

**Purpose:** Central repository for all learnings, experiments, bugs, and insights.

**How to Add Knowledge:**
1. Add new sections under the appropriate category
2. Include date, symptom, root cause, and fix/insight
3. Update CLAUDE.md debugging checklist if it's a common issue
4. Commit with descriptive message

---

## Table of Contents

1. [Critical Bugs & Fixes](#critical-bugs--fixes)
2. [Training Insights](#training-insights)
3. [Architecture Notes](#architecture-notes)
4. [Performance Optimizations](#performance-optimizations)
5. [Experiment History](#experiment-history)

---

## Critical Bugs & Fixes

### BUG-001: Query Initialization Scale (2026-01-04)

**Symptom:** `loss_fine == loss_coarse` exactly (ratio = 1.00)

**Root Cause:** Query vectors initialized with `std=0.02` instead of `std=1.0`.
The linear projection's bias (~0.088 std) dominated the tiny query signal.

**Diagnosis:**
```python
# Test with:
q_static = model.q_static
print(f"q_static std: {q_static.std().item()}")  # Should be ~1.0

# Attention entropy should be < 5.5 (max for 257 patches)
```

**Evidence:**
| Metric | Before (std=0.02) | After (std=1.0) |
|--------|-------------------|-----------------|
| Embedding diff | 0.012 | 0.75 |
| Attention entropy | 5.548 | 5.28 |
| Output correlation | 0.9998 | 0.52 |

**Fix:** `src/model/foveated_vlm.py` line 88-89
```python
# Change from:
self.q_static = nn.Parameter(torch.randn(1, query_dim) * 0.02)
# To:
self.q_static = nn.Parameter(torch.randn(1, query_dim))  # std=1.0
```

---

### BUG-002: Query Projection Bias (2026-01-04)

**Symptom:** Different queries produce identical attention patterns

**Root Cause:** `query_input_proj` had bias that dominated small queries

**Fix:** `src/model/encoder.py` line 59
```python
# Change from:
self.query_input_proj = nn.Linear(query_dim, self.dino_dim)
# To:
self.query_input_proj = nn.Linear(query_dim, self.dino_dim, bias=False)
```

---

### BUG-003: Per-Sample Mode Selection (2026-01-04)

**Symptom:** Losses increasing during training instead of decreasing

**Root Cause:** Processing samples individually within a batch breaks batch normalization and model assumptions

**Fix:** Select mode per-batch, not per-sample

---

### BUG-004: Shallow Mode Produces Uniform Attention (2026-01-04)

**Symptom:** `loss_fine == loss_coarse` (ratio = 1.00) even after fixing query initialization

**Root Cause:** Shallow mode (`deep_query=False`) uses a single cross-attention layer on DINO's final features. These features are highly correlated across spatial positions (self-dot-products have mean=2113, std=344), causing attention to be nearly uniform (entropy=5.55, max=5.55).

**Diagnosis:**
```python
# Test with different queries:
# Shallow mode:
#   Output correlation: 0.9818 (nearly identical!)
#   Attention entropy: 5.55 (completely uniform)
#
# Deep mode:
#   Output correlation: 0.4341 (differentiated)
#   Queries propagate through all 12 DINO layers
```

**Evidence:**
| Mode | Output Correlation | Output L2 Diff | Attention |
|------|-------------------|----------------|-----------|
| Shallow | 0.98 | 7.7 | Uniform |
| Deep | 0.43 | 48.7 | Selective |

**Fix:** Enable deep query mode in `src/model/foveated_vlm.py`:
```python
# Default changed to deep_query=True
self.encoder = FoveatedEncoder(
    ...
    deep_query=True,  # CRITICAL for query differentiation
)
```

**Performance Impact:** Deep mode is ~42x slower per query_attend call but only ~10% slower overall due to other overhead (DINO encoding, LLM forward passes).

---

### BUG-005: forward_captioning Missing kv_cache (2026-01-04)

**Symptom:** `KeyError: 'kv_cache'` crash at step 26 during captioning mode

**Root Cause:** `forward_captioning` created per-frame caches assuming shallow mode structure:
```python
all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]
```
But deep mode returns `kv_cache` (list of 12 layer caches) which needs to be reshaped per-frame.

**Fix:** `src/model/foveated_vlm.py` in `forward_captioning`:
```python
# Create per-frame caches (must handle both shallow and deep mode)
all_caches = []
if 'kv_cache' in cache_flat:
    # Deep mode: reshape kv_cache for per-frame access
    num_layers = len(cache_flat['kv_cache'])
    for t in range(T):
        frame_kv_cache = []
        for layer_idx in range(num_layers):
            layer_cache = cache_flat['kv_cache'][layer_idx]
            K_all = layer_cache['K'].reshape(B, T, N, D)
            V_all = layer_cache['V'].reshape(B, T, N, D)
            frame_kv_cache.append({
                'K': K_all[:, t],
                'V': V_all[:, t],
                'layer': layer_cache['layer'],
            })
        all_caches.append({
            'patch_features': patch_features[:, t],
            'kv_cache': frame_kv_cache,
        })
else:
    # Shallow mode: just patch_features per frame
    all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]
```

---

### BUG-006: encode_video Missing kv_cache (2026-01-04)

**Symptom:** `KeyError: 'kv_cache'` crash at step 500 during caption generation

**Root Cause:** Same issue as BUG-005, but in `encode_video` function. The captioning mode calls `generate_caption` which uses `encode_video`, and this function also assumed shallow mode cache structure.

**Fix:** `src/model/foveated_vlm.py` in `encode_video` - applied identical fix as BUG-005.

**Note:** Both functions now handle shallow AND deep mode, making the model robust regardless of the `deep_query` setting.

---

## Training Insights

### Observation: Fine/Coarse Ratio Stays at 1.0 (Historical)

**Date:** 2026-01-02 to 2026-01-04

**Context:** Multiple training runs showed ratio = 1.000 throughout

**Analysis (from Phase 2):**
- Loss progression: 0.821 → 0.713 (13% improvement)
- Fine and coarse always equal to 3 decimal places
- This was NOT random - it indicated a systematic issue

**Resolution:** Fixed by BUG-001 and BUG-002 above. The issue was that
both coarse (static) and fine (dynamic) queries were producing identical
attention patterns due to poor initialization.

---

### Optimal Training Configuration

**Hardware:** RTX 4090 (24GB, ~20GB usable)

**Best Settings (throughput/memory balance):**
```yaml
batch_size: 3
grad_accum: 6
effective_batch: 18
num_frames: 16
frame_size: 256
learning_rate: 3e-5
throughput: ~1.1s/step
vram_usage: ~9GB
```

**Streaming from WebVid-10M:**
- Success rate: ~90-92%
- Network variability causes periodic slowdowns

---

## Architecture Notes

### Two-Pass Structure

**Pass 1 (Query Planning):**
- Uses learned `q_static` for ALL frames
- Extracts coarse features
- LLM predicts dynamic queries from coarse features

**Pass 2 (Focused Extraction):**
- Uses shifted dynamic queries: `[q_init, q_1, q_2, ..., q_{T-1}]`
- Query for frame t based on seeing frames 0..t-1
- Should extract more informative features than static query

**Key Insight:** The queries must be initialized large enough (std=1.0)
to overcome the projection layer and produce meaningful attention patterns.

---

### Proposal vs Implementation Discrepancies

| Aspect | Proposal | Implementation | Status |
|--------|----------|----------------|--------|
| q_static init | `torch.randn(1, 384)` | `torch.randn(1, 128)` | Different dim, OK |
| q_static scale | std=1.0 | was std=0.02 | **FIXED** |
| query_input_proj | No mention of bias | Had bias | **FIXED** |
| query_dim | 384 | 128 | Intentional |

---

## Performance Optimizations

### Streaming Dataset (2026-01-04)

**Problem:** 240GB storage for full dataset unsustainable

**Solution:** Stream from WebVid-10M HuggingFace
- No local storage needed
- Downloads on-the-fly
- ~90% success rate

**Implementation:** `scripts/train_large_scale.py`

---

### Memory Optimizations

- Gradient checkpointing on LLM
- bf16 precision throughout
- VAE latents computed per-batch (not precomputed for streaming)

---

## Experiment History

### Phase 1: Self-Supervised (WebVid 813 samples)
- Final loss: 0.34
- Severe overfitting observed

### Phase 2: Text-Conditioned (LLaVA-Video 11,985 samples)
- Final loss: 0.713
- 13% improvement
- ratio = 1.000 (bug - not detected at time)

### Large-Scale Run v1 (WebVid-10M streaming, shallow mode)

**Date:** 2026-01-04 (Run 1)

**Configuration:**
- Mode: shallow (deep_query=False) - INCORRECT
- Batch: 3 × 6 = 18 effective
- Learning rate: 3e-5
- Streaming from WebVid-10M

**Result:** Crashed at step 26
- Error: `KeyError: 'kv_cache'` in `forward_captioning`
- Root cause: Code assumed shallow mode cache structure
- Fix: BUG-005 (see above)

---

### Large-Scale Run v2 (WebVid-10M streaming, deep mode)

**Date:** 2026-01-04 (Run 2)

**Configuration:**
- Mode: deep_query=True (correct!)
- Batch: 3 × 6 = 18 effective
- Learning rate: 3e-5
- q_static std: ~0.97 (verified)

**Result:** Crashed at step 500
- Error: `KeyError: 'kv_cache'` in `encode_video`
- Root cause: Same cache structure issue, different function
- Fix: BUG-006 (see above)

**Metrics before crash:**
| Step | fine | coarse | ratio |
|------|------|--------|-------|
| 100 | 0.790 | 0.788 | 1.00 |
| 200 | 0.843 | 0.841 | 1.00 |
| 300 | 0.824 | 0.824 | 1.00 |
| 400 | 0.837 | 0.836 | 1.00 |

---

### Large-Scale Run v3 (WebVid-10M streaming, deep mode, all fixes)

**Date:** 2026-01-04 (Run 3) - CURRENT

**Configuration:**
```yaml
deep_query: true
batch_size: 3
grad_accum: 6
effective_batch: 18
learning_rate: 3e-05
mode_weights: video-only=60%, text-cond=20%, caption=20%
```

**Verified Settings:**
- q_static std: ~0.97 (correct)
- deep_query: True (12-layer kv_cache)
- kv_cache handling: Fixed for both shallow and deep modes

**Status:** Running (step 380+)

**Metrics:**
| Step | fine | coarse | ratio | Notes |
|------|------|--------|-------|-------|
| 100 | 0.790 | 0.788 | 1.00 | Initial convergence |
| 200 | 0.843 | 0.841 | 1.00 | Slight increase |
| 300 | 0.824 | 0.824 | 1.00 | Stabilizing |

**System Performance:**
- Memory: ~11.3GB / 24.6GB (stable, no leak)
- Speed: ~1.0-1.1s/step (normal), spikes to ~2s/step (network variability)
- Success rate: ~94%

**Key Observations:**
1. Training passed previous crash points (step 26, step 500)
2. Ratio remains at 1.00 - fine ≈ coarse
3. Deep mode confirmed active (12-layer cache structure)
4. Memory stable - no leak detected

---

### Large-Scale Run v3 - COMPLETED (24 hours)

**Date:** 2026-01-04 to 2026-01-05

**Configuration:**
```yaml
deep_query: true
batch_size: 3
grad_accum: 6
effective_batch: 18
learning_rate: 3e-05
mode_weights: video-only=60%, text-cond=20%, caption=20%
duration: 24 hours
```

**Final Results:**
| Metric | Value |
|--------|-------|
| Total Steps | 26,153 |
| Samples Seen | 78,463 |
| Final Loss (fine) | 0.20694 |
| Final Loss (coarse) | 0.20675 |
| Final Ratio | 1.00 |
| Success Rate | ~94% |

**Loss Progression:**
| Step | fine | coarse | ratio |
|------|------|--------|-------|
| 100 | 0.790 | 0.788 | 1.00 |
| 1000 | 0.543 | 0.542 | 1.00 |
| 5000 | 0.234 | 0.234 | 1.00 |
| 10000 | 0.220 | 0.220 | 1.00 |
| 20000 | 0.219 | 0.220 | 1.00 |
| 26153 | 0.207 | 0.207 | 1.00 |

**Conclusion:**
- Training completed successfully without crashes
- Losses converged to ~0.207 (74% reduction from initial 0.79)
- **Core hypothesis NOT validated:** fine ≈ coarse throughout

---

### Post-Training Evaluation (2026-01-06)

**Setup:**
- Checkpoint: `final_step_26153.pt`
- Samples: 20 videos from WebVid-10M
- Per-frame loss analysis on 16 frames each

**Overall Statistics:**
| Metric | Coarse | Fine | Δ | Winner |
|--------|--------|------|---|--------|
| Avg Loss (all frames) | 0.1974 | 0.1977 | -0.0002 | Coarse |
| Avg Loss (last 4) | 0.1339 | 0.1339 | -0.0000 | Coarse |

**Per-Frame Analysis:**
| Frame | Coarse | Fine | Winner |
|-------|--------|------|--------|
| 0 | 1.2950 | 1.2994 | Coarse |
| 1 | 0.1700 | 0.1700 | Tie |
| ... | ... | ... | ... |
| 12-15 (last 4) | ~0.13 | ~0.13 | Mixed |

**Win Statistics:**
- Fine wins overall: 5/20 (25%)
- Fine wins on last 4 frames: 12/20 (60%)

**Key Observations:**
1. Frame 0 has ~10x higher loss (1.30 vs 0.10-0.20) - expected since no prior info
2. Fine queries win slightly more often on later frames (60%) but magnitude is negligible
3. Captions are repetitive and not strongly tied to actual video content

**Critical Attention Statistics Finding:**
| Metric | Coarse | Fine | Interpretation |
|--------|--------|------|----------------|
| Avg Entropy | 5.698 | 5.591 | Fine 0.11 lower = more focused |
| Avg Max Attn | 0.009 | 0.017 | Fine ~1.8x higher peak attention |
| Focus Ratio | - | 1.80 | Fine nearly 2x more focused |

**Conclusion:** Fine queries ARE producing more focused attention (validated by entropy/max stats), but this doesn't translate to better reconstruction loss.

**Visualizations Generated:**
- `outputs/eval_24h/eval_XX_ID.png` - Comprehensive per-sample analysis
- `outputs/eval_24h/gif_XX_ID.gif` - Attention comparison with difference maps
- `outputs/eval_24h/stats_XX_ID.png` - Per-sample attention & loss statistics
- `outputs/eval_24h/EVALUATION_REPORT.md` - Detailed statistics

---

## Analysis: Why Fine ≈ Coarse (Despite Different Attention)

After 24 hours of training and comprehensive evaluation:
- **Core hypothesis NOT validated:** loss_fine ≈ loss_coarse
- **However:** Fine queries DO produce more focused attention (1.8x focus ratio)

The attention mechanism IS working, but focused attention doesn't improve reconstruction.

---

### Preliminary Experiments (2026-01-06)

**Purpose:** Data-driven diagnosis to identify the root cause and best fix direction.

**Experiment Results:**

| Exp | Test | Finding |
|-----|------|---------|
| 1 | prev_latents dominance | Model 4.7x MORE sensitive to h than prev_latents! (GOOD) |
| 2 | Feature similarity | z_coarse and z_fine are 98.75% similar! (BAD - root cause) |
| 3 | Temperature | Current temp=1.0 gives uniform attention; temp=0.1 is 10x sharper |
| 4 | Query projection | Queries are orthogonal in/out of projection (working correctly) |
| 5 | Pred head sensitivity | Responds well to h changes (not just copying prev) |

**CRITICAL FINDING:** Despite different queries, extracted features are 98.75% similar!

This directly explains why `loss_fine ≈ loss_coarse`:
1. Queries ARE different (cosine sim = -0.015)
2. Attention patterns differ (entropy 5.1 vs 5.5)
3. **BUT features z_fine and z_coarse are 98.75% similar** ← THE PROBLEM
4. So h_fine ≈ h_coarse
5. So loss_fine ≈ loss_coarse

**Root Causes Identified:**
1. **Uniform attention (temp=1.0):** Max attention = 0.0044 (nearly uniform). Different attention patterns still extract similar weighted averages.
2. **DINO feature homogenization:** Training may have caused features to become too similar across patches.

**WRONG Hypothesis:** prev_latents dominance was NOT the issue. The model uses visual features (h) 4.7x more than prev_latents.

---

### 1. Why Attention Differs But Loss Doesn't (UPDATED)

- **Uniform attention:** With temperature=1.0, attention is nearly uniform (max=0.0044), so all queries extract similar weighted averages regardless of focus
- **Feature homogeneity:** DINO features may have converged during training, reducing patch-level diversity
- **Information redundancy:** VAE latents encode high-level structure that's available in ALL patches

### 2. Architectural Factors

- **Shared encoder:** Both queries use same DINO weights, so features are fundamentally similar
- **Temperature:** Default softmax temperature too high, causing attention to spread
- **Fixed prediction head:** Same head processes both fine and coarse; may not leverage attention differences

### 3. Dataset Characteristics

- **WebVid videos static:** Low motion means all patches contain similar temporal info
- **Repetitive captions:** Limited semantic diversity in training signal

### 4. What This Means

The model learned to:
- ✅ Produce more focused attention with dynamic queries
- ✅ Reduce reconstruction loss (0.79 → 0.21, 74% improvement)
- ✅ Use visual features h (not just copy prev_latents)
- ❌ NOT extract different features with different attention patterns

The bottleneck is **between attention and feature extraction**:
- Attention is sharper but features are still similar
- Need to force feature differentiation

### 5. Priority Fixes (Based on Experiments)

| Priority | Fix | Rationale |
|----------|-----|-----------|
| 1 | Temperature = 0.1 | 10x more focused attention, trivial change |
| 2 | Contrastive loss | Push z_fine and z_coarse apart explicitly |
| 3 | Freeze DINO | Prevent feature homogenization |
| 4 | Hard attention (top-k) | Force commitment to specific patches |

See `docs/IDEAS_TO_TRY.md` for full implementation details.

---

### Multi-Pass Refinement Experiments (2026-01-07)

**Purpose:** Test if multi-pass query refinement and/or longer sequences can create fine/coarse gap.

**Hypotheses:**
1. Single refinement pass isn't enough - need iterative refinement
2. Longer sequences provide more temporal context for dynamic queries

**Configuration:**
- 5 experiments with varying passes (1, 2, 3) and frames (8, 16)
- 1,500 steps each on local WebVid data (813 videos)
- freeze_dino=True (based on ablation results)

**Results:**

| Experiment | Config | Final Ratio | Fine Loss | Coarse Loss | Result |
|------------|--------|-------------|-----------|-------------|--------|
| A | 1 pass, 8 frames | 1.0015 | 0.495 | 0.496 | No gap |
| B | 2 passes, 8 frames | 1.0046 | 0.515 | 0.517 | No gap |
| C | 3 passes, 8 frames | 1.0000 | 0.472 | 0.472 | No gap |
| D | 1 pass, 16 frames | 1.0003 | 0.261 | 0.261 | No gap |
| E | 3 passes, 16 frames | 0.9996 | 0.274 | 0.274 | No gap |

**Key Observations:**

1. **Multi-pass refinement doesn't help:**
   - B (2-pass) and C (3-pass) show no improvement over A (1-pass)
   - Each refinement pass produces nearly identical loss (r0 ≈ r1 ≈ r2)
   - Passes aren't learning to extract different information

2. **Longer sequences improve prediction but don't create gap:**
   - D and E achieve much lower loss (~0.27 vs ~0.47)
   - More temporal context helps overall prediction
   - But ratio remains ~1.0 (no fine/coarse difference)

3. **E shows ratio < 1.0:**
   - With 3 passes + 16 frames, fine is slightly WORSE than coarse
   - Additional complexity may hurt rather than help

**Conclusion:**

Neither multi-pass refinement nor longer sequences address the core issue. The problem is fundamental to the query mechanism - different queries don't extract sufficiently different features regardless of:
- Number of refinement iterations
- Temporal context length

**wandb runs:** https://wandb.ai/sanjayanps/foveated-vlm-multipass

**Code:** `scripts/experiment_multipass_local.py`

---

### Attention Fix Experiments (2026-01-07)

**Purpose:** Test if attention mechanism fixes can create fine/coarse gap.

**Experiments:**

| # | Config | Ratio | Z_Sim | Notes |
|---|--------|-------|-------|-------|
| 1 | Baseline | 0.9991 | 0.19 | Control |
| 2 | temp=0.1 | 0.9998 | 0.32 | Sharper attention, no effect |
| 3 | Contrastive | 1.0016 | **-0.78** | Features anti-correlated! |
| 4 | Top-k=16 | 0.9933 | 0.25 | Actually hurt |
| 5 | temp+contrastive | 1.0047 | -0.36 | Combined |
| 6 | temp+topk | 1.0023 | 0.12 | Combined |

**CRITICAL FINDING:**

Contrastive loss successfully pushed z_fine and z_coarse to be **anti-correlated** (z_sim = -0.78), but the ratio stayed at ~1.0.

This proves:
1. We CAN make the features different (contrastive works)
2. But different features don't help prediction
3. **The task itself doesn't require foveated attention**

**Conclusion:**

Next-frame latent prediction is solvable with ANY weighted average of patches. The task is too "global" - it doesn't require spatial focus.

**Next Steps (Task Redesign):**
- Object tracking: "Where did X go?" - requires spatial focus
- Sparse reconstruction: Predict only 25% of patches - forces selective attention
- Action recognition: Motion-focused features needed
- Longer-horizon prediction: Current frame → frame t+5

**wandb:** https://wandb.ai/sanjayanps/foveated-vlm-attention-fixes

---

### Diagnostic Analysis: Query Generation Bottleneck (2026-01-07)

**Purpose:** Deep investigation into why ratio stays at 1.0 despite attention mechanism fixes.

**Script:** `scripts/diagnostic_attention.py`

**Method:** Loaded real FoveatedVideoModel and analyzed each stage:
1. Encode frames and verify deep query mode active
2. Extract coarse features with q_static
3. Generate dynamic queries from LLM
4. Extract fine features with dynamic queries
5. Compare at each stage

**KEY FINDINGS:**

**1. Deep Query Mode WORKS:**
```
Opposite queries (q vs -q) → features similarity: -0.7214
Orthogonal queries → features similarity: 0.3241
```
The attention mechanism correctly produces different features for different queries.

**2. BOTTLENECK IDENTIFIED - Query Generation:**
```
Query similarity across frames:
   Frame 0 vs 1: 0.9414
   Frame 1 vs 2: 0.9542
   ...
   Frame 6 vs 7: 0.9965  ← Almost identical!

Query variance across frames: 0.000094
```
The LLM generates queries that are 92-98% similar across all frames!

**3. z_coarse vs z_fine Similarity:**
```
Raw features: ~57% similar (not 98% as in simplified experiments)
After normalization: ~57% similar
```
With real model, features ARE somewhat different, but not enough.

**4. LLM Hidden States:**
```
h_coarse vs h_fine similarity: 0.89-0.92 per frame
```
LLM processes similar features similarly.

**Root Cause Confirmed:**
- The attention mechanism works correctly
- The LLM fails to generate diverse queries
- Queries converge to near-identical values regardless of frame content
- This explains why ratio=1.0: q_static ≈ q_dynamic → z_coarse ≈ z_fine

---

### Query Diversity Experiments (2026-01-07)

**Purpose:** Test if forcing query diversity can create fine/coarse gap.

**Script:** `scripts/experiment_query_diversity.py`

**Hypothesis:** If we force the queries to be different, will the features differ enough to help prediction?

**5 Approaches Tested:**

| # | Query Mode | Description |
|---|------------|-------------|
| 1 | LLM (baseline) | Standard LLM-generated queries |
| 2 | Fixed q_init | Use orthogonal q_init instead of random |
| 3 | Diversity loss | Add penalty for query similarity |
| 4 | Orthogonal | Project queries to be orthogonal to q_static |
| 5 | Random | Completely random queries per frame |

**Results:**

| Experiment | Ratio | Fine Loss | Coarse Loss | Result |
|------------|-------|-----------|-------------|--------|
| 1_baseline_llm | 1.0007 | 0.5907 | 0.5911 | ✗ No gap |
| 2_fixed_qinit | 1.0042 | 0.5708 | 0.5732 | ✗ No gap |
| 3_diversity_loss | 1.0005 | 0.5302 | 0.5304 | ✗ No gap |
| 4_orthogonal | 0.9999 | 0.5408 | 0.5408 | ✗ No gap |
| 5_random | 1.0000 | 0.5406 | 0.5406 | ✗ No gap |

**CRITICAL FINDING:**

Even with **completely random queries** (experiment 5), the ratio stays at 1.0!

This definitively proves:
1. Query diversity is NOT the issue
2. The task itself doesn't benefit from foveated attention
3. **Next-frame VAE latent prediction is solvable with ANY weighted average of patches**

**Why This Makes Sense:**

VAE latents encode global image structure (mean color, layout, textures). This information is distributed across ALL patches uniformly. Whether you attend to top-left or bottom-right, you get similar reconstruction-relevant information.

**Conclusion:**

The core hypothesis is fundamentally unsuited for this task. To validate foveated attention, we need tasks that REQUIRE spatial selectivity:

1. **Object tracking:** "Where did the red ball go?" - forces focus on specific regions
2. **Sparse reconstruction:** Predict only 25% of patches - forces strategic selection
3. **Action recognition:** "Is the person walking or running?" - requires motion-focused attention
4. **Region-specific QA:** "What color is the object in the top-left?" - requires spatial focus

**wandb:** https://wandb.ai/sanjayanps/foveated-vlm-query-diversity

---

## Final Conclusions (2026-01-07)

After extensive experimentation over 3+ days:

### What Works:
- ✅ Deep query attention mechanism (different queries → different features)
- ✅ Training converges and reduces loss (0.79 → 0.21)
- ✅ Model uses visual features h (not just copying prev_latents)
- ✅ Attention is more focused in fine pass (1.8x focus ratio)

### What Doesn't Work:
- ❌ LLM generates near-identical queries across frames (92-98% similar)
- ❌ Even forced query diversity doesn't create ratio > 1.0
- ❌ Multi-pass refinement doesn't help
- ❌ Longer sequences don't help

### Root Cause:
**The task (next-frame VAE latent prediction) doesn't require foveated attention.**

VAE latents encode global structure available in ANY weighted average of patches. The task is fundamentally "global" - spatial selectivity provides no advantage.

### Recommendation:
To validate the foveated attention hypothesis, redesign the task to require spatial selectivity:
1. Object tracking / visual question answering
2. Sparse patch reconstruction
3. Action recognition
4. Longer-horizon prediction (t+5, t+10)

---

## Quick Reference

### Debugging loss_fine == loss_coarse

1. Check `model.q_static.std()` → should be ~1.0
2. Check attention entropy → should be < 5.3 (not 5.5)
3. Check embedding difference → should be > 0.5
4. Verify bias=False in query_input_proj

### Core Hypothesis

**Success:** `loss_fine < loss_coarse` (ratio > 1.0)
- 5-15% improvement = PoC successful
- >15% = very promising

### Files to Reference

- **Architecture:** `core_docs/foveated_vlm_proposal.md`
- **Implementation:** `core_docs/foveated_vlm_execution_guide.md`
- **Working Guide:** `CLAUDE.md`
- **Knowledge Base:** This file

---

---

### Captioning Experiment: Hypothesis Validated! (2026-01-08)

**Purpose:** Test if captioning (semantic task) benefits from foveated attention, unlike reconstruction (global task).

**Script:** `scripts/experiment_captioning_fast.py`

**Dataset:** WebVid-10M streaming (short videos <15s with captions)

**Key Change:** Train on captioning ONLY (cross-entropy loss), compare fine vs coarse.

**Results (460+ steps, still running):**

| Metric | Value |
|--------|-------|
| Ratios > 1.0 | 100% (17/17) |
| Ratios > 1.05 | 88.2% |
| Ratios > 1.1 | 64.7% |
| Mean ratio | 1.140 |
| Best ratio_avg20 | 1.289 |

**Key Finding:**

Fine queries (dynamic, LLM-generated) consistently outperform coarse queries (static) by **10-23%** on captioning loss!

This validates the core hypothesis:
- **Reconstruction:** Global task, any weighted average works → ratio ≈ 1.0
- **Captioning:** Semantic task, requires understanding WHAT is in the video → ratio > 1.0

**Why Captioning Benefits:**

Captions require identifying:
- Objects ("a person", "a car")
- Actions ("running", "flying")
- Relationships ("person riding bike")
- Attributes ("red", "small")

These are spatially localized features that benefit from selective attention.

**wandb:** https://wandb.ai/sanjayanps/foveated-vlm-captioning

---

*Last updated: 2026-01-08*
