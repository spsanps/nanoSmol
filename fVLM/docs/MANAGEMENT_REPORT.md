# Foveated VLM: Management Report

**Date:** 2026-01-02
**Project:** Foveated Vision-Language Model Proof-of-Concept
**Hardware:** 1x RTX 4090 (24GB VRAM)
**Training Duration:** ~20 hours total across phases

---

## 1. Executive Summary

**Primary finding:** The dynamic foveated attention mechanism successfully learns to focus on different spatial regions compared to static attention (32x more focused on average), but this **does not translate into improved reconstruction loss**. The core hypothesis (`loss_fine < loss_coarse`) was **NOT validated** - both losses remained nearly equal (ratio ~1.0) throughout training.

**Bottom line:** The attention mechanism works mechanically, but the model has not learned to extract more useful visual information through dynamic attention. The model has severe representation collapse and weak visual grounding, generating stylistically correct but semantically inaccurate captions.

---

## 2. Results Summary

### 2.1 Primary Success Metric

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| `loss_fine < loss_coarse` | Ratio > 1.0 | Ratio = 1.0 | **NOT MET** |
| Improvement % | 5-15% (PoC success) | ~0% | **FAILED** |

**Interpretation per execution guide:**
- < 0%: Foveation hurts → Not observed
- 0-5%: Marginal improvement → **WE ARE HERE (at 0%)**
- 5-15%: Solid improvement (PoC successful) → Not achieved
- > 15%: Strong improvement → Not achieved

### 2.2 Secondary Metrics

| Metric | Good Sign | Observed | Status |
|--------|-----------|----------|--------|
| Attention entropy (dynamic vs static) | Dynamic lower | 0.015 vs 0.018 | **PASS** |
| Attention peak ratio | Fine > Coarse | 32x average (up to 126x) | **PASS** |
| Attention tracks objects | Yes | Partial - content dependent | **PARTIAL** |
| Loss curve | Smooth decrease | Stable, 0.821 → 0.713 | **PASS** |
| Gradient norms | Stable | Stable after fixes | **PASS** |

### 2.3 Caption Quality Analysis

| Metric | Fine (Dynamic) | Coarse (Static) | Notes |
|--------|----------------|-----------------|-------|
| Repetition | 38.2% | 33.4% | Both high - worse for Fine |
| Tech Spam | 11.1% | 11.4% | Both problematic |
| Diversity | 0.485 | 0.541 | Coarse slightly better |
| Fine-Coarse Similarity | 0.09 | - | Very different outputs |
| Match to Ground Truth | ~15% partial | ~15% partial | Both poor |
| Complete Miss Rate | ~85% | ~85% | Critical failure |

---

## 3. Training History

### Phase 1: Reconstruction Only (WebVid)
- **Duration:** ~4 hours
- **Steps:** 50,000
- **Epochs:** 1,230 (severe overfitting)
- **Result:** Attention collapsed to uniform - failed

### Phase 2: Text-Conditioned (LLaVA-Video)
- **Duration:** ~5 hours
- **Steps:** 12,000
- **Epochs:** 16
- **Result:** Non-uniform attention learned, loss improved 13%
- **Loss ratio:** Stayed at 1.0 (fine = coarse)

### Phase 3: Multi-task (Reconstruction + Captioning)
- **Duration:** ~11 hours
- **Steps:** 15,000
- **Samples:** 30,008
- **Result:** Caption generation works syntactically but not semantically
- **Loss ratio:** Still 1.0 (fine = coarse)

---

## 4. Key Visualizations & Findings

### 4.1 Attention Patterns

**Generated:** 64 diverse video examples with attention visualization

**Positive findings:**
- Fine attention IS more focused (32x peak attention vs coarse average)
- Fine entropy lower (0.015) vs coarse (0.018)
- Attention adapts temporally (changes across frames)
- Content-dependent: Works better on clear objects (squirrel, tractor) vs abstract scenes

**Negative findings:**
- Despite focused attention, reconstruction quality identical
- Attention doesn't consistently track moving objects
- Max attention only 2-3x uniform baseline (could be sharper)

### 4.2 Caption Comparison with Ground Truth

**Sample results (showing disconnect):**

| Ground Truth | Fine Caption | Coarse Caption |
|--------------|--------------|----------------|
| "Aerial shot winter forest" | "fisherman fishing in a rod in the sea" | "Aerial view of a beautiful lake" |
| "Red squirrel on tree branch" | "Boy with white hair sitting on bench" | "Woman's hand holding flowers" |
| "Yacht sailing at horizon" | "tourists walk along beach in Korea" | "young woman with beard in cafe" |
| "Aquarium fish on white background" | "Girl doing stretching in gym" | "young woman with laptop on couch" |

**Pattern:** Model generates fluent WebVid-style captions but with **near-random visual content**.

### 4.3 Representation Collapse

**Evidence:**
- Cosine similarity between visual embeddings: 0.88 → 0.96 during training
- All visual tokens becoming increasingly similar
- LLM receives nearly identical inputs regardless of video content

---

## 5. Root Cause Analysis

### Why `loss_fine ≈ loss_coarse`?

1. **Representation Collapse:** Visual embeddings become too similar (cos sim 0.96)
   - Both coarse and fine queries extract essentially the same information
   - The bottleneck (1 token per frame) may be too severe

2. **Weak Visual Grounding:**
   - Model learned WebVid caption style (dates, resolutions, "4k footage")
   - Caption generation doesn't depend on visual input
   - VAE latent prediction may not require fine-grained attention

3. **Information Bottleneck:**
   - 1 token per frame is extremely compressed (256 patches → 1 token)
   - Dynamic attention may not help when bottleneck is too tight
   - Both queries may extract the "gist" equally well

4. **Training Objective Mismatch:**
   - Reconstruction loss rewards overall frame similarity
   - Doesn't specifically reward capturing dynamic/salient regions
   - Coarse "average" representation may be sufficient

---

## 6. What Worked

1. **Non-uniform attention learned** (vs Phase 1 collapse)
2. **Stable training** after gradient fixes
3. **Text conditioning** influences attention patterns
4. **Caption generation** is fluent (stylistically correct)
5. **Attention mechanism** is differentiable and trainable
6. **Streaming data pipeline** enables large-scale training

---

## 7. What Didn't Work

1. **Core hypothesis not validated** - dynamic attention doesn't improve loss
2. **Semantic accuracy** of captions is poor (~85% miss rate)
3. **Visual grounding** is weak - captions don't match content
4. **Representation collapse** - embeddings too similar
5. **Motion tracking** - attention doesn't consistently follow objects

---

## 8. Failure Modes Observed

| Failure Mode | Symptoms | Observed |
|--------------|----------|----------|
| Query collapse | All attention uniform | Fixed in Phase 2 |
| Static query collapse | q_static attention uniform | Fixed |
| Mode token dominance | loss_fine ≈ loss_coarse | **YES - CURRENT STATE** |
| NaN loss | Training crashes | Fixed with grad clipping |
| OOM | CUDA out of memory | Fixed with batch tuning |
| Representation collapse | Visual embeddings similar | **YES - CRITICAL** |

---

## 9. Comparison to Success Criteria

### From Execution Guide:

> **Success metric:** `loss_fine < loss_coarse` consistently after warmup

**Result:** NOT MET. Ratio stayed at 1.0 throughout all training phases.

> **5-15% improvement = PoC successful**

**Result:** 0% improvement. PoC NOT successful by this criterion.

### However:

The **attention mechanism itself works** - dynamic queries produce meaningfully different (and more focused) attention patterns. The failure is in translating this into better representations.

---

## 10. Resource Assessment

### What 20 GPU-hours Achieved:
- Validated architecture is trainable
- Proved attention mechanism learns non-uniform patterns
- Identified representation collapse as key failure mode
- Identified visual grounding as major gap
- Generated comprehensive analysis and visualizations

### What More Compute Would Enable:

| Experiment | GPU Hours | Expected Outcome |
|------------|-----------|------------------|
| Contrastive learning on attention | 20h | May fix representation collapse |
| Attention entropy regularization | 10h | Sharper attention patterns |
| Larger query dimension (384→768) | 20h | More capacity per token |
| 2-4 tokens per frame | 30h | Relaxed bottleneck |
| Different reconstruction target | 20h | May require different attention |
| Video-QA downstream task | 40h | Proper evaluation of utility |

---

## 11. Recommendations

### Immediate (No additional compute):
1. **Document findings** - Current state is well-analyzed
2. **Archive checkpoints** - Preserve for future experiments
3. **Publish negative results** - Valuable to community

### Short-term (10-20 GPU hours):
1. **Add contrastive loss** between coarse/fine representations
2. **Increase tokens per frame** from 1 to 2-4
3. **Try attention entropy regularization**
4. **Evaluate on downstream task** (Video-QA) instead of reconstruction

### Long-term (50+ GPU hours):
1. **Different architecture** - May need multiple queries per frame
2. **Different training objective** - Reconstruction may be wrong target
3. **Larger models** - 135M LLM may lack capacity
4. **Scale up** - Full LLaVA-Video-178K with streaming

---

## 12. Decision Points for Management

### Option A: Pivot Architecture
- Change from 1 token/frame to multiple tokens
- Expected outcome: Better reconstruction, potentially validated hypothesis
- Risk: May lose foveation benefit entirely

### Option B: Change Training Objective
- Train on Video-QA or action recognition instead of reconstruction
- Expected outcome: Forced to use visual content for answers
- Risk: Larger compute requirements

### Option C: Add Regularization
- Contrastive loss between fine/coarse, attention entropy loss
- Expected outcome: May fix representation collapse
- Risk: May not address fundamental bottleneck issue

### Option D: Document and Publish
- Write up negative results
- Expected outcome: Contribution to research community
- Risk: No further progress

### Recommended: Option C first (10-20 hours), then Option B if unsuccessful.

---

## 13. Appendix: Technical Configuration

### Model Architecture
- **Vision Encoder:** DINOv2-small (384-dim, 22M params)
- **LLM:** SmolLM2-135M-Instruct (576-dim hidden)
- **Query Dim:** 128 (changed from 384 for efficiency)
- **Total Params:** 186.9M

### Training Configuration
- **Batch Size:** 2 (effective 8 with grad_accum=4)
- **Learning Rate:** 1e-4
- **Frames per Video:** 16
- **Frame Size:** 256x256
- **Precision:** bfloat16
- **Optimizer:** AdamW

### Dataset
- **Source:** LLaVA-Video-178K, WebVid-10M (streaming)
- **Videos Processed:** ~30,000
- **Caption Source:** WebVid metadata

---

## 14. Files Reference

| File | Purpose |
|------|---------|
| `outputs/multitask/checkpoints/final.pt` | Final trained model (606MB) |
| `outputs/attention_full/` | 20 attention visualizations |
| `outputs/diverse_64/` | 64 diverse example GIFs |
| `outputs/caption_comparison/` | 15 ground truth comparisons |
| `outputs/caption_eval/QUALITY_ANALYSIS.md` | Caption quality metrics |
| `docs/HANDOFF.md` | Technical handoff documentation |
| `core_docs/foveated_vlm_proposal.md` | Architecture specification |
| `core_docs/foveated_vlm_execution_guide.md` | Implementation guide |

---

## 15. Temporal/Speed Sensitivity Analysis

### Experiment Design
Tested how model captions change when given different numbers of frames from the same video:
- **4 frames (sparse):** Simulates fast-forwarded video
- **8 frames (uniform):** Medium density
- **16 frames (dense):** Full temporal detail
- **First half / Second half:** Tests which part of video dominates caption

### Results

| Configuration | Frames | Avg Overlap with 16-frame |
|---------------|--------|---------------------------|
| 4 frames (sparse) | 4 | **0.16** |
| 8 frames (uniform) | 8 | **0.09** |
| 16 frames (dense) | 16 | 1.00 |
| First 8 frames | 8 | 0.13 |
| Last 8 frames | 8 | 0.11 |

### Key Findings

1. **Captions CHANGE significantly with frame count:**
   - Only 16% word overlap between 4-frame and 16-frame captions
   - Model IS sensitive to temporal density

2. **Fine-Coarse overlap is very low (0.10):**
   - Fine and Coarse produce DIFFERENT captions
   - This confirms the attention mechanism IS working differently

3. **Both halves contribute equally:**
   - First half overlap: 0.13, Second half: 0.11
   - No temporal bias toward beginning or end

### Critical Insight

**The model is sensitive to input, but produces equally WRONG outputs regardless of frame count.**

Example from Video 1026437312:
- **4 frames:** "Time lapse of clouds moving through the sky..."
- **8 frames:** "Aerial view of a car driving through the city..."
- **16 frames:** "Dentist on the street in the city..."
- **First half:** "Aerial drone view of a white and green grass field..."
- **Second half:** "Man and woman walking on the street..."

**All 5 captions describe completely different content - the model is generating random outputs that change with input but don't accurately reflect the visual content.**

### Implication for Core Hypothesis

This confirms the root cause: **representation collapse + weak visual grounding**
- The model's outputs are sensitive to input variations
- But outputs don't correspond to actual visual content
- The attention mechanism works, but extracted representations are not semantically meaningful

---

**Prepared by:** Claude Code
**Date:** 2026-01-02
**Status:** Complete
