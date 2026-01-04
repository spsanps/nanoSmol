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

### Large-Scale Run (WebVid-10M streaming)
- In progress as of 2026-01-04
- Bugs fixed, awaiting results

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

*Last updated: 2026-01-04*
