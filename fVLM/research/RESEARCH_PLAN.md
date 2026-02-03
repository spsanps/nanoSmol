# Foveated VLM Research Plan

## Critical Issue: Training vs Inference Mismatch

### The Problem

Our current evaluation measures **training loss**, not **true inference loss**.

**Training (parallel approximation):**
```
1. Coarse pass: q_static → ALL frames (parallel) → z_coarse
2. z_coarse → LLM → generates ALL queries at once
3. Fine pass: shifted queries → ALL frames (parallel) → z_fine
4. z_fine → LLM → caption loss
```

**True Inference (autoregressive):**
```
1. Frame 0: q_init → DINO → z_0 → LLM → q_1
2. Frame 1: q_1 → DINO → z_1 → LLM → q_2
3. Frame 2: q_2 → DINO → z_2 → LLM → q_3
... purely sequential, NO coarse features used
```

**Key Difference:**
- Training: Queries derived from COARSE features (z_coarse)
- Inference: Queries derived from PREVIOUS FINE features (z_{t-1})

### Why This Matters

The whole thesis of foveated attention is that **queries adapt based on what was seen**.
But during training, queries are generated from coarse features, not fine features.

This means:
1. Our training loss may not reflect true inference performance
2. The query generation learns to transform COARSE features, not FINE features
3. There's potential for a train/inference gap

### Research Questions

1. **How big is the gap?** Compare training loss vs autoregressive inference loss
2. **Does it matter?** If gap is small, training approximation is validated
3. **Can we close it?** Multi-fine iterations (coarse→fine₁→fine₂) partially help

---

## Research Tasks

### Task 1: Implement Autoregressive Inference

Create `forward_autoregressive()` method that:
1. Processes frames one at a time
2. Generates query for frame t from LLM hidden state after seeing z_0...z_{t-1}
3. No coarse pass at all - purely fine features

**Implementation location:** `src/model/foveated_vlm.py`

**Pseudocode:**
```python
def forward_autoregressive(self, raw_frames, caption_ids, caption_mask):
    B, T = raw_frames.shape[:2]

    # Encode all frames with DINO (can be parallel - just patch features)
    patch_features, caches = self.encode_all_patches(raw_frames)

    # Process frames autoregressively
    z_fine_list = []
    hidden_states = []
    query = self.q_init.expand(B, -1)  # Start with initial query

    for t in range(T):
        # Extract features with current query
        z_t = self.encoder.query_attend(query, caches[t])
        z_fine_list.append(z_t)

        # Project to LLM space
        z_t_llm = self.dino_to_llm(z_t)
        z_t_llm = z_t_llm / (z_t_llm.std() + 1e-6) * self.visual_scale

        # Update LLM hidden state (incrementally)
        # ... LLM forward with KV cache

        # Generate query for next frame from LLM hidden
        if t < T - 1:
            query = self.llm_to_query(h_t)

    # Now compute caption loss with z_fine sequence
    ...
```

**Challenge:** LLM KV caching for incremental processing

### Task 2: Create Evaluation Script

Create `evaluate_inference_gap.py` that:
1. Loads a checkpoint
2. Computes training-style loss (`forward_captioning` with `use_fine=True`)
3. Computes autoregressive inference loss (`forward_autoregressive`)
4. Computes coarse baseline loss (`forward_captioning` with `use_fine=False`)
5. Reports the gap: training loss vs inference loss

**Metrics to report:**
- `loss_training_fine`: Current training loss (parallel approximation)
- `loss_inference_fine`: True autoregressive inference loss
- `loss_coarse`: Baseline with static query
- `gap = loss_inference_fine - loss_training_fine`: Train/inference gap

### Task 3: Run Gap Analysis

For existing checkpoints:
- S-S foveated: 100, 300, 1000, 3000 steps
- M-S foveated: 100, 300, 1000, 3000 steps

Analyze:
1. Does the gap increase or decrease with training?
2. Does larger LLM (M-S) reduce the gap?
3. Is multi-fine training helping close the gap?

### Task 4: Update Scaling Analysis

Once we have inference loss:
1. Re-plot scaling laws with INFERENCE loss (not training loss)
2. This is the true measure of model capability
3. Update KNOWLEDGE.md with corrected results

---

## Expected Outcomes

**Best case:** Gap is small (<5%), training approximation is validated
- Current results remain valid
- Foveated attention advantage is real

**Moderate case:** Gap is medium (5-15%)
- Training approximation has some cost
- Need to report both metrics
- May need more fine iterations in training

**Worst case:** Gap is large (>15%)
- Training approximation is poor
- May need different training approach
- Results need significant revision

---

## File Organization

```
fVLM/
├── src/model/foveated_vlm.py      # Add forward_autoregressive()
├── scripts/
│   └── evaluate_inference_gap.py  # New evaluation script
├── research/
│   ├── RESEARCH_PLAN.md           # This file
│   └── scaling_laws/
│       ├── results/
│       │   ├── training_loss/     # Current results
│       │   └── inference_loss/    # New results
│       └── plots/
│           └── gap_analysis.png   # Train vs inference gap
└── docs/KNOWLEDGE.md              # Update with gap analysis
```

---

## Timeline

1. **Task 1** (Implement autoregressive): ~2 hours
2. **Task 2** (Evaluation script): ~1 hour
3. **Task 3** (Run analysis): ~1 hour (GPU time)
4. **Task 4** (Update docs): ~30 min

Total: ~4-5 hours

---

## Success Criteria

1. `forward_autoregressive()` works correctly (outputs reasonable loss)
2. Gap analysis complete for all checkpoints
3. Scaling laws re-plotted with inference loss
4. KNOWLEDGE.md updated with findings
5. Clear conclusion about training approximation validity
