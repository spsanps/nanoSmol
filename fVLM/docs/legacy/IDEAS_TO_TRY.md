# Foveated VLM: Ideas to Try

**Date:** 2026-01-06
**Context:** After 24h training (26K steps), fine ≈ coarse loss despite attention being more focused.

---

## Experimental Findings (from quick_experiments.py)

### CRITICAL: Feature Similarity Problem
```
Cosine Similarity of extracted features:
  q_static vs q_init:   0.9875 (98.75% similar!)
  q_static vs random:   0.9899 (98.99% similar!)
```
**Despite different queries, extracted features are nearly identical!**

This directly explains why `loss_fine ≈ loss_coarse` - the features fed to the LLM are the same regardless of query.

### Attention is Too Uniform (Temperature Issue)
```
Temp     Entropy      Max Attn     Focus
0.1      4.967        0.0456       Moderate
1.0      5.772        0.0044       Uniform  ← Current
```
At temperature=1.0, attention is nearly uniform (max=5.78). Using temp=0.1 makes it more focused.

### Prediction Head Uses h, Not Just prev_latents
```
Sensitivity to perturbation:
  h perturbation:    0.0021 (4.7x higher!)
  prev perturbation: 0.0005
```
Good news: The model IS using visual features (h), not just copying prev_latents. The problem is that h_fine ≈ h_coarse because z_fine ≈ z_coarse.

---

## Current State

### What's Working
- Fine queries produce **1.8x more focused attention** (lower entropy, higher peak)
- Loss converged from 0.79 → 0.21 (74% improvement)
- Training is stable, no crashes
- Prediction head responds to visual features (h), not just prev_latents

### What's NOT Working
- `loss_fine ≈ loss_coarse` (core hypothesis not validated)
- **Extracted features are 98.75% similar despite different attention patterns**
- Focused attention doesn't translate to different features (the core problem)

### Root Cause Analysis (UPDATED)
The problem is **between attention and feature extraction**:
1. Queries are different (cosine sim = -0.015)
2. Attention patterns differ (entropy 5.1 vs 5.5)
3. **BUT extracted features are 98.75% similar** ← THE PROBLEM
4. So h_fine ≈ h_coarse
5. So loss_fine ≈ loss_coarse

Root causes:
1. **Uniform attention**: With temp=1.0, attention is nearly uniform, so all queries extract similar weighted averages
2. **DINO feature homogeneity**: Training DINO may have made features too similar across patches
3. **Single query bottleneck**: One query can only capture one "focus point" - may not be enough

---

## Ideas to Try

### 1. Remove/Reduce prev_latents Conditioning ⭐ HIGH PRIORITY

**Problem:** The FiLM prediction head receives `(h, z_vae_prev)`. The prev_latents likely dominate, making visual features irrelevant.

**Solution:**
```python
# Option A: Remove entirely
class PredictionHeadV2(nn.Module):
    def forward(self, h):
        # No prev_latents - force use of visual features
        return self.decoder(self.h_proj(h))

# Option B: Add noise to weaken conditioning
prev_noisy = z_vae_prev + 0.1 * torch.randn_like(z_vae_prev)
pred = pred_head(h, prev_noisy)

# Option C: Dropout on prev_latents
if self.training:
    prev_latents = F.dropout(prev_latents, p=0.5)
```

**Why it helps:** Forces model to actually USE foveated visual features.

**Effort:** Low
**Expected Impact:** High

---

### 2. Predict Frame DELTA Instead of Full Latent ⭐ HIGH PRIORITY

**Problem:** Full latent prediction is dominated by static content. Static background = easy to predict = no need for attention.

**Solution:**
```python
# Current: predict full latent
target = vae_latents[:, t+1]
loss = mse(pred, target)

# New: predict change from previous frame
delta_target = vae_latents[:, t+1] - vae_latents[:, t]
delta_pred = pred_head(h, z_vae_prev)

# Add L1 regularization to prevent collapse to zero
loss = mse(delta_pred, delta_target) + 0.01 * delta_pred.abs().mean()
```

**Why it helps:**
- Static scenes: delta ≈ 0, easy
- Motion: delta is large, requires attending to moving objects
- Model MUST focus on changes to predict correctly

**Note:** Proposal rejected this (Section 6.4) but we can add safeguards.

**Effort:** Medium
**Expected Impact:** High

---

### 3. Skip Connection: Query → Prediction Head

**Problem:** Query encodes WHERE to look, but this info doesn't reach prediction head.

```
Current flow:  query → attention → z → LLM → h → pred_head
                ↑                                    ↑
            WHERE info                          Lost here!
```

**Solution:**
```python
class PredictionHead(nn.Module):
    def __init__(self, ...):
        self.query_proj = nn.Linear(query_dim, 256)

    def forward(self, h, z_vae_prev, query=None):
        feat = self.encoder(z_vae_prev)
        film = self.h_to_film(h)

        # Inject query information
        if query is not None:
            query_feat = self.query_proj(query).view(-1, 256, 1, 1)
            feat = feat + query_feat

        ...
```

**Why it helps:** WHERE you looked should inform WHAT you predict (e.g., if looking at ball, predict ball motion).

**Effort:** Low
**Expected Impact:** Medium

---

### 4. Attention Temperature Scaling

**Problem:** Soft attention spreads weight across many patches, diluting the benefit of selective queries.

**Solution:**
```python
# Current (temperature = 1.0)
attn_weights = softmax(scores / sqrt(d))

# Sharper attention (temperature = 0.5)
attn_weights = softmax(scores / (sqrt(d) * 0.5))

# Learnable temperature
self.temperature = nn.Parameter(torch.tensor(1.0))
attn_weights = softmax(scores / (sqrt(d) * F.softplus(self.temperature)))
```

**Why it helps:** Sharper attention = more selective = bigger difference between fine and coarse.

**Effort:** Low
**Expected Impact:** Medium

---

### 5. Attention Entropy Regularization

**Problem:** No explicit incentive for fine to be MORE focused than coarse.

**Solution:**
```python
def compute_entropy(attn):
    return -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()

# In training loop:
entropy_coarse = compute_entropy(attn_coarse)
entropy_fine = compute_entropy(attn_fine)

# Fine should have LOWER entropy (more focused)
margin = 0.5  # bits
entropy_loss = F.relu(entropy_fine - entropy_coarse + margin)

loss = loss_main + 0.1 * entropy_loss
```

**Why it helps:** Explicitly rewards fine being more focused.

**Effort:** Low
**Expected Impact:** Medium

---

### 6. Separate Prediction Heads for Fine vs Coarse

**Problem:** Shared head might not learn to exploit different feature types.

**Solution:**
```python
# Current (shared)
self.pred_head = PredictionHead(...)
pred_coarse = self.pred_head(h_coarse, prev_latents)
pred_fine = self.pred_head(h_fine, prev_latents)

# New (separate)
self.pred_head_coarse = PredictionHead(...)
self.pred_head_fine = PredictionHead(...)
pred_coarse = self.pred_head_coarse(h_coarse, prev_latents)
pred_fine = self.pred_head_fine(h_fine, prev_latents)
```

**Why it helps:** Fine head can specialize in using focused features; coarse head can specialize in global features.

**Effort:** Low
**Expected Impact:** Low-Medium

---

### 7. Multi-Query Attention

**Problem:** Single query compresses all spatial info to one point. Too aggressive bottleneck.

**Solution:**
```python
# Instead of 1 query, use K queries (e.g., 4 or 9)
class MultiQueryEncoder:
    def __init__(self):
        self.num_queries = 4
        self.query_aggregator = nn.MultiheadAttention(...)

    def forward(self, queries, kv_cache):
        # queries: [B, K, D]
        z_list = []
        for k in range(self.num_queries):
            z_k = self.query_attend(queries[:, k], kv_cache)
            z_list.append(z_k)

        z_multi = torch.stack(z_list, dim=1)  # [B, K, D]

        # Aggregate with attention (LLM hidden as query)
        z_out = self.query_aggregator(h, z_multi, z_multi)
        return z_out
```

**Why it helps:** Multiple queries can capture different spatial regions; LLM decides which to use.

**Effort:** Medium
**Expected Impact:** Medium

---

### 8. Contrastive Loss: Fine vs Coarse

**Problem:** Fine and coarse features converge to be similar despite different attention.

**Solution:**
```python
# Add contrastive loss to PUSH fine and coarse apart
z_fine_flat = z_fine.view(B, -1)
z_coarse_flat = z_coarse.view(B, -1)

# Minimize cosine similarity
similarity = F.cosine_similarity(z_fine_flat, z_coarse_flat, dim=-1)
contrastive_loss = similarity.mean()

loss = loss_main + 0.1 * contrastive_loss
```

**Why it helps:** Forces fine and coarse to extract DIFFERENT information.

**Effort:** Low
**Expected Impact:** Medium

---

### 9. Freeze DINO Encoder

**Problem:** Training DINO causes both coarse and fine paths to converge to same features.

**Solution:**
```python
# Freeze DINO, only train:
# - query_input_proj (attention mechanism)
# - LLM
# - prediction head

for param in self.encoder.dino.parameters():
    param.requires_grad = False

# Only these are trainable:
# self.encoder.query_input_proj
# self.llm
# self.pred_head
```

**Why it helps:** Preserves pretrained DINO feature diversity; only attention selection is learned.

**Effort:** Low
**Expected Impact:** Medium

---

### 10. More Dynamic Dataset ⭐ HIGH PRIORITY

**Problem:** WebVid videos are mostly static (aerial views, slow motion). The proposal's "ball bouncing" example requires actual motion.

**Solution:**
```python
# Instead of WebVid-10M, try:

# Option A: Something-Something v2
# - 220K videos of humans doing actions
# - "Pushing X from left to right", "Picking up X"
# - Lots of motion, clear cause-effect

# Option B: Kinetics-400/700
# - 400/700 action classes
# - Sports, activities, interactions

# Option C: Epic-Kitchens
# - Egocentric (first-person view)
# - Cooking activities, hand movements
# - Very dynamic

# Option D: UCF-101
# - Smaller, good for quick experiments
# - 13K clips, 101 action classes
```

**Why it helps:** Dynamic videos have regions that MATTER for prediction. Static videos = all regions equally predictive.

**Effort:** High (new data pipeline)
**Expected Impact:** High

---

### 11. Predict Optical Flow Instead of VAE Latent

**Problem:** VAE latents encode global appearance; don't require spatial attention.

**Solution:**
```python
# Precompute optical flow targets
flow = compute_optical_flow(frame_t, frame_t+1)  # [B, 2, H, W]

# Predict flow instead of latent
flow_pred = self.flow_head(h)  # [B, 2, H, W]
loss = mse(flow_pred, flow)
```

**Why it helps:** Optical flow is INHERENTLY spatial - you must know WHERE objects are to predict WHERE they go.

**Effort:** Medium (need flow computation)
**Expected Impact:** High

---

### 12. Hard Attention (Gumbel-Softmax)

**Problem:** Soft attention still gives weight to all patches.

**Solution:**
```python
# Instead of soft attention:
attn_weights = softmax(scores / temperature)

# Use Gumbel-softmax for differentiable hard attention:
attn_weights = F.gumbel_softmax(scores, tau=0.5, hard=True)

# Or top-k hard attention:
topk_indices = scores.topk(k=16, dim=-1).indices
attn_mask = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)
attn_weights = softmax(scores) * attn_mask
attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
```

**Why it helps:** Forces model to commit to specific regions; maximizes difference between queries.

**Effort:** Medium
**Expected Impact:** Medium-High

---

## Priority Ranking (UPDATED based on experiments)

| Rank | Idea | Effort | Impact | Rationale |
|------|------|--------|--------|-----------|
| 1 | **Attention temperature (0.1)** | Low | HIGH | Experiments show temp=0.1 gives 10x more focused attention |
| 2 | **Contrastive loss** | Low | HIGH | Features 98.75% similar - MUST push apart |
| 3 | **Freeze DINO** | Low | HIGH | Stop feature homogenization, preserve diversity |
| 4 | **Hard attention (top-k)** | Medium | HIGH | Force commitment to specific patches |
| 5 | Entropy regularization | Low | Medium | Explicit focus incentive |
| 6 | Multi-query (4-9 queries) | Medium | Medium | Less aggressive bottleneck |
| 7 | Predict delta | Medium | Medium | Forces attention to motion (data-dependent) |
| 8 | ~~Remove prev_latents~~ | Low | LOW | Experiments show model USES h, not prev! |
| 9 | Dynamic dataset | High | Medium | Current model doesn't exploit attention anyway |
| 10 | Query→pred skip | Low | Low | Secondary issue |
| 11 | Separate heads | Low | Low | Secondary issue |
| 12 | Optical flow target | Medium | Medium | Different task, defer for now |

**Key insight:** The prev_latents hypothesis was WRONG. The model responds 4.7x more to h than prev_latents. The real problem is **features from different queries are too similar** (98.75% cosine similarity).

---

## Recommended Experiment Plan (UPDATED)

### Phase 1: Immediate Fixes (implement together)
1. **Temperature = 0.1** - 10x more focused attention, trivial change
2. **Freeze DINO** - prevent feature collapse during training
3. **Contrastive loss** - push z_fine and z_coarse apart

### Phase 2: If Phase 1 insufficient
4. **Hard attention (top-k=16)** - force patch selection
5. **Entropy margin loss** - explicit incentive for fine < coarse
6. **Multi-query (4 queries)** - reduce bottleneck

### Phase 3: Data/Task Changes (only if needed)
7. **Something-Something v2** - more dynamic videos
8. **Optical flow prediction** - inherently spatial task

---

## Success Criteria

After implementing fixes:
- `loss_fine < loss_coarse` by **>5%** consistently
- OR clear qualitative evidence that fine attention tracks important objects

If still no improvement after Phase 2, the **prediction task itself** may need rethinking (not just VAE latent prediction).

---

*Last updated: 2026-01-06*
