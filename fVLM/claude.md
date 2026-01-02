# Claude Working Guide: Foveated VLM Project

## Critical References

**AUTHORITATIVE DOCUMENTS** (Adhere strictly):
1. `core_docs/foveated_vlm_proposal.md` - Architecture specification
2. `core_docs/foveated_vlm_execution_guide.md` - Implementation guide

**Always refer to these docs before making architectural or implementation decisions.**

---

## Code Organization

```
fVLM/
├── src/                      # Core library (imports as modules)
│   ├── model/
│   │   ├── foveated_vlm.py   # Main model class (FoveatedVLM)
│   │   ├── encoder.py        # Vision encoder (FoveatedEncoder)
│   │   └── prediction.py     # Prediction head (FiLM-conditioned)
│   ├── data/
│   │   ├── dataset.py        # WebVid dataset
│   │   ├── streaming_dataset.py
│   │   └── sampling.py       # Frame sampling utilities
│   └── training/
│       └── visualization.py  # Attention & prediction viz
│
├── scripts/                  # Runnable scripts
│   ├── train_multitask.py    # MAIN: Multi-task training (reconstruction + caption)
│   ├── train_phase1.py       # Phase 1: Reconstruction only
│   ├── train_phase2.py       # Phase 2: Text-conditioned
│   ├── test_captioning.py    # Evaluate captioning
│   ├── visualize_attention.py
│   ├── analyze_results.py
│   ├── setup/                # Data & model setup
│   │   ├── download_models.py
│   │   ├── download_webvid.py
│   │   └── precompute_latents.py
│   └── archive/              # Old/experimental (don't use)
│
├── configs/                  # Training configs (YAML)
├── core_docs/                # AUTHORITATIVE design docs
├── docs/                     # Analysis, handoff, experiments
├── outputs/                  # Training outputs (gitignored)
├── logs/                     # Log files (gitignored)
└── data/                     # Data files (gitignored)
```

**Key Entry Points:**
- Training: `python scripts/train_multitask.py --config configs/train_webvid.yaml`
- Evaluate: `python scripts/test_captioning.py`
- Setup: `python scripts/setup/download_models.py`

---

## Project Scope

### What We're Building
A novel vision-language model that:
- Processes video frame-by-frame with ONE token per frame (not 196+ patches)
- LLM controls WHERE to look in each frame (biological foveated attention)
- Two-pass parallel training: static query planning → dynamic query execution
- Training objective: next-frame VAE latent prediction

### Core Hypothesis
**Success metric:** `loss_fine < loss_coarse`
- Pass 2 (dynamic queries) should outperform Pass 1 (static query)
- This validates that foveated attention extracts better information

### Model Stack (DO NOT DEVIATE)
- **Vision Encoder:** DINOv3 ViT-S/16 (384-dim, trainable)
- **Core LLM:** SmolLM2-135M-Instruct (576-dim hidden)
- **Reconstruction Target:** Stable Diffusion VAE (frozen)
- **Dataset:** LLaVA-Video-178K (0-30s academic subset)

---

## CRITICAL: GPU Efficiency & Memory Management

### Hardware Constraints
- **GPU:** Single RTX 4090 (24GB VRAM, ~20GB usable)
- **Failure mode:** VRAM overflow stops everything - must avoid at all costs

### Proactive Memory Management Strategy

**1. Conservative Configuration (Start Here)**
```python
batch_size = 2          # Never start higher
grad_accum = 4          # Effective batch = 8
num_frames = 8          # Can reduce to 4 if needed
dtype = bfloat16        # Always use mixed precision
```

**2. Before Full Training**
- Run memory profiler on single batch (see Appendix A in execution guide)
- Verify peak memory < 18GB (leaves 2GB headroom)
- Test gradient accumulation loop without OOM

**3. During Training - Active Monitoring**
- Log `torch.cuda.max_memory_allocated()` every 100 steps
- Set up alerts if memory > 18GB
- Monitor GPU utilization (`nvidia-smi dmon -s mu`)
- If utilization < 80%, something's inefficient - investigate

**4. OOM Mitigation Hierarchy (Apply in Order)**
```python
# Level 1: Reduce batch size
batch_size = 1, grad_accum = 8

# Level 2: Reduce sequence length
num_frames = 4  # Instead of 8

# Level 3: Gradient checkpointing
llm.gradient_checkpointing_enable()

# Level 4: More aggressive settings
max_text_tokens = 32  # Truncate text context
```

**5. Efficiency Checks**
- ✓ VAE latents precomputed (not computed during training)
- ✓ DINO KV caching implemented correctly
- ✓ No unnecessary `.cpu()` / `.cuda()` transfers in training loop
- ✓ DataLoader num_workers set appropriately (2-4)
- ✓ Pin memory enabled in DataLoader

**6. Red Flags - Stop Immediately If:**
- Memory usage creeping up over time (memory leak)
- Training slower than expected (check GPU utilization)
- Frequent OOM crashes (batch size too high)
- Loss becomes NaN (gradient explosion)

---

## Implementation Milestones

### Phase 1: Setup (Milestone 1)
- [ ] Project structure created
- [ ] Download pretrained models (DINOv3, SmolLM2, SD-VAE)
- [ ] Download dataset (start with videos_1.tar.gz)
- [ ] Implement frame sampling
- [ ] Precompute VAE latents
- [ ] DataLoader working, verify shapes

**Checkpoint:** Can load batch with correct shapes, no errors

### Phase 2: Model (Milestone 2)
- [ ] FoveatedEncoder with query mechanism
- [ ] Two-pass forward pass (Pass 1 + Pass 2)
- [ ] PredictionHead (FiLM conditioning)
- [ ] Loss computation (loss_fine + lambda_coarse * loss_coarse)
- [ ] Single training step works, gradients flow

**Checkpoint:** Single step runs, loss decreases, no NaN, memory < 18GB

### Phase 3: Training (Milestone 3)
- [ ] Full training loop with logging
- [ ] Checkpoint saving/loading
- [ ] Attention visualization code
- [ ] Training runs stably for 1K steps
- [ ] Verify loss_fine vs loss_coarse divergence

**Checkpoint:** Stable training, attention patterns non-uniform

### Phase 4: Analysis (Milestone 4)
- [ ] Attention visualizations
- [ ] Decode predictions to pixels
- [ ] Loss analysis plots
- [ ] Write technical report

---

## Key Principles

1. **Architecture Fidelity:** Follow proposal document exactly
   - Query-guided attention via asymmetric masking
   - Two-pass structure with shared prediction head
   - Auxiliary loss on Pass 1

2. **Memory First:** Always verify memory before scaling up
   - Profile first, train second
   - Conservative defaults, scale carefully

3. **Gradient Flow:** Verify end-to-end differentiability
   - Auxiliary loss provides short path to q_static
   - Main loss flows: Pass 2 → Pass 1 → encoder

4. **Training Stability:** Monitor for collapse modes
   - Attention entropy (should be lower for dynamic vs static)
   - Loss ratio (loss_coarse / loss_fine should be > 1.0)
   - Gradient norms (should be stable)

5. **Incremental Development:** Test each component independently
   - DataLoader → Model forward → Single step → Full training
   - Don't build everything then debug

---

## Success Criteria

**Primary:** `loss_fine < loss_coarse` consistently after warmup
- 5-15% improvement = PoC successful
- >15% improvement = very promising

**Secondary:**
- Attention patterns track moving objects (visualizations)
- Training stable (no NaN, no divergence)
- Predictions improve over training (qualitative assessment)

---

## When in Doubt

1. Check the proposal doc for architecture details
2. Check the execution guide for implementation specifics
3. Prioritize memory efficiency over speed
4. Test small before scaling up
5. Visualize attention early and often

---

## Common Pitfalls to Avoid

- ❌ Starting with large batch size without profiling
- ❌ Skipping VAE latent precomputation
- ❌ Not monitoring memory during training
- ❌ Implementing architecture differently than proposal
- ❌ Ignoring attention collapse (uniform attention)
- ❌ Not logging loss_fine vs loss_coarse separately

---

*Last updated: 2026-01-02*
*For questions, refer to core_docs/ or consult with team*
