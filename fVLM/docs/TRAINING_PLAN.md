# Full Training Plan — Authoritative

**This is the SINGLE source of truth for all training decisions.** If other docs (SCALING_PLAN.md, GPU_PHASE1_PLAN.md, DATA_SOURCES.md) conflict with this, THIS wins.

**Last updated:** 2026-02-17

---

## Current Status

```
Phase 1a: Ablations           ✓ DONE (13 runs, config decisions locked)
Phase 1b: Scaling grid         ← IN PROGRESS (135M constant done ✓, 360M/1.7B LR sweep running)
  → Pick model size
Full training: Stage 1 → 2 → 3
Eval: 3-mode benchmarks
Release: HuggingFace + blog
```

---

## Phase 1a Results → Locked Decisions

| Decision | Winner | Evidence |
|----------|--------|----------|
| **Freeze strategy** | Full unfreeze | F1 (1.999), F2 (1.938) vs baseline (1.361) — frozen much worse |
| **Connector LR ratio** | 100:1 (connector=1e-3, backbone=1e-5) | LR2 (1.332) beat baseline 10:1 (1.361) |
| **Architecture** | Deep query + fine pass (full two-pass) | Kept for thesis. A1v2 shallow was competitive (1.349) but deep query needed at scale |
| **Data mix** | Video-heavy (55% video / 30% image / 15% text) | D1 (1.312) was best overall |
| **Model size** | TBD from Phase 1b scaling curves | 135M done (1.1923), 360M/1.7B LR sweep in progress |
| **Image frame replication** | **N=8** | A8 sweep: N=8 (2.858) > N=4 (2.868) > N=1 (3.027) > N=16 (3.049) |

---

## Phase 1b: What Remains

### Scaling

1. **135M constant-LR rerun** (~8h) — current 135M used cosine, 360M used constant. Can't compare until both match.
2. **Analyze scaling curves** → pick model size (135M vs 360M)
3. 1.7B deferred to A100 if budget allows

### Experiments blocking full training decisions

4. **A8 image frame replication: N=4 and N=8** — We have N=1 (A8a) and N=16 (A8b) from Phase 1a. Need intermediate points to find the sweet spot for `replicate_image_frames` in Stage 2-3. Run on Cauldron image data, same setup as A8a/A8b. There is NO multi-token experiment — the architecture is always foveated. Replicated frames with diverse dynamic queries IS how foveated handles images.

5. **Coarse vs fine on long videos only** — Validates fine pass where it should matter most.

   **Dataset:** Vript Long (400K samples, `/workspace/data/vript_long_shards/`). Filter to `frame_count >= 30` (30-64 frames, i.e. 30+ second clips). Vript Long is purpose-built as long video clips — better than OpenVid (avg ~7 frames, mostly short) or LLaVA-Video (67% truncated at 64-frame cap = lossy).

   **Runs:** baseline (deep+fine) vs A6-style (deep+coarse_only) on this filtered subset, same sample count, same config otherwise.

   **Hypothesis:** Fine pass gap widens with video length because more temporal diversity means static queries can't cover everything. At 300K mixed samples (short+long, Phase 1a) the gap was only 0.008 — it should be larger when isolated to long videos where temporal attention actually matters.

   **Paper result:** If coarse≈fine on short videos but fine>>coarse on long videos → "foveation matters most when temporal complexity is high." Clean, publishable finding.

### Priority order
Run (1) first (already planned), then (4) and (5) can run in parallel. Benchmarks come after full Stage 3 training.

---

## Full Training Pipeline

### Stage 1: Visual Alignment (video pre-training)

| Item | Spec |
|------|------|
| **Purpose** | Teach foveated query mechanism + visual grounding |
| **Loss** | All-text CE (predict all tokens) |
| **LR** | Converging: connector=1e-3 → 3e-5, backbone=1e-5 → 3e-5 (100:1 → 1:1) |
| **Freeze** | Nothing — full fine-tuning |
| **Samples** | ~1M (shorter Stage 1 — just alignment, not full training) |

**Data:**

| Dataset | Samples | Type | Notes |
|---------|---------|------|-------|
| OpenVid-1M | 905K | Video caption | Primary video data |
| WebVid (valid) | 19K | Video caption | Small supplement |
| SmolTalk S1 | ~280K (14%) | Text only | Instruction retention |

**Honest conditioning (prompt format):**

Stage 1 captions are noisy stock-video-ese ("4K aerial drone footage..."). Do NOT pretend these are natural descriptions. Frame the task honestly:

```
<|user|>What would be the WebVid caption for this video?<|end|>
<|assistant|>4K aerial drone footage of beautiful sunset over ocean waves stock video<|end|>
```

This teaches visual grounding without polluting the model's conversational style. Stage 2-3 teaches real description/QA skills via properly formatted data.

**Do NOT rewrite captions.** Expensive and unnecessary — the honest framing handles the distribution mismatch.

---

### Stage 2: Vision-Language SFT

| Item | Spec |
|------|------|
| **Purpose** | Learn to answer questions about images and video |
| **Loss** | Answer-only CE (mask user/system tokens) |
| **LR** | Flat 3e-5 all components (1:1, SmolVLM2 style) + cosine decay |
| **Freeze** | Full fine-tuning |
| **Samples** | ~1M |

**Data:**

| Dataset | Samples | Type | Notes |
|---------|---------|------|-------|
| Cauldron (full) | ~2.0M → subsample to ~850K | Image VQA | Already instruction-formatted |
| SmolTalk S2 | ~140K (14%) | Text only | Instruction retention |

**Prompt format:** Cauldron data is already `{"user": "...", "assistant": "..."}` formatted. Use as-is.

---

### Stage 3: Video SFT

| Item | Spec |
|------|------|
| **Purpose** | Temporal reasoning, narrative understanding |
| **Loss** | Answer-only CE |
| **LR** | Flat 3e-5 all components (1:1, SmolVLM2 style) + cosine decay |
| **Freeze** | Full fine-tuning |
| **Samples** | ~0.5M |
| **Mix target** | ~55% video / ~30% image / ~15% text (video-heavy, per D1 winner — not ablated for Stage 3) |

**Data:**

| Dataset | Samples | Type | Notes |
|---------|---------|------|-------|
| VISTA (main + extra) | ~237K | Video temporal QA | MIT license |
| Vript (long + short) | ~411K → subsample | Video caption | Temporal narration |
| ShareGPT4Video | ~37K | Video caption | Short clips |
| LLaVA YouTube | ~22K | Video SFT | Pre-tokenized |
| Cauldron (subsample) | ~image portion | Image VQA | Retain image capability |
| SmolTalk S3 | ~70K (14%) | Text only | Instruction retention |

**Prompt format:** These datasets are already instruction-formatted. Use as-is.

### Video Filtering Rules

**Exclude truncated videos from Stage 3.** Videos that hit the 64-frame cap (i.e., videos >64 seconds that got uniformly subsampled) lose temporal information. This matters most in Stage 3 where temporal reasoning is the goal.

- **LLaVA-Video-178K: EXCLUDED from Stage 3.** 67% of its videos hit the 64-frame cap → noisy signal for temporal reasoning.
- General rule: if a dataset has >50% cap-hitting videos, exclude from Stage 3 or filter to short clips only.
- Stage 1 is less sensitive (learning visual grounding, not temporal reasoning), so truncated videos are acceptable there.

---

### Optional: DPO

| Item | Spec |
|------|------|
| **Data** | RLAIF-V (83K image preference pairs) |
| **Method** | DPO with LoRA or full fine-tuning |
| **When** | After Stage 3, only if hallucination is a problem |

---

## Frame Handling

| Item | Spec |
|------|------|
| **Frame rate** | 1 FPS (constant, matches SmolVLM2) |
| **Frame count** | Variable: `min(video_duration_seconds, 64)` |
| **Long video fallback** | Videos >64s: uniform spacing (evenly spaced, still 64 max) |
| **Resolution** | 224×224 (DINOv2-small native) |
| **Images in training** | Replicated to **8 frames** (A8 sweep: N=8 best at 2.858, N=4 close at 2.868, N=1/16 worse) |

---

## Architecture (DO NOT CHANGE)

| Decision | Value | Why |
|----------|-------|-----|
| deep_query | True | Shallow = uniform attention = no foveation |
| query_dim | 384 | Matches DINO-small hidden dim |
| bias on query_input_proj | False | Bias dominated signal (BUG-002) |
| q_static/q_init init std | 1.0 | 0.02 killed gradients (BUG-001) |
| Mode selection | Per-batch | Per-sample broke DDP sync |
| visual_scale | 0.14 | Matches LLM embedding std |
| Loss | Text CE only | No reconstruction, no VAE, no auxiliary loss |
| Text retention | 14% SmolTalk in ALL stages | Removing hurts 3.7-6.5% (SmolVLM2 finding) |
| Connector LR | Stage 1: converging 100:1→1:1. Stage 2+: flat 1:1 (3e-5) | LR2 ablation + SmolVLM2 recipe |
| Data mix | Video-heavy | Ablation winner (D1) — Stage 1 only, Stage 2-3 not ablated |

---

## SmolTalk Text Retention

SmolTalk is split into 3 stages that match our training stages:

| Our Stage | SmolTalk Subset | Samples | Content |
|-----------|----------------|---------|---------|
| Stage 1 | SmolTalk S1 | ~280K | General instruction-following |
| Stage 2 | SmolTalk S2 | ~140K | Curated text Q&A |
| Stage 3 | SmolTalk S3 | ~70K | Refined instruction data |

**Critical:** Use curated text Q&A, NOT raw SmolTalk dump. Reusing LLM-SFT text hurts small VLMs (SmolVLM2 finding).

---

## Eval (AFTER Stage 3)

### Three Evaluation Modes

| Mode | Query Source | Execution | Purpose |
|------|-------------|-----------|---------|
| **Coarse-only** | Static q_static | Parallel | Lower bound |
| **Coarse→Fine** | Coarse → LLM → dynamic queries | Parallel | Training approximation |
| **Autoregressive Fine** | Sequential: q→frame→LLM→q→frame→... | Sequential | True deployment mode |

### Benchmarks

| Benchmark | Type | Status |
|-----------|------|--------|
| Video-MME | Video MCQ (2,700 Q) | Videos downloaded |
| MVBench | Video MCQ (4,000 Q) | Videos downloaded |
| MLVU | Video MCQ | Annotations only (videos deferred) |
| Val 10K | Mixed held-out | Ready |

### Baselines

| Baseline | Source | Purpose |
|----------|--------|---------|
| SmolVLM2-256M | HuggingFace (pretrained) | Direct param-count comparison |
| SmolVLM2-2.2B | HuggingFace (pretrained) | Quality upper bound |

---

## Go/No-Go Before Full Training

Before spending ~$250+ on full 3-stage training:

- [x] Foveated beats coarse-only — confirmed (small but consistent gap)
- [x] Val loss curves converge at ablation budget — yes
- [x] No divergence with full unfreeze — stable across all runs
- [ ] Scaling grid shows clear compute-optimal size — 135M done (1.1923), 360M/1.7B LR sweep running, then full scaling reruns
- [x] Training infrastructure stable — 13 successful ablation runs

---

## Budget Estimate

| Phase | Est. Cost | Status |
|-------|-----------|--------|
| Phase 1a: Ablations | ~$30 | ✓ Done |
| Phase 1b: Scaling grid | ~$40 | In progress |
| Stage 1 training | ~$40-60 | Pending |
| Stage 2 training | ~$25-35 | Pending |
| Stage 3 training | ~$20-30 | Pending |
| Eval + paper figures | ~$15 | Pending |
| Buffer (20%) | ~$40 | — |
| **Total** | **~$210-250** | ~$400 remaining of $500 |

---

## Document Index

| Doc | Role | Status |
|-----|------|--------|
| **TRAINING_PLAN.md** (this) | Authoritative training decisions | **CANONICAL** |
| DATA_SOURCES.md | Dataset provenance, citations, licenses | Reference (stage assignments defer to this doc) |
| GPU_PHASE1_PLAN.md | Phase 1 ablation + scaling design | Historical (Phase 1a decisions baked into this doc) |
| SCALING_PLAN.md | Original training plan (v3) | **SUPERSEDED by this doc** for training decisions |
| KNOWLEDGE.md | Experiment history, bugs, architecture notes | Reference |
| HANDOFF_2026-02-16.md | System migration snapshot | Temporal (will go stale) |
| runpod/BRIEFING.md | GPU pod setup context | Reference |
| CLAUDE.md | Claude Code project guide | Reference |

---

*Consolidates decisions from: SCALING_PLAN.md (Feb 9), GPU_PHASE1_PLAN.md (Feb 14), Phase 1a ablation results (Feb 16), and conversation context.*
