# Full Training Plan — Authoritative

**This is the SINGLE source of truth for all training decisions.** If other docs (SCALING_PLAN.md, GPU_PHASE1_PLAN.md, DATA_SOURCES.md) conflict with this, THIS wins.

**Last updated:** 2026-02-20

---

## Current Status

```
Phase 1a: Ablations           ✓ DONE (13 runs, config decisions locked)
Phase 1b: Scaling grid         ✓ DONE (135M selected, 360M LR=7e-4, 1.7B deferred/OOM)
Full training: Stage 1          ✓ DONE (1M samples, 4.5h, val_loss=1.2358)
Full training: Stage 2          ✓ DONE (1M samples, 7.6h, train_loss=0.040)
Full training: Stage 3 (DPO)    ← FINISHING (RLAIF-V 83K, step 1500/2593, resuming)
Eval: 3-mode benchmarks         Pending
Release: HuggingFace + blog     Pending
```

---

## Changes Made During Training (Feb 18-20)

These are decision changes from the original plan, made during actual training:

| Change | Old | New | Rationale |
|--------|-----|-----|-----------|
| **Val eval** | Val every 500 steps on val_10k | Train loss only (no val) for Stages 2-3 | Val set mismatch: val_10k is mixed data, misleading when training distribution differs. Train loss rankings matched val rankings in all experiments except longvid. Saves ~6min/1M samples. |
| **Stage 2 data** | Cauldron-only (image VQA) | Cauldron + ALL video data (~55% image, ~45% video) | Video SFT folded into Stage 2 instead of separate Stage 3. Maximizes data utilization. |
| **Stage 3** | Video SFT (VISTA + Vript + ShareGPT4Video) | DPO on RLAIF-V (83K preference pairs, beta=0.1) | Video data moved to Stage 2. Stage 3 becomes preference alignment. |
| **Stage 1 workers** | 2 workers, prefetch 2 | 6 workers, prefetch 4 | Throughput optimization (BUG-011 fix) |
| **Stage 1 system prompt** | SmolLM default | "You are a helpful AI assistant." | Override SmolLM2's "named SmolLM, trained by Hugging Face" |
| **Text retention (Stage 1)** | All text wrapped in WebVid prompt | Proper chat format for text-only samples | Bug: SmolTalk was getting "What would be the WebVid caption?" prompt |
| **torch.compile** | Enabled | Disabled for all stages | 135M too small: 40% throughput regression. Dynamic frame counts cause recompilation. |
| **Longvid conclusion** | "Coarse-only wins" | INCONCLUSIVE | Train losses equivalent (fine=1.981, coarse=1.994). Val difference was distribution artifact. |

### Config (yaml) changes with rationale:

**stage1_135M.yaml:**
- `num_workers: 2→6, prefetch_factor: 2→4` — throughput bottleneck was data loading, not GPU. Fixed ~30% speedup.
- `eval.every_steps: 250→500` — evals were too frequent, wasting ~6 min per 1M samples.
- `compile: false` — was already false but comment clarified: 135M too small, dynamic frame counts cause recompilation.

**stage2_135M.yaml:**
- `train_shards: single cauldron → list of ALL data sources` — video SFT data folded into Stage 2 (Cauldron + OpenVid + WebVid + VISTA + Vript + ShareGPT4Video + LLaVA YouTube). Rationale: avoids a separate video SFT stage, model sees everything in one pass.
- `val_shards: removed` — no validation. Train loss only. Val set mismatch makes val_loss misleading when training distribution is broader than val_10k.
- `compile: true→false` — 135M model is too small for torch.compile, 40% throughput regression measured.
- `metric: val_loss→train_loss` — follows from removing val.

**stage3_135M.yaml (complete rewrite):**
- Was: Video SFT (VISTA + Vript + ShareGPT4Video + Cauldron + SmolTalk, 500K samples, answer-only CE, LR 3e-5)
- Now: DPO on RLAIF-V (83K preference pairs, beta=0.1, LR 1e-6, frozen reference model)
- Rationale: Video data moved to Stage 2. Stage 3 becomes preference alignment. DPO LR is 30x lower than SFT (standard practice). Batch size halved (4 from 8) because DPO processes chosen+rejected per sample (2x memory).
- `text_shards/text_ratio: removed` — DPO doesn't use text retention interleaving.
- `loss.type: text_ce_answer_only→dpo` — triggers DPO training loop with reference model.

### Code changes (uncommitted, +812 lines):
- `train.py`: DPO training loop, reference model builder, DPO loss function (+285 lines)
- `foveated_vlm.py`: `forward_dpo()` method — shared DINO encoding for chosen/rejected (+129 lines)
- `collate.py`: `collate_dpo()` for preference pair batching (+81 lines)
- `webdataset_loader.py`: `create_dpo_webdataset()` + DPO sample decoding (+255 lines)
- `precompute.py`: Updates for RLAIF-V shard generation (+50 lines)

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

## Phase 1b: Results (DONE)

| Experiment | Result | Status |
|-----------|--------|--------|
| 135M constant-LR | val_loss=1.1923, 82 eval points | Done |
| 360M LR sweep | 3e-4: 1.631, 5e-4: 1.530, **7e-4: 1.497** | Done, 7e-4 wins |
| 1.7B LR sweep | OOM on RTX 5090 (32GB), even bs=1 | Deferred to A100 |
| A8 frame replication | N=8 (2.858) > N=4 (2.868) | Done, N=8 locked |
| Longvid fine vs coarse | INCONCLUSIVE (train losses equivalent, val artifact) | Done |

### Val Set Mismatch Finding
val_10k is ~49% video (mean 5.7 frames), ~51% image/text. Train-loss rankings agree with val-loss rankings for ALL experiments EXCEPT longvid (trained on 30-64 frame videos, evaluated on mixed). Lesson: cross-check val conclusions against train loss when distributions differ.

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

**System prompt:** All stages use an explicit system message: `"You are a helpful AI assistant."` This overrides SmolLM2's default chat template which injects "named SmolLM, trained by Hugging Face."

**Per-source conditioning (prompt format):**

Different datasets have different caption/response styles. Each source gets a prompt that matches its output style, so the model learns *when* to use each style rather than conflating them:

| Source | User Prompt | Response Style |
|--------|------------|----------------|
| OpenVid | "Write a brief caption for this video." | Stock video captions ("4K aerial drone footage...") |
| WebVid | "What would be the WebVid caption for this video?" | Stock video captions (honest conditioning) |
| Vript | "Provide a detailed narration of what happens in this video." | Temporal narrations ("The clip begins with...") |
| ShareGPT4Video | "Describe what happens in this video in detail." | Detailed video descriptions |
| VISTA / Cauldron / LLaVA YouTube | *(use existing user field)* | Proper QA — already instruction-formatted |
| SmolTalk | *(use existing user field)* | Chat responses — already instruction-formatted |

Stage 1 captions are noisy stock-video-ese. The per-source prompts teach visual grounding without polluting the model's conversational style. Stage 2-3 teaches real description/QA skills via properly formatted data.

**Do NOT rewrite captions.** Expensive and unnecessary — the per-source conditioning handles the distribution mismatch.

---

### Stage 2: Vision-Language SFT (DONE)

| Item | Spec |
|------|------|
| **Purpose** | Learn to answer questions about images AND video |
| **Loss** | Answer-only CE (mask user/system tokens) |
| **LR** | Flat 3e-5 all components (1:1, SmolVLM2 style) + cosine decay |
| **Freeze** | Full fine-tuning |
| **Samples** | ~1M |
| **Result** | train_loss=0.040 @ step 27000, 7.6h on RTX 5090 |
| **Eval** | Train loss only (no val — see val mismatch finding) |

**Data (CHANGED from original plan — video folded in):**

| Dataset | Samples | Type | Notes |
|---------|---------|------|-------|
| Cauldron (full) | ~2.0M | Image VQA | Already instruction-formatted |
| OpenVid-1M | 905K | Video caption | From Stage 1 data |
| WebVid | 19K | Video caption | Small supplement |
| VISTA (main + extra) | ~237K | Video temporal QA | MIT license |
| Vript (long + short) | ~411K | Video caption | Temporal narration |
| ShareGPT4Video | ~37K | Video caption | Short clips |
| LLaVA YouTube | ~22K | Video SFT | Pre-tokenized |
| SmolTalk S2 | ~140K (14%) | Text only | Instruction retention |

**Rationale for merging video into Stage 2:** Avoids a separate video SFT stage. The model sees images + video + text simultaneously, learning answer-only CE on all of them. Stage 3 then becomes preference alignment (DPO).

---

### Stage 3: DPO (IN PROGRESS — was originally "Video SFT")

| Item | Spec |
|------|------|
| **Purpose** | Preference alignment, reduce hallucination |
| **Loss** | DPO (beta=0.1) |
| **LR** | 1e-6 all components + cosine decay |
| **Freeze** | Full fine-tuning (reference model frozen) |
| **Samples** | 83K (1 epoch RLAIF-V) |
| **Init** | Stage 2 best checkpoint (step 27000) |
| **Reference** | Frozen copy of Stage 2 best |
| **Progress** | Resuming from step 1500/2593, rew_acc ~0.68 |

**Data:**

| Dataset | Samples | Type | Notes |
|---------|---------|------|-------|
| RLAIF-V | 83K | Image preference pairs (chosen + rejected) | Full fine-tuning DPO |

**Note:** Original plan had Video SFT here. Video data was moved to Stage 2 instead. DPO was originally "optional after Stage 3" but promoted to Stage 3 itself.

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

## Execution Strategy

### Step 1: 135M full pipeline ← DONE
- Stage 1: 1M samples, converging LR, 4.5h ✓
- Stage 2: 1M samples, flat 3e-5, 7.6h ✓
- Stage 3: DPO 83K samples, finishing ← IN PROGRESS
- Throughput: ~48 samp/s Stage 1, ~43 samp/s Stage 2, ~23 samp/s Stage 3 (DPO)
- torch.compile: disabled (135M too small, 40% regression)

### Step 2: Eval + HuggingFace upload ← NEXT
- Upload final model to HuggingFace
- Run 3-mode benchmarks (coarse, coarse→fine, autoregressive)
- MVBench + Video-MME

### Step 3: Scale to larger model
- 360M with LR=7e-4 (confirmed from sweep)
- 1.7B needs A100 80GB (OOM on RTX 5090)

## Go/No-Go Checklist

- [x] Foveated beats coarse-only — confirmed (small but consistent gap)
- [x] Val loss curves converge at ablation budget — yes
- [x] No divergence with full unfreeze — stable across all runs
- [x] 360M LR sweep complete — 7e-4 wins
- [x] 135M Stage 1 complete — val_loss=1.2358
- [x] 135M Stage 2 complete — train_loss=0.040
- [ ] 135M Stage 3 (DPO) complete — in progress
- [ ] Benchmark eval — pending
- [x] Training infrastructure stable — 13 ablation runs + 3 full stages

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
