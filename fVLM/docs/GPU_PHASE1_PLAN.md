# GPU Phase 1: Ablation Sweeps

**Date:** 2026-02-14 (v3)
**Hardware:** 2xA100-80GB
**Model:** SmolLM2-135M + DINOv2-small (~160M params)
**Budget per run:** 1M samples, ~4-5h each
**Val set:** Fixed val_10k (10K samples, mixed sources)
**wandb project:** `foveated-vlm-ablations`

---

## Sample Budget Rationale

Past experiments had serious methodology problems (see KNOWLEDGE.md):
- Ran on 18-50K samples with streaming data
- Different batch sizes, step counts, data sources between runs
- "Best" results came from the shortest run (2.3K steps)
- Cannot attribute differences to architecture vs. training noise

**This time: 1M samples per run.** Why:
- We have 4.3M samples available — no data shortage
- 1M is ~25% of Stage 1's full budget (good signal without full training)
- At 135M on 2xA100, throughput is ~500-1000 samples/sec → ~4-5h per run with eval
- Curves will clearly converge or diverge at 1M, giving reliable rankings
- All runs see exactly the same 1M samples (deterministic, seeded)

---

## Overview

Five ablation sweeps, 12 unique runs, ~3 days wall-clock, ~$65-85 total.

Every run serves double duty: (1) informs our config decisions, (2) produces data for the paper/blog.

```
Sweep 1 (Freeze)       →  How much to unfreeze in Stage 1?
Sweep 2 (LR)           →  Differential vs uniform LR, ratio sweep
Sweep 3 (Architecture) →  Confirm deep query + two-pass are essential
Sweep 4 (Image + Mix)  →  Frame replication + data composition
Sweep 5 (Baseline)     →  Multi-token baseline for efficiency comparison
```

---

## Sweep 1: Freeze Strategy (3 runs)

**Question:** How much should we freeze in Stage 1?

Idefics3 diverged with full unfreeze at 8B and fell back to DoRA. Our models are 50x smaller — but needs verification.

| Run | Frozen | Trainable | LR |
|-----|--------|-----------|-----|
| **F1** | DINO + LLM | Connector only | connector=1e-4 |
| **F2** | LLM | Connector + DINO | connector=1e-4, dino=1e-5 |
| **F3** | Nothing | All (differential LR) | connector=1e-4, dino=1e-5, llm=1e-5 |

**Metrics:** val_loss convergence curve, grad norm stability, loss_fine/loss_coarse ratio.

**Decision tree:**
- F3 diverges → fall back to F1 or F2
- F1 ≈ F3 → freeze backbones (simpler, cheaper)
- F3 clearly better → full unfreeze justified
- F2 > F3 → LLM unfreezing is harmful

**Note:** F3 = baseline config = LR1 (10:1 ratio). No duplicate run.

**Paper use:** Freeze comparison figure (convergence curves, 3 lines on one plot).

---

## Sweep 2: Learning Rate (3 runs)

**Question:** Does differential LR matter? What ratio?

| Run | Connector | DINO | LLM | Ratio | Notes |
|-----|-----------|------|-----|-------|-------|
| **LR1** | 1e-4 | 1e-5 | 1e-5 | 10:1 | = F3 (no extra run) |
| **LR2** | 1e-3 | 1e-5 | 1e-5 | 100:1 | Aggressive connector |
| **LR3** | 3e-5 | 1e-5 | 1e-5 | 3:1 | Near-uniform |
| **LR4** | 1e-5 | 1e-5 | 1e-5 | 1:1 (uniform) | Is differential LR even needed? |

**Key question LR4 answers:** If uniform LR matches differential, we can simplify the whole pipeline. If it's clearly worse, differential LR is justified.

**What to watch:**
- LR2 diverges → 1e-3 too aggressive for connector
- LR4 ≈ LR1 → differential LR doesn't matter, simplify
- LR4 clearly worse → differential LR is load-bearing

**Conditional:** If F1 (freeze backbones) wins Sweep 1, skip LR2/LR3/LR4 (only connector LR matters when backbones are frozen).

**Paper use:** LR sensitivity table, "differential LR is necessary/unnecessary" finding.

---

## Sweep 3: Architecture Validation (2 runs)

**Question:** Are deep query and two-pass training both essential?

| Run | Change from Baseline | Hypothesis |
|-----|---------------------|------------|
| **A1** | `deep_query=False` (shallow) | Shallow gives uniform attention → no foveation benefit |
| **A6** | No fine pass (coarse only, train with q_static only) | Coarse-only = no foveation. How much does the fine pass add? |

**A1 expected:** loss_ratio ≈ 1.0 (fine = coarse, no differentiation). Confirms deep query is essential for the mechanism to work.

**A6 rationale:** Even though we're committed to coarse+fine for the final model, we NEED the coarse-only comparison data for the paper. The core claim is "foveated two-pass beats static single-pass." Without A6, we can't prove that claim. This is the paper's control condition.

**Paper use:**
- **Figure: "Foveated attention matters"** — F3 (two-pass) vs A6 (coarse-only) vs A1 (shallow). Three lines showing the architecture design is load-bearing.
- loss_fine/loss_coarse ratio plot over training steps (from F3) — shows foveation benefit emerging during training.

---

## Sweep 4: Image Handling + Data Mix (3 runs)

**Question:** How should images enter training? What's the right video:image ratio?

| Run | Focus | Config | Eval |
|-----|-------|--------|------|
| **A8a** | 1-frame images | `replicate_image_frames: 1` | Image VQA |
| **A8b** | 16-frame replication | `replicate_image_frames: 16` | Image VQA |
| **D1** | Video-heavy mix | 55% video / 30% image / 15% text | Val loss (video + image subsets) |

**A8 runs:** Cauldron-heavy data, Stage 2 answer-only loss, evaluated on image VQA.
**D1 run:** Uses winning config, compared against F3 baseline (which uses default mix).

**Paper use:** Image handling table, data mix sensitivity.

---

## Sweep 5: Multi-Token Baseline (1 run)

**Question:** How does 1 foveated token compare to a standard multi-token approach?

| Run | Architecture | Tokens/Frame | Notes |
|-----|-------------|-------------|-------|
| **B1** | Standard (no foveation) | 16 | Feed 16 DINO patch tokens per frame directly to LLM. No query mechanism, no two-pass. Same DINO, same LLM, same data. |

**Implementation:** Skip the foveated encoder entirely. Use average-pooled or linearly-projected DINO patch tokens (4×4 grid → 16 tokens per frame). Project each to LLM dim. This is what a standard VLM does.

**Paper use:** This is the key efficiency claim figure:
- **"1 foveated token ≈ 16 standard tokens"** — compare F3 vs B1 at same FLOP budget
- **Token efficiency frontier plot:** quality vs visual tokens/frame
- If B1 with 16x more tokens only marginally beats F3 → efficiency win proven
- If B1 clearly beats F3 → foveation trades quality for efficiency (still publishable, different framing)

---

## Summary Table

| # | ID | Sweep | Description | Paper Figure |
|---|-----|-------|------------|-------------|
| 1 | F1 | Freeze | Connector only (DINO+LLM frozen) | Freeze comparison |
| 2 | F2 | Freeze | Connector + DINO (LLM frozen) | Freeze comparison |
| 3 | F3/LR1 | Freeze + LR | Full unfreeze, 10:1 ratio (baseline) | Baseline for all |
| 4 | LR2 | LR | Full unfreeze, 100:1 ratio | LR sensitivity |
| 5 | LR3 | LR | Full unfreeze, 3:1 ratio | LR sensitivity |
| 6 | LR4 | LR | Full unfreeze, 1:1 uniform | LR sensitivity |
| 7 | A1 | Architecture | Shallow query (`deep_query=False`) | "Deep query essential" |
| 8 | A6 | Architecture | Coarse-only (no fine pass) | **"Foveation helps" (core claim)** |
| 9 | B1 | Baseline | Multi-token (16 tok/frame, no foveation) | **"1 tok ≈ 16 tok" efficiency** |
| 10 | A8a | Image | 1-frame images | Image handling |
| 11 | A8b | Image | 16-frame replication | Image handling |
| 12 | D1 | Data mix | 55/30/15 video-heavy | Data mix |

**Total: 12 unique runs × ~4-5h = ~48-60h GPU time, ~$65-85**

---

## Paper Figures Produced by These Runs

| Figure | Data Source | What It Shows |
|--------|------------|---------------|
| **Fig 1: Foveated vs alternatives** | F3 + A6 + A1 + B1 | Core result: two-pass foveated beats coarse-only, shallow, and matches 16-token |
| **Fig 2: loss_fine < loss_coarse** | F3 (logged every step) | Foveation benefit emerges during training (ratio plot over steps) |
| **Fig 3: Token efficiency frontier** | F3 (1 tok) vs B1 (16 tok) | Quality vs visual tokens at same compute |
| **Fig 4: Attention heatmaps** | F3 eval (save attention weights) | Model learns to track objects, static→dynamic query difference |
| **Fig 5: Freeze strategy** | F1 + F2 + F3 | Which freeze pattern works for small VLMs |
| **Fig 6: Attention entropy** | All runs (log entropy per step) | Attention sharpens over training |
| **Fig 7: 3-mode eval** | Phase 7 (later) | coarse_only vs coarse+fine vs autoregressive gap |
| **Fig 8: Scaling curves** | Phase 3 (later) | Advantage grows with model size |

### Logging Requirements for Paper Figures

Every ablation run must log (per eval step):
1. `val_loss` — primary metric
2. `loss_fine`, `loss_coarse`, `loss_ratio` — foveation benefit tracking
3. `grad_norm` — stability monitoring
4. `attention_entropy` — selectivity metric (Fig 6)
5. `learning_rate` per param group — for LR sweep analysis

At eval time (every N steps):
6. **Attention weight maps** — save raw attention weights from encoder for a fixed set of ~10 eval videos. These become heatmap visualizations (Fig 4).

---

## Execution Schedule

```
Day 1 (~15h):
  Block 1:  F1 + F2 + F3/LR1       (Sweep 1: freeze strategy)
  → Analyze: pick freeze winner

Day 2 (~15h):
  Block 2:  LR2 + LR3 + LR4        (Sweep 2: LR ratio, skip if F1 won)
  Block 3:  A1                      (Sweep 3a: shallow query)

Day 3 (~15h):
  Block 4:  A6 + B1                 (Sweep 3b + 5: coarse-only + multi-token baseline)
  Block 5:  A8a + A8b + D1          (Sweep 4: image handling + data mix)
  → Final config locked
```

---

## What We're NOT Sweeping (and why)

| Dropped | Reason |
|---------|--------|
| Multi-fine iterations | From reconstruction era, not relevant to text-CE pipeline |
| Attention temperature | Not a meaningful training knob during fine-tuning |
| Text retention ratio (14%) | SmolVLM2 established this empirically at our model family/scale |
| Query dimension (384) | Matches DINO dim naturally, second-order effect |
| Visual scale (0.14) | Can tune cheaply post-training |

---

## After Ablations: Full Pipeline

```
Phase 1: Ablations          →  Pick best config (this document)
Phase 2: Speed optimization →  torch.compile, profile, 2x throughput target
Phase 3: Scaling grid       →  135M / 360M / 1.7B × 4 FLOP budgets
                                → Determines final model size

Phase 4: Stage 1 (alignment)
  Data:  OpenVid + Vript + ShareGPT4V + Cauldron + 14% SmolTalk
  Loss:  All-token CE
  Freeze: Winner from Sweep 1

Phase 5: Stage 2 (instruction)
  Data:  LLaVA-Video + VISTA + Cauldron + 14% SmolTalk
  Loss:  Answer-only CE
  Freeze: Full unfreeze (differential LR)

Phase 6: Stage 3 (refinement)
  Data:  VISTA + Stage3 YT + Cauldron + RLAIF-V + 14% SmolTalk
  Loss:  Answer-only CE + optional DPO
  Note:  LLaVA-Video excluded (67% hit 64-frame cap → noisy signal)

Phase 7: Eval
  Modes: coarse_only, coarse_fine, autoregressive
  Benchmarks: Video-MME, MVBench, MLVU
  Baselines: SmolVLM2-256M, SmolVLM2-2.2B
```

---

## Data Inventory (as of 2026-02-14)

| Dataset | Samples | Type | Stage | Status |
|---------|---------|------|-------|--------|
| Cauldron | 2.0M | Image QA | 1, 2, 3 | DONE |
| OpenVid-1M | 905K | Video caption | 1 | DONE |
| LLaVA-Video | 266K | Video QA | 2 | DONE |
| Vript (short + long) | 207K | Video caption | 1, 3 | Long still running |
| VISTA-400K | 152K+ | Video temporal QA | 2, 3 | Running (~12h ETA) |
| ShareGPT4Video | 61K | Video caption | 1 | DONE |
| SmolTalk | 490K | Text-only | All (14%) | DONE |
| RLAIF-V | 84K | Image DPO | 3 (optional) | DONE |
| Others (WebVid, YT, stage3) | 91K | Mixed | Various | DONE |
| **Total** | **~4.3M** | | | |

**Video: ~1.69M / 4.31M = 39.2%** (target 40-50%, on track)

---

## Code Changes Needed Before Running

1. **Freeze config** — add `freeze_dino: true/false` and `freeze_llm: true/false` to train.py
2. **Dataloader update** — unified `user`/`assistant` JSON format (RunPod Claude already converted all shards)
3. **Loss routing** — `text_ce_all` (Stage 1) vs `text_ce_answer_only` (Stage 2-3) based on config
4. **A8 wiring** — `replicate_image_frames` param exists in train.py, verify dataloader implements it
5. **B1 baseline model** — implement multi-token forward (16 DINO patches → project → LLM, no query mechanism)
6. **Attention logging** — save attention weights + entropy at eval time for visualization figures

---

*Last updated: 2026-02-14 (v3)*
