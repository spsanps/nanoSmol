# GPU Phase 1: Ablations + Scaling Grid

**Date:** 2026-02-14 (v4)
**Hardware:** 2xA100-80GB
**Val set:** Fixed val_10k (10K samples, mixed sources)
**wandb projects:** `foveated-vlm-ablations`, `foveated-vlm-scaling`

**Two phases before full training:**
```
Phase 1a: Ablation Sweeps   →  12 runs @ 135M, pick best config
Phase 1b: Scaling Grid      →  24 runs @ 3 sizes, pick best model size
Then: Full 3-stage training with winning config + size
```

---

---

# Phase 1a: Ablation Sweeps

**Model:** SmolLM2-135M + DINOv2-small (~160M params)
**Budget per run:** 1M samples, ~4-5h each

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

---

# Phase 1b: Scaling Grid

**Goal:** Determine optimal model size for final training. Not pre-committed to any size.
**Depends on:** Winning config from Phase 1a ablations (freeze strategy, LR ratio, architecture).
**wandb project:** `foveated-vlm-scaling`

## Design

Modeled on Chinchilla / nanochat's `scaling_laws.sh`. For each (size, FLOP budget) pair, compute how many samples that budget allows, train, and evaluate.

| Axis | Values | Count |
|------|--------|-------|
| **LLM size** | SmolLM2-135M (~157M total), SmolLM2-360M (~383M), SmolLM2-1.7B (~1.72B) | 3 |
| **FLOP budget** | 4 levels spanning ~20x range | 4 |
| **Architecture** | Foveated (1 tok/frame), Multi-token baseline (16 tok/frame) | 2 |

**Total: 3 × 4 × 2 = 24 runs**

## FLOP Budget Calibration

Approximate FLOPs per sample: `2 × total_params × tokens_per_sample`
Average tokens/sample ≈ 500 (visual + text, variable frame count).

| Budget Level | Total FLOPs | 135M samples | 360M samples | 1.7B samples |
|-------------|-------------|-------------|-------------|-------------|
| **C1 (small)** | ~1.6e16 | 100K | 40K | 9K |
| **C2 (medium)** | ~5.6e16 | 360K | 140K | 30K |
| **C3 (large)** | ~1.6e17 | 1M | 400K | 90K |
| **C4 (full)** | ~3.1e17 | 2M | 800K | 180K |

Levels are ~3-3.5x apart, spanning ~20x total compute range. This gives enough dynamic range for clear scaling trends.

**C2 = existing iso-FLOP configs** (360K/140K/30K). C3 matches the ablation budget (1M at 135M).

## Run Matrix (24 runs)

### Foveated (winning config from Phase 1a)

| Run | Size | Budget | Samples | Est. Time | Notes |
|-----|------|--------|---------|-----------|-------|
| S-C1-F | 135M | C1 | 100K | ~5min | |
| S-C2-F | 135M | C2 | 360K | ~15min | = existing config |
| S-C3-F | 135M | C3 | 1M | ~30min | = ablation baseline F3 |
| S-C4-F | 135M | C4 | 2M | ~1h | |
| M-C1-F | 360M | C1 | 40K | ~5min | |
| M-C2-F | 360M | C2 | 140K | ~15min | = existing config |
| M-C3-F | 360M | C3 | 400K | ~30min | |
| M-C4-F | 360M | C4 | 800K | ~1h | |
| L-C1-F | 1.7B | C1 | 9K | ~5min | gradient checkpointing |
| L-C2-F | 1.7B | C2 | 30K | ~15min | = existing config |
| L-C3-F | 1.7B | C3 | 90K | ~30min | gradient checkpointing |
| L-C4-F | 1.7B | C4 | 180K | ~1h | gradient checkpointing |

### Multi-token baseline (B1 architecture, 16 tok/frame)

Same matrix but with `multi_token: true, tokens_per_frame: 16`:

| Run | Size | Budget | Samples | Notes |
|-----|------|--------|---------|-------|
| S-C1-B | 135M | C1 | 100K | 16x more visual tokens → ~16x more DINO FLOPs |
| S-C2-B | 135M | C2 | 360K | |
| S-C3-B | 135M | C3 | 1M | |
| S-C4-B | 135M | C4 | 2M | |
| M-C1-B | 360M | C1 | 40K | |
| M-C2-B | 360M | C2 | 140K | |
| M-C3-B | 360M | C3 | 400K | |
| M-C4-B | 360M | C4 | 800K | |
| L-C1-B | 1.7B | C1 | 9K | |
| L-C2-B | 1.7B | C2 | 30K | |
| L-C3-B | 1.7B | C3 | 90K | |
| L-C4-B | 1.7B | C4 | 180K | |

**Note on iso-FLOP for baseline:** The multi-token baseline uses ~16x more visual FLOPs per sample (16 tokens vs 1). Two options:
1. **Same sample count** — baseline sees same data but costs more FLOPs (unfair to baseline in FLOP terms, fair in data terms)
2. **Same FLOP budget** — baseline sees fewer samples to match total FLOPs (fair in compute, unfair in data)

**Decision:** Run **both** for the most informative plots. For the 135M runs at C2-C3 (cheap), run iso-sample AND iso-FLOP. For expensive runs (1.7B), iso-sample only. This gives us both the "same compute" and "same data" comparison.

## Time & Cost Estimate

| Category | Runs | Est. GPU Hours | Est. Cost |
|----------|------|---------------|-----------|
| Foveated (12 runs) | 12 | ~5h | ~$7 |
| Baseline (12 runs) | 12 | ~8h (16x more visual tokens) | ~$11 |
| Iso-FLOP extras (4 runs) | 4 | ~2h | ~$3 |
| **Total Phase 1b** | **28** | **~15h** | **~$21** |

Most runs are fast (5-30min). Only the C4 runs at 1.7B take ~1h each.

## Paper Figures from Scaling Grid

| Figure | X-axis | Y-axis | Curves | What It Shows |
|--------|--------|--------|--------|---------------|
| **Iso-FLOP curves** | Model params | Val loss | One per FLOP budget (C1-C4) | Compute-optimal model size |
| **Foveated vs baseline crossover** | Total FLOPs | Val loss | Foveated vs baseline per size | **Foveated advantage grows with scale** (thesis) |
| **Token efficiency frontier** | Visual tokens/frame (1 vs 16) | Val loss at matched FLOPs | One per model size | 1 foveated ≈ N standard |
| **Data scaling curves** | Samples seen | Val loss (from intermediate checkpoints) | Foveated vs baseline | Sample efficiency |
| **Chinchilla-style allocation** | Total compute C | Optimal N (params) at iso-FLOP minimum | Foveated vs baseline | N ∝ C^a exponents differ? |

## Decision from Scaling Grid

The scaling grid answers: **at our budget, what model size gives the best quality?**

| Outcome | Action |
|---------|--------|
| 135M is compute-optimal at our budget | Ship 135M, fast inference, efficiency story |
| 360M is compute-optimal | Ship 360M, balanced story |
| 1.7B is compute-optimal | Ship 1.7B, quality-first story |
| Foveated advantage grows with scale | Strong paper thesis — "foveation is more valuable at scale" |
| Foveated advantage flat across scale | Efficiency story — "foveation is a free compression" |
| Foveated advantage shrinks with scale | Pivot — "foveation helps small models most" (still publishable) |

## DINOv2-base Decision

Existing configs have both DINOv2-small (384d, 22M) and DINOv2-base (768d, 86M). **Recommendation: skip DINOv2-base in the main scaling grid.** Reason:
- Doubles the grid from 24 to 48 runs
- DINO size is a secondary axis — the LLM dominates parameter count
- Run 2-3 DINOv2-base comparison runs at the **winning model size only** after the grid, as a quick follow-up

## Execution Schedule (Phase 1b)

```
Day 4 (~8h, after Phase 1a ablations complete):
  Morning:   Apply winning config from ablations to all scaling configs
  Block 6:   All 135M runs (8 runs: 4 foveated + 4 baseline) → ~2h
  Block 7:   All 360M runs (8 runs: 4 foveated + 4 baseline) → ~3h

Day 5 (~7h):
  Block 8:   All 1.7B runs (8 runs: 4 foveated + 4 baseline) → ~4h
  Block 9:   Iso-FLOP extras (4 runs) → ~2h
  → Analyze: pick optimal model size
  → Optional: 2-3 DINOv2-base comparison runs at winning size
```

## Config Changes for Scaling Grid

1. **Update existing 6 scaling configs** — apply winning LR, freeze strategy from ablations
2. **Create 18 new configs** — the other FLOP budget levels (C1, C3, C4 for each size) + all baseline variants
3. **Or: parametric runner script** — `run_scaling_grid.py` that generates configs from template + matrix, more maintainable than 24 YAML files

---

## After Phase 1a + 1b: Full Pipeline

```
Phase 1a: Ablations          →  Pick best config (this document, 12 runs)
Phase 1b: Scaling grid       →  Pick best model size (this document, 24+ runs)
Phase 2:  Speed optimization →  torch.compile, profile, 2x throughput target

Phase 3: Stage 1 (alignment)
  Data:  OpenVid + Vript + ShareGPT4V + Cauldron + 14% SmolTalk
  Loss:  All-token CE
  Freeze: Winner from Sweep 1

Phase 4: Stage 2 (instruction)
  Data:  LLaVA-Video + VISTA + Cauldron + 14% SmolTalk
  Loss:  Answer-only CE
  Freeze: Full unfreeze (differential LR)

Phase 5: Stage 3 (refinement)
  Data:  VISTA + Stage3 YT + Cauldron + RLAIF-V + 14% SmolTalk
  Loss:  Answer-only CE + optional DPO
  Note:  LLaVA-Video excluded (67% hit 64-frame cap → noisy signal)

Phase 6: Eval
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

### For Phase 1a (Ablations)
1. **Freeze config** — add `freeze_dino: true/false` and `freeze_llm: true/false` to train.py
2. **Dataloader update** — unified `user`/`assistant` JSON format (RunPod Claude already converted all shards)
3. **Loss routing** — `text_ce_all` (Stage 1) vs `text_ce_answer_only` (Stage 2-3) based on config
4. **A8 wiring** — `replicate_image_frames` param exists in train.py, verify dataloader implements it
5. **B1 baseline model** — implement multi-token forward (16 DINO patches → project → LLM, no query mechanism)

### For Both Phases (Logging & Metrics)
6. **Comprehensive metric logging** — every run must log to both wandb and structured CSV:
   - Per training step: `train_loss`, `loss_fine`, `loss_coarse`, `loss_ratio`, `grad_norm`, `learning_rate` (per param group), `throughput_samples_sec`, `gpu_memory_allocated`, `gpu_memory_reserved`
   - Per eval step: `val_loss`, `val_loss_fine`, `val_loss_coarse`, `val_loss_ratio`, `attention_entropy` (mean across heads/layers)
   - Per eval step (save to disk): raw attention weight maps for ~10 fixed eval videos (for heatmap figures)
7. **Structured CSV output** — one row per eval point, columns: `run_id, step, samples_seen, wall_time_sec, train_loss, val_loss, loss_fine, loss_coarse, loss_ratio, grad_norm, attention_entropy, lr_connector, lr_dino, lr_llm, throughput, gpu_mem_gb`. Auto-saved alongside wandb for offline analysis.
8. **Checkpoint metadata** — each checkpoint saves: config used, git commit hash, exact samples seen, best metrics so far, data shard position (for deterministic resume)
9. **Run summary JSON** — at end of each run, auto-generate `run_summary.json` with: final metrics, best metrics, convergence step, total wall time, total samples, config hash. For quick comparison across runs without opening wandb.

### For Phase 1b (Scaling Grid)
10. **FLOP counter** — `estimate_flops(config)` function that returns FLOPs per sample for any (LLM size, DINO size, tokens/frame) combo. Used to calibrate iso-FLOP sample counts.
11. **Scaling grid runner** — `run_scaling_grid.py` script that:
    - Takes the template + a matrix definition (sizes, budgets, architectures)
    - Generates configs on the fly (no need for 24 YAML files)
    - Launches runs sequentially, collects results into a single CSV
    - Produces scaling law plots automatically
12. **Intermediate checkpoint eval** — save and evaluate intermediate checkpoints at ~5 evenly-spaced points per run. Needed for Plot 4 (data scaling curves) where we plot val_loss vs samples_seen across different model sizes.

### For Paper/Blog Figures
13. **Attention heatmap saver** — at eval time, save raw attention weights from encoder cross-attention for a fixed set of ~10 eval videos. These become heatmap visualizations (Fig 4: object tracking, static vs dynamic query difference).
14. **Auto-plot generation** — script that reads structured CSVs and produces publication-quality matplotlib/seaborn figures:
    - Convergence curves (loss vs samples for ablation comparison)
    - Loss ratio plots (loss_fine/loss_coarse over training)
    - Attention entropy over training
    - Scaling law iso-FLOP curves
    - Token efficiency frontier
    - Freeze strategy comparison
15. **Eval mode comparison** — after Phase 1b, evaluate best checkpoint in all 3 modes (coarse_only, coarse_fine, autoregressive) and log the gap. This is Fig 7 data.

---

## Post Phase 1: Decision Framework

After ablations + scaling grid, we have all the data to make final decisions before committing to expensive full training (~$250+).

### Decision 1: Architecture Config (from Phase 1a)

| Question | Data Source | Decision |
|----------|------------|----------|
| Freeze strategy | F1 vs F2 vs F3 convergence curves | Pick lowest val_loss with stable grad norms |
| Learning rate | LR1-LR4 final val_loss + divergence check | Pick best ratio (or uniform if LR4 matches) |
| Deep query essential? | A1 loss_ratio (should be ~1.0 = bad) | If A1 ≈ F3, deep query isn't load-bearing (unlikely) |
| Two-pass essential? | A6 vs F3 val_loss gap | Quantifies the foveation benefit |
| Image handling | A8a vs A8b image VQA scores | Pick 1-frame or 16-frame replication |
| Data mix | D1 vs F3 val_loss on video/image subsets | Adjust video:image:text ratio |
| Efficiency claim | F3 (1 tok) vs B1 (16 tok) at matched FLOPs | The core paper result |

**Deliverable:** Locked YAML config for full training. Blog table: "ablation results at a glance."

### Decision 2: Model Size (from Phase 1b)

| Question | Data Source | Decision |
|----------|------------|----------|
| Compute-optimal size | Iso-FLOP curves (val_loss vs params at each C level) | Pick size at minimum of each curve |
| Does foveation scale? | Foveated vs baseline gap across sizes | Core thesis for the paper |
| How much data? | Data scaling curves (val_loss vs samples at each size) | Determines Stage 1-3 sample counts |
| Token efficiency | 1-tok vs 16-tok at matched FLOPs per size | Does efficiency advantage hold at scale? |

**Deliverable:** Chosen model size (135M/360M/1.7B) + Stage 1-3 sample budgets. Blog post: scaling law story.

### Decision 3: Training Schedule (derived)

Once we know config + size, compute:
- Stage 1 sample count (from data scaling curve — where does loss plateau?)
- Stage 2-3 sample counts (proportional to SmolVLM2 ratios)
- Total GPU hours and cost estimate for full pipeline
- Whether we stay on 2xA100 or upgrade to 4x

### What Gets Published

| Artifact | Source | When |
|----------|--------|------|
| **Ablation table** | Phase 1a CSVs | Blog post 1 |
| **Scaling law plots** (5 figures) | Phase 1b CSVs | Blog post 2 / paper |
| **Attention heatmaps** | Saved attention weights from best runs | Paper Fig 4 |
| **Architecture comparison figure** | F3 vs A6 vs A1 vs B1 | Paper Fig 1 (core result) |
| **Loss ratio emergence plot** | F3 per-step loss_fine/loss_coarse | Paper Fig 2 |
| **Token efficiency frontier** | F3 vs B1 at multiple sizes | Paper Fig 3 |
| **Freeze comparison** | F1 vs F2 vs F3 | Paper Fig 5 |
| **Entropy over training** | All runs attention_entropy | Paper Fig 6 |
| **Locked config + rationale** | All of the above | Internal doc / appendix |

### Go/No-Go Criteria

Before committing to full training ($250+), verify:
- [ ] Foveated (F3) clearly beats coarse-only (A6) — the mechanism works
- [ ] Val loss curves converge (not still falling) at 1M samples — ablation budget was sufficient
- [ ] No divergence in any winning config — training is stable
- [ ] Scaling grid shows clear compute-optimal point — we know which size to train
- [ ] Foveated has reasonable efficiency vs baseline — the story is publishable regardless of direction
- [ ] All logging works — wandb, CSV, attention maps, checkpoint metadata all verified

If any of these fail, re-evaluate before spending more compute.

---

## Total Budget Summary

| Phase | Runs | GPU Hours | Cost |
|-------|------|-----------|------|
| **Phase 1a: Ablations** | 12 | ~48-60h | ~$65-85 |
| **Phase 1b: Scaling Grid** | 24-28 | ~15h | ~$21 |
| **Speed optimization** | — | ~8h | ~$11 |
| **Buffer (20%)** | — | ~16h | ~$22 |
| **Total pre-training** | **36-40** | **~90-100h** | **~$120-140** |

Remaining budget for full 3-stage training + eval: ~$360-380 of $500.

---

*Last updated: 2026-02-14 (v4)*
