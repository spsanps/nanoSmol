# Scaling Experiment Data Export

Raw data and fitted parameters from the fVLM Phase 1b scaling grid experiments.

## Files

| File | Description |
|------|-------------|
| `all_eval_points.csv` | Combined evaluation metrics from all 5 runs (169 eval points). Each row is one validation evaluation with the closest preceding training loss, LR, and throughput. |
| `run_summaries.csv` | One row per run (5 runs). Best val loss, total samples, wall time, etc. |
| `scaling_fits.json` | Chinchilla-style power-law fit parameters (per-size and joint). |
| `scaling_law_plot.png` | Loss-vs-tokens plot with fitted scaling curves. |
| `_build_export.py` | Script used to generate the CSVs from raw checkpoint metrics. |

## Experiment Setup

### Phase 1a: Ablations (prior work)

Seven ablation runs (A1–A7, B1–B2) established the optimal architecture configuration:

- `deep_query=True` (shallow query was a no-op due to BUG-004)
- `query_dim=384`, `bias=False`, `std=1.0` init
- `lambda_coarse=0.0` (coarse loss disabled — fine-only is sufficient)
- Full unfrozen fine-tuning, no LoRA/DoRA

### Phase 1b: Scaling Grid

Five scaling runs at two model sizes with increasing compute budgets:

| Run ID | Model Size | Params | LR Schedule | Compute Budget | Total Samples | Best Val Loss | Status |
|--------|-----------|--------|-------------|----------------|---------------|---------------|--------|
| 135M-C1-F | 135M | 157.6M | cosine | 1.6e16 FLOPs | 67K | 1.4178 | Complete |
| 135M-C2-F | 135M | 157.6M | cosine | 5.6e16 FLOPs | 228K | 1.3251 | Complete |
| 135M-C3-F | 135M | 157.6M | cosine | 1.6e17 FLOPs | 648K | 1.3132 | Complete |
| 135M-C4-F | 135M | 157.6M | cosine | 5.6e17 FLOPs | 386K* | 1.2286 | Interrupted |
| 135M-scaling | 135M | 157.6M | constant | — | 656K | 1.1923 | Complete |
| 360M-scaling | 360M | 382.6M | constant | — | 702K | 1.3501 | Complete (LR too high) |

### LR Sweeps (100K samples, cosine schedule)

| Run ID | Model Size | Connector LR | Best Val Loss | Status |
|--------|-----------|-------------|---------------|--------|
| 360M-lr-3e4 | 360M | 3e-4 | 1.631 | Complete |
| 360M-lr-5e4 | 360M | 5e-4 | 1.530 | Complete |
| 360M-lr-7e4 | 360M | 7e-4 | 1.497 | Complete — **winner** |
| 1.7B-lr-* | 1.7B | {1-5}e-4 | — | Aborted (OOM on 32GB) |

### Long Video Ablation (50K samples, long videos only)

| Run ID | Mode | Best Val Loss | Trend | Status |
|--------|------|---------------|-------|--------|
| longvid_fine | Fine-only | 2.1619 @ step 200 | Overfitting (2.16→2.23) | Complete |
| longvid_coarse | Coarse-only | **2.1465** @ step 600 | Stable (~2.15) | Complete |

### Frame Replication (A8 sweep, 135M, mixed data)

| Run ID | Frames | Best Val Loss | Status |
|--------|--------|---------------|--------|
| A8c | 4 | 2.868 | Complete |
| A8d | 8 | **2.858** | Complete — **winner** |

*C4 was interrupted at step 12060/~39100 (~31% complete). Its 24 eval points are included.
**135M-scaling is the corrected rerun with constant LR (matches 360M design).
***360M-scaling used 135M-tuned LRs (1e-3 connector) — too aggressive. LR sweep confirmed 7e-4 as optimal.

## Model Configurations

- **135M**: SmolLM2-135M-Instruct (LLM) + DINOv2-small (vision encoder)
- **360M**: SmolLM2-360M-Instruct (LLM) + DINOv2-small (vision encoder)
- **Architecture**: Foveated VLM — 1 token per video frame via learned query cross-attention on DINOv2 patch features. Two-pass: coarse (static queries) then fine (dynamic queries).

## Training Details

- **Stage**: Stage 1 (video captioning, all-text loss)
- **Data mix**: OpenVid-1M + Vript (long captions) + ShareGPT4Video, with 14% SmolTalk text retention
- **Optimizer**: AdamW (lr=1e-3 for 135M, lr=1.52e-3 for 360M)
- **Batch size**: 8 (with gradient accumulation to effective batch = 32)
- **Hardware**: 1x RTX 5090 32GB
- **Precision**: TF32 + cuDNN benchmark + channels_last

## Key Findings

1. **135M constant-LR (1.1923) is the best result so far** — 82 clean eval points, every one valid for scaling law fits.
2. **360M used wrong LR** — same connector LR (1e-3) as 135M was too aggressive. LR sweep confirmed 7e-4 as optimal for 360M (1.497 vs 1.631 at 3e-4).
3. **360M LR sweep complete** — 7e-4 wins. Need full rerun with correct LR for proper scaling law fit.
4. **1.7B doesn't fit on RTX 5090** — even bs=1 with gradient checkpointing OOMs. Deferred to A100 80GB.
5. **Longvid fine vs coarse is INCONCLUSIVE** — val_loss showed coarse winning (2.147 vs 2.162) but train losses are equivalent (1.994 vs 1.981). The val_loss difference was an artifact of distribution mismatch (trained on 30-64 frame videos, evaluated on mixed data with mean 5.7 frames). See "Val Set Mismatch" section below.
6. **N=8 frame replication wins** for images in video training — confirmed on both train loss (1.330 vs 1.371) and val loss (2.858 vs 2.868).
7. **C1-C3 (cosine)** have LR schedule artifacts making them non-comparable with constant-LR runs.

## Val Set Mismatch Analysis

**val_10k composition**: ~49% video (mean 5.7 frames), ~51% image/text (Cauldron, SmolTalk). Sources include OpenVid, LLaVA-Video, WebVid, Cauldron, SmolTalk.

Most experiments train on video data (OpenVid + Vript + ShareGPT4Video) but val_10k includes Cauldron images not in training. This creates a consistent bias: val_loss is inflated for all video-training runs. However, since the bias is consistent, **relative rankings remain valid** — confirmed by checking that train-loss rankings match val-loss rankings across all experiment groups.

| Experiment group | Train-val gap | Rankings agree? | Notes |
|-----------------|---------------|-----------------|-------|
| Phase 1a ablations | ~0.7 | YES | All conclusions hold |
| A8 frame replication | ~1.5 | YES (N=8 wins both) | Large gap: trains on images only |
| 360M LR sweep | 0.8–1.0 | YES (7e-4 wins both) | Large gap: short runs (100K) |
| 135M scaling | 0.07 | — | Healthy gap |
| 360M scaling | 0.51 | — | Overfit (wrong LR) |
| **Longvid** | **0.15–0.18** | **NO** | **Train: fine≈coarse; val: coarse "wins"** |

**Lesson**: Always cross-check val_loss conclusions against train_loss, especially when training data distribution differs from val_10k. For future experiments with specialized data, consider building a matched validation set.

## Column Reference

### all_eval_points.csv

| Column | Description |
|--------|-------------|
| model_size | `135M` or `360M` |
| run_id | Unique run identifier |
| lr_schedule | `cosine` or `constant` |
| step | Training step at evaluation |
| samples_seen | Cumulative training samples |
| val_loss | Validation loss (text cross-entropy) |
| train_loss | Training loss from closest preceding train step |
| lr | Learning rate (connector LR; DINO/LLM are 0.01x this) |
| throughput_sps | Training throughput (samples/sec) |
| attention_entropy | Mean attention entropy in foveated cross-attention |

### run_summaries.csv

| Column | Description |
|--------|-------------|
| run_id | Unique run identifier |
| model_size | `135M` or `360M` |
| total_params | Total trainable parameters |
| lr_schedule | `cosine` or `constant` |
| total_samples | Total training samples processed |
| total_steps | Total optimizer steps |
| best_val_loss | Lowest validation loss achieved |
| best_val_step | Step at which best val loss occurred |
| final_train_loss | Training loss at the last logged step |
| wall_time_hours | Total wall-clock training time |

## Reproducing

To regenerate the scaling law plots and fits from these CSVs:

```bash
python release/scripts/scaling_law_analysis.py
```

To rebuild these CSVs from raw checkpoint metrics (requires access to `/workspace/checkpoints/scaling/`):

```bash
python release/docs/scaling_data/_build_export.py
```
