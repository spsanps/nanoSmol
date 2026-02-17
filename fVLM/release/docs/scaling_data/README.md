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
| 360M-scaling | 360M | 382.6M | constant | — | 702K | 1.3501 | Complete |

*C4 was interrupted at step 12060/~39100 (~31% complete). Its 24 eval points are included.

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

1. **Val loss decreases with more data** across all runs, as expected.
2. **LR schedule mismatch is a critical caveat**: The 135M runs used cosine decay while 360M used constant LR. Cosine decay artificially reduces loss late in training (lower LR → lower instantaneous loss), making 135M appear to scale better than 360M. The joint Chinchilla fit is unreliable for this reason.
3. **Per-size fits are more trustworthy** than the joint fit. The 360M constant-LR fit (RMSE=0.023) is cleaner than the 135M cosine fit (RMSE=0.058).
4. **C4 (interrupted)** was on a strong trajectory — its best val loss of 1.229 at 386K samples is already better than C3's final 1.313 at 648K, likely due to seeing the steepest part of the cosine-boosted learning curve.

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
