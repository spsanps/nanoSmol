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

*C4 was interrupted at step 12060/~39100 (~31% complete). Its 24 eval points are included.
**135M-scaling is the corrected rerun with constant LR (matches 360M design).
***360M-scaling used 135M-tuned LRs (1e-3 connector) — too aggressive. LR sweep in progress.

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
2. **360M used wrong LR** — same connector LR (1e-3) as 135M. 360M best (1.3501) barely beats 135M ablation baselines (~1.36). Curves never cross: 135M beats 360M at every sample count. This is a hyperparameter problem, not a model capacity problem.
3. **LR sweep in progress** — testing 360M with {3e-4, 5e-4, 7e-4} and 1.7B with {1e-4, 2e-4, 3e-4, 5e-4}. Standard heuristic: LR ~ 1/sqrt(N/N_base). Each sweep is 100K samples with cosine schedule + dense evals.
4. **After LR sweep**: re-run 360M and 1.7B full scaling runs with winning LRs → proper 3-point Chinchilla fit.
5. **C1-C3 (cosine)** have LR schedule artifacts making them non-comparable with constant-LR runs. C4 (interrupted) was on a strong trajectory but only 31% complete.

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
