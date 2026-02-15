# Foveated VLM — Claude Code Guide

## Project Overview

Foveated Vision-Language Model: a novel VLM that compresses each video frame to **1 visual token** via query-guided cross-attention on DINOv2 features. The LLM controls WHERE to look by generating the query for the next frame.

- **Architecture**: DINOv2 encoder + foveated cross-attention + SmolLM2 LLM
- **Two-pass training**: coarse (static query, parallel) → fine (dynamic queries, parallel)
- **Key claim**: 1 foveated token ≈ 16 standard tokens in quality

## Codebase Layout

```
release/                    ← CANONICAL codebase (ignore everything else)
  model/
    foveated_vlm.py         ← Main model (FoveatedVLM)
    encoder.py              ← DINOv2 + deep query cross-attention (FoveatedEncoder)
    multi_token_vlm.py      ← B1 baseline: 16 tok/frame, no foveation (MultiTokenVLM)
    __init__.py
  data/
    webdataset_loader.py    ← WebDataset loader (handles unified user/assistant format)
    text_interleave.py      ← Interleaves vision + text batches for retention
  utils/
    logging_utils.py        ← TrainingLogger (wandb + CSV + run summary JSON)
    checkpoint.py           ← Save/load/resume with best tracking
    distributed.py          ← DDP helpers
    lr_schedule.py          ← Cosine warmup scheduler
    flop_counter.py         ← FLOP estimation + iso-FLOP sample calculator
    attention_viz.py        ← Attention entropy + heatmap saver
  scripts/
    run_scaling_grid.py     ← Phase 1b scaling grid runner (24 runs from template)
    precompute.py           ← Tokenization helpers (tokenize_stage1, tokenize_sft)
  configs/
    ablations/              ← Phase 1a ablation configs (21 YAML files)
    scaling/                ← Phase 1b scaling configs

docs/
  GPU_PHASE1_PLAN.md        ← Full plan: ablations + scaling grid + decision framework
  runpod/
    GPU_HANDOFF.md          ← Config→RunID mapping, execution order, known gotchas
    REPO_SYNC_INSTRUCTIONS.md
  KNOWLEDGE.md              ← Bug history + experiment learnings
```

**Ignore**: `src/`, `scripts/`, `research/`, `core_docs/` — all legacy, superseded by `release/`.

## Current State (2026-02-15)

**Phase: Ready for GPU Phase 1a ablation runs.**

All code and data are ready. Next step is running 12 ablation experiments on 2xA100-80GB.

### What to do next

1. Read `docs/runpod/GPU_HANDOFF.md` for the complete run guide
2. Read `docs/GPU_PHASE1_PLAN.md` for the full plan with decision framework
3. Run ablations: `torchrun --nproc_per_node=2 release/train.py --config release/configs/ablations/<CONFIG>.yaml`

### Phase 1a: 12 Ablation Runs

| Run | Config | What |
|-----|--------|------|
| F1 | `F1_freeze_both.yaml` | Connector only (both backbones frozen) |
| F2 | `F2_freeze_llm.yaml` | Connector + DINO (LLM frozen) |
| F3/LR1 | `LR1.yaml` | Full unfreeze, 10:1 LR ratio (**BASELINE**) |
| LR2 | `LR2.yaml` | Full unfreeze, 100:1 |
| LR3 | `LR3.yaml` | Full unfreeze, 3:1 |
| LR4 | `LR4.yaml` | Full unfreeze, 1:1 uniform |
| A1 | `A1_deep_query_off.yaml` | Shallow query (proves deep query essential) |
| A6 | `A6_coarse_only.yaml` | No fine pass (proves foveation helps) |
| B1 | `B1_multi_token.yaml` | 16 tok/frame baseline (efficiency comparison) |
| A8a | `A8_static_1frame.yaml` | Image with 1 frame (control) |
| A8b | `A8_static_16frames.yaml` | Image replicated to 16 frames |
| D1 | `D1_video_heavy.yaml` | 55% video / 30% image / 15% text |

**Execution order**: F1+F2+LR1 first → analyze → LR2+LR3+LR4+A1 → A6+B1+A8a+A8b+D1

**Ignore these legacy configs**: `baseline.yaml`, `A2_*`, `A3_*`, `A4_*`, `A5_*`, `A8_static_8frames.yaml`, `T1_*`, `T2_*`, `T3_*`

## Data (all at /workspace/data/)

| Dataset | Path | Shards | Samples |
|---------|------|--------|---------|
| OpenVid-1M (video) | `openvid/*.tar` | 905 | 905K |
| Cauldron (image QA) | `cauldron_full/*.tar` | 2,001 | 2.0M |
| SmolTalk (text, 3 stages) | `text_retention/stage{1,2,3}/*.tar` | 490 | 490K |
| Eval | `eval/val_10k/*.tar` | 10 | 10K |
| Total across all sources | | 4,554 | 4.53M |

## Models (at /workspace/models/)

- `SmolLM2-135M-Instruct`, `SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct`
- `dinov2-small`, `dinov2-base`

## Key Architecture Decisions

- Loss: Stage 1 = all-text CE, Stage 2-3 = answer-only CE
- 14% SmolTalk text retention in ALL stages
- deep_query=True, query_dim=384, bias=False, std=1.0 init
- No reconstruction loss, no VAE, no DoRA

## Environment

- **Cache redirect**: Source `/workspace/.bashrc_runpod` to redirect HF/torch caches to /workspace (system disk is only 5GB)
- **Disk**: ~506GB used of 1TB quota (`df -h` shows 1.7PB but actual quota is 1TB)
- **No root**: Can't apt install

## Critical Bugs (already fixed in release/ code)

- BUG-001: Query init must be std=1.0 (not 0.02) — uniform attention otherwise
- BUG-002: query_input_proj must have bias=False — bias dominates small queries
- BUG-004: Must use deep_query=True — shallow gives uniform attention (correlation ~0.98)

These are baked into the code. Listed here so you don't accidentally reintroduce them.
