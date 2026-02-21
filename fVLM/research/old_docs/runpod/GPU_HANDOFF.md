# GPU Pod Handoff Guide

**Date:** 2026-02-15
**From:** RunPod Claude (CPU pod)
**To:** LOCAL Claude (GPU pod)

Everything is pushed. After `git pull`, you can start Phase 1a immediately.

---

## Quick Start

```bash
git pull origin main
# Verify
python release/train.py --config release/configs/ablations/LR1.yaml --dry-run
```

---

## Phase 1a: Config → Run ID Mapping

**12 runs to execute. The plan file (`docs/GPU_PHASE1_PLAN.md`) defines them.**

| Plan ID | Config File | Sweep | What It Tests |
|---------|------------|-------|---------------|
| **F1** | `F1_freeze_both.yaml` | Freeze | Connector only (DINO + LLM frozen) |
| **F2** | `F2_freeze_llm.yaml` | Freeze | Connector + DINO (LLM frozen) |
| **F3/LR1** | `LR1.yaml` | Freeze + LR | Full unfreeze, 10:1 ratio (BASELINE) |
| **LR2** | `LR2.yaml` | LR | Full unfreeze, 100:1 (aggressive connector) |
| **LR3** | `LR3.yaml` | LR | Full unfreeze, 3:1 (near-uniform) |
| **LR4** | `LR4.yaml` | LR | Full unfreeze, 1:1 (uniform) |
| **A1** | `A1_deep_query_off.yaml` | Architecture | Shallow query (no deep propagation) |
| **A6** | `A6_coarse_only.yaml` | Architecture | Coarse-only (no fine pass) |
| **B1** | `B1_multi_token.yaml` | Baseline | Multi-token (16 tok/frame, no foveation) |
| **A8a** | `A8_static_1frame.yaml` | Image | 1-frame images (control) |
| **A8b** | `A8_static_16frames.yaml` | Image | 16-frame replication |
| **D1** | `D1_video_heavy.yaml` | Data mix | 55% video / 30% image / 15% text |

**Ignore these legacy configs** (not in Phase 1a plan):
`baseline.yaml`, `A2_*`, `A3_*`, `A4_*`, `A5_*`, `A8_static_8frames.yaml`, `T1_*`, `T2_*`, `T3_*`

---

## Execution Order (from plan)

```
Day 1:  F1 + F2 + LR1       → Sweep 1 (freeze), analyze
Day 2:  LR2 + LR3 + LR4     → Sweep 2 (LR), then A1
Day 3:  A6 + B1, then A8a + A8b + D1 → Sweeps 3-5
```

Each run: `torchrun --nproc_per_node=2 release/train.py --config release/configs/ablations/<CONFIG>.yaml`

---

## How New Code Works

### Multi-Token Model (B1)
- Config has `multi_token: true` → `train.py:build_model()` creates `MultiTokenVLM` instead of `FoveatedVLM`
- Same `forward()` API: accepts `mode=` param but ignores it
- Returns `fine_loss`, `coarse_loss` for logging compatibility (coarse_loss = 0)
- File: `release/model/multi_token_vlm.py`

### Freeze Config (F1, F2)
- Config has `freeze_dino: true` and/or `freeze_llm: true`
- Applied in `build_model()` after model creation, before DDP
- Works for both FoveatedVLM (has `model.encoder.dino`) and MultiTokenVLM (has `model.dino`)

### Coarse-Only Mode (A6)
- Config has `model.coarse_only: true`
- `train.py` sets `train_mode = "coarse_only"` → passed to `model.forward(mode=...)`
- `forward_coarse_only` no longer has `@torch.no_grad()` so it works for training
- Returns `coarse_loss = loss`, `fine_loss = 0`

### Deep Query Off (A1)
- Config has `model.deep_query: false`
- Handled inside `FoveatedEncoder` (falls back to shallow single-layer cross-attention)

### Logging
- **CSV**: auto-saved to checkpoint dir, one row per log event
- **wandb**: project `foveated-vlm-ablations`, run names match config IDs
- **Run summary**: JSON saved at end of training with best metrics, wall time
- **Attention maps**: saved to `{checkpoint_dir}/attention_maps/` during eval
- **Attention entropy**: computed during eval, logged to CSV + wandb

### Scaling Grid (Phase 1b)
```bash
# After Phase 1a, using winning config as template:
python release/scripts/run_scaling_grid.py --template release/configs/ablations/LR1.yaml --dry-run
python release/scripts/run_scaling_grid.py --template release/configs/ablations/LR1.yaml
# Or filter: --filter "135M" to run only small model configs
```

---

## Data Paths (all at /workspace/data/)

All configs use these paths (already set):
- **Stage 1 video**: `/workspace/data/openvid/*.tar` (905 shards, 905K samples)
- **Stage 2 image**: `/workspace/data/cauldron_full/*.tar` (2001 shards, 2M samples)
- **Eval**: `/workspace/data/eval/val_10k/*.tar` (10 shards, 10K samples)
- **Text retention**: `/workspace/data/text_retention/stage{1,2,3}/*.tar`

D1 (video-heavy) also uses: `/workspace/data/vript_long_shards/*.tar`

---

## Environment Notes

- **Cache redirect**: Source `/workspace/.bashrc_runpod` to redirect HF/torch caches to /workspace
- **Models**: `/workspace/models/SmolLM2-135M-Instruct`, `dinov2-small`, etc.
- **Disk**: 506GB used of 1TB — plenty of room for checkpoints
- **No root**: Can't apt install. All Python deps should already be installed.

---

## What Each Config Produces

Every run auto-generates:
1. `{checkpoint_dir}/metrics_{run_id}_{timestamp}.csv` — full training metrics
2. `{checkpoint_dir}/run_summary_{run_id}_{timestamp}.json` — final summary
3. `{checkpoint_dir}/attention_maps/` — raw attention weight .pt files (foveated models only)
4. `{checkpoint_dir}/step_*.pt` — model checkpoints (keeps last 2 + best)
5. wandb logs at `foveated-vlm-ablations` project

---

## Known Gotchas

1. **B1 sequences are long**: 16 tokens/frame × 64 frames = 1024 visual tokens. May need `gradient_checkpointing: true` or smaller batch if OOM.
2. **A8 configs use Stage 2**: They train on Cauldron (image QA) with answer-only loss, not Stage 1 video captions.
3. **LR1 = F3 = baseline**: Same config, no need to run twice. F3 in the freeze comparison IS LR1 in the LR comparison.
4. **Conditional skip**: If F1 (freeze both) wins Sweep 1, skip LR2/LR3/LR4 (only connector LR matters when backbones frozen).
5. **torchvision**: On CPU pod, `import torchvision` failed with `torch.library.register_fake` error. Should be fine on GPU pod with proper CUDA torch install.
