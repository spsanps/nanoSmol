# RunPod GPU Briefing: Foveated VLM Project

**READ THIS FIRST.** You are Claude Code on a RunPod GPU pod (2xA100-80GB). All data is prepared, all code is ready. Your job: run ablation experiments and scaling grid, then train the final model.

**Date:** 2026-02-15
**Budget:** ~$400 remaining of $500
**Network volume:** `/workspace` persists across pods (506GB used / 1TB)

---

## Quick Start (first 5 minutes)

```bash
# 1. Environment setup
source /workspace/.bashrc_runpod

# 2. Verify GPUs
nvidia-smi   # Should show 2x A100 80GB

# 3. Activate venv
source /workspace/venv311/bin/activate

# 4. Pull latest code
cd /workspace/workdir/nanoSmol/fVLM
git pull origin main

# 5. Read the execution plan
cat docs/GPU_PHASE1_PLAN.md
```

---

## What Is This Project?

**fVLM** = Foveated Vision-Language Model. 1 token per video frame via query-guided cross-attention on DINOv2 features. The LLM generates queries that control WHERE to look — like biological foveated vision.

```
Standard VLM:  frame → 196 patches → pixel_shuffle → ~81 tokens → LLM
Foveated VLM:  frame → DINO features → query_cross_attention → 1 token → LLM
```

**Thesis:** 1 foveated token matches 16 standard tokens at same compute.

### Two-Pass Training

1. **Coarse pass:** `q_static` (learnable, same for all frames) → DINO cross-attention → `z_coarse` → LLM → dynamic queries
2. **Fine pass:** dynamic queries → DINO cross-attention → `z_fine` → LLM → text prediction → CE loss

### Three Eval Modes
- `coarse_only` — fastest, static query only
- `coarse_fine` — matches training (two parallel passes)
- `autoregressive` — true sequential inference, query per frame

---

## Current State: Everything Ready

### Code (release/ is THE codebase)

```
release/
├── train.py                     # torchrun --nproc_per_node=2 release/train.py --config ...
├── evaluate.py                  # python release/evaluate.py --checkpoint ... --mode coarse_fine
├── model/
│   ├── foveated_vlm.py          # FoveatedVLM: coarse_fine, coarse_only, autoregressive modes
│   ├── multi_token_vlm.py       # MultiTokenVLM: B1 baseline (16 tok/frame, no foveation)
│   └── encoder.py               # FoveatedEncoder: deep query on DINOv2, attention extraction
├── data/
│   ├── webdataset_loader.py     # Tar shard loader, unified user/assistant format, frame replication
│   ├── collate.py               # Pads frames + text, creates masks
│   └── text_interleave.py       # 14% SmolTalk text retention mixing
├── utils/
│   ├── logging_utils.py         # wandb + CSV + run_summary JSON + git hash tracking
│   ├── attention_viz.py         # Attention entropy + heatmap saver for paper figures
│   ├── flop_counter.py          # FLOP estimation, iso-FLOP sample calculator
│   ├── checkpoint.py            # Save/load with best-metric tracking
│   ├── distributed.py           # DDP setup
│   └── lr_schedule.py           # Cosine with linear warmup
├── scripts/
│   ├── run_scaling_grid.py      # 24-run scaling grid runner (Phase 1b)
│   └── ...                      # Data pipeline scripts (already run, not needed on GPU)
└── configs/
    ├── ablations/               # 21 YAML files (F1, F2, F3/baseline, LR1-4, A1, A6, A8a/b, B1, D1, + extras)
    ├── scaling/                 # 6 scaling configs + template
    ├── stage1_webvid.yaml       # Full Stage 1 training
    ├── stage2_vl_sft.yaml       # Full Stage 2 training
    └── stage3_video_sft.yaml    # Full Stage 3 training
```

**All code changes from GPU_PHASE1_PLAN.md are implemented:**
- Freeze config (freeze_dino/freeze_llm in train.py)
- A6 coarse-only training mode
- Unified user/assistant format in dataloader
- Loss routing (all-token vs answer-only)
- B1 multi-token baseline model
- Comprehensive metric logging (wandb + CSV + attention entropy + run summary)
- Attention heatmap saver
- FLOP counter + scaling grid runner

### Data (4.53M samples, 450GB)

| Dataset | Path | Shards | Samples | Type | Stage |
|---------|------|--------|---------|------|-------|
| Cauldron | cauldron_full/ | 2,001 | 2.0M | Image QA | 1, 2, 3 |
| OpenVid-1M | openvid/ | 905 | 905K | Video caption | 1 |
| Vript long | vript_long_shards/ | 400 | 398K | Video caption | 1, 3 |
| LLaVA-Video | llava_video_shards/ | 266 | 266K | Video QA | 2 |
| SmolTalk S1 | text_retention/stage1/ | 280 | 280K | Text | 1 |
| SmolTalk S2 | text_retention/stage2/ | 140 | 140K | Text | 2 |
| VISTA main | vista_shards/ | 163 | 146K | Video QA | 2, 3 |
| SmolTalk S3 | text_retention/stage3/ | 70 | 70K | Text | 3 |
| VISTA extra | vista_extra_shards/ | 66 | 58K | Video QA | 2, 3 |
| RLAIF-V DPO | rlaif_v/ | 84 | 84K | Image DPO | 3 (opt) |
| ShareGPT4Video | sharegpt4video_shards/ | 61 | 61K | Video caption | 1 |
| Stage3 video | stage3/ | 50 | 50K | Video QA | 3 |
| Stage3 YouTube | stage3_youtube/ | 22 | 22K | Video QA | 3 |
| WebVid | webvid/ | 19 | 19K | Video caption | 1 |
| Vript short | vript_shards/ | 11 | 11K | Video caption | 1, 3 |
| Eval val_10k | eval/val_10k/ | 10 | 10K | Mixed | Eval |

All paths are under `/workspace/data/`. All shards use unified `{"user": "...", "assistant": "...", "source": "...", "frame_count": N}` JSON format. Video ratio: 44.1%.

### Models (41GB at /workspace/models/)

| Model | Path | Purpose |
|-------|------|---------|
| SmolLM2-135M-Instruct | models/SmolLM2-135M-Instruct | Ablation workhorse |
| SmolLM2-360M-Instruct | models/SmolLM2-360M-Instruct | Scaling midpoint |
| SmolLM2-1.7B-Instruct | models/SmolLM2-1.7B-Instruct | Scaling upper bound |
| DINOv2-small | models/dinov2-small | Vision encoder (384d) |
| DINOv2-base | models/dinov2-base | Vision encoder (768d, optional) |
| SmolVLM2-256M-Video | models/SmolVLM2-256M-Video-Instruct | Eval baseline |
| SmolVLM2-2.2B | models/SmolVLM2-2.2B-Instruct | Eval baseline |

---

## 3 Bugs to Fix First

Before running any experiments, fix these (found in code audit):

### Bug 1: dtype mismatch in foveated_vlm.py
In `forward_coarse_fine()`, the loss mask concat uses mismatched dtypes:
```python
# Find this line (around line 314):
visual_no_loss = torch.zeros(B, T, dtype=torch.long, device=frames.device)
# Change to:
visual_no_loss = torch.zeros(B, T, dtype=attention_mask.dtype, device=attention_mask.device)
```

### Bug 2: dtype mismatch in multi_token_vlm.py
Same issue, around line 192:
```python
# Find:
visual_no_loss = torch.zeros(B, V_tokens, dtype=torch.long, device=frames.device)
# Change to:
visual_no_loss = torch.zeros(B, V_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
```

### Bug 3: CSV header in run_scaling_grid.py
Around line 226, the header logic corrupts CSV on re-runs:
```python
# Find:
write_header = not results_csv.exists() or i == 0
# Change to:
write_header = not results_csv.exists()
```

---

## The Execution Plan

**Full details in `docs/GPU_PHASE1_PLAN.md` (v4).** Here's the summary:

### Phase 1a: Ablation Sweeps (12 runs, ~3 days, ~$65-85)

All runs: SmolLM2-135M, 1M samples, same eval set, seed 42.

| Day | Runs | Sweep |
|-----|------|-------|
| Day 1 | F1 + F2 + F3/LR1 | Freeze strategy → pick winner |
| Day 2 | LR2 + LR3 + LR4, A1 | LR ratio + shallow query |
| Day 3 | A6 + B1, A8a + A8b + D1 | Coarse-only + baseline + image handling |

**How to run:**
```bash
torchrun --nproc_per_node=2 release/train.py --config release/configs/ablations/F1_freeze_both.yaml
torchrun --nproc_per_node=2 release/train.py --config release/configs/ablations/F2_freeze_llm.yaml
torchrun --nproc_per_node=2 release/train.py --config release/configs/ablations/baseline.yaml
```

**Dry run first:**
```bash
python release/train.py --config release/configs/ablations/baseline.yaml --dry-run
```

### Phase 1b: Scaling Grid (24 runs, ~1-2 days, ~$21)

After ablations pick the best config:
```bash
python release/scripts/run_scaling_grid.py --template release/configs/scaling/template.yaml --dry-run
python release/scripts/run_scaling_grid.py --template release/configs/scaling/template.yaml
```

### Then: Full 3-Stage Training (~$250)

With winning config + model size from above.

---

## Key Architecture Decisions (DO NOT CHANGE)

| Decision | Setting | Why |
|----------|---------|-----|
| deep_query | `True` | Shallow = uniform attention = no foveation (BUG-004) |
| query_dim | 384 | Matches DINO-small hidden dim |
| bias on query_input_proj | `False` | Bias dominated signal (BUG-002) |
| q_static/q_init init std | 1.0 | 0.02 killed gradients (BUG-001) |
| Mode selection | Per-batch | Per-sample broke DDP sync (BUG-003) |
| visual_scale | 0.14 | Matches LLM embedding std |
| Loss | Text CE only | No reconstruction, no VAE |
| Text retention | 14% SmolTalk | Removing hurts 3.7-6.5% |
| Frame rate | 1 FPS, cap 64 | Matches SmolVLM2 |
| Frame size | 224x224 | DINOv2-small native |

---

## What to Log (for paper figures)

Every run must log to both wandb and structured CSV:
- **Per step:** train_loss, loss_fine, loss_coarse, loss_ratio, grad_norm, LR per group, throughput, GPU memory
- **Per eval:** val_loss, val_loss_fine, val_loss_coarse, attention_entropy
- **Saved to disk:** attention weight maps for ~10 eval videos (heatmap figures)
- **End of run:** run_summary.json with final/best metrics, git hash, wall time

This is all implemented in logging_utils.py and attention_viz.py.

---

## Communication

| Channel | How |
|---------|-----|
| **Comms board** | `/workspace/comms/BOARD.md` — append-only, timestamped |
| **Git** | Push after each logical change. LOCAL Claude pulls regularly. |
| **wandb** | Project `foveated-vlm-ablations` (Phase 1a), `foveated-vlm-scaling` (Phase 1b) |

**Read the comms board** before starting — it has the full history of CPU pod work.

---

## Reference Docs

| Doc | Purpose |
|-----|---------|
| **docs/GPU_PHASE1_PLAN.md** | THE execution plan — ablations, scaling grid, metrics, decisions |
| **CLAUDE.md** | Project guide, architecture, debugging checklists, data paths |
| **docs/KNOWLEDGE.md** | All experiment history, bugs found, data pipeline details |
| **docs/SCALING_PLAN.md** | Overall 3-stage training plan, scaling study design |
| **docs/runpod/SMOLVLM2_REFERENCE.md** | SmolVLM2 training details (staged freeze, DoRA, text retention) |
| **core_docs/foveated_vlm_proposal.md** | Original architecture specification |

**Read order:** This file → GPU_PHASE1_PLAN.md → CLAUDE.md → start running.

---

*Last updated: 2026-02-15*
