# RunPod Claude: Repo Sync & Cleanup Instructions

**Date:** 2026-02-14
**Source:** LOCAL Claude
**Priority:** Do this BEFORE starting any GPU ablation work

---

## 1. Pull Latest from Git

The local repo has been cleaned up and pushed. Pull to get:
- `docs/GPU_PHASE1_PLAN.md` — the definitive ablation plan (12 runs, 5 sweeps)
- `docs/legacy/` — 9 reconstruction-era docs moved here
- `docs/SCALING_PLAN.md` — Section 6 now points to GPU_PHASE1_PLAN.md
- `CLAUDE.md` — updated references
- 22 stale `codex/*` branches deleted from remote

```bash
cd /workspace/workdir/nanoSmol/fVLM
git pull origin main
git fetch --prune
```

---

## 2. Verify Clean State

```bash
git status          # should be clean
git log --oneline -5  # should show "docs: Add GPU Phase 1 plan..." as latest
```

---

## 3. Code Changes Needed Before Ablation Runs

These are the 6 items from GPU_PHASE1_PLAN.md that need implementation.
**Work in release/ only.** All changes should be committed and pushed.

### 3a. Freeze Config (train.py)
Add `freeze_dino: true/false` and `freeze_llm: true/false` to config.
In train.py, after model creation, freeze parameters based on config:
```python
if config.model.get('freeze_dino', False):
    for p in model.dino.parameters():
        p.requires_grad = False
if config.model.get('freeze_llm', False):
    for p in model.llm.parameters():
        p.requires_grad = False
```
Create configs: `release/configs/ablations/F1.yaml` (freeze both), `release/configs/ablations/F2.yaml` (freeze LLM only).
F3 = baseline.yaml (nothing frozen).

### 3b. Loss Routing (train.py)
Add config option `loss_mode: all_tokens | answer_only`.
- `all_tokens`: CE on all text tokens (Stage 1 behavior)
- `answer_only`: CE only on assistant response tokens (Stage 2-3)
The dataloader already has user/assistant structure — use it to create the loss mask.

### 3c. A6 Coarse-Only Fix (train.py)
**BUG:** A6_coarse_only.yaml sets `model.coarse_only: true` but train.py ignores it — always uses `mode="coarse_fine"`.
Fix: read `config.model.get('coarse_only', False)` and pass `mode="coarse_only"` to forward() when True.

### 3d. A8 Wiring Verification
`replicate_image_frames` is in the dataloader code. Verify it actually works:
- Load a Cauldron shard with `replicate_image_frames: 16`
- Check the batch has 16 identical frames
- Quick smoke test, not a full training run

### 3e. B1 Multi-Token Baseline
New model variant: skip foveated encoder entirely.
- Take DINOv2 patch tokens (14x14 = 196 patches for 224x224 input)
- Average-pool to 4x4 grid = 16 tokens per frame
- Linear project each to LLM dim
- Feed directly to LLM (no query mechanism, no two-pass)
- Config: `release/configs/ablations/B1_multi_token.yaml`

Implementation options:
1. Add a `multi_token` mode to FoveatedVLM, OR
2. Create a small `MultiTokenVLM` class that reuses DINO + LLM but skips the encoder

### 3f. Attention Logging
At eval time (every N steps), save:
- Raw attention weights from encoder cross-attention for ~10 fixed eval samples
- Attention entropy per step (scalar, logged to wandb)
- These are for paper figures (heatmaps + entropy plot)

---

## 4. Config Updates for Phase 1

Existing configs that need updating to 1M samples:
- All ablation configs currently have `max_samples: 360000` — change to `max_samples: 1000000`
- All should use the same seed for deterministic data ordering
- Add `seed: 42` to all configs if not present

New configs needed:
| Config | Description |
|--------|-------------|
| `F1.yaml` | freeze_dino: true, freeze_llm: true (connector only) |
| `F2.yaml` | freeze_dino: false, freeze_llm: true (connector + DINO) |
| `D1.yaml` | 55% video / 30% image / 15% text mix |
| `B1_multi_token.yaml` | 16 tokens/frame, no foveation |

Existing configs to verify match the plan:
| Config | Verify |
|--------|--------|
| baseline.yaml (=F3/LR1) | full unfreeze, connector=1e-4, backbone=1e-5 |
| LR2.yaml | connector=1e-3, backbone=1e-5 |
| LR3.yaml | connector=3e-5, backbone=1e-5 |
| LR4.yaml | all=1e-5 (uniform) |
| A1_deep_query_off.yaml | deep_query: false |
| A6_coarse_only.yaml | coarse_only: true (+ fix train.py to honor it) |
| A8_static_16frames.yaml (=A8b) | replicate_image_frames: 16 |

Need to create or rename:
- `A8_static_1frame.yaml` (=A8a) — `replicate_image_frames: 1` (or just don't replicate)

---

## 5. Version Control Workflow

- **Always commit before and after each change**
- **Push to origin/main** after each logical unit of work
- **Commit messages:** `type: description` (e.g., `feat: Add freeze config support to train.py`)
- **No force pushes** — LOCAL Claude may pull at any time
- **Tag completed milestones:** `git tag phase1-ready` when all 6 code changes are done

---

## 6. Priority Order

1. **Git pull + verify** (5 min)
2. **Read GPU_PHASE1_PLAN.md** thoroughly (10 min)
3. **Freeze config + F1/F2 yamls** (30 min) — unblocks Sweep 1
4. **A6 coarse-only fix** (15 min) — unblocks Sweep 3
5. **Loss routing** (30 min) — needed for A8 runs
6. **Config updates** (1M samples, seed) (20 min)
7. **B1 multi-token baseline** (1-2h) — can defer, Sweep 5 runs last
8. **Attention logging** (1h) — can defer, runs in eval phase
9. **A8 smoke test** (15 min)
10. **Dry run** all 12 configs (30 min) — `python release/train.py --config X --dry-run`

---

*Written by LOCAL Claude, 2026-02-14*
