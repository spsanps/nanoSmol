# Claude Working Guide: Foveated VLM Project

## RunPod Infrastructure

**Pod Specs:**
- CPU pod: 8 vCPU, 30GB RAM, ~1TB workspace at /workspace, 5GB system disk
- GPU pod (future): 2xA100 80GB, ~$1.39/hr

**CRITICAL: Cache redirects (run FIRST on every new pod):**
```bash
source /workspace/.bashrc_runpod
```
This redirects HOME, pip, HuggingFace, and tmp to /workspace (system disk is only 5GB).

**Auth:** Credentials in `/workspace/.env` (gitignored). Re-login with:
```bash
source /workspace/.bashrc_runpod
/workspace/venv311/bin/python -c "from huggingface_hub import login; import os; login(token=os.environ.get('HF_TOKEN', open('/workspace/.env').read().split('HF_TOKEN=')[1].split('\n')[0]))"
```

**Inter-Claude Communication:** `/workspace/comms/BOARD.md` — append-only, timestamped, `LOCAL` vs `RUNPOD` source tags.

---

## Canonical Codebase: `release/`

The `release/` folder is THE codebase. Old `src/` and `scripts/` are legacy reference only.

```
release/
├── train.py                    # torchrun --nproc_per_node=2 release/train.py --config ...
├── evaluate.py                 # python release/evaluate.py --checkpoint ... --mode coarse_fine
├── model/
│   ├── foveated_vlm.py         # FoveatedVLM: 3 forward modes, text CE loss only
│   └── encoder.py              # FoveatedEncoder: deep query on DINOv2, all bugs pre-fixed
├── data/
│   ├── webdataset_loader.py    # Tar shard loader, variable frames (1-64)
│   ├── collate.py              # Pads frames + text, creates masks
│   └── text_interleave.py      # 14% SmolTalk text retention mixing
├── utils/
│   ├── distributed.py          # DDP setup, rank helpers, reduce_mean
│   ├── checkpoint.py           # Save/load with best-metric tracking
│   ├── lr_schedule.py          # Cosine with linear warmup
│   └── logging_utils.py        # wandb + CSV + stdout
├── eval/
│   └── metrics.py              # CIDEr, BLEU, METEOR, VQA accuracy
├── configs/
│   ├── stage1_webvid.yaml      # Stage 1: WebVid captioning
│   ├── stage2_vl_sft.yaml      # Stage 2: VL SFT (Cauldron)
│   ├── stage3_video_sft.yaml   # Stage 3: Video SFT
│   ├── ablations/              # A1-A7, LR1-LR4
│   └── scaling/                # Scaling grid configs
└── scripts/
    ├── precompute.py           # Data preprocessing (tokenize + shard)
    ├── validate_shards.py      # Data integrity checks
    └── create_shard_manifest.py # Shard metadata for bucketed batching
```

**Key commands:**
```bash
# Training
torchrun --nproc_per_node=2 release/train.py --config release/configs/stage1_webvid.yaml

# Dry run (verify shapes, no training)
python release/train.py --config release/configs/stage1_webvid.yaml --dry-run

# Evaluation (3 modes: coarse_only, coarse_fine, autoregressive)
python release/evaluate.py --config release/configs/stage1_webvid.yaml \
    --checkpoint /workspace/checkpoints/stage1/best.pt --mode coarse_fine

# Data preprocessing
python release/scripts/precompute.py smoltalk --stage 1
python release/scripts/precompute.py cauldron
python release/scripts/precompute.py webvid --workers 4
```

---

## Data Paths

| Dataset | Path | Purpose |
|---------|------|---------|
| WebVid shards | `/workspace/data/webvid/*.tar` | Stage 1 training |
| Cauldron shards | `/workspace/data/cauldron/*.tar` | Stage 2 training |
| Video SFT shards | `/workspace/data/stage3/*.tar` | Stage 3 training |
| SmolTalk (stage N) | `/workspace/data/text_retention/stageN/*.tar` | 14% text retention |
| Eval (10K frozen) | `/workspace/data/eval/val_10k/*.tar` | Validation |
| Benchmarks | `/workspace/data/eval/benchmarks/` | Video-MME, MLVU, etc. |

## Model Paths

| Model | Path |
|-------|------|
| SmolLM2-135M-Instruct | `/workspace/models/SmolLM2-135M-Instruct` |
| SmolLM2-360M-Instruct | `/workspace/models/SmolLM2-360M-Instruct` |
| SmolLM2-1.7B-Instruct | `/workspace/models/SmolLM2-1.7B-Instruct` |
| DINOv2-small | `/workspace/models/dinov2-small` |
| DINOv2-base | `/workspace/models/dinov2-base` |
| SmolVLM2-256M (eval) | `/workspace/models/SmolVLM2-256M-Video-Instruct` |
| SmolVLM2-2.2B (eval) | `/workspace/models/SmolVLM2-2.2B-Instruct` |

---

## Architecture Quick Reference

**Core idea:** 1 token/frame via query-guided cross-attention on DINOv2 features.

**Two-pass training (parallel approximation):**
1. Coarse: q_static → all frames → z_coarse → LLM → dynamic queries
2. Fine: shifted queries → all frames → z_fine → LLM + text → loss

**Three eval modes:**
- `coarse_only` — fastest, single static-query pass
- `coarse_fine` — matches training (two parallel passes)
- `autoregressive` — true sequential inference with KV cache

**Train/inference gap < 0.6%** — parallel approximation is valid.

---

## Key Decisions (DO NOT CHANGE without discussion)

### Loss Masking
- **Stage 1 (WebVid captioning):** Loss on ALL text tokens (prompt + caption)
- **Stage 2-3 (SFT):** Loss on ANSWER tokens only (mask user prompts)
- **Visual tokens:** NEVER have loss (DINO features, not predicted text)
- **SmolTalk retention:** Follows same loss rule as the stage it's in

### Architecture Settings
- `deep_query=True` (shallow = uniform attention — BUG-004)
- `query_dim=384`, `bias=False` on query_input_proj (BUG-002)
- `q_static`/`q_init` init with `std=1.0` (BUG-001)
- Mode selection per-batch, not per-sample (BUG-003)

### Training
- No reconstruction loss, no VAE — text CE only
- Full weight fine-tuning (no DoRA/LoRA)
- Differential LR: connector (1e-4) > LLM (1e-5) > DINO (1e-5)
- 14% SmolTalk in ALL stages
- 1 FPS, variable frames 1-64, cap at 64 (matches SmolVLM2)
- Frame size 224x224 (DINOv2 native, NOT 384 like SigLIP)

### Prompt Format
- Stage 1: `<|user|>What would be the WebVid caption for this video?<|end|><|assistant|>{caption}<|end|>`
- Stage 2-3: Datasets have instruction format, use as-is

---

## Debugging Checklist

When `loss_fine == loss_coarse` (or ratio stays at 1.0):

1. **Query init scale:** `model.q_static.std()` should be ~1.0
2. **Attention entropy:** High (near log N) = uniform = BAD
3. **Embedding difference:** `(embed_q1 - embed_q2).abs().mean()` should be > 0.5
4. **Per-batch mode selection:** NOT per-sample
5. **deep_query=True:** Shallow mode → uniform attention

When loss is NaN or exploding:
1. Check gradient norm (should be < 10 after clipping)
2. Check learning rates (connector shouldn't be > 1e-3)
3. Check visual_scale (should match LLM embedding std ~0.14)

---

## Critical References

| Document | Purpose |
|----------|---------|
| `core_docs/foveated_vlm_proposal.md` | Architecture specification |
| `docs/KNOWLEDGE.md` | Bugs, fixes, experiments, insights |
| `docs/SCALING_PLAN.md` | 3-stage training plan, scaling study |
| `docs/runpod/BRIEFING.md` | Full project context for fresh invocations |
| `docs/runpod/SMOLVLM2_REFERENCE.md` | SmolVLM2 training details |
| `docs/RESEARCH_PLAYBOOK.md` | Research methodology |

---

*Last updated: 2026-02-11*
