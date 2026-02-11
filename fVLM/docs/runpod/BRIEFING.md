# RunPod Briefing: Foveated VLM Project

**READ THIS FIRST.** You are Claude Code, freshly invoked on a RunPod instance with GPUs. This document contains everything you need from prior conversation context. Your job is to build and train a release-quality foveated VLM.

---

## What Is This Project?

**fVLM** = Foveated Vision-Language Model. A novel video VLM that processes each video frame with **1 token** (not 196+ patches like standard VLMs). The LLM generates queries that control WHERE to look in each frame via cross-attention with DINOv2 features — like biological foveated vision.

**Novel architecture:**
```
Standard VLM:  frame → 196 patches → pixel_shuffle → ~81 tokens → LLM
Foveated VLM:  frame → DINO features → query_cross_attention → 1 token → LLM
```

**The thesis:** Foveated attention (1 token/frame, query-guided) matches multi-token baselines at 16x fewer visual tokens. The LLM learns to look at what matters.

---

## Key Architecture Components

| Component | What It Does | In Code |
|-----------|-------------|---------|
| **DINOv2 encoder** | Extracts patch features from video frames | `src/model/encoder.py` |
| **Query cross-attention** | Uses LLM-generated query to attend to DINO features → 1 token | `src/model/encoder.py` |
| **`q_static`** | Learnable static query (coarse pass, same for all frames) | `src/model/foveated_vlm.py` |
| **`q_init`** | Initial query for fine/autoregressive pass | `src/model/foveated_vlm.py` |
| **Query head** | LLM hidden state → next query (enables autoregressive refinement) | `src/model/foveated_vlm.py` |
| **SmolLM2 LLM** | Processes visual tokens + text, generates queries | Pretrained from HuggingFace |
| **Prediction head** | FiLM-conditioned head (used in prior reconstruction experiments, may not be needed for text-only loss) | `src/model/prediction.py` |

**The "connector" in fVLM** = `query_input_proj` + cross-attention weights + `q_static` + `q_init` + query_head. This is the novel part, randomly initialized, needs highest learning rate.

### Three Evaluation Modes (CRITICAL)

| Mode | How | Speed |
|------|-----|-------|
| **Coarse-only** | Static `q_static` → all frames parallel | Fast |
| **Coarse→Fine** | Coarse → LLM → queries → fine pass (training approx) | Medium |
| **Autoregressive Fine** | `q_init` → frame₁ → LLM → q₁ → frame₂ → ... (sequential) | Slow but true inference |

All three modes must be evaluated and compared. The gap between coarse→fine and autoregressive fine measures the train/inference mismatch.

---

## Current State (What's Been Done)

### Completed Experiments (on local RTX 4090, 5K samples — TOY SCALE)
- 10/12 scaling law runs across 3 configs × 2 frame counts × 2 architectures
- Foveated is 1.6-2.4% worse (parameter-corrected) but 16x fewer visual tokens
- LLM size matters most (7% gain) vs DINO size (<1%)
- Train/inference gap < 0.6% (parallel approximation works)
- Results in `docs/KNOWLEDGE.md` (2100+ lines of experiment history)

### Key Bugs Found & Fixed (see CLAUDE.md for details)
1. Query init too small (std=0.02 → std=1.0)
2. Query projection bias dominated (removed bias)
3. Per-sample mode selection broke batching (fixed to per-batch)

### What's NOT Done (your job)
- Everything at real scale (3M+ samples, proper hardware)
- Full training pipeline (3-stage, matching SmolVLM2)
- Proper evaluation suite (CIDEr, BLEU, Video-MME)
- Speed optimization
- HuggingFace release

---

## The Plan (SCALING_PLAN.md has full details)

Read `docs/SCALING_PLAN.md` for the complete plan. Here's the executive summary:

### Training Pipeline (3 stages, NO reconstruction, NO DoRA)

| Stage | Loss | Data | Trainable |
|-------|------|------|-----------|
| **1. WebVid captioning** | Text CE | 2M WebVid + 14% text | ALL params (diff LR: high connector, low backbone) |
| **2. VL SFT** | Text CE (answer tokens only) | 1M Cauldron + 14% text | ALL params (full fine-tuning) |
| **3. Video SFT** | Text CE (answer tokens only) | 0.5M LLaVA-Video + Vista-400K + mix | ALL params |

**Critical decisions:**
- **No SD-VAE / no reconstruction.** Text CE loss only. This is a VLM for understanding, not generation.
- **No DoRA/LoRA.** Full weight fine-tuning. Our models (135M-1.7B) are small enough. Monitor for divergence; freeze DINO first if it happens.
- **Model size TBD by scaling study.** Don't pre-commit to 1.7B. Run ablations at 135M, scaling grid at all 3 sizes, then decide.
- **1 FPS, variable frame count.** 5s video = 5 frames, 30s = 30 frames, cap 64 (matches SmolVLM2). Pre-sort shards by duration for efficient bucketed batching.
- **14% text data in ALL stages.** Preserves instruction-following. Removing it hurts 3.7-6.5%.
- **WebVid captions are noisy.** Format as "What would be the WebVid caption for this video?" — don't pretend they're natural descriptions.

### Execution Order on RunPod

```
Phase 0: CPU precompute (frames + tokenization, no GPU needed)
Phase 1: Speed optimization (profile, compile, benchmark — target 2x throughput)
Phase 2: Ablations (6-10 runs at SmolLM2-135M, 360K samples each)
Phase 3: Scaling grid (24 runs across 3 model sizes — determines final model)
Phase 4: Final model training (3-stage, winning config)
Phase 5: Evaluation (3 eval modes + benchmarks + Q-before-video experiment)
Phase 6: Release (HuggingFace model + blog + report)
```

### Ablations to Run (at 135M, cheap)

**Architecture:** deep query on/off, multi-fine iterations, tokens/frame, DINO frozen/tuned, attention temperature, prediction head capacity, Stage 1 frozen/unfrozen backbone

**Learning rate:** 4 runs testing connector:backbone LR ratio (10x, 100x, 50x, uniform)

**Task:** captioning only, captioning+VQA, full SmolVLM2 mix, Q-before-video ordering

### Q-Before-Video Experiment

For QA tasks, put the question BEFORE the video frames. The fine queries become task-aware AND temporally-aware (coarse queries remain task-agnostic). Train with 50/50 mix of both orderings, eval both separately. If fine improves with Q-first but coarse doesn't, that's direct evidence the query mechanism uses task context.

---

## Engineering Patterns (from nanochat/SmolLM Playbook)

Read `docs/RESEARCH_PLAYBOOK.md` for full methodology. Key patterns:

| Pattern | Implementation |
|---------|---------------|
| **One variable at a time** | Each ablation changes exactly one thing |
| **Fixed sample count** | All ablation runs see same number of samples |
| **Mandatory wandb** | Every run logged, project: `foveated-vlm-{phase}` |
| **Structured CSV** | One row per eval point for scaling analysis |
| **CPU-only precompute** | Frames + tokenization to disk. DINO runs during training (features change when fine-tuning) |
| **`torch.compile`** | On DINO forward + query attention. Target >80% GPU utilization |
| **nanochat speedrun.sh** | One script runs entire pipeline end-to-end |
| **Iso-FLOP analysis** | Auto-compute training steps from FLOP budget. Smaller models train longer at same budget |

### Precompute Strategy (CPU only)

| Precompute (CPU) | During Training (GPU) |
|------------------|--------------------|
| Video download → disk | DINO forward pass |
| Frame extraction (ffmpeg) | Query cross-attention |
| Frame resize/normalize | LLM forward + backward |
| Caption tokenization | Loss computation |

Do NOT precompute DINO features — storage would be TBs, and features change when fine-tuning the encoder.

---

## SmolVLM2 Reference (what we're matching)

SmolVLM2's pipeline for reference:
- **Stage 1 (Vision):** SigLIP frozen, train connector, then DoRA on backbones
- **Stage 2 (Video):** 3.3M samples, 33% video / 34% image / 20% text / 12% multi-image
- **Loss:** Always text CE, answer tokens only
- **Benchmarks:** MMMU 42.0, MathVista 51.5, Video-MME 52.1, MLVU 55.2, DocVQA 80.0
- **SmolTalk** (~1.3M samples) is their instruction SFT dataset (published on HuggingFace)
- **14% text retention** prevents catastrophic forgetting of language
- **<0.05% CoT** — more chain-of-thought hurts small models
- **Idefics3 diverged with full unfreezing** at 8B — but we're smaller, should be fine with monitoring

---

## File Map

```
fVLM/
├── CLAUDE.md                           # Project guide, architecture, debugging checklists
├── src/model/                          # Core model code
│   ├── foveated_vlm.py                 # Main model (FoveatedVLM)
│   ├── encoder.py                      # Vision encoder + query cross-attention
│   └── prediction.py                   # Prediction head (FiLM)
├── src/data/                           # Data loading
├── src/training/                       # Training utilities
├── scripts/                            # Runnable scripts
│   ├── train_multitask.py              # Current training script (needs rewrite for 3-stage)
│   └── setup/                          # Data/model download
├── configs/                            # YAML training configs
├── core_docs/                          # AUTHORITATIVE architecture docs
│   ├── foveated_vlm_proposal.md        # Architecture specification
│   └── foveated_vlm_execution_guide.md # Implementation guide
├── docs/
│   ├── runpod/                         # THIS FOLDER — read first
│   │   ├── BRIEFING.md                 # This file
│   │   └── SMOLVLM2_REFERENCE.md      # SmolVLM2 training details
│   ├── SCALING_PLAN.md                 # Full plan (phases, budget, ablations, scaling grid)
│   ├── RESEARCH_PLAYBOOK.md            # Methodology (from nanochat/SmolLM/SmolVLM)
│   ├── KNOWLEDGE.md                    # All experiment history, bugs, insights
│   ├── HANDOFF.md                      # Previous session handoff
│   └── IDEAS_TO_TRY.md                # Architecture ideas (some outdated)
└── research/scaling_laws/              # Previous scaling experiments (toy scale)
```

**Read order for full context:**
1. This file (`docs/runpod/BRIEFING.md`)
2. `CLAUDE.md` (project guide + debugging)
3. `docs/SCALING_PLAN.md` (the plan)
4. `docs/RESEARCH_PLAYBOOK.md` (methodology)
5. `core_docs/foveated_vlm_proposal.md` (architecture spec, if needed)
6. `docs/KNOWLEDGE.md` (experiment history, if needed)

---

## Budget

$500 total on RunPod. Estimated ~$300-310 for the full pipeline. See `docs/SCALING_PLAN.md` Section 4 for detailed breakdown.

| What | Cost |
|------|------|
| Network volume (500GB, ~1 month) | ~$35 |
| CPU pod (~15h for data precompute) | ~$5-15 |
| GPU pod (2xA100, ~185h for training) | ~$256 |
| **Total** | **~$300-310** |

Start with **2xA100 80GB**. Scale to 4x only if training is the bottleneck.

---

## RunPod Infrastructure Setup

### Step 1: Network Volume

Create a **500GB network volume** on RunPod Secure Cloud.
- Cost: $0.07/GB/month = ~$35/month
- Pick a datacenter that has A100 80GB available (volume + pods must be same datacenter)
- Volume persists independently of pods — data survives pod termination
- Mounted at `/workspace` by default

### Step 2: CPU Pod (Data Precompute)

Spin up a **CPU-only pod**, attach the network volume. This does ALL data prep without paying for GPUs.

**What WebVid-10M is:** The HuggingFace dataset (`TempoFunk/webvid-10M`) is **metadata only** — 10.7M rows of video URLs + captions (2.9GB). Actual videos are on Shutterstock CDN and must be downloaded via the URLs.

**Pipeline (all CPU, ~10-15h with 16 workers):**

```bash
# 1. Install tools
pip install video2dataset huggingface_hub datasets

# 2. Download WebVid metadata from HuggingFace
python -c "from datasets import load_dataset; ds = load_dataset('TempoFunk/webvid-10M')"

# 3. Bulk download videos + extract frames at 1 FPS + resize + shard
#    video2dataset handles the full pipeline: download → ffmpeg → resize → webdataset shards
#    ~230 videos/sec on 16 cores
video2dataset --url_list webvid_urls.parquet \
    --output_folder /workspace/webvid_frames \
    --output_format webdataset \
    --input_format webvid \
    --url_col contentUrl \
    --caption_col name \
    --frame_rate 1 \
    --resize_mode center_crop \
    --resize 224 \
    --number_sample_per_shard 1000 \
    --processes_count 16

# 4. Tokenize captions (add SmolLM2 token IDs to shards)
python scripts/setup/tokenize_captions.py --shards /workspace/webvid_frames

# Output: /workspace/webvid_frames/
#   ├── 00000.tar (1000 samples each: JPEG frames + captions + token_ids)
#   ├── 00001.tar
#   └── ... (~3000 shards for 3M samples)
```

**What gets precomputed (CPU) vs what stays in training loop (GPU):**

| Precompute on CPU pod | Runs during GPU training |
|-----------------------|--------------------------|
| Video download from Shutterstock CDN | DINO forward pass (features change with fine-tuning) |
| Frame extraction (ffmpeg, 1 FPS) | Query cross-attention |
| Frame resize to 224×224 | LLM forward + backward |
| Caption tokenization | Loss computation |
| Tar sharding (webdataset format) | — |

**Why NOT precompute DINO features:** Would be ~TBs of storage (3M videos × variable frames × 196 patches × 384 dim × 2 bytes). Also, DINO features change when we fine-tune the encoder in Stage 2+. Not worth it.

**Output size:** ~50GB compressed (JPEG frames + tokenized captions). Fits easily in 500GB volume with room for checkpoints.

**After precompute:** Terminate the CPU pod. Volume stays. Data stays.

### Step 3: GPU Pod (Training)

Spin up **2xA100 80GB**, attach the **same network volume**.

```bash
# Verify setup
nvidia-smi                    # Should show 2x A100 80GB
ls /workspace/webvid_frames/  # Should show precomputed shards
```

Data is already at `/workspace`. No download or preprocessing needed during training. GPU utilization should be >80% from the start.

### Step 4: Also Download Stage 2-3 Datasets

These are smaller and can be downloaded directly on the GPU pod (or precomputed on CPU pod if you want):

| Dataset | Size | Source | For |
|---------|------|--------|-----|
| The Cauldron | ~50GB | HuggingFace (HuggingFaceM4/the_cauldron) | Stage 2: VL SFT |
| LLaVA-Video-178K | ~30GB | HuggingFace | Stage 3: Video SFT |
| Vista-400K | ~20GB | HuggingFace | Stage 3: Temporal reasoning |
| FineVideo | ~10GB | HuggingFace | Stage 3: Narrative |
| SmolTalk subset | ~2GB | HuggingFace (HuggingFaceTB/smoltalk) | 14% text retention |
| RLAIF-V | ~1GB | HuggingFace (HuggingFaceH4/rlaif-v_formatted) | Optional DPO |

These are instruction-formatted and don't need the video2dataset pipeline — they download directly via `huggingface_hub` or `datasets` library.

---

## CRITICAL: Redirect All Caches to /workspace

**System disk (`/`) is only 5GB.** It WILL fill up and crash the pod if you install things to default locations. **ALL caches, packages, and temp files must go to `/workspace`.**

Run this FIRST on every new pod (before installing anything):

```bash
# Create workspace directories
mkdir -p /workspace/.cache /workspace/.local /workspace/tmp /workspace/.npm /workspace/.pip

# Redirect everything
export HOME=/workspace
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.pip/cache
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export XDG_CACHE_HOME=/workspace/.cache
export NPM_CONFIG_PREFIX=/workspace/.npm-global
export PATH="/workspace/.npm-global/bin:/workspace/.local/bin:$PATH"

# Make persistent across shell sessions
cat >> /workspace/.bashrc_runpod << 'ENVEOF'
export HOME=/workspace
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.pip/cache
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export XDG_CACHE_HOME=/workspace/.cache
export NPM_CONFIG_PREFIX=/workspace/.npm-global
export PATH="/workspace/.npm-global/bin:/workspace/.local/bin:$PATH"
ENVEOF
echo 'source /workspace/.bashrc_runpod' >> ~/.bashrc
```

**This persists on the network volume** — when you terminate a CPU pod and start a GPU pod on the same volume, the caches and installed packages survive. Only system packages (apt) need reinstalling.

---

## First Steps When Invoked on RunPod

**If on CPU pod (data precompute):**
1. Read this file + `CLAUDE.md`
2. **Run the cache redirect script above FIRST**
3. Set up environment
4. Run the CPU data prep pipeline above
5. Verify shards: count, sample a few, check frame quality
6. Done — terminate pod, tell user to spin up GPU pod

**If on GPU pod (training):**
1. **Run the cache redirect script above FIRST** (or `source /workspace/.bashrc_runpod` if it exists from CPU pod)
2. Verify GPU access (`nvidia-smi`) and data at `/workspace`
3. Read this file + `CLAUDE.md` + `docs/SCALING_PLAN.md`
4. Clone repo, set up environment (Python, PyTorch, transformers, wandb)
5. **Phase 1:** Profile and optimize training throughput (target 2x improvement)
6. **Phase 2:** Run ablations at SmolLM2-135M
7. **Phase 3:** Scaling grid → determines final model size
8. **Phase 4:** Train final model (3-stage pipeline)
9. **Phase 5:** Evaluate (3 modes + benchmarks)
10. **Phase 6:** Package and release to HuggingFace

---

## Communication Between Claude Instances

There are two Claude Code instances running on this project:

| Instance | Where | Can do |
|----------|-------|--------|
| **LOCAL** | User's laptop (WSL) | SSH to RunPod, git push/pull, read/write `/workspace` via SSH |
| **RUNPOD** | On the pod | Direct filesystem access, runs jobs, git push/pull |

**Shared message board:** `/workspace/comms/BOARD.md`

- **Check this file** when you start a session or before major decisions
- **Append messages** at the bottom (never delete previous messages)
- Format: `## [YYYY-MM-DD HH:MM] LOCAL` or `## [YYYY-MM-DD HH:MM] RUNPOD`
- Use for: status updates, questions, handoffs, blocker reports
- For code/config changes: use git (push from one side, pull from the other)

**The user may not always be available.** When blocked, write to the board and continue on other tasks. The other Claude or the user will respond.

---

*Created: 2026-02-09*
*Source: Multi-session conversation covering architecture, scaling, SmolVLM2 analysis, nanochat patterns*
