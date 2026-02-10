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
- **1 FPS, variable frame count.** 5s video = 5 frames, 30s = 30 frames, cap ~50.
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

$500 total on RunPod. Estimated ~$256 for the full pipeline (2xA100 80GB). See `docs/SCALING_PLAN.md` Section 4 for detailed breakdown.

Start with **2xA100 80GB**. Scale to 4x only if training is the bottleneck.

---

## First Steps on RunPod

1. Clone the repo, verify GPU access (`nvidia-smi`)
2. Read this file + `CLAUDE.md` + `docs/SCALING_PLAN.md`
3. Set up environment (Python, PyTorch, dependencies)
4. **Phase 0:** Build CPU data prep pipeline, single training script, eval harness
5. **Phase 1:** Profile and optimize training throughput (target 2x improvement)
6. Then proceed through phases 2-6 as planned

---

*Created: 2026-02-09*
*Source: Multi-session conversation covering architecture, scaling, SmolVLM2 analysis, nanochat patterns*
