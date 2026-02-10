# fVLM Scaling Plan

**Date:** 2026-02-09 (v3)
**Target:** Release-quality foveated VLM on HuggingFace
**Budget:** $500 on RunPod
**Reference scale:** SmolVLM2 (~3.3M samples), nanochat (~$100 full pipeline)

> Supersedes: `SCALING_RECOMMENDATIONS.md` (Jan 18, was 100K samples / 4090-only scope)

---

## 1. What We're Shipping

| Item | Detail |
|------|--------|
| **Model** | Foveated VLM: 1 token/frame video understanding |
| **Backbone** | SmolLM2 (size TBD by scaling study) + DINOv2-S |
| **Loss** | Next-token prediction on text (never reconstruction) |
| **Training data** | ~3.3M samples across 3 stages (see Section 3) |
| **Release** | HuggingFace model card, safetensors, inference scripts, demo |
| **Blog** | Scaling story, ablation tables, attention visualizations |
| **Novel claim** | Foveated attention matches multi-token baselines at 16x fewer visual tokens |

> **No SD-VAE in the pipeline.** This is a video *understanding* model, not generation. All losses are next-token prediction on text (captions, answers). SmolVLM2 never does reconstruction either.

---

## 2. Model Configs

Three SmolLM2 sizes for scaling study. **Final model size determined by scaling law results, not pre-committed.**

| Config | LLM | DINO | Est. Params | Role |
|--------|-----|------|-------------|------|
| **S** | SmolLM2-135M | dinov2-small | ~160M | Ablation workhorse |
| **M** | SmolLM2-360M | dinov2-small | ~390M | Scaling midpoint |
| **L** | SmolLM2-1.7B | dinov2-small | ~1.8B | Scaling upper bound |

Visual tokens per frame: **1** (foveated) vs **4, 16** (baseline comparisons).

> **LLM variant:** Use `-Instruct` for final model. BUT instruction-following capability is maintained via **14% text instruction data** in all training stages (SmolTalk subset), NOT "free" from the backbone. Fine-tuning without text retention will lose instruct capabilities.

---

## 3. Training Pipeline (modeled on SmolVLM2 / Idefics3)

### 3a. How SmolVLM2 Actually Trains (reference)

SmolVLM2 follows Idefics3's staged unfreezing. Key: **loss is always next-token prediction on text. Never reconstruction.**

| Stage | Vision Encoder | Connector | LLM | Loss | Data |
|-------|---------------|-----------|-----|------|------|
| **Idefics3 Stage 1** | Frozen | Train (new) | Frozen | Text CE | Cauldron + Docmatix |
| **Idefics3 Stage 2-3** | DoRA | Train | DoRA | Text CE | + synthetic data |
| **Idefics3 Stage 4 (SFT)** | DoRA | Train | DoRA | Text CE (answer tokens only) | Cauldron extended |
| **SmolVLM2 Video SFT** | DoRA | Train | DoRA | Text CE (answer tokens only) | 3.3M: 33% video, 34% image, 20% text, 12% multi-image |

> Idefics3 tried full unfreezing — it **diverged**. DoRA on backbones was the stable solution. However, Idefics3 is 8B and was adding vision to a fully-converged LLM. Our situation is different (smaller models, novel architecture), so we opt for **full weight fine-tuning** with continuous monitoring.

### 3b. Our Training Stages

**Full weight fine-tuning throughout. No DoRA/LoRA.** Monitor loss + gradient norms continuously; fall back to freezing components only if divergence is observed.

```
SmolLM2-{size}-Instruct (pretrained)
    │
    ├── Stage 1: Visual Connector + WebVid Captioning
    │     Trains:   ALL params (full fine-tuning) — see note below
    │     Loss:     Next-token prediction on captions
    │     Data:     ~2M WebVid-10M video-caption pairs + 14% text
    │     Format:   "What would be the WebVid caption for this video?" → caption
    │     LR:       Differential — higher for connector, lower for DINO/LLM
    │     Purpose:  Learn the foveated query mechanism + visual grounding
    │
    ├── Stage 2: Vision-Language SFT
    │     Trains:   ALL params (full fine-tuning)
    │     Loss:     Next-token prediction on answer tokens only
    │     Data:     ~1M from The Cauldron subset + 14% text
    │     Monitor:  Loss, grad norms, attention entropy — watch for divergence
    │     Purpose:  Learn to answer questions about images/video
    │
    ├── Stage 3: Video SFT
    │     Trains:   ALL params (full fine-tuning)
    │     Loss:     Next-token prediction on answer tokens only
    │     Data:     ~0.5M from LLaVA-Video-178K + Vista-400K + FineVideo
    │     Mix:      33% video + 34% image + 20% text + 12% multi-image
    │     Purpose:  Temporal reasoning, narrative understanding
    │
    └── Optional: DPO (full fine-tuning or LoRA)
          Data:     RLAIF-V (83K preference pairs)
          Purpose:  Reduce hallucinations
```

> **Why NOT freeze backbones in Stage 1?**
> The standard rationale (LLaVA, Idefics3) is: random connector → garbage visual tokens → corrupt the LLM. But:
> - At 135M-1.7B, models are much more plastic than 8B
> - Differential learning rates (10x lower for backbone) prevent corruption
> - Gradient clipping + monitoring catches problems early
> - Full fine-tuning lets DINO adapt its features to be useful for query attention from the start
>
> **Ablation A7:** Test frozen-backbone Stage 1 vs full-unfreeze Stage 1 at 135M. Cheap to test.

> **Why full fine-tuning, not DoRA?**
> - Idefics3 diverged at 8B. We're 5-10x smaller — less fragile
> - Our foveated connector is novel — full gradients needed for query mechanism to learn
> - DoRA constrains the model's ability to learn new attention patterns
> - Fallback plan: if divergence occurs, freeze DINO first, then freeze LLM layers bottom-up
> - **Continuous monitoring required**: log grad norms + loss every step, set alerts for NaN or spikes

### 3c. Caption Formatting

**Problem:** WebVid captions are noisy stock-video-ese, not natural language:
```
"4K aerial drone footage of beautiful sunset over ocean waves stock video"
"Slow motion cinematic shot of coffee being poured into cup B-roll"
"HD timelapse of clouds moving over mountain landscape royalty free"
```

Training with "Describe this video" → these captions would teach the model to always respond in stock-footage language.

**Solution:** Frame the prompt honestly so the model knows it's producing a WebVid-style caption:

```
<|user|>What would be the WebVid caption for this video?<|end|>
<|assistant|>4K aerial drone footage of beautiful sunset over ocean waves stock video<|end|>
```

Or minimal format:
```
<|user|>Caption:<|end|>
<|assistant|>4K aerial drone footage of beautiful sunset...<|end|>
```

**Why this works:** Stage 1 is just to train the visual connector — the model learns to extract visual features, not conversational style. In Stage 2-3, properly instruction-formatted data (Cauldron, LLaVA-Video) teaches real description and QA skills.

**Do NOT rewrite captions** — expensive and unnecessary. Just frame the task honestly.

**Stage 2-3 format:** Already instruction-formatted in The Cauldron / LLaVA-Video datasets.

**14% text retention:** SmolTalk-style instruction data (general Q&A, math, coding) to preserve instruction-following.

### 3d. SmolVLM2 Training Insights (apply to our pipeline)

| Insight | Detail | Action |
|---------|--------|--------|
| **14% text retention** | Removing text from VLM stages hurts by 3.7-6.5% | Always keep 14% text-only data in mix |
| **Text source matters** | Reusing LLM-SFT text hurts small VLMs | Use curated text Q&A, not raw SmolTalk dump |
| **CoT is harmful at small scale** | >0.05% CoT examples degrade small models | Keep CoT minimal |
| **Instruct needs text data** | Fine-tuning loses instruct capabilities without text retention | 14% text in every stage preserves instruction-following |
| **Answer-only loss** | Masking user prompts in SFT improved accuracy | Compute loss only on assistant answer tokens |
| **Checkpoint selection** | Weighted composite metric across benchmarks | Not all metrics improve monotonically (DocVQA degrades) |
| **Full unfreeze risk** | Idefics3 (8B) diverged with full unfreezing | Monitor continuously; our smaller models may be fine |

### 3e. Frame Sampling

| Item | Spec |
|------|------|
| **Frame rate** | Fixed **1 FPS** (industry standard: SmolVLM2, LLaVA-Video, LLaVA-OneVision) |
| **Variable frame count** | 5s video = 5 frames, 30s video = 30 frames, cap at ~50 |
| **No fixed frame count** | Different videos get different numbers of frames |
| **Batching** | Pad to max frames in batch + attention mask, OR best-fit packing |
| **Resolution** | DINO input resolution (224×224 for ViT-S, 518×518 for ViT-L) |

> SmolVLM2 uses 1 FPS up to 50 frames. LLaVA-Video found >100 frames needed for 30s+ dynamic content. Our foveated architecture may need fewer frames since queries can focus on what matters.

### 3f. Data Plan

| Stage | Source | Samples | Notes |
|-------|--------|---------|-------|
| Stage 1 | WebVid-10M + 14% SmolTalk | ~2M | Video-caption pairs |
| Stage 2 | The Cauldron subset + 14% text | ~1M | Instruction-formatted VQA |
| Stage 3 | LLaVA-Video-178K + Vista-400K + FineVideo | ~0.5M | 33% video / 34% image / 20% text / 12% multi-image |
| DPO (opt.) | RLAIF-V | 83K | Preference pairs |
| **Val set** | Fixed, never changes | 10K | Same across all stages |

### 3g. What to Precompute (CPU vs GPU)

Follow nanochat's pattern: **precompute CPU-bound work, keep GPU-bound work in training loop.**

| What | Where | Precompute? | Why |
|------|-------|-------------|-----|
| Video download | Network | **Yes** → disk | Eliminate network variability |
| Frame extraction (ffmpeg) | CPU | **Yes** → disk | Biggest CPU bottleneck (0.2s/sample) |
| Frame resize/normalize | CPU | **Yes** → disk | Trivial but adds up |
| Caption tokenization | CPU | **Yes** → disk | nanochat precomputes this |
| DINO forward pass | GPU | **No** — run during training | Storage too large (TBs for 3M×variable frames×196 patches×384d). Also, DINO gets DoRA in Stage 2+, so features change |
| LLM forward pass | GPU | **No** | Always trainable |

**Storage format:** Sharded tar files (nanochat-style). Each shard = ~1000 samples with pre-extracted frames + tokenized captions.

**Estimated storage:** 3M videos × ~10 frames avg × 224×224×3 bytes ≈ **~450 GB** (frames only, JPEG compressed much less).

---

## 4. Compute Budget

### Recommended Starting System: 2xA100 80GB

Start with 2xA100 for data prep, ablations, and optimization. Scale to 4x only if needed for final training.

| Phase | System | Wall-clock | Cost est. | Notes |
|-------|--------|-----------|-----------|-------|
| **CPU precompute** | 2xA100 (CPU cores) | ~24h | $33 | Frame extraction + tokenization (CPU-only) |
| **Speed optimization** | 2xA100 | ~8h | $11 | Profile, compile, benchmark — treat like a competition |
| **Ablations** | 2xA100 | ~16h | $22 | 6-10 runs at 135M |
| **Scaling grid** | 2xA100 | ~36h | $50 | 24 runs across 3 sizes → determines final model size |
| **Stage 1: Connector pretrain** | 2xA100 | ~24h | $33 | Winning config on 2M WebVid |
| **Stage 2: VL SFT** | 2xA100 | ~16h | $22 | 1M Cauldron + 14% text |
| **Stage 3: Video SFT** | 2xA100 | ~12h | $17 | 0.5M video mix |
| **Eval + report** | 2xA100 | ~12h | $17 | 3 eval modes + full benchmark suite |
| **Buffer** (25%) | — | ~37h | $51 | Reruns, debugging, optional DPO |
| **TOTAL** | | **~185h** | **~$256** | Well within $500 |

> **Why 2xA100 not 4x?** Most phases only need 1-2 GPUs (ablations, eval, profiling). 4x wastes 2 GPUs during single-GPU tasks. Upgrade to 4x only if Stage 1-3 training is the bottleneck.

### Speed Optimization Phase (IMPORTANT)

Allocate dedicated time to aggressively optimize throughput **before** running expensive training.

| Optimization | Expected Speedup | Effort |
|-------------|-----------------|--------|
| `torch.compile` on DINO + query attention | 1.3-1.5x | Medium |
| Pinned memory + async HtoD transfer | 1.1-1.2x | Low |
| Optimal batch size / grad accum sweep | 1.1-1.3x | Low |
| Variable-length packing (avoid padding waste) | 1.1-1.2x | Medium |
| DataLoader workers + prefetch tuning | 1.1-1.2x | Low |
| Profile with `torch.profiler` → fix hotspots | Varies | Medium |

**Target:** Measure samples/sec at start of optimization, then beat it by 2x+ before committing to long runs.

### Checkpoint Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Save frequency** | Decide after profiling (based on step time and disk speed) | Too aggressive = I/O bottleneck |
| **What to save** | Model state dict + optimizer state + data position | Full resume capability |
| **Keep** | Last 2 + best by composite metric | Don't fill disk |
| **Composite metric** | Weighted average of CIDEr + BLEU + Video-MME | Per-benchmark weights TBD from ablations |

### Efficiency Targets (nanochat-grade)

| Metric | Target | How |
|--------|--------|-----|
| GPU utilization | >80% | Precomputed frames, no I/O bottleneck |
| MFU | >40% | `torch.compile` on hot paths |
| Data loading | 0 GPU-wait | Sharded tar + pinned memory + prefetch workers |
| Mixed precision | bfloat16 | Always |
| DDP | 2-way (or 4-way if scaled up) | `torchrun --nproc_per_node=2` |

---

## 5. Engineering Prep (Local, Free)

Do all of this BEFORE renting GPUs.

| Task | Description | Depends On |
|------|-------------|------------|
| **E1. Single training script** | One `train.py` + YAML configs. Supports all 3 stages via config. Loss = text CE only | - |
| **E2. Sharded dataloader** | nanochat-style: tar shards of pre-extracted frames + tokenized captions. Variable frame count (1 FPS). Pinned buffers, prefetch workers | - |
| **E3. Eval harness** | CIDEr, BLEU-4, METEOR on fixed val set. Categorical + generative modes | - |
| **E4. FLOP counter** | `estimate_flops()` per sample for iso-FLOP analysis | E1 |
| **E5. Report generator** | Auto-generate markdown report card after each run | - |
| **E6. `speedrun.sh`** | End-to-end: CPU precompute → optimize → ablate → Stage 1-3 → eval → report | E1-E5 |
| **E7. HuggingFace packaging** | `AutoModel` loading, model card template, safetensors export | E1 |
| **E8. `torch.compile` integration** | Compile DINO forward + query attention + prediction head | E1 |
| **E9. Q-before-video dataloader mode** | Support question→video token ordering for QA eval | E2 |
| **E10. Autoregressive fine inference** | `forward_autoregressive()` — true sequential inference, no coarse pass | E1 |
| **E11. CPU data prep pipeline** | Download videos → ffmpeg frame extraction → resize → tokenize captions → shard to tar. All CPU, no GPU | - |
| **E12. Speed profiling harness** | `torch.profiler` integration, samples/sec benchmark, memory tracking | E1 |

---

## 6. Ablation Plan (Phase 1)

**Goal:** Establish best architecture config at small scale before spending compute on 1.7B.

All runs: SmolLM2-135M, 360K samples, 3M-sample LR schedule, same eval set.

### 6a. Architecture Ablations

One change at a time vs baseline (current best config).

| ID | Variable | Values | Hypothesis |
|----|----------|--------|------------|
| A1 | Deep query | on / off | Deep query enables selective attention |
| A2 | Multi-fine iterations | 1 / 2 / 3 | More refinement → better queries |
| A3 | Tokens per frame | 1 / 4 | More tokens help baseline, less help for foveated |
| A4 | DINO frozen vs finetuned | frozen / tuned | Frozen preserves feature diversity |
| A5 | Attention temperature | 1.0 / 0.5 / learnable | Sharper attention → bigger fine/coarse gap |
| A6 | Prediction head capacity | 2-layer / 4-layer | More capacity in decoder |
| A7 | Stage 1 backbone freezing | frozen / unfrozen (diff LR) | Is frozen connector bootstrap necessary at small scale? |

### 6b. Learning Rate Study

Three components, three learning rates. The **ratio** matters more than absolute values.

| Component | Why It's Different | Intuition |
|-----------|-------------------|-----------|
| **Connector** (query attn, projection) | Randomly initialized, must learn from scratch | Highest LR |
| **DINO encoder** | Pretrained, good features already | Lowest LR (don't destroy features) |
| **LLM** | Pretrained+Instruct, must adapt to vision tokens | Medium LR (adapt but don't forget) |

**Runs to test** (4 runs at 135M, ~2h total):

| ID | Connector LR | DINO LR | LLM LR | Hypothesis |
|----|-------------|---------|--------|------------|
| LR1 | 1e-4 | 1e-5 | 1e-5 | Conservative backbone, 10x ratio |
| LR2 | 1e-4 | 1e-6 | 1e-5 | Very conservative DINO, 100x ratio |
| LR3 | 5e-4 | 1e-5 | 5e-5 | Aggressive connector, 50x/10x ratio |
| LR4 | 1e-4 | 1e-5 | 1e-5 (uniform) | Baseline: is differential even needed? |

> **Intuition:** LR1 or LR2 likely wins. DINO features are good out of the box — a 100x ratio (LR2) may be right because we want DINO to barely move while the connector learns to use its features. The LLM needs moderate adaptation to accept visual tokens without forgetting language.

**What to skip** (use standard choices):
| Decision | Standard Choice | Why Skip |
|----------|----------------|----------|
| Schedule shape | Cosine decay | Universal default, rarely worth ablating at our scale |
| Warmup | 5-10% of total steps | Standard; only matters if diverging at start |
| Weight decay | 0.01-0.1 | Second-order effect vs LR ratios |
| Batch size vs LR scaling | Linear scaling rule | Well-established, not worth testing |

**What to watch for:**
- If LR4 (uniform) matches LR1-LR3 → differential LR doesn't matter, simplify the pipeline
- If DINO LR matters a lot → DINO features are fragile, consider freezing it entirely
- If LLM LR matters a lot → language forgetting is a real concern, may need more text retention

### 6c. Task Ablations


| ID | Training Mix | Hypothesis |
|----|-------------|------------|
| T1 | Captioning only | Baseline text prediction |
| T2 | Captioning + VQA | Multi-task helps generalization |
| T3 | Captioning + VQA + QA (question-first) | Task-aware foveation (see Section 8) |
| T4 | Full mix (33% video, 34% image, 20% text, 12% multi-image) | SmolVLM2 recipe |

### 6d. Ablation Methodology

| Rule | Implementation |
|------|---------------|
| **One variable at a time** | Each ablation changes exactly one thing from baseline |
| **Fixed sample count** | All runs see 360K samples (same schedule) |
| **Same eval** | Fixed 10K val set, same metrics |
| **Mandatory wandb** | Project: `foveated-vlm-ablations` |
| **Structured CSV** | One row per eval point, columns: run_id, step, samples_seen, loss_fine, loss_coarse, ratio, CIDEr, BLEU, attention_entropy |
| **Document everything** | KNOWLEDGE.md entry within 1 hour of run completion |

---

## 7. Scaling Law Grid (Phase 2)

### Design (modeled on nanochat's `scaling_laws.sh`)

| Axis | Values | Count |
|------|--------|-------|
| **LLM size** | 135M, 360M, 1.7B | 3 |
| **FLOP budget** | 4 levels (calibrated from ablation costs) | 4 |
| **Architecture** | foveated, baseline | 2 |

**Total: 3 x 4 x 2 = 24 runs** (but smallest ~minutes each, largest ~hours).

### How It Works

For each (LLM_size, FLOP_budget, architecture) triplet:
1. Compute `flops_per_sample` via `estimate_flops()`
2. Compute `num_samples = target_flops / flops_per_sample`
3. Train for exactly that many samples
4. Evaluate on fixed val set
5. Log to CSV: `flop_budget, llm_size, arch, params, samples_trained, val_loss, CIDEr, ratio`

### Axes for Plots

**Plot 1 — Iso-FLOP Curves (main scaling law):**
- X: model params (scaling params, Kaplan-style)
- Y: val captioning loss
- Curves: one per FLOP budget
- Shows: compute-optimal model size

**Plot 2 — Foveated vs Baseline Crossover:**
- X: total compute (FLOPs)
- Y: `loss_fine / loss_coarse` ratio
- Curves: foveated vs baseline at each model size
- Shows: **foveated advantage grows with scale** (the thesis)

**Plot 3 — Token Efficiency Frontier:**
- X: visual tokens per frame (1, 4, 16)
- Y: CIDEr at fixed FLOP budget
- Curves: one per LLM size
- Shows: 1 foveated token ≈ N baseline tokens

**Plot 4 — Data Scaling:**
- X: training samples seen
- Y: eval metrics (from intermediate checkpoints)
- Curves: foveated vs baseline
- Shows: sample efficiency / crossover point

**Plot 5 — Optimal Allocation:**
- X: total compute C
- Y: optimal N (params) and D (data) at iso-FLOP minimum
- Shows: Chinchilla-style N ∝ C^a, D ∝ C^b exponents for video VLMs

---

## 8. Question-Before-Video Experiment

### The Argument

Foveated attention = "look at what matters." If the model knows the question FIRST, it can generate task-aware queries.

| Ordering | Coarse (static) | Fine (foveated) |
|----------|-----------------|-----------------|
| Video → Question | task-agnostic | temporally-aware only |
| **Question → Video** | **still task-agnostic** | **temporally-aware AND task-aware** |

### Implementation

- **Training:** Mix both orderings 50/50
- **Eval:** Test both orderings separately on same QA benchmark

### Expected Result (2x2 table for paper)

| | Video→Q | Q→Video | Delta |
|---|---------|---------|-------|
| **Coarse** | X | ~X | ~0 (static query ignores question) |
| **Fine** | Y | Y+ | **positive** (queries use question) |

If delta > 0 for fine but ~0 for coarse, that's direct evidence the query mechanism leverages task context. Clean, publishable result.

### Data Source for QA

- Use existing video-QA datasets (ActivityNet-QA, MSRVTT-QA) for eval
- Or generate QA pairs from WebVid captions (cheaper)

---

## 9. Evaluation Suite

### Metrics

| Metric | Type | What It Measures |
|--------|------|-----------------|
| **CIDEr** | Captioning | Primary quality metric |
| **BLEU-4** | Captioning | N-gram overlap |
| **METEOR** | Captioning | Semantic similarity |
| **Video-MME** | Video understanding | Temporal reasoning benchmark |
| **MLVU** | Video understanding | Multi-task video language understanding |
| **loss_fine / loss_coarse** | Internal | Foveation benefit (should be < 1.0) |
| **Attention entropy** | Diagnostic | Selectivity (lower = more focused) |

### Eval Principles (from SmolLM Playbook)

| Principle | Our Implementation |
|-----------|--------------------|
| **Monotonic** | Track all metrics over training — should improve steadily |
| **Low noise** | Fixed val set, deterministic eval, report std across 3 seeds |
| **Above random** | CIDEr > 0 early in training (not just at end) |
| **Ranking consistent** | If config A > B at 100K samples, A > B at 3M samples |

### Comparison Baselines

| Baseline | Source | Why |
|----------|--------|-----|
| **SmolVLM2-256M** | HuggingFace (pretrained) | Direct comparison at similar param count |
| **SmolVLM2-2.2B** | HuggingFace (pretrained) | Upper bound (what full multi-token VLM achieves) |
| **Our baseline (16 tok/frame)** | Trained alongside | Same data, same compute, different architecture |
| **Random query control** | Trained alongside | Proves queries matter (not just architecture) |

### Three Evaluation Modes (CRITICAL)

The foveated architecture has a **train/inference mismatch** that MUST be measured explicitly.

| Mode | Query Source | Execution | When Used |
|------|-------------|-----------|-----------|
| **Coarse-only** | Static `q_static` for all frames | Parallel (all frames at once) | Baseline lower bound |
| **Coarse→Fine** | Coarse pass → LLM → queries → fine pass | Parallel (training approximation) | Training eval |
| **Autoregressive Fine** | `q_init` → frame₁ → LLM → `q_1` → frame₂ → LLM → `q_2` → ... | **Sequential** (true inference) | **Deployment mode** |

```
Coarse-only:        q_static → [F1, F2, F3, F4] → z_coarse (parallel)

Coarse→Fine:        q_static → [F1, F2, F3, F4] → z_coarse (parallel)
                    z_coarse → LLM → [q1, q2, q3, q4] (parallel)
                    [q1→F1, q2→F2, q3→F3, q4→F4] → z_fine (parallel)

Autoregressive:     q_init → F1 → z1 → LLM → q1
                    q1 → F2 → z2 → LLM → q2
                    q2 → F3 → z3 → LLM → q3    (sequential!)
                    q3 → F4 → z4 → LLM → q4
```

### What to Measure Across All Three Modes

| Metric | Coarse-only | Coarse→Fine | Autoregressive Fine |
|--------|-------------|-------------|---------------------|
| **Captioning loss** | X₁ | X₂ | X₃ |
| **CIDEr** | C₁ | C₂ | C₃ |
| **Attention entropy** | E₁ | E₂ | E₃ |
| **Inference time/sample** | T₁ (fast) | T₂ (medium) | T₃ (slowest) |

**Key questions this answers:**

| Question | How to Read |
|----------|-------------|
| Does foveation help at all? | X₂ < X₁ (coarse→fine beats coarse-only) |
| Does true autoregressive help more? | X₃ < X₂ (AR beats training approximation) |
| How big is the train/inference gap? | \|X₃ - X₂\| (should be small if training works) |
| Is AR worth the latency cost? | (X₂ - X₃) vs (T₃ - T₂) — quality gain per latency cost |
| Does multi-fine bridge the gap? | Train with coarse→fine₁→fine₂, measure if X₃ improves |

### Engineering Requirement

Need to implement `forward_autoregressive()` in the model:

```python
def forward_autoregressive(self, frames, text_embeds=None):
    """True inference mode: no coarse pass, sequential fine queries."""
    q = self.q_init  # Starting query
    fine_features = []
    for t in range(num_frames):
        z_t = self.encoder(frames[:, t], query=q)       # One frame
        fine_features.append(z_t)
        h_t = self.llm(torch.stack(fine_features, dim=1)) # LLM on all so far
        q = self.query_head(h_t[:, -1])                   # Next query from latest hidden
    # Final prediction from full sequence
    return self.predict(fine_features, text_embeds)
```

This goes into **E10** in the engineering prep list.

---

## 10. Deliverables & Timeline

### Phase 0: Local Prep (free, before renting GPUs)

| Deliverable | Status |
|-------------|--------|
| Single `train.py` + YAML configs (all 3 stages, text CE loss) | TODO |
| CPU data prep pipeline (E11: download → ffmpeg → resize → tokenize → shard) | TODO |
| Sharded dataloader with variable frame count (1 FPS) | TODO |
| Eval harness (CIDEr, BLEU, METEOR, Video-MME) | TODO |
| Autoregressive fine inference mode (E10) | TODO |
| FLOP counter | TODO |
| Speed profiling harness (E12) | TODO |
| `speedrun.sh` (full pipeline) | TODO |
| HuggingFace model packaging | TODO |
| Q-before-video dataloader mode | TODO |

### Phase 1: Speed Optimization (~$11, ~8 GPU-hours)

| Deliverable |
|-------------|
| `torch.compile` on DINO + query attention |
| Optimal batch size / grad accum / workers sweep |
| Profiler results identifying hotspots |
| Benchmark: samples/sec before and after |
| **Goal: 2x throughput improvement over naive baseline** |

### Phase 2: Ablations (~$22, ~16 GPU-hours)

| Deliverable |
|-------------|
| 6-10 ablation runs at SmolLM2-135M |
| Best config identified |
| Ablation table for blog/paper |
| KNOWLEDGE.md entries for all runs |

### Phase 3: Scaling Grid (~$50, ~36 GPU-hours)

| Deliverable |
|-------------|
| 24 scaling law runs across 3 model sizes |
| 5 publication-quality plots |
| **Determines final model size** (not pre-committed to 1.7B) |
| Chinchilla-style optimal allocation analysis |
| Foveated vs baseline crossover evidence |

### Phase 4: Final Model — 3-Stage Training (~$72, ~52 GPU-hours)

| Stage | Data | Trainable | Monitor |
|-------|------|-----------|---------|
| Stage 1: WebVid captioning | 2M WebVid + 14% text | All params (diff LR for backbone) | Loss convergence |
| Stage 2: VL SFT | 1M Cauldron + 14% text | All params (full weights) | Grad norms, loss spikes, NaN |
| Stage 3: Video SFT | 0.5M video mix | All params | Same + per-benchmark eval |
| Checkpoint selection | — | Best by weighted composite metric | — |

> **Divergence fallback:** If training diverges, freeze DINO first. If still unstable, freeze bottom LLM layers. Only use DoRA/LoRA as last resort.

### Phase 5: Evaluation (~$17, ~12 GPU-hours)

| Deliverable |
|-------------|
| **3-mode eval**: coarse-only vs coarse→fine vs autoregressive fine |
| **Timing comparison**: inference latency per mode |
| **Q-before-video**: 2x2 ablation (ordering × architecture) |
| Full benchmark suite (CIDEr, BLEU, METEOR, Video-MME, MLVU) |
| Comparison table vs SmolVLM2 at similar param count |

### Phase 6: Release (~$17, ~12 GPU-hours)

| Deliverable |
|-------------|
| HuggingFace model card + safetensors |
| Inference scripts + demo notebook |
| Blog post with scaling story |
| Auto-generated report card |
| Optional: DPO via RLAIF-V (LoRA, 83K pairs) |

---

## 11. Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Loss function | **Text CE only, no reconstruction** | This is a VLM (understanding), not generation. Matches SmolVLM2 |
| LLM family | SmolLM2 only (3 sizes) | Consistency; interpolate for scaling curves |
| Final model size | **TBD by scaling law study** | Not pre-committed; data-driven decision |
| LLM variant | Instruct for final model | BUT must retain 14% text data to preserve instruct capability |
| GPU system | **Start 2xA100**, scale to 4x if needed | Most phases need 1-2 GPUs; don't waste $$$ |
| Frame sampling | **1 FPS, variable frame count** | Industry standard; videos get natural number of frames |
| Training stages | 3-stage (Idefics3/SmolVLM2 pattern) | Stage 1: WebVid captioning → Stage 2: VL SFT → Stage 3: Video SFT. All full unfreeze (ablate frozen vs not) |
| Backbone training | **Full weight fine-tuning** (not DoRA) | Smaller models less fragile; foveated connector needs full gradients; monitor for divergence |
| Caption formatting | **Honest framing** ("WebVid caption for this video?") | Don't pretend noisy captions are descriptions; Stage 2-3 teaches real conversational style |
| Data scale | ~3.5M total across stages | Matches SmolVLM2 scale |
| Data mix | 14% text retention always | SmolVLM2 showed removing text hurts 3.7-6.5% |
| Precompute | **CPU-only** (frames + tokenization) | GPU precompute (DINO features) = too much storage + can't reuse when DoRA changes encoder |
| Speed optimization | **Dedicated phase before long runs** | Aggressive profiling + compile, target 2x throughput gain |
| Checkpoint saving | **TBD after profiling** | Decide frequency based on actual step time and disk speed |
| Eval modes | 3-way (coarse / coarse→fine / AR fine) | Must quantify train/inference gap explicitly |
| Eval baseline | SmolVLM2 (pretrained) | Free comparison; no need to train our own baseline VLM |
| Q-before-video | Train mixed, eval both | Clean 2x2 ablation; isolates task-aware foveation |

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1.7B OOMs on 4xA100 | Low | High | Gradient checkpointing; 80GB VRAM is generous |
| Foveated advantage doesn't grow with scale | Medium | High | This is the thesis — if it fails, paper pivots to efficiency story |
| 3M WebVid download too slow | Medium | Medium | Start download immediately; use streaming fallback |
| Eval suite too noisy for ablations | Low | Medium | Fixed val set + 3 seeds; validate ranking consistency first |
| Precompute takes too long | Low | Medium | 4-GPU parallel; ~11h is manageable |
| Budget overrun | Low | Medium | 20% buffer built in; can skip extended scaling grid |
| AR fine much worse than coarse→fine | Medium | Medium | Multi-fine bridging (coarse→fine₁→fine₂); measure gap early at 135M |
| AR fine too slow for practical use | Low | Medium | KV caching in LLM; only encoder is sequential, not LLM |
| The Cauldron / LLaVA-Video data access | Low | Medium | All on HuggingFace, Apache-2.0 licensed |
| Stage 2/3 catastrophic forgetting | Medium | Medium | Keep 14% text; eval between stages; lower LR for later stages |
| Full unfreeze divergence | Medium | Medium | Log grad norms + loss every step; freeze DINO first if spikes; DoRA as last resort |

---

## 13. What Changed from Previous Plans

| Before (toy-scale) | Now (release-scale) |
|--------------------|---------------------|
| 5K samples on local 4090 | ~3.5M samples across 3 stages on 2-4xA100 |
| **SD-VAE reconstruction loss** | **Text CE loss only** (understanding, not generation) |
| **Pre-committed to 1.7B** | **Model size TBD by scaling law study** |
| Fixed 8 or 16 frames per video | **1 FPS, variable frame count** (5-50 frames per video) |
| Single-stage training (reconstruction only) | 3-stage: connector pretrain → VL SFT (DoRA) → Video SFT (DoRA) |
| "Free" instruct from backbone | **14% text retention in all stages** to preserve instruct capability |
| Raw captions as training data | **Instruction-formatted** captions with varied prompts + system prompt |
| DoRA on backbones (Idefics3 pattern) | **Full weight fine-tuning** — smaller models, novel arch, continuous monitoring |
| GPU precompute (VAE latents, DINO KV) | **CPU-only precompute** (frames + tokenization). DINO runs during training |
| No speed optimization phase | **Dedicated optimization sprint** before long runs (target 2x throughput) |
| Aggressive checkpoint saving | **TBD after profiling** (based on actual step time + disk speed) |
| 10 training scripts | 1 script + YAML configs |
| Only eval coarse→fine loss | **3 eval modes**: coarse-only, coarse→fine, autoregressive fine |
| Ad-hoc eval (loss ratio only) | Full suite: CIDEr, BLEU, METEOR, Video-MME, MLVU + loss ratio |
| Optional wandb | Mandatory wandb for every run |
| 43% GPU utilization | Target >80% via compile + DDP + precompute |
| No reproducibility | `speedrun.sh` end-to-end |
| No release plan | HuggingFace model + blog + scripts |

---

*Last updated: 2026-02-09 (v3)*
