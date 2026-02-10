# Research Playbook: fVLM Scaling & Presentation Guide

Distilled from nanochat (Karpathy), SmolLM Training Playbook (HuggingFace), and SmolVLM.
These are the patterns we follow for experiments, scaling, and writing up results.

---

## 1. How to Do Good Research

### 1.1 The Training Compass: Why -> What -> How

Before running any experiment, answer three questions in order:

1. **Why** are we running this experiment? What hypothesis does it test?
2. **What** specifically are we changing? (One variable at a time)
3. **How** will we measure success? (Define metrics before running)

> "Many failed training projects didn't fail because of bad hyperparameters or buggy code;
> they failed because someone decided to train a model they didn't need." -- SmolLM Playbook

**For fVLM**: Our "why" is clear -- validate that foveated attention (dynamic queries informed by temporal context) outperforms static attention for video understanding. Every experiment should connect back to this thesis.

### 1.2 Ablation Methodology

**The discipline of derisking: Never change anything unless you've tested that it helps.**

Core rules (from SmolLM Playbook "Rules of Engagement"):

| Rule | Why It Matters |
|------|---------------|
| **Change one thing at a time** | If you change multiple things and performance improves, you won't know what caused it |
| **Test every change, no matter how small** | "Seemingly innocent" changes can introduce subtle bugs that contaminate results |
| **Train on enough tokens/steps** | Cutting corners leads to noisy results and bad decisions |
| **Validate your evaluation suite first** | Evals guide every decision; getting them wrong poisons everything downstream |
| **Be paranoid** | The golden principle: once you have a good setup, no change should go untested |

**Practical ablation workflow:**
1. Start with a proven baseline (our current best config)
2. Test promising changes one at a time against baseline
3. When something works, integrate it as the new baseline
4. Test next change against new baseline
5. Document everything -- positive AND negative results

**Cost awareness** (from SmolLM Playbook):
- SmolLM3 spent 161,280 GPU hours on ablations and debugging vs. 276,480 on the main run
- Ablations = ~58% of the main training cost
- Plan for: training cost + ablations + buffer for surprises
- For us on a single 4090: budget ~2x the main run time for ablations

### 1.3 Evaluation Design

Choose evaluation metrics that satisfy these four principles (from SmolLM/FineWeb):

| Principle | Definition | Why It Matters |
|-----------|-----------|---------------|
| **Monotonicity** | Scores consistently improve as training progresses | If scores jump around, you can't tell if a change helped |
| **Low noise** | Same setup with different seeds gives similar scores | High variance = chasing noise |
| **Above-random performance** | Shows signal early in training | Tasks at random baseline waste ablation compute |
| **Ranking consistency** | If A > B early on, A > B stays true later | Otherwise early ablations don't predict final performance |

**For fVLM evaluation metrics:**
- **Primary**: `loss_fine / loss_coarse` ratio (should be < 1.0, i.e. fine beats coarse)
- **Secondary**: Absolute reconstruction loss (MSE on VAE latents)
- **Diagnostic**: Attention entropy (lower = more selective = good)
- **Downstream**: Captioning quality (CIDEr, BLEU) when applicable
- Track all of these over training steps -- they should be monotonic

### 1.4 Scaling Law Experiments

**nanochat's approach** -- the gold standard for small-scale scaling studies:

1. **Single control variable**: nanochat uses `--depth` to derive everything (width, LR, heads, batch size, weight decay, training horizon)
2. **Grid sweep**: 4 FLOP budgets x 7 model depths = 28 runs
3. **Chinchilla-optimal ratio**: For each model size, train on `N * ratio` tokens where ratio is the data:params optimal point
4. **Power Laws batch size scaling**: `B_opt proportional to D^0.383` -- batch size should scale with dataset size
5. **muP LR transfer**: Learning rate transfers across depths via `LR proportional to 1/sqrt(dim)`

**What this means for fVLM scaling:**
- Identify our "single dial" -- likely `num_frames` or `model_capacity` (encoder depth + LLM size)
- Run a grid: vary model size x training compute budget
- For each point, measure: loss ratio, reconstruction quality, attention selectivity
- Plot iso-performance curves to find optimal model size for a given compute budget
- Track wall-clock time, not just loss -- a faster model at slightly worse loss may be better

### 1.5 Document Negative Results

From nanochat's `dev/LOG.md` -- every experiment gets logged, including failures:

```
| Date | Experiment | Result | Impact |
| 2/5  | SwiGLU Activation | NEGATIVE | Worse than ReLU^2 on all metrics |
| 1/15 | Olmo Pretraining Mix | NEGATIVE | CORE 15.5 -> 13.8 |
| 1/13 | Varlen Attention | NEGATIVE | No measurable improvement |
| 1/12 | Multi-Token Prediction | NEGATIVE | +13GB memory, worse wall-clock |
```

**Negative results are as valuable as positive ones.** They prevent others (including future-you) from wasting time. Always log: hypothesis, config, result, conclusion (VALIDATED/FAILED).

---

## 2. How to Do Good Presentation

### 2.1 The Report Card Pattern

**nanochat's auto-generated report** is the model here. After every pipeline stage, metrics are collected into a structured markdown document:

```
| Metric        | BASE  | SFT   | RL    |
|---------------|-------|-------|-------|
| CORE          | 0.22  | -     | -     |
| ARC-Easy      | -     | 0.39  | -     |
| GSM8K         | -     | 0.05  | 0.08  |
| ChatCORE      | -     | 0.09  | -     |
| Wall clock    | 3h    | +8min | +1.5h |
```

**Key elements of a good report card:**
1. **Environment info**: Hardware, software versions, git commit
2. **Configuration**: All hyperparameters that define the run
3. **Metrics at each stage**: Progression visible across pipeline stages
4. **Cost tracking**: Wall-clock time, GPU hours, estimated dollar cost
5. **Bloat metric**: Lines of code, dependencies -- keeps the project honest
6. **Sample outputs**: Qualitative examples showing what the model can do

### 2.2 Blog/Paper Presentation Patterns

**From SmolLM Playbook** (a masterclass in research communication):

1. **Start with "Why"**: Don't lead with architecture details. Lead with the problem and why it matters.
2. **Training Compass framing**: Guide the reader through Why -> What -> How
3. **Show the messy reality**: "This is not an ordinary guide, but rather the untangling of a spiderweb of decisions, discoveries, and dead ends"
4. **Interactive elements**: SmolLM includes interactive parameter calculators, browsable benchmark examples, clickable ablation plots
5. **Comparison tables**: Always compare against established baselines with clear metrics
6. **Honest about failures**: "We restarted a training run after 1T tokens" -- admitting mistakes builds credibility

**From SmolVLM blog:**

1. **TLDR first**: One paragraph that says everything essential
2. **Capability showcase**: Show concrete input/output examples before architecture details
3. **Memory story**: For efficiency-focused models, the memory/throughput comparison IS the story
4. **Benchmark table**: Standardized comparison with clear wins highlighted
5. **Checkpoint selection**: Explain HOW you picked the final model, not just what it scored
6. **Reproducibility**: Fine-tuning scripts, notebooks, CLI commands -- let people use your work

**From nanochat announcement:**

1. **Cost framing**: "$100 ChatGPT" -- immediately tells you what to expect
2. **Walk-through narrative**: Step-by-step guide that readers can follow along
3. **Progressive complexity**: Start with tokenizer, then pretrain, then SFT, then RL
4. **Show actual outputs**: Model completions on concrete prompts
5. **"Your turn" section**: Explicitly invite experimentation with clear knobs to turn
6. **Community results**: Show what other people achieved with your work

### 2.3 What Our fVLM Blog Should Include

Based on these patterns, our writeup should have:

1. **Hook**: "Can a VLM understand video with just ONE token per frame?"
2. **The thesis**: Foveated attention -- biological inspiration, architectural bet
3. **The journey**: How we went from idea -> bugs -> validation (including the 3 critical bugs)
4. **Scaling story**: How performance changes with model size, frames, compute
5. **Ablation table**: What worked and what didn't (deep queries, captioning vs reconstruction, etc.)
6. **Visualization showcase**: Attention heatmaps showing foveated behavior on actual video
7. **Comparison table**: vs. standard multi-patch VLMs on efficiency metrics
8. **Limitations & future work**: Honest about train/inference gap, what remains to be done
9. **Reproducibility**: All code, configs, and checkpoints

### 2.4 Visualization Best Practices

**For fVLM specifically:**
- **Attention heatmaps**: Show where the model "looks" (coarse vs fine) on video frames
- **Loss curves**: `loss_fine` vs `loss_coarse` over training (the core thesis validation)
- **Ratio plots**: `loss_fine / loss_coarse` across different configs (captioning > reconstruction)
- **Scaling plots**: Loss vs compute (iso-FLOP curves for different model sizes)
- **Prediction quality**: Decoded VAE latent predictions vs ground truth frames
- **Temporal tracking**: Show how dynamic queries follow objects across frames

---

## 3. How to Do Good Engineering for Scaled Runs

### 3.1 The Speedrun Pattern

**nanochat's `speedrun.sh`** is the template: one script that runs the entire pipeline end-to-end.

```bash
#!/bin/bash
# 1. Setup environment
# 2. Download data
# 3. Train tokenizer (if applicable)
# 4. Pretrain base model
# 5. Finetune (SFT)
# 6. Evaluate
# 7. Generate report
```

**Key engineering patterns:**
- Background data download while training tokenizer (overlap I/O with compute)
- `torchrun` for distributed training with automatic gradient accumulation
- `torch.compile` for fused kernels (nanochat gets ~48% MFU)
- Pre-allocated pinned CPU buffers with single HtoD transfer (dataloader)
- `@torch.compile(dynamic=False, fullgraph=True)` on optimizer steps for zero Python overhead
- Resume support via state dicts tracking exact data position

### 3.2 Single-Dial Scaling

**The most powerful engineering pattern from nanochat**: derive everything from one number.

nanochat's `--depth` parameter controls:
- `model_dim = depth * 64`
- `n_heads = depth` (with head_dim=64 fixed)
- Number of KV heads (GQA ratio)
- Sliding window attention pattern
- Learning rate (scaled as `1/sqrt(dim)`)
- Batch size (Power Laws: `B_opt proportional to D^0.383`)
- Weight decay
- Training horizon (Chinchilla optimal `tokens = params * ratio`)

**For fVLM, our "dials" should be:**
- **Primary**: Model capacity tier (e.g., "small"=current, "medium"=2x encoder, "large"=bigger LLM)
- **Secondary**: `num_frames` (4, 8, 16, 32)
- Each tier auto-determines: batch size, learning rate, training steps, memory budget

### 3.3 Memory Management

**Critical for single-GPU work (our RTX 4090 constraint).**

From our CLAUDE.md + nanochat patterns:

```
OOM Mitigation Hierarchy (apply in order):
1. Reduce batch size (batch_size=1, grad_accum=8)
2. Reduce sequence length (num_frames=4)
3. Gradient checkpointing
4. Mixed precision (bfloat16 always)
5. Precompute everything possible (VAE latents, DINO features)
```

**Monitoring** (from nanochat):
- Log `torch.cuda.max_memory_allocated()` every N steps
- Track MFU (model FLOPS utilization) -- if < 30%, something is wrong
- Track tokens/second and compare across configs
- If memory creeps up over time, you have a leak

### 3.4 Checkpoint & Evaluation Strategy

**From SmolVLM** -- checkpoint selection is non-trivial:
- Save checkpoints frequently (every N steps)
- Evaluate multiple metrics at each checkpoint
- Create a composite score with weighted metrics
- The best checkpoint is NOT always the last one

**From nanochat** -- evaluation during training:
- CORE metric evaluated periodically during pretraining
- ChatCORE evaluated after SFT
- Separate generative and categorical evaluation loops
- DDP-distributed evaluation for speed

**For fVLM:**
- Save every 500 steps (or every epoch for short runs)
- Evaluate: loss_ratio, reconstruction_MSE, attention_entropy
- Track best checkpoint by loss_ratio (primary metric)
- Keep last 3 checkpoints + best checkpoint

### 3.5 wandb Logging

**Every experiment needs structured logging.** From nanochat + SmolLM:

```python
wandb.log({
    "step": step,
    "loss_fine": loss_fine,
    "loss_coarse": loss_coarse,
    "loss_ratio": loss_fine / loss_coarse,
    "attention_entropy_fine": entropy_fine,
    "attention_entropy_coarse": entropy_coarse,
    "learning_rate": lr,
    "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
    "tokens_per_second": tokens_per_sec,
    "wall_time_minutes": elapsed / 60,
})
```

**Project naming**: `foveated-vlm-{experiment-type}` (e.g., `foveated-vlm-scaling`, `foveated-vlm-captioning`)

### 3.6 Fused & Compiled Optimization

From nanochat's `optim.py` -- the performance pattern:

1. Use `@torch.compile(dynamic=False, fullgraph=True)` on hot paths
2. Use 0-D CPU tensors for hyperparameters to avoid recompilation
3. Fuse multiple operations into single kernels (momentum + orthogonalization + update)
4. Stack parameters of the same shape for batched operations
5. Use `torch._foreach_copy_` for efficient multi-tensor updates

**For fVLM**: Consider compiling the encoder forward pass and the prediction head. These run every step and benefit most from fusion.

### 3.7 Data Pipeline Efficiency

From nanochat's `dataloader.py`:

1. **Best-fit packing**: Maximize GPU utilization by packing variable-length sequences
2. **Pre-allocated buffers**: `torch.empty` once, reuse forever
3. **Pinned memory**: CPU buffers with `pin_memory=True` for fast HtoD transfer
4. **Single HtoD copy**: Stage everything in CPU buffer, then one `gpu_buffer.copy_(cpu_buffer, non_blocking=True)`
5. **Background data loading**: Overlap I/O with compute via separate processes

**For fVLM**: Precomputed VAE latents are already our biggest efficiency win. Next: ensure DataLoader workers overlap with GPU training steps.

---

## 4. Scaling Experiment Plan for fVLM

Based on all the above patterns, here's what a systematic scaling study looks like:

### Phase 1: Establish Baselines
- Fix best known config from existing experiments
- Run 3 seeds to establish variance bounds
- Define composite metric: `M = w1*loss_ratio + w2*captioning_score + w3*attention_selectivity`

### Phase 2: Architecture Ablations
One change at a time against baseline:
- [ ] Encoder depth (DINO layers used)
- [ ] LLM size (SmolLM2-135M vs 360M)
- [ ] Number of frames (4, 8, 16)
- [ ] Deep query vs shallow query
- [ ] Multi-fine iterations (1, 2, 3 fine passes)
- [ ] Prediction head capacity

### Phase 3: Scaling Laws
Grid sweep:
- **Model sizes**: 3-4 configs (vary encoder + LLM jointly)
- **Compute budgets**: 3-4 levels (vary training steps)
- **For each point**: measure loss_ratio, reconstruction quality, wall-clock time
- **Output**: Iso-performance curves, optimal model size per compute budget

### Phase 4: Best Model Training
- Train the winning config for full duration
- Frequent checkpointing + evaluation
- Track all metrics for the report card

### Phase 5: Report Card & Blog
- Auto-generate metrics table across all pipeline stages
- Produce visualizations (attention, predictions, loss curves)
- Write up following the patterns in Section 2

---

## Quick Reference: Patterns by Source

| Pattern | Source | Key Insight |
|---------|--------|-------------|
| Single-dial scaling | nanochat | Derive everything from one control variable |
| Training Compass | SmolLM | Why -> What -> How before every experiment |
| Change one thing | SmolLM | Never modify two variables simultaneously |
| 4 eval principles | SmolLM/FineWeb | Monotonicity, low noise, above-random, ranking consistency |
| Speedrun script | nanochat | End-to-end pipeline in one script |
| Report card | nanochat | Auto-generated metrics summary at each stage |
| Cost tracking | nanochat + SmolLM | Wall-clock time, GPU hours, dollar cost |
| Negative results | nanochat LOG.md | Document what DIDN'T work, not just what did |
| Checkpoint selection | SmolVLM | Best checkpoint != last checkpoint |
| Memory efficiency story | SmolVLM | For efficiency models, memory IS the story |
| Compiled optimizer | nanochat | `@torch.compile` on hot paths, 0-D CPU tensor trick |
| Best-fit packing | nanochat | Maximize utilization with variable-length data |
| Ablation cost budget | SmolLM | Plan for ablations = ~50-60% of main run cost |
| TLDR first | SmolVLM | Lead with one-paragraph summary |
| Progressive narrative | nanochat | Walk through pipeline step by step |
| Honest about failures | SmolLM | Admitting mistakes builds credibility |

---

*Distilled: 2026-02-09*
*Sources: nanochat (Karpathy), SmolLM Training Playbook (HuggingFace), SmolVLM (HuggingFace)*
