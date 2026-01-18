# Scaling Recommendations for Scientific Validity

**Date:** 2026-01-18
**Author:** Profiling Analysis
**Status:** DRAFT - For discussion

---

## Executive Summary

Current experiments lack scientific rigor due to:
1. **Inconsistent training configurations** (steps, batches, durations)
2. **Data loading bottleneck** (43% GPU utilization)
3. **No controlled comparisons** (multiple variables changed between experiments)

This document provides recommendations for running scientifically valid experiments at scale.

---

## Current Bottleneck Analysis

### Profiling Results (batch_size=8, num_frames=8)

| Stage | Time/Sample | % of Total |
|-------|-------------|------------|
| Network download | 0.115s | 20% |
| ffmpeg extraction | 0.205s | 36% |
| VAE encoding | 0.105s | 19% |
| Model forward | 0.073s | 13% |
| Model backward | 0.065s | 12% |
| **TOTAL** | **0.563s** | 100% |

### Key Findings

1. **Data loading is the bottleneck**
   - Serial data loading: 0.32s/sample (download + extract)
   - GPU compute: 0.24s/sample (VAE + forward + backward)
   - GPU waits for data ~57% of the time

2. **Memory underutilization**
   - Peak usage: 14 GB
   - Available: 24 GB
   - Headroom: 10 GB (41% unused!)

3. **Current throughput**
   - ~6400 samples/hour
   - ~800 steps/hour (batch_size=8)

### Optimal Throughput (Theoretical)

With fully parallel data loading:
- ~10,500 samples/hour (1.6x speedup)
- ~1,300 steps/hour

---

## Recommendation 1: Fix Data Loading

### Option A: Parallel Download Threads (Easy)

```python
# Current: 1 thread
"num_prefetch_threads": 1

# Better: 4+ threads
"num_prefetch_threads": 4
"prefetch_size": 64  # Larger buffer
```

**Expected improvement:** ~1.5-2x throughput

### Option B: Precompute Dataset Locally (Best)

1. Download videos once
2. Extract frames to disk
3. Precompute VAE latents
4. Load from disk during training

**Benefits:**
- No network variability
- Reproducible experiments
- ~3x faster data loading (disk vs network)

**Storage estimate:**
- 100K videos × 8 frames × 256×256 × 3 bytes ≈ 150 GB (frames)
- 100K videos × 8 frames × 4×32×32 × 2 bytes ≈ 13 GB (VAE latents)

### Option C: Use WebDataset (Production)

```python
# Package data into shards
webdataset.ShardWriter("data/shard-%05d.tar")

# Load with parallel workers
wds.WebDataset(shards).decode().shuffle(1000)
```

**Benefits:**
- Scales to any size
- Parallel I/O built-in
- Used by LAION, OpenAI, etc.

---

## Recommendation 2: Standardize Training Config

### Fixed Configuration for All Experiments

```python
STANDARD_CONFIG = {
    # Fixed across ALL experiments
    "total_samples": 500_000,  # Train for exactly this many samples
    "batch_size": 8,
    "grad_accum": 4,  # Effective batch = 32
    "num_frames": 8,
    "learning_rate": 3e-5,
    "warmup_samples": 10_000,

    # Variable (one at a time)
    "architecture_variant": "...",  # The thing being tested

    # Logging
    "eval_every_samples": 10_000,
    "checkpoint_every_samples": 50_000,
}
```

### Comparison Table Template

| Experiment | Architecture | Samples Seen | Cap Ratio | Rec Ratio |
|------------|--------------|--------------|-----------|-----------|
| baseline | single-fine | 500K | X.XX ± 0.XX | X.XX ± 0.XX |
| multi-fine | coarse→fine₁→fine₂ | 500K | X.XX ± 0.XX | X.XX ± 0.XX |
| joint | cap+rec loss | 500K | X.XX ± 0.XX | X.XX ± 0.XX |

**All experiments must have identical:**
- Total samples seen
- Batch size
- Learning rate schedule
- Evaluation points

---

## Recommendation 3: Proper Ablation Study

### Phase 1: Baseline Establishment (1 week)

1. **Baseline A:** Coarse-only (q_static, no fine pass)
2. **Baseline B:** Single-fine (current architecture)
3. **Baseline C:** Random-query fine (control for query learning)

Each trained for 500K samples, 3 seeds each.

### Phase 2: Architecture Ablations (2 weeks)

Change ONE thing at a time:

| Ablation | Change | Hypothesis |
|----------|--------|------------|
| A1 | Multi-fine (2 iterations) | More refinement helps |
| A2 | Multi-fine (3 iterations) | Diminishing returns? |
| A3 | Deeper LLM queries | Better query diversity |
| A4 | Multiple queries (4-way) | Better coverage |
| A5 | Larger batch size (16) | Smoother gradients |

### Phase 3: Task Ablations (1 week)

| Task Mix | Cap Loss | Rec Loss | Hypothesis |
|----------|----------|----------|------------|
| Recon only | 0.0 | 1.0 | Baseline |
| Caption only | 1.0 | 0.0 | Semantic focus |
| Joint 50/50 | 0.5 | 0.5 | Balance |
| Joint 80/20 | 0.8 | 0.2 | Caption-heavy |

---

## Recommendation 4: Metrics & Logging

### Per-Step Metrics

```python
log_dict = {
    # Primary metrics
    "loss/caption_fine": ...,
    "loss/caption_coarse": ...,
    "loss/recon_fine": ...,
    "loss/recon_coarse": ...,
    "ratio/caption": coarse / fine,
    "ratio/recon": coarse / fine,

    # Samples seen (NOT steps!)
    "samples_seen": step * batch_size * grad_accum,

    # Throughput
    "samples_per_second": ...,
    "gpu_utilization": ...,

    # Model internals
    "attention/entropy_coarse": ...,
    "attention/entropy_fine": ...,
    "query/diversity": ...,  # Measure how different queries are
}
```

### Validation Set

Create a **fixed validation set** (1000 samples) that:
- Never changes between experiments
- Evaluates every 10K training samples
- Reports all metrics

---

## Recommendation 5: Resource Requirements

### Minimum Viable Experiment

| Resource | Requirement |
|----------|-------------|
| GPU | 1× RTX 4090 (24GB) |
| Storage | 200 GB (precomputed data) |
| Time per experiment | ~20 hours (500K samples) |
| Total for full ablation | ~3 weeks |

### For Statistical Significance

| Requirement | Value |
|-------------|-------|
| Seeds per experiment | 3 |
| Total experiments | 15 (baselines + ablations) |
| Total GPU hours | ~900 hours |

### If Multi-GPU Available

- Data parallel across 4× GPUs
- ~4x speedup
- Full ablation in ~1 week

---

## Immediate Action Items

### This Week

1. [ ] Download and precompute 100K video samples locally
2. [ ] Create fixed validation set (1K samples)
3. [ ] Implement WebDataset or multi-threaded loading
4. [ ] Verify GPU utilization >80%

### Next Week

1. [ ] Run 3 baseline experiments (3 seeds each)
2. [ ] Establish variance bounds
3. [ ] Create automated experiment runner

### Following Weeks

1. [ ] Run full ablation study
2. [ ] Statistical analysis
3. [ ] Write up results with confidence intervals

---

## Scripts to Create

### 1. `scripts/setup/precompute_webvid_local.py`
- Download N videos from WebVid-10M
- Extract frames uniformly
- Compute VAE latents
- Save to disk in efficient format

### 2. `scripts/train_controlled.py`
- Fixed config template
- Samples-based training (not steps)
- Automatic checkpointing at sample milestones
- Validation every N samples

### 3. `scripts/run_ablation_suite.py`
- Launch multiple experiments with different seeds
- Track all experiments in wandb
- Aggregate results automatically

---

## Summary

The current experiments provide **directional evidence** that:
- Captioning benefits from foveated attention
- Joint training may help reconstruction

But we **cannot make quantitative claims** until:
- Training configurations are standardized
- Data loading is not the bottleneck
- Multiple seeds are run
- Controlled ablations are performed

The path forward is clear: fix infrastructure first, then run proper experiments.
