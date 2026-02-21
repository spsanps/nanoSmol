# fVLM-135M Benchmark Results

## Model
- **Architecture**: SmolLM2-135M-Instruct + DINOv2-small (157.6M total params)
- **Training**: 3-stage pipeline (Visual Alignment → VL-SFT → DPO)
- **Checkpoint**: Stage 3 (DPO) best, step 2000
- **HuggingFace**: `sanps/fVLM-135M`

## Inference Modes
- **coarse_only**: Single static-query pass (fastest, no foveation)
- **coarse_fine**: Two-pass parallel (training mode, with foveated attention)
- **autoregressive**: True sequential inference with KV cache (highest quality)

## Results Summary

| Benchmark | Coarse-Only | Coarse→Fine | Autoregressive |
|-----------|------------|-------------|----------------|
| Val 10K (loss ↓) | 1.8790 | 1.5327 | **1.5308** |
| MVBench (3800) | 27.4% | **28.0%** | 27.9% |
| Video-MME (2700) | 26.2% | **29.5%** | 28.7% |
| POPE (9000) | 50.0% | 50.0% | 50.0% |
| ScienceQA (2017) | **36.4%** | 35.6% | 35.4% |

### Val 10K Loss (1000 samples)
| Mode | Avg Loss |
|------|----------|
| coarse_only | 1.8790 |
| coarse_fine | 1.5327 |
| autoregressive | 1.5308 |

### MVBench (3800 MCQ, 20 video categories)
| Mode | Accuracy |
|------|----------|
| coarse_only | 27.4% |
| coarse_fine | 28.0% |
| autoregressive | 27.9% |
| *random baseline* | *25.0%* |

### Video-MME (2700 MCQ)
| Mode | Accuracy |
|------|----------|
| coarse_only | 26.2% |
| coarse_fine | 29.5% |
| autoregressive | 28.7% |
| *random baseline* | *25.0%* |

### POPE (9000 yes/no, hallucination detection)
| Mode | Accuracy |
|------|----------|
| coarse_only | 50.0% |
| coarse_fine | 50.0% |
| autoregressive | 50.0% |
| *random baseline* | *50.0%* |

### ScienceQA (2017 image MCQ)
| Mode | Accuracy |
|------|----------|
| coarse_only | 36.4% |
| coarse_fine | 35.6% |
| autoregressive | 35.4% |
| *random baseline* | *~25%* |

## Comparison with SmolVLM2

| Benchmark | fVLM-135M (best) | SmolVLM2-256M | SmolVLM2-500M | SmolVLM2-2.2B |
|-----------|-----------------|---------------|---------------|---------------|
| MVBench | 28.0% | 32.7% | 40.0% | 47.0% |
| Video-MME | 29.5% | 33.7% | 42.5% | 52.2% |
| MLVU | — | 40.6% | — | — |
| POPE | 50.0% | — | — | — |
| ScienceQA | 36.4% | — | — | — |

## Analysis
- **Val loss**: Foveated modes (coarse_fine, autoregressive) improve loss by ~19% over coarse_only
- **MVBench**: +3% above random (25%) — modest but expected for 135M. Foveation helps slightly
- **Video-MME**: coarse_fine is best at 29.5%, +4.5% above random. Foveation adds +3.3% over coarse_only
- **POPE**: Exactly at random (50%) — model consistently predicts one answer. Expected for 135M
- **ScienceQA**: Best result at 36.4% (coarse_only). Image MCQ doesn't benefit from foveation
- **vs SmolVLM2-256M**: fVLM-135M scores ~4-5% lower despite using only 1 visual token per frame
  (SmolVLM2 uses 64-256 tokens per image). Efficiency vs quality tradeoff

## Benchmark Timing (v2 optimized, A100 80GB)
- Val 10K: 323s (~5.4 min)
- MVBench: 2305s (~38 min)
- Video-MME: 5539s (~92 min)
- POPE: 1405s (~23 min)
- ScienceQA: 319s (~5 min)
- **Total: 9658s (~161 min / 2.7 hours)**
