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

## Results

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
*Running...*

### POPE (9000 yes/no, hallucination detection)
*Pending...*

### ScienceQA (2017 image MCQ)
*Pending...*

## Comparison with SmolVLM2

| Benchmark | fVLM-135M (coarse_fine) | SmolVLM2-256M | SmolVLM2-500M | SmolVLM2-2.2B |
|-----------|------------------------|---------------|---------------|---------------|
| MVBench | 28.0% | 32.7% | 40.0% | 47.0% |
| Video-MME | running... | 33.7% | 42.5% | 52.2% |
| MLVU | — | 40.6% | — | — |
| POPE | pending | — | — | — |
| ScienceQA | pending | — | — | — |

## Notes
- fVLM-135M is 157.6M total (135M LLM + 22M DINO + connector)
- SmolVLM2-256M is comparable in parameter count
- fVLM uses 1 visual token per frame (extremely efficient); SmolVLM2 uses ~64-256 per image
- MVBench ~28% is modestly above random (25%) for a 135M model — expected given model size
- coarse_fine and autoregressive modes consistently outperform coarse_only on val loss

## Benchmark Timing (v2 optimized, A100 80GB)
- Val 10K: 323s (~5.4 min)
- MVBench: 2305s (~38 min)
- Video-MME: running
- POPE: pending
- ScienceQA: pending
