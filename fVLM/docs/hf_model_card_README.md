---
license: apache-2.0
language:
  - en
tags:
  - vision-language
  - video-understanding
  - foveated-attention
  - multimodal
  - smollm2
  - dinov2
library_name: pytorch
pipeline_tag: image-text-to-text
---

# fVLM-135M (Foveated Vision-Language Model)

A compact vision-language model that uses **foveated attention** to compress each video frame into a single visual token, enabling efficient processing of long videos.

## Benchmark Results

### Summary

| Benchmark | Metric | fVLM-135M | SmolVLM2-256M | SmolVLM2-500M | SmolVLM2-2.2B |
|-----------|--------|-----------|---------------|---------------|---------------|
| **MVBench** | Accuracy (3800 MCQ) | 28.0% | 32.7% | 40.0% | 47.0% |
| **Video-MME** | Accuracy (2700 MCQ) | 29.5% | 33.7% | 42.5% | 52.2% |
| **ScienceQA** | Accuracy (2017 MCQ) | 36.4% | — | — | — |
| **POPE** | Accuracy (9000 Y/N) | 50.0% | — | — | — |
| **Val 10K** | Loss ↓ (1000 samples) | 1.531 | — | — | — |

> **Note**: fVLM-135M uses **1 visual token per frame** vs SmolVLM2's 64–256 tokens per image.
> Despite being 2× smaller than SmolVLM2-256M, fVLM-135M scores within 4–5% on video benchmarks — demonstrating the efficiency of foveated attention.

### Results by Inference Mode

fVLM supports three inference modes with different speed/quality tradeoffs:

| Benchmark | Coarse-Only | Coarse→Fine | Autoregressive |
|-----------|------------|-------------|----------------|
| Val 10K (loss ↓) | 1.879 | 1.533 | **1.531** |
| MVBench | 27.4% | **28.0%** | 27.9% |
| Video-MME | 26.2% | **29.5%** | 28.7% |
| POPE | 50.0% | 50.0% | 50.0% |
| ScienceQA | **36.4%** | 35.6% | 35.4% |

- **Coarse-Only**: Single static-query pass (fastest, no foveation)
- **Coarse→Fine**: Two-pass parallel forward (training mode, with foveated attention)
- **Autoregressive**: Sequential inference with KV cache (highest quality)

### Analysis

- **Foveation helps on video**: coarse→fine adds +3.3% on Video-MME over coarse-only, confirming that learned "where to look" queries improve video understanding
- **MVBench**: +3% above random baseline (25%), modest but expected for 135M params
- **POPE**: At random baseline — model consistently predicts one class (expected at this scale)
- **ScienceQA**: Best at 36.4% with coarse-only — static images don't benefit from foveation

## Architecture

| Component | Details |
|-----------|---------|
| **Language Model** | SmolLM2-135M-Instruct |
| **Vision Encoder** | DINOv2-small |
| **Attention** | Deep query-guided foveated cross-attention |
| **Visual Tokens** | 1 token per frame (query-compressed) |
| **Total Parameters** | 157.6M |
| **Query Dimension** | 384 |
| **Visual Scale** | 0.14 |

### How Foveated Attention Works

Unlike standard VLMs that use many visual tokens per image (e.g., 576 for LLaVA), fVLM compresses each frame to a **single visual token** using a learned query mechanism:

1. **DINOv2** encodes each frame into patch features and caches K/V at every layer
2. A **query vector** is propagated through all 12 DINO layers, attending to patch K/V at each layer (deep query attention)
3. The single output token is projected to LLM dimension and prepended to the text sequence
4. The **LLM generates the next query** from its hidden state, creating a feedback loop where the model learns *where to look*

This enables processing **64+ frames** with the same memory as a few frames in traditional VLMs.

## Training Pipeline

### Stage 1: Visual Alignment
- **Data**: OpenVid-1M (905K) + WebVid (19K) + 14% SmolTalk text retention
- **Loss**: Full-text cross-entropy (predict all tokens)
- **LR**: Converging schedule — connector 1e-3 to 3e-5, backbone 1e-5 to 3e-5

### Stage 2: Vision-Language SFT
- **Data**: Cauldron (2M images) + video datasets (~1.6M) + 14% SmolTalk text retention
- **Loss**: Answer-only cross-entropy (mask user/system tokens)
- **LR**: Flat 3e-5 all components with cosine decay

### Stage 3: DPO (Direct Preference Optimization)
- **Data**: RLAIF-V (83K preference pairs)
- **Loss**: DPO with beta=0.1
- **LR**: 1e-6 all components

## Training Performance

Optimized for A100 80GB with coarse-pass optimization (skip text in coarse LLM — causal attention makes it mathematically equivalent):

| Config | Throughput | Memory |
|--------|-----------|--------|
| 135M, bs=32 | ~30 samp/s | 8 GB |
| 1.7B, bs=32, grad_ckpt | 15.7 samp/s | 26.5 GB |

## Model Components

The checkpoint contains the full `FoveatedVLM` model:

- `encoder.dino.*` — DINOv2-small vision backbone
- `encoder.query_input_proj.*` — Query projection (bias=False)
- `encoder.output_proj.*` — Output projection
- `dino_to_llm.*` — DINO→LLM dimension projection
- `llm_to_query.*` — LLM→query dimension projection
- `q_static` — Learnable static query for coarse pass
- `q_init` — Learnable initial query for fine pass
- `llm.*` — SmolLM2-135M-Instruct language model

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from release.model import FoveatedVLM

# Download checkpoint
ckpt_path = hf_hub_download("sanps/fVLM-135M", "model.safetensors")

# Build model
model = FoveatedVLM(
    llm_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    dino_name="facebook/dinov2-small",
    query_dim=384,
    visual_scale=0.14,
    deep_query=True,
)

# Load weights
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

## License

Apache 2.0
