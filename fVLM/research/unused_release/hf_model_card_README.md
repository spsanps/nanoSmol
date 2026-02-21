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

### Video Benchmarks

| Benchmark | fVLM-135M | SmolVLM2-256M | SmolVLM2-500M | SmolVLM2-2.2B |
|-----------|:---------:|:------------:|:------------:|:------------:|
| **MVBench** (3800 MCQ) | 28.0% | 32.7% | 39.7% | 46.3% |
| **Video-MME** (2700 MCQ) | 29.5% | 33.7% | 42.2% | 52.1% |

### Image Benchmarks

| Benchmark | fVLM-135M | SmolVLM2-256M | SmolVLM2-500M | SmolVLM2-2.2B |
|-----------|:---------:|:------------:|:------------:|:------------:|
| **ScienceQA** (2017 MCQ) | 36.0% | 73.8% | 80.0% | 89.6% |

> **Key context**: fVLM-135M uses **1 visual token per frame** vs SmolVLM2's 64-256 tokens per image. fVLM-135M has 158M params total — 1.6x smaller than SmolVLM2-256M. The gap on video benchmarks (4-5%) is modest given the extreme compression.

### Results by Inference Mode

fVLM supports three inference modes with different speed/quality tradeoffs:

| Benchmark | Coarse-Only | Coarse→Fine | Autoregressive |
|-----------|:----------:|:-----------:|:--------------:|
| MVBench | 27.4% | **28.0%** | 27.9% |
| Video-MME | 26.2% | **29.5%** | 28.7% |
| ScienceQA | 34.7% | **36.0%** | **36.0%** |

- **Coarse-Only**: Single static-query pass (fastest, no foveation)
- **Coarse→Fine**: Two-pass parallel forward (training mode, with foveated attention)
- **Autoregressive**: Sequential inference with KV cache (highest quality)

### Analysis

- **Foveation helps on video**: coarse→fine adds +3.3% on Video-MME over coarse-only, confirming that learned "where to look" queries improve video understanding
- **ScienceQA**: Best at 36.0% with coarse_fine/autoregressive modes — foveated attention provides a small benefit even on static images when replicated to 8 frames
- **Scale gap**: The large gap on ScienceQA (36% vs 74%) shows the 135M backbone limits image reasoning. Video benchmarks are closer because foveated compression is highly efficient for temporal tasks

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

## Usage

### Setup

```python
import torch
from torchvision import transforms
from transformers import AutoTokenizer
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
model = model.to("cuda").to(torch.bfloat16).eval()

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Standard DINO preprocessing
frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Image Input

**Important**: fVLM treats all inputs as video. Static images must be **replicated to 8 frames** to match training distribution (Stage 2 and 3 used `replicate_image_frames: 8`). Passing a single frame for an image will produce degraded results.

```python
from PIL import Image

img = Image.open("photo.jpg").convert("RGB")
frame_tensor = frame_transform(img)                      # [3, 224, 224]
frames = frame_tensor.unsqueeze(0).repeat(8, 1, 1, 1)   # [8, 3, 224, 224] — replicate to 8
frames = frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)  # [1, 8, 3, 224, 224]
```

### Video Input

For video, sample up to 64 frames uniformly. No replication needed.

```python
# video_frames: list of PIL Images (sampled from video)
tensors = [frame_transform(f) for f in video_frames]
frames = torch.stack(tensors).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
# frames shape: [1, T, 3, 224, 224] where T = number of frames (1-64)
```

### Inference

```python
# Tokenize prompt
messages = [
    {"role": "user", "content": "Describe what is happening in this image."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
attention_mask = torch.ones_like(input_ids)
loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

# Forward pass (coarse_fine mode recommended for best quality)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    result = model(
        frames=frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        mode="coarse_fine",
    )
# result["logits"]: [B, S, V] text logits
# result["loss"]: scalar cross-entropy loss
```

### Inference Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `coarse_only` | Single static-query pass | Fastest; good for images |
| `coarse_fine` | Two-pass parallel forward | Best overall; uses foveated attention |
| `autoregressive` | Sequential with KV cache | Highest quality for video |

## License

Apache 2.0
