# fVLM: Foveated Vision-Language Model

A VLM that compresses each video frame into **1 visual token** via query-guided foveated attention on DINOv2 features. The LLM learns *where to look* by generating the query for the next frame.

## Models

| Model | Params | LLM | HuggingFace |
|-------|--------|-----|-------------|
| fVLM-135M | 157.6M | SmolLM2-135M-Instruct | [sanps/fVLM-135M](https://huggingface.co/sanps/fVLM-135M) |
| fVLM-1.7B | ~1.73B | SmolLM2-1.7B-Instruct | *training* |

## Quick Start

```python
import torch
from torchvision import transforms
from model import FoveatedVLM

model = FoveatedVLM(
    llm_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    dino_name="facebook/dinov2-small",
    query_dim=384, visual_scale=0.14, deep_query=True,
)
```

### Image Input (Important)

**Images must be replicated to 8 frames** to match the training distribution. The model was trained with `replicate_image_frames: 8` in Stages 2-3. Passing a single frame for an image will produce degraded results.

```python
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("photo.jpg").convert("RGB")
frame = transform(img)                              # [3, 224, 224]
frames = frame.unsqueeze(0).repeat(8, 1, 1, 1)      # [8, 3, 224, 224]
frames = frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)  # [1, 8, 3, H, W]
```

### Video Input

Sample up to 64 frames uniformly. No replication needed.

```python
frames = torch.stack([transform(f) for f in video_frames])  # [T, 3, 224, 224]
frames = frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
```

## Architecture

| Component | Details |
|-----------|---------|
| Vision Encoder | DINOv2-small (22M) |
| Attention | Deep query-guided cross-attention (12 layers) |
| Visual Tokens | 1 per frame (query-compressed) |
| Language Model | SmolLM2-135M-Instruct or 1.7B-Instruct |
| Query Dimension | 384 |

Each frame is encoded by DINOv2, then a learned query propagates through all 12 DINO layers via cross-attention to produce a single foveated token. The LLM generates the next query from its hidden state, creating a feedback loop.

## Training Pipeline

3-stage pipeline (same for 135M and 1.7B):

| Stage | Data | Loss | LR |
|-------|------|------|----|
| 1. Visual Alignment | OpenVid-1M + WebVid + SmolTalk | All-text CE | Converging schedule |
| 2. Vision-Language SFT | Cauldron + video datasets + SmolTalk | Answer-only CE | Flat + cosine decay |
| 3. DPO | RLAIF-V (83K pairs) | DPO (beta=0.1) | 1e-6 |

```bash
python train.py --config configs/stage1_135M.yaml
```

## Project Structure

```
model.py          # FoveatedVLM — the model (~1000 lines)
encoder.py        # FoveatedEncoder — DINO cross-attention
train.py          # Training loop (all 3 stages + DPO)
data.py           # WebDataset video/image loader
collate.py        # Variable-length batch padding
text_interleave.py # 14% text-only data mixing
tokenization.py   # Chat-template tokenization for all stages
benchmark.py      # MCQ evaluation (MVBench, Video-MME, ScienceQA)
checkpoint.py     # Save/load training state
logger.py         # wandb + CSV + stdout logging
schedule.py       # LR schedules (cosine, converging)
distributed.py    # Multi-GPU DDP setup
attention_viz.py  # Attention heatmaps + entropy
configs/          # Training configs (6 YAML files)
```

## Benchmark Results (fVLM-135M)

### Video

| Benchmark | fVLM-135M | SmolVLM2-256M | SmolVLM2-2.2B |
|-----------|:---------:|:------------:|:------------:|
| MVBench (3800 MCQ) | 28.0% | 32.7% | 46.3% |
| Video-MME (2700 MCQ) | 29.5% | 33.7% | 52.1% |

### Image

| Benchmark | fVLM-135M | SmolVLM2-256M | SmolVLM2-2.2B |
|-----------|:---------:|:------------:|:------------:|
| ScienceQA (2017 MCQ) | 36.0% | 73.8% | 89.6% |

## Hardware

- **135M training**: A100 80GB, bf16, ~27 samp/s (optimized)
- **1.7B training**: A100 80GB, bf16, gradient checkpointing + torch.compile

## License

Apache 2.0
