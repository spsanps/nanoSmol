# Foveated VLM: MLE Execution Guide

## Document Purpose

This document provides ML Engineers with everything needed to implement and train the Foveated VLM proof-of-concept. It complements the architecture proposal (`foveated_vlm_proposal_v4.md`) with practical execution details.

**Target audience:** ML Engineers joining the project
**Hardware allocation:** 1× NVIDIA RTX 4090 (24GB VRAM, ~20GB usable)
**Goal:** Demonstrate that dynamic foveated attention improves next-frame prediction over static attention (loss_fine < loss_coarse)

---

## 1. Project Context

### 1.1 What We're Building

A vision-language model that:
1. Processes video frame-by-frame with ONE token per frame (not 196+ patches)
2. Uses the LLM to decide WHERE to look in the next frame
3. Predicts next-frame VAE latents as the training objective

### 1.2 Core Hypothesis

> Dynamic, content-aware visual attention (Pass 2) extracts more useful information than static attention (Pass 1).

**Success metric:** `loss_fine < loss_coarse` consistently during training.

### 1.3 Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| GPU | 1× RTX 4090 | PoC allocation |
| Usable VRAM | ~20GB | System monitoring overhead |
| CPU offload | Disabled | Performance requirement |
| Baselines | Deferred | Focus on core PoC first |

---

## 2. Hardware Budget Analysis

### 2.1 Model Memory Footprint (bf16)

| Component | Parameters | Memory (bf16) |
|-----------|------------|---------------|
| DINOv3 ViT-S/16 | ~22M | ~44MB |
| SmolLM2-135M | ~135M | ~270MB |
| Prediction Head | ~5M | ~10MB |
| Projections + Embeddings | ~2M | ~4MB |
| **Total Model** | **~164M** | **~328MB** |

### 2.2 Activation Memory (the real constraint)

For T=8 frames, batch_size=B:

| Component | Shape | Memory per sample |
|-----------|-------|-------------------|
| DINO patches (all frames) | [T, 256, 384] | ~3MB |
| DINO KV cache (12 layers) | [12, 2, 256, 384] | ~9MB |
| LLM activations (2 passes) | ~2× [seq_len, 576] | ~2MB |
| VAE latents | [T, 4, 32, 32] | ~0.13MB |
| Gradients | ~1.5× model | ~500MB |

**Conservative estimate:** ~1.5GB per sample for T=8

### 2.3 Recommended Configuration

```
Batch size: 2
Gradient accumulation: 4
Effective batch: 8
Frames per clip (T): 8
Precision: bf16
VRAM usage: ~15-18GB (leaves headroom)
```

If OOM occurs:
1. Reduce batch_size to 1, increase grad_accum to 8
2. Reduce T to 4 frames
3. Enable gradient checkpointing on LLM

---

## 3. Frame Sampling Strategy

### 3.1 FPS Considerations

Source videos are typically 24-30 fps. Using all frames is:
- Computationally wasteful (consecutive frames nearly identical)
- Memory prohibitive
- Not how humans or other video models process video

**Standard practice:** Sample 1-4 fps effective rate.

### 3.2 Recommended Sampling

For LLaVA-Video-178K (0-30 second clips):

```python
# Target: T=8 frames uniformly sampled from video
# This gives ~0.25-0.5 fps for 30s video, ~1-2 fps for 10s video

def sample_frames(video_path, num_frames=8, target_size=256):
    """
    Uniformly sample frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (T)
        target_size: Resize to target_size × target_size
    
    Returns:
        frames: [T, 3, H, W] tensor in [-1, 1] range
    """
    import decord
    decord.bridge.set_bridge('torch')
    
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    
    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices)  # [T, H, W, C]
    
    # Resize and normalize
    frames = resize(frames, target_size)  # [T, 3, H, W]
    frames = frames / 127.5 - 1.0  # [-1, 1] for VAE
    
    return frames
```

### 3.3 Why This Works

| Video Length | Effective FPS | Frame Interval |
|--------------|---------------|----------------|
| 5 seconds | 1.6 fps | ~0.6s between frames |
| 15 seconds | 0.5 fps | ~2s between frames |
| 30 seconds | 0.25 fps | ~4s between frames |

This captures temporal dynamics while keeping computation manageable. The model learns to predict across these time gaps, which is actually harder (more change between frames) and more useful.

---

## 4. Data Pipeline

### 4.1 Dataset: LLaVA-Video-178K

Using the 0-30s academic subset. Already part of SmolVLM2's training mixture.

**Download (~5-6GB starter slice):**
```bash
huggingface-cli download lmms-lab/LLaVA-Video-178K \
  --repo-type dataset \
  --include "0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json" \
  --include "0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_1.tar.gz" \
  --local-dir data/llava_video --local-dir-use-symlinks False

# Extract
mkdir -p data/videos
tar -xzf data/llava_video/0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_1.tar.gz -C data/videos
```

**Scale up later:** Add `_videos_2.tar.gz`, `_videos_3.tar.gz`, etc.

### 4.2 Preprocessing Pipeline (Run Once)

Precompute VAE latents to avoid VAE forward pass during training:

```python
"""
precompute_latents.py

Run once before training. Saves ~10x training speed.
Output: data/latents/{video_id}.pt containing [T, 4, 32, 32] tensor
"""

import torch
from diffusers.models import AutoencoderKL
from pathlib import Path
from tqdm import tqdm

# Config
INPUT_DIR = "data/videos"
OUTPUT_DIR = "data/latents"
NUM_FRAMES = 8
FRAME_SIZE = 256

# Load frozen VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", 
    torch_dtype=torch.bfloat16
).cuda().eval()

@torch.no_grad()
def encode_video(video_path):
    frames = sample_frames(video_path, NUM_FRAMES, FRAME_SIZE)  # [T, 3, 256, 256]
    frames = frames.to(dtype=torch.bfloat16, device='cuda')
    
    # Encode frame by frame (memory efficient)
    latents = []
    for t in range(frames.shape[0]):
        lat = vae.encode(frames[t:t+1]).latent_dist.mean
        lat = lat * vae.config.scaling_factor
        latents.append(lat.cpu())
    
    return torch.cat(latents, dim=0)  # [T, 4, 32, 32]

# Process all videos
Path(OUTPUT_DIR).mkdir(exist_ok=True)
video_paths = list(Path(INPUT_DIR).glob("**/*.mp4"))

for video_path in tqdm(video_paths):
    output_path = Path(OUTPUT_DIR) / f"{video_path.stem}.pt"
    if output_path.exists():
        continue
    
    try:
        latents = encode_video(video_path)
        torch.save(latents, output_path)
    except Exception as e:
        print(f"Failed {video_path}: {e}")
```

**Storage estimate:** 8 frames × 4×32×32 × 2 bytes (bf16) = ~64KB per video. 10K videos = ~640MB.

### 4.3 DataLoader

```python
class FoveatedVideoDataset(torch.utils.data.Dataset):
    """
    Loads precomputed VAE latents + raw frames for DINO.
    Text is optional for Phase 1 (self-supervised).
    """
    
    def __init__(self, video_dir, latent_dir, caption_json=None):
        self.video_dir = Path(video_dir)
        self.latent_dir = Path(latent_dir)
        
        # Find all videos with precomputed latents
        self.samples = []
        for latent_path in self.latent_dir.glob("*.pt"):
            video_path = self.video_dir / f"{latent_path.stem}.mp4"
            if video_path.exists():
                self.samples.append({
                    'video': video_path,
                    'latent': latent_path,
                })
        
        # Optional: load captions for Phase 2
        self.captions = {}
        if caption_json:
            # Load and index by video_id
            pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load precomputed VAE latents
        vae_latents = torch.load(sample['latent'])  # [T, 4, 32, 32]
        
        # Load raw frames for DINO (need original pixels)
        frames = sample_frames(sample['video'], num_frames=len(vae_latents))
        # Normalize for DINO: ImageNet stats
        frames = normalize_imagenet(frames)  # [T, 3, 256, 256]
        
        return {
            'frames': frames,           # For DINO encoder
            'vae_latents': vae_latents, # Precomputed targets
            'video_id': sample['video'].stem,
        }
```

---

## 5. Model Checkpoints

### 5.1 Pretrained Models

```python
# DINOv3 ViT-S/16
from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

# SmolLM2-135M
from transformers import AutoModelForCausalLM
llm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# SD VAE (frozen, for preprocessing only)
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

### 5.2 Pre-download (Recommended)

```bash
python -c "from huggingface_hub import snapshot_download as s; \
s('facebook/dinov3-vits16-pretrain-lvd1689m'); \
s('HuggingFaceTB/SmolLM2-135M-Instruct'); \
s('stabilityai/sd-vae-ft-mse')"
```

---

## 6. Training Configuration

### 6.1 Phase 1: Self-Supervised (Next-Frame Prediction)

**Objective:** Learn foveated attention via next-frame VAE latent prediction. Text ignored.

```python
# config_phase1.py

config = {
    # Data
    'num_frames': 8,           # T
    'frame_size': 256,
    'batch_size': 2,
    'grad_accum': 4,           # Effective batch = 8
    
    # Model
    'dino_model': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
    'dino_dim': 384,
    'llm_dim': 576,
    'query_dim': 384,
    
    # Training
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'max_steps': 50000,        # Adjust based on dataset size
    'lambda_coarse': 1.0,      # Auxiliary loss weight
    
    # Precision
    'dtype': 'bfloat16',
    'grad_checkpoint': False,  # Enable if OOM
    
    # Logging
    'log_every': 100,
    'save_every': 5000,
    'eval_every': 1000,
}
```

### 6.2 Training Loop Skeleton

```python
def train_step(model, batch, optimizer, scaler, config):
    frames = batch['frames'].cuda()           # [B, T, 3, 256, 256]
    vae_latents = batch['vae_latents'].cuda() # [B, T, 4, 32, 32]
    
    # Phase 1: no text, use dummy embeddings
    text_embeds = model.get_empty_text_embeds(batch_size=frames.shape[0])
    
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, loss_fine, loss_coarse = model(text_embeds, frames, vae_latents)
        loss = loss / config['grad_accum']
    
    scaler.scale(loss).backward()
    
    return {
        'loss': loss.item() * config['grad_accum'],
        'loss_fine': loss_fine.item(),
        'loss_coarse': loss_coarse.item(),
        'loss_ratio': loss_coarse.item() / (loss_fine.item() + 1e-8),
    }
```

### 6.3 Key Metrics to Log

```python
# Every log_every steps:
wandb.log({
    'loss/total': loss,
    'loss/fine': loss_fine,      # Pass 2 (dynamic queries)
    'loss/coarse': loss_coarse,  # Pass 1 (static query)
    'loss/ratio': loss_coarse / loss_fine,  # >1 means foveation helps!
    
    # Attention diagnostics
    'attention/entropy_static': ...,   # Should be higher (less focused)
    'attention/entropy_dynamic': ...,  # Should be lower (more focused)
    
    # Training dynamics
    'grad_norm': ...,
    'lr': ...,
})
```

---

## 7. Project Milestones

### Milestone 1: Setup & Data Pipeline

| Task | Deliverable |
|------|-------------|
| Environment setup, download models | Working imports |
| Download dataset, implement frame sampling | `sample_frames()` working |
| Precompute VAE latents | `data/latents/*.pt` files |
| DataLoader, verify shapes | DataLoader returning correct shapes |

**Checkpoint:** Can load batch of (frames, vae_latents) with correct shapes.

### Milestone 2: Model Implementation

| Task | Deliverable |
|------|-------------|
| FoveatedEncoder (DINO + query mechanism) | `encoder.encode_patches()`, `encoder.query()` working |
| Two-pass forward (Pass 1 + Pass 2) | Both passes producing correct shapes |
| Prediction head (FiLM conditioning) | End-to-end forward pass |
| Loss computation, backward pass | Gradients flowing, no NaN |

**Checkpoint:** Single training step runs without error, loss decreases.

### Milestone 3: Training & Debugging

| Task | Deliverable |
|------|-------------|
| Training loop, logging, checkpointing | Training runs continuously |
| Debug attention patterns, verify non-collapse | Attention visualizations |
| Full Phase 1 training run | loss_fine vs loss_coarse curves |

**Checkpoint:** Training stable, loss decreasing, attention non-uniform.

### Milestone 4: Analysis & Report

| Task | Deliverable |
|------|-------------|
| Attention visualization, loss analysis | Plots and figures |
| Qualitative results (decode predictions) | Sample reconstructions |
| Write-up: what worked, what didn't | Technical report |
| Prepare resource request | Presentation for management |

---

## 8. Success Criteria

### 8.1 Primary Success Metric

```
loss_fine < loss_coarse (consistently after warmup)
```

This means dynamic foveated attention extracts more useful information than static attention.

**Quantify as:** `improvement = (loss_coarse - loss_fine) / loss_coarse`

| Improvement | Interpretation |
|-------------|----------------|
| < 0% | Foveation hurts (debug needed) |
| 0-5% | Marginal improvement |
| 5-15% | Solid improvement (PoC successful) |
| > 15% | Strong improvement (very promising) |

### 8.2 Secondary Metrics

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Attention entropy (dynamic) | Lower than static | Same as static |
| Attention patterns | Track moving objects | Uniform/fixed |
| Loss curve | Smooth decrease | Spiky/divergent |
| Gradient norms | Stable | Exploding/vanishing |

### 8.3 Failure Modes to Watch

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Query collapse | All attention uniform | Check entropy, add regularization |
| Static query collapse | q_static attention uniform | Verify gradient flow to q_static |
| Mode token dominance | loss_fine ≈ loss_coarse always | Check mode token embeddings |
| NaN loss | Training crashes | Reduce LR, check grad norms |
| OOM | CUDA out of memory | Reduce batch, enable grad checkpoint |

---

## 9. Results & Analysis Plan

### 9.1 Quantitative Results

**Table 1: Loss Comparison**
```
| Metric      | Value |
|-------------|-------|
| loss_coarse | X.XXX |
| loss_fine   | X.XXX |
| Improvement | X.X%  |
```

**Figure 1: Training Curves**
- Plot loss_fine and loss_coarse over training steps
- Should see gap emerge after initial warmup

**Figure 2: Loss Ratio Over Time**
- Plot loss_coarse / loss_fine
- Should increase and stabilize > 1.0

### 9.2 Qualitative Results

**Figure 3: Attention Visualizations**
- For 5-10 sample videos, show attention heatmaps over time
- Compare static (q_static) vs dynamic (q_t) attention patterns
- Look for: dynamic attention tracking motion

**Figure 4: Prediction Quality**
- Decode predicted VAE latents to pixels (using frozen VAE decoder)
- Show: Frame t, Predicted Frame t+1, Actual Frame t+1
- Qualitative assessment of prediction quality

### 9.3 Analysis Questions

1. **Does dynamic attention help?**
   - Compare loss_fine vs loss_coarse
   - Visualize attention patterns

2. **What does the model attend to?**
   - Attention heatmaps on videos with clear motion
   - Does it track moving objects?

3. **Where does it fail?**
   - Find videos with highest loss
   - Analyze: scene cuts? fast motion? static scenes?

4. **Is the bottleneck too severe?**
   - Compare reconstruction quality to baselines (future work)
   - Does 1 token capture enough?

### 9.4 Report Structure (for Management)

```
1. Executive Summary
   - One sentence: "Dynamic foveated attention improves prediction by X%"
   - Key finding: loss_fine < loss_coarse
   
2. Method (brief)
   - Architecture diagram
   - Training setup
   
3. Results
   - Table 1: Loss comparison
   - Figure 1-2: Training curves
   - Figure 3-4: Visualizations
   
4. Resource Request
   - What we learned
   - What more compute would enable
   - Proposed next steps
```

---

## 10. Code Organization

```
foveated_vlm/
├── README.md
├── requirements.txt
├── configs/
│   ├── phase1.yaml
│   └── phase2.yaml
├── data/
│   ├── videos/              # Raw videos
│   ├── latents/             # Precomputed VAE latents
│   └── llava_video/         # Downloaded dataset
├── src/
│   ├── model/
│   │   ├── encoder.py       # FoveatedEncoder (DINO + query)
│   │   ├── prediction.py    # PredictionHead (FiLM)
│   │   └── foveated_vlm.py  # Main model
│   ├── data/
│   │   ├── dataset.py       # FoveatedVideoDataset
│   │   ├── sampling.py      # Frame sampling utilities
│   │   └── precompute.py    # VAE latent preprocessing
│   └── training/
│       ├── train.py         # Training loop
│       └── utils.py         # Logging, checkpointing
├── scripts/
│   ├── download_data.sh
│   ├── precompute_latents.py
│   └── train_phase1.py
└── notebooks/
    ├── visualize_attention.ipynb
    └── analyze_results.ipynb
```

---

## 11. Quick Start Commands

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets accelerate diffusers decord opencv-python tqdm

# 2. Download models
python -c "from huggingface_hub import snapshot_download as s; \
s('facebook/dinov3-vits16-pretrain-lvd1689m'); \
s('HuggingFaceTB/SmolLM2-135M-Instruct'); \
s('stabilityai/sd-vae-ft-mse')"

# 3. Download dataset
bash scripts/download_data.sh

# 4. Precompute VAE latents
python scripts/precompute_latents.py

# 5. Train
python scripts/train_phase1.py --config configs/phase1.yaml

# 6. Monitor (in another terminal)
tensorboard --logdir outputs/
```

---

## 12. Troubleshooting

### OOM Errors

```python
# Option 1: Reduce batch size
config['batch_size'] = 1
config['grad_accum'] = 8

# Option 2: Reduce frames
config['num_frames'] = 4

# Option 3: Gradient checkpointing
llm.gradient_checkpointing_enable()

# Option 4: Reduce sequence length by truncating text
config['max_text_tokens'] = 32
```

### Training Instability

```python
# Reduce learning rate
config['learning_rate'] = 5e-5

# Increase warmup
config['warmup_steps'] = 1000

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Loss Not Decreasing

1. Verify data pipeline (visualize frames, check latent shapes)
2. Check gradient flow (`torch.autograd.detect_anomaly()`)
3. Simplify: train prediction head only first, freeze rest
4. Verify loss computation indexing

### Attention Collapse (Uniform)

1. Check attention entropy metric
2. Verify q_static receiving gradients
3. Add entropy regularization:
   ```python
   attention_entropy = -(attn * attn.log()).sum(dim=-1).mean()
   loss += 0.01 * attention_entropy  # Encourage non-uniform
   ```

---

## 13. Contact & Resources

- Architecture proposal: `foveated_vlm_proposal_v4.md`
- Model checkpoints: HuggingFace Hub (see Section 5)
- Dataset: LLaVA-Video-178K on HuggingFace
- Questions: [Team Slack/Email]

---

## Appendix A: Memory Profiling

Run this to verify memory usage before full training:

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Single batch test
model = FoveatedVideoModel().cuda()
batch = next(iter(dataloader))

with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
    loss, _, _ = model(batch['text'], batch['frames'].cuda(), batch['vae_latents'].cuda())
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## Appendix B: Attention Visualization

```python
def visualize_attention(model, frames, save_path):
    """
    Visualize attention patterns for a single video.
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # Get attention weights from both passes
        attn_static, attn_dynamic = model.get_attention_maps(frames)
    
    fig, axes = plt.subplots(2, len(frames), figsize=(3*len(frames), 6))
    
    for t in range(len(frames)):
        # Static attention (Pass 1)
        axes[0, t].imshow(attn_static[t].reshape(16, 16).cpu())
        axes[0, t].set_title(f'Static t={t}')
        axes[0, t].axis('off')
        
        # Dynamic attention (Pass 2)
        axes[1, t].imshow(attn_dynamic[t].reshape(16, 16).cpu())
        axes[1, t].set_title(f'Dynamic t={t}')
        axes[1, t].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```
