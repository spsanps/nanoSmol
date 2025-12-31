# Foveated Vision-Language Model

A novel VLM architecture that processes video frame-by-frame using foveated attention, where the LLM controls WHERE to look in each frame.

## Documentation

- `core_docs/foveated_vlm_proposal.md` - Architecture specification
- `core_docs/foveated_vlm_execution_guide.md` - Implementation guide
- `claude.md` - Development guide for AI assistant

## Quick Start

```bash
# 1. Setup environment
conda activate fVLM
pip install -r requirements.txt

# 2. Download models and data
bash scripts/download_data.sh

# 3. Precompute VAE latents
python scripts/precompute_latents.py

# 4. Train
python scripts/train_phase1.py --config configs/phase1.yaml

# 5. Monitor
tensorboard --logdir outputs/logs
```

## Project Structure

```
fVLM/
├── core_docs/          # Architecture & execution docs
├── configs/            # Training configurations
├── data/              # Videos and preprocessed latents
├── src/               # Source code
│   ├── model/         # Model components
│   ├── data/          # Data pipeline
│   └── training/      # Training utilities
├── scripts/           # Executable scripts
├── notebooks/         # Analysis notebooks
└── outputs/           # Checkpoints, logs, visualizations
```

## Hardware Requirements

- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Precision: bfloat16
- Batch size: 2 (with gradient accumulation)

## Success Metric

**Core hypothesis**: Dynamic foveated attention (Pass 2) outperforms static attention (Pass 1)

Validation: `loss_fine < loss_coarse` consistently during training
