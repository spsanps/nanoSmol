#!/bin/bash
# =============================================================================
# GPU Pod Bootstrap Script
# =============================================================================
# Run this FIRST on the GPU pod to set up the environment and start
# background data downloads while training begins.
#
# Usage:
#   source /workspace/.bashrc_runpod    # ALWAYS first
#   bash release/scripts/gpu_bootstrap.sh
#
# This script:
#   1. Installs GPU PyTorch
#   2. Verifies all data and model weights
#   3. Starts WebVid download in background (if not already done)
#   4. Runs a quick dry-run to verify training works
#   5. Prints instructions for starting training
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[bootstrap]${NC} $*"; }
warn() { echo -e "${YELLOW}[bootstrap]${NC} $*"; }
err() { echo -e "${RED}[bootstrap]${NC} $*" >&2; }

# Check cache redirects
if [[ ! -f /workspace/.bashrc_runpod ]]; then
    err "Missing /workspace/.bashrc_runpod!"
    err "Run these commands first:"
    echo 'export HOME=/workspace'
    echo 'export HF_HOME=/workspace/.cache/huggingface'
    echo 'export XDG_CACHE_HOME=/workspace/.cache'
    echo 'export PIP_CACHE_DIR=/workspace/.pip/cache'
    echo 'export TMPDIR=/workspace/tmp'
    exit 1
fi

source /workspace/.bashrc_runpod
log "Cache redirects loaded"

# Step 1: Install GPU PyTorch
log "Step 1: Installing GPU PyTorch..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "  GPU PyTorch already installed"
else
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install torchvision --index-url https://download.pytorch.org/whl/cu124
    log "  GPU PyTorch installed"
fi

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
log "  Found $GPU_COUNT GPU(s)"

# Step 2: Verify model weights
log "Step 2: Checking model weights..."
MODELS=("SmolLM2-135M-Instruct" "SmolLM2-360M-Instruct" "SmolLM2-1.7B-Instruct"
        "dinov2-small" "dinov2-base" "SmolVLM2-256M-Video-Instruct" "SmolVLM2-2.2B-Instruct")
for model in "${MODELS[@]}"; do
    if [[ -d "/workspace/models/$model" ]]; then
        echo "  [OK] $model"
    else
        warn "  [MISSING] $model"
    fi
done

# Step 3: Verify data
log "Step 3: Checking data..."
echo "  SmolTalk:"
for s in stage1 stage2 stage3; do
    count=$(ls /workspace/data/text_retention/$s/*.tar 2>/dev/null | wc -l)
    echo "    $s: $count shards"
done

echo "  Cauldron: $(ls /workspace/data/cauldron_full/*.tar 2>/dev/null | wc -l) shards"
echo "  Stage 3:  $(ls /workspace/data/stage3_youtube/*.tar 2>/dev/null | wc -l) shards"
echo "  OpenVid:  $(ls /workspace/data/openvid/*.tar 2>/dev/null | wc -l) shards"
echo "  Eval:     $(ls /workspace/data/eval/val_10k/*.tar 2>/dev/null | wc -l) shards"

# Step 4: Check OpenVid data availability
OPENVID_SHARDS=$(ls /workspace/data/openvid/*.tar 2>/dev/null | wc -l)
log "Step 4: OpenVid has $OPENVID_SHARDS shards (Stage 1 video data)"
if [[ $OPENVID_SHARDS -lt 50 ]]; then
    warn "  Low shard count — Stage 1 may need more OpenVid data."
    warn "  Consider resuming download: python release/scripts/download_openvid.py --resume"
fi

# Step 5: Dry run
log "Step 5: Running training dry-run..."
python -c "
import yaml, sys
sys.path.insert(0, '.')
# Test config loading
with open('release/configs/stage1_lite.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'  Config loaded: stage={cfg[\"stage\"]}, model={cfg[\"model\"][\"llm\"][-20:]}')

# Test model creation
from release.model.foveated_vlm import FoveatedVLM
model = FoveatedVLM(
    llm_name=cfg['model']['llm'],
    dino_name=cfg['model']['dino'],
    query_dim=cfg['model']['query_dim'],
)
total = sum(p.numel() for p in model.parameters())
print(f'  Model created: {total/1e6:.1f}M params')
print('  Dry run PASSED')
"

# Summary
echo ""
log "=========================================="
log "GPU Pod Ready!"
log "=========================================="
echo ""
echo "Recommended training sequence:"
echo ""
echo "  # Option A: Full Stage 1 with OpenVid video data"
echo "  torchrun --nproc_per_node=$GPU_COUNT release/train.py --config release/configs/stage1_webvid.yaml"
echo ""
echo "  # Option B: Stage 1 Lite warmup (uses Cauldron images, 2h)"
echo "  torchrun --nproc_per_node=$GPU_COUNT release/train.py --config release/configs/stage1_lite.yaml"
echo ""
echo "  # Option C: Run ablation grid"
echo "  bash release/speedrun.sh --ablations"
echo ""
echo "  # Option D: Full 3-stage pipeline"
echo "  bash release/speedrun.sh"
echo ""
