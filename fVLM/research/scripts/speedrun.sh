#!/bin/bash
# =============================================================================
# fVLM Speedrun: End-to-End Training Pipeline
# =============================================================================
# Runs the complete training pipeline on 2xA100-80GB.
# MUST source /workspace/.bashrc_runpod FIRST on RunPod pods.
#
# Usage:
#   bash release/speedrun.sh              # Full pipeline
#   bash release/speedrun.sh --stage 2    # Resume from Stage 2
#   bash release/speedrun.sh --ablations  # Run ablation grid only
#   bash release/speedrun.sh --scaling    # Run scaling grid only
#
# Total expected time: ~30h (ablations) + ~25h (main pipeline) + ~20h (scaling)
# Total expected cost: ~$105 at $1.39/hr for 2xA100
# =============================================================================

set -euo pipefail

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

log() { echo -e "${GREEN}[speedrun]${NC} $*"; }
warn() { echo -e "${YELLOW}[speedrun]${NC} $*"; }
err() { echo -e "${RED}[speedrun]${NC} $*" >&2; }

# Parse args
START_STAGE=${START_STAGE:-1}
RUN_ABLATIONS=false
RUN_SCALING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) START_STAGE=$2; shift 2 ;;
        --ablations) RUN_ABLATIONS=true; shift ;;
        --scaling) RUN_SCALING=true; shift ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=${NPROC:-2}

# Verify environment
if [[ ! -f /workspace/.bashrc_runpod ]]; then
    err "Missing /workspace/.bashrc_runpod -- source cache redirect first!"
    exit 1
fi
source /workspace/.bashrc_runpod

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    err "No CUDA available. This script requires GPU."
    exit 1
fi

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
log "Found $GPU_COUNT GPU(s)"

# =========================================================================
# ABLATION GRID  (~30h total on 2xA100)
# =========================================================================
if $RUN_ABLATIONS; then
    log "=== Running Ablation Grid ==="

    ABLATION_CONFIGS=(
        "baseline"
        "A1_deep_query_off"
        "A2_query_dim_192"
        "A3_bias_on"
        "A4_std_002_init"
        "A5_frozen_dino"
        "A6_coarse_only"
        "LR1"
        "LR2"
        "LR3"
        "LR4"
        "T1_no_text_retention"
        "T2_text_7pct"
        "T3_text_28pct"
    )

    for config in "${ABLATION_CONFIGS[@]}"; do
        log "--- Ablation: $config ---"
        torchrun --nproc_per_node=$NPROC \
            "$RELEASE_DIR/train.py" \
            --config "$RELEASE_DIR/configs/ablations/${config}.yaml"
        log "--- $config DONE ---"
    done

    log "=== Ablation Grid Complete ==="
    exit 0
fi

# =========================================================================
# SCALING GRID  (~20h total on 2xA100)
# =========================================================================
if $RUN_SCALING; then
    log "=== Running Scaling Grid ==="

    for config_file in "$RELEASE_DIR"/configs/scaling/run_*.yaml; do
        if [[ ! -f "$config_file" ]]; then
            warn "No scaling configs found. Generate them first."
            break
        fi
        name=$(basename "$config_file" .yaml)
        log "--- Scaling: $name ---"
        torchrun --nproc_per_node=$NPROC \
            "$RELEASE_DIR/train.py" \
            --config "$config_file"
        log "--- $name DONE ---"
    done

    log "=== Scaling Grid Complete ==="
    exit 0
fi

# =========================================================================
# MAIN PIPELINE  (~25h total on 2xA100)
# =========================================================================
log "=== Main Training Pipeline (starting from Stage $START_STAGE) ==="

# --- Stage 1: WebVid Captioning (~12h) ---
if [[ $START_STAGE -le 1 ]]; then
    log "=== Stage 1: WebVid Captioning ==="
    torchrun --nproc_per_node=$NPROC \
        "$RELEASE_DIR/train.py" \
        --config "$RELEASE_DIR/configs/stage1_webvid.yaml"
    log "=== Stage 1 Complete ==="
fi

# --- Stage 2: Vision-Language SFT (~8h) ---
if [[ $START_STAGE -le 2 ]]; then
    log "=== Stage 2: VL SFT ==="
    torchrun --nproc_per_node=$NPROC \
        "$RELEASE_DIR/train.py" \
        --config "$RELEASE_DIR/configs/stage2_vl_sft.yaml"
    log "=== Stage 2 Complete ==="
fi

# --- Stage 3: Video SFT (~5h) ---
if [[ $START_STAGE -le 3 ]]; then
    log "=== Stage 3: Video SFT ==="
    torchrun --nproc_per_node=$NPROC \
        "$RELEASE_DIR/train.py" \
        --config "$RELEASE_DIR/configs/stage3_video_sft.yaml"
    log "=== Stage 3 Complete ==="
fi

# --- Evaluation (all 3 modes) ---
log "=== Running Evaluation ==="
for MODE in coarse_only coarse_fine autoregressive; do
    log "  Eval mode: $MODE"
    python "$RELEASE_DIR/evaluate.py" \
        --config "$RELEASE_DIR/configs/stage3_video_sft.yaml" \
        --checkpoint /workspace/checkpoints/stage3/best.pt \
        --mode "$MODE"
done

log "=== Pipeline Complete ==="
log "Results saved to /workspace/checkpoints/"
