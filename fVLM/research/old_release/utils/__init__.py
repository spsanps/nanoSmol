"""
Training utilities for foveated VLM.

- distributed: DDP process group setup and rank helpers
- checkpoint: Save/load training state with best-metric tracking
- lr_schedule: Cosine decay with linear warmup, converging schedule
- logging_utils: Unified wandb + CSV + stdout logging (rank-0 only)
- attention_viz: Attention entropy computation and heatmap saving
"""

from release.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)
from release.utils.checkpoint import (
    save_checkpoint,
    load_latest_checkpoint,
    cleanup_checkpoints,
)
from release.utils.lr_schedule import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_converging_schedule
from release.utils.logging_utils import TrainingLogger
from release.utils.attention_viz import compute_attention_entropy, save_attention_maps

__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "save_checkpoint",
    "load_latest_checkpoint",
    "cleanup_checkpoints",
    "get_cosine_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_converging_schedule",
    "TrainingLogger",
    "compute_attention_entropy",
    "save_attention_maps",
]
