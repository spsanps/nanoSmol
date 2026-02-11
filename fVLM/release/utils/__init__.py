"""
Release utility modules for foveated VLM training.

Components:
- distributed: DDP process group setup and rank helpers
- checkpoint: Save/load/cleanup training checkpoints with best-metric tracking
- lr_schedule: Cosine decay with linear warmup
- logging_utils: Unified wandb + CSV + stdout logging (rank-0 only)
- flop_counter: Per-sample FLOP estimation for throughput reporting
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
from release.utils.lr_schedule import get_cosine_schedule_with_warmup
from release.utils.logging_utils import TrainingLogger
from release.utils.flop_counter import estimate_flops_per_sample

__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "save_checkpoint",
    "load_latest_checkpoint",
    "cleanup_checkpoints",
    "get_cosine_schedule_with_warmup",
    "TrainingLogger",
    "estimate_flops_per_sample",
]
