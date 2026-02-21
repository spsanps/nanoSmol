"""
Checkpoint management for foveated VLM training.

Saves full training state (model, optimizer, scaler, scheduler, step, and
data position) so that runs can be resumed exactly where they left off after
preemption.  Keeps the last N checkpoints plus the single best checkpoint by
a tracked metric (e.g. validation loss).

File layout inside ``save_dir``::

    checkpoints/
        step_001000.pt
        step_002000.pt
        best.pt        -> symlink to the best checkpoint
        latest.pt      -> symlink to the most recent checkpoint

Each ``.pt`` file is a dict with keys:
    model_state_dict, optimizer_state_dict, scaler_state_dict,
    scheduler_state_dict, step, data_position, metric_value,
    config (optional), timestamp.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

from release.utils.distributed import is_main_process


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scaler,
    scheduler,
    step: int,
    data_position: int,
    save_dir: str,
    metric_value: Optional[float] = None,
    config: Optional[dict] = None,
):
    """
    Save a full training checkpoint.

    Only rank-0 writes to disk.  Other ranks return immediately so the caller
    does not need to guard this with ``is_main_process()``.

    Parameters
    ----------
    model : nn.Module or DDP-wrapped module
        The model whose ``state_dict()`` will be saved.  If wrapped in
        ``DistributedDataParallel`` the underlying ``module`` is accessed
        automatically.
    optimizer : torch.optim.Optimizer
    scaler : torch.amp.GradScaler
    scheduler : torch.optim.lr_scheduler._LRScheduler
    step : int
        Global training step (number of optimizer updates).
    data_position : int
        Number of *samples* consumed so far (across all epochs).  Used to
        fast-forward the data pipeline on resume.
    save_dir : str or Path
        Directory where checkpoint files are written.
    metric_value : float, optional
        Tracked metric value (e.g. eval loss).  When provided, the checkpoint
        is compared against the current best and ``best.pt`` is updated if
        this value is lower.
    config : dict, optional
        Training configuration dict.  Stored inside the checkpoint for
        provenance tracking.
    """
    if not is_main_process():
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed.
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "data_position": data_position,
        "metric_value": metric_value,
        "timestamp": datetime.now().isoformat(),
    }
    if config is not None:
        checkpoint["config"] = config

    # Write to a step-numbered file.
    ckpt_path = save_dir / f"step_{step:07d}.pt"
    torch.save(checkpoint, ckpt_path)

    # Update ``latest.pt`` symlink.
    _update_symlink(save_dir / "latest.pt", ckpt_path)

    # Update ``best.pt`` if this checkpoint beats the current best.
    if metric_value is not None:
        _maybe_update_best(save_dir, ckpt_path, metric_value)

    # Prune old checkpoints (default: keep last 2 + best).
    cleanup_checkpoints(save_dir, keep_last=2, keep_best=1)

    print(
        f"  [{datetime.now().strftime('%H:%M:%S')}] "
        f"Checkpoint saved: step {step}"
        + (f", metric={metric_value:.4f}" if metric_value is not None else ""),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Load / resume
# ---------------------------------------------------------------------------

def load_latest_checkpoint(
    save_dir: str,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    map_location: Optional[str] = None,
):
    """
    Auto-resume from the latest checkpoint in ``save_dir``.

    Parameters
    ----------
    save_dir : str or Path
    model : nn.Module or DDP-wrapped module
    optimizer, scaler, scheduler : optional
        If provided, their ``state_dict`` is restored.  Pass ``None`` to skip
        (useful for evaluation-only loads).
    map_location : str, optional
        Device to map tensors to (e.g. ``"cuda:0"``).  Defaults to
        ``"cpu"`` for safety; the caller moves the model after.

    Returns
    -------
    dict or None
        ``{"step": int, "data_position": int, "metric_value": float|None}``
        if a checkpoint was found, otherwise ``None``.
    """
    save_dir = Path(save_dir)

    # Try latest.pt symlink first, then fall back to the highest-numbered file.
    ckpt_path = save_dir / "latest.pt"
    if not ckpt_path.exists():
        ckpt_path = _find_latest_numbered(save_dir)

    if ckpt_path is None or not ckpt_path.exists():
        return None

    if map_location is None:
        map_location = "cpu"

    print(f"  Resuming from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Model
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model_state_dict"])

    # Optimizer
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Scaler
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Scheduler
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = ckpt.get("step", 0)
    data_position = ckpt.get("data_position", 0)
    metric_value = ckpt.get("metric_value", None)

    print(
        f"  Resumed at step {step}, data_position {data_position}"
        + (f", metric={metric_value:.4f}" if metric_value is not None else "")
    )

    return {
        "step": step,
        "data_position": data_position,
        "metric_value": metric_value,
    }


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_checkpoints(save_dir, keep_last: int = 2, keep_best: int = 1):
    """
    Remove old checkpoints, keeping the last ``keep_last`` numbered files plus
    the best checkpoint.

    Parameters
    ----------
    save_dir : str or Path
    keep_last : int
        Number of most-recent step_*.pt files to keep.
    keep_best : int
        If >= 1, the ``best.pt`` target is also kept (even if it is not among
        the last N).
    """
    if not is_main_process():
        return

    save_dir = Path(save_dir)
    if not save_dir.exists():
        return

    # Find all numbered checkpoint files.
    numbered = sorted(
        save_dir.glob("step_*.pt"),
        key=_step_from_path,
    )

    if len(numbered) <= keep_last:
        return  # Nothing to prune.

    # Identify protected paths.
    protected = set()

    # Protect the last N.
    for p in numbered[-keep_last:]:
        protected.add(p.resolve())

    # Protect the best checkpoint target.
    if keep_best >= 1:
        best_link = save_dir / "best.pt"
        if best_link.exists():
            target = best_link.resolve() if best_link.is_symlink() else best_link
            protected.add(target)

    # Delete the rest.
    for p in numbered:
        if p.resolve() not in protected:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def _step_from_path(path: Path) -> int:
    """Extract the step number from a checkpoint filename."""
    m = _STEP_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _find_latest_numbered(save_dir: Path) -> Optional[Path]:
    """Return the step_*.pt file with the highest step number, or None."""
    candidates = sorted(save_dir.glob("step_*.pt"), key=_step_from_path)
    return candidates[-1] if candidates else None


def _update_symlink(link_path: Path, target_path: Path):
    """Create or update a symlink atomically."""
    # Use a temp link then rename (atomic on POSIX).
    tmp = link_path.with_suffix(".tmp")
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(target_path.resolve())
        tmp.rename(link_path)
    except OSError:
        # Fallback: just copy the file (Windows or weird fs).
        import shutil
        shutil.copy2(str(target_path), str(link_path))


def _maybe_update_best(save_dir: Path, ckpt_path: Path, metric_value: float):
    """
    Update ``best.pt`` if ``metric_value`` is lower than the current best.

    The current best metric is stored in a small JSON sidecar file
    ``best_metric.json`` next to the checkpoint directory.
    """
    meta_path = save_dir / "best_metric.json"
    current_best = float("inf")

    if meta_path.exists():
        try:
            with open(meta_path) as f:
                current_best = json.load(f).get("metric_value", float("inf"))
        except (json.JSONDecodeError, KeyError):
            pass

    if metric_value < current_best:
        _update_symlink(save_dir / "best.pt", ckpt_path)
        with open(meta_path, "w") as f:
            json.dump(
                {"metric_value": metric_value, "step": _step_from_path(ckpt_path)},
                f,
            )
