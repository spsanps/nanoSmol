"""
Training logger: wandb + CSV + stdout + run summary JSON.

All logging is gated on ``enabled`` (typically ``is_main_process()``).
Wandb is optional -- if ``wandb`` is not installed or fails to init,
logging falls back to CSV + stdout silently.

CSV columns (one row per logged event):
  run_id, step, samples_seen, wall_time_sec, event_type,
  train_loss, loss_fine, loss_coarse, loss_ratio,
  grad_norm, lr_connector, lr_dino, lr_llm,
  throughput_samples_sec, gpu_mem_gb,
  val_loss, val_loss_fine, val_loss_coarse, val_loss_ratio,
  attention_entropy
"""

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _gpu_memory_gb() -> float:
    """Get current GPU memory allocated in GB, or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


CSV_COLUMNS = [
    "run_id", "step", "samples_seen", "wall_time_sec", "event_type",
    "train_loss", "loss_fine", "loss_coarse", "loss_ratio",
    "grad_norm", "lr_connector", "lr_dino", "lr_llm",
    "throughput_samples_sec", "gpu_mem_gb",
    "val_loss", "val_loss_fine", "val_loss_coarse", "val_loss_ratio",
    "attention_entropy",
]


class TrainingLogger:
    """
    Unified logger that writes to wandb, structured CSV, and stdout.

    Parameters
    ----------
    project : str
        wandb project name.
    config : dict
        Training config to log as wandb config / CSV header metadata.
    enabled : bool
        If False, all log calls are no-ops (use for non-rank-0 processes).
    log_dir : str
        Directory for the CSV log file.
    """

    def __init__(
        self,
        project: str = "foveated-vlm",
        config: Optional[dict] = None,
        enabled: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.enabled = enabled
        self._wandb_run = None
        self._csv_path = None
        self._csv_writer = None
        self._csv_file = None
        self._start_time = time.time()
        self._config = config or {}
        self._run_id = ""
        self._best_val_loss = float("inf")
        self._best_step = 0
        self._last_step = 0
        self._last_samples = 0
        self._git_hash = _get_git_hash()

        if not enabled:
            return

        # ---- Run ID ----
        run_name = (config or {}).get("wandb", {}).get("run_name", "run")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{run_name}_{timestamp}"

        # ---- wandb ----
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                resume="allow",
            )
        except Exception:
            pass

        # ---- CSV ----
        if log_dir is None:
            log_dir = (config or {}).get("checkpoint", {}).get(
                "save_dir", "/workspace/logs"
            )
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._log_dir / f"metrics_{self._run_id}.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore",
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def _write_csv_row(self, row: dict):
        if self._csv_writer is not None:
            row.setdefault("run_id", self._run_id)
            row.setdefault("wall_time_sec", f"{time.time() - self._start_time:.1f}")
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def log_step(
        self,
        step: int,
        loss: float,
        fine_loss: float = 0.0,
        coarse_loss: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        samples_seen: int = 0,
        samples_per_sec: float = 0.0,
        lr_groups: Optional[dict] = None,
    ):
        """Log a training step with full metrics."""
        if not self.enabled:
            return

        self._last_step = step
        self._last_samples = samples_seen

        loss_ratio = fine_loss / max(coarse_loss, 1e-8) if coarse_loss > 0 else 0.0
        gpu_mem = _gpu_memory_gb()

        # Parse per-group LRs
        lr_connector = lr
        lr_dino = lr
        lr_llm = lr
        if lr_groups:
            lr_connector = lr_groups.get("connector", lr)
            lr_dino = lr_groups.get("dino", lr)
            lr_llm = lr_groups.get("llm", lr)

        # stdout
        print(
            f"  step {step:6d} | loss {loss:.4f} | "
            f"fine {fine_loss:.4f} | ratio {loss_ratio:.3f} | "
            f"lr {lr:.2e} | gnorm {grad_norm:.2f} | "
            f"{samples_per_sec:.0f} samp/s | {gpu_mem:.1f}GB",
            flush=True,
        )

        # wandb
        if self._wandb_run is not None:
            try:
                import wandb
                log_dict = {
                    "train/loss": loss,
                    "train/fine_loss": fine_loss,
                    "train/coarse_loss": coarse_loss,
                    "train/loss_ratio": loss_ratio,
                    "train/lr": lr,
                    "train/lr_connector": lr_connector,
                    "train/lr_dino": lr_dino,
                    "train/lr_llm": lr_llm,
                    "train/grad_norm": grad_norm,
                    "train/samples_seen": samples_seen,
                    "train/throughput": samples_per_sec,
                    "train/gpu_mem_gb": gpu_mem,
                }
                wandb.log(log_dict, step=step)
            except Exception:
                pass

        # CSV
        self._write_csv_row({
            "step": step,
            "samples_seen": samples_seen,
            "event_type": "train",
            "train_loss": f"{loss:.6f}",
            "loss_fine": f"{fine_loss:.6f}",
            "loss_coarse": f"{coarse_loss:.6f}",
            "loss_ratio": f"{loss_ratio:.4f}",
            "grad_norm": f"{grad_norm:.4f}",
            "lr_connector": f"{lr_connector:.2e}",
            "lr_dino": f"{lr_dino:.2e}",
            "lr_llm": f"{lr_llm:.2e}",
            "throughput_samples_sec": f"{samples_per_sec:.1f}",
            "gpu_mem_gb": f"{gpu_mem:.2f}",
        })

    def log_eval(
        self,
        step: int,
        val_loss: float,
        val_fine_loss: float = 0.0,
        val_coarse_loss: float = 0.0,
        attention_entropy: float = 0.0,
    ):
        """Log a validation result with extended metrics."""
        if not self.enabled:
            return

        val_ratio = val_fine_loss / max(val_coarse_loss, 1e-8) if val_coarse_loss > 0 else 0.0

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_step = step

        print(
            f"  [eval] step {step:6d} | val_loss {val_loss:.4f} | "
            f"fine {val_fine_loss:.4f} | ratio {val_ratio:.3f} | "
            f"entropy {attention_entropy:.4f} | "
            f"best {self._best_val_loss:.4f}@{self._best_step}",
            flush=True,
        )

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "eval/val_loss": val_loss,
                    "eval/val_fine_loss": val_fine_loss,
                    "eval/val_coarse_loss": val_coarse_loss,
                    "eval/val_loss_ratio": val_ratio,
                    "eval/attention_entropy": attention_entropy,
                    "eval/best_val_loss": self._best_val_loss,
                }, step=step)
            except Exception:
                pass

        self._write_csv_row({
            "step": step,
            "samples_seen": self._last_samples,
            "event_type": "eval",
            "val_loss": f"{val_loss:.6f}",
            "val_loss_fine": f"{val_fine_loss:.6f}",
            "val_loss_coarse": f"{val_coarse_loss:.6f}",
            "val_loss_ratio": f"{val_ratio:.4f}",
            "attention_entropy": f"{attention_entropy:.6f}",
        })

    def save_run_summary(self, final_loss: float = 0.0, total_samples: int = 0):
        """Save run summary JSON at end of training."""
        if not self.enabled:
            return

        elapsed = time.time() - self._start_time
        summary = {
            "run_id": self._run_id,
            "git_hash": self._git_hash,
            "config_file": self._config.get("_config_path", ""),
            "final_train_loss": final_loss,
            "best_val_loss": self._best_val_loss,
            "best_val_step": self._best_step,
            "total_steps": self._last_step,
            "total_samples": total_samples,
            "wall_time_sec": elapsed,
            "wall_time_hours": elapsed / 3600,
            "csv_path": str(self._csv_path) if self._csv_path else "",
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = self._log_dir / f"run_summary_{self._run_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Run summary saved to {summary_path}", flush=True)

    def finish(self):
        """Flush and close all logging backends."""
        if not self.enabled:
            return

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
