"""
Training logger: wandb + CSV + stdout.

All logging is gated on ``enabled`` (typically ``is_main_process()``).
Wandb is optional -- if ``wandb`` is not installed or fails to init,
logging falls back to CSV + stdout silently.
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class TrainingLogger:
    """
    Unified logger that writes to wandb, a CSV file, and stdout.

    Parameters
    ----------
    project : str
        wandb project name (also used as the CSV filename prefix).
    config : dict
        Training config to log as wandb config / CSV header metadata.
    enabled : bool
        If False, all log calls are no-ops (use for non-rank-0 processes).
    log_dir : str
        Directory for the CSV log file.  Defaults to the checkpoint dir
        from config, or ``/workspace/logs``.
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

        if not enabled:
            return

        # ---- wandb ----
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project,
                config=config or {},
                resume="allow",
            )
        except Exception:
            # wandb not installed or init failed -- degrade gracefully
            pass

        # ---- CSV ----
        if log_dir is None:
            log_dir = (config or {}).get("checkpoint", {}).get(
                "save_dir", "/workspace/logs"
            )
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = log_dir / f"train_log_{timestamp}.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "step", "loss", "fine_loss", "coarse_loss", "lr",
            "grad_norm", "samples_seen", "samples_per_sec",
            "val_loss", "elapsed_sec",
        ])
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
    ):
        """Log a training step."""
        if not self.enabled:
            return

        elapsed = time.time() - self._start_time

        # stdout
        print(
            f"  step {step:6d} | loss {loss:.4f} | "
            f"fine {fine_loss:.4f} | lr {lr:.2e} | "
            f"gnorm {grad_norm:.2f} | {samples_per_sec:.0f} samp/s",
            flush=True,
        )

        # wandb
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "train/loss": loss,
                    "train/fine_loss": fine_loss,
                    "train/coarse_loss": coarse_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/samples_seen": samples_seen,
                    "train/samples_per_sec": samples_per_sec,
                }, step=step)
            except Exception:
                pass

        # CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                step, f"{loss:.6f}", f"{fine_loss:.6f}", f"{coarse_loss:.6f}",
                f"{lr:.2e}", f"{grad_norm:.4f}", samples_seen,
                f"{samples_per_sec:.1f}", "", f"{elapsed:.1f}",
            ])
            self._csv_file.flush()

    def log_eval(self, step: int, val_loss: float):
        """Log a validation result."""
        if not self.enabled:
            return

        elapsed = time.time() - self._start_time

        print(f"  [eval] step {step:6d} | val_loss {val_loss:.4f}", flush=True)

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log({"eval/val_loss": val_loss}, step=step)
            except Exception:
                pass

        if self._csv_writer is not None:
            self._csv_writer.writerow([
                step, "", "", "", "", "", "", "", f"{val_loss:.6f}",
                f"{elapsed:.1f}",
            ])
            self._csv_file.flush()

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
