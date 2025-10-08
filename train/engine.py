"""Accelerate-powered trainer with NanoGPT-style clarity."""
from __future__ import annotations

import json
import math
import os
import itertools
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:  # Accelerate is optional for unit tests
    from accelerate import Accelerator
except Exception:  # pragma: no cover - optional import for tests
    Accelerator = None  # type: ignore

try:  # W&B optional dependency
    import wandb
except Exception:  # pragma: no cover - optional import for tests
    wandb = None  # type: ignore


@dataclass
class TrainingConfig:
    """Hyper-parameters shared across datasets/models."""

    max_steps: int = 1000
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    betas: Sequence[float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_grad_norm: Optional[float] = 1.0
    compile_model: bool = False
    log_every: int = 10
    log_dir: Optional[str] = "artifacts/train"
    mixed_precision: str = "bf16"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    resume_wandb: bool = False
    track_tokens: bool = True
    track_samples: bool = True
    grad_clip_eps: float = 1e-3


def _init_accelerator(cfg: TrainingConfig) -> Accelerator:
    if Accelerator is None:  # pragma: no cover - accelerate optional for import
        raise ImportError("accelerate is required for multi-GPU training")
    return Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps, mixed_precision=cfg.mixed_precision)


class TrainingLogger:
    """Collect scalars, print nicely, and optionally talk to W&B."""

    def __init__(self, cfg: TrainingConfig, accelerator: Accelerator) -> None:
        self.cfg = cfg
        self.accelerator = accelerator
        self.history: List[Dict[str, float]] = []
        self.log_dir: Optional[Path] = Path(cfg.log_dir) if cfg.log_dir else None
        self._wandb_run = None
        if self.log_dir and accelerator.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
        if (
            accelerator.is_main_process
            and cfg.wandb_project
            and wandb is not None
        ):  # pragma: no cover - wandb optional
            self._wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                group=cfg.wandb_group,
                resume="allow" if cfg.resume_wandb else False,
            )

    def log(self, step: int, scalars: Dict[str, float]) -> None:
        if self.accelerator.is_main_process:
            message = " | ".join([f"{key} {value:.4f}" for key, value in scalars.items() if not math.isnan(value)])
            print(f"step {step:05d} | {message}")
            record = {"step": float(step)}
            record.update({k: float(v) for k, v in scalars.items()})
            self.history.append(record)
            if self._wandb_run is not None:  # pragma: no cover - wandb optional
                wandb.log({"step": step, **scalars})

    def finalize(self) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.log_dir:
            log_json = self.log_dir / "training_history.json"
            with log_json.open("w", encoding="utf-8") as handle:
                json.dump(self.history, handle, indent=2)
            try:  # pragma: no cover - plotting is optional for tests
                import matplotlib.pyplot as plt

                if self.history:
                    steps = [entry["step"] for entry in self.history]
                    losses = [entry.get("loss", float("nan")) for entry in self.history]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(steps, losses, marker="o", linewidth=1.5)
                    ax.set_xlabel("step")
                    ax.set_ylabel("loss")
                    ax.set_title("training loss")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(self.log_dir / "training_curve.png", dpi=150)
                    plt.close(fig)
            except Exception:
                pass
        if self._wandb_run is not None:  # pragma: no cover - wandb optional
            self._wandb_run.finish()


class Trainer:
    """Single-file training loop built around ``accelerate``."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        cfg: TrainingConfig,
        *,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        self.cfg = cfg
        self.accelerator = accelerator or _init_accelerator(cfg)
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        if cfg.compile_model:
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.logger = TrainingLogger(cfg, self.accelerator)

    def _warmup_factor(self, step: int) -> float:
        if self.cfg.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / float(self.cfg.warmup_steps))

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.accelerator.device, non_blocking=True) for key, value in batch.items()}

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
        )
        vocab = logits.size(-1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def train(self) -> None:
        data_iterator = itertools.cycle(self.dataloader)
        start_time = perf_counter()
        last_log_time = start_time
        total_tokens = 0
        total_samples = 0
        step = 0
        running_tokens = 0
        running_samples = 0
        running_loss = 0.0
        window_updates = 0

        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", group.get("lr", self.cfg.lr))

        while step < self.cfg.max_steps:
            batch = next(data_iterator)
            batch = self._prepare_batch(batch)

            attention = batch.get("attention_mask")
            if attention is None:
                attention = torch.ones_like(batch["input_ids"])
            tokens_in_batch = int(attention.sum().item())
            samples_in_batch = int(batch["input_ids"].size(0))

            with self.accelerator.accumulate(self.model):
                loss = self._forward(batch) / self.cfg.grad_accum_steps
                self.accelerator.backward(loss)

            loss_value = self.accelerator.gather(loss.detach()).mean().item()
            running_loss += loss_value
            running_tokens += tokens_in_batch
            running_samples += samples_in_batch

            if self.accelerator.sync_gradients:
                step += 1
                window_updates += 1
                if self.cfg.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm + self.cfg.grad_clip_eps)

                lr_scale = self._warmup_factor(step)
                for group in self.optimizer.param_groups:
                    base_lr = group.get("base_lr", self.cfg.lr)
                    group["lr"] = base_lr * lr_scale

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_tokens += running_tokens
                total_samples += running_samples

                if step % self.cfg.log_every == 0 or step == 1:
                    now = perf_counter()
                    elapsed = max(now - start_time, 1e-6)
                    window = max(now - last_log_time, 1e-6)
                    avg_loss = running_loss / max(1, window_updates)
                    metrics = {
                        "loss": avg_loss,
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "tokens_per_sec": (running_tokens / window) if self.cfg.track_tokens else float("nan"),
                        "samples_per_sec": (running_samples / window) if self.cfg.track_samples else float("nan"),
                        "tokens_total": float(total_tokens),
                        "samples_total": float(total_samples),
                        "wall": elapsed,
                    }
                    self.logger.log(step, metrics)
                    running_loss = 0.0
                    running_tokens = 0
                    running_samples = 0
                    window_updates = 0
                    last_log_time = now

        self.accelerator.wait_for_everyone()
        self.logger.finalize()

