"""Minimal training loop utilities (NanoGPT-inspired)."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence

import torch
from torch import nn

try:  # accelerate is optional for documentation builds
    from accelerate import Accelerator
except Exception:  # pragma: no cover - accelerate optional for import
    Accelerator = None  # type: ignore[misc]


@dataclass
class TrainingConfig:
    """Hyper-parameters that are agnostic to the experiment wiring."""

    max_steps: int = 1000
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    max_grad_norm: Optional[float] = 1.0
    log_every: int = 10
    compile_model: bool = False
    mixed_precision: str = "bf16"


@dataclass
class StepOutput:
    """Return type expected from a ``step_fn`` provided by the experiment."""

    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)
    tokens: Optional[int] = None
    samples: Optional[int] = None


StepFn = Callable[[nn.Module, MutableMapping[str, torch.Tensor]], StepOutput]


class TrainerCallback:
    """Extensible hooks so experiments can plug in logging/checkpointing."""

    def on_train_start(self, trainer: "Trainer", state: "TrainingState") -> None:  # pragma: no cover - optional override
        pass

    def on_step_end(
        self,
        trainer: "Trainer",
        state: "TrainingState",
        step_output: StepOutput,
    ) -> None:  # pragma: no cover - optional override
        pass

    def on_log(self, trainer: "Trainer", state: "TrainingState", metrics: Dict[str, float]) -> None:  # pragma: no cover
        pass

    def on_train_end(self, trainer: "Trainer", state: "TrainingState") -> None:  # pragma: no cover - optional override
        pass


class ConsoleMetricLogger(TrainerCallback):
    """Print aggregated metrics whenever the trainer emits a log event."""

    def on_log(self, trainer: "Trainer", state: "TrainingState", metrics: Dict[str, float]) -> None:
        if not trainer.is_main_process:
            return
        ordered = " | ".join(f"{key} {value:.4f}" for key, value in metrics.items() if isinstance(value, float))
        print(f"step {state.step:05d} | {ordered}")


@dataclass
class TrainingState:
    """Lightweight container describing the loop progress."""

    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    samples_seen: int = 0
    start_time: float = 0.0
    last_log_time: float = 0.0


@dataclass
class _Window:
    updates: int = 0
    loss: float = 0.0
    tokens: int = 0
    samples: int = 0
    scalars: Dict[str, float] = field(default_factory=dict)


def _init_accelerator(cfg: TrainingConfig) -> Accelerator:
    if Accelerator is None:  # pragma: no cover - accelerate optional for import
        raise ImportError("accelerate is required for the Trainer")
    return Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps, mixed_precision=cfg.mixed_precision)


def _warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / float(warmup_steps))


def _to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(value, device) for value in batch)
    return batch


class Trainer:
    """NanoGPT-style single-file training loop with pluggable experiment logic."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: Iterable[MutableMapping[str, torch.Tensor]],
        cfg: TrainingConfig,
        step_fn: StepFn,
        *,
        accelerator: Optional[Accelerator] = None,
        callbacks: Sequence[TrainerCallback] = (),
    ) -> None:
        self.cfg = cfg
        self.accelerator = accelerator or _init_accelerator(cfg)
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.step_fn = step_fn
        self.callbacks = list(callbacks)

        if cfg.compile_model:
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
        prepared = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        self.model, self.optimizer, self.dataloader = prepared

        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", group.get("lr", cfg.learning_rate))

        self.state = TrainingState()
        self.window = _Window()

    # ------------------------------------------------------------------ helpers
    @property
    def is_main_process(self) -> bool:
        return getattr(self.accelerator, "is_main_process", True)

    def _run_callbacks(self, hook: str, *args) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook, None)
            if fn is not None:
                fn(self, *args)

    def _next_batch(self, iterator: Iterator[MutableMapping[str, torch.Tensor]]) -> MutableMapping[str, torch.Tensor]:
        try:
            return next(iterator)
        except StopIteration:
            self.state.epoch += 1
            sampler = getattr(self.dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.state.epoch)
            iterator = iter(self.dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:  # pragma: no cover - empty dataloader is a user error
                raise RuntimeError("dataloader yielded no batches") from exc
            self._iterator = iterator
            return batch

    def _update_window(self, step_output: StepOutput, loss_value: float) -> None:
        self.window.updates += 1
        self.window.loss += loss_value
        if step_output.tokens is not None:
            self.window.tokens += int(step_output.tokens)
            self.state.tokens_seen += int(step_output.tokens)
        if step_output.samples is not None:
            self.window.samples += int(step_output.samples)
            self.state.samples_seen += int(step_output.samples)
        for key, value in step_output.metrics.items():
            self.window.scalars[key] = self.window.scalars.get(key, 0.0) + float(value)

    def _maybe_log(self, *, force: bool = False) -> None:
        if not force and self.state.step % self.cfg.log_every != 0 and self.state.step != 1:
            return
        now = perf_counter()
        elapsed = max(now - self.state.start_time, 1e-9)
        window_time = max(now - self.state.last_log_time, 1e-9)
        avg_loss = self.window.loss / max(1, self.window.updates)
        metrics: Dict[str, float] = {
            "loss": avg_loss,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "wall": elapsed,
        }
        if self.window.tokens > 0:
            metrics["tokens_per_sec"] = self.window.tokens / window_time
            metrics["tokens_total"] = float(self.state.tokens_seen)
        if self.window.samples > 0:
            metrics["samples_per_sec"] = self.window.samples / window_time
            metrics["samples_total"] = float(self.state.samples_seen)
        for key, value in self.window.scalars.items():
            metrics[key] = value / max(1, self.window.updates)
        self._run_callbacks("on_log", self.state, metrics)
        self.window = _Window()
        self.state.last_log_time = now

    # ------------------------------------------------------------------ main loop
    def train(self) -> TrainingState:
        self.state.start_time = perf_counter()
        self.state.last_log_time = self.state.start_time
        self._iterator = iter(self.dataloader)
        self._run_callbacks("on_train_start", self.state)

        while self.state.step < self.cfg.max_steps:
            batch = self._next_batch(self._iterator)
            batch = _to_device(batch, self.accelerator.device)

            with self.accelerator.accumulate(self.model):
                step_output = self.step_fn(self.model, batch)
                loss = step_output.loss / self.cfg.grad_accum_steps
                self.accelerator.backward(loss)

            gathered = self.accelerator.gather(step_output.loss.detach())
            loss_value = gathered.mean().item()
            self._update_window(step_output, loss_value)

            if self.accelerator.sync_gradients:
                self.state.step += 1
                if self.cfg.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm
                    )
                scale = _warmup_factor(self.state.step, self.cfg.warmup_steps)
                for group in self.optimizer.param_groups:
                    base_lr = group.get("base_lr", self.cfg.learning_rate)
                    group["lr"] = base_lr * scale
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self._run_callbacks("on_step_end", self.state, step_output)
                self._maybe_log()

        if self.window.updates:
            self._maybe_log(force=True)
        self.accelerator.wait_for_everyone()
        self._run_callbacks("on_train_end", self.state)
        return self.state


__all__ = [
    "TrainingConfig",
    "Trainer",
    "TrainerCallback",
    "ConsoleMetricLogger",
    "StepOutput",
    "StepFn",
    "TrainingState",
]
