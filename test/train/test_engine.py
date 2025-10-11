import contextlib

import pytest

pytest.importorskip("torch")

import torch
from torch import nn
from torch.utils.data import DataLoader

from train.engine import ConsoleMetricLogger, StepOutput, Trainer, TrainerCallback, TrainingConfig


class DummyAccelerator:
    def __init__(self, grad_accum_steps: int = 1) -> None:
        self.grad_accum_steps = grad_accum_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main_process = True
        self.sync_gradients = True

    def prepare(self, model, optimizer, dataloader):
        return model.to(self.device), optimizer, dataloader

    def accumulate(self, model):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def gather(self, tensor):
        return tensor.detach()

    def wait_for_everyone(self):
        pass


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int):
        return {"x": self.inputs[index], "y": self.targets[index]}


def build_toy_step_fn():
    def step_fn(model: nn.Module, batch):
        preds = model(batch["x"])
        loss = torch.nn.functional.mse_loss(preds, batch["y"])
        return StepOutput(loss=loss, samples=batch["x"].size(0), metrics={"mse": float(loss.detach())})

    return step_fn


class Recorder(TrainerCallback):
    def __init__(self) -> None:
        self.logged_steps = []
        self.final_step = None

    def on_log(self, trainer, state, metrics):
        self.logged_steps.append((state.step, metrics["loss"]))

    def on_train_end(self, trainer, state):
        self.final_step = state.step


def test_trainer_runs_and_logs(tmp_path) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 2)
    dataset = TinyDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step_fn = build_toy_step_fn()

    recorder = Recorder()
    cfg = TrainingConfig(max_steps=5, log_every=1, grad_accum_steps=1, learning_rate=1e-3, warmup_steps=2)
    trainer = Trainer(
        model,
        optimizer,
        dataloader,
        cfg,
        step_fn,
        callbacks=[recorder, ConsoleMetricLogger()],
        accelerator=DummyAccelerator(),
    )
    state = trainer.train()

    assert state.step == cfg.max_steps
    assert recorder.final_step == cfg.max_steps
    assert len(recorder.logged_steps) >= 1


def test_trainer_warmup_adjusts_learning_rate() -> None:
    torch.manual_seed(0)
    dataset = TinyDataset(torch.randn(4, 3), torch.randn(4, 1))
    dataloader = DataLoader(dataset, batch_size=2)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    cfg = TrainingConfig(max_steps=3, warmup_steps=2, learning_rate=1.0, grad_accum_steps=1)
    trainer = Trainer(
        model,
        optimizer,
        dataloader,
        cfg,
        build_toy_step_fn(),
        accelerator=DummyAccelerator(),
    )
    trainer.train()

    lrs = [group["lr"] for group in optimizer.param_groups]
    assert all(abs(lr - cfg.learning_rate) < 1e-5 for lr in lrs)
