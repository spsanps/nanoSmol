from __future__ import annotations

import pytest

pytest.importorskip("torch")

from torch import nn

from train.engine import TrainingConfig, Trainer, _resolve_checkpoint_interval


def test_resolve_checkpoint_interval_uses_explicit_override() -> None:
    cfg = TrainingConfig(checkpoint_interval=5, num_checkpoints=99, max_steps=100)
    assert _resolve_checkpoint_interval(cfg) == 5


def test_resolve_checkpoint_interval_derives_from_count() -> None:
    cfg = TrainingConfig(checkpoint_interval=None, num_checkpoints=10, max_steps=1000)
    assert _resolve_checkpoint_interval(cfg) == 100


def test_resolve_checkpoint_interval_handles_disabled() -> None:
    cfg = TrainingConfig(checkpoint_interval=None, num_checkpoints=0, max_steps=1000)
    assert _resolve_checkpoint_interval(cfg) is None


def test_save_final_model_falls_back_to_state_dict(tmp_path) -> None:
    class DummyAccelerator:
        is_main_process = True

        def unwrap_model(self, model: nn.Module) -> nn.Module:
            return model

        def print(self, *args, **kwargs) -> None:
            pass

        def wait_for_everyone(self) -> None:
            pass

    class TinyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainingConfig(final_model_dir=str(tmp_path / "final"))
    trainer.accelerator = DummyAccelerator()
    trainer.model = TinyModule()
    trainer.tokenizer = None

    output_dir = trainer._save_final_model()
    expected_path = tmp_path / "final" / "pytorch_model.bin"
    assert output_dir == tmp_path / "final"
    assert expected_path.exists(), "Fallback torch.save should create pytorch_model.bin"
