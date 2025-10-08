from __future__ import annotations

import pytest

pytest.importorskip("torch")

from train.engine import TrainingConfig, _resolve_checkpoint_interval


def test_resolve_checkpoint_interval_uses_explicit_override() -> None:
    cfg = TrainingConfig(checkpoint_interval=5, num_checkpoints=99, max_steps=100)
    assert _resolve_checkpoint_interval(cfg) == 5


def test_resolve_checkpoint_interval_derives_from_count() -> None:
    cfg = TrainingConfig(checkpoint_interval=None, num_checkpoints=10, max_steps=1000)
    assert _resolve_checkpoint_interval(cfg) == 100


def test_resolve_checkpoint_interval_handles_disabled() -> None:
    cfg = TrainingConfig(checkpoint_interval=None, num_checkpoints=0, max_steps=1000)
    assert _resolve_checkpoint_interval(cfg) is None
