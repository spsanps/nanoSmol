from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytest.importorskip("torch")
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.nanoeval.common import SimpleModel, set_seed  # noqa: E402


class _DummyEncoding(dict):
    def __init__(self, input_ids: list[int]):
        tensor = torch.tensor([input_ids], dtype=torch.long)
        super().__init__({"input_ids": tensor})
        self.input_ids = tensor

    def to(self, device: str):
        return self


class _DummyTextTokenizer:
    def __init__(self) -> None:
        self.mapping = {
            "prompt": [0],
            " correct": [1, 2],
            " wrong": [3, 4],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.mapping[text])


class _DummyTextModel:
    def __call__(self, *, input_ids: torch.Tensor):
        tokens = input_ids.tolist()[0]
        if tokens[1] == 1:  # option "correct"
            logits = torch.tensor(
                [[[5.0, 5.0, 5.0, 0.0, 0.0], [5.0, 5.0, 5.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
        else:  # option "wrong"
            logits = torch.tensor(
                [[[0.0, 0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 0.0, 5.0, 5.0]]],
                dtype=torch.float32,
            )
        return SimpleNamespace(logits=logits)


class _DummyProcessor:
    def apply_chat_template(self, messages, add_generation_prompt: bool = True) -> str:
        if add_generation_prompt:
            return "prompt"
        assistant = messages[-1]["content"][0]["text"]
        return f"prompt|{assistant}"

    def __call__(self, *, text: str, images, return_tensors: str = "pt"):
        mapping = {
            "prompt": [0, 5],
            "prompt|": [0, 5],
            "prompt| cat": [0, 5, 1, 2],
            "prompt| dog": [0, 5, 3, 4],
        }
        return _DummyEncoding(mapping[text])


class _DummyVLMModel:
    def __call__(self, *, input_ids: torch.Tensor):
        tokens = input_ids.tolist()[0]
        if tokens[2] == 1:  # option "cat"
            logits = torch.tensor(
                [[[5.0, 5.0, 5.0, 0.0, 0.0], [5.0, 5.0, 5.0, 0.0, 0.0], [5.0, 5.0, 5.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
        else:  # option "dog"
            logits = torch.tensor(
                [[[0.0, 0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 0.0, 5.0, 5.0]]],
                dtype=torch.float32,
            )
        return SimpleNamespace(logits=logits)


def _build_simple_text_model() -> SimpleModel:
    instance = SimpleModel.__new__(SimpleModel)
    instance.processor = None
    instance.tokenizer = _DummyTextTokenizer()
    instance.model = _DummyTextModel()
    instance.device = "cpu"
    return instance


def _build_simple_vlm_model() -> SimpleModel:
    instance = SimpleModel.__new__(SimpleModel)
    instance.processor = _DummyProcessor()
    instance.tokenizer = None
    instance.model = _DummyVLMModel()
    instance.device = "cpu"
    return instance


def test_rank_log_likelihood_prefers_high_prob_text_option():
    model = _build_simple_text_model()
    result = model.rank_log_likelihood("prompt", ["correct", "wrong"])
    assert result == 0


def test_rank_log_likelihood_multimodal_prefers_high_prob_option():
    model = _build_simple_vlm_model()
    messages = [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]
    result = model.rank_log_likelihood_multimodal(
        messages, images=(), options=["cat", "dog"]
    )
    assert result == 0


def test_set_seed_cpu_only(monkeypatch):
    calls = {"cuda": False}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)

    def fake_manual_seed_all(seed):  # pragma: no cover - defensive
        calls["cuda"] = True

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fake_manual_seed_all)

    set_seed(123)
    assert calls["cuda"] is False
