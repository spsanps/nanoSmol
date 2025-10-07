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
        self.reverse = {1: "c", 2: "orrect", 3: "w", 4: "rong"}
        self.pad_token_id = 0
        self.eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.mapping[text])

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.reverse.get(token, "") for token in tokens)


class _DummyTextModel:
    def __call__(self, *, input_ids: torch.Tensor):
        tokens = input_ids.tolist()[0]
        seq_len = len(tokens)
        vocab_size = max(max(tokens) + 1, 6)
        logits = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32)
        if tokens[1] == 1:  # option "correct"
            weight = 5.0
        else:  # option "wrong"
            weight = -5.0
        for index, token in enumerate(tokens[1:], start=0):
            logits[0, index, token] = weight
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        suffix = torch.tensor([[1, 2]], dtype=torch.long)
        return torch.cat([input_ids, suffix[:, : max_new_tokens]], dim=1)


class _DummyProcessor:
    def __init__(self) -> None:
        self.last_messages = None
        self.last_add_generation_prompt = None
        self.last_images = None

    def apply_chat_template(self, messages, add_generation_prompt: bool = True) -> str:
        self.last_messages = messages
        self.last_add_generation_prompt = add_generation_prompt
        if add_generation_prompt:
            return "prompt"
        assistant = messages[-1]["content"][0]["text"]
        return f"prompt|{assistant}"

    def __call__(self, *, text: str, images, return_tensors: str = "pt"):
        self.last_images = images
        mapping = {
            "prompt": [0, 5],
            "prompt|": [0, 5],
            "prompt| cat": [0, 5, 1, 2],
            "prompt| dog": [0, 5, 3, 4],
        }
        return _DummyEncoding(mapping[text])

    @property
    def tokenizer(self) -> _DummyTextTokenizer:  # type: ignore[override]
        return _DummyTextTokenizer()


class _DummyVLMModel:
    def __call__(self, *, input_ids: torch.Tensor):
        tokens = input_ids.tolist()[0]
        seq_len = len(tokens)
        vocab_size = max(max(tokens) + 1, 6)
        logits = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32)
        if tokens[2] == 1:  # option "cat"
            weight = 5.0
        else:  # option "dog"
            weight = -5.0
        for index, token in enumerate(tokens[1:], start=0):
            logits[0, index, token] = weight
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        suffix = torch.tensor([[1, 2]], dtype=torch.long)
        return torch.cat([input_ids, suffix[:, : max_new_tokens]], dim=1)


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


def test_generate_text_for_text_model_uses_greedy_suffix():
    model = _build_simple_text_model()
    result = model.generate_text("prompt", max_new_tokens=2)
    assert result == "correct"


def test_generate_text_for_vlm_accepts_string_prompt():
    model = _build_simple_vlm_model()
    dummy_image = object()
    result = model.generate_text("prompt", images=[dummy_image], max_new_tokens=1)
    assert result == "c"
    assert model.processor.last_images == [dummy_image]
    assert model.processor.last_messages is not None
    content = model.processor.last_messages[0]["content"]
    image_placeholders = [item for item in content if item.get("type") == "image"]
    assert len(image_placeholders) == 1


def test_set_seed_cpu_only(monkeypatch):
    calls = {"cuda": False}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)

    def fake_manual_seed_all(seed):  # pragma: no cover - defensive
        calls["cuda"] = True

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fake_manual_seed_all)

    set_seed(123)
    assert calls["cuda"] is False
