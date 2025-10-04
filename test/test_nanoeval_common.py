"""Unit tests covering shared NanoEval helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from eval.nanoeval.common import SimpleModel, set_seed
from eval.nanoeval.config import MMMUProRunConfig, ModelConfig, ScoringConfig
from eval.nanoeval.run_mmmu_pro import run


class _DummyEncoding(dict):
    def __init__(self, input_ids: list[int]):
        tensor = torch.tensor([input_ids], dtype=torch.long)
        super().__init__({"input_ids": tensor})
        self.input_ids = tensor

    def to(self, device: str):
        return self


class _DummyTokenizer:
    def __init__(self, prompt_ids: list[int], vocab: dict[int, str]):
        self.prompt_ids = prompt_ids
        self.vocab = vocab
        self.pad_token = None
        self.eos_token = "<eos>"

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        return _DummyEncoding(self.prompt_ids)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self.vocab.get(token, "") for token in token_ids)


class _DummyProcessor:
    def __init__(self, prompt_ids: list[int], vocab: dict[int, str]):
        self.prompt_ids = prompt_ids
        self.vocab = vocab

    def apply_chat_template(self, messages, add_generation_prompt: bool = True) -> str:
        return "dummy"

    def __call__(self, *, text: str, images, return_tensors: str = "pt"):
        return _DummyEncoding(self.prompt_ids)

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        results = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            results.append("".join(self.vocab.get(token, "") for token in seq))
        return results


class _DummyModel:
    def __init__(self, output_ids: list[int]):
        self.output = torch.tensor([output_ids], dtype=torch.long)

    def generate(self, **_: dict):
        return self.output


def _build_simple_model_for_text() -> SimpleModel:
    instance = SimpleModel.__new__(SimpleModel)
    vocab = {10: "A", 11: ". ", 12: "Option ", 20: "C"}
    instance.processor = None
    instance.tokenizer = _DummyTokenizer([10, 11, 12], vocab)
    instance.model = _DummyModel([10, 11, 12, 20])
    instance.device = "cpu"
    return instance


def _build_simple_model_for_vlm() -> SimpleModel:
    instance = SimpleModel.__new__(SimpleModel)
    vocab = {30: "A", 31: ". ", 32: "Picture ", 40: "D"}
    instance.processor = _DummyProcessor([30, 31, 32], vocab)
    instance.tokenizer = None
    instance.model = _DummyModel([30, 31, 32, 40])
    instance.device = "cpu"
    return instance


def test_generate_letter_text_ignores_prompt_letters():
    model = _build_simple_model_for_text()
    result = model.generate_letter_text("prompt", ["A", "B", "C"], max_new_tokens=1)
    assert result == "C"


def test_generate_letter_vlm_ignores_prompt_letters():
    model = _build_simple_model_for_vlm()
    messages = [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]
    result = model.generate_letter_vlm(messages, images=(), allowed_letters=["A", "B", "C", "D"], max_new_tokens=1)
    assert result == "D"


def test_set_seed_cpu_only(monkeypatch):
    calls = {"cuda": False}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)

    def fake_manual_seed_all(seed):  # pragma: no cover - defensive
        calls["cuda"] = True

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fake_manual_seed_all)

    set_seed(123)
    assert calls["cuda"] is False


def test_mmmu_pro_rejects_rank_ll_strategy(monkeypatch):
    monkeypatch.setattr("eval.nanoeval.run_mmmu_pro.load_dataset", lambda *args, **kwargs: None)
    config = MMMUProRunConfig(
        task="mmmu_pro",
        model=ModelConfig(model_id="dummy", is_vlm=True),
        scoring=ScoringConfig(strategy="rank_ll"),
    )
    with pytest.raises(ValueError):
        run(config)
