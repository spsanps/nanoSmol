from __future__ import annotations

from typing import Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from train.data import (
    ConversationTokenizer,
    FineVisionCollator,
    FineVisionDataConfig,
    _build_image_transform,
    _convert_turns_to_messages,
    _passes_quality_filter,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self._vocab: Dict[str, int] = {
            "<|user|>:": 3,
            "<|assistant|>:": 4,
            "": 5,
            "\n": 6,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        tokens = []
        for chunk in text.split(" "):
            if chunk not in self._vocab:
                self._vocab[chunk] = len(self._vocab) + 1
            tokens.append(self._vocab[chunk])
        return tokens


def test_convert_turns_to_messages_attaches_images() -> None:
    record = {
        "images": [Image.new("RGB", (8, 8), color=1), Image.new("RGB", (8, 8), color=2)],
        "texts": [
            {"user": "What do you see?", "assistant": "A chart."},
            {"user": "Describe it", "assistant": "It trends up."},
        ],
    }
    sample = _convert_turns_to_messages(record, max_images=1, max_turns=None)
    assert sample is not None
    first_user = sample["messages"][0]
    assert first_user["role"] == "user"
    assert first_user["content"][0]["type"] == "image"
    assert sample["messages"][1]["role"] == "assistant"


def test_quality_filter_uses_minimum_score() -> None:
    record = {"relevance_ratings": [5, 4, 3]}
    assert not _passes_quality_filter(record, min_quality=4)
    assert _passes_quality_filter(record, min_quality=2)


def test_collator_batches_and_masks_user_tokens() -> None:
    tokenizer = DummyTokenizer()
    cfg = FineVisionDataConfig(max_images=2, image_size=16)
    transform = _build_image_transform(cfg)
    wrapper = ConversationTokenizer(
        tokenizer,
        image_token_id=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    collator = FineVisionCollator(
        wrapper,
        image_transform=transform,
        max_images=cfg.max_images,
        image_size=cfg.image_size,
    )
    record = {
        "images": [Image.new("RGB", (8, 8), color=1)],
        "texts": [{"user": "Hello", "assistant": "World"}],
    }
    sample = _convert_turns_to_messages(record, max_images=cfg.max_images, max_turns=None)
    batch = collator([sample, sample])

    assert batch["input_ids"].shape[0] == 2
    assert batch["pixel_values"].shape == (2, cfg.max_images, 3, cfg.image_size, cfg.image_size)
    assert batch["pixel_attention_mask"].dtype == torch.bool
    # user tokens should be ignored by the loss mask
    labels = batch["labels"]
    assert (labels == -100).any()
    assert (labels != -100).any()
