"""Dataset adapters: turn raw records into chat message transcripts."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Type

from PIL import Image

from .config import ChatDataConfig


ChatRecord = Dict[str, object]
Message = Dict[str, object]


def _convert_turns_to_messages(
    record: ChatRecord,
    *,
    max_images: int,
    max_turns: Optional[int],
) -> Optional[Dict[str, object]]:
    """Convert FineVision style QA pairs into a NanoGPT-style chat transcript."""

    images = list(record.get("images", []))[:max_images]
    if not images:
        return None

    qa_pairs = list(record.get("texts", []))
    if max_turns is not None:
        qa_pairs = qa_pairs[:max_turns]
    if not qa_pairs:
        return None

    messages: List[Message] = []
    first_user_message = True
    for pair in qa_pairs:
        question = pair.get("user") or pair.get("question") or ""
        answer = pair.get("assistant") or pair.get("answer") or ""
        if not question or not answer:
            continue

        user_chunks: List[Dict[str, object]] = []
        if first_user_message:
            for image in images:
                if isinstance(image, Image.Image):
                    user_chunks.append({"type": "image", "image": image})
            first_user_message = False
        user_chunks.append({"type": "text", "text": str(question).strip()})
        messages.append({"role": "user", "content": user_chunks})

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer).strip()}],
            }
        )

    if not messages:
        return None

    return {
        "images": images,
        "messages": messages,
        "source": record.get("source"),
    }


def _passes_quality_filter(record: ChatRecord, min_quality: Optional[int]) -> bool:
    """Drop samples with low quality ratings (if the dataset provides them)."""

    if min_quality is None:
        return True
    ratings = record.get("relevance_ratings")
    if ratings is None:
        return True
    try:
        scores = [int(score) for score in ratings]
    except Exception:
        return True
    return min(scores) >= min_quality if scores else True


class ChatAdapter:
    """Base class for dataset-specific hooks."""

    name: str = "base"

    def __init__(self, cfg: ChatDataConfig) -> None:
        self.cfg = cfg

    def filter(self, record: ChatRecord) -> bool:  # pragma: no cover - override in subclasses
        return True

    def convert(self, record: ChatRecord) -> Optional[Dict[str, object]]:
        raise NotImplementedError


class FineVisionAdapter(ChatAdapter):
    """Adapter implementing FineVision's multi-turn structure."""

    name = "finevision"

    def filter(self, record: ChatRecord) -> bool:
        return _passes_quality_filter(record, self.cfg.min_quality)

    def convert(self, record: ChatRecord) -> Optional[Dict[str, object]]:
        return _convert_turns_to_messages(
            record,
            max_images=self.cfg.max_images,
            max_turns=self.cfg.max_turns,
        )


_ADAPTERS: Dict[str, Type[ChatAdapter]] = {FineVisionAdapter.name: FineVisionAdapter}


def register_adapter(adapter_cls: Type[ChatAdapter]) -> None:
    """Expose new dataset adapters without touching the core pipeline."""

    _ADAPTERS[adapter_cls.name] = adapter_cls


def available_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_ADAPTERS.keys()))


def get_adapter(name: str) -> Type[ChatAdapter]:
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        known = ", ".join(available_adapters())
        raise ValueError(f"Unknown adapter '{name}'. Known: {known}")
    return adapter


__all__ = [
    "ChatAdapter",
    "ChatRecord",
    "Message",
    "FineVisionAdapter",
    "register_adapter",
    "available_adapters",
    "get_adapter",
]
