"""Streaming datasets for multimodal chat fine-tuning."""
from __future__ import annotations

from typing import Dict, Iterator

from torch.utils.data import IterableDataset

try:  # Optional for tests without datasets installed
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from .adapters import get_adapter
from .config import ChatDataConfig


class StreamingChatDataset(IterableDataset):
    """Stream chat data through an adapter to keep memory usage tiny."""

    def __init__(self, cfg: ChatDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if load_dataset is None:  # pragma: no cover - protects unit tests
            raise ImportError("datasets is required to stream Hugging Face corpora")

    # ------------------------------------------------------------------ helpers
    def _dataset_stream(self):
        data = load_dataset(
            self.cfg.repo_id,
            name=self.cfg.subset,
            streaming=self.cfg.streaming,
            split=self.cfg.split,
        )
        if self.cfg.streaming:
            data = data.shuffle(seed=self.cfg.seed, buffer_size=self.cfg.shuffle_buffer_size)
        elif hasattr(data, "shuffle"):
            data = data.shuffle(seed=self.cfg.seed)
        return data

    # ------------------------------------------------------------------ iterator
    def __iter__(self) -> Iterator[Dict[str, object]]:  # pragma: no cover - requires datasets
        adapter_cls = get_adapter(self.cfg.adapter)
        adapter = adapter_cls(self.cfg)
        for record in self._dataset_stream():
            if not adapter.filter(record):
                continue
            sample = adapter.convert(record)
            if sample is None:
                continue
            yield sample


__all__ = ["StreamingChatDataset"]
