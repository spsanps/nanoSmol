"""Configuration objects for chat datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ChatDataConfig:
    """Describe how a multimodal chat dataset should be streamed and filtered."""

    repo_id: str = "HuggingFaceM4/FineVision-1.0"
    subset: Optional[str] = None
    split: str = "train"
    streaming: bool = True
    shuffle_buffer_size: int = 1024
    seed: int = 0
    max_turns: Optional[int] = None
    max_images: int = 1
    image_size: int = 384
    image_mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)
    min_quality: Optional[int] = 4
    adapter: str = "finevision"


__all__ = ["ChatDataConfig"]
