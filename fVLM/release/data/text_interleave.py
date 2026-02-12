"""
Interleaved data loader that mixes vision and text-only batches.

SmolVLM2 retains 14% text-only data in ALL training stages to preserve
instruction-following capability.  Removing it hurts 3.7-6.5% on language
benchmarks (see docs/runpod/SMOLVLM2_REFERENCE.md).

Usage:
    from release.data.text_interleave import InterleavedDataLoader

    loader = InterleavedDataLoader(
        vision_loader=vision_dl,
        text_loader=text_dl,
        text_ratio=0.14,
    )
    for batch in loader:
        # ~86% of batches come from vision_loader (frames + text)
        # ~14% of batches come from text_loader (text-only, zero frames)
        train_step(batch)

Text-only batches are formatted identically to vision batches:
  - frames:         [B, 1, 3, 224, 224]  all zeros
  - frame_mask:     [B, 1]               all False
  - input_ids:      [B, S]               real tokens
  - attention_mask:  [B, S]               real mask
  - loss_mask:      [B, S]               real mask
  - num_frames:     [B]                  all zeros
"""

import random
from typing import Iterator, Optional

import torch


def _wrap_text_batch_as_vision(text_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Take a text-only batch (which may lack frame-related keys) and add
    dummy frame tensors so it has the same schema as a vision batch.

    If the text_batch already has a ``frames`` key (e.g. the text loader
    already produces the full schema), it is returned unchanged.
    """
    if "frames" in text_batch and "frame_mask" in text_batch:
        return text_batch

    B = text_batch["input_ids"].shape[0]

    # Single dummy frame per sample (zeros).  Using T=1 instead of T=0
    # avoids edge-case tensor shape issues in downstream code.
    text_batch["frames"] = torch.zeros(B, 1, 3, 224, 224, dtype=torch.float32)
    text_batch["frame_mask"] = torch.zeros(B, 1, dtype=torch.bool)
    text_batch["num_frames"] = torch.zeros(B, dtype=torch.long)

    # Ensure attention_mask exists (some text loaders use it, some don't).
    if "attention_mask" not in text_batch:
        S = text_batch["input_ids"].shape[1]
        text_batch["attention_mask"] = (text_batch["input_ids"] != 0).bool()

    # Ensure loss_mask exists.
    if "loss_mask" not in text_batch:
        text_batch["loss_mask"] = text_batch["attention_mask"].float()

    return text_batch


class InterleavedDataLoader:
    """
    Yields batches from two data loaders with a configurable mixing ratio.

    Parameters
    ----------
    vision_loader : iterable
        Primary data loader producing vision+text batches (from
        webdataset_loader + collate_foveated).
    text_loader : iterable
        Secondary data loader producing text-only batches (e.g. SmolTalk).
        May or may not include dummy frame tensors -- they will be added
        automatically if missing.
    text_ratio : float
        Target fraction of batches drawn from text_loader.  Default 0.14
        matches SmolVLM2's 14% text retention.
    seed : int
        Random seed for reproducible interleaving order.
    """

    def __init__(
        self,
        vision_loader,
        text_loader,
        text_ratio: float = 0.14,
        seed: int = 42,
    ):
        if not 0.0 <= text_ratio <= 1.0:
            raise ValueError(f"text_ratio must be in [0, 1], got {text_ratio}")

        self.vision_loader = vision_loader
        self.text_loader = text_loader
        self.text_ratio = text_ratio
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)

        vision_iter = iter(self.vision_loader)
        text_iter = iter(self.text_loader)

        # Track exhaustion of each source.
        vision_exhausted = False
        text_exhausted = False

        while True:
            # Decide which source to draw from.
            draw_text = rng.random() < self.text_ratio

            if draw_text and not text_exhausted:
                batch = self._next_text(text_iter)
                if batch is not None:
                    yield batch
                    continue
                # Text loader exhausted -- restart it (text data is small,
                # cycling is expected and intentional).
                text_exhausted = True
                text_iter = iter(self.text_loader)
                batch = self._next_text(text_iter)
                if batch is not None:
                    text_exhausted = False
                    yield batch
                    continue
                # If restart also fails, fall through to vision.

            # Draw from vision loader.
            if not vision_exhausted:
                batch = self._next_vision(vision_iter)
                if batch is not None:
                    yield batch
                    continue
                # Vision loader exhausted -- epoch is done.
                vision_exhausted = True

            # If vision is exhausted we stop the epoch.  The training loop
            # should call iter() again for the next epoch.
            if vision_exhausted:
                return

    @staticmethod
    def _next_vision(it: Iterator) -> Optional[dict]:
        """Pull the next vision batch, return None on StopIteration."""
        try:
            return next(it)
        except StopIteration:
            return None

    @staticmethod
    def _next_text(it: Iterator) -> Optional[dict]:
        """Pull the next text batch and ensure it has the vision-batch schema."""
        try:
            batch = next(it)
            return _wrap_text_batch_as_vision(batch)
        except StopIteration:
            return None

    def __len__(self) -> int:
        """
        Approximate length (number of batches per epoch).

        This is an estimate: the actual count depends on the random draw
        sequence.  It equals the vision loader length scaled by 1/(1-text_ratio)
        to account for interleaved text batches.
        """
        vision_len = getattr(self.vision_loader, "__len__", None)
        if vision_len is None:
            raise TypeError(
                "Cannot compute len() -- the vision_loader does not support __len__. "
                "This is normal for streaming webdataset loaders."
            )
        vision_batches = vision_len()  if callable(vision_len) else vision_len
        if self.text_ratio >= 1.0:
            return 0
        return int(vision_batches / (1.0 - self.text_ratio))
