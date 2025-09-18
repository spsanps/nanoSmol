"""Rotary positional embedding helper for the SmolVLM language model."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Lightweight rotary embedding helper.

    The head dimension is conceptually split into two contiguous halves that
    store the cosine and sine channels of the embedding.  Rotating those halves
    by an angle derived from the token position injects relative offsets into
    the attention scores while leaving the vector norm unchanged.  Packaging the
    trigonometric work in this module keeps the attention block focused on the
    linear algebra.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        # Inverse frequencies follow the LLaMA recipe: we sample every other
        # feature to produce half as many frequencies and reuse them for both
        # halves of the head vector.  ``persistent=False`` avoids storing the
        # buffer inside checkpoints unnecessarily.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by swapping halves and flipping signs."""

        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat([-second, first], dim=-1)

    def _compute_cos_sin(
        self, positions: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``positions`` is integer valued with shape [batch, seq_len].  We cast to
        # float so ``einsum`` can multiply by the inverse frequencies.  The result
        # is a grid of angles; concatenating the tensor with itself aligns the
        # values with the cosine half followed by the sine half of the head
        # dimension.
        angles = torch.einsum("bt,d->btd", positions.to(torch.float32), self.inv_freq.to(device))
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._compute_cos_sin(positions, device=query.device, dtype=query.dtype)
        # Broadcast ``cos``/``sin`` across the head dimension so each head shares
        # the same rotation for a given token.
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (self._rotate_half(query) * sin)
        key = (key * cos) + (self._rotate_half(key) * sin)
        return query, key
