"""Utilities for mapping vision features into the language space."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmolVLMMultiModalProjector(nn.Module):
    """Pixel-shuffle followed by a small MLP to match the text dimension."""

    def __init__(self, *, vision_dim: int, text_dim: int, scale_factor: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.scale_factor = max(scale_factor, 1)
        self.hidden_dim = hidden_dim
        self.out_features = text_dim
        if hidden_dim is None:
            self.proj = nn.Linear(vision_dim, text_dim, bias=False)
        else:
            self.linear1 = nn.Linear(vision_dim, hidden_dim, bias=True)
            self.linear2 = nn.Linear(hidden_dim, text_dim, bias=True)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        if image_hidden_states.numel() == 0:
            return image_hidden_states.new_zeros((0, 0, self.out_features))

        shuffled = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        if self.hidden_dim is None:
            return self.proj(shuffled)
        hidden = F.silu(self.linear1(shuffled))
        return self.linear2(hidden)

    @staticmethod
    def pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        if scale_factor <= 1:
            return x
        batch, seq_len, dim = x.shape
        side = int(seq_len ** 0.5)
        if side * side != seq_len:
            raise ValueError(f"Sequence length {seq_len} is not a perfect square; cannot pixel-shuffle.")
        if side % scale_factor != 0:
            raise ValueError("Scale factor must divide the patch grid size")
        new_side = side // scale_factor
        x = x.view(batch, side, side, dim)
        x = x.reshape(batch, new_side, scale_factor, new_side, scale_factor, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(batch, new_side * new_side, dim * (scale_factor ** 2))
        return x
