"""Normalization layers used by the SmolLM2 blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm used in SmolLM2."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The mean is taken over the last dimension only.
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
