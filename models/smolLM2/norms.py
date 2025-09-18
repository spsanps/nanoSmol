"""Normalization layers used by the SmolLM2 blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm used in SmolLM2.

    RMSNorm normalises each token vector ``x`` using

    ``x / sqrt(mean(x^2) + eps)``

    This omits the learned bias from standard LayerNorm and only keeps a single
    learned scale parameter, which is stored in ``self.weight``.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension of ``x`` using its root mean square."""

        # Compute mean(x^2) along the feature dimension, keep dims for broadcast.
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        normalised = x * torch.rsqrt(rms + self.eps)
        return self.weight * normalised
