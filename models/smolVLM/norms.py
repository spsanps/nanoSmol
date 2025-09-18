"""Normalisation layers purpose-built for the SmolVLM language stack."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRMSNorm(nn.Module):
    """Root-mean-square normalisation with a single learnable scale.

    ``LayerNorm`` subtracts the mean and divides by the standard deviation.
    LLaMA-style models instead scale each feature by the *root mean square* of
    the activations.  This variant is marginally cheaper (no mean subtraction)
    while keeping the same stabilising effect.  The implementation fits in a
    handful of lines which makes it ideal for an educational code base.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        # ``weight`` stretches or shrinks each feature after normalisation.
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # ``inputs`` has shape [batch, seq_len, hidden_size].  We reduce over the
        # last dimension to obtain the per-token variance.
        variance = inputs.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        normalised = inputs * inv_rms
        return normalised * self.weight
