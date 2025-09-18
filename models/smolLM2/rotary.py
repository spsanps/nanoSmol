"""Rotary positional embeddings (RoPE) for SmolLM2 attention heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Apply RoPE in the LLaMA/SmolLM2 style.

    We split the head dimension into two contiguous halves.  The first half acts
    as the "cosine" coordinates, the second half provides the "sine" component.
    Rotating these halves by an angle derived from the token position injects
    relative offsets into the attention dot-product while preserving the vector
    norm.  This mirrors the complex-number intuition often used to explain RoPE
    without requiring the even/odd interleaving used in some implementations.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        # Pre-compute inverse frequencies for half of the features.  Each value
        # corresponds to the ``1 / base^(i / head_dim)`` term in the RoPE
        # formulation and is reused for both halves of the head vector.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Swap and negate halves to emulate multiplication by ``i``."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _cos_sin(self, positions: torch.Tensor, device, dtype):
        """Compute ``cos(theta)``/``sin(theta)`` for each position and feature pair."""
        freqs = torch.einsum("bt,d->btd", positions.to(torch.float32), self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the rotary transformation to the query/key tensors.

        Args:
            q: ``(batch, seq_len, num_heads, head_dim)`` query tensor.
            k: ``(batch, seq_len, num_heads, head_dim)`` key tensor.
            positions: ``(batch, seq_len)`` integer tensor with absolute positions.
        Returns:
            Tuple ``(q_rotated, k_rotated)`` with the same shapes as the inputs.
        """

        cos, sin = self._cos_sin(positions, q.device, q.dtype)
        cos = cos.unsqueeze(2)  # broadcast across attention heads
        sin = sin.unsqueeze(2)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
