"""Rotary positional embeddings (RoPE) for SmolLM2 attention heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Apply RoPE in the LLaMA/SmolLM2 style.

    The implementation mirrors Hugging Face's `RotaryEmbedding` module but keeps
    the code small and well-commented for educational purposes.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        # Pre-compute inverse frequencies for even positions.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Helper used to apply the complex rotation in a real-valued space.
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _cos_sin(self, positions: torch.Tensor, device, dtype):
        freqs = torch.einsum("bt,d->btd", positions.to(torch.float32), self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, T, H, head_dim]; positions: [B, T]
        cos, sin = self._cos_sin(positions, q.device, q.dtype)
        cos = cos.unsqueeze(2)  # broadcast across heads
        sin = sin.unsqueeze(2)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
