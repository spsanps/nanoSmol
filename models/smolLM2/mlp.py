"""Feed-forward network (SwiGLU) for the SmolLM2 block."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolLM2Config


class SmolLM2MLP(nn.Module):
    """SwiGLU MLP used in SmolLM2.

    Hugging Face's implementation splits the gate and up projections.  We keep
    that layout but pack them into a single linear layer for simplicity.
    """

    def __init__(self, cfg: SmolLM2Config) -> None:
        super().__init__()
        hidden = cfg.d_ff
        self.in_proj = nn.Linear(cfg.d_model, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.out_proj(x)
        return self.dropout(x)
