"""Feed-forward network used inside the SmolVLM language block."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolVLMLanguageConfig


class SmolVLMFeedForward(nn.Module):
    """SwiGLU feed-forward network used inside each language block."""

    def __init__(self, cfg: SmolVLMLanguageConfig) -> None:
        super().__init__()
        hidden = cfg.intermediate_size
        self.in_proj = nn.Linear(cfg.hidden_size, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ``chunk`` splits the projection into gate + candidate tensors.
        gate_values, candidate_values = self.in_proj(hidden_states).chunk(2, dim=-1)
        activated = F.silu(gate_values) * candidate_values
        activated = self.out_proj(activated)
        return self.dropout(activated)
