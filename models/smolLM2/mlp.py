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
        hidden_dim = cfg.d_ff
        self.in_proj = nn.Linear(cfg.d_model, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the SwiGLU feed-forward network used in SmolLM2."""

        gate_values, candidate_values = self.in_proj(hidden_states).chunk(2, dim=-1)
        # SwiGLU activation: silu(gate) provides a smooth gating curve that is
        # multiplied with the candidate branch to introduce elementwise sparsity.
        activated = F.silu(gate_values) * candidate_values
        activated = self.out_proj(activated)
        return self.dropout(activated)
