"""Feed-forward network used inside the SmolVLM language block."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolVLMLanguageConfig


class SmolVLMFeedForward(nn.Module):
    """SwiGLU feed-forward network used inside each language block.

    The projection expands the hidden dimension by ``2 * intermediate_size`` so
    we can form a gate and a candidate tensor.  Applying ``SiLU`` to the gate and
    multiplying by the candidate implements the SwiGLU activation popularised by
    LLaMA.  A final linear layer brings the representation back to
    ``hidden_size`` so it can be added to the residual stream.
    """

    def __init__(self, cfg: SmolVLMLanguageConfig) -> None:
        super().__init__()
        hidden = cfg.intermediate_size
        self.in_proj = nn.Linear(cfg.hidden_size, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ``hidden_states`` has shape [batch, seq_len, hidden_size].  ``chunk``
        # splits the expanded projection into gate + candidate tensors.
        gate_values, candidate_values = self.in_proj(hidden_states).chunk(2, dim=-1)
        activated = F.silu(gate_values) * candidate_values
        activated = self.out_proj(activated)
        return self.dropout(activated)
