"""Feed-forward network (SwiGLU) for the SmolLM2 block."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolLM2Config


class SmolLM2MLP(nn.Module):
    """SwiGLU MLP used in SmolLM2.

    The feed-forward network implements the following transformation for each
    token independently:

    ``SwiGLU(x) = (W_g x) * silu(W_f x)`` where the multiplication is elementwise.
    The combined projection in ``self.in_proj`` stores the
    ``W_g`` (gate) and ``W_f`` (feature) weights back-to-back so we can split
    them with a single ``chunk`` operation.
    """

    def __init__(self, cfg: SmolLM2Config) -> None:
        super().__init__()
        hidden_dim = cfg.d_ff
        self.in_proj = nn.Linear(cfg.d_model, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the SwiGLU feed-forward network used in SmolLM2.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` tensor.
        Returns:
            Tensor with the same shape containing the per-token MLP update.
        """

        gate_pre_activation, feature_values = self.in_proj(hidden_states).chunk(2, dim=-1)
        # SwiGLU activation: silu(gate) = gate * sigmoid(gate).  Multiplying this
        # smooth gate with the candidate features allows the model to learn a
        # data-dependent sparsity pattern.
        gated_features = F.silu(gate_pre_activation) * feature_values
        projected_update = self.out_proj(gated_features)
        return self.dropout(projected_update)
