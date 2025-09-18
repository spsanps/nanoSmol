"""Transformer block (attention + MLP) for SmolLM2.

Each block follows the "pre-norm" layout popularised by GPT-style models:

1. Normalise the residual stream so gradients stay well behaved.
2. Apply multi-head attention and add the result back to the residual stream.
3. Repeat the normalise/transform/add pattern with the SwiGLU feed-forward
   network.

This mirrors the standard transformer equations and gives readers a clear map
between the math and the code.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import SmolLM2Attention
from .config import SmolLM2Config
from .mlp import SmolLM2MLP
from .norms import RMSNorm


class SmolLM2Block(nn.Module):
    """A standard pre-norm transformer block."""

    def __init__(self, cfg: SmolLM2Config) -> None:
        super().__init__()
        self.ln_attn = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = SmolLM2Attention(cfg)
        self.ln_mlp = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SmolLM2MLP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the attention and MLP sub-layers with residual connections.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` residual stream.
            attn_mask: Optional attention mask broadcastable to
                ``(batch, num_heads, query_len, key_len)``.
            position_ids: Rotary position indices passed straight to attention.
        """

        # --- Attention sub-layer -------------------------------------------------
        # Pre-norm: stabilise the statistics of the residual stream before the
        # attention projections so that scaling is consistent across layers.
        normalized_for_attention = self.ln_attn(hidden_states)
        attention_update = self.attn(
            normalized_for_attention,
            attn_mask=attn_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attention_update

        # --- Feed-forward sub-layer ----------------------------------------------
        # Another pre-norm step followed by the SwiGLU MLP.  The output is added
        # back to the residual stream, completing the transformer block update.
        normalized_for_mlp = self.ln_mlp(hidden_states)
        mlp_update = self.mlp(normalized_for_mlp)
        return hidden_states + mlp_update
