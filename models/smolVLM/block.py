"""Transformer block wiring the SmolVLM attention + feed-forward."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import SmolVLMSelfAttention
from .config import SmolVLMLanguageConfig
from .mlp import SmolVLMFeedForward
from .norms import SimpleRMSNorm


class SmolVLMLanguageBlock(nn.Module):
    """Pre-normalised transformer block (attention + feed-forward).

    The structure mirrors GPT-style decoder blocks:

    1. Apply RMSNorm to stabilise the residual stream before the projections.
    2. Run multi-head attention and add the result back to the stream.
    3. Repeat the norm/transform/add pattern with the SwiGLU feed-forward MLP.

    Keeping the steps separate makes it easier to trace tensor shapes and
    understand where each transformation fits into the overall computation.
    """

    def __init__(self, cfg: SmolVLMLanguageConfig) -> None:
        super().__init__()
        self.ln_attn = SimpleRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = SmolVLMSelfAttention(cfg)
        self.ln_mlp = SimpleRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SmolVLMFeedForward(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Attention branch: normalise, transform, and add back to the residual
        # stream.  ``hidden_states`` and ``attn_output`` both have shape
        # [batch, seq_len, hidden_size].
        attn_input = self.ln_attn(hidden_states)
        attn_output = self.attn(attn_input, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = hidden_states + attn_output

        # Feed-forward branch mirrors the same pre-norm pattern.
        mlp_input = self.ln_mlp(hidden_states)
        mlp_output = self.mlp(mlp_input)
        return hidden_states + mlp_output
