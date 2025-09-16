"""Transformer block (attention + MLP) for SmolLM2."""
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
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x), attn_mask=attn_mask, position_ids=position_ids)
        x = x + self.mlp(self.ln_mlp(x))
        return x
