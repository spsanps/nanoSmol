"""Multi-head attention used inside the SmolLM2 decoder blocks."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolLM2Config
from .rotary import RotaryEmbedding


class SmolLM2Attention(nn.Module):
    """Minimal multi-head attention with RoPE and grouped key/value heads."""

    def __init__(self, cfg: SmolLM2Config) -> None:
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim()
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rotary = RotaryEmbedding(hd, cfg.max_seq_len, base=cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        hd = self.cfg.head_dim()
        h = self.cfg.n_heads
        hk = self.cfg.n_kv_heads

        # Linear projections -> [B, T, (#heads * head_dim)]
        q = self.q_proj(x).view(B, T, h, hd)
        k = self.k_proj(x).view(B, T, hk, hd)
        v = self.v_proj(x).view(B, T, hk, hd)

        # Apply rotary embeddings in-place on q/k.
        q, k = self.rotary(q, k, positions=position_ids)

        # Grouped-query attention: duplicate key/value heads as needed.
        if hk != h:
            repeat = h // hk
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # Switch to [B, h, T, head_dim] for the attention computation.
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, h * hd)
        attn = self.o_proj(attn)
        return self.resid_dropout(attn)
