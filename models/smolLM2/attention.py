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
        head_dim = cfg.head_dim()
        # Separate projection matrices are used so we can fold grouped-query
        # attention (more query than key/value heads) into a minimal module.
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * head_dim, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rotary = RotaryEmbedding(head_dim, cfg.max_seq_len, base=cfg.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the attention output for a batch of hidden states.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            attn_mask: Optional attention mask broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``.  ``None`` means no
                padding mask.
            position_ids: Pre-computed rotary position ids of shape
                ``(batch, seq_len)``.
        """

        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.cfg.head_dim()
        num_query_heads = self.cfg.n_heads
        num_kv_heads = self.cfg.n_kv_heads

        # Linear projections -> [batch, seq_len, (#heads * head_dim)]
        query = self.q_proj(hidden_states).view(batch_size, seq_len, num_query_heads, head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)

        # Apply rotary position encodings before the attention score dot-product.
        query, key = self.rotary(query, key, positions=position_ids)

        # Grouped-query attention: duplicate key/value heads if there are fewer
        # kv-heads than query-heads.  repeat_interleave keeps tensors contiguous.
        if num_kv_heads != num_query_heads:
            repeat_factor = num_query_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=2)
            value = value.repeat_interleave(repeat_factor, dim=2)

        # Switch to [batch, heads, seq_len, head_dim] for the attention kernel.
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Restore [batch, seq_len, hidden_size] and combine the heads.
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, num_query_heads * head_dim
        )
        attention_output = self.o_proj(attention_output)
        return self.resid_dropout(attention_output)
