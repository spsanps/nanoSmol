"""Attention module used by the SmolVLM language transformer."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolVLMLanguageConfig
from .rotary import RotaryPositionalEmbedding


class SmolVLMSelfAttention(nn.Module):
    """Grouped-query causal self-attention layer.

    The implementation mirrors the textbook transformer equations so beginners
    can connect the code to the math:

    * Query/key/value projections implement ``Q = XW_Q`` (and friends) where
      ``X`` is the incoming hidden state matrix.
    * Rotary position embeddings rotate the cosine and sine halves of each head
      vector so the dot-product encodes relative position.
    * When ``num_attention_heads`` is a multiple of ``num_key_value_heads`` we
      duplicate key/value heads to emulate grouped-query attention.
    * ``scaled_dot_product_attention`` performs ``softmax(QK^T / sqrt(d))V`` in
      a fused, numerically stable kernel.
    """

    def __init__(self, cfg: SmolVLMLanguageConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_query_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        projection_dim = self.num_query_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Separate linear layers keep the logic easy to follow.
        self.q_proj = nn.Linear(cfg.hidden_size, projection_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(projection_dim, cfg.hidden_size, bias=False)

        self.attention_dropout = nn.Dropout(cfg.dropout)
        self.residual_dropout = nn.Dropout(cfg.dropout)
        self.rotary = RotaryPositionalEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project the input to query/key/value tensors and reshape to explicitly
        # expose the head dimension.  Shape after ``view``: [batch, seq, heads, dim].
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings before the dot-product is taken.
        query, key = self.rotary(query, key, positions=position_ids)

        # ``repeat_interleave`` duplicates key/value heads so every query head has
        # a matching partner.  This is what turns grouped-query attention into an
        # efficient implementation detail instead of a behavioural change.
        if self.num_kv_heads != self.num_query_heads:
            replication = self.num_query_heads // self.num_kv_heads
            key = key.repeat_interleave(replication, dim=2)
            value = value.repeat_interleave(replication, dim=2)

        # ``scaled_dot_product_attention`` expects [batch, heads, seq, dim].
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attn_mask = attention_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # ``scaled_dot_product_attention`` treats boolean masks
                # differently from the additive masks produced by Hugging
                # Face.  Converting to large negative biases matches the
                # reference implementation and avoids numerical drift even
                # when no padding positions are masked.
                if not attn_mask.any():
                    attn_mask = None
                else:
                    mask_val = torch.finfo(query.dtype).min
                    attn_mask = attn_mask.to(dtype=query.dtype)
                    attn_mask = attn_mask.masked_fill(attention_mask, mask_val)
            else:
                attn_mask = attn_mask.to(dtype=query.dtype)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Collapse the heads back into the hidden dimension.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return self.residual_dropout(attn_output)
