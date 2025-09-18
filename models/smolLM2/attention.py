"""Multi-head attention used inside the SmolLM2 decoder blocks.

The implementation intentionally mirrors the equations from the Transformer
paper so that new readers can connect the PyTorch code with the math:

* The query/key/value projections implement ``Q = XW_Q`` (and analogous
  equations for ``K`` and ``V``) where ``X`` is the incoming hidden state.
* Rotary position embeddings (RoPE) inject relative position information by
  rotating the first and second halves of each head vector, which store the
  cosine and sine parts of the complex representation.
* Grouped-query attention is supported by duplicating the key/value heads when
  ``n_heads`` is a multiple of ``n_kv_heads``.
* The PyTorch ``scaled_dot_product_attention`` kernel performs
  ``softmax(QK^T / sqrt(d_head))V`` in a numerically stable fused operation.
"""
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
            hidden_states: ``(batch, seq_len, hidden_size)`` tensor containing the
                activations produced by the previous layer.
            attn_mask: Optional mask broadcastable to
                ``(batch, num_heads, query_len, key_len)``. ``True`` entries mark
                positions that should be ignored.
            position_ids: ``(batch, seq_len)`` tensor with per-token positions
                used by rotary embeddings.
        """

        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.cfg.head_dim()
        num_query_heads = self.cfg.n_heads
        num_kv_heads = self.cfg.n_kv_heads

        # Perform the learned linear projections.  After the view the tensor is
        # shaped as [batch, seq_len, num_heads, head_dim] so each head owns a
        # contiguous slice of the hidden dimension.
        query_vectors = self.q_proj(hidden_states).view(
            batch_size, seq_len, num_query_heads, head_dim
        )
        key_vectors = self.k_proj(hidden_states).view(
            batch_size, seq_len, num_kv_heads, head_dim
        )
        value_vectors = self.v_proj(hidden_states).view(
            batch_size, seq_len, num_kv_heads, head_dim
        )

        # Apply RoPE so the dot-product between queries and keys encodes their
        # relative position.  The vector is split into cosine and sine halves
        # which are rotated by an angle derived from ``position_ids``.
        query_vectors, key_vectors = self.rotary(query_vectors, key_vectors, positions=position_ids)

        # Grouped-query attention: duplicate key/value heads if there are fewer
        # kv-heads than query-heads.  repeat_interleave keeps tensors contiguous.
        if num_kv_heads != num_query_heads:
            repeat_factor = num_query_heads // num_kv_heads
            key_vectors = key_vectors.repeat_interleave(repeat_factor, dim=2)
            value_vectors = value_vectors.repeat_interleave(repeat_factor, dim=2)

        # Rearrange to the kernel's preferred [batch, heads, seq_len, head_dim]
        # layout.  ``permute`` only reorders views; ``contiguous`` below ensures
        # the tensor is laid out as expected before the final projection.
        query_vectors = query_vectors.permute(0, 2, 1, 3)
        key_vectors = key_vectors.permute(0, 2, 1, 3)
        value_vectors = value_vectors.permute(0, 2, 1, 3)

        attention_output = F.scaled_dot_product_attention(
            query_vectors,
            key_vectors,
            value_vectors,
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
