"""Unit tests that lock in the SmolVLM attention mask semantics."""
from __future__ import annotations

import math

import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.smolVLM.attention import SmolVLMSelfAttention
from models.smolVLM.config import SmolVLMLanguageConfig


def _build_attention_module() -> SmolVLMSelfAttention:
    cfg = SmolVLMLanguageConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    module = SmolVLMSelfAttention(cfg)
    module.eval()
    return module


def _convert_mask(attention_mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        if not attention_mask.any():
            return None
        mask = attention_mask.to(dtype=dtype)
        return mask.masked_fill(mask.bool(), torch.finfo(dtype).min)
    return attention_mask.to(dtype=dtype)


def _reference_attention(
    module: SmolVLMSelfAttention,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    query = module.q_proj(hidden_states).view(batch_size, seq_len, module.num_query_heads, module.head_dim)
    key = module.k_proj(hidden_states).view(batch_size, seq_len, module.num_kv_heads, module.head_dim)
    value = module.v_proj(hidden_states).view(batch_size, seq_len, module.num_kv_heads, module.head_dim)

    query, key = module.rotary(query, key, positions=position_ids)

    if module.num_kv_heads != module.num_query_heads:
        replication = module.num_query_heads // module.num_kv_heads
        key = key.repeat_interleave(replication, dim=2)
        value = value.repeat_interleave(replication, dim=2)

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    mask = _convert_mask(attention_mask, query.dtype)
    if mask is not None and mask.dim() == 4:
        mask = mask[:, :, :, : key.shape[-2]]

    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(module.head_dim)
    if mask is not None:
        attn_scores = attn_scores + mask

    mask_value = torch.finfo(query.dtype).min
    causal = torch.triu(torch.ones(seq_len, key.shape[-2], dtype=torch.bool, device=query.device), diagonal=1)
    attn_scores = attn_scores.masked_fill(causal, mask_value)

    attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_probs, value)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_output = module.o_proj(attn_output)
    return module.residual_dropout(attn_output)


@torch.no_grad()
def test_attention_matches_reference_with_and_without_padding() -> None:
    module = _build_attention_module()
    torch.manual_seed(0)

    seq_len = 8
    lengths = [seq_len, seq_len - 2, seq_len - 3, seq_len]
    hidden_states = torch.randn(len(lengths), seq_len, module.cfg.hidden_size)

    for idx, length in enumerate(lengths):
        mask = torch.ones(seq_len, dtype=torch.long)
        if length < seq_len:
            mask[length:] = 0

        padding = (~mask.bool()).view(1, 1, 1, -1)
        position_ids = (mask.view(1, -1).cumsum(-1) - 1).clamp_min(0)

        expected = _reference_attention(
            module,
            hidden_states[idx : idx + 1],
            attention_mask=padding,
            position_ids=position_ids,
        )
        actual = module(
            hidden_states[idx : idx + 1],
            attention_mask=padding,
            position_ids=position_ids,
        )
        assert torch.allclose(actual, expected, atol=1e-6)
