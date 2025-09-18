"""Vision encoder mirroring the SmolVLM SigLIP backbone."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolVLMVisionConfig


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """Implementation of the tanh-approximated GELU used by SigLIP."""

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def get_activation(name: str):
    if name == "gelu" or name == "gelu_new":
        return F.gelu
    if name == "gelu_pytorch_tanh":
        return gelu_pytorch_tanh
    raise ValueError(f"Unsupported activation: {name}")


class SmolVLMVisionAttention(nn.Module):
    """Standard multi-head self-attention for the vision transformer."""

    def __init__(self, cfg: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        if self.head_dim * self.num_heads != cfg.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.dropout = nn.Dropout(cfg.attention_dropout)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        if attn_mask is not None:
            neg_inf = torch.finfo(hidden_states.dtype).min
            mask = (~attn_mask).to(dtype=hidden_states.dtype) * neg_inf
            mask = mask.unsqueeze(1).unsqueeze(2)

        attn_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.o_proj(attn_out)


class SmolVLMVisionMLP(nn.Module):
    """Feed-forward network used inside each vision transformer block."""

    def __init__(self, cfg: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        self.activation = get_activation(cfg.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.fc2(hidden_states)


class SmolVLMVisionBlock(nn.Module):
    """Pre-norm transformer block for the vision encoder."""

    def __init__(self, cfg: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = SmolVLMVisionAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = SmolVLMVisionMLP(cfg)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_input = self.ln1(hidden_states)
        hidden_states = hidden_states + self.attn(attn_input, attn_mask)
        mlp_input = self.ln2(hidden_states)
        return hidden_states + self.mlp(mlp_input)


class SmolVLMVisionEncoder(nn.Module):
    """Minimal SigLIP-style encoder used by SmolVLM."""

    def __init__(self, cfg: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(cfg.num_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(cfg.num_patches, cfg.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))  # kept for checkpoint compatibility
        self.blocks = nn.ModuleList([SmolVLMVisionBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.ln_post = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = pixel_values.size(0)
        device = pixel_values.device

        if patch_attention_mask is None:
            h = pixel_values.shape[2] // self.cfg.patch_size
            w = pixel_values.shape[3] // self.cfg.patch_size
            patch_attention_mask = torch.ones((batch, h, w), dtype=torch.bool, device=device)

        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        position_ids = self._build_position_ids(patch_attention_mask)
        pos_embeds = F.embedding(position_ids, self.pos_embed)
        hidden_states = hidden_states + pos_embeds

        attn_mask = patch_attention_mask.view(batch, -1)
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask)
        return self.ln_post(hidden_states)

    def _build_position_ids(self, patch_mask: torch.Tensor) -> torch.Tensor:
        batch, h, w = patch_mask.shape
        device = patch_mask.device
        dtype = torch.float32
        boundaries = torch.arange(
            1 / self.cfg.num_patches_per_side,
            1.0,
            1 / self.cfg.num_patches_per_side,
            device=device,
            dtype=dtype,
        )
        position_ids = torch.zeros(batch, h * w, dtype=torch.long, device=device)
        for batch_idx, mask in enumerate(patch_mask):
            valid_flat = mask.view(-1)
            if not torch.any(valid_flat):
                continue
            patches_h = int(mask[:, 0].sum().item())
            patches_w = int(mask[0].sum().item())
            if patches_h == 0 or patches_w == 0:
                continue
            frac_h = torch.arange(patches_h, device=device, dtype=dtype)
            frac_w = torch.arange(patches_w, device=device, dtype=dtype)
            frac_h = frac_h / max(patches_h, 1) * (1.0 - 1e-6)
            frac_w = frac_w / max(patches_w, 1) * (1.0 - 1e-6)
            bucket_h = torch.bucketize(frac_h, boundaries, right=True)
            bucket_w = torch.bucketize(frac_w, boundaries, right=True)
            pos = (bucket_h[:, None] * self.cfg.num_patches_per_side + bucket_w).reshape(-1)
            position_ids[batch_idx, valid_flat] = pos[: int(valid_flat.sum().item())]
        return position_ids
