"""Top-level decoder-only language model stitched from modular building blocks."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .block import SmolVLMLanguageBlock
from .config import SmolVLMLanguageConfig
from .norms import SimpleRMSNorm


class SmolVLMLanguageModel(nn.Module):
    """Decoder-only transformer rebuilt with verbose, reader-friendly comments."""

    def __init__(self, cfg: SmolVLMLanguageConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding table translating vocabulary ids into dense vectors.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SmolVLMLanguageBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm_out = SimpleRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Sharing parameters keeps the number of learnable tensors small and
            # mirrors what Hugging Face does for LLaMA checkpoints.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ helpers
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_attention_inputs(
        self,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Build attention masks + rotary position ids from user input.

        Hugging Face follows the convention where ``1`` marks valid tokens and
        ``0`` marks padding.  We flip that into a boolean mask that PyTorch's
        attention kernel expects (``True`` = ignore).  For positional encodings we
        either reuse the provided tensor or create contiguous positions that skip
        over padding tokens.
        """

        if attention_mask is not None:
            is_padding = ~attention_mask.bool()
            attn_mask = is_padding[:, None, None, :]
            if position_ids is None:
                # ``cumsum`` assigns increasing numbers to non-padding tokens.
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        else:
            attn_mask = None
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        return attn_mask, position_ids

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience wrapper that starts from integer token ids."""

        embeddings = self.tok_emb(input_ids)
        return self.forward_from_embeddings(
            embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the transformer using pre-computed token embeddings."""

        batch_size, seq_len, _ = embeddings.shape
        attn_mask, position_ids = self._prepare_attention_inputs(
            attention_mask,
            position_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            device=embeddings.device,
        )

        hidden_states = self.dropout(embeddings)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attn_mask, position_ids=position_ids)
        hidden_states = self.norm_out(hidden_states)
        return self.lm_head(hidden_states)
