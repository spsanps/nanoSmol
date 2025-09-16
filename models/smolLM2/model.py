"""Top-level language model composed of the SmolLM2 decoder blocks."""
from __future__ import annotations

import re
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import SmolLM2Block
from .config import SmolLM2Config
from .norms import RMSNorm


class SmolLM2(nn.Module):
    """A minimal decoder-only transformer mirroring SmolLM2."""

    def __init__(self, cfg: SmolLM2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SmolLM2Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self._tie_lm_head_to_embedding()

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ utils
    def _tie_lm_head_to_embedding(self) -> None:
        # Share storage so any update to the embedding weights is reflected in
        # the output projection.  This mimics Hugging Face's tie-weights logic.
        self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    @staticmethod
    def from_config_dict(cfg_dict: Dict) -> "SmolLM2":
        return SmolLM2(SmolLM2Config(**cfg_dict))

    # ---------------------------------------------------------------- forward
    def _prepare_attention_inputs(
        self,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        *,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
            attn_mask = pad_mask[:, None, None, :]
            if position_ids is None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        else:
            attn_mask = None
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        return attn_mask, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        attn_mask, position_ids = self._prepare_attention_inputs(attention_mask, position_ids, batch=B, seq_len=T, device=input_ids.device)

        x = self.tok_emb(input_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, position_ids=position_ids)
        x = self.norm_out(x)
        return self.lm_head(x)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    # ---------------------------------------------------------------- loading
    def load_hf_state_dict(
        self,
        hf_state: Dict[str, torch.Tensor],
        *,
        strict: bool = False,
        verbose: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load tensors from a Hugging Face SmolLM2 checkpoint."""

        ref_param = next(self.parameters(), None)
        if dtype is None:
            dtype = ref_param.dtype if ref_param is not None else torch.float32
        if device is None:
            device = ref_param.device if ref_param is not None else torch.device("cpu")

        used_keys = set()
        missing = []
        mismatched = []

        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        def fetch(key: str, *, mark_only: bool = False) -> Optional[torch.Tensor]:
            if key not in hf_state:
                return None
            used_keys.add(key)
            if mark_only:
                return None
            tensor = hf_state[key]
            return tensor.to(device=device, dtype=dtype)

        def assign(target_name: str, value: Optional[torch.Tensor]) -> None:
            if value is None:
                missing.append(target_name)
                return
            target = params.get(target_name) or buffers.get(target_name)
            if target is None:
                missing.append(target_name)
                return
            if target.shape != value.shape:
                mismatched.append((target_name, tuple(target.shape), tuple(value.shape)))
                return
            target.data.copy_(value)

        # --- top-level tensors -------------------------------------------------
        tok_weight = fetch("model.embed_tokens.weight")
        if tok_weight is None and self.cfg.tie_embeddings:
            tok_weight = fetch("lm_head.weight")
        assign("tok_emb.weight", tok_weight)

        if self.cfg.tie_embeddings:
            fetch("lm_head.weight", mark_only=True)
        else:
            assign("lm_head.weight", fetch("lm_head.weight"))

        assign("norm_out.weight", fetch("model.norm.weight"))

        # --- per-layer tensors -------------------------------------------------
        layer_re = re.compile(r"^model\.layers\.(\d+)\.")
        layer_indices = {int(m.group(1)) for key in hf_state for m in [layer_re.search(key)] if m}
        if verbose and layer_indices:
            lo, hi = min(layer_indices), max(layer_indices)
            print(f"[load_hf] src layers detected: {lo}..{hi} (count={len(layer_indices)}) ; dst expects {self.cfg.n_layers}")

        for layer_idx, block in enumerate(self.blocks):
            prefix = f"model.layers.{layer_idx}"
            assign(f"blocks.{layer_idx}.ln_attn.weight", fetch(f"{prefix}.input_layernorm.weight"))
            assign(f"blocks.{layer_idx}.ln_mlp.weight", fetch(f"{prefix}.post_attention_layernorm.weight"))

            assign(f"blocks.{layer_idx}.attn.q_proj.weight", fetch(f"{prefix}.self_attn.q_proj.weight"))
            assign(f"blocks.{layer_idx}.attn.k_proj.weight", fetch(f"{prefix}.self_attn.k_proj.weight"))
            assign(f"blocks.{layer_idx}.attn.v_proj.weight", fetch(f"{prefix}.self_attn.v_proj.weight"))
            assign(f"blocks.{layer_idx}.attn.o_proj.weight", fetch(f"{prefix}.self_attn.o_proj.weight"))

            assign(f"blocks.{layer_idx}.mlp.in_proj.weight", _cat_or_fetch(
                fetch(f"{prefix}.mlp.gate_proj.weight"),
                fetch(f"{prefix}.mlp.up_proj.weight"),
                fetch(f"{prefix}.mlp.w1.weight"),
            ))
            assign(f"blocks.{layer_idx}.mlp.out_proj.weight", fetch(f"{prefix}.mlp.down_proj.weight"))

        unused = [key for key in hf_state if key not in used_keys]
        if verbose:
            print(f"[load_hf] copied tensors: {len(used_keys)}")
            if missing:
                print(f"[load_hf] missing target params: {len(missing)} (up to 10) -> {missing[:10]}")
            if mismatched:
                print(f"[load_hf] shape mismatches: {len(mismatched)} (first) -> {mismatched[0]}")
            if unused:
                print(f"[load_hf] unused source tensors: {len(unused)} (up to 10) -> {unused[:10]}")

        if strict and (missing or mismatched):
            msg = []
            if missing:
                msg.append(f"missing targets: {missing[:10]}")
            if mismatched:
                msg.append(f"shape mismatches: {mismatched[:3]}")
            raise RuntimeError("[load_hf strict] " + " ; ".join(msg))


def _cat_or_fetch(
    gate: Optional[torch.Tensor],
    up: Optional[torch.Tensor],
    combined: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if gate is not None and up is not None:
        return torch.cat([gate, up], dim=0)
    return combined
