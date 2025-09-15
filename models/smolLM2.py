"""
model.py — Minimal SmolLM2-style decoder-only LLM in a nanoGPT-like style.

Design goals
- Keep it small, readable, single-file.
- Use modern LLM ingredients: RMSNorm, RoPE, GQA (n_kv_head), SwiGLU MLP, weight tying.
- Make config-driven so we can instantiate sizes matching released SmolLM2 checkpoints.
- Keep attention path compatible with PyTorch 2.x scaled_dot_product_attention (flash if available).
- Provide light helpers for loading HF-style state_dicts later (name mapping hooks).

Note: This is the *LLM-only* core (no vision). A VLM connector can sit on top later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Config
# ------------------------------

@dataclass
class SmolConfig:
    # Model dims
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int  # for GQA/MQA; set equal to n_head for MHA
    d_ff: int

    # Positional / context
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    rope_pct: float = 1.0  # fraction of head_dim using RoPE (1.0 = full)

    # Training niceties
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # Projections / activations
    tie_embeddings: bool = False
    qkv_proj_bias: bool = False
    out_proj_bias: bool = False
    mlp_activation: str = "silu"  # "silu" or "gelu"
    gated_mlp: bool = True         # True -> SwiGLU (SiLU-Gated)
    mlp_bias: bool = False

    # Initialization
    init_std: float = 0.02

    # Token ids
    pad_token_id: int = 0

    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head

    def rope_dim(self) -> int:
        return int(self.head_dim() * self.rope_pct)


# ------------------------------
# Layers
# ------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """
    HF-compatible RoPE for LLaMA-style models.
    Applies rotation to the first rope_dim of q,k.
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0, rope_dim: Optional[int] = None):
        super().__init__()
        self.head_dim = head_dim
        self.rope_dim = rope_dim if rope_dim is not None else head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seqlen: int, device, dtype):
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, rope_dim/2]
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin  # [T, D/2]

    @staticmethod
    def _rotate_half(x):  # x[..., D]
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)  # [-x1, x0] interleave

    def _apply_one(self, x, cos, sin, pos_ids=None):
        x_rope, x_pass = x[..., :self.rope_dim], x[..., self.rope_dim:]
        if pos_ids is not None:
            cos = F.embedding(pos_ids, cos)  # [B,T,D/2]
            sin = F.embedding(pos_ids, sin)
        # broadcast to [B,T,1,D/2]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        # interleave back to D by duplicating cos/sin across pairs
        x_cos = torch.empty_like(x_rope)
        x_cos[..., ::2] = cos
        x_cos[..., 1::2] = cos
        x_sin = torch.empty_like(x_rope)
        x_sin[..., ::2] = sin
        x_sin[..., 1::2] = sin
        x_applied = (x_rope * x_cos) + (self._rotate_half(x_rope) * x_sin)
        return torch.cat([x_applied, x_pass], dim=-1)

    def forward(self, q, k, positions: Optional[torch.Tensor] = None):
        # q,k: [B, T, H, hd]
        T = q.size(1)
        cos, sin = self._cos_sin(T, q.device, q.dtype)  # [T, D/2]
        q = self._apply_one(q, cos, sin, positions)
        k = self._apply_one(k, cos, sin, positions)
        return q, k


class MLP(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        act = cfg.mlp_activation.lower()
        self.act = F.silu if act == "silu" else F.gelu
        self.gated = cfg.gated_mlp
        hidden = cfg.d_ff
        use_bias = cfg.mlp_bias
        if self.gated:
            self.w1 = nn.Linear(cfg.d_model, hidden * 2, bias=use_bias)
        else:
            self.w1 = nn.Linear(cfg.d_model, hidden, bias=use_bias)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            u, v = self.w1(x).chunk(2, dim=-1)  # SwiGLU
            return self.w2(self.act(u) * v)
        else:
            return self.w2(self.act(self.w1(x)))


class MultiheadAttention(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.n_head
        hk = cfg.n_kv_head
        d = cfg.d_model
        hd = cfg.head_dim()
        self.q_proj = nn.Linear(d, h * hd, bias=cfg.qkv_proj_bias)
        self.k_proj = nn.Linear(d, hk * hd, bias=cfg.qkv_proj_bias)
        self.v_proj = nn.Linear(d, hk * hd, bias=cfg.qkv_proj_bias)
        self.o_proj = nn.Linear(h * hd, d, bias=cfg.out_proj_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(hd, cfg.max_seq_len, base=cfg.rope_base, rope_dim=cfg.rope_dim())

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        h, hk, hd = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim()

        q = self.q_proj(x).view(B, T, h, hd)
        k = self.k_proj(x).view(B, T, hk, hd)
        v = self.v_proj(x).view(B, T, hk, hd)

        # RoPE on q,k
        q, k = self.rope(q, k, positions=positions)

        # GQA: expand k,v to n_head
        if hk != h:
            repeat = h // hk
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # [B, h, T, hd]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # PyTorch 2.x SDPA (uses Flash/Math depending on dtype/device)
        # attn_mask can be additive or boolean; we expect None for simple causal here.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, h * hd)
        out = self.resid_dropout(self.o_proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = MultiheadAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask, positions=positions)
        x = x + self.mlp(self.ln2(x))
        return x


class SmolLM(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    # ---- Forward ----
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute logits for next-token prediction.
        input_ids: [B, T]
        attention_mask: optional additive/boolean mask (we keep causal by default).
        positions: optional absolute positions [B, T]; if None uses arange(T).
        returns: logits [B, T, vocab]
        """
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.tok_emb(input_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask, positions=positions)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    # ---- Utils ----
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def from_config_dict(cfg_dict: dict) -> "SmolLM":
        cfg = SmolConfig(**cfg_dict)
        return SmolLM(cfg)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Greedy/top-k sampling without KV cache (simple but fine for smoke tests)."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    # ---- HF loading hook ----
    def load_hf_state_dict(
        self,
        hf_state: dict,
        *,
        strict: bool = False,
        verbose: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Load a Hugging Face SmolLM2-style checkpoint into this minimal model.

        - Embeddings / final norm / (optionally) lm_head
        - Per-layer: LN, Q/K/V/O projections
        - MLP (SwiGLU): gate_proj + up_proj -> w1 (concat along dim=0), down_proj -> w2
        - Biases are copied if both source and target have them; otherwise ignored.

        Args:
            hf_state: state_dict from a HF model (AutoModelForCausalLM or safetensors).
            strict:  if True, raise if a mapped param is missing or mismatched.
            verbose: print a short mapping report.
            dtype, device: optional casts for loaded tensors (defaults to module's).
        """
        # Resolve dtype/device defaults to current module
        ref_param = next(self.parameters(), None)
        if dtype is None:
            dtype = (ref_param.dtype if ref_param is not None else torch.float32)
        if device is None:
            device = (ref_param.device if ref_param is not None else torch.device("cpu"))

        used_keys = set()
        missing_targets = []
        shape_mismatches = []

        def get_src(key: str) -> torch.Tensor | None:
            if key in hf_state:
                used_keys.add(key)
                t = hf_state[key]
                # Some HF checkpoints store on meta/cpu; bring to our dtype/device
                return t.to(device=device, dtype=dtype)
            return None

        def copy_(target_name: str, src: torch.Tensor | None) -> None:
            if src is None:
                missing_targets.append(target_name)
                return
            tgt = dict(self.named_parameters()).get(target_name, None)
            if tgt is None:
                # allow copying into buffers (e.g., norms) if needed
                tgt = dict(self.named_buffers()).get(target_name, None)
            if tgt is None:
                missing_targets.append(target_name)
                return
            if tgt.shape != src.shape:
                shape_mismatches.append((target_name, tuple(tgt.shape), tuple(src.shape)))
                return
            with torch.no_grad():
                tgt.copy_(src)

        # ---- Top-level mappings ----
        copy_("tok_emb.weight", get_src("model.embed_tokens.weight"))

        # lm_head: skip if tied
        if not self.cfg.tie_embeddings:
            copy_("lm_head.weight", get_src("lm_head.weight"))

        copy_("ln_f.weight", get_src("model.norm.weight"))

        # ---- Detect number of layers present in HF state ----
        # (We still iterate over self.cfg.n_layer to ensure target shape)
        import re
        layer_idxs = {int(m.group(1)) for k in hf_state.keys()
                    for m in [re.search(r"^model\.layers\.(\d+)\.", k)] if m}
        if verbose and layer_idxs:
            print(f"[load_hf] src layers detected: {min(layer_idxs)}..{max(layer_idxs)} "
                f"(count={len(layer_idxs)}) ; dst expects {self.cfg.n_layer}")

        # ---- Per-layer mapping ----
        for i in range(self.cfg.n_layer):
            pfx = f"model.layers.{i}"

            # norms
            copy_(f"blocks.{i}.ln1.weight", get_src(f"{pfx}.input_layernorm.weight"))
            copy_(f"blocks.{i}.ln2.weight", get_src(f"{pfx}.post_attention_layernorm.weight"))

            # attention projections
            copy_(f"blocks.{i}.attn.q_proj.weight", get_src(f"{pfx}.self_attn.q_proj.weight"))
            copy_(f"blocks.{i}.attn.k_proj.weight", get_src(f"{pfx}.self_attn.k_proj.weight"))
            copy_(f"blocks.{i}.attn.v_proj.weight", get_src(f"{pfx}.self_attn.v_proj.weight"))
            copy_(f"blocks.{i}.attn.o_proj.weight", get_src(f"{pfx}.self_attn.o_proj.weight"))

            # optional attn biases (only if both sides have them)
            tgt_params = dict(self.named_parameters())
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tname = f"blocks.{i}.attn.{proj}.bias"
                if tname in tgt_params:
                    copy_(tname, get_src(f"{pfx}.self_attn.{proj}.bias"))

            # MLP: gate + up -> w1 (concat dim=0 so first half = gate, second = up)
            gate_w = get_src(f"{pfx}.mlp.gate_proj.weight")
            up_w   = get_src(f"{pfx}.mlp.up_proj.weight")
            down_w = get_src(f"{pfx}.mlp.down_proj.weight")

            if gate_w is not None and up_w is not None:
                w1 = torch.cat([gate_w, up_w], dim=0)  # [2*hidden, d_model]
                copy_(f"blocks.{i}.mlp.w1.weight", w1)
            else:
                # Fallback: if a combined w1 exists in source (less common)
                comb = get_src(f"{pfx}.mlp.w1.weight")
                if comb is not None:
                    copy_(f"blocks.{i}.mlp.w1.weight", comb)
                else:
                    missing_targets.append(f"blocks.{i}.mlp.w1.weight")

            if down_w is not None:
                copy_(f"blocks.{i}.mlp.w2.weight", down_w)
            else:
                missing_targets.append(f"blocks.{i}.mlp.w2.weight")

            # MLP biases if present
            tgt_params = dict(self.named_parameters())
            if f"blocks.{i}.mlp.w1.bias" in tgt_params:
                gate_b = get_src(f"{pfx}.mlp.gate_proj.bias")
                up_b   = get_src(f"{pfx}.mlp.up_proj.bias")
                if gate_b is not None and up_b is not None:
                    copy_(f"blocks.{i}.mlp.w1.bias", torch.cat([gate_b, up_b], dim=0))
                else:
                    comb_b = get_src(f"{pfx}.mlp.w1.bias")
                    if comb_b is not None:
                        copy_(f"blocks.{i}.mlp.w1.bias", comb_b)
            if f"blocks.{i}.mlp.w2.bias" in tgt_params:
                copy_(f"blocks.{i}.mlp.w2.bias", get_src(f"{pfx}.mlp.down_proj.bias"))

        # ---- Report ----
        unused = [k for k in hf_state.keys() if k not in used_keys]
        if verbose:
            print(f"[load_hf] copied tensors: {len(used_keys)}")
            if missing_targets:
                print(f"[load_hf] missing target params: {len(missing_targets)} "
                    f"(showing up to 10) -> {missing_targets[:10]}")
            if shape_mismatches:
                print(f"[load_hf] shape mismatches: {len(shape_mismatches)} "
                    f"(first) -> {shape_mismatches[0]}")
            if unused:
                print(f"[load_hf] unused source tensors: {len(unused)} "
                    f"(showing up to 10) -> {unused[:10]}")

        if strict and (missing_targets or shape_mismatches):
            msg = []
            if missing_targets:
                msg.append(f"missing targets: {missing_targets[:10]}")
            if shape_mismatches:
                msg.append(f"shape mismatches: {shape_mismatches[:3]}")
            raise RuntimeError("[load_hf strict] " + " ; ".join(msg))


# ------------------------------
# Tiny smoke test
# ------------------------------
if __name__ == "__main__":
    cfg = SmolConfig(vocab_size=32000, d_model=512, n_layer=12, n_head=8, n_kv_head=4, d_ff=1536, max_seq_len=1024)
    model = SmolLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    print("logits:", logits.shape)
    print("params:", model.num_parameters())
