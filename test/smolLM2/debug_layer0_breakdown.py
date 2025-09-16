#!/usr/bin/env python3
"""
Layer-0 stage-by-stage parity check between HF (Llama/SmolLM2) and your minimal SmolLM.

Usage:
  python test/smolLM2/debug_layer0_breakdown.py \
    --hf-model HuggingFaceTB/SmolLM2-135M \
    --prompt "The capital of France is" \
    --dtype float32
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- import your model from repo root ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.smolLM2 import SmolConfig, SmolLM  # noqa: E402


# ---------- helpers ----------
def resolve_pad_id(hf_cfg, tok) -> int:
    for src in (getattr(hf_cfg, "pad_token_id", None),
                getattr(tok, "pad_token_id", None),
                getattr(hf_cfg, "eos_token_id", None),
                0):
        if src is not None:
            return int(src)
    return 0

def map_hf_config_to_smol(cfg_hf, pad_id: int) -> SmolConfig:
    def _get(a, d=None): return getattr(cfg_hf, a, d)
    return SmolConfig(
        vocab_size=int(_get("vocab_size")),
        d_model=int(_get("hidden_size", _get("n_embd"))),
        n_layer=int(_get("num_hidden_layers", _get("n_layer"))),
        n_head=int(_get("num_attention_heads", _get("n_head"))),
        n_kv_head=int(_get("num_key_value_heads", _get("num_attention_heads"))),
        d_ff=int(_get("intermediate_size", 4 * int(_get("hidden_size", _get("n_embd"))))),
        max_seq_len=int(_get("max_position_embeddings", 4096)),
        rope_base=float(_get("rope_theta", _get("rope_base", 10000.0))),
        rope_pct=1.0,
        dropout=0.0,
        norm_eps=float(_get("rms_norm_eps", _get("layer_norm_eps", 1e-6))),
        tie_embeddings=bool(_get("tie_word_embeddings", True)),
        qkv_proj_bias=False, out_proj_bias=False,
        mlp_activation="silu", gated_mlp=True, mlp_bias=False,
        init_std=0.02, pad_token_id=pad_id,
    )

@torch.no_grad()
def maxdiff(a, b): return float((a - b).abs().max().item())

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)

def repeat_kv(x, n_rep):
    b, h, t, d = x.shape
    if n_rep == 1: return x
    x = x[:, :, None].expand(b, h, n_rep, t, d)
    return x.reshape(b, h * n_rep, t, d)

def make_additive_causal_mask(attn_mask, T, device, dtype):
    # causal part: [1,1,T,T], 0 on allowed, -inf above diagonal
    m = torch.full((1, 1, T, T), float("-inf"), device=device, dtype=torch.float32)
    m = torch.triu(m, diagonal=1)
    # key padding part: [B,1,1,T]
    if attn_mask is not None:
        kpm_bool = (~attn_mask.bool()).unsqueeze(1).unsqueeze(2)
        kpm = torch.zeros_like(kpm_bool, dtype=torch.float32)
        kpm[kpm_bool] = float("-inf")
        m = m + kpm
    return m.to(dtype=torch.float32)  # scores/softmax we’ll do in fp32


# ---------- main ----------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32","float16","bfloat16"])
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype  = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # HF model & tok
    hf_cfg = AutoConfig.from_pretrained(args.hf_model, revision=args.revision)
    tok = AutoTokenizer.from_pretrained(args.hf_model, revision=args.revision, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    pad_id = resolve_pad_id(hf_cfg, tok)

    hf = AutoModelForCausalLM.from_pretrained(
        args.hf_model, revision=args.revision, dtype=dtype, device_map={"": "cpu"}
    ).eval()
    hf.to(device)
    print(hf.config.attention_bias)  # True/False
# also: do these keys exist?
    sd = hf.state_dict()
    print("has_q_bias", "model.layers.0.self_attn.q_proj.bias" in sd)
    print("has_k_bias", "model.layers.0.self_attn.k_proj.bias" in sd)
    print("has_v_bias", "model.layers.0.self_attn.v_proj.bias" in sd)       
    hf_dev = next(hf.parameters()).device
    
    # Your minimal model
    smcfg = map_hf_config_to_smol(hf_cfg, pad_id)
    print(f"[cfg] d_model={smcfg.d_model} layers={smcfg.n_layer} heads={smcfg.n_head} "
          f"kv_heads={smcfg.n_kv_head} ffn={smcfg.d_ff} rope_theta={smcfg.rope_base} pad_id={smcfg.pad_token_id}")
    mine = SmolLM(smcfg).to(device=device, dtype=dtype).eval()
    mine.load_hf_state_dict(hf.state_dict(), strict=False, verbose=True, dtype=dtype, device=device)

    m = mine  # <-- your minimal model
    sd = hf.state_dict()
    dev = next(m.parameters()).device
    dt  = next(m.parameters()).dtype

    def md(param, key):
        return (param - sd[key].to(dev, dt)).abs().max().item()

    # layer 0 checks
    print("Δ q_proj.w[0]", md(m.blocks[0].attn.q_proj.weight, "model.layers.0.self_attn.q_proj.weight"))
    print("Δ k_proj.w[0]", md(m.blocks[0].attn.k_proj.weight, "model.layers.0.self_attn.k_proj.weight"))
    print("Δ v_proj.w[0]", md(m.blocks[0].attn.v_proj.weight, "model.layers.0.self_attn.v_proj.weight"))

    # single prompt
    enc = tok("The capital of France is", return_tensors="pt")
    input_ids = enc["input_ids"].to(next(mine.parameters()).device)
    with torch.no_grad():
    # embeddings -> LN1 (fp32 inside HF LN)
        x_my = mine.tok_emb(input_ids)
        x_hf = hf.model.embed_tokens(input_ids)
        ln1_my = mine.blocks[0].ln1(x_my)
        ln1_hf = hf.model.layers[0].input_layernorm(x_hf)
        print("LN1 max Δ:", (ln1_my - ln1_hf).abs().max().item())

        # Q/K linear projections BEFORE any RoPE/permutes
        q_my = mine.blocks[0].attn.q_proj(ln1_my)
        k_my = mine.blocks[0].attn.k_proj(ln1_my)
        q_hf = hf.model.layers[0].self_attn.q_proj(ln1_hf)
        k_hf = hf.model.layers[0].self_attn.k_proj(ln1_hf)
        print("Q lin max Δ:", (q_my - q_hf).abs().max().item())
        print("K lin max Δ:", (k_my - k_hf).abs().max().item())
    
    # Inputs
    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None: attention_mask = attention_mask.to(device)
    B, T = input_ids.shape

    # ===== HF layer 0 (manual, no HF masking utils) =====
    x_hf = hf.model.embed_tokens(input_ids).to(dtype)  # [B,T,D]
    cache_position = torch.arange(T, device=device)
    position_ids = cache_position.unsqueeze(0)  # [1,T]
    # cos/sin
    cos, sin = hf.model.rotary_emb(x_hf, position_ids)  # [B,T,hd]
    # mask (additive)
    add_mask = make_additive_causal_mask(attention_mask, T, device, dtype)

    l0 = hf.model.layers[0]
    in0_hf = x_hf.clone()
    ln1_hf = l0.input_layernorm(in0_hf)

    # qkv -> [B,H,T,hd]
    H = hf.config.num_attention_heads
    HK = hf.config.num_key_value_heads
    hd = ln1_hf.shape[-1] // H
    q_hf = l0.self_attn.q_proj(ln1_hf).view(B, T, H, hd).transpose(1, 2)
    k_hf = l0.self_attn.k_proj(ln1_hf).view(B, T, HK, hd).transpose(1, 2)
    v_hf = l0.self_attn.v_proj(ln1_hf).view(B, T, HK, hd).transpose(1, 2)

    q_hf, k_hf = apply_rope(q_hf, k_hf, cos, sin)
    k_hf = repeat_kv(k_hf, H // HK)
    v_hf = repeat_kv(v_hf, H // HK)

    # attention (shared eager impl in fp32)
    scores = (q_hf @ k_hf.transpose(2,3)) * (hd ** -0.5)     # [B,H,T,T]
    scores = scores + add_mask                               # add causal+keypad
    aw = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_hf.dtype)
    attn_out_hf = aw @ v_hf                                  # [B,H,T,hd]
    attn_out_hf = attn_out_hf.transpose(1,2).contiguous().view(B, T, H*hd)
    oproj_hf = l0.self_attn.o_proj(attn_out_hf)
    resid1_hf = in0_hf + oproj_hf
    ln2_hf = l0.post_attention_layernorm(resid1_hf)
    mlp_hf = l0.mlp(ln2_hf)
    out_hf = resid1_hf + mlp_hf

    # ===== Your layer 0 (use the same eager attention path) =====
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    x_my = mine.tok_emb(input_ids)
    in0_my = mine.drop(x_my)
    ln1_my = mine.blocks[0].ln1(in0_my)

    h  = mine.cfg.n_head
    hk = mine.cfg.n_kv_head
    hd = mine.cfg.head_dim()

    q = mine.blocks[0].attn.q_proj(ln1_my).view(B, T, h, hd).permute(0,2,1,3)   # [B,h,T,hd]
    k = mine.blocks[0].attn.k_proj(ln1_my).view(B, T, hk, hd).permute(0,2,1,3)
    v = mine.blocks[0].attn.v_proj(ln1_my).view(B, T, hk, hd).permute(0,2,1,3)
    # RoPE (your module expects [B,T,H,hd], so permute back temporarily)
    q_r, k_r = mine.blocks[0].attn.rope(q.permute(0,2,1,3), k.permute(0,2,1,3), positions=positions)
    q_r, k_r = q_r.permute(0,2,1,3), k_r.permute(0,2,1,3)
    if hk != h:
        rep = h // hk
        k_r = repeat_kv(k_r, rep)
        v   = repeat_kv(v,   rep)

    # same eager attention in fp32
    scores_my = (q_r.float() @ k_r.float().transpose(2,3)) * (hd ** -0.5)
    scores_my = scores_my + add_mask
    aw_my = F.softmax(scores_my, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_out_my = aw_my @ v.float()
    attn_out_my = attn_out_my.to(q.dtype).transpose(1,2).contiguous().view(B, T, h*hd)
    oproj_my = mine.blocks[0].attn.o_proj(attn_out_my)
    resid1_my = in0_my + mine.blocks[0].attn.resid_dropout(oproj_my)
    ln2_my = mine.blocks[0].ln2(resid1_my)
    mlp_my = mine.blocks[0].mlp(ln2_my)
    out_my = resid1_my + mlp_my

    # -------- diffs --------
    def report(name, a, b):
        print(f"{name:<12} max|Δ|={maxdiff(a,b):.6g}")

    print("stage diffs (max abs):")
    report("LN1", ln1_my, ln1_hf)
    report("Q (all)", q.permute(0,2,1,3), q_hf.permute(0,2,1,3))
    report("K (all)", k.permute(0,2,1,3), k_hf.permute(0,2,1,3)[:, :, :hk, :])
    report("V (all)", v.permute(0,2,1,3)[:, :, :hk, :], v_hf.permute(0,2,1,3)[:, :, :hk, :])
    report("Q_rope", q_r.permute(0,2,1,3), q_hf.permute(0,2,1,3))
    report("K_rope", k_r.permute(0,2,1,3)[:, :, :hk, :], k_hf.permute(0,2,1,3)[:, :, :hk, :])
    report("AttnOut", attn_out_my, attn_out_hf)
    report("O_proj", oproj_my, oproj_hf)
    report("Resid1", resid1_my, resid1_hf)
    report("LN2", ln2_my, ln2_hf)
    report("MLP(out)", mlp_my, mlp_hf)
    report("Layer0 out", out_my, out_hf)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
