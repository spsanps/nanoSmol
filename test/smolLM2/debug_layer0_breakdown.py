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
        n_layers=int(_get("num_hidden_layers", _get("n_layer"))),
        n_heads=int(_get("num_attention_heads", _get("n_head"))),
        n_kv_heads=int(_get("num_key_value_heads", _get("num_attention_heads"))),
        d_ff=int(_get("intermediate_size", 4 * int(_get("hidden_size", _get("n_embd"))))),
        max_seq_len=int(_get("max_position_embeddings", 4096)),
        rope_theta=float(_get("rope_theta", _get("rope_base", 10000.0))),
        dropout=0.0,
        norm_eps=float(_get("rms_norm_eps", _get("layer_norm_eps", 1e-6))),
        tie_embeddings=bool(_get("tie_word_embeddings", True)),
        init_std=0.02,
        pad_token_id=pad_id,
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
        ln1_my = mine.blocks[0].ln_attn(x_my)
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
    in0_my = x_my  # Remove drop reference since SmolLM doesn't have this attribute
    ln1_my = mine.blocks[0].ln_attn(in0_my)

    h  = mine.cfg.n_head
    hk = mine.cfg.n_kv_head
    hd = mine.cfg.head_dim()

    # -------- Staged comparisons with proper shapes --------
    def report(name, a, b):
        print(f"{name:<20} max|Δ|={maxdiff(a,b):.6g}")

    print("stage diffs (max abs):")
    report("LN1", ln1_my, ln1_hf)

    # ----- Stage A: pre-RoPE, pre-permute -----
    q_my_A = mine.blocks[0].attn.q_proj(ln1_my).view(B, T, h,  hd)   # [B,T,h,hd]
    k_my_A = mine.blocks[0].attn.k_proj(ln1_my).view(B, T, hk, hd)   # [B,T,hk,hd]
    v_my_A = mine.blocks[0].attn.v_proj(ln1_my).view(B, T, hk, hd)   # [B,T,hk,hd]
    
    # Get fresh HF projections (before any RoPE/permute operations)
    q_hf_A = l0.self_attn.q_proj(ln1_hf).view(B, T, H, hd)   # [B,T,H,hd]
    k_hf_A = l0.self_attn.k_proj(ln1_hf).view(B, T, HK, hd)  # [B,T,HK,hd]
    v_hf_A = l0.self_attn.v_proj(ln1_hf).view(B, T, HK, hd)  # [B,T,HK,hd]

    # Only compare if dimensions match (h==H and hk==HK)
    if h == H and hk == HK:
        report("Q pre-RoPE", q_my_A, q_hf_A)
        report("K pre-RoPE", k_my_A, k_hf_A)
        report("V pre-RoPE", v_my_A, v_hf_A)
    else:
        print(f"Skipping Stage A comparison: head count mismatch (mine: h={h}, hk={hk} vs HF: H={H}, HK={HK})")

    # ----- Stage B: post-RoPE, pre-repeat -----
    q_my_B, k_my_B = mine.blocks[0].attn.rotary(q_my_A, k_my_A, positions=positions)   # [B,T,h,hd], [B,T,hk,hd]

    # Apply RoPE to fresh HF projections
    q_hf_A_permuted = q_hf_A.permute(0,2,1,3)  # [B,H,T,hd] for apply_rope
    k_hf_A_permuted = k_hf_A.permute(0,2,1,3)  # [B,HK,T,hd] for apply_rope
    q_hf_B_permuted, k_hf_B_permuted = apply_rope(q_hf_A_permuted, k_hf_A_permuted, cos, sin)
    q_hf_B = q_hf_B_permuted.permute(0,2,1,3)  # back to [B,T,H,hd]
    k_hf_B = k_hf_B_permuted.permute(0,2,1,3)  # back to [B,T,HK,hd]

    # Only compare if dimensions match
    if h == H and hk == HK:
        report("Q post-RoPE", q_my_B, q_hf_B)
        report("K post-RoPE", k_my_B, k_hf_B)
    else:
        print(f"Skipping Stage B comparison: head count mismatch (mine: h={h}, hk={hk} vs HF: H={H}, HK={HK})")

    # ----- Stage C: post-RoPE, post-repeat (GQA expanded to H heads) -----
    def repeat_kv_for_compare(x, rep):  # x: [B,T,HK,hd] -> [B,T,H,hd]
        if rep == 1: return x
        return x.repeat_interleave(rep, dim=2)

    rep_my = h // hk
    rep_hf = H // HK
    k_my_C  = repeat_kv_for_compare(k_my_B, rep_my).permute(0,2,1,3)  # [B,H,T,hd]
    v_my_C  = repeat_kv_for_compare(v_my_A, rep_my).permute(0,2,1,3)  # [B,H,T,hd]
    k_hf_C  = repeat_kv_for_compare(k_hf_B, rep_hf).permute(0,2,1,3)  # [B,H,T,hd]
    v_hf_C  = repeat_kv_for_compare(v_hf_A, rep_hf).permute(0,2,1,3)  # [B,H,T,hd]
    q_my_C  = q_my_B.permute(0,2,1,3)                     # [B,H,T,hd]
    q_hf_C  = q_hf_B.permute(0,2,1,3)                     # [B,H,T,hd]

    # All tensors should now be [B,H,T,hd] format
    report("Q post-repeat", q_my_C, q_hf_C)
    report("K post-repeat", k_my_C, k_hf_C)
    report("V post-repeat", v_my_C, v_hf_C)

    # Continue with attention computation using the properly shaped tensors
    # same eager attention in fp32
    scores_my = (q_my_C.float() @ k_my_C.float().transpose(2,3)) * (hd ** -0.5)
    scores_my = scores_my + add_mask
    aw_my = F.softmax(scores_my, dim=-1, dtype=torch.float32).to(q_my_C.dtype)
    attn_out_my = aw_my @ v_my_C.float()
    attn_out_my = attn_out_my.to(q_my_C.dtype).transpose(1,2).contiguous().view(B, T, h*hd)
    oproj_my = mine.blocks[0].attn.o_proj(attn_out_my)
    resid1_my = in0_my + mine.blocks[0].attn.resid_dropout(oproj_my)
    ln2_my = mine.blocks[0].ln_mlp(resid1_my)
    mlp_my = mine.blocks[0].mlp(ln2_my)
    out_my = resid1_my + mlp_my

    # Final stage comparisons
    report("AttnOut", attn_out_my, attn_out_hf)
    report("O_proj", oproj_my, oproj_hf)
    report("Resid1", resid1_my, resid1_hf)
    report("LN2", ln2_my, ln2_hf)
    report("MLP(out)", mlp_my, mlp_hf)
    report("Layer0 out", out_my, out_hf)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
