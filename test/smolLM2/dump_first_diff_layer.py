#!/usr/bin/env python3
"""
Find the first layer where your minimal SmolLM2 output diverges from HF Llama/SmolLM2.

Usage:
  python dump_first_diff_layer.py --hf-model HuggingFaceTB/SmolLM2-135M --prompt "The capital of France is"
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- adjust this import path to match your repo layout ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.smolLM2 import SmolConfig, SmolLM  # noqa: E402

# -------------------------
# Config mapping (HF -> yours)
# -------------------------
def map_hf_config_to_smol(cfg_hf) -> SmolConfig:
    def _get(attr, default=None): return getattr(cfg_hf, attr, default)
    vocab_size = int(_get("vocab_size"))
    d_model    = int(_get("hidden_size", _get("n_embd")))
    n_layer    = int(_get("num_hidden_layers", _get("n_layer")))
    n_head     = int(_get("num_attention_heads", _get("n_head")))
    n_kv_head  = int(_get("num_key_value_heads", n_head))
    d_ff       = int(_get("intermediate_size", 4 * int(d_model)))
    max_seq_len = int(_get("max_position_embeddings", 4096))
    rope_base   = float(_get("rope_theta", _get("rope_base", 10000.0)))
    norm_eps    = float(_get("rms_norm_eps", _get("layer_norm_eps", 1e-6)))  # HF often uses 1e-6
    tie_word_embeddings = bool(_get("tie_word_embeddings", True))
    pad_id = _get("pad_token_id", None)
    if pad_id is None:
        pad_id = _get("eos_token_id", 0) or 0
    pad_id = int(pad_id)
    return SmolConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layer,
        n_heads=n_head,
        n_kv_heads=n_kv_head,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        rope_theta=rope_base,
        dropout=0.0,
        norm_eps=norm_eps,
        tie_embeddings=tie_word_embeddings,
        init_std=0.02,
        pad_token_id=pad_id,
    )

# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def run_minimal_collect_layers(model: SmolLM, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    """Return list of hidden states after each block (post-MLP residual), same shape as HF per-layer outputs."""
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    # positions like HF (arange), and SDPA key-padding mask for your model
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    attn_mask = None
    if attention_mask is not None:
        attn_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)  # [B,1,1,T_k]

    x = model.tok_emb(input_ids)
    x = model.drop(x)
    outs = []
    for blk in model.blocks:
        x = blk(x, attn_mask=attn_mask, position_ids=positions)
        outs.append(x.clone())
    x = model.norm_out(x)
    return outs, x  # per-layer outputs, and final normalized last_hidden_state

@torch.no_grad()
def run_hf_collect_layers(hf_model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    """Return list of hidden states after each HF decoder layer (post-MLP residual), and final."""
    hf_model.eval()
    # ask HF to give us hidden states
    out = hf_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # HF hidden_states: [embeddings, layer1_out, layer2_out, ..., layerN_out]
    hs = list(out.hidden_states)
    per_layer = hs[1:]              # post-MLP residual after each layer
    final = hf_model.model.norm(per_layer[-1]) if len(per_layer) else hf_model.model.norm(hs[-1])
    return per_layer, final

def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).abs()
    return float(diff.max().item()), float(diff.mean().item())

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--tol", type=float, default=1e-3, help="per-layer max|Δ| tolerance")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] HF config/model: {args.hf_model}")
    hf_cfg = AutoConfig.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map={"": "cpu"} if use_cuda else {"": "cpu"},  # keep HF on CPU to save GPU RAM
    )

    tok = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Build your minimal model from HF cfg and load weights
    smol_cfg = map_hf_config_to_smol(hf_cfg)
    print(f"[init] minimal: d_model={smol_cfg.d_model}, layers={smol_cfg.n_layer}, "
          f"heads={smol_cfg.n_head}, kv_heads={smol_cfg.n_kv_head}, ffn={smol_cfg.d_ff}, "
          f"rope_theta={smol_cfg.rope_base}, vocab={smol_cfg.vocab_size}")
    mine = SmolLM(smol_cfg).to(device=device, dtype=dtype)

    # Copy weights from HF to your model
    with torch.no_grad():
        mine.load_hf_state_dict(hf_model.state_dict(), strict=False, verbose=True, dtype=dtype, device=device)

    # Encode prompt (no padding needed for single example; attention_mask still ok)
    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Run both and collect per-layer outputs
    print("[run] collecting per-layer outputs…")
    ours_layers, ours_final = run_minimal_collect_layers(mine, input_ids, attention_mask)
    hf_layers, hf_final = run_hf_collect_layers(hf_model.to(device), input_ids, attention_mask)

    # Compare per-layer
    print("\n[cmp] per-layer max|Δ|, mean|Δ|")
    first_bad = None
    for i, (a, b) in enumerate(zip(ours_layers, hf_layers)):
        m, avg = compare_tensors(a, b)
        flag = " <-- DIFF" if m > args.tol else ""
        print(f"  layer {i:02d}: max={m:.6g}  mean={avg:.6g}{flag}")
        if first_bad is None and m > args.tol:
            first_bad = i

    # Final norm / logits sanity
    m_final, avg_final = compare_tensors(ours_final, hf_final)
    print(f"\n[cmp] final normed state: max={m_final:.6g}  mean={avg_final:.6g}")

    # Optionally compare logits for one step
    with torch.no_grad():
        logits_hf = hf_model(input_ids.to(device)).logits
        logits_ours = mine(input_ids.to(device))
        m_logits, avg_logits = compare_tensors(logits_ours, logits_hf)
    print(f"[cmp] logits (full seq): max={m_logits:.6g}  mean={avg_logits:.6g}")

    if first_bad is None:
        print("\n[ok] No per-layer diffs above tolerance. (Try a stricter --tol or fp32.)")
    else:
        print(f"\n[first-diff] First layer exceeding tolerance: layer {first_bad}")
        # quick extra clues
        with torch.no_grad():
            # Check whether your lm_head equals embedding (tied behavior)
            if mine.lm_head.weight.shape == mine.tok_emb.weight.shape:
                head_diff = (mine.lm_head.weight - mine.tok_emb.weight).abs().max().item()
                print(f"[hint] lm_head vs tok_emb max|Δ|: {head_diff:.6g}")

            # RoPE head_dim / theta
            print(f"[hint] head_dim={smol_cfg.d_model // smol_cfg.n_head}, rope_theta={smol_cfg.rope_base}, "
                  f"kv_heads={smol_cfg.n_kv_head}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
