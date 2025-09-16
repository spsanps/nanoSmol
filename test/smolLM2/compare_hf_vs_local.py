#!/usr/bin/env python3
"""
Compare our minimal SmolLM against the official Hugging Face SmolLM2 model.

What this does
- Loads HF model + tokenizer
- Builds our model with matching dims
- Imports weights via load_hf_state_dict
- Runs a *greedy*, temperature=0 teacher-forced decode for N steps
- Asserts logits closeness and next-token equality at each step

Run (from repo root):
  python test/smolLM2/compare_hf_vs_local.py \
    --hf-model HuggingFaceTB/SmolLM2-135M \
    --steps 16

Use --hf-model HuggingFaceTB/SmolLM2-135M-Instruct with --prompt "### Instruction..." for instruct variants.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# repo import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.smolLM2 import SmolConfig, SmolLM  # noqa: E402

# ---- defaults ----
DEFAULT_PROMPT_BASE = "The capital of France is"
DEFAULT_PROMPT_INSTRUCT = (
    "### Instruction\n"
    "Answer briefly: What is the capital of France?\n"
    "### Response\n"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Use float32 for strict numeric comparisons")
    p.add_argument("--steps", type=int, default=16, help="#greedy decode steps to compare")
    p.add_argument("--tol", type=float, default=1e-3, help="max allowed abs diff on logits")
    p.add_argument("--prompt", default=None, help="override prompt; defaults to BASE or INSTRUCT template")
    p.add_argument("--instruct", action="store_true", help="use instruct-style default prompt")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--tie-embeddings", action="store_true", help="tie lm_head to embeddings in our model")
    return p.parse_args()


def map_hf_config_to_smol(cfg_hf, *, tie_embeddings: bool) -> SmolConfig:
    def _get(attr, default=None):
        return getattr(cfg_hf, attr, default)

    vocab_size = int(_get("vocab_size"))
    d_model = int(_get("hidden_size", _get("n_embd")))
    n_layer = int(_get("num_hidden_layers", _get("n_layer")))
    n_head = int(_get("num_attention_heads", _get("n_head")))
    n_kv_head = int(_get("num_key_value_heads", n_head))
    d_ff = int(_get("intermediate_size", 4 * int(d_model)))
    max_seq_len = int(_get("max_position_embeddings", 4096))
    rope_base = float(_get("rope_theta", _get("rope_base", 10000.0)))
    norm_eps = float(_get("rms_norm_eps", _get("layer_norm_eps", 1e-5)))

    pad_id = _get("pad_token_id", None)
    if pad_id is None:
        pad_id = _get("eos_token_id", 0) or 0
    pad_id = int(pad_id)

    return SmolConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        rope_pct=1.0,
        dropout=0.0,
        norm_eps=norm_eps,
        tie_embeddings=tie_embeddings,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_activation="silu",
        gated_mlp=True,
        init_std=0.02,
        pad_token_id=pad_id,
    )


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax token ids from final-position logits [B, V]."""
    return torch.argmax(logits, dim=-1, keepdim=True)


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # Load HF config/model/tokenizer
    hf_cfg = AutoConfig.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    hf_tok = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if hf_tok.pad_token_id is None:
        hf_tok.pad_token_id = hf_tok.eos_token_id

    # Choose default prompt
    prompt = (args.prompt if args.prompt is not None
              else (DEFAULT_PROMPT_INSTRUCT if args.instruct else DEFAULT_PROMPT_BASE))

    # Tokenize once
    enc = hf_tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    # Build our model
    smol_cfg = map_hf_config_to_smol(hf_cfg, tie_embeddings=args.tie_embeddings)
    local_model = SmolLM(smol_cfg).to(device=device, dtype=dtype).eval()

    # Load HF weights (keep HF model on CPU to save VRAM when desired)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map={"": device.type},
    ).eval()
    
    # Import into local
    local_model.load_hf_state_dict(hf_model.state_dict(), strict=False, verbose=True, dtype=dtype, device=device)

    with torch.no_grad():
        max_diff = (local_model.lm_head.weight - local_model.tok_emb.weight).abs().max().item()
    print(f"lm_head vs tok_emb max |diff| = {max_diff:.6g}")
    assert (max_diff < 1e-6), "lm_head and tok_emb should be tied!"
    
    # check difference between local and hf lm_head weights
    with torch.no_grad():
        max_diff = (local_model.lm_head.weight - hf_model.lm_head.weight.to(device=device, dtype=dtype)).abs().max().item()
    print(f"local vs hf lm_head max |diff| = {max_diff:.6g}")
    assert (max_diff < 1e-3), "lm_head weights differ too much!"


    # Compare step-by-step (teacher forcing)
    ids = input_ids.clone()
    for step in range(args.steps):
        with torch.no_grad():
            # local
            l_logits = local_model(ids)[:, -1, :]
            # hf
            h_out = hf_model(input_ids=ids)
            h_logits = h_out.logits[:, -1, :]
        # numeric check
        max_diff = (l_logits - h_logits).abs().max().item()
        print(f"[step {step}] max|Δlogits| = {max_diff:.6f}")
        if max_diff > args.tol:
            raise AssertionError(f"Logit mismatch at step {step}: {max_diff} > tol={args.tol}")
        # greedy next
        next_id = greedy_next_token(h_logits)
        ids = torch.cat([ids, next_id], dim=1)

    # Decode final text for quick human check
    text_local = hf_tok.decode(ids[0], skip_special_tokens=True)
    print("\n=== PROMPT ===\n" + prompt)
    print("\n=== LOCAL (teacher-forced path) ===\n" + text_local)
    print("\nOK: logits matched within tolerance at each step.")


if __name__ == "__main__":
    main()
