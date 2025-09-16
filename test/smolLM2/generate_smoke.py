#!/usr/bin/env python3
"""
Quick generate test: load our minimal SmolLM model, import weights from a
Hugging Face SmolLM2 checkpoint, then generate a short sample.

Suggested filename: test/smolLM2/generate_demo.py
Run from repo root:
  python test/smolLM2/generate_smoke.py \
    --hf-model HuggingFaceTB/SmolLM2-135M

Notes
- The default prompt is an *instruction-style* template. Override with --prompt as needed.
- Works with base or *-Instruct checkpoints (instruction-tuned will follow the template better).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Make sure we can import models.smolLM2 when run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.smolLM2 import SmolConfig, SmolLM  # noqa: E402

# ---- Default test prompt (instruction style) ----
DEFAULT_PROMPT = (
    "The capital of France is"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M",
                   help="HF repo id or local path to SmolLM2 checkpoint")
    p.add_argument("--revision", default=None, help="Optional HF revision/tag/commit")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                   help="dtype to load weights into")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="device for the model")
    p.add_argument("--prompt", default=None, help="override prompt text (defaults to instruction-style)")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def map_hf_config_to_smol(cfg_hf) -> SmolConfig:
    """Translate HF SmolLM2/LLaMA-like config into our SmolConfig."""
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
        n_layers=n_layer,
        n_heads=n_head,
        n_kv_heads=n_kv_head,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        rope_theta=rope_base,
        dropout=0.0,
        norm_eps=norm_eps,
        tie_embeddings=False,
        init_std=0.02,
        pad_token_id=pad_id,
    )


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[schema] loading HF config: {args.hf_model}")
    hf_cfg = AutoConfig.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    smol_cfg = map_hf_config_to_smol(hf_cfg)

    print("[init] building minimal SmolLM with:")
    print(f"       d_model={smol_cfg.d_model}, layers={smol_cfg.n_layer}, heads={smol_cfg.n_head}, kv_heads={smol_cfg.n_kv_head}, d_ff={smol_cfg.d_ff}")
    model = SmolLM(smol_cfg).to(device=device, dtype=dtype)

    print("[weights] loading HF weights…")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        #device_map="cpu",
    )
    state = hf_model.state_dict()

    model.load_hf_state_dict(state, strict=False, verbose=True, dtype=dtype, device=device)
    model.eval()

    tok = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
    
    # for instruction-style prompts, we could do:
    # messages = [
    #     {"role": "system", "content": "You are a concise, helpful assistant."},
    #     {"role": "user", "content": "Write a short, friendly greeting for a developer who just set up a minimal SmolLM baseline."}
    # ]
    # prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    print("[gen] generating…")
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    text = tok.decode(out_ids[0], skip_special_tokens=True)

    print(" \
===== PROMPT ===== \
" + prompt)
    print(" \
===== OUTPUT ===== \
" + text)


if __name__ == "__main__":
    main()
