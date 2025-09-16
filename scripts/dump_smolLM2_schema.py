#!/usr/bin/env python3
"""
Dump the *original* SmolLM2 (HF) checkpoint schema so you can map it into
the ``models/smolLM2/`` package. Inspired by nanoGPT's simple tooling.

What it does:
 - Loads a SmolLM2 (or any GPT-style) HF checkpoint from Hub or local path
 - Saves:
   * config.json         — the HF config dict
   * state_schema.json   — {parameter_name: shape}
   * state_shapes.tsv    — tabular view (name\tshape\tnumel)
   * module_tree.txt     — nested module tree with parameter counts
   * keymap_template.yaml— suggested key-name mapping to models/smolLM2/

Usage
-----
python scripts/dump_smolLM2_schema.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --out artifacts/smolllm2_135m_schema

Or with a local path:
python scripts/dump_smolLM2_schema.py --model /path/to/smollm2 --out artifacts/local_schema

Notes
-----
- No network is required if you already have the model locally and pass --local-files-only.
- This does not modify any weights; it's a read-only inspector.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo id or local directory path")
    p.add_argument("--revision", default=None, help="Optional HF revision/tag/commit")
    p.add_argument("--out", default="artifacts/smollm2_schema", help="Output directory")
    p.add_argument("--local-files-only", action="store_true", help="Do not download from Hub")
    p.add_argument("--trust-remote-code", action="store_true", help="Allow custom model classes if needed")
    p.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32"], help="Load dtype")
    return p.parse_args()


def pick_dtype(name: str):
    if name == "auto":
        return None
    return getattr(torch, name)


def safe_shape(t: torch.Tensor) -> List[int]:
    try:
        return list(t.shape)
    except Exception:
        return []


def numel_str(n: int) -> str:
    if n < 1_000: return str(n)
    if n < 1_000_000: return f"{n/1_000:.1f}K"
    if n < 1_000_000_000: return f"{n/1_000_000:.2f}M"
    return f"{n/1_000_000_000:.2f}B"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_state_schema(out_dir: Path, state: Dict[str, torch.Tensor]) -> None:
    schema = {k: safe_shape(v) for k, v in state.items()}
    write_json(out_dir / "state_schema.json", schema)
    # TSV for quick grep
    lines = ["name\tshape\tnumel\tdtype"]
    for k, v in state.items():
        lines.append(f"{k}\t{list(v.shape)}\t{v.numel()}\t{str(v.dtype).replace('torch.','')}")
    (out_dir / "state_shapes.tsv").write_text("\n".join(lines))


def save_module_tree(out_dir: Path, model: torch.nn.Module) -> None:
    lines: List[str] = []

    def recurse(mod: torch.nn.Module, name: str, prefix: str = ""):
        # Count parameters for this module (non-recursive and recursive)
        own = sum(p.numel() for p in mod.parameters(recurse=False))
        total = sum(p.numel() for p in mod.parameters())
        cls = mod.__class__.__name__
        lines.append(f"{prefix}{name}  <{cls}>  params: own={numel_str(own)} total={numel_str(total)}")
        children = list(mod.named_children())
        for i, (n, ch) in enumerate(children):
            is_last = i == len(children) - 1
            branch = "└─ " if is_last else "├─ "
            indent = "   " if is_last else "│  "
            recurse(ch, n, prefix + branch)
            # add a spacer line for readability
            if i != len(children) - 1:
                lines.append(prefix + indent)

    recurse(model, name="model", prefix="")
    (out_dir / "module_tree.txt").write_text("\n".join(lines))


def detect_smol_style_keys(state: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    keys = state.keys()
    def any_has(sub: str) -> bool:
        return any(sub in k for k in keys)
    return {
        "has_gate_proj": any_has("mlp.gate_proj"),
        "has_up_proj": any_has("mlp.up_proj"),
        "has_down_proj": any_has("mlp.down_proj"),
        "has_input_ln": any_has("input_layernorm"),
        "has_post_ln": any_has("post_attention_layernorm"),
        "has_o_proj": any_has("self_attn.o_proj"),
        "has_qkv_bias": any(k.endswith(".bias") and ("q_proj" in k or "k_proj" in k or "v_proj" in k) for k in keys),
    }


def make_keymap_template(out_dir: Path, state: Dict[str, torch.Tensor]) -> None:
    """Emit a YAML-ish template mapping HF keys -> your models/smolLM2 package keys.
    This is a *suggestion*; adjust to your checkpoint.
    """
    hints = detect_smol_style_keys(state)
    gate_note = "(concat gate_proj & up_proj along dim=-1 to models/MLP.in_proj)" if hints["has_gate_proj"] else "(use up_proj -> mlp.in_proj)"
    lines = [
        "# keymap_template.yaml — edit and use for manual remap in models/smolLM2/",
        "# Left side: HF checkpoint key, Right side: your module key",
        "",
        "# Embeddings / final norm / head",
        "model.embed_tokens.weight: tok_emb.weight",
        "model.norm.weight: ln_f.weight",
        "lm_head.weight: lm_head.weight   # ignored if you tie embeddings",
        "",
        "# Per-layer mapping (layer index i)",
        "# input LN",
        "model.layers.{i}.input_layernorm.weight: blocks.{i}.ln_attn.weight",
        "# attention projections",
        "model.layers.{i}.self_attn.q_proj.weight: blocks.{i}.attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight: blocks.{i}.attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight: blocks.{i}.attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight: blocks.{i}.attn.o_proj.weight",
        "# post-attn LN",
        "model.layers.{i}.post_attention_layernorm.weight: blocks.{i}.ln_mlp.weight",
        "# MLP (SwiGLU) {gate_note}",
        "model.layers.{i}.mlp.gate_proj.weight + model.layers.{i}.mlp.up_proj.weight -> blocks.{i}.mlp.in_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight: blocks.{i}.mlp.out_proj.weight",
        "",
        "# If your checkpoint has biases, map *.bias similarly.",
    ]
    (out_dir / "keymap_template.yaml").write_text("\n".join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dump] loading config: {args.model}")
    cfg = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    # Save raw config
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        f.write(cfg.to_json_string())

    print(f"[dump] loading weights (dtype={args.dtype})…")
    dtype = pick_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map=None,
    )
    state = model.state_dict()

    print(f"[dump] saving state schema with {len(state)} tensors…")
    save_state_schema(out_dir, state)

    print("[dump] writing module tree…")
    save_module_tree(out_dir, model)

    print("[dump] writing keymap template…")
    make_keymap_template(out_dir, state)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    write_json(out_dir / "summary.json", meta)

    print("[done] artifacts written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
