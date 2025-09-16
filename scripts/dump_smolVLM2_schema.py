#!/usr/bin/env python3
"""
Dump the *original* SmolVLM2 (HF) checkpoint schema so you can map it into
a new ``models/smolVLM2/`` package. Analyzes the vision-language architecture.

What it does:
 - Loads a SmolVLM2 checkpoint from Hub or local path
 - Saves:
   * config.json         — the HF config dict
   * state_schema.json   — {parameter_name: shape}
   * state_shapes.tsv    — tabular view (name\tshape\tnumel)
   * module_tree.txt     — nested module tree with parameter counts
   * keymap_template.yaml— suggested key-name mapping to models/smolVLM2/
   * architecture_analysis.json — detailed analysis of the VL architecture

Usage
-----
python scripts/dump_smolVLM2_schema.py \
  --model HuggingFaceTB/SmolVLM2-1.7B \
  --out artifacts/smolvlm2_1_7b_schema

Or with a local path:
python scripts/dump_smolVLM2_schema.py --model /path/to/smolvlm2 --out artifacts/local_schema

Notes
-----
- No network is required if you already have the model locally and pass --local-files-only.
- This does not modify any weights; it's a read-only inspector.
- Analyzes both vision and language components of the VL model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo id or local directory path")
    p.add_argument("--revision", default=None, help="Optional HF revision/tag/commit")
    p.add_argument("--out", default="artifacts/smolvlm2_schema", help="Output directory")
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


def analyze_vlm_architecture(config: Any, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze the Vision-Language Model architecture components."""
    analysis = {
        "model_type": getattr(config, "model_type", "unknown"),
        "architecture_type": getattr(config, "architectures", ["unknown"]),
        "components": {},
        "parameter_breakdown": {},
        "key_patterns": {},
        "vision_config": {},
        "language_config": {},
        "multimodal_config": {}
    }
    
    # Extract vision configuration
    if hasattr(config, "vision_config"):
        vision_cfg = config.vision_config
        if hasattr(vision_cfg, "to_dict"):
            analysis["vision_config"] = vision_cfg.to_dict()
        else:
            analysis["vision_config"] = dict(vision_cfg) if vision_cfg else {}
    
    # Extract language configuration  
    if hasattr(config, "text_config"):
        text_cfg = config.text_config
        if hasattr(text_cfg, "to_dict"):
            analysis["language_config"] = text_cfg.to_dict()
        else:
            analysis["language_config"] = dict(text_cfg) if text_cfg else {}
    
    # Extract multimodal projector config
    for attr in ["mm_projector_lr", "mm_projector_type", "mm_hidden_size", "mm_vision_select_layer", "mm_vision_select_feature"]:
        if hasattr(config, attr):
            analysis["multimodal_config"][attr] = getattr(config, attr)
    
    # Analyze parameter patterns
    keys = list(state.keys())
    
    # Group parameters by component
    vision_params = [k for k in keys if any(pattern in k for pattern in ["vision", "visual", "image", "patch", "clip"])]
    language_params = [k for k in keys if any(pattern in k for pattern in ["language", "text", "lm_head", "embed_tokens", "layers"])]
    projector_params = [k for k in keys if any(pattern in k for pattern in ["projector", "mm_projector", "multi_modal"])]
    
    analysis["components"] = {
        "vision_parameters": len(vision_params),
        "language_parameters": len(language_params), 
        "projector_parameters": len(projector_params),
        "total_parameters": len(keys)
    }
    
    # Parameter count breakdown
    vision_count = sum(state[k].numel() for k in vision_params)
    language_count = sum(state[k].numel() for k in language_params)
    projector_count = sum(state[k].numel() for k in projector_params)
    total_count = sum(p.numel() for p in state.values())
    
    analysis["parameter_breakdown"] = {
        "vision_params": vision_count,
        "language_params": language_count,
        "projector_params": projector_count,
        "other_params": total_count - vision_count - language_count - projector_count,
        "total_params": total_count,
        "vision_percentage": (vision_count / total_count * 100) if total_count > 0 else 0,
        "language_percentage": (language_count / total_count * 100) if total_count > 0 else 0,
        "projector_percentage": (projector_count / total_count * 100) if total_count > 0 else 0
    }
    
    # Identify key patterns
    patterns = {
        "attention_patterns": [k for k in keys if "attn" in k or "attention" in k],
        "mlp_patterns": [k for k in keys if "mlp" in k or "feed_forward" in k],
        "norm_patterns": [k for k in keys if "norm" in k or "ln" in k],
        "embedding_patterns": [k for k in keys if "embed" in k],
        "projection_patterns": [k for k in keys if "proj" in k],
        "bias_patterns": [k for k in keys if k.endswith(".bias")],
        "vision_encoder_patterns": [k for k in keys if "vision" in k and ("encoder" in k or "layer" in k)],
        "language_decoder_patterns": [k for k in keys if "language" in k and ("decoder" in k or "layer" in k)]
    }
    
    analysis["key_patterns"] = {name: len(params) for name, params in patterns.items()}
    
    return analysis


def make_vlm_keymap_template(out_dir: Path, state: Dict[str, torch.Tensor], analysis: Dict[str, Any]) -> None:
    """Generate a keymap template for SmolVLM2 architecture."""
    lines = [
        "# keymap_template.yaml — SmolVLM2 Vision-Language Model mapping",
        "# Left side: HF checkpoint key, Right side: your module key",
        "",
        "# ===== VISION ENCODER =====",
        "# Vision embeddings and patch projection",
        "vision_model.embeddings.patch_embedding.weight: vision_encoder.patch_embed.weight",
        "vision_model.embeddings.position_embedding.weight: vision_encoder.pos_embed.weight", 
        "vision_model.embeddings.class_embedding: vision_encoder.cls_token",
        "",
        "# Vision transformer layers (layer index i)",
        "vision_model.encoder.layers.{i}.layer_norm1.weight: vision_encoder.blocks.{i}.ln1.weight",
        "vision_model.encoder.layers.{i}.layer_norm1.bias: vision_encoder.blocks.{i}.ln1.bias",
        "vision_model.encoder.layers.{i}.self_attn.q_proj.weight: vision_encoder.blocks.{i}.attn.q_proj.weight", 
        "vision_model.encoder.layers.{i}.self_attn.k_proj.weight: vision_encoder.blocks.{i}.attn.k_proj.weight",
        "vision_model.encoder.layers.{i}.self_attn.v_proj.weight: vision_encoder.blocks.{i}.attn.v_proj.weight",
        "vision_model.encoder.layers.{i}.self_attn.out_proj.weight: vision_encoder.blocks.{i}.attn.o_proj.weight",
        "vision_model.encoder.layers.{i}.layer_norm2.weight: vision_encoder.blocks.{i}.ln2.weight",
        "vision_model.encoder.layers.{i}.layer_norm2.bias: vision_encoder.blocks.{i}.ln2.bias",
        "vision_model.encoder.layers.{i}.mlp.fc1.weight: vision_encoder.blocks.{i}.mlp.fc1.weight",
        "vision_model.encoder.layers.{i}.mlp.fc1.bias: vision_encoder.blocks.{i}.mlp.fc1.bias",
        "vision_model.encoder.layers.{i}.mlp.fc2.weight: vision_encoder.blocks.{i}.mlp.fc2.weight",
        "vision_model.encoder.layers.{i}.mlp.fc2.bias: vision_encoder.blocks.{i}.mlp.fc2.bias",
        "",
        "# Vision final norm",
        "vision_model.post_layernorm.weight: vision_encoder.ln_post.weight",
        "vision_model.post_layernorm.bias: vision_encoder.ln_post.bias",
        "",
        "# ===== MULTIMODAL PROJECTOR =====", 
        "multi_modal_projector.linear_1.weight: mm_projector.linear1.weight",
        "multi_modal_projector.linear_1.bias: mm_projector.linear1.bias",
        "multi_modal_projector.linear_2.weight: mm_projector.linear2.weight", 
        "multi_modal_projector.linear_2.bias: mm_projector.linear2.bias",
        "",
        "# ===== LANGUAGE MODEL =====",
        "# Token embeddings and final norm",
        "language_model.model.embed_tokens.weight: language_model.tok_emb.weight",
        "language_model.model.norm.weight: language_model.ln_f.weight",
        "language_model.lm_head.weight: language_model.lm_head.weight",
        "",
        "# Language transformer layers (layer index i)", 
        "language_model.model.layers.{i}.input_layernorm.weight: language_model.blocks.{i}.ln_attn.weight",
        "language_model.model.layers.{i}.self_attn.q_proj.weight: language_model.blocks.{i}.attn.q_proj.weight",
        "language_model.model.layers.{i}.self_attn.k_proj.weight: language_model.blocks.{i}.attn.k_proj.weight", 
        "language_model.model.layers.{i}.self_attn.v_proj.weight: language_model.blocks.{i}.attn.v_proj.weight",
        "language_model.model.layers.{i}.self_attn.o_proj.weight: language_model.blocks.{i}.attn.o_proj.weight",
        "language_model.model.layers.{i}.post_attention_layernorm.weight: language_model.blocks.{i}.ln_mlp.weight",
        "language_model.model.layers.{i}.mlp.gate_proj.weight: language_model.blocks.{i}.mlp.gate_proj.weight",
        "language_model.model.layers.{i}.mlp.up_proj.weight: language_model.blocks.{i}.mlp.up_proj.weight", 
        "language_model.model.layers.{i}.mlp.down_proj.weight: language_model.blocks.{i}.mlp.down_proj.weight",
        "",
        "# Notes:",
        "# - Adjust layer indices {i} based on actual model depth",
        "# - Some models may have different naming conventions",
        "# - Vision encoder may use different attention/MLP structures",
        "# - Projector architecture varies (linear, MLP, etc.)",
    ]
    (out_dir / "keymap_template.yaml").write_text("\n".join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dump] loading config: {args.model}")
    try:
        cfg = AutoConfig.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        # Save raw config
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            f.write(cfg.to_json_string())
    except Exception as e:
        print(f"[error] Failed to load config: {e}")
        return

    print(f"[dump] loading weights (dtype={args.dtype})…")
    dtype = pick_dtype(args.dtype)
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
            device_map=None,
        )
        state = model.state_dict()
    except Exception as e:
        print(f"[error] Failed to load model: {e}")
        print("[info] Trying with AutoModelForCausalLM as fallback...")
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=dtype,
                device_map=None,
            )
            state = model.state_dict()
        except Exception as e2:
            print(f"[error] Fallback also failed: {e2}")
            return

    print(f"[dump] analyzing VLM architecture...")
    analysis = analyze_vlm_architecture(cfg, state)
    write_json(out_dir / "architecture_analysis.json", analysis)

    print(f"[dump] saving state schema with {len(state)} tensors…")
    save_state_schema(out_dir, state)

    print("[dump] writing module tree…")
    save_module_tree(out_dir, model)

    print("[dump] writing VLM keymap template…")
    make_vlm_keymap_template(out_dir, state, analysis)

    # Try to load processor for additional info
    try:
        print("[dump] loading processor for additional config...")
        processor = AutoProcessor.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        processor_info = {
            "processor_class": processor.__class__.__name__,
            "tokenizer_info": {
                "vocab_size": getattr(processor.tokenizer, "vocab_size", None),
                "model_max_length": getattr(processor.tokenizer, "model_max_length", None),
                "pad_token": getattr(processor.tokenizer, "pad_token", None),
                "eos_token": getattr(processor.tokenizer, "eos_token", None),
                "bos_token": getattr(processor.tokenizer, "bos_token", None),
            },
            "image_processor_info": {
                "size": getattr(processor.image_processor, "size", None),
                "do_resize": getattr(processor.image_processor, "do_resize", None),
                "do_normalize": getattr(processor.image_processor, "do_normalize", None),
                "image_mean": getattr(processor.image_processor, "image_mean", None),
                "image_std": getattr(processor.image_processor, "image_std", None),
            }
        }
        write_json(out_dir / "processor_info.json", processor_info)
    except Exception as e:
        print(f"[warning] Could not load processor: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_human": numel_str(total_params),
        "trainable_params_human": numel_str(trainable_params),
        "model_type": analysis["model_type"],
        "architecture_type": analysis["architecture_type"],
    }
    write_json(out_dir / "summary.json", meta)

    print(f"\n[summary] SmolVLM2 Analysis Complete:")
    print(f"  Total parameters: {numel_str(total_params)}")
    print(f"  Vision params: {numel_str(analysis['parameter_breakdown']['vision_params'])} ({analysis['parameter_breakdown']['vision_percentage']:.1f}%)")
    print(f"  Language params: {numel_str(analysis['parameter_breakdown']['language_params'])} ({analysis['parameter_breakdown']['language_percentage']:.1f}%)")
    print(f"  Projector params: {numel_str(analysis['parameter_breakdown']['projector_params'])} ({analysis['parameter_breakdown']['projector_percentage']:.1f}%)")
    print(f"  Model type: {analysis['model_type']}")
    print(f"  Architecture: {analysis['architecture_type']}")

    print(f"\n[done] artifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
