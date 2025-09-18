#!/usr/bin/env python3
"""Quick smoke-test generation for the minimal SmolVLM implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.smolVLM import SmolVLM, SmolVLMConfig  # noqa: E402

DEFAULT_PROMPT = "Summarise the scene in a few words."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="HuggingFaceTB/SmolVLM-256M-Base")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", default=None, help="path to an RGB image (defaults to a grey square)")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def build_prompt(processor: AutoProcessor, text: str) -> str:
    bos = processor.tokenizer.bos_token or ""
    image_token = getattr(processor, "image_token", "<image>")
    terminator = processor.tokenizer.unk_token or ""
    return f"{bos}User:{image_token}{text}{terminator}Assistant:"


def load_image(path: Optional[str]) -> Image.Image:
    if path is None:
        return Image.new("RGB", (512, 512), color=(180, 180, 180))
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    hf_cfg = AutoConfig.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    prompt_text = args.prompt if args.prompt is not None else DEFAULT_PROMPT
    prompt = build_prompt(processor, prompt_text)
    image = load_image(args.image)
    proc_inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    input_ids = proc_inputs["input_ids"].to(device)
    attention_mask = proc_inputs["attention_mask"].to(device)
    pixel_values = proc_inputs["pixel_values"].to(device=device, dtype=dtype)
    pixel_attention_mask = proc_inputs["pixel_attention_mask"].to(device)

    smol_cfg = SmolVLMConfig.from_hf_config(hf_cfg)
    local_model = SmolVLM(smol_cfg).to(device=device, dtype=dtype).eval()
    hf_model = AutoModelForVision2Seq.from_pretrained(
        args.hf_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map={"": device.type},
    ).eval()

    local_model.load_hf_state_dict(hf_model.state_dict(), strict=False, verbose=True, dtype=dtype, device=device)

    with torch.no_grad():
        local_ids = local_model.generate(
            input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        hf_ids = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print("=== Prompt ===")
    print(prompt_text)
    print("\n=== Local ===")
    print(processor.tokenizer.decode(local_ids[0], skip_special_tokens=True))
    print("\n=== HF ===")
    print(processor.tokenizer.decode(hf_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
