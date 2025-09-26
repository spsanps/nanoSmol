"""Shared runtime utilities for the NanoEval tasks.

The repository aims for "one-screen" evaluation scripts.  Everything that would
otherwise be duplicated between the MMLU, HellaSwag, and MMMU-Pro runners lives
here: device/precision handling, deterministic seeding, and the tiny wrappers
around Hugging Face models used to score multiple-choice prompts.
"""

from __future__ import annotations

import re
from string import ascii_uppercase
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from .config import ModelConfig

LETTER4: tuple[str, ...] = ("A", "B", "C", "D")
LETTER10: tuple[str, ...] = LETTER4 + ("E", "F", "G", "H", "I", "J")


def build_letter_choices(count: int) -> tuple[str, ...]:
    """Return an ordered tuple of letter labels long enough for ``count`` options."""

    if count < 1:
        raise ValueError("Multiple-choice prompts must include at least one option")
    if count <= len(ascii_uppercase):
        return tuple(ascii_uppercase[:count])
    raise ValueError(
        "NanoEval currently supports up to 26 answer choices; "
        f"received {count}"
    )

_DTYPE_LOOKUP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def set_seed(seed: int) -> None:
    """Seed all torch RNGs so greedy decoding stays deterministic."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> str:
    """Return the actual device to run on given the user's preference."""

    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cpu"


class SimpleModel:
    """Minimal wrapper around either a text LLM or a vision-language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        torch_dtype = _DTYPE_LOOKUP.get(cfg.dtype, torch.float16)

        if cfg.is_vlm:
            self.processor = AutoProcessor.from_pretrained(
                cfg.model_id, trust_remote_code=cfg.trust_remote_code
            )
            model_kwargs = {"dtype": torch_dtype}
            if self.device.startswith("cuda") and cfg.attn_impl:
                model_kwargs["_attn_implementation"] = cfg.attn_impl
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_id, **model_kwargs
            ).to(self.device)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_id, trust_remote_code=cfg.trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id, dtype=torch_dtype
            ).to(self.device)
            self.processor = None

        self.model.eval()

    # --------------------------------------------------------------------- text
    @torch.no_grad()
    def generate_letter_text(
        self,
        prompt: str,
        allowed_letters: Sequence[str],
        max_new_tokens: int,
    ) -> str:
        """Greedily decode a short answer and return the first valid letter."""

        if self.processor is not None:
            # Vision-language checkpoints (e.g. SmolVLM) still need to answer
            # text-only prompts for MMLU / HellaSwag.  We mirror the chat-style
            # interface but provide no images so the call path stays uniform.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ]
            return self.generate_letter_vlm(
                messages,
                images=(),
                allowed_letters=allowed_letters,
                max_new_tokens=max_new_tokens,
            )

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return _extract_choice(decoded, allowed_letters)

    @torch.no_grad()
    def rank_log_likelihood(self, prompt: str, options: Sequence[str]) -> int:
        """Return the index of the option with the highest summed log-prob."""

        if self.processor is not None:
            raise RuntimeError("Log-likelihood scoring only applies to text models")

        scores: List[float] = []
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        for candidate in options:
            option_ids = self.tokenizer.encode(" " + candidate, add_special_tokens=False)
            input_ids = torch.tensor([prompt_ids + option_ids], device=self.device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            scores.append(gathered[0, -len(option_ids) :].sum().item())
        best_index = max(range(len(options)), key=scores.__getitem__)
        return int(best_index)

    # ---------------------------------------------------------------------- vlm
    @torch.no_grad()
    def generate_letter_vlm(
        self,
        messages: Sequence[dict],
        images: Sequence[object],
        allowed_letters: Sequence[str],
        max_new_tokens: int,
    ) -> str:
        """Mirror chat-style prompting for VLMs such as SmolVLM."""

        if self.processor is None:
            raise RuntimeError("Text models must use generate_letter_text")
        prompt = self.processor.apply_chat_template(
            list(messages), add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=list(images) if images else None,
            return_tensors="pt",
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return _extract_choice(decoded, allowed_letters)


def _extract_choice(decoded: str, allowed_letters: Sequence[str]) -> str:
    """Return the first allowed letter found in ``decoded`` text."""

    if not allowed_letters:
        return ""
    allowed_lookup = {letter.upper(): letter for letter in allowed_letters}
    pattern = r"\\b(" + "|".join(re.escape(letter) for letter in allowed_lookup) + r")\\b"
    match = re.search(pattern, decoded, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        return allowed_lookup.get(candidate, "")
    for char in decoded:
        letter = allowed_lookup.get(char.upper())
        if letter:
            return letter
    return ""


__all__ = [
    "LETTER4",
    "LETTER10",
    "build_letter_choices",
    "set_seed",
    "SimpleModel",
]
