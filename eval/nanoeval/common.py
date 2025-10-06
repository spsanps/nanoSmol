"""Shared runtime utilities for the NanoEval tasks.

The repository aims for "one-screen" evaluation scripts.  Everything that would
otherwise be duplicated between the MMLU, HellaSwag, and MMMU-Pro runners lives
here: device/precision handling, deterministic seeding, and the tiny wrappers
around Hugging Face models used to score multiple-choice prompts.
"""

from __future__ import annotations

import random
from string import ascii_uppercase
from typing import List, Sequence

import numpy as np
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
    """Return ``count`` Excel-style letter labels (A..Z, AA, AB, ...)."""

    if count < 1:
        raise ValueError("Multiple-choice prompts must include at least one option")

    labels: List[str] = []
    for index in range(count):
        # Excel-style base-26 conversion that stays within uppercase ASCII.
        value = index
        label_parts: List[str] = []
        while True:
            value, remainder = divmod(value, len(ascii_uppercase))
            label_parts.append(ascii_uppercase[remainder])
            if value == 0:
                break
            value -= 1
        labels.append("".join(reversed(label_parts)))
    return tuple(labels)

_DTYPE_LOOKUP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def set_seed(seed: int) -> None:
    """Seed all torch RNGs so greedy decoding stays deterministic."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    def rank_log_likelihood(self, prompt: str, options: Sequence[str], normalize: bool = False) -> int:
        """Return the index of the option with the highest summed log-prob.

        Args:
            prompt: The prompt/context to score against
            options: List of option texts to rank
            normalize: If True, divide log-prob sum by sequence length (lighteval's loglikelihood_acc_norm)
        """

        if self.processor is None:
            return self._rank_text(prompt, options, normalize=normalize)

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
        return self.rank_log_likelihood_multimodal(messages, (), options, normalize=normalize)

    @torch.no_grad()
    def rank_log_likelihood_multimodal(
        self,
        messages: Sequence[dict],
        images: Sequence[object],
        options: Sequence[str],
        normalize: bool = False,
    ) -> int:
        """Rank ``options`` given multimodal ``messages`` and ``images``.

        Args:
            messages: Chat-style messages
            images: Image inputs for VLM
            options: List of option texts to rank
            normalize: If True, divide log-prob sum by sequence length
        """

        if self.processor is None:
            raise RuntimeError("Text models must call rank_log_likelihood()")
        if not options:
            raise ValueError("Multiple-choice prompts must include at least one option")

        base_messages = list(messages)
        images_list = list(images)
        # ``apply_chat_template`` appends closing tokens (for example
        # ``<end_of_utterance>``) even when the assistant message is empty.  We
        # cache the tokenised length of that "prompt + empty assistant" variant
        # so we can later subtract it from each scored conversation and obtain
        # the exact number of tokens contributed by the candidate option alone.
        blank_convo = base_messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "",
                    }
                ],
            }
        ]
        blank_text = self.processor.apply_chat_template(
            blank_convo, add_generation_prompt=False
        )
        blank_inputs = self.processor(
            text=blank_text,
            images=images_list if images_list else None,
            return_tensors="pt",
        )
        base_len = blank_inputs["input_ids"].shape[1]

        scores: List[float] = []
        for candidate in options:
            assistant_text = (
                candidate
                if not candidate or candidate[0].isspace()
                else " " + candidate
            )
            convo = list(messages) + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_text,
                        }
                    ],
                }
            ]
            convo_text = self.processor.apply_chat_template(
                convo, add_generation_prompt=False
            )
            convo_inputs = self.processor(
                text=convo_text,
                images=images_list if images_list else None,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in convo_inputs.items()}
            input_ids = inputs["input_ids"]
            outputs = self.model(**inputs)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            option_token_count = input_ids.shape[1] - base_len
            if option_token_count <= 0:
                raise RuntimeError(
                    "Failed to isolate option tokens when ranking multimodal options"
                )
            raw_score = gathered[0, -option_token_count:].sum().item()
            # Apply length normalization if requested
            score = raw_score / option_token_count if normalize else raw_score
            scores.append(score)

        best_index = max(range(len(options)), key=scores.__getitem__)
        return int(best_index)

    def _rank_text(self, prompt: str, options: Sequence[str], normalize: bool = False) -> int:
        if not options:
            raise ValueError("Multiple-choice prompts must include at least one option")
        scores: List[float] = []
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        for candidate in options:
            option_ids = self.tokenizer.encode(
                " " + candidate, add_special_tokens=False
            )
            input_ids = torch.tensor([prompt_ids + option_ids], device=self.device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            raw_score = gathered[0, -len(option_ids) :].sum().item()
            # Apply length normalization if requested (like lighteval's loglikelihood_acc_norm)
            score = raw_score / len(option_ids) if normalize and len(option_ids) > 0 else raw_score
            scores.append(score)
        best_index = max(range(len(options)), key=scores.__getitem__)
        return int(best_index)


__all__ = [
    "LETTER4",
    "LETTER10",
    "build_letter_choices",
    "set_seed",
    "SimpleModel",
]
