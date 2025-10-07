"""Shared runtime utilities for the NanoEval tasks.

The repository aims for "one-screen" evaluation scripts.  Everything that would
otherwise be duplicated between the MMLU, HellaSwag, and MMMU-Pro runners lives
here: device/precision handling, deterministic seeding, and the tiny wrappers
around Hugging Face models used to score multiple-choice prompts.
"""

from __future__ import annotations

import random
from string import ascii_uppercase
from typing import Any, List, Mapping, Sequence

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
    def generate_text(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        *,
        images: Sequence[object] | None = None,
        max_new_tokens: int = 32,
    ) -> str:
        """Greedily decode a response for ``prompt``.

        ``SimpleModel`` instances abstract over text-only causal LMs and
        multimodal chat-style models.  ``generate_text`` accepts either a raw
        string prompt (for text models) or a chat-style conversation
        (``[{"role": ..., "content": ...}, ...]``) for VLMs.  The method always
        performs deterministic greedy decoding so repeated calls with the same
        seed produce identical outputs.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        tokenizer = self.tokenizer
        if self.processor is None:
            if not isinstance(prompt, str):
                raise TypeError("Text models expect `prompt` to be a string")
            # ``transformers`` tokenizers expose both ``__call__`` and ``encode``.
            # ``SimpleModel`` supports either to keep dummy unit-test tokenizers
            # lightweight.
            tokenized_inputs: Mapping[str, torch.Tensor]
            try:
                tokenized_inputs = self.tokenizer(  # type: ignore[call-arg]
                    prompt, return_tensors="pt"
                )
            except TypeError:
                prompt_ids = self.tokenizer.encode(  # type: ignore[call-arg]
                    prompt, add_special_tokens=False
                )
                tensor = torch.tensor([prompt_ids], dtype=torch.long)
                tokenized_inputs = {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                }
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in tokenized_inputs.items()
            }
        else:
            images_list = list(images) if images is not None else []
            if isinstance(prompt, str):
                content: List[Mapping[str, Any]] = []
                for _ in images_list:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            else:
                messages = list(prompt)
            convo_text = self.processor.apply_chat_template(  # type: ignore[arg-type]
                messages, add_generation_prompt=True
            )
            batch_encoding = self.processor(  # type: ignore[call-arg]
                text=convo_text,
                images=images_list if images_list else None,
                return_tensors="pt",
            )
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in batch_encoding.items()
            }
            tokenizer = getattr(self.processor, "tokenizer", tokenizer)

        pad_token_id = None
        if hasattr(self.model, "config"):
            pad_token_id = getattr(self.model.config, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.model.config, "eos_token_id", None)
        if pad_token_id is None and tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)

        generated_ids = self.model.generate(  # type: ignore[call-arg]
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=pad_token_id,
        )

        if isinstance(generated_ids, torch.Tensor):
            full_sequence = generated_ids
        else:
            full_sequence = torch.tensor(generated_ids, device=self.device)

        input_length = inputs["input_ids"].shape[1]
        new_token_ids = full_sequence[0, input_length:]
        new_token_list = new_token_ids.tolist()

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            text = tokenizer.decode(new_token_list, skip_special_tokens=True)  # type: ignore[arg-type]
        elif hasattr(self.processor, "decode"):
            text = self.processor.decode(new_token_list, skip_special_tokens=True)  # type: ignore[arg-type]
        elif hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode([new_token_list], skip_special_tokens=True)[0]  # type: ignore[arg-type]
        else:
            raise RuntimeError("Could not find a decode method for generated tokens")

        return text.strip()

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
