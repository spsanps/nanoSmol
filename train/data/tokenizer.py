"""Utilities that render chat transcripts into token tensors."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .adapters import Message


class ConversationTokenizer:
    """Render chat transcripts into ``input_ids``/``labels`` tensors."""

    def __init__(
        self,
        tokenizer,
        *,
        image_token_id: int,
        bos_token_id: Optional[int],
        eos_token_id: Optional[int],
        pad_token_id: int,
        user_prefix: str = "<|user|>: ",
        assistant_prefix: str = "<|assistant|>: ",
        inline_sep: str = "\n",
        message_sep: str = "\n",
        append_eos_to_assistant: bool = True,
        image_tokens_per_image: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_token_id = int(image_token_id)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.inline_sep = inline_sep
        self.message_sep = message_sep
        self.append_eos = append_eos_to_assistant
        self.image_tokens_per_image = max(int(image_tokens_per_image), 1)

    # ------------------------------------------------------------------ helpers
    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(tok) for tok in tokens]

    # ------------------------------------------------------------------ API
    def render(self, messages: Sequence[Message]) -> Dict[str, torch.Tensor]:
        """Return ``input_ids``, ``labels`` and ``attention_mask`` for a transcript."""

        input_ids: List[int] = []
        labels: List[int] = []

        if self.bos_token_id is not None:
            input_ids.append(self.bos_token_id)
            labels.append(-100)

        for message in messages:
            prefix = self.user_prefix if message["role"] == "user" else self.assistant_prefix
            prefix_ids = self._encode(prefix)
            input_ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))

            content = list(message.get("content", []))
            for idx, chunk in enumerate(content):
                if chunk.get("type") == "image":
                    repeat = self.image_tokens_per_image
                    input_ids.extend([self.image_token_id] * repeat)
                    labels.extend([-100] * repeat)
                else:
                    text = str(chunk.get("text", ""))
                    if not text:
                        continue
                    encoded = self._encode(text)
                    input_ids.extend(encoded)
                    if message["role"] == "assistant":
                        labels.extend(encoded)
                    else:
                        labels.extend([-100] * len(encoded))
                if idx != len(content) - 1:
                    sep_ids = self._encode(self.inline_sep)
                    input_ids.extend(sep_ids)
                    labels.extend([-100] * len(sep_ids))

            sep_ids = self._encode(self.message_sep)
            input_ids.extend(sep_ids)
            labels.extend([-100] * len(sep_ids))

            if message["role"] == "assistant" and self.append_eos and self.eos_token_id is not None:
                input_ids.append(self.eos_token_id)
                labels.append(self.eos_token_id)

        tensor_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        attention = torch.ones_like(tensor_ids, dtype=torch.long)
        return {"input_ids": tensor_ids, "labels": tensor_labels, "attention_mask": attention}


__all__ = ["ConversationTokenizer"]
