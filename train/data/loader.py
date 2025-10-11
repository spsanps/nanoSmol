"""Factories that assemble the complete chat DataLoader."""
from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .collator import ChatCollator, build_image_transform
from .config import ChatDataConfig
from .dataset import StreamingChatDataset
from .tokenizer import ConversationTokenizer


def _infer_image_token_count(processor, tokenizer, model_cfg, image_size: int) -> int:
    candidates = []
    proc_tokenizer = None
    if processor is not None:
        proc_tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_like = proc_tokenizer if proc_tokenizer is not None else tokenizer

    for attr in ("num_image_tokens", "image_seq_length", "image_tokens_per_image", "image_length"):
        value = getattr(tokenizer_like, attr, None)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            candidates.append(value)

    image_token_id = getattr(tokenizer_like, "image_token_id", None)
    if processor is not None and image_token_id is not None:
        try:
            dummy = Image.new("RGB", (image_size, image_size), color=0)
            probe_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": dummy},
                        {"type": "text", "text": "Describe"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
            ]
            apply_template = getattr(processor, "apply_chat_template", None)
            if callable(apply_template):
                rendered = apply_template(probe_messages, add_generation_prompt=False)
                batch = processor(text=rendered, images=[dummy], return_tensors="pt")
            else:
                rendered = "<|user|>: <image> Describe\n<|assistant|>: Done."
                batch = processor(text=rendered, images=[dummy], return_tensors="pt")
            input_ids = batch.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                image_tokens = (input_ids[0] == int(image_token_id)).sum().item()
                if image_tokens > 0:
                    candidates.append(int(image_tokens))
        except Exception:
            pass

    if candidates:
        return max(candidates)

    vision_cfg = getattr(model_cfg, "vision", None)
    if vision_cfg is not None and hasattr(vision_cfg, "num_patches"):
        try:
            value = int(vision_cfg.num_patches)
            if value > 0:
                return value
        except Exception:
            pass
    return 1


def build_chat_dataloader(
    cfg: ChatDataConfig,
    *,
    tokenizer,
    model_cfg,
    batch_size: int,
    num_workers: int = 0,
    processor=None,
) -> DataLoader:
    """Factory for multimodal chat dataloaders that can swap adapters."""

    image_token_repeat = _infer_image_token_count(processor, tokenizer, model_cfg, cfg.image_size)

    tokenizer_wrapper = ConversationTokenizer(
        tokenizer,
        image_token_id=model_cfg.image_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=int(getattr(tokenizer, "pad_token_id", getattr(model_cfg, "pad_token_id", 0))),
        image_tokens_per_image=image_token_repeat,
    )
    collator = ChatCollator(
        tokenizer_wrapper,
        image_transform=build_image_transform(cfg),
        max_images=cfg.max_images,
        image_size=cfg.image_size,
        processor=processor,
    )
    dataset = StreamingChatDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


__all__ = ["build_chat_dataloader"]
