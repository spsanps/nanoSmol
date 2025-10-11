"""Batch collation utilities for multimodal chat data."""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from .config import ChatDataConfig
from .tokenizer import ConversationTokenizer


def build_image_transform(cfg: ChatDataConfig):
    """Return a closure that maps PIL images -> normalised tensors."""

    mean = torch.tensor(cfg.image_mean).view(-1, 1, 1)
    std = torch.tensor(cfg.image_std).view(-1, 1, 1)

    def transform(image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        resized = resize(rgb, [cfg.image_size, cfg.image_size], interpolation=InterpolationMode.BICUBIC)
        tensor = to_tensor(resized).to(dtype=torch.float32)
        return (tensor - mean) / std

    return transform


class ChatCollator:
    """Pack variable-length transcripts and image grids into dense tensors."""

    def __init__(
        self,
        tokenizer: ConversationTokenizer,
        *,
        image_transform,
        max_images: int,
        image_size: int,
        processor=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_images = max_images
        self.image_size = image_size
        self.processor = processor

        self._processor_image_transform = None
        if processor is not None:
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is not None:
                def _process(image):
                    batch = image_processor([image], return_tensors="pt")
                    pixels = batch.get("pixel_values")
                    if isinstance(pixels, torch.Tensor):
                        tensor = pixels[0]
                    else:
                        tensor = torch.tensor(pixels[0])
                    return tensor.to(dtype=torch.float32)

                self._processor_image_transform = _process

        self.image_transform = image_transform
        if self._processor_image_transform is None:
            if self.image_transform is None:
                raise ValueError("image_transform must be provided when processor has no image processor")
            dummy = self.image_transform(Image.new("RGB", (image_size, image_size), color=0))
        else:
            dummy = self._processor_image_transform(Image.new("RGB", (image_size, image_size), color=0))
        self._blank_pixels = torch.zeros_like(dummy)
        self._pixel_hw = dummy.shape[1:]

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        encoded = [self.tokenizer.render(sample["messages"]) for sample in batch]
        input_ids = [item["input_ids"] for item in encoded]
        attention = [item["attention_mask"] for item in encoded]
        labels = [item["labels"] for item in encoded]

        max_len = max(vec.size(0) for vec in input_ids)
        pad_id = self.tokenizer.pad_token_id

        def pad_stack(tensors: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
            padded: List[torch.Tensor] = []
            for tensor in tensors:
                pad_width = max_len - tensor.size(0)
                if pad_width > 0:
                    pad_tensor = torch.full((pad_width,), pad_value, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, pad_tensor], dim=0)
                padded.append(tensor)
            return torch.stack(padded, dim=0)

        batch_input = pad_stack(input_ids, pad_id)
        batch_attention = pad_stack(attention, 0)
        batch_labels = pad_stack(labels, -100)

        pixel_tensors: List[torch.Tensor] = []
        pixel_masks: List[torch.Tensor] = []
        for sample in batch:
            images = list(sample.get("images", []))[: self.max_images]
            slots: List[torch.Tensor] = []
            mask_flags: List[int] = []
            for image in images:
                if self._processor_image_transform is not None:
                    slots.append(self._processor_image_transform(image))
                else:
                    slots.append(self.image_transform(image))
                mask_flags.append(1)
            while len(slots) < self.max_images:
                slots.append(self._blank_pixels.clone())
                mask_flags.append(0)
            pixel_tensors.append(torch.stack(slots, dim=0))

            mask_planes: List[torch.Tensor] = []
            for flag in mask_flags:
                plane = torch.ones(self._pixel_hw, dtype=torch.bool) if flag else torch.zeros(self._pixel_hw, dtype=torch.bool)
                mask_planes.append(plane)
            pixel_masks.append(torch.stack(mask_planes, dim=0))

        pixels = torch.stack(pixel_tensors, dim=0)
        attention_mask = torch.stack(pixel_masks, dim=0)
        return {
            "input_ids": batch_input,
            "attention_mask": batch_attention,
            "labels": batch_labels,
            "pixel_values": pixels,
            "pixel_attention_mask": attention_mask,
        }


__all__ = ["ChatCollator", "build_image_transform"]
