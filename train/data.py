"""Multimodal chat dataset utilities (NanoGPT style, but modular)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type

import torch
from torch.utils.data import DataLoader, IterableDataset

try:  # 🤗 Datasets is optional for unit tests
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional import for tests
    load_dataset = None  # type: ignore

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor


ChatRecord = Dict[str, object]
Message = Dict[str, object]


def _infer_image_token_repeats(model_cfg) -> int:
    """Best-effort heuristic for matching the model's expected <image> span length."""

    def _coerce_int(value) -> Optional[int]:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            return None
        return integer

    for attr in ("image_token_length", "image_seq_length", "image_seq_len", "vision_token_length"):
        value = _coerce_int(getattr(model_cfg, attr, None))
        if value is not None and value > 0:
            return value

    vision = getattr(model_cfg, "vision", None)
    if vision is not None:
        num_patches = getattr(vision, "num_patches", None)
        if not isinstance(num_patches, int):
            num_patches = _coerce_int(num_patches)
        if num_patches is None or num_patches <= 0:
            image_size = _coerce_int(getattr(vision, "image_size", None))
            patch_size = _coerce_int(getattr(vision, "patch_size", None))
            if image_size and patch_size and patch_size > 0:
                side = image_size // patch_size
                num_patches = side * side

        if num_patches and num_patches > 0:
            scale = _coerce_int(getattr(model_cfg, "scale_factor", None))
            if scale is None or scale < 1:
                scale = 1
            repeats = num_patches // (scale * scale)
            return max(repeats, 1)

    return 1


@dataclass
class ChatDataConfig:
    """Configuration describing how to stream and post-process chat data."""

    repo_id: str = "HuggingFaceM4/FineVision-1.0"
    subset: Optional[str] = None
    split: str = "train"
    streaming: bool = True
    shuffle_buffer_size: int = 1024
    seed: int = 0
    max_turns: Optional[int] = None
    max_images: int = 1
    image_size: int = 384
    image_mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)
    min_quality: Optional[int] = 4
    adapter: str = "finevision"


def _convert_turns_to_messages(
    record: ChatRecord,
    *,
    max_images: int,
    max_turns: Optional[int],
) -> Optional[Dict[str, object]]:
    """Turn raw records into chat-style message transcripts."""

    images = list(record.get("images", []))[:max_images]
    if not images:
        return None

    qa_pairs = list(record.get("texts", []))
    if max_turns is not None:
        qa_pairs = qa_pairs[:max_turns]
    if not qa_pairs:
        return None

    messages: List[Message] = []
    first_user_message = True
    for pair in qa_pairs:
        question = pair.get("user") or pair.get("question") or ""
        answer = pair.get("assistant") or pair.get("answer") or ""
        if not question or not answer:
            continue

        user_chunks: List[Dict[str, object]] = []
        if first_user_message:
            for image in images:
                if isinstance(image, Image.Image):
                    user_chunks.append({"type": "image", "image": image})
            first_user_message = False
        user_chunks.append({"type": "text", "text": str(question).strip()})
        messages.append({"role": "user", "content": user_chunks})

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer).strip()}],
            }
        )

    if not messages:
        return None

    return {
        "images": images,
        "messages": messages,
        "source": record.get("source"),
    }


def _passes_quality_filter(record: ChatRecord, min_quality: Optional[int]) -> bool:
    """Drop low-quality samples by checking optional quality ratings."""

    if min_quality is None:
        return True
    ratings = record.get("relevance_ratings")
    if ratings is None:
        return True
    try:
        scores = [int(score) for score in ratings]
    except Exception:
        return True
    return min(scores) >= min_quality if scores else True


class ChatAdapter:
    """Base class for dataset-specific hooks."""

    name: str = "base"

    def __init__(self, cfg: ChatDataConfig) -> None:
        self.cfg = cfg

    def filter(self, record: ChatRecord) -> bool:
        return True

    def convert(self, record: ChatRecord) -> Optional[Dict[str, object]]:
        raise NotImplementedError


class FineVisionAdapter(ChatAdapter):
    """Adapter implementing FineVision's multi-turn structure."""

    name = "finevision"

    def filter(self, record: ChatRecord) -> bool:
        return _passes_quality_filter(record, self.cfg.min_quality)

    def convert(self, record: ChatRecord) -> Optional[Dict[str, object]]:
        return _convert_turns_to_messages(
            record,
            max_images=self.cfg.max_images,
            max_turns=self.cfg.max_turns,
        )


_ADAPTERS: Dict[str, Type[ChatAdapter]] = {FineVisionAdapter.name: FineVisionAdapter}


def register_adapter(adapter_cls: Type[ChatAdapter]) -> None:
    """Expose new dataset adapters without touching the core pipeline."""

    _ADAPTERS[adapter_cls.name] = adapter_cls


def available_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_ADAPTERS.keys()))


class StreamingChatDataset(IterableDataset):
    """Stream chat data through an adapter to keep memory usage tiny."""

    def __init__(self, cfg: ChatDataConfig):
        super().__init__()
        self.cfg = cfg
        if load_dataset is None:
            raise ImportError("datasets is required to stream Hugging Face corpora")

    def _dataset_stream(self):
        data = load_dataset(
            self.cfg.repo_id,
            name=self.cfg.subset,
            streaming=self.cfg.streaming,
            split=self.cfg.split,
        )
        if self.cfg.streaming:
            data = data.shuffle(seed=self.cfg.seed, buffer_size=self.cfg.shuffle_buffer_size)
        elif hasattr(data, "shuffle"):
            data = data.shuffle(seed=self.cfg.seed)
        return data

    def __iter__(self) -> Iterator[Dict[str, object]]:  # pragma: no cover - requires datasets
        adapter_cls = _ADAPTERS.get(self.cfg.adapter)
        if adapter_cls is None:
            raise ValueError(f"Unknown adapter '{self.cfg.adapter}'. Known: {available_adapters()}")
        adapter = adapter_cls(self.cfg)
        for record in self._dataset_stream():
            if not adapter.filter(record):
                continue
            sample = adapter.convert(record)
            if sample is None:
                continue
            yield sample


class ConversationTokenizer:
    """Render chat transcripts into token/label tensors."""

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
        image_token_repeats: int = 1,
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
        repeats = int(image_token_repeats)
        if repeats < 1:
            raise ValueError("image_token_repeats must be at least 1")
        self.image_token_repeats = repeats

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(tok) for tok in tokens]

    def render(self, messages: Sequence[Message]) -> Dict[str, torch.Tensor]:
        """Return ``input_ids``, ``labels``, ``attention_mask`` for a transcript."""

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
                    input_ids.extend([self.image_token_id] * self.image_token_repeats)
                    labels.extend([-100] * self.image_token_repeats)
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


def _build_image_transform(cfg: ChatDataConfig):
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
    ) -> None:
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_images = max_images
        self.image_size = image_size
        dummy = self.image_transform(Image.new("RGB", (image_size, image_size), color=0))
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


def build_chat_dataloader(
    cfg: ChatDataConfig,
    *,
    tokenizer,
    model_cfg,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Factory for multimodal chat dataloaders that can swap adapters."""

    tokenizer_wrapper = ConversationTokenizer(
        tokenizer,
        image_token_id=model_cfg.image_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=int(getattr(tokenizer, "pad_token_id", getattr(model_cfg, "pad_token_id", 0))),
        image_token_repeats=_infer_image_token_repeats(model_cfg),
    )
    collator = ChatCollator(
        tokenizer_wrapper,
        image_transform=_build_image_transform(cfg),
        max_images=cfg.max_images,
        image_size=cfg.image_size,
    )
    dataset = StreamingChatDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


# Backwards compatible aliases for earlier modules/tests.
FineVisionDataConfig = ChatDataConfig
FineVisionCollator = ChatCollator
build_finevision_dataloader = build_chat_dataloader

