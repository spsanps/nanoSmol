"""NanoGPT-style chat data pipeline broken into bite-sized modules."""
from .adapters import (
    ChatAdapter,
    ChatRecord,
    FineVisionAdapter,
    Message,
    _convert_turns_to_messages,
    _passes_quality_filter,
    available_adapters,
    get_adapter,
    register_adapter,
)
from .collator import ChatCollator, build_image_transform
from .config import ChatDataConfig
from .dataset import StreamingChatDataset
from .loader import build_chat_dataloader
from .tokenizer import ConversationTokenizer

# Backwards compatible aliases for earlier modules/tests.
FineVisionDataConfig = ChatDataConfig
FineVisionCollator = ChatCollator
build_finevision_dataloader = build_chat_dataloader
_build_image_transform = build_image_transform

# Private helper compatibility layer.
__private_aliases__ = [
    "_build_image_transform",
    "_convert_turns_to_messages",
    "_passes_quality_filter",
]

__all__ = [
    "ChatAdapter",
    "ChatRecord",
    "FineVisionAdapter",
    "Message",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "ChatCollator",
    "build_image_transform",
    "ChatDataConfig",
    "StreamingChatDataset",
    "build_chat_dataloader",
    "ConversationTokenizer",
    "FineVisionDataConfig",
    "FineVisionCollator",
    "build_finevision_dataloader",
] + __private_aliases__
