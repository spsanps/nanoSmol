"""
Release data pipeline for foveated VLM training.

Components:
- webdataset_loader: Streams tar shards of video frames + tokenized text
- collate: Pads variable-length batches (frames, text, masks)
- text_interleave: Mixes ~14% text-only batches to preserve language ability
"""

from release.data.webdataset_loader import create_webdataset, decode_sample
from release.data.collate import collate_foveated
from release.data.text_interleave import InterleavedDataLoader

__all__ = [
    "create_webdataset",
    "decode_sample",
    "collate_foveated",
    "InterleavedDataLoader",
]
