"""Minimal SmolLM2 building blocks exposed as a small Python package."""
from .config import SmolLM2Config
from .model import SmolLM2

# Backwards-compatible aliases: historical code imported these names directly.
SmolConfig = SmolLM2Config
SmolLM = SmolLM2

__all__ = ["SmolLM2Config", "SmolLM2", "SmolConfig", "SmolLM"]
