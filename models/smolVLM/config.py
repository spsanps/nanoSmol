"""Configuration dataclasses that describe the minimal SmolVLM layout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only needed for static type checkers
    from transformers import Idefics3Config


@dataclass
class SmolVLMVisionConfig:
    """Hyper-parameters steering the SigLIP-style vision transformer.

    The vision encoder is a straightforward ViT: images are split into patches,
    each patch is linearly projected, and a stack of transformer blocks operates
    over the resulting sequence.  The fields below provide just enough
    information to rebuild that architecture from scratch in ``vision.py``.
    """

    image_size: int
    patch_size: int
    num_channels: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"

    @property
    def num_patches_per_side(self) -> int:
        """Number of patches that fit along the height/width of the image."""

        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total patch count in a square grid (used to size positional tables)."""

        side = self.num_patches_per_side
        return side * side


@dataclass
class SmolVLMLanguageConfig:
    """Parameters that define the decoder-only language stack.

    SmolVLM follows a LLaMA-like design: grouped-query attention with rotary
    embeddings and SwiGLU feed-forward networks.  We keep the dataclass explicit
    so each hyper-parameter is visible at call sites, mirroring the educational
    tone of the rest of the project.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    dropout: float = 0.0
    tie_embeddings: bool = False
    pad_token_id: int = 0

    @property
    def head_dim(self) -> int:
        """Dimension of a single attention head (convenience helper)."""

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


@dataclass
class SmolVLMConfig:
    """Bundle vision + language settings together with VL glue metadata."""

    vision: SmolVLMVisionConfig
    language: SmolVLMLanguageConfig
    image_token_id: int
    pad_token_id: int
    scale_factor: int = 1
    mm_hidden_size: Optional[int] = None

    @classmethod
    def from_hf_config(cls, hf_cfg: "Idefics3Config") -> "SmolVLMConfig":
        """Translate a Hugging Face ``Idefics3Config`` into light-weight dataclasses."""

        vision_cfg = hf_cfg.vision_config
        text_cfg = hf_cfg.text_config

        # --- vision -----------------------------------------------------------------
        # Hugging Face stores the maximum processed size separately from the original
        # training image size.  Prefer the explicit ``image_size`` attribute when
        # present, otherwise fall back to ``size["longest_edge"]``.
        vision = SmolVLMVisionConfig(
            image_size=int(getattr(vision_cfg, "image_size", 0) or vision_cfg.size["longest_edge"]),
            patch_size=int(vision_cfg.patch_size),
            num_channels=int(vision_cfg.num_channels),
            hidden_size=int(vision_cfg.hidden_size),
            intermediate_size=int(vision_cfg.intermediate_size),
            num_hidden_layers=int(vision_cfg.num_hidden_layers),
            num_attention_heads=int(vision_cfg.num_attention_heads),
            layer_norm_eps=float(vision_cfg.layer_norm_eps),
            attention_dropout=float(getattr(vision_cfg, "attention_dropout", 0.0)),
            hidden_act=str(getattr(vision_cfg, "hidden_act", "gelu_pytorch_tanh")),
        )

        # --- language ---------------------------------------------------------------
        language = SmolVLMLanguageConfig(
            vocab_size=int(text_cfg.vocab_size),
            hidden_size=int(text_cfg.hidden_size),
            intermediate_size=int(text_cfg.intermediate_size),
            num_hidden_layers=int(text_cfg.num_hidden_layers),
            num_attention_heads=int(text_cfg.num_attention_heads),
            num_key_value_heads=int(getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)),
            max_position_embeddings=int(text_cfg.max_position_embeddings),
            rope_theta=float(getattr(text_cfg, "rope_theta", 100000.0)),
            rms_norm_eps=float(getattr(text_cfg, "rms_norm_eps", 1e-5)),
            dropout=float(getattr(text_cfg, "attention_dropout", 0.0)),
            tie_embeddings=bool(
                getattr(hf_cfg, "tie_word_embeddings", getattr(text_cfg, "tie_word_embeddings", False))
            ),
            pad_token_id=int(getattr(text_cfg, "pad_token_id", getattr(hf_cfg, "pad_token_id", 0) or 0)),
        )

        # --- multimodal glue --------------------------------------------------------
        scale_factor = int(getattr(hf_cfg, "scale_factor", getattr(text_cfg, "pixel_shuffle_factor", 1)))
        mm_hidden = getattr(hf_cfg, "mm_hidden_size", None)
        if mm_hidden is not None:
            mm_hidden = int(mm_hidden)

        pad_id = int(getattr(hf_cfg, "pad_token_id", language.pad_token_id))
        return cls(
            vision=vision,
            language=language,
            image_token_id=int(getattr(hf_cfg, "image_token_id", 0)),
            pad_token_id=pad_id,
            scale_factor=max(scale_factor, 1),
            mm_hidden_size=mm_hidden,
        )

