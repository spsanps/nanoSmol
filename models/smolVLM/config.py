"""Configuration objects describing the SmolVLM architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from models.smolLM2 import SmolLM2Config

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from transformers import Idefics3Config


@dataclass
class SmolVLMVisionConfig:
    """Hyper-parameters for the SigLIP-style vision transformer."""

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
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        side = self.num_patches_per_side
        return side * side


@dataclass
class SmolVLMLanguageConfig:
    """Subset of LLaMA-style parameters used by the language decoder."""

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

    def to_smol_lm_config(self) -> SmolLM2Config:
        return SmolLM2Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            d_ff=self.intermediate_size,
            max_seq_len=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            dropout=self.dropout,
            norm_eps=self.rms_norm_eps,
            tie_embeddings=self.tie_embeddings,
            pad_token_id=self.pad_token_id,
        )


@dataclass
class SmolVLMConfig:
    """Bundle of vision + language parameters plus VL glue settings."""

    vision: SmolVLMVisionConfig
    language: SmolVLMLanguageConfig
    image_token_id: int
    pad_token_id: int
    scale_factor: int = 1
    mm_hidden_size: Optional[int] = None

    @classmethod
    def from_hf_config(cls, hf_cfg: "Idefics3Config") -> "SmolVLMConfig":
        """Create a SmolVLMConfig from a Hugging Face Idefics3Config."""

        vision_cfg = hf_cfg.vision_config
        text_cfg = hf_cfg.text_config

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
            tie_embeddings=bool(getattr(hf_cfg, "tie_word_embeddings", getattr(text_cfg, "tie_word_embeddings", False))),
            pad_token_id=int(getattr(text_cfg, "pad_token_id", getattr(hf_cfg, "pad_token_id", 0) or 0)),
        )

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

    def to_language_config(self) -> SmolLM2Config:
        cfg = self.language.to_smol_lm_config()
        cfg.pad_token_id = self.pad_token_id
        return cfg

