"""Configuration dataclass describing the published SmolLM2 architecture."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmolLM2Config:
    """Hyper-parameters required to instantiate a SmolLM2 decoder.

    The goal of this project is to mirror the released SmolLM2 checkpoints, so
    the configuration deliberately only exposes the knobs that appear in those
    models.  Anything more experimental can live in a separate folder later.
    """

    # --- model dimensions -------------------------------------------------
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int

    # --- positional encodings / context ----------------------------------
    max_seq_len: int = 4096
    rope_theta: float = 100_000.0

    # --- normalisation / regularisation ----------------------------------
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # --- misc behaviour ---------------------------------------------------
    tie_embeddings: bool = True
    init_std: float = 0.02
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be an integer multiple of n_kv_heads")

    # Historically these were methods; keep methods for drop-in compatibility.
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def rope_dim(self) -> int:
        # SmolLM2 applies rotary embeddings to the entire head dimension.
        return self.head_dim()

    # --- backwards compatibility with the previous single-file module -------
    @property
    def n_layer(self) -> int:
        return self.n_layers

    @property
    def n_head(self) -> int:
        return self.n_heads

    @property
    def n_kv_head(self) -> int:
        return self.n_kv_heads

    @property
    def rope_base(self) -> float:
        return self.rope_theta

    @property
    def rope_pct(self) -> float:
        return 1.0
