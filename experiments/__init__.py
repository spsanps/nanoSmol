"""Experiment registry for nanoSmol training scripts."""
from __future__ import annotations

from typing import Dict, Iterable

from .smolvlm_siglip import SmolVLMSiglipExperiment

_EXPERIMENTS: Dict[str, SmolVLMSiglipExperiment] = {
    "smolvlm-siglip": SmolVLMSiglipExperiment(),
}


def available_experiments() -> Iterable[str]:
    """Return the experiment registry keys for CLI help."""

    return sorted(_EXPERIMENTS.keys())


def get_experiment(name: str) -> SmolVLMSiglipExperiment:
    try:
        return _EXPERIMENTS[name]
    except KeyError as exc:  # pragma: no cover - sanity guard
        raise ValueError(f"Unknown experiment: {name}") from exc
