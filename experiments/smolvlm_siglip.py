"""SmolVLM + SigLIP fine-tuning experiment wiring."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from models.smolVLM.config import SmolVLMConfig
from models.smolVLM.model import SmolVLM


@dataclass
class ExperimentArtifacts:
    """Bundle of objects returned by :func:`SmolVLMSiglipExperiment.build`."""

    model: SmolVLM
    tokenizer: PreTrainedTokenizerBase
    model_config: SmolVLMConfig


class SmolVLMSiglipExperiment:
    """Configure SmolVLM for multimodal fine-tuning with SigLIP vision."""

    default_model_id = "HuggingFaceM4/smolvlm-instruct"
    default_dataset = "HuggingFaceM4/FineVision-1.0"
    default_adapter = "finevision"

    def apply_defaults(self, args) -> None:
        """Fill any unset CLI flags with experiment-specific defaults."""

        if getattr(args, "model", None) is None:
            args.model = self.default_model_id
        if getattr(args, "dataset", None) is None:
            args.dataset = self.default_dataset
        if getattr(args, "adapter", None) is None:
            args.adapter = self.default_adapter

    # ------------------------------------------------------------------ builder
    def build(self, args) -> ExperimentArtifacts:
        """Instantiate SmolVLM and load pretrained weights from Hugging Face."""

        hf_cfg = AutoConfig.from_pretrained(args.model, revision=args.model_revision, token=args.model_token)
        model_cfg = SmolVLMConfig.from_hf_config(hf_cfg)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.model_revision,
            token=args.model_token,
            use_fast=True,
        )

        model = SmolVLM(model_cfg)
        state_dict = self._download_state_dict(
            args.model,
            revision=args.model_revision,
            token=args.model_token,
            local_only=args.model_local_only,
        )
        try:
            model.load_hf_state_dict(state_dict, strict=False, verbose=True)
        finally:
            # Release the (potentially multi-gigabyte) checkpoint immediately.
            state_dict.clear()
        return ExperimentArtifacts(model=model, tokenizer=tokenizer, model_config=model_cfg)

    # ----------------------------------------------------------------- weights
    def _download_state_dict(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        local_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Fetch a Hugging Face checkpoint and return it as a state dict."""

        for index_name in self._index_search_paths():
            try:
                index_path = hf_hub_download(
                    repo_id,
                    index_name,
                    revision=revision,
                    token=token,
                    local_files_only=local_only,
                )
            except Exception:
                continue
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_files = sorted(set(index_data.get("weight_map", {}).values()))
            if not weight_files:
                continue
            return self._load_weight_shards(
                repo_id,
                weight_files,
                revision=revision,
                token=token,
                local_only=local_only,
            )

        for single_name in self._single_file_candidates():
            try:
                file_path = hf_hub_download(
                    repo_id,
                    single_name,
                    revision=revision,
                    token=token,
                    local_files_only=local_only,
                )
            except Exception:
                continue
            return self._load_state_file(Path(file_path))

        tried = list(self._index_search_paths()) + list(self._single_file_candidates())
        raise RuntimeError(
            "Could not locate model weights in repo "
            f"'{repo_id}'. Checked: {', '.join(tried)}"
        )

    @staticmethod
    def _index_search_paths() -> Iterable[str]:
        return (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )

    @staticmethod
    def _single_file_candidates() -> Iterable[str]:
        return (
            "model.safetensors",
            "pytorch_model.bin",
        )

    def _load_weight_shards(
        self,
        repo_id: str,
        shard_names: Iterable[str],
        *,
        revision: Optional[str],
        token: Optional[str],
        local_only: bool,
    ) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            shard_path = hf_hub_download(
                repo_id,
                shard_name,
                revision=revision,
                token=token,
                local_files_only=local_only,
            )
            state.update(self._load_state_file(Path(shard_path)))
        return state

    @staticmethod
    def _load_state_file(path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix == ".safetensors":
            return load_safetensors(str(path))
        return torch.load(str(path), map_location="cpu")
