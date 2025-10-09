import json
from types import SimpleNamespace

import pytest

# Skip the suite entirely if torch is unavailable in the execution env.
torch = pytest.importorskip("torch")

from experiments import smolvlm_siglip


def test_apply_defaults_backfills_missing_fields():
    exp = smolvlm_siglip.SmolVLMSiglipExperiment()
    args = SimpleNamespace(model=None, dataset=None, adapter=None)
    exp.apply_defaults(args)
    assert args.model == exp.default_model_id
    assert args.dataset == exp.default_dataset
    assert args.adapter == exp.default_adapter


def test_apply_defaults_respects_user_values():
    exp = smolvlm_siglip.SmolVLMSiglipExperiment()
    args = SimpleNamespace(model="custom-model", dataset="custom-dataset", adapter="my-adapter")
    exp.apply_defaults(args)
    assert args.model == "custom-model"
    assert args.dataset == "custom-dataset"
    assert args.adapter == "my-adapter"


def test_download_state_dict_uses_index(monkeypatch, tmp_path):
    exp = smolvlm_siglip.SmolVLMSiglipExperiment()

    index = tmp_path / "pytorch_model.bin.index.json"
    shard = tmp_path / "pytorch_model-00001-of-00001.bin"
    torch.save({"layer.weight": torch.ones(2)}, shard)
    index.write_text(json.dumps({"weight_map": {"layer.weight": shard.name}}), encoding="utf-8")

    def fake_download(repo_id, filename, **kwargs):
        mapping = {
            "pytorch_model.bin.index.json": index,
            shard.name: shard,
        }
        path = mapping.get(filename)
        if path is None:
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr(smolvlm_siglip, "hf_hub_download", fake_download)

    state = exp._download_state_dict("repo")
    assert state["layer.weight"].shape == (2,)


def test_download_state_dict_falls_back_to_single_file(monkeypatch, tmp_path):
    exp = smolvlm_siglip.SmolVLMSiglipExperiment()

    safetensor_path = tmp_path / "model.safetensors"
    from safetensors.torch import save_file

    save_file({"layer.bias": torch.zeros(3)}, str(safetensor_path))

    def fake_download(repo_id, filename, **kwargs):
        if filename in {"model.safetensors.index.json", "pytorch_model.bin.index.json", "pytorch_model.bin"}:
            raise FileNotFoundError(filename)
        if filename == "model.safetensors":
            return str(safetensor_path)
        raise FileNotFoundError(filename)

    monkeypatch.setattr(smolvlm_siglip, "hf_hub_download", fake_download)

    state = exp._download_state_dict("repo")
    assert state["layer.bias"].shape == (3,)
