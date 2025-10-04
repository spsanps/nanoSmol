"""Config-driven MMMU-Pro evaluation with optional vision inputs."""

from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .common import SimpleModel, build_letter_choices, set_seed
from .config import MMMUProRunConfig, load_task_config
from .prompts import MMMU_VQA_ZERO_SHOT
from .reporting import ReportWriter

IMAGE_COLUMNS = tuple(f"image_{idx}" for idx in range(1, 8))


def _parse_options(raw) -> List[str]:
    """Return a list of option strings regardless of how ``raw`` is stored."""

    if isinstance(raw, list):
        return [str(option) for option in raw]

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text[0] in "[({":
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return [str(option) for option in parsed]
        candidates: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) >= 2 and stripped[1] in ".):" and stripped[0].isalnum():
                stripped = stripped[2:].strip()
            candidates.append(stripped)
        if candidates:
            return candidates
        return [text]

    raise TypeError(f"Unsupported options payload of type {type(raw)!r}")


def _ensure_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if hasattr(value, "numpy"):
        return Image.fromarray(value.numpy()).convert("RGB")
    raise TypeError(f"Unsupported image payload of type {type(value)!r}")


def _collect_images(example) -> List[Image.Image]:
    images: List[Image.Image] = []
    for key in IMAGE_COLUMNS:
        if key in example and example[key] is not None:
            images.append(_ensure_image(example[key]))
    return images


def run(config: MMMUProRunConfig) -> Dict[str, object]:
    """Evaluate MMMU-Pro according to ``config`` and return a summary payload."""

    set_seed(config.scoring.seed)
    if not config.model.is_vlm:
        raise ValueError("MMMU-Pro requires a vision-language model")
    if config.scoring.strategy != "gen_letter":
        raise ValueError(
            "MMMU-Pro currently supports only 'gen_letter' scoring; "
            f"received {config.scoring.strategy!r}"
        )

    model = SimpleModel(config.model)
    dataset = load_dataset("MMMU/MMMU_Pro", name=config.dataset.subset_name, split=config.dataset.split)
    if config.dataset.subset_size is not None:
        dataset = dataset.select(range(min(config.dataset.subset_size, len(dataset))))

    records: List[Dict[str, object]] = []
    correct = 0

    for example in tqdm(dataset, desc=f"mmmu_pro:{config.dataset.subset_name}"):
        options = _parse_options(example["options"])
        if not options:
            raise ValueError(
                "Encountered an MMMU-Pro example without options: "
                f"id={example.get('id', '')}"
            )
        letters = build_letter_choices(len(options))
        prompt = MMMU_VQA_ZERO_SHOT.format(
            letters=", ".join(letters),
            question=example["question"],
            options_block="\n".join(
                f"{letters[idx]}. {choice}" for idx, choice in enumerate(options)
            ),
        )

        images = _collect_images(example)
        message_content = [{"type": "image"} for _ in images]
        message_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": message_content}]

        prediction = (
            model.generate_letter_vlm(messages, images, letters, config.scoring.max_new_tokens)
            or "?"
        )
        gold = str(example["answer"]).strip().upper()
        if gold not in letters:
            raise ValueError(
                "Gold answer label not present in generated choice labels: "
                f"gold={gold!r}, labels={letters}, id={example.get('id', '')}",
            )
        is_correct = prediction == gold
        correct += int(is_correct)
        records.append(
            {
                "task": "mmmu_pro",
                "id": example.get("id", ""),
                "gold": gold,
                "prediction": prediction,
                "correct": bool(is_correct),
            }
        )

    total = len(records)
    accuracy = correct / max(1, total)

    writer = ReportWriter(config.report, title="MMMU-Pro accuracy")
    writer.write_predictions(records)
    writer.write_summary(
        {
            "task": "mmmu_pro",
            "accuracy": accuracy,
            "total_examples": total,
            "subset_name": config.dataset.subset_name,
            "split": config.dataset.split,
            "subset_size": config.dataset.subset_size,
            "scoring": config.scoring.strategy,
            "max_new_tokens": config.scoring.max_new_tokens,
            "seed": config.scoring.seed,
            "model_id": config.model.model_id,
        }
    )
    writer.write_metrics_table([(config.dataset.subset_name, accuracy)])
    writer.plot_metrics([(config.dataset.subset_name, accuracy)])

    return {"accuracy": accuracy, "total_examples": total}


def run_from_yaml(config_path: Path) -> Dict[str, object]:
    cfg = load_task_config(config_path)
    if not isinstance(cfg, MMMUProRunConfig):
        raise TypeError("MMMU-Pro runner received a configuration for a different task")
    return run(cfg)


def _main() -> None:
    config_env = os.environ.get("NANOEVAL_CONFIG")
    if config_env is None:
        raise SystemExit("Set NANOEVAL_CONFIG to the path of an MMMU-Pro config YAML")
    summary = run_from_yaml(Path(config_env))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
