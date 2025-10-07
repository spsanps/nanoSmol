"""MME yes/no evaluation with greedy generation."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .common import SimpleModel, set_seed
from .config import MMERunConfig, load_task_config
from .reporting import ReportWriter

_YES_TOKENS = {"yes", "yeah", "y", "true", "correct", "yep"}
_NO_TOKENS = {"no", "nah", "n", "false", "incorrect", "nope"}
_WORD_PATTERN = re.compile(r"[a-zA-Z]+")


def _extract_yes_no(text: str) -> str | None:
    """Return ``"yes"``/``"no"`` if the string contains a confident answer."""

    lowered = text.lower()
    for match in _WORD_PATTERN.findall(lowered):
        if match in _YES_TOKENS:
            return "yes"
        if match in _NO_TOKENS:
            return "no"
    return None


def _format_prompt(question: str) -> str:
    return (
        "You are a vision-language assistant. Answer the question with yes or no.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def run(config: MMERunConfig) -> Dict[str, object]:
    """Evaluate MME according to ``config`` and return summary metrics."""

    set_seed(config.scoring.seed)
    model = SimpleModel(config.model)

    dataset = load_dataset("lmms-lab/MME", split=config.dataset.split)
    if config.dataset.subset_size is not None:
        dataset = dataset.select(range(min(config.dataset.subset_size, len(dataset))))

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    rows: List[Dict[str, object]] = []

    for example in tqdm(dataset, desc=f"mme:{config.dataset.split}"):
        category = str(example.get("category", "unknown"))
        question = str(example.get("question", ""))
        gold_raw = str(example.get("answer", "")).strip().lower()
        gold = "yes" if gold_raw.startswith("y") else "no"
        image = example.get("image")
        prompt = _format_prompt(question)
        if model.processor is None:
            completion = model.generate_text(
                prompt, max_new_tokens=config.generation.max_new_tokens
            )
        else:
            completion = model.generate_text(
                prompt,
                images=[image] if image is not None else None,
                max_new_tokens=config.generation.max_new_tokens,
            )
        predicted_label = _extract_yes_no(completion) or "unknown"
        is_correct = predicted_label == gold
        per_category[category]["correct"] += int(is_correct)
        per_category[category]["total"] += 1
        rows.append(
            {
                "task": "mme",
                "category": category,
                "question": question,
                "prediction": predicted_label,
                "raw_prediction": completion,
                "gold": gold,
                "correct": bool(is_correct),
            }
        )

    total_correct = sum(item["correct"] for item in per_category.values())
    total_seen = sum(item["total"] for item in per_category.values())
    accuracy = total_correct / max(1, total_seen)

    per_category_accuracy = [
        (category, counts["correct"] / max(1, counts["total"]))
        for category, counts in sorted(per_category.items())
    ]

    writer = ReportWriter(config.report, title="MME yes/no accuracy")
    writer.write_predictions(rows)
    writer.write_summary(
        {
            "task": "mme",
            "accuracy": accuracy,
            "total_examples": total_seen,
            "split": config.dataset.split,
            "subset_size": config.dataset.subset_size,
            "max_new_tokens": config.generation.max_new_tokens,
            "seed": config.scoring.seed,
            "model_id": config.model.model_id,
            "per_category": {
                category: {
                    "accuracy": counts["correct"] / max(1, counts["total"]),
                    "correct": counts["correct"],
                    "total": counts["total"],
                }
                for category, counts in per_category.items()
            },
        }
    )
    writer.write_metrics_table(per_category_accuracy)
    writer.plot_metrics(per_category_accuracy)

    return {"accuracy": accuracy, "total_examples": total_seen, "per_category": dict(per_category)}


def run_from_yaml(config_path: Path) -> Dict[str, object]:
    cfg = load_task_config(config_path)
    if not isinstance(cfg, MMERunConfig):
        raise TypeError("MME runner received a configuration for a different task")
    return run(cfg)


def _main() -> None:
    config_env = os.environ.get("NANOEVAL_CONFIG")
    if config_env is None:
        raise SystemExit("Set NANOEVAL_CONFIG to the path of an MME config YAML")
    summary = run_from_yaml(Path(config_env))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
