"""HellaSwag evaluation driven entirely by YAML configs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .common import LETTER4, SimpleModel, set_seed
from .config import HellaSwagRunConfig, load_task_config
from .prompts import HELLASWAG_ZERO_SHOT
from .reporting import ReportWriter


def run(config: HellaSwagRunConfig) -> Dict[str, object]:
    """Evaluate HellaSwag according to ``config`` and return a summary payload."""

    set_seed(config.scoring.seed)

    model = SimpleModel(config.model)
    dataset = load_dataset("Rowan/hellaswag", split=config.dataset.split)
    if config.dataset.subset_size is not None:
        dataset = dataset.select(range(min(config.dataset.subset_size, len(dataset))))

    correct = 0
    rows: List[Dict[str, object]] = []

    for index, example in enumerate(tqdm(dataset, desc=f"hellaswag:{config.dataset.split}")):
        prompt = HELLASWAG_ZERO_SHOT.format(
            context=example["ctx"],
            A=example["endings"][0],
            B=example["endings"][1],
            C=example["endings"][2],
            D=example["endings"][3],
        )
        gold_index = int(example["label"])

        scoring_options = [
            f"{LETTER4[idx]}. {ending}" for idx, ending in enumerate(example["endings"])
        ]
        predicted_index = model.rank_log_likelihood(prompt, scoring_options)

        is_correct = predicted_index == gold_index
        correct += int(is_correct)
        rows.append(
            {
                "task": "hellaswag",
                "index": index,
                "gold": LETTER4[gold_index],
                "prediction": LETTER4[predicted_index] if 0 <= predicted_index < 4 else "?",
                "correct": bool(is_correct),
            }
        )

    total = len(rows)
    accuracy = correct / max(1, total)

    writer = ReportWriter(config.report, title="HellaSwag accuracy")
    writer.write_predictions(rows)
    writer.write_summary(
        {
            "task": "hellaswag",
            "accuracy": accuracy,
            "total_examples": total,
            "split": config.dataset.split,
            "subset_size": config.dataset.subset_size,
            "scoring": "rank_ll",
            "seed": config.scoring.seed,
            "model_id": config.model.model_id,
        }
    )
    writer.write_metrics_table([(config.dataset.split, accuracy)])
    writer.plot_metrics([(config.dataset.split, accuracy)])

    return {"accuracy": accuracy, "total_examples": total}


def run_from_yaml(config_path: Path) -> Dict[str, object]:
    cfg = load_task_config(config_path)
    if not isinstance(cfg, HellaSwagRunConfig):
        raise TypeError("HellaSwag runner received a configuration for a different task")
    return run(cfg)


def _main() -> None:
    config_env = os.environ.get("NANOEVAL_CONFIG")
    if config_env is None:
        raise SystemExit("Set NANOEVAL_CONFIG to the path of a HellaSwag config YAML")
    summary = run_from_yaml(Path(config_env))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
