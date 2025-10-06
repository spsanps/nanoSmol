"""Config-driven MMLU evaluation following the NanoGPT philosophy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset
from tqdm import tqdm

from .common import LETTER4, SimpleModel, set_seed
from .config import MMLURunConfig, load_task_config
from .reporting import ReportWriter
from .prompts import MMLU_ZERO_SHOT

SUBJECTS: Sequence[str] = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def _select_subjects(config: MMLURunConfig) -> Sequence[str]:
    requested = config.dataset.subjects
    if requested:
        return tuple(requested)
    return SUBJECTS


def _summarise(per_subject: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    macro = sum(item["correct"] / max(1, item["total"]) for item in per_subject.values()) / max(
        1, len(per_subject)
    )
    total_correct = sum(item["correct"] for item in per_subject.values())
    total_seen = sum(item["total"] for item in per_subject.values())
    per_subject_accuracy = {
        subject: {
            "accuracy": counts["correct"] / max(1, counts["total"]),
            "correct": counts["correct"],
            "total": counts["total"],
        }
        for subject, counts in per_subject.items()
    }
    return {
        "task": "mmlu",
        "macro_accuracy": macro,
        "micro_accuracy": total_correct / max(1, total_seen),
        "total_examples": total_seen,
        "per_subject": per_subject_accuracy,
    }


def run(config: MMLURunConfig) -> Dict[str, object]:
    """Evaluate MMLU according to ``config`` and return the summary payload."""

    set_seed(config.scoring.seed)

    model = SimpleModel(config.model)
    subjects = _select_subjects(config)
    per_subject_counts: Dict[str, Dict[str, int]] = {}
    records: List[Dict[str, object]] = []

    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split=config.dataset.split)
        if config.dataset.subset_size is not None:
            dataset = dataset.select(range(min(config.dataset.subset_size, len(dataset))))

        correct = 0
        total = 0
        for index, example in enumerate(tqdm(dataset, desc=f"mmlu:{subject}", leave=False)):
            prompt = MMLU_ZERO_SHOT.format(
                question=example["question"],
                A=example["choices"][0],
                B=example["choices"][1],
                C=example["choices"][2],
                D=example["choices"][3],
            )
            gold_index = int(example["answer"])
            gold_letter = LETTER4[gold_index]
            scoring_options = [str(choice) for choice in example["choices"]]
            choice_index = model.rank_log_likelihood(prompt, scoring_options, normalize=config.scoring.normalize_by_length)
            predicted = LETTER4[choice_index]
            is_correct = predicted == gold_letter
            correct += int(is_correct)
            total += 1
            records.append(
                {
                    "task": "mmlu",
                    "subject": subject,
                    "index": index,
                    "gold": gold_letter,
                    "prediction": predicted,
                    "correct": bool(is_correct),
                }
            )

        per_subject_counts[subject] = {"correct": correct, "total": total}

    summary = _summarise(per_subject_counts)

    writer = ReportWriter(config.report, title="MMLU macro accuracy")
    writer.write_predictions(records)
    writer.write_summary(
        {
            **summary,
            "scoring": "rank_ll",
            "seed": config.scoring.seed,
            "subjects": list(subjects),
            "split": config.dataset.split,
            "subset_size": config.dataset.subset_size,
            "model_id": config.model.model_id,
        }
    )
    metrics = [
        (subject.replace("_", " "), counts["correct"] / max(1, counts["total"]))
        for subject, counts in per_subject_counts.items()
    ]
    writer.write_metrics_table(metrics)
    writer.plot_metrics(metrics)

    return summary


def run_from_yaml(config_path: Path) -> Dict[str, object]:
    cfg = load_task_config(config_path)
    if not isinstance(cfg, MMLURunConfig):
        raise TypeError("MMLU runner received a configuration for a different task")
    return run(cfg)


def _main() -> None:
    config_env = os.environ.get("NANOEVAL_CONFIG")
    if config_env is None:
        raise SystemExit("Set NANOEVAL_CONFIG to the path of an MMLU config YAML")
    summary = run_from_yaml(Path(config_env))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
