"""Dataclasses describing the YAML configuration for NanoEval tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml


@dataclass
class ModelConfig:
    """Configuration needed to load a Hugging Face model."""

    model_id: str
    is_vlm: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = True
    attn_impl: str | None = None


@dataclass
class ScoringConfig:
    """Options controlling deterministic scoring."""

    seed: int = 123


@dataclass
class ReportConfig:
    """Where to store predictions, summaries, and diagnostic plots."""

    output_dir: Path
    summary_filename: str = "summary.json"
    predictions_filename: str = "predictions.jsonl"
    table_filename: str = "metrics.csv"
    plot_filename: str = "accuracy.png"
    save_predictions: bool = True
    save_table: bool = True
    save_plot: bool = True


@dataclass
class MMLUDatasetConfig:
    split: str = "test"
    subjects: Sequence[str] | None = None
    subset_size: int | None = None


@dataclass
class HellaSwagDatasetConfig:
    split: Literal["validation", "test"] = "validation"
    subset_size: int | None = None


@dataclass
class MMMUProDatasetConfig:
    subset_name: Literal["standard (10 options)", "standard (4 options)", "vision"] = (
        "standard (10 options)"
    )
    split: str = "test"
    subset_size: int | None = None


@dataclass
class MMLURunConfig:
    task: Literal["mmlu"]
    model: ModelConfig
    dataset: MMLUDatasetConfig = field(default_factory=MMLUDatasetConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    report: ReportConfig = field(
        default_factory=lambda: ReportConfig(Path("artifacts/nanoeval/mmlu"))
    )


@dataclass
class HellaSwagRunConfig:
    task: Literal["hellaswag"]
    model: ModelConfig
    dataset: HellaSwagDatasetConfig = field(default_factory=HellaSwagDatasetConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    report: ReportConfig = field(
        default_factory=lambda: ReportConfig(Path("artifacts/nanoeval/hellaswag"))
    )


@dataclass
class MMMUProRunConfig:
    task: Literal["mmmu_pro"]
    model: ModelConfig
    dataset: MMMUProDatasetConfig = field(default_factory=MMMUProDatasetConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    report: ReportConfig = field(
        default_factory=lambda: ReportConfig(Path("artifacts/nanoeval/mmmu_pro"))
    )


TaskConfig = MMLURunConfig | HellaSwagRunConfig | MMMUProRunConfig


def _ensure_report_config(
    task: str, data: Mapping[str, Any], default_dir: Path | None = None
) -> ReportConfig:
    output_dir_raw = data.get("output_dir")
    if output_dir_raw is not None:
        target_dir = Path(str(output_dir_raw))
    elif default_dir is not None:
        target_dir = Path(default_dir)
    else:
        target_dir = Path("artifacts") / "nanoeval" / task
    return ReportConfig(
        output_dir=target_dir,
        summary_filename=data.get("summary_filename", "summary.json"),
        predictions_filename=data.get("predictions_filename", "predictions.jsonl"),
        table_filename=data.get("table_filename", "metrics.csv"),
        plot_filename=data.get("plot_filename", "accuracy.png"),
        save_predictions=bool(data.get("save_predictions", True)),
        save_table=bool(data.get("save_table", True)),
        save_plot=bool(data.get("save_plot", True)),
    )


def build_model_config(data: Mapping[str, Any]) -> ModelConfig:
    return ModelConfig(
        model_id=data["model_id"],
        is_vlm=bool(data.get("is_vlm", False)),
        dtype=str(data.get("dtype", "bfloat16")),
        device=str(data.get("device", "cuda")),
        trust_remote_code=bool(data.get("trust_remote_code", True)),
        attn_impl=data.get("attn_impl"),
    )


def _build_scoring_config(data: Mapping[str, Any] | None) -> ScoringConfig:
    if not data:
        return ScoringConfig()
    return ScoringConfig(seed=int(data.get("seed", 123)))


def _build_mmlu_dataset(data: Mapping[str, Any] | None) -> MMLUDatasetConfig:
    if not data:
        return MMLUDatasetConfig()
    subjects = data.get("subjects")
    if isinstance(subjects, str):
        subjects = [subjects]
    elif subjects is not None and not isinstance(subjects, Sequence):
        raise TypeError("`subjects` must be a sequence of subject names or omitted")
    return MMLUDatasetConfig(
        split=str(data.get("split", "test")),
        subjects=tuple(subjects) if subjects else None,
        subset_size=(int(data["subset_size"]) if data.get("subset_size") is not None else None),
    )


def _build_hellaswag_dataset(data: Mapping[str, Any] | None) -> HellaSwagDatasetConfig:
    if not data:
        return HellaSwagDatasetConfig()
    subset_raw = data.get("subset_size")
    return HellaSwagDatasetConfig(
        split=str(data.get("split", "validation")),
        subset_size=(int(subset_raw) if subset_raw is not None else None),
    )


def _build_mmmu_dataset(data: Mapping[str, Any] | None) -> MMMUProDatasetConfig:
    if not data:
        return MMMUProDatasetConfig()
    subset_raw = data.get("subset_size")
    return MMMUProDatasetConfig(
        subset_name=str(data.get("subset_name", "standard (10 options)")),
        split=str(data.get("split", "test")),
        subset_size=(int(subset_raw) if subset_raw is not None else None),
    )


def load_task_config(path: Path) -> TaskConfig:
    """Parse ``path`` into the strongly-typed configuration objects."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] = yaml.safe_load(handle)

    task_type = payload.get("task")
    if task_type not in {"mmlu", "hellaswag", "mmmu_pro"}:
        raise ValueError("Config must set task to 'mmlu', 'hellaswag', or 'mmmu_pro'")

    return build_task_config(payload)


def build_task_config(
    payload: Mapping[str, Any],
    *,
    default_model: ModelConfig | None = None,
    default_report_dir: Path | None = None,
) -> TaskConfig:
    """Construct a :class:`TaskConfig` from a mapping.

    ``load_task_config`` uses this helper for single-task YAML files, while the
    suite runner reuses it to compose multiple tasks that share one model.
    """

    task_type = payload.get("task")
    if task_type not in {"mmlu", "hellaswag", "mmmu_pro"}:
        raise ValueError("Config must set task to 'mmlu', 'hellaswag', or 'mmmu_pro'")

    model_payload = payload.get("model")
    if model_payload is not None:
        model = build_model_config(model_payload)
    elif default_model is not None:
        model = default_model
    else:
        raise ValueError("Task entry is missing 'model' configuration")

    scoring = _build_scoring_config(payload.get("scoring"))
    report = _ensure_report_config(task_type, payload.get("report", {}), default_report_dir)

    if task_type == "mmlu":
        dataset = _build_mmlu_dataset(payload.get("dataset"))
        return MMLURunConfig(task="mmlu", model=model, dataset=dataset, scoring=scoring, report=report)
    if task_type == "hellaswag":
        dataset = _build_hellaswag_dataset(payload.get("dataset"))
        return HellaSwagRunConfig(
            task="hellaswag", model=model, dataset=dataset, scoring=scoring, report=report
        )
    dataset = _build_mmmu_dataset(payload.get("dataset"))
    return MMMUProRunConfig(task="mmmu_pro", model=model, dataset=dataset, scoring=scoring, report=report)


__all__ = [
    "ModelConfig",
    "ScoringConfig",
    "ReportConfig",
    "MMLUDatasetConfig",
    "HellaSwagDatasetConfig",
    "MMMUProDatasetConfig",
    "MMLURunConfig",
    "HellaSwagRunConfig",
    "MMMUProRunConfig",
    "TaskConfig",
    "load_task_config",
    "build_model_config",
    "build_task_config",
]
