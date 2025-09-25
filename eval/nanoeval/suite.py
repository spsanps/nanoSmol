"""Run multiple NanoEval tasks for a single model using one YAML file."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

from .config import ModelConfig, TaskConfig, build_model_config, build_task_config
from .run_hellaswag import run as run_hellaswag
from .run_mmlu import run as run_mmlu
from .run_mmmu_pro import run as run_mmmu_pro


@dataclass
class SuiteConfig:
    """Describe a bundle of evaluation tasks that share a base model."""

    name: str
    output_root: Path
    model: ModelConfig
    tasks: Sequence[TaskConfig]


def _default_output_root(name: str) -> Path:
    return Path("artifacts") / "nanoeval" / "suites" / name


def load_suite_config(path: Path) -> SuiteConfig:
    """Parse ``path`` into a :class:`SuiteConfig`."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise TypeError("Suite YAML must contain a mapping")

    suite_meta = payload.get("suite") or {}
    name = str(suite_meta.get("name", Path(path).stem))

    output_root_raw = suite_meta.get("output_root")
    output_root = _default_output_root(name) if output_root_raw is None else Path(output_root_raw)

    model_payload = payload.get("model")
    if not model_payload:
        raise ValueError("Suite config must provide a 'model' section")
    model = build_model_config(model_payload)

    task_entries = payload.get("tasks")
    if not isinstance(task_entries, list) or not task_entries:
        raise ValueError("Suite config must list at least one task")

    tasks: List[TaskConfig] = []
    for entry in task_entries:
        if not isinstance(entry, dict):
            raise TypeError("Each task entry must be a mapping")
        task_cfg = build_task_config(entry, default_model=model, default_report_dir=None)
        # Unless the user explicitly overrides ``output_dir`` we tuck the
        # artefacts under ``<output_root>/<task-name>/``.
        report_section = entry.get("report") if isinstance(entry.get("report"), dict) else {}
        if "output_dir" not in report_section:
            task_cfg.report.output_dir = output_root / task_cfg.task
        tasks.append(task_cfg)

    return SuiteConfig(name=name, output_root=output_root, model=model, tasks=tuple(tasks))


def _run_task(task: TaskConfig) -> Dict[str, object]:
    if task.task == "mmlu":  # type: ignore[comparison-overlap]
        summary = run_mmlu(task)  # type: ignore[arg-type]
    elif task.task == "hellaswag":
        summary = run_hellaswag(task)  # type: ignore[arg-type]
    else:
        summary = run_mmmu_pro(task)  # type: ignore[arg-type]
    return {"task": task.task, **summary}


def run_suite(config: SuiteConfig) -> Sequence[Dict[str, object]]:
    """Execute every task in ``config`` and persist a suite-level summary."""

    summaries = [_run_task(task) for task in config.tasks]

    config.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": config.name,
        "model_id": config.model.model_id,
        "num_tasks": len(summaries),
        "results": summaries,
    }
    summary_path = config.output_root / "suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return summaries


def run_suite_from_yaml(path: Path) -> Sequence[Dict[str, object]]:
    return run_suite(load_suite_config(path))


def _main() -> None:
    suite_env = os.environ.get("NANOEVAL_SUITE")
    if suite_env is None:
        raise SystemExit("Set NANOEVAL_SUITE to the path of a suite YAML file")
    results = run_suite_from_yaml(Path(suite_env))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()
