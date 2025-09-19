#!/usr/bin/env python
"""Utility to summarise LightEval `results_*.json` payloads.

The script prints a compact table, optionally emits a CSV dump, and can draw a
horizontal bar plot with error bars (if the stderr columns are present).  It is
agnostic to the task suite, so any LightEval run that follows the standard
tracker layout is supported.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PRIMARY_METRIC_CANDIDATES = ("acc", "accuracy", "score", "exact_match", "em", "f1")


def _load_results(results_path: Path) -> dict[str, dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "results" not in payload:
        raise KeyError(f"`results` key not found in {results_path}")
    return payload["results"]


def _is_summary_task(task_name: str) -> bool:
    if task_name == "all":
        return True
    parts = task_name.split("|")
    if len(parts) <= 2:
        return True
    task_segment = parts[1]
    return ":" not in task_segment or task_segment.endswith("_average")


def _filter_tasks(
    raw_results: dict[str, dict[str, Any]], mode: str
) -> dict[str, dict[str, Any]]:
    if mode == "full":
        return raw_results
    return {name: metrics for name, metrics in raw_results.items() if _is_summary_task(name)}


def _pick_metric(metrics: dict[str, Any], preferred: str | None) -> tuple[str, float, float | None]:
    def _valid(key: str) -> bool:
        value = metrics.get(key)
        return not key.endswith("_stderr") and isinstance(value, (int, float))

    if preferred:
        if not _valid(preferred):
            raise KeyError(f"Metric '{preferred}' not present in {metrics}")
        return preferred, float(metrics[preferred]), metrics.get(f"{preferred}_stderr")

    for candidate in PRIMARY_METRIC_CANDIDATES:
        if _valid(candidate):
            return candidate, float(metrics[candidate]), metrics.get(f"{candidate}_stderr")

    for key, value in metrics.items():
        if _valid(key):
            return key, float(value), metrics.get(f"{key}_stderr")
    raise ValueError("No scalar metrics found in LightEval results")


def _format_task_label(task_name: str) -> str:
    if task_name == "all":
        return "all"
    parts = task_name.split("|")
    if len(parts) == 1:
        return task_name
    label = f"{parts[0]} · {parts[1]}"
    if len(parts) >= 3:
        label = f"{label} · {parts[2]}"
    return label.replace(":_average", ":avg")


def _prepare_rows(
    results: dict[str, dict[str, Any]],
    metric_name: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, metrics in results.items():
        try:
            chosen_metric, value, stderr = _pick_metric(metrics, metric_name)
        except (KeyError, ValueError):
            continue
        rows.append(
            {
                "task": task_name,
                "label": _format_task_label(task_name),
                "metric": chosen_metric,
                "value": value,
                "stderr": float(stderr) if isinstance(stderr, (int, float)) else None,
            }
        )
    if not rows:
        raise ValueError("No metrics survived filtering; check the results JSON and CLI options")
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'Task':<60} {'Metric':<12} {'Value':>8} {'StdErr':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        stderr = "" if row["stderr"] is None else f"{row['stderr']:.4f}"
        print(f"{row['label']:<60} {row['metric']:<12} {row['value']:>8.4f} {stderr:>8}")


def _write_csv(destination: Path, rows: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "metric", "value", "stderr"])
        writer.writeheader()
        writer.writerows(
            {k: ("" if v is None else v) for k, v in row.items() if k in writer.fieldnames}
            for row in rows
        )


def _plot(rows: list[dict[str, Any]], output_path: Path, title: str | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in rows]
    values = [row["value"] for row in rows]
    errors = [row["stderr"] or 0.0 for row in rows]

    fig_height = max(2.5, 0.5 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    positions = range(len(rows))
    ax.barh(positions, values, xerr=errors, color="#377eb8", alpha=0.85)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title(title or "LightEval summary")
    ax.invert_yaxis()  # Highest score at the top

    xmax = max(values) if values else 1.0
    ax.set_xlim(0, xmax * 1.05)

    for pos, value in zip(positions, values):
        ax.text(value + xmax * 0.01, pos, f"{value:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LightEval results")
    parser.add_argument("--results-json", type=Path, required=True, help="Path to results_*.json from LightEval")
    parser.add_argument("--metric", type=str, default=None, help="Metric name to plot (defaults to the first scalar metric)")
    parser.add_argument("--mode", choices=["summary", "full"], default="summary", help="Filter strategy for task rows")
    parser.add_argument("--sort", choices=["task", "value"], default="task", help="Sort order for the output table")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the generated plot image")
    parser.add_argument("--table", type=Path, default=None, help="Optional CSV dump of the aggregated scores")
    parser.add_argument("--title", type=str, default=None, help="Custom title for the plot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = _load_results(args.results_json)
    filtered = _filter_tasks(results, args.mode)
    rows = _prepare_rows(filtered, args.metric)

    if args.sort == "value":
        rows.sort(key=lambda row: row["value"], reverse=True)
    else:
        rows.sort(key=lambda row: row["label"].lower())

    _print_table(rows)

    if args.table:
        _write_csv(args.table, rows)
    if args.output:
        _plot(rows, args.output, args.title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
