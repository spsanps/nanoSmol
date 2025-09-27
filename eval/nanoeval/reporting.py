"""Utility helpers for persisting NanoEval results and diagnostic plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from .config import ReportConfig


class ReportWriter:
    """Persist predictions, aggregate metrics, and quick-look bar plots."""

    def __init__(self, cfg: ReportConfig, title: str) -> None:
        self.cfg = cfg
        self.title = title
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- predictions
    def write_predictions(self, rows: Iterable[Mapping[str, object]]) -> None:
        if not self.cfg.save_predictions:
            return
        destination = self.cfg.output_dir / self.cfg.predictions_filename
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------- summary
    def write_summary(self, payload: Mapping[str, object]) -> None:
        destination = self.cfg.output_dir / self.cfg.summary_filename
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------- table
    def write_metrics_table(self, metrics: Sequence[tuple[str, float]]) -> None:
        if not self.cfg.save_table:
            return
        destination = self.cfg.output_dir / self.cfg.table_filename
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "value"])
            for label, value in metrics:
                writer.writerow([label, value])

    # ---------------------------------------------------------------------- plot
    def plot_metrics(self, metrics: Sequence[tuple[str, float]]) -> None:
        if not (self.cfg.save_plot and metrics):
            return

        destination = self.cfg.output_dir / self.cfg.plot_filename
        labels = [label for label, _ in metrics]
        values = [value for _, value in metrics]

        fig_height = max(2.5, 0.45 * len(metrics) + 1.0)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        positions = range(len(metrics))
        ax.barh(positions, values, color="#4f81bd", alpha=0.85)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Accuracy")
        ax.set_title(self.title)
        ax.invert_yaxis()

        for pos, value in zip(positions, values):
            ax.text(value + 0.01, pos, f"{value:.3f}", va="center")

        xmax = max(values)
        if xmax <= 0:
            xmax = 1.0
        ax.set_xlim(0.0, min(1.0, xmax * 1.05) if xmax <= 1.0 else xmax * 1.05)

        fig.tight_layout()
        fig.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(fig)


__all__ = ["ReportWriter"]
