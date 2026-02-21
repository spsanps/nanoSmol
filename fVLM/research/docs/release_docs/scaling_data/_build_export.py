#!/usr/bin/env python3
"""Build scaling data export CSVs from raw metrics and run summaries."""

import csv
import json
import os
import sys

# ── Configuration ──────────────────────────────────────────────────────────

METRICS_FILES = {
    "135M-C1-F": ("135M", "135M-C1-F", "cosine",
                   "/workspace/checkpoints/scaling/135M-C1-F/metrics_135M-C1-F_20260216_080531.csv"),
    "135M-C2-F": ("135M", "135M-C2-F", "cosine",
                   "/workspace/checkpoints/scaling/135M-C2-F/metrics_135M-C2-F_20260216_193400.csv"),
    "135M-C3-F": ("135M", "135M-C3-F", "cosine",
                   "/workspace/checkpoints/scaling/135M-C3-F/metrics_135M-C3-F_20260216_205554.csv"),
    "135M-C4-F": ("135M", "135M-C4-F", "cosine",
                   "/workspace/checkpoints/scaling/135M-C4-F/metrics_135M-C4-F_20260217_041440.csv"),
    "360M-scaling": ("360M", "360M-scaling", "constant",
                     "/workspace/checkpoints/scaling/360M-scaling/metrics_360M-scaling-constant-lr_20260217_063408.csv"),
}

RUN_SUMMARY_FILES = [
    "/workspace/checkpoints/scaling/135M-C1-F/run_summary_135M-C1-F_20260216_080531.json",
    "/workspace/checkpoints/scaling/135M-C2-F/run_summary_135M-C2-F_20260216_193400.json",
    "/workspace/checkpoints/scaling/135M-C3-F/run_summary_135M-C3-F_20260216_205554.json",
    "/workspace/checkpoints/scaling/360M-scaling/run_summary_360M-scaling-constant-lr_20260217_063408.json",
]

PARAM_COUNTS = {"135M": 157600000, "360M": 382600000}

OUTPUT_DIR = "/workspace/workdir/nanoSmol/fVLM/release/docs/scaling_data"


def parse_metrics_file(path, model_size, run_id, lr_schedule):
    """Parse a metrics CSV and return (train_rows, eval_rows) with metadata."""
    train_rows = []
    eval_rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = row.get("event_type", "").strip()
            step = int(row["step"])
            samples = int(row["samples_seen"])

            if event_type == "train":
                train_rows.append({
                    "step": step,
                    "samples_seen": samples,
                    "train_loss": float(row["train_loss"]) if row.get("train_loss") else None,
                    "lr": row.get("lr_connector", ""),
                    "throughput_sps": row.get("throughput_samples_sec", ""),
                })
            elif event_type == "eval":
                eval_rows.append({
                    "model_size": model_size,
                    "run_id": run_id,
                    "lr_schedule": lr_schedule,
                    "step": step,
                    "samples_seen": samples,
                    "val_loss": float(row["val_loss"]) if row.get("val_loss") else None,
                    "attention_entropy": float(row["attention_entropy"]) if row.get("attention_entropy") else None,
                })

    return train_rows, eval_rows


def build_all_eval_points():
    """Build combined eval CSV with closest preceding train loss."""
    all_eval = []

    for key, (model_size, run_id, lr_schedule, path) in METRICS_FILES.items():
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue

        train_rows, eval_rows = parse_metrics_file(path, model_size, run_id, lr_schedule)

        for ev in eval_rows:
            closest_train = None
            for tr in train_rows:
                if tr["step"] <= ev["step"]:
                    if closest_train is None or tr["step"] > closest_train["step"]:
                        closest_train = tr

            ev["train_loss"] = closest_train["train_loss"] if closest_train else ""
            ev["lr"] = closest_train["lr"] if closest_train else ""
            ev["throughput_sps"] = closest_train["throughput_sps"] if closest_train else ""
            all_eval.append(ev)

    all_eval.sort(key=lambda x: (x["model_size"], x["run_id"], x["step"]))

    out_path = os.path.join(OUTPUT_DIR, "all_eval_points.csv")
    fieldnames = ["model_size", "run_id", "lr_schedule", "step", "samples_seen",
                  "val_loss", "train_loss", "lr", "throughput_sps", "attention_entropy"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_eval:
            writer.writerow(row)

    print(f"  Wrote {len(all_eval)} eval points to {out_path}")
    return all_eval


def build_run_summaries():
    """Build run summary CSV from JSON summaries + grid results."""
    summaries = []
    seen_run_ids = set()

    lr_map = {
        "135M-C1-F": "cosine",
        "135M-C2-F": "cosine",
        "135M-C3-F": "cosine",
        "135M-C4-F": "cosine",
        "360M-scaling": "constant",
    }

    for path in RUN_SUMMARY_FILES:
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        raw_id = data["run_id"]
        short_id = None
        for key in lr_map:
            if key in raw_id or raw_id.startswith(key):
                short_id = key
                break
        if short_id is None:
            short_id = raw_id.rsplit("_", 2)[0] if "_20" in raw_id else raw_id

        model_size = "135M" if "135M" in short_id else "360M"

        summaries.append({
            "run_id": short_id,
            "model_size": model_size,
            "total_params": PARAM_COUNTS[model_size],
            "lr_schedule": lr_map.get(short_id, "unknown"),
            "total_samples": data.get("total_samples", ""),
            "total_steps": data.get("total_steps", ""),
            "best_val_loss": data.get("best_val_loss", ""),
            "best_val_step": data.get("best_val_step", ""),
            "final_train_loss": data.get("final_train_loss", ""),
            "wall_time_hours": round(data.get("wall_time_hours", 0), 4),
        })
        seen_run_ids.add(short_id)

    # C4 has no run summary — construct from metrics
    if "135M-C4-F" not in seen_run_ids:
        c4_path = METRICS_FILES["135M-C4-F"][3]
        if os.path.exists(c4_path):
            train_rows, eval_rows = parse_metrics_file(
                c4_path, "135M", "135M-C4-F", "cosine")

            best_eval = min(eval_rows, key=lambda x: x["val_loss"]) if eval_rows else {}
            last_train = train_rows[-1] if train_rows else {}

            # Get wall time from the last row in the raw CSV
            wall_hours = ""
            with open(c4_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    wt = row.get("wall_time_sec", "")
                if wt:
                    wall_hours = round(float(wt) / 3600, 4)

            summaries.append({
                "run_id": "135M-C4-F",
                "model_size": "135M",
                "total_params": PARAM_COUNTS["135M"],
                "lr_schedule": "cosine",
                "total_samples": last_train.get("samples_seen", ""),
                "total_steps": last_train.get("step", ""),
                "best_val_loss": best_eval.get("val_loss", ""),
                "best_val_step": best_eval.get("step", ""),
                "final_train_loss": last_train.get("train_loss", ""),
                "wall_time_hours": wall_hours,
            })
            seen_run_ids.add("135M-C4-F")
            print("  Constructed C4 summary from metrics (run was interrupted at step 12060/~39100)")

    summaries.sort(key=lambda x: (x["model_size"], x["run_id"]))

    out_path = os.path.join(OUTPUT_DIR, "run_summaries.csv")
    fieldnames = ["run_id", "model_size", "total_params", "lr_schedule",
                  "total_samples", "total_steps", "best_val_loss", "best_val_step",
                  "final_train_loss", "wall_time_hours"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"  Wrote {len(summaries)} run summaries to {out_path}")
    return summaries


if __name__ == "__main__":
    print("Building all_eval_points.csv...")
    build_all_eval_points()
    print()
    print("Building run_summaries.csv...")
    build_run_summaries()
    print()
    print("Done.")
