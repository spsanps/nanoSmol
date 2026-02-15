#!/usr/bin/env python3
"""Analyze Phase 1a ablation results.

Reads run summaries and metrics CSVs from checkpoint directories,
produces comparison tables and key analysis.

Usage:
    python release/scripts/analyze_ablations.py
    python release/scripts/analyze_ablations.py --ckpt-dir /workspace/checkpoints/ablations
    python release/scripts/analyze_ablations.py --verbose
"""

import argparse
import json
import glob
import os
import csv
from pathlib import Path


def load_run_summary(ckpt_dir: str) -> dict:
    """Load run_summary JSON from a checkpoint directory."""
    jsons = sorted(glob.glob(os.path.join(ckpt_dir, "run_summary_*.json")))
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        return json.load(f)


def load_metrics_csv(ckpt_dir: str) -> list:
    """Load metrics CSV rows from a checkpoint directory."""
    csvs = sorted(glob.glob(os.path.join(ckpt_dir, "metrics_*.csv")))
    if not csvs:
        return []
    rows = []
    with open(csvs[-1]) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(dict(zip(header, row)))
    return rows


def get_eval_rows(rows: list) -> list:
    """Filter to eval rows only."""
    return [r for r in rows if r.get("event_type") == "eval"]


def get_last_train_row(rows: list) -> dict:
    """Get the last training row."""
    train_rows = [r for r in rows if r.get("event_type") == "train"]
    return train_rows[-1] if train_rows else None


def analyze(ckpt_base: str, verbose: bool = False):
    """Main analysis."""
    # Find all ablation directories
    dirs = sorted(glob.glob(os.path.join(ckpt_base, "*")))
    dirs = [d for d in dirs if os.path.isdir(d)]

    # Known broken runs (BUG-010: deep_query=false was a no-op before fix)
    BROKEN_RUNS = {"A1_deep_query_off"}

    results = []
    for d in dirs:
        name = os.path.basename(d)
        summary = load_run_summary(d)
        rows = load_metrics_csv(d)
        last_train = get_last_train_row(rows)
        eval_rows = get_eval_rows(rows)

        if summary is None and not rows:
            continue

        entry = {
            "name": name,
            "best_val_loss": None,
            "best_val_step": None,
            "final_train_loss": None,
            "total_steps": None,
            "wall_hours": None,
            "samp_per_sec": None,
            "attention_entropy": None,
            "status": "unknown",
        }

        if summary:
            entry["best_val_loss"] = summary.get("best_val_loss")
            entry["best_val_step"] = summary.get("best_val_step")
            entry["final_train_loss"] = summary.get("final_train_loss")
            entry["total_steps"] = summary.get("total_steps")
            entry["wall_hours"] = summary.get("wall_time_hours")
            entry["status"] = "complete"
        elif last_train:
            entry["total_steps"] = int(last_train.get("step", 0))
            entry["final_train_loss"] = float(last_train.get("train_loss", 0))
            entry["status"] = f"partial ({entry['total_steps']}/9375)"

        if last_train:
            entry["samp_per_sec"] = float(last_train.get("throughput_samples_sec", 0))
            entry["gpu_mem_gb"] = float(last_train.get("gpu_mem_gb", 0))

        # Get best eval from CSV (more reliable than summary for in-progress)
        if eval_rows:
            best_eval = min(eval_rows, key=lambda r: float(r.get("val_loss", 999)))
            entry["best_val_loss"] = float(best_eval["val_loss"])
            entry["best_val_step"] = int(best_eval["step"])
            if best_eval.get("attention_entropy"):
                entry["attention_entropy"] = float(best_eval["attention_entropy"])

        if name in BROKEN_RUNS:
            entry["status"] = "BROKEN (BUG-010)"

        results.append(entry)

    if not results:
        print("No results found!")
        return

    # Sort by val loss (best first)
    results.sort(key=lambda r: r["best_val_loss"] if r["best_val_loss"] else 999)

    # === Main comparison table ===
    print("=" * 90)
    print("PHASE 1a ABLATION RESULTS")
    print("=" * 90)
    print(f"{'Run':<25} {'Val Loss':>10} {'Train Loss':>11} {'Steps':>7} "
          f"{'Hours':>6} {'samp/s':>7} {'Entropy':>8} {'Status':<15}")
    print("-" * 90)

    baseline_val = None
    for r in results:
        if r["name"] == "baseline":
            baseline_val = r["best_val_loss"]

        val = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] else "N/A"
        train = f"{r['final_train_loss']:.4f}" if r["final_train_loss"] else "N/A"
        steps = f"{r['total_steps']}" if r["total_steps"] else "N/A"
        hours = f"{r['wall_hours']:.2f}" if r["wall_hours"] else "N/A"
        sps = f"{r['samp_per_sec']:.1f}" if r["samp_per_sec"] else "N/A"
        ent = f"{r['attention_entropy']:.3f}" if r["attention_entropy"] else "N/A"

        # Delta from baseline
        delta = ""
        if baseline_val and r["best_val_loss"] and r["name"] != "baseline":
            d = r["best_val_loss"] - baseline_val
            delta = f" ({d:+.4f})"

        print(f"{r['name']:<25} {val:>10}{delta:<10} {train:>11} {steps:>7} "
              f"{hours:>6} {sps:>7} {ent:>8} {r['status']:<15}")

    # === Key comparisons ===
    print("\n" + "=" * 90)
    print("KEY COMPARISONS")
    print("=" * 90)

    by_name = {r["name"]: r for r in results}

    # 2x2 Factorial: deep_query × coarse_only
    print("\n2×2 Factorial: deep_query × coarse_only")
    print("-" * 50)
    grid_names = {
        "deep+fine": "baseline",
        "deep+coarse": "A6_coarse_only",
        "shallow+fine": "A1_deep_query_off_v2",
        "shallow+coarse": "A7_shallow_coarse_only",
    }
    grid_vals = {}
    for label, name in grid_names.items():
        if name in by_name and by_name[name]["best_val_loss"]:
            grid_vals[label] = by_name[name]["best_val_loss"]
            print(f"  {label:<20} = {grid_vals[label]:.4f}  ({name})")
        else:
            print(f"  {label:<20} = PENDING  ({name})")

    if len(grid_vals) == 4:
        # Interaction effect
        deep_effect = grid_vals["deep+fine"] - grid_vals["shallow+fine"]
        coarse_effect = grid_vals["deep+coarse"] - grid_vals["shallow+coarse"]
        fine_effect_deep = grid_vals["deep+fine"] - grid_vals["deep+coarse"]
        fine_effect_shallow = grid_vals["shallow+fine"] - grid_vals["shallow+coarse"]
        interaction = (grid_vals["deep+fine"] - grid_vals["deep+coarse"]) - \
                      (grid_vals["shallow+fine"] - grid_vals["shallow+coarse"])

        print(f"\n  Deep query effect (at fine):    {deep_effect:+.4f}")
        print(f"  Deep query effect (at coarse):  {coarse_effect:+.4f}")
        print(f"  Fine pass effect (at deep):     {fine_effect_deep:+.4f}")
        print(f"  Fine pass effect (at shallow):  {fine_effect_shallow:+.4f}")
        print(f"  Interaction (deep×fine):         {interaction:+.4f}")
        if abs(interaction) > 0.02:
            print("  → SIGNIFICANT interaction: deep query + fine pass are synergistic")
        else:
            print("  → Weak interaction: effects are approximately additive")

    # LR sweep
    print("\nLR Sweep (connector:backbone ratio)")
    print("-" * 50)
    lr_names = [
        ("10:1 (baseline)", "baseline"),
        ("100:1", "LR2"),
        ("3:1", "LR3"),
        ("1:1", "LR4"),
    ]
    for label, name in lr_names:
        if name in by_name and by_name[name]["best_val_loss"]:
            val = by_name[name]["best_val_loss"]
            print(f"  {label:<25} = {val:.4f}")
        else:
            print(f"  {label:<25} = PENDING")

    # Freeze sweep
    print("\nFreeze Sweep")
    print("-" * 50)
    freeze_names = [
        ("Full finetune (baseline)", "baseline"),
        ("Freeze LLM", "F2_freeze_llm"),
        ("Freeze both", "F1_freeze_both"),
    ]
    for label, name in freeze_names:
        if name in by_name and by_name[name]["best_val_loss"]:
            val = by_name[name]["best_val_loss"]
            print(f"  {label:<25} = {val:.4f}")
        else:
            print(f"  {label:<25} = PENDING")

    # Static frames
    print("\nFrame Count (image handling)")
    print("-" * 50)
    frame_names = [
        ("Dynamic (baseline)", "baseline"),
        ("Static 1 frame", "A8_static_1frame"),
        ("Static 16 frames", "A8_static_16frames"),
    ]
    for label, name in frame_names:
        if name in by_name and by_name[name]["best_val_loss"]:
            val = by_name[name]["best_val_loss"]
            print(f"  {label:<25} = {val:.4f}")
        else:
            print(f"  {label:<25} = PENDING")

    # Data mix
    print("\nData Mix")
    print("-" * 50)
    data_names = [
        ("Balanced (baseline)", "baseline"),
        ("Video heavy", "D1_video_heavy"),
    ]
    for label, name in data_names:
        if name in by_name and by_name[name]["best_val_loss"]:
            val = by_name[name]["best_val_loss"]
            print(f"  {label:<25} = {val:.4f}")
        else:
            print(f"  {label:<25} = PENDING")

    # Summary recommendations
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    complete = [r for r in results if r["status"] == "complete"]
    pending = [r for r in results if r["status"] != "complete"]
    print(f"Complete: {len(complete)}/{len(results)} runs")
    if pending:
        print(f"Pending: {', '.join(r['name'] for r in pending)}")

    if complete:
        best = min(complete, key=lambda r: r["best_val_loss"])
        print(f"\nBest so far: {best['name']} (val_loss={best['best_val_loss']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="/workspace/checkpoints/ablations",
                        help="Base checkpoint directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    analyze(args.ckpt_dir, args.verbose)
