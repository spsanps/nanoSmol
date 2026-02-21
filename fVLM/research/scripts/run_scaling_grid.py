#!/usr/bin/env python
"""
Scaling grid runner for Phase 1b.

Generates configs from a template + matrix definition, launches runs
sequentially, and collects results into a single CSV for scaling law plots.

Usage:
    python release/scripts/run_scaling_grid.py --template release/configs/ablations/LR1_10to1.yaml
    python release/scripts/run_scaling_grid.py --template release/configs/ablations/LR1_10to1.yaml --dry-run
    python release/scripts/run_scaling_grid.py --template release/configs/ablations/LR1_10to1.yaml --filter "135M"

The template should be the winning config from Phase 1a ablations. The script
overrides model size, sample count, and architecture (foveated vs multi-token)
for each cell in the grid.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from release.utils.flop_counter import (
    LLM_CONFIGS,
    DINO_CONFIGS,
    estimate_flops_from_config,
    compute_samples_for_budget,
)


# ---- Scaling grid definition ----

LLM_SIZES = {
    "135M": "/workspace/models/SmolLM2-135M-Instruct",
    "360M": "/workspace/models/SmolLM2-360M-Instruct",
    "1.7B": "/workspace/models/SmolLM2-1.7B-Instruct",
}

FLOP_BUDGETS = {
    "C1": 1.6e16,
    "C2": 5.6e16,
    "C3": 1.6e17,
    "C4": 3.1e17,
}

ARCHITECTURES = ["foveated", "multi_token"]


def generate_run_configs(template_cfg: dict) -> list:
    """Generate all scaling grid configs from a template."""
    runs = []

    for size_name, llm_path in LLM_SIZES.items():
        for budget_name, flop_budget in FLOP_BUDGETS.items():
            for arch in ARCHITECTURES:
                cfg = deepcopy(template_cfg)

                # Set model
                cfg["model"]["llm"] = llm_path
                is_multi = arch == "multi_token"
                cfg["model"]["multi_token"] = is_multi
                if is_multi:
                    cfg["model"]["tokens_per_frame"] = 16
                else:
                    cfg["model"].pop("tokens_per_frame", None)
                    cfg["model"].pop("multi_token", None)

                # Adjust batch/memory settings per model size
                if "1.7B" in size_name:
                    cfg["model"]["gradient_checkpointing"] = True
                    cfg["training"]["batch_size"] = 2
                    cfg["training"]["grad_accum"] = 16
                    cfg["training"]["compile"] = True  # worth it for large models
                elif "360M" in size_name:
                    cfg["training"]["batch_size"] = 4
                    cfg["training"]["grad_accum"] = 8
                    cfg["training"]["compile"] = True  # worth it for 360M+

                # Compute sample count for this FLOP budget
                num_samples = compute_samples_for_budget(flop_budget, cfg)
                cfg["training"]["total_samples"] = num_samples

                # Run ID
                arch_tag = "B" if is_multi else "F"
                run_id = f"{size_name}-{budget_name}-{arch_tag}"
                cfg["wandb"] = cfg.get("wandb", {})
                cfg["wandb"]["project"] = "foveated-vlm-scaling"
                cfg["wandb"]["run_name"] = run_id

                # Checkpoint dir
                cfg["checkpoint"]["save_dir"] = f"/workspace/checkpoints/scaling/{run_id}"

                # Metadata
                run_info = {
                    "run_id": run_id,
                    "size": size_name,
                    "budget": budget_name,
                    "arch": arch,
                    "flop_budget": flop_budget,
                    "total_samples": num_samples,
                    "config": cfg,
                }
                runs.append(run_info)

    return runs


def run_training(cfg: dict, run_id: str, output_dir: str, dry_run: bool = False) -> dict:
    """Run a single training job and return results."""
    # Write config to temp file
    config_dir = Path(output_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    if dry_run:
        flops = estimate_flops_from_config(cfg)
        print(f"  [DRY RUN] {run_id}: {cfg['training']['total_samples']:,} samples, "
              f"{flops:.2e} FLOPs/sample, config at {config_path}")
        return {"run_id": run_id, "status": "dry_run"}

    # Skip if already completed (run_summary exists with valid data)
    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    summary_files = sorted(ckpt_dir.glob("run_summary_*.json"))
    if summary_files:
        with open(summary_files[-1]) as f:
            prev_summary = json.load(f)
        if prev_summary.get("best_val_loss") is not None:
            print(f"\n  [SKIP] {run_id} already completed "
                  f"(val_loss={prev_summary['best_val_loss']:.4f})")
            return {
                "run_id": run_id,
                "status": "success",
                "returncode": 0,
                "wall_time_sec": prev_summary.get("wall_time_sec", 0),
                "final_train_loss": prev_summary.get("final_train_loss"),
                "best_val_loss": prev_summary.get("best_val_loss"),
                "best_val_step": prev_summary.get("best_val_step"),
                "total_samples": cfg["training"]["total_samples"],
            }

    # Run training
    cmd = [
        sys.executable, "-u", "release/train.py",
        "--config", str(config_path),
    ]

    print(f"\n{'='*60}")
    print(f"Starting: {run_id}")
    print(f"  Samples: {cfg['training']['total_samples']:,}")
    print(f"  LLM: {cfg['model']['llm']}")
    print(f"  Arch: {'multi_token' if cfg['model'].get('multi_token') else 'foveated'}")
    print(f"  Batch: {cfg['training']['batch_size']} × {cfg['training']['grad_accum']}")
    print(f"{'='*60}")

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        cmd, capture_output=False, text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        env=env,
    )
    elapsed = time.time() - t0

    # Collect results from run summary JSON
    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    summary_files = sorted(ckpt_dir.glob("run_summary_*.json"))
    summary = {}
    if summary_files:
        with open(summary_files[-1]) as f:
            summary = json.load(f)

    return {
        "run_id": run_id,
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_time_sec": elapsed,
        "final_train_loss": summary.get("final_train_loss", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "best_val_step": summary.get("best_val_step", None),
        "total_samples": cfg["training"]["total_samples"],
    }


def main():
    parser = argparse.ArgumentParser(description="Scaling grid runner")
    parser.add_argument("--template", required=True, help="Template YAML config (winning ablation)")
    parser.add_argument("--output-dir", default="/workspace/checkpoints/scaling",
                        help="Output directory for configs and results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--filter", default=None,
                        help="Only run configs matching this substring (e.g. '135M', 'C2')")
    parser.add_argument("--arch", default=None, choices=["foveated", "multi_token"],
                        help="Only run this architecture (default: both)")
    args = parser.parse_args()

    # Load template
    with open(args.template) as f:
        template = yaml.safe_load(f)

    # Generate grid
    all_runs = generate_run_configs(template)

    # Filter by run_id substring
    if args.filter:
        all_runs = [r for r in all_runs if args.filter in r["run_id"]]

    # Filter by architecture
    if args.arch:
        all_runs = [r for r in all_runs if r["arch"] == args.arch]

    print(f"Scaling grid: {len(all_runs)} runs")
    print(f"Template: {args.template}")
    print()

    # Print matrix
    print(f"{'Run ID':<20} {'Size':<8} {'Budget':<8} {'Arch':<12} {'Samples':>12} {'FLOPs/sample':>14}")
    print("-" * 80)
    for r in all_runs:
        flops = estimate_flops_from_config(r["config"])
        print(f"{r['run_id']:<20} {r['size']:<8} {r['budget']:<8} {r['arch']:<12} "
              f"{r['total_samples']:>12,} {flops:>14.2e}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would run the above configs. Exiting.")
        return

    # Run sequentially, collect results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results_csv = output_dir / "scaling_grid_results.csv"

    for i, run_info in enumerate(all_runs):
        print(f"\n[{i+1}/{len(all_runs)}] {run_info['run_id']}")
        result = run_training(
            run_info["config"], run_info["run_id"],
            str(output_dir), dry_run=args.dry_run,
        )
        result.update({
            "size": run_info["size"],
            "budget": run_info["budget"],
            "arch": run_info["arch"],
            "flop_budget": run_info["flop_budget"],
        })
        results.append(result)

        # Append to CSV after each run (for incremental monitoring)
        # Skip if this run_id is already in the CSV (from a previous invocation)
        fieldnames = [
            "run_id", "size", "budget", "arch", "flop_budget",
            "total_samples", "status", "returncode", "wall_time_sec",
            "final_train_loss", "best_val_loss", "best_val_step",
        ]
        existing_ids = set()
        if results_csv.exists():
            with open(results_csv, newline="") as f:
                for row in csv.DictReader(f):
                    existing_ids.add(row.get("run_id"))
        write_header = not results_csv.exists() or os.path.getsize(results_csv) == 0
        if result["run_id"] not in existing_ids:
            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(result)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Scaling grid complete: {len(results)} runs")
    print(f"Results saved to: {results_csv}")
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    print(f"  Success: {successful}, Failed: {failed}")
    total_time = sum(r.get("wall_time_sec", 0) for r in results)
    print(f"  Total wall time: {total_time/3600:.1f}h")


if __name__ == "__main__":
    main()
