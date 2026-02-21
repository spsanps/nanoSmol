#!/usr/bin/env python
"""
Scaling Law Analysis for fVLM.

Implements the Chinchilla parametric form (Hoffmann et al., 2022):

    L(N, D) = E + A / N^α + B / D^β

where:
    L = validation loss
    N = trainable parameter count
    D = training samples (analogous to tokens for VLMs)
    E = irreducible loss (entropy floor of the data)
    A, B = scaling coefficients
    α, β = scaling exponents

Additionally fits iso-FLOP curves:
    For each compute budget C, find N* that minimizes L(N, D(C,N))
    where D = C / (flops_per_sample(N)) and C ≈ 6*N*D for transformers.

Reference constants from Chinchilla (text LLMs):
    E=1.69, A=406.4, B=410.7, α=0.34, β=0.28

Our constants will differ (video VLM, different architecture + data).

Usage:
    python release/scripts/scaling_law_analysis.py
    python release/scripts/scaling_law_analysis.py --results-csv /path/to/results.csv
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Model parameter counts (trainable params, not total)
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "135M": 135_000_000,
    "360M": 360_000_000,
    "1.7B": 1_700_000_000,
}

# DINO-small params (always present, trainable)
DINO_PARAMS = 22_000_000
CONNECTOR_PARAMS = 600_000  # rough: dino_to_llm + llm_to_query + q_static + q_init

def total_trainable_params(size_name: str) -> int:
    """Total trainable params: LLM + DINO + connector."""
    return MODEL_PARAMS[size_name] + DINO_PARAMS + CONNECTOR_PARAMS


# ---------------------------------------------------------------------------
# Chinchilla parametric loss function
# ---------------------------------------------------------------------------

def chinchilla_loss(N: float, D: float, E: float, A: float, B: float,
                    alpha: float, beta: float) -> float:
    """L(N, D) = E + A/N^α + B/D^β"""
    return E + A / (N ** alpha) + B / (D ** beta)


def fit_chinchilla(data_points: list[dict], verbose: bool = True) -> dict:
    """
    Fit L(N,D) = E + A/N^α + B/D^β to observed data.

    data_points: list of dicts with keys: N, D, loss
    Returns: dict with E, A, B, alpha, beta, residuals
    """
    from scipy.optimize import minimize

    if len(data_points) < 3:
        print(f"WARNING: Only {len(data_points)} data points. Need 5+ for reliable fit.")
        print("Showing preliminary estimates with constrained fit.\n")

    N_arr = np.array([d["N"] for d in data_points], dtype=np.float64)
    D_arr = np.array([d["D"] for d in data_points], dtype=np.float64)
    L_arr = np.array([d["loss"] for d in data_points], dtype=np.float64)

    def objective(params):
        E, log_A, log_B, alpha, beta = params
        A, B = np.exp(log_A), np.exp(log_B)
        pred = E + A / (N_arr ** alpha) + B / (D_arr ** beta)
        # Huber loss for robustness (Chinchilla replication uses this)
        residuals = pred - L_arr
        delta = 0.01
        loss = np.where(
            np.abs(residuals) < delta,
            0.5 * residuals ** 2,
            delta * (np.abs(residuals) - 0.5 * delta)
        )
        return loss.sum()

    # Initial guess (Chinchilla-like but adapted for our scale)
    x0 = [
        0.8,       # E: irreducible loss (our data ~ 1.3, so floor ~ 0.8)
        np.log(50),  # log(A): smaller than Chinchilla (406) since our N is smaller
        np.log(50),  # log(B)
        0.34,      # alpha (Chinchilla)
        0.28,      # beta (Chinchilla)
    ]

    bounds = [
        (0.1, 2.0),      # E
        (np.log(1), np.log(1e6)),  # log(A)
        (np.log(1), np.log(1e6)),  # log(B)
        (0.05, 1.0),     # alpha
        (0.05, 1.0),     # beta
    ]

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    E, log_A, log_B, alpha, beta = result.x
    A, B = np.exp(log_A), np.exp(log_B)

    # Compute residuals
    pred = np.array([chinchilla_loss(N, D, E, A, B, alpha, beta)
                     for N, D in zip(N_arr, D_arr)])
    residuals = pred - L_arr

    fit = {
        "E": E, "A": A, "B": B, "alpha": alpha, "beta": beta,
        "rmse": np.sqrt(np.mean(residuals ** 2)),
        "max_error": np.max(np.abs(residuals)),
        "n_points": len(data_points),
    }

    if verbose:
        print("=" * 60)
        print("Chinchilla Parametric Fit: L(N,D) = E + A/N^α + B/D^β")
        print("=" * 60)
        print(f"  E (irreducible loss) = {E:.4f}")
        print(f"  A (param coeff)      = {A:.4f}")
        print(f"  B (data coeff)       = {B:.4f}")
        print(f"  α (param exponent)   = {alpha:.4f}")
        print(f"  β (data exponent)    = {beta:.4f}")
        print(f"  RMSE                 = {fit['rmse']:.6f}")
        print(f"  Max error            = {fit['max_error']:.6f}")
        print(f"  Data points          = {fit['n_points']}")
        print()
        print(f"  {'Point':<20} {'N':>12} {'D':>12} {'L_actual':>10} {'L_pred':>10} {'Error':>10}")
        print(f"  {'-'*74}")
        for dp, p, r in zip(data_points, pred, residuals):
            print(f"  {dp.get('name',''):<20} {dp['N']:>12,.0f} {dp['D']:>12,.0f} "
                  f"{dp['loss']:>10.6f} {p:>10.6f} {r:>+10.6f}")
        print()

        # Chinchilla comparison
        print("  Chinchilla reference (text LLMs):")
        print(f"    E=1.69, A=406.4, B=410.7, α=0.34, β=0.28")
        print()

    return fit


# ---------------------------------------------------------------------------
# Iso-FLOP analysis (Chinchilla Approach 1)
# ---------------------------------------------------------------------------

def fit_isoflop_curves(data_points: list[dict], verbose: bool = True) -> dict:
    """
    For each FLOP budget, fit a parabola in log(N) to find optimal model size.

    Groups data by flop_budget, fits: L = a * (log N)^2 + b * log N + c
    The minimum is at log N* = -b / (2a), giving N* = exp(-b/2a).
    """
    from collections import defaultdict

    by_budget = defaultdict(list)
    for dp in data_points:
        by_budget[dp["flop_budget"]].append(dp)

    results = {}
    if verbose:
        print("=" * 60)
        print("Iso-FLOP Analysis (parabola fit in log N)")
        print("=" * 60)

    for budget_name, points in sorted(by_budget.items()):
        if len(points) < 2:
            if verbose:
                print(f"\n  {budget_name}: only {len(points)} point(s), need 2+ for parabola")
            continue

        log_N = np.array([np.log(p["N"]) for p in points])
        losses = np.array([p["loss"] for p in points])

        if len(points) >= 3:
            # Full parabola fit
            coeffs = np.polyfit(log_N, losses, 2)
            a, b, c = coeffs
        else:
            # 2 points: linear interpolation, can estimate direction
            coeffs = np.polyfit(log_N, losses, 1)
            b, c = coeffs
            a = 0  # can't fit curvature with 2 points

        if a > 0:
            # Parabola opens up — minimum exists
            log_N_opt = -b / (2 * a)
            N_opt = np.exp(log_N_opt)
            L_opt = a * log_N_opt ** 2 + b * log_N_opt + c
        else:
            N_opt = None
            L_opt = None

        results[budget_name] = {
            "coeffs": coeffs.tolist() if hasattr(coeffs, 'tolist') else list(coeffs),
            "N_opt": N_opt,
            "L_opt": L_opt,
            "points": [(p["N"], p["loss"], p.get("name", "")) for p in points],
        }

        if verbose:
            flops = points[0].get("flops", "?")
            print(f"\n  Budget {budget_name} (FLOPs={flops}):")
            for p in sorted(points, key=lambda x: x["N"]):
                print(f"    N={p['N']:>12,.0f} ({p.get('name',''):>8}) → loss={p['loss']:.6f}")
            if N_opt:
                print(f"    → Optimal N* = {N_opt:,.0f} (loss* = {L_opt:.6f})")
            elif len(points) == 2:
                # With 2 points, report direction
                if losses[0] < losses[1] and log_N[0] < log_N[1]:
                    print(f"    → Smaller model wins at this budget (data-constrained)")
                else:
                    print(f"    → Larger model wins at this budget (param-constrained)")

    return results


# ---------------------------------------------------------------------------
# Compute-optimal predictions
# ---------------------------------------------------------------------------

def predict_optimal_allocation(fit: dict, C_flops: float,
                               flops_per_sample_fn=None) -> dict:
    """
    Given fitted constants and a FLOP budget, find optimal (N*, D*).

    Uses: C = flops_per_sample(N) * D, minimize L(N, D(C,N))
    """
    E, A, B, alpha, beta = fit["E"], fit["A"], fit["B"], fit["alpha"], fit["beta"]

    if flops_per_sample_fn is None:
        # Approximate: FLOPs ≈ 6*N per sample (standard transformer approx)
        flops_per_sample_fn = lambda N: 6 * N

    from scipy.optimize import minimize_scalar

    def loss_at_N(log_N):
        N = np.exp(log_N)
        fps = flops_per_sample_fn(N)
        D = C_flops / fps
        if D < 1:
            return 1e10
        return chinchilla_loss(N, D, E, A, B, alpha, beta)

    # Search over reasonable N range
    result = minimize_scalar(loss_at_N, bounds=(np.log(1e6), np.log(1e11)),
                            method="bounded")
    N_opt = np.exp(result.x)
    fps = flops_per_sample_fn(N_opt)
    D_opt = C_flops / fps
    L_opt = chinchilla_loss(N_opt, D_opt, E, A, B, alpha, beta)

    return {
        "N_opt": N_opt,
        "D_opt": D_opt,
        "L_opt": L_opt,
        "D_over_N": D_opt / N_opt,
    }


# ---------------------------------------------------------------------------
# Load results from scaling grid CSV
# ---------------------------------------------------------------------------

def load_scaling_results(csv_path: str) -> list[dict]:
    """Load completed scaling grid results into data points for fitting."""
    data_points = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["best_val_loss"] in (None, "", "None"):
                continue
            size = row["size"]
            N = total_trainable_params(size)
            D = int(row["total_samples"])
            loss = float(row["best_val_loss"])
            flops = float(row["flop_budget"])

            data_points.append({
                "name": row["run_id"],
                "N": N,
                "D": D,
                "loss": loss,
                "flops": flops,
                "flop_budget": row["budget"],
                "size": size,
            })
    return data_points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scaling law analysis for fVLM")
    parser.add_argument("--results-csv",
                        default="/workspace/checkpoints/scaling/scaling_grid_results.csv",
                        help="Path to scaling grid results CSV")
    parser.add_argument("--include-running", action="store_true",
                        help="Include in-progress runs (uses best val loss so far)")
    args = parser.parse_args()

    print("=" * 70)
    print("fVLM Scaling Law Analysis")
    print("=" * 70)
    print(f"\nParametric form: L(N,D) = E + A/N^α + B/D^β")
    print(f"Reference: Hoffmann et al. (2022) 'Training Compute-Optimal LLMs'")
    print()

    # Load data
    data_points = load_scaling_results(args.results_csv)

    # Optionally add in-progress runs
    if args.include_running:
        import glob
        for run_dir in sorted(glob.glob("/workspace/checkpoints/scaling/*/metrics_*.csv")):
            run_name = Path(run_dir).parent.name
            if any(d["name"] == run_name for d in data_points):
                continue
            # Get best val loss
            best_val = float("inf")
            with open(run_dir) as f:
                for row in csv.DictReader(f):
                    if row["event_type"] == "eval" and row["val_loss"]:
                        val = float(row["val_loss"])
                        if val < best_val:
                            best_val = val
            if best_val < float("inf"):
                # Parse size from run name (e.g., "135M-C3-F")
                parts = run_name.split("-")
                size = parts[0]
                budget = parts[1]
                if size in MODEL_PARAMS:
                    N = total_trainable_params(size)
                    # Get config for sample count
                    cfg_path = f"/workspace/checkpoints/scaling/configs/{run_name}.yaml"
                    if os.path.exists(cfg_path):
                        import yaml
                        with open(cfg_path) as f:
                            cfg = yaml.safe_load(f)
                        D = cfg["training"]["total_samples"]
                        data_points.append({
                            "name": f"{run_name} (running)",
                            "N": N, "D": D, "loss": best_val,
                            "flops": 0, "flop_budget": budget, "size": size,
                        })

    print(f"Data points loaded: {len(data_points)}")
    for dp in data_points:
        print(f"  {dp['name']:<25} N={dp['N']:>12,.0f}  D={dp['D']:>10,.0f}  "
              f"loss={dp['loss']:.6f}  budget={dp['flop_budget']}")
    print()

    if len(data_points) < 2:
        print("Need at least 2 data points. Waiting for more runs to complete.")
        return

    # --- Iso-FLOP analysis ---
    isoflop = fit_isoflop_curves(data_points)
    print()

    # --- Parametric fit ---
    if len(data_points) >= 3:
        fit = fit_chinchilla(data_points)

        # --- Predictions ---
        print("=" * 60)
        print("Compute-Optimal Predictions")
        print("=" * 60)
        for label, C in [("C3", 1.6e17), ("C4", 3.1e17),
                         ("C5 (hypothetical)", 1e18),
                         ("A100 budget (est)", 1e19)]:
            pred = predict_optimal_allocation(fit, C)
            print(f"  {label}: N*={pred['N_opt']:,.0f} params, "
                  f"D*={pred['D_opt']:,.0f} samples, "
                  f"L*={pred['L_opt']:.4f}, D/N={pred['D_over_N']:.1f}")
    else:
        print("Need 3+ points for parametric fit. Current observations:")
        # Simple power-law extrapolation with 2 points
        if len(data_points) >= 2:
            pts = sorted(data_points, key=lambda x: x["D"])
            D0, L0 = pts[0]["D"], pts[0]["loss"]
            D1, L1 = pts[-1]["D"], pts[-1]["loss"]
            if D1 > D0 and L1 < L0:
                # L ≈ a * D^(-b) → b = log(L0/L1) / log(D1/D0)
                b = np.log(L0 / L1) / np.log(D1 / D0)
                a = L0 * (D0 ** b)
                print(f"\n  Simple power law (2-point): L ≈ {a:.2f} * D^(-{b:.4f})")
                print(f"  Extrapolation to 1M samples: L ≈ {a * (1e6 ** (-b)):.4f}")
                print(f"  Extrapolation to 2M samples: L ≈ {a * (2e6 ** (-b)):.4f}")


if __name__ == "__main__":
    main()
