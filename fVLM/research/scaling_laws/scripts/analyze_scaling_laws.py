#!/usr/bin/env python3
"""
Analyze scaling laws from experiment data.

Generates plots and derives empirical scaling relationships.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

RESEARCH_DIR = Path(__file__).parent.parent
DATA_DIR = RESEARCH_DIR / "data"
PLOTS_DIR = RESEARCH_DIR / "plots"
RESULTS_DIR = RESEARCH_DIR / "results"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'foveated_opt': '#2ecc71',      # Green
    'baseline': '#3498db',           # Blue
    'foveated_orig': '#e74c3c',     # Red
}
LABELS = {
    'foveated_opt': 'Foveated (224px, 1 fine)',
    'baseline': 'Baseline (16 tok/frame)',
    'foveated_orig': 'Foveated (256px, 2 fine)',
}
MARKERS = {
    'foveated_opt': 'o',
    'baseline': 's',
    'foveated_orig': '^',
}


def load_data():
    """Load scaling data from CSV."""
    # Try different file names
    for name in ["scaling_data_S-S.csv", "scaling_data.csv"]:
        csv_path = DATA_DIR / name
        if csv_path.exists():
            return pd.read_csv(csv_path)

    # Fallback to JSON
    json_path = DATA_DIR / "scaling_data.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    raise FileNotFoundError("No scaling data found. Run evaluate_all_checkpoints.py first.")


def power_law(x, a, b):
    """Power law: L = a * x^(-b)"""
    return a * np.power(x, -b)


def log_linear(x, a, b):
    """Log-linear: L = a - b * log(x)"""
    return a - b * np.log(x)


def fit_scaling_law(df, x_col, y_col, model_name):
    """Fit scaling law to data for a specific model."""
    subset = df[df['experiment'] == model_name].copy()
    if len(subset) < 3:
        return None

    x = subset[x_col].values
    y = subset[y_col].values

    # Try power law fit
    try:
        popt, pcov = curve_fit(power_law, x, y, p0=[10, 0.1], maxfev=10000)
        y_pred = power_law(x, *popt)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        return {
            'type': 'power_law',
            'params': {'a': popt[0], 'b': popt[1]},
            'r2': r2,
            'formula': f"L = {popt[0]:.4f} * x^(-{popt[1]:.4f})",
        }
    except Exception as e:
        print(f"  Power law fit failed for {model_name}: {e}")

    # Fallback to log-linear
    try:
        slope, intercept, r, p, se = linregress(np.log(x), y)
        return {
            'type': 'log_linear',
            'params': {'a': intercept, 'b': -slope},
            'r2': r**2,
            'formula': f"L = {intercept:.4f} - {-slope:.4f} * log(x)",
        }
    except Exception as e:
        print(f"  Log-linear fit failed for {model_name}: {e}")

    return None


def plot_loss_vs_steps(df):
    """Plot validation loss vs training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('step')
        ax.errorbar(
            subset['step'], subset['val_loss'],
            yerr=subset['val_loss_se'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=8, capsize=3
        )

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Validation Loss (CE)', fontsize=12)
    ax.set_title('Validation Loss vs Training Steps', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'loss_vs_steps.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'loss_vs_steps.png'}")


def plot_loss_vs_flops(df):
    """Plot validation loss vs training FLOPs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('total_training_flops')
        x = subset['total_training_flops'] / 1e15  # Convert to PFLOPs
        ax.errorbar(
            x, subset['val_loss'],
            yerr=subset['val_loss_se'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=8, capsize=3
        )

    ax.set_xlabel('Training FLOPs (PFLOPs)', fontsize=12)
    ax.set_ylabel('Validation Loss (CE)', fontsize=12)
    ax.set_title('Validation Loss vs Training Compute', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'loss_vs_flops.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'loss_vs_flops.png'}")


def plot_ppl_vs_flops(df):
    """Plot perplexity vs training FLOPs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('total_training_flops')
        x = subset['total_training_flops'] / 1e15
        ax.plot(
            x, subset['perplexity'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=8
        )

    ax.set_xlabel('Training FLOPs (PFLOPs)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Perplexity vs Training Compute', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ppl_vs_flops.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'ppl_vs_flops.png'}")


def plot_efficiency_ratio(df):
    """Plot quality per FLOP over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get blind baseline loss (approximate)
    blind_loss = 6.05  # From earlier experiments

    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('step')
        # Quality = blind_loss - model_loss
        quality = blind_loss - subset['val_loss']
        # Efficiency = quality per inference FLOP (in GFLOPs)
        efficiency = quality / (subset['flops_per_sample'] / 1e9)

        ax.plot(
            subset['step'], efficiency,
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=8
        )

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Visual Contribution per GFLOPs', fontsize=12)
    ax.set_title('Inference Efficiency Over Training', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'efficiency_ratio.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'efficiency_ratio.png'}")


def plot_iso_loss_curves(df):
    """Plot iso-loss curves (compute required for same quality)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define target loss levels
    all_losses = df['val_loss'].values
    target_losses = np.linspace(all_losses.min() * 1.05, all_losses.max() * 0.95, 5)

    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('val_loss')

        # Interpolate to find FLOPs for each target loss
        flops_at_target = []
        valid_targets = []

        for target in target_losses:
            if target >= subset['val_loss'].min() and target <= subset['val_loss'].max():
                # Linear interpolation
                flops = np.interp(target, subset['val_loss'][::-1], subset['total_training_flops'][::-1])
                flops_at_target.append(flops / 1e15)
                valid_targets.append(target)

        if len(valid_targets) > 1:
            ax.plot(
                valid_targets, flops_at_target,
                marker=MARKERS.get(exp_name, 'o'),
                color=COLORS.get(exp_name, 'gray'),
                label=LABELS.get(exp_name, exp_name),
                linewidth=2, markersize=8
            )

    ax.set_xlabel('Target Validation Loss', fontsize=12)
    ax.set_ylabel('Training FLOPs Required (PFLOPs)', fontsize=12)
    ax.set_title('Iso-Loss Curves: Compute Required for Same Quality', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.invert_xaxis()  # Lower loss is better, so invert

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'iso_loss_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'iso_loss_curves.png'}")


def plot_token_efficiency(df):
    """Plot loss vs total visual tokens processed during training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('step')

        # Total visual tokens = steps * batch_size * frames * tokens_per_frame
        batch_size = 16
        num_frames = 8
        tokens_per_frame = subset['visual_tokens_per_frame'].iloc[0]
        total_tokens = subset['step'] * batch_size * num_frames * tokens_per_frame / 1e6  # Millions

        ax.errorbar(
            total_tokens, subset['val_loss'],
            yerr=subset['val_loss_se'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=8, capsize=3
        )

    ax.set_xlabel('Total Visual Tokens Processed (Millions)', fontsize=12)
    ax.set_ylabel('Validation Loss (CE)', fontsize=12)
    ax.set_title('Validation Loss vs Visual Tokens Processed', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'loss_vs_tokens.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'loss_vs_tokens.png'}")


def plot_combined_summary(df):
    """Create a 2x2 summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Loss vs Steps
    ax = axes[0, 0]
    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('step')
        ax.errorbar(
            subset['step'], subset['val_loss'],
            yerr=subset['val_loss_se'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=6, capsize=2
        )
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('(a) Loss vs Training Steps')
    ax.legend(fontsize=8)
    ax.set_xscale('log')

    # 2. Loss vs FLOPs
    ax = axes[0, 1]
    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('total_training_flops')
        x = subset['total_training_flops'] / 1e15
        ax.errorbar(
            x, subset['val_loss'],
            yerr=subset['val_loss_se'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=6, capsize=2
        )
    ax.set_xlabel('Training FLOPs (PFLOPs)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('(b) Loss vs Training Compute')
    ax.legend(fontsize=8)
    ax.set_xscale('log')

    # 3. Perplexity vs FLOPs
    ax = axes[1, 0]
    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('total_training_flops')
        x = subset['total_training_flops'] / 1e15
        ax.plot(
            x, subset['perplexity'],
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=6
        )
    ax.set_xlabel('Training FLOPs (PFLOPs)')
    ax.set_ylabel('Perplexity')
    ax.set_title('(c) Perplexity vs Training Compute')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 4. Efficiency
    ax = axes[1, 1]
    blind_loss = 6.05
    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name].sort_values('step')
        quality = blind_loss - subset['val_loss']
        efficiency = quality / (subset['flops_per_sample'] / 1e9)
        ax.plot(
            subset['step'], efficiency,
            marker=MARKERS.get(exp_name, 'o'),
            color=COLORS.get(exp_name, 'gray'),
            label=LABELS.get(exp_name, exp_name),
            linewidth=2, markersize=6
        )
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Visual Contribution / GFLOPs')
    ax.set_title('(d) Inference Efficiency')
    ax.legend(fontsize=8)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'scaling_summary.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'scaling_summary.png'}")


def generate_analysis_report(df, scaling_laws):
    """Generate markdown analysis report."""
    report = []
    report.append("# Scaling Law Analysis Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("\n## Data Summary\n")
    report.append(f"- Total data points: {len(df)}\n")
    report.append(f"- Experiments: {', '.join(df['experiment'].unique())}\n")
    report.append(f"- Step range: {df['step'].min()} - {df['step'].max()}\n")

    report.append("\n## Results by Experiment\n")
    for exp_name in df['experiment'].unique():
        subset = df[df['experiment'] == exp_name]
        report.append(f"\n### {LABELS.get(exp_name, exp_name)}\n")
        report.append(f"- Visual tokens/frame: {subset['visual_tokens_per_frame'].iloc[0]}\n")
        report.append(f"- Frame size: {subset['frame_size'].iloc[0]}\n")
        report.append(f"- FLOPs/sample: {subset['flops_per_sample'].iloc[0]/1e9:.1f} GFLOPs\n")
        report.append("\n| Step | Loss | SE | PPL | Training FLOPs |\n")
        report.append("|------|------|-----|-----|----------------|\n")
        for _, row in subset.sort_values('step').iterrows():
            report.append(f"| {row['step']} | {row['val_loss']:.4f} | {row['val_loss_se']:.4f} | {row['perplexity']:.1f} | {row['total_training_flops']/1e15:.3f} PF |\n")

    report.append("\n## Scaling Laws\n")
    for exp_name, law in scaling_laws.items():
        if law:
            report.append(f"\n### {LABELS.get(exp_name, exp_name)}\n")
            report.append(f"- Type: {law['type']}\n")
            report.append(f"- Formula: {law['formula']}\n")
            report.append(f"- R²: {law['r2']:.4f}\n")

    report.append("\n## Key Findings\n")

    # Compare at highest step count
    max_step = df['step'].max()
    at_max = df[df['step'] == max_step]

    if len(at_max) > 1:
        report.append(f"\n### At {max_step} steps:\n")
        report.append("| Model | Loss | PPL | Training FLOPs |\n")
        report.append("|-------|------|-----|----------------|\n")
        for _, row in at_max.sort_values('val_loss').iterrows():
            report.append(f"| {LABELS.get(row['experiment'], row['experiment'])} | {row['val_loss']:.4f} | {row['perplexity']:.1f} | {row['total_training_flops']/1e15:.3f} PF |\n")

        # Find best model
        best = at_max.loc[at_max['val_loss'].idxmin()]
        report.append(f"\n**Best model at {max_step} steps:** {LABELS.get(best['experiment'], best['experiment'])}\n")
        report.append(f"- Loss: {best['val_loss']:.4f}\n")
        report.append(f"- PPL: {best['perplexity']:.1f}\n")

    report.append("\n## Plots Generated\n")
    for plot_file in sorted(PLOTS_DIR.glob("*.png")):
        report.append(f"- [{plot_file.name}](../plots/{plot_file.name})\n")

    # Write report
    report_path = RESULTS_DIR / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.writelines(report)
    print(f"Saved: {report_path}")


def main():
    print("=" * 80)
    print("SCALING LAW ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} data points")
    print(f"Experiments: {df['experiment'].unique()}")
    print(f"Steps: {sorted(df['step'].unique())}")

    # Fit scaling laws
    print("\nFitting scaling laws...")
    scaling_laws = {}
    for exp_name in df['experiment'].unique():
        print(f"  {exp_name}...")
        law = fit_scaling_law(df, 'total_training_flops', 'val_loss', exp_name)
        scaling_laws[exp_name] = law
        if law:
            print(f"    {law['formula']} (R²={law['r2']:.4f})")

    # Save scaling laws
    laws_path = RESULTS_DIR / "scaling_laws.json"
    with open(laws_path, 'w') as f:
        json.dump(scaling_laws, f, indent=2)
    print(f"\nSaved: {laws_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_loss_vs_steps(df)
    plot_loss_vs_flops(df)
    plot_ppl_vs_flops(df)
    plot_efficiency_ratio(df)
    plot_iso_loss_curves(df)
    plot_token_efficiency(df)
    plot_combined_summary(df)

    # Generate report
    print("\nGenerating analysis report...")
    generate_analysis_report(df, scaling_laws)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
