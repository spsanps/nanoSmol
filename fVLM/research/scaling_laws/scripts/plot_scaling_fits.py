#!/usr/bin/env python3
"""
Scaling Law Analysis: Curve Fitting & Extrapolation

Fits Chinchilla-style scaling laws to experimental data and extrapolates
to SmolVLM-scale compute budgets.

Scaling law form: L(C) = A * C^(-alpha) + L_inf
Where C = compute (FLOPs), L = loss, L_inf = irreducible loss
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

RESULTS_PATH = "/mnt/d/projects/fVLM/outputs/scaling_comprehensive/results.json"
OUTPUT_DIR = Path("/mnt/d/projects/fVLM/outputs/scaling_comprehensive/scaling_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SmolVLM reference points for extrapolation
SMOLVLM_CONFIGS = {
    'SmolVLM-256M': {
        'params_M': 256, 'llm': '135M', 'vision': 'SigLIP-SO400M',
        'tokens_per_image': 81, 'train_samples': 4_400_000,
        # Approximate FLOPs: 6*N*seq_len*samples, seq_len~81+64=145
        'approx_flops': 6 * 256e6 * 145 * 4_400_000,  # ~9.8e17
    },
    'SmolVLM-500M': {
        'params_M': 500, 'llm': '360M', 'vision': 'SigLIP-SO400M',
        'tokens_per_image': 81, 'train_samples': 4_400_000,
        'approx_flops': 6 * 500e6 * 145 * 4_400_000,  # ~1.9e18
    },
    'SmolVLM-2.2B': {
        'params_M': 2200, 'llm': '1.7B', 'vision': 'SigLIP-SO400M',
        'tokens_per_image': 81, 'train_samples': 4_400_000,
        'approx_flops': 6 * 2200e6 * 145 * 4_400_000,  # ~8.4e18
    },
}

# Our model configs for extrapolation
OUR_CONFIGS = {
    'Fov-S-S': {'params_M': 159, 'flops_per_sample_8f': 4.87e11, 'flops_per_sample_64f': 1.68e12},
    'Fov-M-S': {'params_M': 386, 'flops_per_sample_8f': 1.08e12, 'flops_per_sample_64f': 2.73e12},
    'Fov-S-B': {'params_M': 224, 'flops_per_sample_8f': 8.73e11, 'flops_per_sample_64f': None},
    'Bas-S-S': {'params_M': 172, 'flops_per_sample_8f': 6.01e11, 'flops_per_sample_64f': 3.70e12},
    'Bas-M-S': {'params_M': 411, 'flops_per_sample_8f': 1.38e12, 'flops_per_sample_64f': 8.11e12},
    'Bas-S-B': {'params_M': 251, 'flops_per_sample_8f': 9.87e11, 'flops_per_sample_64f': None},
}

# Style
COLORS = {
    'fov': '#2196F3',   # blue
    'bas': '#FF5722',   # orange-red
}
MARKERS = {
    'S-S': 'o',
    'M-S': 's',
    'S-B': '^',
    'B-L': 'D',
}
FRAME_STYLES = {
    8: '-',
    64: '--',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Group by run
    runs = {}
    for r in results:
        key = r['run']
        if key not in runs:
            runs[key] = []
        runs[key].append(r)

    return results, runs


def parse_run_name(name):
    """Parse 'M-S_8f_fov' -> (config='M-S', frames=8, arch='fov')"""
    parts = name.split('_')
    config = parts[0]
    frames = int(parts[1].replace('f', ''))
    arch = parts[2]
    return config, frames, arch


# ============================================================================
# SCALING LAW FITTING
# ============================================================================

def power_law(C, A, alpha, L_inf):
    """L(C) = A * C^(-alpha) + L_inf"""
    return A * np.power(C, -alpha) + L_inf


def fit_scaling_law(flops, losses, name=""):
    """Fit power law to data. Returns (params, r_squared)."""
    # Initial guesses
    A0 = (max(losses) - min(losses)) * flops[0]**0.1
    alpha0 = 0.1
    L_inf0 = min(losses) * 0.95

    try:
        popt, pcov = curve_fit(
            power_law, flops, losses,
            p0=[A0, alpha0, L_inf0],
            bounds=([0, 0.001, 0], [1e30, 2.0, min(losses)]),
            maxfev=10000,
        )
        # R-squared
        y_pred = power_law(flops, *popt)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt, r2
    except Exception as e:
        print(f"  Warning: fit failed for {name}: {e}")
        return None, 0


def fit_param_scaling(params_list, losses_list):
    """Fit L(N) = B * N^(-beta) + L_inf for model size scaling."""
    params = np.array(params_list)
    losses = np.array(losses_list)

    try:
        popt, _ = curve_fit(
            power_law, params, losses,
            p0=[100, 0.1, min(losses) * 0.95],
            bounds=([0, 0.001, 0], [1e10, 2.0, min(losses)]),
            maxfev=10000,
        )
        y_pred = power_law(params, *popt)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt, r2
    except:
        return None, 0


# ============================================================================
# PLOT 1: Loss vs FLOPs with power law fits
# ============================================================================

def plot_loss_vs_flops(results, runs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, frame_count in enumerate([8, 64]):
        ax = axes[ax_idx]
        fit_data = {}

        for name, pts in sorted(runs.items()):
            config, frames, arch = parse_run_name(name)
            if frames != frame_count:
                continue

            pts_sorted = sorted(pts, key=lambda x: x['step'])
            flops = np.array([p['total_flops'] for p in pts_sorted])
            losses = np.array([p['loss_train'] for p in pts_sorted])

            color = COLORS[arch]
            marker = MARKERS.get(config, 'o')
            label = f"{config} {'foveated' if arch=='fov' else 'baseline'}"

            ax.plot(flops, losses, marker=marker, color=color,
                    markersize=4, linewidth=1.5, alpha=0.7, label=label)

            # Collect for fitting
            key = (arch, frame_count)
            if key not in fit_data:
                fit_data[key] = {'flops': [], 'losses': []}
            fit_data[key]['flops'].extend(flops.tolist())
            fit_data[key]['losses'].extend(losses.tolist())

        # Fit and extrapolate per architecture
        for (arch, fc), data in fit_data.items():
            flops = np.array(data['flops'])
            losses = np.array(data['losses'])
            sort_idx = np.argsort(flops)
            flops, losses = flops[sort_idx], losses[sort_idx]

            popt, r2 = fit_scaling_law(flops, losses, f"{arch}_{fc}f")
            if popt is not None:
                A, alpha, L_inf = popt
                color = COLORS[arch]
                arch_label = 'foveated' if arch == 'fov' else 'baseline'

                # Extrapolation range
                f_min = flops.min() * 0.5
                f_max = flops.max() * 100  # 100x extrapolation
                f_range = np.logspace(np.log10(f_min), np.log10(f_max), 200)
                l_pred = power_law(f_range, *popt)

                ax.plot(f_range, l_pred, '--', color=color, alpha=0.4, linewidth=2)

                # Annotate
                ax.annotate(
                    f"L={A:.1f}·C^(-{alpha:.3f})+{L_inf:.2f}\nR²={r2:.3f}",
                    xy=(f_range[-1], l_pred[-1]),
                    fontsize=7, color=color, alpha=0.8,
                    ha='right', va='bottom',
                )

        ax.set_xscale('log')
        ax.set_xlabel('Training FLOPs')
        ax.set_ylabel('Eval Loss (Cross-Entropy)')
        ax.set_title(f'{frame_count}-Frame: Loss vs Compute')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=3.0)

    fig.suptitle('Scaling Laws: Loss vs Training FLOPs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'loss_vs_flops_fitted.png')
    fig.savefig(OUTPUT_DIR / 'loss_vs_flops_fitted.pdf')
    plt.close(fig)
    print("Saved: loss_vs_flops_fitted.png")


# ============================================================================
# PLOT 2: Perplexity vs FLOPs (log-log) — classic Chinchilla style
# ============================================================================

def plot_perplexity_vs_flops(results, runs):
    fig, ax = plt.subplots(figsize=(10, 7))

    all_fits = {}

    for name, pts in sorted(runs.items()):
        config, frames, arch = parse_run_name(name)
        pts_sorted = sorted(pts, key=lambda x: x['step'])

        flops = np.array([p['total_flops'] for p in pts_sorted])
        ppl = np.array([p['ppl_train'] for p in pts_sorted])

        color = COLORS[arch]
        marker = MARKERS.get(config, 'o')
        ls = FRAME_STYLES.get(frames, '-')
        label = f"{config}_{frames}f {'fov' if arch=='fov' else 'bas'}"

        ax.plot(flops, ppl, marker=marker, color=color, linestyle=ls,
                markersize=4, linewidth=1.5, alpha=0.7, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training FLOPs')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity vs Training Compute (Log-Log)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'perplexity_vs_flops.png')
    fig.savefig(OUTPUT_DIR / 'perplexity_vs_flops.pdf')
    plt.close(fig)
    print("Saved: perplexity_vs_flops.png")


# ============================================================================
# PLOT 3: Model size scaling (params vs final loss)
# ============================================================================

def plot_model_size_scaling(results, runs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, frame_count in enumerate([8, 64]):
        ax = axes[ax_idx]

        for arch in ['fov', 'bas']:
            params_list = []
            losses_list = []
            configs = []

            for name, pts in sorted(runs.items()):
                config, frames, a = parse_run_name(name)
                if frames != frame_count or a != arch:
                    continue
                final = max(pts, key=lambda x: x['step'])
                if final['step'] < 280:
                    continue

                params_list.append(final['param_count_M'])
                losses_list.append(final['loss_train'])
                configs.append(config)

            if len(params_list) < 2:
                continue

            color = COLORS[arch]
            arch_label = 'Foveated' if arch == 'fov' else 'Baseline'

            ax.scatter(params_list, losses_list, color=color, s=100,
                      marker='o', zorder=5, label=arch_label)

            # Label points
            for p, l, c in zip(params_list, losses_list, configs):
                ax.annotate(c, (p, l), textcoords="offset points",
                           xytext=(8, 5), fontsize=8, color=color)

            # Fit power law
            popt, r2 = fit_param_scaling(params_list, losses_list)
            if popt is not None and r2 > 0.5:
                B, beta, L_inf = popt
                p_range = np.linspace(min(params_list)*0.5, 2000, 100)
                l_pred = power_law(p_range, *popt)
                ax.plot(p_range, l_pred, '--', color=color, alpha=0.4, linewidth=2)

                # Extrapolate to 1.7B
                l_1700 = power_law(1700, *popt)
                ax.axhline(l_1700, color=color, alpha=0.2, linestyle=':')
                ax.annotate(f'1.7B pred: {l_1700:.3f}',
                           xy=(1700, l_1700), fontsize=8, color=color,
                           ha='right')

        ax.set_xlabel('Model Parameters (M)')
        ax.set_ylabel('Final Eval Loss')
        ax.set_title(f'{frame_count}-Frame: Loss vs Model Size')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Model Size Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'model_size_scaling.png')
    fig.savefig(OUTPUT_DIR / 'model_size_scaling.pdf')
    plt.close(fig)
    print("Saved: model_size_scaling.png")


# ============================================================================
# PLOT 4: ISO-FLOP curves — at same compute, what loss?
# ============================================================================

def plot_iso_flop(results, runs):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Choose ISO-FLOP budgets
    iso_budgets = [2e15, 5e15, 1e16, 2e16, 4e16]

    for name, pts in sorted(runs.items()):
        config, frames, arch = parse_run_name(name)
        pts_sorted = sorted(pts, key=lambda x: x['total_flops'])

        flops = np.array([p['total_flops'] for p in pts_sorted])
        losses = np.array([p['loss_train'] for p in pts_sorted])

        color = COLORS[arch]
        marker = MARKERS.get(config, 'o')
        ls = FRAME_STYLES.get(frames, '-')
        label = f"{config}_{frames}f {'fov' if arch=='fov' else 'bas'}"

        ax.plot(flops, losses, marker=marker, color=color, linestyle=ls,
                markersize=4, linewidth=1.5, alpha=0.7, label=label)

    # Draw ISO-FLOP lines
    for budget in iso_budgets:
        ax.axvline(budget, color='gray', alpha=0.15, linestyle='-', linewidth=2)
        ax.annotate(f'{budget:.0e}', xy=(budget, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1] > 4 else 4.5),
                   fontsize=7, color='gray', ha='center', rotation=90)

    ax.set_xscale('log')
    ax.set_xlabel('Training FLOPs')
    ax.set_ylabel('Eval Loss')
    ax.set_title('ISO-FLOP Analysis: Loss at Same Compute Budget')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'iso_flop_curves.png')
    fig.savefig(OUTPUT_DIR / 'iso_flop_curves.pdf')
    plt.close(fig)
    print("Saved: iso_flop_curves.png")


# ============================================================================
# PLOT 5: Foveated efficiency — same loss at fewer FLOPs
# ============================================================================

def plot_efficiency_frontier(results, runs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, frame_count in enumerate([8, 64]):
        ax = axes[ax_idx]

        # Collect final points
        fov_points = []
        bas_points = []

        for name, pts in sorted(runs.items()):
            config, frames, arch = parse_run_name(name)
            if frames != frame_count:
                continue
            final = max(pts, key=lambda x: x['step'])
            if final['step'] < 280:
                continue

            point = {
                'config': config,
                'params': final['param_count_M'],
                'loss': final['loss_train'],
                'flops': final['total_flops'],
                'tokens': final['visual_tokens'],
            }

            if arch == 'fov':
                fov_points.append(point)
            else:
                bas_points.append(point)

        # Plot with size proportional to params
        for pts, color, label in [(fov_points, COLORS['fov'], 'Foveated'),
                                    (bas_points, COLORS['bas'], 'Baseline')]:
            flops = [p['flops'] for p in pts]
            losses = [p['loss'] for p in pts]
            sizes = [p['params'] / 2 for p in pts]

            ax.scatter(flops, losses, c=color, s=sizes, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label=label, zorder=5)

            for p in pts:
                ax.annotate(f"{p['config']}\n{p['params']:.0f}M",
                           (p['flops'], p['loss']),
                           textcoords="offset points", xytext=(10, -5),
                           fontsize=8, color=color)

        # Draw arrows showing efficiency gain
        for fp in fov_points:
            for bp in bas_points:
                if fp['config'] == bp['config']:
                    # Arrow from baseline to foveated
                    ax.annotate('',
                              xy=(fp['flops'], fp['loss']),
                              xytext=(bp['flops'], bp['loss']),
                              arrowprops=dict(arrowstyle='->', color='green',
                                            alpha=0.3, lw=1.5))
                    # Efficiency label
                    ratio = bp['flops'] / fp['flops']
                    mid_x = np.sqrt(fp['flops'] * bp['flops'])
                    mid_y = (fp['loss'] + bp['loss']) / 2
                    ax.annotate(f'{ratio:.1f}x', (mid_x, mid_y),
                              fontsize=8, color='green', fontweight='bold',
                              ha='center')

        ax.set_xscale('log')
        ax.set_xlabel('Training FLOPs')
        ax.set_ylabel('Final Eval Loss')
        ax.set_title(f'{frame_count}-Frame: Efficiency Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Foveated Efficiency: Same Config, Fewer FLOPs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'efficiency_frontier.png')
    fig.savefig(OUTPUT_DIR / 'efficiency_frontier.pdf')
    plt.close(fig)
    print("Saved: efficiency_frontier.png")


# ============================================================================
# PLOT 6: Extrapolation to SmolVLM scale
# ============================================================================

def plot_smolvlm_extrapolation(results, runs):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Fit per-architecture scaling laws using ALL data
    arch_data = {'fov': {'flops': [], 'losses': []}, 'bas': {'flops': [], 'losses': []}}

    for name, pts in sorted(runs.items()):
        config, frames, arch = parse_run_name(name)
        final = max(pts, key=lambda x: x['step'])
        if final['step'] < 280:
            continue
        for p in pts:
            arch_data[arch]['flops'].append(p['total_flops'])
            arch_data[arch]['losses'].append(p['loss_train'])

    fits = {}
    for arch in ['fov', 'bas']:
        flops = np.array(arch_data[arch]['flops'])
        losses = np.array(arch_data[arch]['losses'])
        popt, r2 = fit_scaling_law(flops, losses, arch)
        if popt is not None:
            fits[arch] = {'popt': popt, 'r2': r2}

    # Plot data points
    for name, pts in sorted(runs.items()):
        config, frames, arch = parse_run_name(name)
        pts_sorted = sorted(pts, key=lambda x: x['total_flops'])
        flops = [p['total_flops'] for p in pts_sorted]
        losses = [p['loss_train'] for p in pts_sorted]

        color = COLORS[arch]
        marker = MARKERS.get(config, 'o')
        ls = FRAME_STYLES.get(frames, '-')
        label = f"{config}_{frames}f {'fov' if arch=='fov' else 'bas'}"

        ax.plot(flops, losses, marker=marker, color=color, linestyle=ls,
                markersize=4, linewidth=1.2, alpha=0.6, label=label)

    # Extrapolate to SmolVLM scale
    smolvlm_flops_range = np.logspace(14, 19.5, 500)

    for arch, style in [('fov', '-'), ('bas', '--')]:
        if arch not in fits:
            continue
        popt = fits[arch]['popt']
        r2 = fits[arch]['r2']
        A, alpha, L_inf = popt

        l_pred = power_law(smolvlm_flops_range, *popt)
        color = COLORS[arch]
        arch_label = 'Foveated' if arch == 'fov' else 'Baseline'

        ax.plot(smolvlm_flops_range, l_pred, style, color=color, alpha=0.5,
                linewidth=3, label=f'{arch_label} fit (R²={r2:.3f})')

        # Mark SmolVLM equivalent compute points
        for sv_name, sv_cfg in SMOLVLM_CONFIGS.items():
            sv_flops = sv_cfg['approx_flops']
            sv_loss = power_law(sv_flops, *popt)

            ax.scatter([sv_flops], [sv_loss], marker='*', s=200,
                      color=color, edgecolors='black', linewidth=0.5, zorder=10)

            offset_y = 0.1 if arch == 'fov' else -0.15
            ax.annotate(
                f"{sv_name}\n({arch_label})\nL={sv_loss:.2f}",
                xy=(sv_flops, sv_loss),
                xytext=(15, 20 if arch == 'fov' else -30),
                textcoords='offset points',
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.5),
            )

    # Our model configs at 1M and 4.4M samples
    for n_samples, marker_style in [(1_000_000, 'P'), (4_400_000, 'X')]:
        for cfg_name, cfg in OUR_CONFIGS.items():
            if cfg['flops_per_sample_8f'] is None:
                continue
            arch = 'fov' if 'Fov' in cfg_name else 'bas'
            if arch not in fits:
                continue

            total_flops = cfg['flops_per_sample_8f'] * n_samples
            pred_loss = power_law(total_flops, *fits[arch]['popt'])
            color = COLORS[arch]

            ax.scatter([total_flops], [pred_loss], marker=marker_style, s=120,
                      color=color, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=8)

    # Legend entries for sample counts
    ax.scatter([], [], marker='P', s=80, color='gray', label='Our models @ 1M samples')
    ax.scatter([], [], marker='X', s=80, color='gray', label='Our models @ 4.4M samples')
    ax.scatter([], [], marker='*', s=150, color='gray', label='SmolVLM equiv. compute')

    ax.set_xscale('log')
    ax.set_xlabel('Training FLOPs')
    ax.set_ylabel('Eval Loss (Cross-Entropy)')
    ax.set_title('Scaling Law Extrapolation to SmolVLM Scale')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # Shade our data region vs extrapolation
    data_max_flops = max(max(p['total_flops'] for p in pts)
                         for pts in runs.values())
    ax.axvspan(ax.get_xlim()[0], data_max_flops, alpha=0.05, color='green')
    ax.axvspan(data_max_flops, ax.get_xlim()[1], alpha=0.05, color='red')
    ax.text(data_max_flops * 0.3, ax.get_ylim()[0] + 0.1, 'Measured',
            fontsize=9, color='green', alpha=0.5, ha='center')
    ax.text(data_max_flops * 5, ax.get_ylim()[0] + 0.1, 'Extrapolated',
            fontsize=9, color='red', alpha=0.5, ha='center')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'smolvlm_extrapolation.png')
    fig.savefig(OUTPUT_DIR / 'smolvlm_extrapolation.pdf')
    plt.close(fig)
    print("Saved: smolvlm_extrapolation.png")


# ============================================================================
# PLOT 7: Token efficiency — loss per visual token
# ============================================================================

def plot_token_efficiency(results, runs):
    fig, ax = plt.subplots(figsize=(10, 7))

    # For each completed run, plot (visual_tokens_total, final_loss)
    for name, pts in sorted(runs.items()):
        config, frames, arch = parse_run_name(name)
        final = max(pts, key=lambda x: x['step'])
        if final['step'] < 280:
            continue

        total_vis_tokens = final['visual_tokens'] * frames * 280 * 16  # tokens × frames × steps × EB

        color = COLORS[arch]
        marker = MARKERS.get(config, 'o')

        ax.scatter(total_vis_tokens, final['loss_train'],
                  c=color, marker=marker, s=150, edgecolors='black',
                  linewidth=0.5, zorder=5)

        arch_short = 'fov' if arch == 'fov' else 'bas'
        ax.annotate(f"{config}_{frames}f_{arch_short}",
                   (total_vis_tokens, final['loss_train']),
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=8)

    # Add SmolVLM reference
    for sv_name, sv_cfg in SMOLVLM_CONFIGS.items():
        total_vis = sv_cfg['tokens_per_image'] * sv_cfg['train_samples']
        ax.axvline(total_vis, color='purple', alpha=0.2, linestyle='--')
        ax.annotate(sv_name, xy=(total_vis, ax.get_ylim()[1] if ax.get_ylim()[1] > 4 else 4.2),
                   fontsize=8, color='purple', rotation=90, ha='right')

    ax.set_xscale('log')
    ax.set_xlabel('Total Visual Tokens Processed')
    ax.set_ylabel('Final Eval Loss')
    ax.set_title('Token Efficiency: Loss vs Total Visual Tokens')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['fov'],
               markersize=10, label='Foveated (1 tok/frame)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['bas'],
               markersize=10, label='Baseline (16 tok/frame)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'token_efficiency.png')
    fig.savefig(OUTPUT_DIR / 'token_efficiency.pdf')
    plt.close(fig)
    print("Saved: token_efficiency.png")


# ============================================================================
# PLOT 8: Training curves (loss vs step) — all runs overlaid
# ============================================================================

def plot_training_curves(results, runs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, frame_count in enumerate([8, 64]):
        ax = axes[ax_idx]

        for name, pts in sorted(runs.items()):
            config, frames, arch = parse_run_name(name)
            if frames != frame_count:
                continue

            pts_sorted = sorted(pts, key=lambda x: x['step'])
            steps = [p['step'] for p in pts_sorted]
            losses = [p['loss_train'] for p in pts_sorted]

            color = COLORS[arch]
            marker = MARKERS.get(config, 'o')
            label = f"{config} {'fov' if arch=='fov' else 'bas'} ({pts_sorted[-1]['param_count_M']:.0f}M)"

            ax.plot(steps, losses, marker=marker, color=color,
                    markersize=3, linewidth=1.5, alpha=0.7, label=label)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Eval Loss')
        ax.set_title(f'{frame_count}-Frame: Training Curves')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training Curves by Frame Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'training_curves.png')
    fig.savefig(OUTPUT_DIR / 'training_curves.pdf')
    plt.close(fig)
    print("Saved: training_curves.png")


# ============================================================================
# PLOT 9: Chinchilla optimal — data vs params tradeoff
# ============================================================================

def plot_chinchilla_optimal(results, runs):
    """Estimate optimal data/params ratio from our measurements."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # For each model size, what's the final loss at different data amounts?
    # We only have 1 data point per config (280 steps × EB16 = 4480 samples)
    # But we can use the loss curve to estimate where more data would help

    # Collect (params, samples_seen, final_loss) for 8F runs
    points = []
    for name, pts in sorted(runs.items()):
        config, frames, arch = parse_run_name(name)
        if frames != 8:
            continue
        final = max(pts, key=lambda x: x['step'])
        if final['step'] < 280:
            continue

        samples = final['step'] * final['effective_batch']
        points.append({
            'config': config,
            'arch': arch,
            'params_M': final['param_count_M'],
            'samples': samples,
            'loss': final['loss_train'],
            'flops_per_sample': final['flops_per_sample'],
        })

    # Plot loss vs params/data ratio
    for p in points:
        color = COLORS[p['arch']]
        marker = MARKERS.get(p['config'], 'o')
        # Chinchilla suggests D ≈ 20 * N for optimal
        ratio = p['samples'] / (p['params_M'] * 1e6)

        ax.scatter(ratio, p['loss'], c=color, marker=marker, s=150,
                  edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f"{p['config']}_{p['arch']}\n{p['params_M']:.0f}M",
                   (ratio, p['loss']),
                   textcoords="offset points", xytext=(10, 5), fontsize=8)

    # Chinchilla optimal line
    ax.axvline(20, color='purple', alpha=0.3, linestyle='--', linewidth=2)
    ax.annotate('Chinchilla optimal\n(D/N ≈ 20)', xy=(20, ax.get_ylim()[0] + 0.1),
               fontsize=10, color='purple', ha='center')

    ax.set_xlabel('Data/Params Ratio (samples / params)')
    ax.set_ylabel('Final Eval Loss')
    ax.set_title('Data Efficiency: Are We Under-Training?')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['fov'],
               markersize=10, label='Foveated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['bas'],
               markersize=10, label='Baseline'),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'chinchilla_ratio.png')
    fig.savefig(OUTPUT_DIR / 'chinchilla_ratio.pdf')
    plt.close(fig)
    print("Saved: chinchilla_ratio.png")


# ============================================================================
# TABLE: Summary with fits and predictions
# ============================================================================

def print_summary_table(results, runs):
    print("\n" + "=" * 100)
    print("SCALING LAW SUMMARY")
    print("=" * 100)

    # Per-architecture fits
    print("\n--- Power Law Fits: L(C) = A·C^(-α) + L∞ ---\n")

    for group_name, filter_fn in [
        ("All foveated", lambda c, f, a: a == 'fov'),
        ("All baseline", lambda c, f, a: a == 'bas'),
        ("Foveated 8F", lambda c, f, a: a == 'fov' and f == 8),
        ("Baseline 8F", lambda c, f, a: a == 'bas' and f == 8),
        ("Foveated 64F", lambda c, f, a: a == 'fov' and f == 64),
        ("Baseline 64F", lambda c, f, a: a == 'bas' and f == 64),
    ]:
        flops_all, losses_all = [], []
        for name, pts in runs.items():
            config, frames, arch = parse_run_name(name)
            if not filter_fn(config, frames, arch):
                continue
            final = max(pts, key=lambda x: x['step'])
            if final['step'] < 280:
                continue
            for p in pts:
                flops_all.append(p['total_flops'])
                losses_all.append(p['loss_train'])

        if len(flops_all) < 5:
            continue

        popt, r2 = fit_scaling_law(np.array(flops_all), np.array(losses_all), group_name)
        if popt is not None:
            A, alpha, L_inf = popt
            print(f"  {group_name:20s}: A={A:.2f}, α={alpha:.4f}, L∞={L_inf:.3f}, R²={r2:.4f}")

            # Predict at SmolVLM scale
            for sv_name, sv_cfg in SMOLVLM_CONFIGS.items():
                pred = power_law(sv_cfg['approx_flops'], *popt)
                print(f"    → {sv_name} equiv ({sv_cfg['approx_flops']:.1e} FLOPs): L={pred:.3f}, PPL={np.exp(pred):.1f}")

    # Efficiency summary
    print("\n--- Foveated Efficiency at 280 steps ---\n")
    print(f"  {'Config':12s} {'Fov Loss':>10s} {'Bas Loss':>10s} {'Gap':>8s} {'FLOPs Ratio':>12s} {'Token Ratio':>12s}")

    pairs = {}
    for name, pts in runs.items():
        config, frames, arch = parse_run_name(name)
        base = f"{config}_{frames}f"
        final = max(pts, key=lambda x: x['step'])
        if final['step'] < 280:
            continue
        if base not in pairs:
            pairs[base] = {}
        pairs[base][arch] = final

    for base, archs in sorted(pairs.items()):
        if 'fov' not in archs or 'bas' not in archs:
            continue
        fov, bas = archs['fov'], archs['bas']
        gap = (fov['loss_train'] - bas['loss_train']) / bas['loss_train'] * 100
        flops_ratio = bas['total_flops'] / fov['total_flops']
        token_ratio = bas['visual_tokens'] / fov['visual_tokens']
        print(f"  {base:12s} {fov['loss_train']:10.4f} {bas['loss_train']:10.4f} {gap:+7.1f}% {flops_ratio:11.2f}x {token_ratio:11.0f}x")

    # Data requirements estimate
    print("\n--- Data Requirements for Target Losses ---\n")

    # Using M-S foveated 8F fit to estimate data needed
    fov_8f_flops = []
    fov_8f_losses = []
    for name, pts in runs.items():
        config, frames, arch = parse_run_name(name)
        if arch != 'fov' or frames != 8:
            continue
        final = max(pts, key=lambda x: x['step'])
        if final['step'] < 280:
            continue
        for p in pts:
            fov_8f_flops.append(p['total_flops'])
            fov_8f_losses.append(p['loss_train'])

    if fov_8f_flops:
        popt, r2 = fit_scaling_law(np.array(fov_8f_flops), np.array(fov_8f_losses), "fov_8f")
        if popt is not None:
            A, alpha, L_inf = popt
            print(f"  Using foveated 8F fit: L = {A:.2f}·C^(-{alpha:.4f}) + {L_inf:.3f}")
            print(f"  Irreducible loss estimate: {L_inf:.3f} (PPL {np.exp(L_inf):.1f})")
            print()

            flops_per_sample_ms = 1.08e12  # M-S fov 8F
            for target_loss in [3.5, 3.0, 2.5, 2.0]:
                if target_loss <= L_inf:
                    print(f"  Target L={target_loss}: BELOW irreducible loss ({L_inf:.3f})")
                    continue
                # Solve: target = A * C^(-alpha) + L_inf
                # C = (A / (target - L_inf))^(1/alpha)
                needed_flops = (A / (target_loss - L_inf)) ** (1 / alpha)
                needed_samples = needed_flops / flops_per_sample_ms
                print(f"  Target L={target_loss} (PPL {np.exp(target_loss):.1f}): "
                      f"need {needed_flops:.2e} FLOPs = {needed_samples/1e6:.1f}M samples (M-S fov 8F)")

    print("\n" + "=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading results...")
    results, runs = load_results()

    complete = sum(1 for name, pts in runs.items() if max(p['step'] for p in pts) >= 280)
    print(f"Total: {len(results)} data points, {len(runs)} runs ({complete} complete)")
    print()

    # Generate all plots
    plot_loss_vs_flops(results, runs)
    plot_perplexity_vs_flops(results, runs)
    plot_model_size_scaling(results, runs)
    plot_iso_flop(results, runs)
    plot_efficiency_frontier(results, runs)
    plot_smolvlm_extrapolation(results, runs)
    plot_token_efficiency(results, runs)
    plot_training_curves(results, runs)
    plot_chinchilla_optimal(results, runs)

    # Summary table
    print_summary_table(results, runs)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
