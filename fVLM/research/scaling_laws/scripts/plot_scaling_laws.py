#!/usr/bin/env python3
"""
Generate comprehensive scaling law plots.
Combines S-S and M-S configurations for foveated vs baseline comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
def load_all_data():
    """Load and combine all scaling data."""
    all_data = []

    # Load S-S data
    ss_file = DATA_DIR / 'scaling_data_S-S.json'
    if ss_file.exists():
        with open(ss_file) as f:
            ss_data = json.load(f)
        for d in ss_data:
            # Normalize field names
            all_data.append({
                'config': d.get('model_config', d.get('config', 'S-S')),
                'model_type': d['model_type'],
                'step': d['step'],
                'loss': d.get('val_loss', d.get('loss_mean')),
                'loss_se': d.get('val_loss_se', d.get('loss_se')),
                'perplexity': d['perplexity'],
                'flops': d.get('total_training_flops', d.get('total_flops')),
                'visual_tokens': d.get('visual_tokens_per_frame', d.get('visual_tokens')),
            })

    # Load M-S data
    ms_file = DATA_DIR / 'scaling_data_combined.json'
    if ms_file.exists():
        with open(ms_file) as f:
            ms_data = json.load(f)
        for d in ms_data:
            all_data.append({
                'config': d.get('config', 'M-S'),
                'model_type': d['model_type'],
                'step': d['step'],
                'loss': d.get('loss_mean'),
                'loss_se': d.get('loss_se'),
                'perplexity': d['perplexity'],
                'flops': d.get('total_flops'),
                'visual_tokens': d.get('visual_tokens'),
            })

    return all_data


def plot_loss_vs_flops(data):
    """Plot loss vs FLOPs for all configurations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        ('S-S', 'foveated'): 'C0',
        ('S-S', 'baseline'): 'C0',
        ('M-S', 'foveated'): 'C1',
        ('M-S', 'baseline'): 'C1',
    }
    linestyles = {'foveated': '-', 'baseline': '--'}
    markers = {'foveated': 'o', 'baseline': 's'}

    plotted = set()

    for config in ['S-S', 'M-S']:
        for model_type in ['foveated', 'baseline']:
            subset = [d for d in data if d['config'] == config and d['model_type'] == model_type]
            if not subset:
                continue

            subset = sorted(subset, key=lambda x: x['flops'])
            flops = np.array([d['flops'] for d in subset]) / 1e15  # Convert to PFLOPs
            losses = [d['loss'] for d in subset]
            loss_se = [d['loss_se'] for d in subset]

            color = colors[(config, model_type)]
            label = f'{config} {model_type}'

            ax.errorbar(flops, losses, yerr=loss_se,
                       color=color, linestyle=linestyles[model_type],
                       marker=markers[model_type], markersize=8, capsize=3,
                       label=label, linewidth=2)
            plotted.add((config, model_type))

    ax.set_xlabel('Training FLOPs (PFLOPs)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Scaling Laws: Loss vs Compute', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_vs_flops.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'loss_vs_flops.pdf')
    print(f"Saved: {OUTPUT_DIR / 'loss_vs_flops.png'}")
    plt.close()


def plot_perplexity_vs_flops(data):
    """Plot perplexity vs FLOPs for all configurations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        ('S-S', 'foveated'): 'C0',
        ('S-S', 'baseline'): 'C0',
        ('M-S', 'foveated'): 'C1',
        ('M-S', 'baseline'): 'C1',
    }
    linestyles = {'foveated': '-', 'baseline': '--'}
    markers = {'foveated': 'o', 'baseline': 's'}

    for config in ['S-S', 'M-S']:
        for model_type in ['foveated', 'baseline']:
            subset = [d for d in data if d['config'] == config and d['model_type'] == model_type]
            if not subset:
                continue

            subset = sorted(subset, key=lambda x: x['flops'])
            flops = np.array([d['flops'] for d in subset]) / 1e15
            ppl = [d['perplexity'] for d in subset]

            color = colors[(config, model_type)]
            label = f'{config} {model_type}'

            ax.plot(flops, ppl, color=color, linestyle=linestyles[model_type],
                   marker=markers[model_type], markersize=8, label=label, linewidth=2)

    ax.set_xlabel('Training FLOPs (PFLOPs)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Scaling Laws: Perplexity vs Compute', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'perplexity_vs_flops.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'perplexity_vs_flops.pdf')
    print(f"Saved: {OUTPUT_DIR / 'perplexity_vs_flops.png'}")
    plt.close()


def plot_foveated_vs_baseline(data):
    """Plot direct comparison of foveated vs baseline at same steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, config in enumerate(['S-S', 'M-S']):
        ax = axes[ax_idx]

        fov = [d for d in data if d['config'] == config and d['model_type'] == 'foveated']
        base = [d for d in data if d['config'] == config and d['model_type'] == 'baseline']

        if not fov or not base:
            continue

        fov = sorted(fov, key=lambda x: x['step'])
        base = sorted(base, key=lambda x: x['step'])

        steps = [d['step'] for d in fov]
        fov_loss = [d['loss'] for d in fov]
        fov_se = [d['loss_se'] for d in fov]
        base_loss = [d['loss'] for d in base]
        base_se = [d['loss_se'] for d in base]

        x = np.arange(len(steps))
        width = 0.35

        bars1 = ax.bar(x - width/2, fov_loss, width, yerr=fov_se,
                       label='Foveated (1 token/frame)', color='C0', capsize=3)
        bars2 = ax.bar(x + width/2, base_loss, width, yerr=base_se,
                       label='Baseline (16 tokens/frame)', color='C1', capsize=3)

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(f'{config}: Foveated vs Baseline', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add delta labels
        for i, (fl, bl) in enumerate(zip(fov_loss, base_loss)):
            delta = fl - bl
            delta_pct = (fl - bl) / bl * 100
            color = 'red' if delta > 0 else 'green'
            ax.annotate(f'{delta_pct:+.1f}%', xy=(i, max(fl, bl) + 0.1),
                       ha='center', fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'foveated_vs_baseline.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'foveated_vs_baseline.pdf')
    print(f"Saved: {OUTPUT_DIR / 'foveated_vs_baseline.png'}")
    plt.close()


def plot_model_size_scaling(data):
    """Plot model size scaling (S-S vs M-S) at same training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, model_type in enumerate(['foveated', 'baseline']):
        ax = axes[ax_idx]

        ss = [d for d in data if d['config'] == 'S-S' and d['model_type'] == model_type]
        ms = [d for d in data if d['config'] == 'M-S' and d['model_type'] == model_type]

        if not ss or not ms:
            continue

        ss = sorted(ss, key=lambda x: x['step'])
        ms = sorted(ms, key=lambda x: x['step'])

        steps = [d['step'] for d in ss]
        ss_loss = [d['loss'] for d in ss]
        ss_se = [d['loss_se'] for d in ss]
        ms_loss = [d['loss'] for d in ms]
        ms_se = [d['loss_se'] for d in ms]

        x = np.arange(len(steps))
        width = 0.35

        bars1 = ax.bar(x - width/2, ss_loss, width, yerr=ss_se,
                       label='S-S (135M LLM)', color='C0', capsize=3)
        bars2 = ax.bar(x + width/2, ms_loss, width, yerr=ms_se,
                       label='M-S (360M LLM)', color='C1', capsize=3)

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(f'{model_type.capitalize()}: Model Size Scaling', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add delta labels
        for i, (sl, ml) in enumerate(zip(ss_loss, ms_loss)):
            delta_pct = (ml - sl) / sl * 100
            color = 'green' if delta_pct < 0 else 'red'
            ax.annotate(f'{delta_pct:+.1f}%', xy=(i, max(sl, ml) + 0.1),
                       ha='center', fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_size_scaling.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'model_size_scaling.pdf')
    print(f"Saved: {OUTPUT_DIR / 'model_size_scaling.png'}")
    plt.close()


def plot_efficiency_comparison(data):
    """Plot compute efficiency: loss vs visual tokens."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get final step (3000) for each config
    final_data = [d for d in data if d['step'] == 3000]

    for d in final_data:
        marker = 'o' if d['model_type'] == 'foveated' else 's'
        color = 'C0' if d['config'] == 'S-S' else 'C1'
        ax.scatter(d['visual_tokens'], d['loss'],
                  s=200, marker=marker, c=color,
                  label=f"{d['config']} {d['model_type']}")
        ax.annotate(f"Loss: {d['loss']:.3f}",
                   (d['visual_tokens'], d['loss']),
                   xytext=(10, 10), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Visual Tokens per Frame', fontsize=12)
    ax.set_ylabel('Validation Loss @ 3000 steps', fontsize=12)
    ax.set_title('Efficiency: Loss vs Visual Token Count', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_comparison.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'efficiency_comparison.pdf')
    print(f"Saved: {OUTPUT_DIR / 'efficiency_comparison.png'}")
    plt.close()


def plot_iso_loss_curves(data):
    """Plot iso-loss curves showing FLOPs required to reach target losses."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    configs = ['S-S', 'M-S']
    model_types = ['foveated', 'baseline']

    # Interpolate to find FLOPs at target losses
    target_losses = [4.2, 4.0, 3.9, 3.8, 3.7]

    results = {}
    for config in configs:
        for mt in model_types:
            subset = [d for d in data if d['config'] == config and d['model_type'] == mt]
            if len(subset) < 2:
                continue
            subset = sorted(subset, key=lambda x: x['flops'])
            flops = np.array([d['flops'] for d in subset])
            losses = np.array([d['loss'] for d in subset])

            # Interpolate
            interp_flops = []
            for tl in target_losses:
                if losses.min() <= tl <= losses.max():
                    # Linear interpolation in log space
                    idx = np.searchsorted(-losses, -tl)  # losses are decreasing
                    if idx == 0:
                        interp_flops.append(flops[0])
                    elif idx >= len(losses):
                        interp_flops.append(flops[-1])
                    else:
                        # Interpolate
                        l1, l2 = losses[idx-1], losses[idx]
                        f1, f2 = flops[idx-1], flops[idx]
                        t = (tl - l1) / (l2 - l1)
                        interp_flops.append(f1 * (1-t) + f2 * t)
                else:
                    interp_flops.append(np.nan)

            results[(config, mt)] = interp_flops

    # Plot
    x = np.arange(len(target_losses))
    width = 0.2

    for i, (config, mt) in enumerate(results.keys()):
        flops = np.array(results[(config, mt)]) / 1e15  # PFLOPs
        offset = (i - 1.5) * width
        color = 'C0' if config == 'S-S' else 'C1'
        pattern = '' if mt == 'foveated' else '//'
        ax.bar(x + offset, flops, width, label=f'{config} {mt}',
               color=color, alpha=0.8 if mt == 'foveated' else 0.5,
               hatch=pattern)

    ax.set_xlabel('Target Loss', fontsize=12)
    ax.set_ylabel('FLOPs Required (PFLOPs)', fontsize=12)
    ax.set_title('Iso-Loss Curves: Compute Required to Reach Target Loss', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l:.1f}' for l in target_losses])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'iso_loss_curves.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'iso_loss_curves.pdf')
    print(f"Saved: {OUTPUT_DIR / 'iso_loss_curves.png'}")
    plt.close()


def generate_summary_table(data):
    """Generate a summary markdown table."""
    output = "# Scaling Law Results Summary\n\n"
    output += "## Raw Data\n\n"
    output += "| Config | Type | Step | Loss | ±SE | PPL | FLOPs (P) | Visual Tokens |\n"
    output += "|--------|------|------|------|-----|-----|-----------|---------------|\n"

    for d in sorted(data, key=lambda x: (x['config'], x['model_type'], x['step'])):
        flops_p = d['flops'] / 1e15
        output += f"| {d['config']} | {d['model_type']} | {d['step']} | "
        output += f"{d['loss']:.4f} | {d['loss_se']:.4f} | {d['perplexity']:.2f} | "
        output += f"{flops_p:.2f} | {d['visual_tokens']} |\n"

    output += "\n## Key Findings\n\n"

    # Compare at step 3000
    final = [d for d in data if d['step'] == 3000]
    output += "### At 3000 Steps (Final)\n\n"

    for config in ['S-S', 'M-S']:
        fov = next((d for d in final if d['config'] == config and d['model_type'] == 'foveated'), None)
        base = next((d for d in final if d['config'] == config and d['model_type'] == 'baseline'), None)

        if fov and base:
            delta = fov['loss'] - base['loss']
            delta_pct = delta / base['loss'] * 100
            direction = "higher" if delta > 0 else "lower"
            output += f"- **{config}**: Foveated loss {fov['loss']:.4f} vs Baseline {base['loss']:.4f} "
            output += f"({delta_pct:+.2f}% {direction})\n"
            output += f"  - Foveated uses {fov['visual_tokens']} tokens, Baseline uses {base['visual_tokens']} tokens\n"
            output += f"  - Efficiency ratio: {base['visual_tokens']/fov['visual_tokens']:.0f}x fewer tokens for "
            output += f"{abs(delta_pct):.1f}% {'worse' if delta > 0 else 'better'} loss\n"

    output += "\n### Model Size Effect\n\n"

    for mt in ['foveated', 'baseline']:
        ss = next((d for d in final if d['config'] == 'S-S' and d['model_type'] == mt), None)
        ms = next((d for d in final if d['config'] == 'M-S' and d['model_type'] == mt), None)

        if ss and ms:
            delta = ms['loss'] - ss['loss']
            delta_pct = delta / ss['loss'] * 100
            output += f"- **{mt.capitalize()}**: M-S (360M) vs S-S (135M) = {delta_pct:+.2f}% "
            output += f"({'improvement' if delta < 0 else 'regression'})\n"

    # Save
    summary_path = OUTPUT_DIR / 'summary.md'
    with open(summary_path, 'w') as f:
        f.write(output)
    print(f"Saved: {summary_path}")

    return output


def main():
    print("Loading data...")
    data = load_all_data()
    print(f"Loaded {len(data)} data points")

    print("\nGenerating plots...")
    plot_loss_vs_flops(data)
    plot_perplexity_vs_flops(data)
    plot_foveated_vs_baseline(data)
    plot_model_size_scaling(data)
    plot_efficiency_comparison(data)
    plot_iso_loss_curves(data)

    print("\nGenerating summary...")
    summary = generate_summary_table(data)
    print("\n" + summary)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
