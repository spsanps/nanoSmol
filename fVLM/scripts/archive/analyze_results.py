"""
Post-training analysis script.

Analyzes training results and generates visualizations for management report.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.foveated_vlm import FoveatedVideoModel
from src.data.dataset import create_dataloader
from src.training.visualization import (
    get_attention_maps,
    visualize_attention_video,
    analyze_attention_focus,
)


def plot_training_curves(log_dir, save_dir):
    """
    Plot training curves from W&B logs or tensorboard.

    For now, creates placeholder plots. TODO: Parse actual logs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load actual training logs
    # For now, create example plots

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves (placeholder)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Loss Ratio (coarse/fine)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Attention Entropy')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Memory Usage')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('GB')
    axes[1, 1].axhline(y=18, color='r', linestyle='--', label='Safety Limit')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved training curves to {save_dir / 'training_curves.png'}")


def analyze_checkpoint(checkpoint_path, dataloader, num_samples=10):
    """
    Analyze a trained checkpoint.

    Returns:
        dict with analysis results
    """
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    # Load model
    config = checkpoint['config']
    model = FoveatedVideoModel(
        dino_model=config['model']['dino_model'],
        llm_model=config['model']['llm_model'],
        dino_dim=config['model']['dino_dim'],
        llm_dim=config['model']['llm_dim'],
        query_dim=config['model']['query_dim'],
        lambda_coarse=config['model']['lambda_coarse'],
    ).cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   Step: {checkpoint['step']}")
    print(f"   Model loaded successfully")

    # Evaluate on samples
    print(f"\n🔄 Evaluating on {num_samples} samples...")

    results = {
        'losses_fine': [],
        'losses_coarse': [],
        'entropy_static': [],
        'entropy_dynamic': [],
        'attention_focus': [],
    }

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            frames = batch['frames'].cuda()
            vae_latents = batch['vae_latents'].cuda()
            text_embeds = model.get_empty_text_embeds(frames.shape[0]).cuda()

            # Compute losses
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss_fine, loss_coarse = model(text_embeds, frames, vae_latents)

            results['losses_fine'].append(loss_fine.item())
            results['losses_coarse'].append(loss_coarse.item())

            # Analyze attention
            attn_static, attn_dynamic = get_attention_maps(model, frames)
            focus_stats = analyze_attention_focus(attn_static, attn_dynamic)

            results['entropy_static'].append(focus_stats['entropy_static_mean'])
            results['entropy_dynamic'].append(focus_stats['entropy_dynamic_mean'])
            results['attention_focus'].append(focus_stats['is_more_focused'])

    # Compute statistics
    analysis = {
        'loss_fine_mean': np.mean(results['losses_fine']),
        'loss_fine_std': np.std(results['losses_fine']),
        'loss_coarse_mean': np.mean(results['losses_coarse']),
        'loss_coarse_std': np.std(results['losses_coarse']),
        'loss_ratio_mean': np.mean(results['losses_coarse']) / np.mean(results['losses_fine']),
        'improvement_pct': (np.mean(results['losses_coarse']) - np.mean(results['losses_fine'])) / np.mean(results['losses_coarse']) * 100,
        'entropy_static_mean': np.mean(results['entropy_static']),
        'entropy_dynamic_mean': np.mean(results['entropy_dynamic']),
        'entropy_reduction': np.mean(results['entropy_static']) - np.mean(results['entropy_dynamic']),
        'pct_more_focused': np.mean(results['attention_focus']) * 100,
    }

    return analysis, results


def generate_report(analysis, save_path):
    """
    Generate management report.
    """
    report = f"""
# Foveated VLM - Phase 1 Results

## Executive Summary

**Core Hypothesis:** Dynamic foveated attention (Pass 2) outperforms static attention (Pass 1)

**Result:** {"✓ VALIDATED" if analysis['improvement_pct'] > 0 else "✗ NOT VALIDATED"}

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Loss (Fine)** | {analysis['loss_fine_mean']:.4f} ± {analysis['loss_fine_std']:.4f} |
| **Loss (Coarse)** | {analysis['loss_coarse_mean']:.4f} ± {analysis['loss_coarse_std']:.4f} |
| **Loss Ratio** | {analysis['loss_ratio_mean']:.3f} |
| **Improvement** | **{analysis['improvement_pct']:.1f}%** |

---

## Attention Analysis

| Metric | Static | Dynamic | Reduction |
|--------|--------|---------|-----------|
| **Entropy** | {analysis['entropy_static_mean']:.3f} | {analysis['entropy_dynamic_mean']:.3f} | {analysis['entropy_reduction']:.3f} |

**Interpretation:**
- Lower entropy = more focused attention
- Dynamic attention is more focused: **{analysis['pct_more_focused']:.0f}%** of samples

---

## Conclusion

{"✓ The model successfully learned to use dynamic queries for focused attention." if analysis['improvement_pct'] > 5 else "⚠ Marginal improvement. Further investigation needed."}

**Next Steps:**
- Scale up training with more compute
- Add text conditioning (Phase 2)
- Compare against SmolVLM2 baseline
- Explore larger models (SmolLM2-360M, DINOv2-B)
"""

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Generated report: {save_path}")
    return report


def main():
    """Run full analysis pipeline."""
    print("=" * 70)
    print("Foveated VLM - Results Analysis")
    print("=" * 70)

    # Check for checkpoint
    checkpoint_dir = Path("outputs/checkpoints")
    if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.pt")):
        print("\n⚠️  No checkpoints found. Train the model first.")
        print(f"   Looking in: {checkpoint_dir}")
        return

    # Use latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    latest = checkpoints[-1]

    # Create dataloader
    print("\n📦 Creating dataloader...")
    dataloader = create_dataloader(
        video_dir="data/videos",
        latent_dir="data/latents",
        batch_size=2,
        num_workers=2,
        shuffle=False,
        num_frames=8,
    )

    # Analyze
    analysis, results = analyze_checkpoint(latest, dataloader, num_samples=20)

    # Print results
    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)
    for k, v in analysis.items():
        print(f"{k:30s}: {v:.4f}" if isinstance(v, float) else f"{k:30s}: {v}")

    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    viz_dir = Path("outputs/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    plot_training_curves("outputs/logs", viz_dir)

    # Sample attention visualizations
    print("\n🎨 Generating attention visualizations...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
    ).cuda()

    checkpoint = torch.load(latest, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for i, batch in enumerate(dataloader):
        if i >= 5:  # Generate 5 samples
            break

        frames = batch['frames'].cuda()
        video_id = batch['video_id'][0]

        attn_static, attn_dynamic = get_attention_maps(model, frames)

        save_path = viz_dir / f"attention_{i}_{video_id}.png"
        visualize_attention_video(
            frames[0],
            attn_static[0],
            attn_dynamic[0],
            save_path=save_path,
            video_id=video_id,
        )
        print(f"   ✓ Saved: {save_path.name}")

    # Generate report
    print("\n📝 Generating management report...")
    report_path = Path("outputs/REPORT.md")
    report = generate_report(analysis, report_path)
    print(report)

    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print(f"   Report: {report_path}")
    print(f"   Visualizations: {viz_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
