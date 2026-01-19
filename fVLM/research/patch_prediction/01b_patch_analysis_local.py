#!/usr/bin/env python3
"""
Diagnostic 01b: DINO Patch Analysis (using local precomputed data)

Uses the 19K samples already on D drive - no network needed.

Questions:
1. How similar are patches across temporal gaps?
2. Which patches change most?
3. What's the baseline MSE for patch prediction?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(device):
    """Load frozen DINO for patch extraction."""
    print("Loading DINO...", flush=True)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    print("DINO loaded!", flush=True)
    return dino


def load_frames_from_dir(frame_dir, device):
    """Load and normalize frames from directory.

    Returns:
        frames: [T, 3, 256, 256] normalized for DINO
    """
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    frames = []
    for fp in frame_paths:
        img = Image.open(fp).convert('RGB')
        # Resize to 256 if needed
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
        frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(frame)

    frames = torch.stack(frames)  # [T, 3, H, W]
    # Normalize for DINO
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames.to(device)


def extract_patches(dino, frames):
    """Extract DINO patches from frames.

    Args:
        frames: [T, 3, H, W] normalized frames
    Returns:
        patches: [T, 257, 384] (CLS + 256 spatial patches)
    """
    with torch.no_grad():
        outputs = dino(frames, output_hidden_states=True)
        patches = outputs.last_hidden_state  # [T, 257, 384]
    return patches


def analyze_single_video(patches, gaps=[1, 2, 4, 8, 12]):
    """Analyze patch similarity for a single video.

    Args:
        patches: [T, N, 384] where N=325 for dinov2-small with 256x256 input
        gaps: frame gaps to analyze

    Returns:
        dict with per-gap statistics
    """
    T, N, D = patches.shape  # N=325 (1 CLS + 18x18 patches)
    results = {}

    for gap in gaps:
        if gap >= T:
            continue

        patch_t = patches[:-gap]  # [T-gap, N, D]
        patch_t_plus_gap = patches[gap:]  # [T-gap, N, D]

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            patch_t.reshape(-1, D),
            patch_t_plus_gap.reshape(-1, D),
            dim=-1
        ).reshape(-1, N)  # [T-gap, N]

        # MSE
        mse = ((patch_t - patch_t_plus_gap) ** 2).mean(dim=-1)  # [T-gap, N]

        # L1 distance
        l1 = (patch_t - patch_t_plus_gap).abs().mean(dim=-1)  # [T-gap, N]

        results[gap] = {
            'cos_sim_mean': cos_sim.mean().item(),
            'cos_sim_std': cos_sim.std().item(),
            'cos_sim_per_patch': cos_sim.mean(dim=0).cpu().numpy(),  # [N]
            'mse_mean': mse.mean().item(),
            'mse_std': mse.std().item(),
            'mse_per_patch': mse.mean(dim=0).cpu().numpy(),  # [N]
            'l1_mean': l1.mean().item(),
            'num_patches': N,
            # CLS vs spatial
            'cls_cos_sim': cos_sim[:, 0].mean().item(),
            'spatial_cos_sim': cos_sim[:, 1:].mean().item(),
            'cls_mse': mse[:, 0].mean().item(),
            'spatial_mse': mse[:, 1:].mean().item(),
        }

    return results


def run_analysis(data_dir, num_videos=100, device='cuda'):
    """Run analysis on local precomputed data."""

    data_dir = Path(data_dir)
    frame_dirs = sorted(data_dir.glob("*"))[:num_videos * 2]  # Sample more in case some fail

    print(f"Found {len(frame_dirs)} video directories")

    dino = load_dino(device)

    all_results = []
    videos_processed = 0

    # Shuffle and sample
    random.shuffle(frame_dirs)

    for frame_dir in tqdm(frame_dirs, desc="Processing videos"):
        if videos_processed >= num_videos:
            break

        if not frame_dir.is_dir():
            continue

        try:
            frames = load_frames_from_dir(frame_dir, device)
            if frames.shape[0] < 8:  # Need at least 8 frames
                continue

            patches = extract_patches(dino, frames)
            results = analyze_single_video(patches)
            all_results.append(results)
            videos_processed += 1

        except Exception as e:
            print(f"Error processing {frame_dir}: {e}")
            continue

    print(f"\nProcessed {videos_processed} videos")
    return all_results


def summarize_and_plot(all_results, output_dir):
    """Create summary statistics and plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaps = [1, 2, 4, 8, 12]

    print("\n" + "=" * 70)
    print("PATCH SIMILARITY ANALYSIS")
    print("=" * 70)

    print(f"\n{'Gap':<6} {'Frames':<8} {'Cos Sim':<12} {'MSE':<12} {'L1':<12} {'CLS Sim':<10} {'Spatial Sim':<12}")
    print("-" * 80)

    summary = {}
    for gap in gaps:
        data = [r[gap] for r in all_results if gap in r]
        if not data:
            continue

        # At 8fps, gap frames = gap/8 seconds
        time_gap = gap / 8.0  # seconds

        summary[gap] = {
            'cos_sim': np.mean([d['cos_sim_mean'] for d in data]),
            'cos_sim_std': np.std([d['cos_sim_mean'] for d in data]),
            'mse': np.mean([d['mse_mean'] for d in data]),
            'l1': np.mean([d['l1_mean'] for d in data]),
            'cls_sim': np.mean([d['cls_cos_sim'] for d in data]),
            'spatial_sim': np.mean([d['spatial_cos_sim'] for d in data]),
            'per_patch_sim': np.mean([d['cos_sim_per_patch'] for d in data], axis=0),
            'per_patch_mse': np.mean([d['mse_per_patch'] for d in data], axis=0),
        }

        s = summary[gap]
        print(f"{gap:<6} {time_gap:.2f}s    {s['cos_sim']:.4f}±{s['cos_sim_std']:.3f}  "
              f"{s['mse']:.4f}       {s['l1']:.4f}       {s['cls_sim']:.4f}     {s['spatial_sim']:.4f}")

    # === PLOTS ===

    # 1. Similarity vs Gap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    gaps_plot = sorted(summary.keys())
    times = [g/8 for g in gaps_plot]

    cos_sims = [summary[g]['cos_sim'] for g in gaps_plot]
    mses = [summary[g]['mse'] for g in gaps_plot]
    l1s = [summary[g]['l1'] for g in gaps_plot]

    axes[0].plot(times, cos_sims, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0].fill_between(times,
                          [summary[g]['cos_sim'] - summary[g]['cos_sim_std'] for g in gaps_plot],
                          [summary[g]['cos_sim'] + summary[g]['cos_sim_std'] for g in gaps_plot],
                          alpha=0.2)
    axes[0].set_xlabel('Temporal Gap (seconds)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Patch Similarity vs Temporal Gap')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.5, 1.0)

    axes[1].plot(times, mses, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Temporal Gap (seconds)')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Patch MSE vs Temporal Gap')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, l1s, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Temporal Gap (seconds)')
    axes[2].set_ylabel('L1 Distance')
    axes[2].set_title('Patch L1 vs Temporal Gap')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_vs_gap.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'similarity_vs_gap.png'}")

    # 2. CLS vs Spatial comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    cls_sims = [summary[g]['cls_sim'] for g in gaps_plot]
    spatial_sims = [summary[g]['spatial_sim'] for g in gaps_plot]

    x = np.arange(len(gaps_plot))
    width = 0.35

    ax.bar(x - width/2, cls_sims, width, label='CLS Token', color='blue', alpha=0.7)
    ax.bar(x + width/2, spatial_sims, width, label='Spatial Patches', color='orange', alpha=0.7)

    ax.set_xlabel('Temporal Gap (seconds)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('CLS vs Spatial Patch Similarity')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g/8:.2f}s' for g in gaps_plot])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cls_vs_spatial.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'cls_vs_spatial.png'}")

    # 3. Per-patch similarity heatmaps
    # DINOv2 with 256x256 gives 18x18 = 324 spatial patches (+ 1 CLS = 325)
    grid_size = int(np.sqrt(len(summary[gaps_plot[0]]['per_patch_sim']) - 1))
    print(f"Detected grid size: {grid_size}x{grid_size}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, gap in enumerate(gaps_plot[:6]):
        ax = axes[idx // 3, idx % 3]

        per_patch = summary[gap]['per_patch_sim'][1:]  # Remove CLS
        sim_map = per_patch.reshape(grid_size, grid_size)

        im = ax.imshow(sim_map, cmap='RdYlGn', vmin=0.6, vmax=1.0)
        ax.set_title(f'Gap = {gap/8:.2f}s')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Per-Patch Cosine Similarity Across Temporal Gaps', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_patch_heatmaps.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'per_patch_heatmaps.png'}")

    # 4. Per-patch MSE heatmaps (which regions change most)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, gap in enumerate(gaps_plot[:6]):
        ax = axes[idx // 3, idx % 3]

        per_patch_mse = summary[gap]['per_patch_mse'][1:]  # Remove CLS
        mse_map = per_patch_mse.reshape(grid_size, grid_size)

        im = ax.imshow(mse_map, cmap='hot')
        ax.set_title(f'Gap = {gap/8:.2f}s')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Per-Patch MSE (Where Change Happens)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'change_heatmaps.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'change_heatmaps.png'}")

    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("DINO Patch Analysis Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Videos analyzed: {len(all_results)}\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")

        if 1 in summary:
            f.write(f"Gap 1 (0.125s): {summary[1]['cos_sim']:.3f} cosine similarity\n")
        if 8 in summary:
            f.write(f"Gap 8 (1.0s):   {summary[8]['cos_sim']:.3f} cosine similarity\n")
        if 12 in summary:
            f.write(f"Gap 12 (1.5s):  {summary[12]['cos_sim']:.3f} cosine similarity\n")

        f.write(f"\nCLS token is {'MORE' if summary[8]['cls_sim'] > summary[8]['spatial_sim'] else 'LESS'} "
                f"stable than spatial patches\n")

        f.write("\n\nIMPLICATIONS FOR PATCH PREDICTION:\n")
        f.write("-" * 40 + "\n")

        # Calculate prediction difficulty
        baseline_mse = summary[8]['mse'] if 8 in summary else summary[list(summary.keys())[0]]['mse']
        f.write(f"Baseline MSE at 1s gap: {baseline_mse:.4f}\n")
        f.write(f"This is the 'copy previous' baseline loss\n")
        f.write(f"Model must beat this to show learning\n")

        f.write("\n\nPER-GAP STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Gap':<6} {'Time':<8} {'Cos Sim':<12} {'MSE':<12} {'L1':<12}\n")
        for gap in gaps_plot:
            s = summary[gap]
            f.write(f"{gap:<6} {gap/8:.2f}s    {s['cos_sim']:.4f}       {s['mse']:.4f}       {s['l1']:.4f}\n")

    print(f"Saved: {output_dir / 'summary.txt'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/d/projects/fVLM/data/precomputed/frames")
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--output_dir", type=str,
                        default="research/patch_prediction/results/01_analysis")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = run_analysis(args.data_dir, args.num_videos, device)
    summary = summarize_and_plot(results, args.output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
