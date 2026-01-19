#!/usr/bin/env python3
"""
Diagnostic 01: Understanding DINO Patch Space

Questions to answer:
1. How similar are patches across temporal gaps?
2. Which patches change most (moving objects vs static background)?
3. What's the baseline difficulty of patch prediction?
4. Does patch similarity correlate with spatial position?

This will inform our prediction architecture design.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from datasets import load_dataset
import requests
import subprocess
import tempfile
from PIL import Image
import re
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(device):
    """Load frozen DINO for patch extraction."""
    print("Loading DINO...", flush=True)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    return dino


def extract_patches(dino, frames, device):
    """Extract DINO patches from frames.

    Args:
        frames: [T, C, H, W] normalized frames
    Returns:
        patches: [T, 257, 384] (CLS + 256 spatial patches)
    """
    frames = frames.to(device)
    with torch.no_grad():
        outputs = dino(frames, output_hidden_states=True)
        patches = outputs.last_hidden_state  # [T, 257, 384]
    return patches


def parse_duration(dur_str):
    try:
        match = re.match(r'PT(\d+)M(\d+)S', dur_str)
        if match:
            return int(match[1]) * 60 + int(match[2])
        match = re.match(r'PT(\d+)S', dur_str)
        if match:
            return int(match[1])
    except:
        pass
    return 0


def download_video(url, timeout=20):
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            content = b''
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 50 * 1024 * 1024:
                    break
            return content
    except:
        pass
    return None


def extract_frames(video_bytes, num_frames=32, size=256):
    """Extract frames uniformly from video."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            # Get duration
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0', f.name
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip())

            if duration < 4:  # Need at least 4 seconds
                return None

            with tempfile.TemporaryDirectory() as tmpdir:
                frames = []
                for i in range(num_frames):
                    t = 0.5 + (duration - 1) * i / (num_frames - 1)
                    cmd = [
                        'ffmpeg', '-ss', str(t), '-i', f.name,
                        '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                        '-frames:v', '1', '-q:v', '2',
                        f'{tmpdir}/frame_{i:04d}.jpg',
                        '-y', '-loglevel', 'error'
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=10)

                    frame_path = Path(tmpdir) / f'frame_{i:04d}.jpg'
                    if not frame_path.exists():
                        return None

                    img = Image.open(frame_path).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)

                frames = torch.stack(frames)
                # Normalize for DINO
                frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
                return frames
    except Exception as e:
        return None


def analyze_patch_similarity(patches, gaps=[1, 4, 8, 16, 24]):
    """Analyze how patches change across different temporal gaps.

    Args:
        patches: [T, 257, 384]
        gaps: list of frame gaps to analyze

    Returns:
        dict with analysis results
    """
    T = patches.shape[0]
    results = {}

    for gap in gaps:
        if gap >= T:
            continue

        # Compare patches at different gaps
        patch_t = patches[:-gap]  # [T-gap, 257, 384]
        patch_t_plus_gap = patches[gap:]  # [T-gap, 257, 384]

        # Cosine similarity per patch position
        cos_sim = F.cosine_similarity(
            patch_t.reshape(-1, 384),
            patch_t_plus_gap.reshape(-1, 384),
            dim=-1
        ).reshape(-1, 257)  # [T-gap, 257]

        # MSE per patch
        mse = ((patch_t - patch_t_plus_gap) ** 2).mean(dim=-1)  # [T-gap, 257]

        results[gap] = {
            'cos_sim_mean': cos_sim.mean().item(),
            'cos_sim_std': cos_sim.std().item(),
            'cos_sim_per_patch': cos_sim.mean(dim=0).cpu().numpy(),  # [257]
            'mse_mean': mse.mean().item(),
            'mse_std': mse.std().item(),
            'mse_per_patch': mse.mean(dim=0).cpu().numpy(),  # [257]
            # CLS token specifically
            'cls_cos_sim': cos_sim[:, 0].mean().item(),
            'cls_mse': mse[:, 0].mean().item(),
            # Spatial patches only
            'spatial_cos_sim': cos_sim[:, 1:].mean().item(),
            'spatial_mse': mse[:, 1:].mean().item(),
        }

    return results


def analyze_which_patches_change(patches, gap=8):
    """Find which patches change most across the gap.

    Returns spatial map of patch change magnitude.
    """
    T = patches.shape[0]
    if gap >= T:
        return None

    patch_t = patches[:-gap, 1:]  # Remove CLS, [T-gap, 256, 384]
    patch_t_plus_gap = patches[gap:, 1:]

    # Change magnitude per spatial position
    change = ((patch_t - patch_t_plus_gap) ** 2).mean(dim=(0, 2))  # [256]

    # Reshape to 16x16 spatial grid
    change_map = change.reshape(16, 16).cpu().numpy()

    return change_map


def run_analysis(num_videos=10, device='cuda'):
    """Run patch analysis on multiple videos."""

    print("=" * 60)
    print("DINO PATCH ANALYSIS")
    print("=" * 60)

    dino = load_dino(device)

    # Stream WebVid
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    all_results = []
    videos_processed = 0
    samples_checked = 0

    for sample in ds:
        samples_checked += 1

        if videos_processed >= num_videos:
            break

        duration = parse_duration(sample.get('duration', ''))
        if duration < 8 or duration > 30:
            continue

        url = sample.get('contentUrl')
        if not url:
            continue

        video_bytes = download_video(url)
        if video_bytes is None:
            continue

        frames = extract_frames(video_bytes, num_frames=32)
        if frames is None:
            continue

        # Extract patches
        patches = extract_patches(dino, frames, device)

        # Analyze
        results = analyze_patch_similarity(patches)
        change_map = analyze_which_patches_change(patches, gap=8)

        all_results.append({
            'video_id': videos_processed,
            'similarity': results,
            'change_map': change_map,
        })

        videos_processed += 1
        print(f"Processed video {videos_processed}/{num_videos} (checked {samples_checked})", flush=True)

    return all_results


def summarize_results(all_results, output_dir):
    """Create summary statistics and plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaps = [1, 4, 8, 16, 24]

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\n{'Gap':<8} {'Cos Sim':<12} {'MSE':<12} {'CLS Sim':<12} {'Spatial Sim':<12}")
    print("-" * 60)

    summary = {}
    for gap in gaps:
        cos_sims = [r['similarity'][gap]['cos_sim_mean'] for r in all_results if gap in r['similarity']]
        mses = [r['similarity'][gap]['mse_mean'] for r in all_results if gap in r['similarity']]
        cls_sims = [r['similarity'][gap]['cls_cos_sim'] for r in all_results if gap in r['similarity']]
        spatial_sims = [r['similarity'][gap]['spatial_cos_sim'] for r in all_results if gap in r['similarity']]

        if cos_sims:
            summary[gap] = {
                'cos_sim': np.mean(cos_sims),
                'mse': np.mean(mses),
                'cls_sim': np.mean(cls_sims),
                'spatial_sim': np.mean(spatial_sims),
            }
            print(f"{gap:<8} {np.mean(cos_sims):<12.4f} {np.mean(mses):<12.4f} "
                  f"{np.mean(cls_sims):<12.4f} {np.mean(spatial_sims):<12.4f}")

    # Plot 1: Similarity vs Gap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gaps_plot = list(summary.keys())
    cos_sims_plot = [summary[g]['cos_sim'] for g in gaps_plot]
    mses_plot = [summary[g]['mse'] for g in gaps_plot]

    axes[0].plot(gaps_plot, cos_sims_plot, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Temporal Gap (frames)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Patch Similarity vs Temporal Gap')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].plot(gaps_plot, mses_plot, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Temporal Gap (frames)')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Patch MSE vs Temporal Gap')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_vs_gap.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'similarity_vs_gap.png'}")

    # Plot 2: Average change map (which patches change most)
    change_maps = [r['change_map'] for r in all_results if r['change_map'] is not None]
    if change_maps:
        avg_change_map = np.mean(change_maps, axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(avg_change_map, cmap='hot')
        ax.set_title('Average Patch Change Magnitude (gap=8)\n(brighter = more change)')
        plt.colorbar(im, ax=ax, label='MSE')
        plt.savefig(output_dir / 'change_heatmap.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'change_heatmap.png'}")

    # Plot 3: Per-patch similarity for different gaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for idx, gap in enumerate([1, 8, 16, 24]):
        if gap not in summary:
            continue
        ax = axes[idx // 2, idx % 2]

        # Average per-patch similarity
        per_patch_sims = [r['similarity'][gap]['cos_sim_per_patch'] for r in all_results if gap in r['similarity']]
        avg_per_patch = np.mean(per_patch_sims, axis=0)[1:]  # Remove CLS

        sim_map = avg_per_patch.reshape(16, 16)
        im = ax.imshow(sim_map, cmap='RdYlGn', vmin=0.5, vmax=1.0)
        ax.set_title(f'Per-patch Cosine Similarity (gap={gap})')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_patch_similarity.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'per_patch_similarity.png'}")

    # Save raw numbers
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("DINO Patch Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Videos analyzed: {len(all_results)}\n\n")
        f.write(f"{'Gap':<8} {'Cos Sim':<12} {'MSE':<12} {'CLS Sim':<12} {'Spatial Sim':<12}\n")
        f.write("-" * 60 + "\n")
        for gap, stats in summary.items():
            f.write(f"{gap:<8} {stats['cos_sim']:<12.4f} {stats['mse']:<12.4f} "
                    f"{stats['cls_sim']:<12.4f} {stats['spatial_sim']:<12.4f}\n")

        f.write("\n\nKey Findings:\n")
        if gaps_plot:
            f.write(f"- Gap 1: {summary[1]['cos_sim']:.3f} similarity (very high - adjacent frames similar)\n")
            if 8 in summary:
                f.write(f"- Gap 8: {summary[8]['cos_sim']:.3f} similarity (~1 second at 8fps)\n")
            if 16 in summary:
                f.write(f"- Gap 16: {summary[16]['cos_sim']:.3f} similarity (~2 seconds)\n")
            if 24 in summary:
                f.write(f"- Gap 24: {summary[24]['cos_sim']:.3f} similarity (~3 seconds)\n")

    print(f"Saved: {output_dir / 'summary.txt'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="research/patch_prediction/results/01_analysis")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_analysis(num_videos=args.num_videos, device=device)
    summary = summarize_results(results, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey questions answered:")
    print("1. How similar are patches across temporal gaps? → See similarity_vs_gap.png")
    print("2. Which patches change most? → See change_heatmap.png")
    print("3. Is change uniform or localized? → See per_patch_similarity.png")
