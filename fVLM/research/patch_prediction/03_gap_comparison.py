#!/usr/bin/env python3
"""
Diagnostic 03: Prediction Difficulty vs Temporal Gap

Key question: At what temporal gap does prediction become easier than copying?

For short gaps (1s): patches very similar → copying wins
For long gaps (3s+): patches very different → model can learn useful patterns?

This will inform the optimal temporal gap for training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(device):
    print("Loading DINO...", flush=True)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    return dino


def load_frames_from_dir(frame_dir, device):
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    frames = []
    for fp in frame_paths:
        img = Image.open(fp).convert('RGB')
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
        frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    frames = torch.stack(frames)
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames.to(device)


def extract_patches(dino, frames):
    with torch.no_grad():
        outputs = dino(frames)
        patches = outputs.last_hidden_state
    return patches


class MLPPredictor(nn.Module):
    """MLP with residual connection."""
    def __init__(self, dim=384, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, patches):
        delta = self.net(patches)
        # Blend: weighted combination of input and delta
        return self.residual_weight * patches + (1 - self.residual_weight) * delta


def train_and_evaluate(model, dino, train_dirs, eval_dirs, gap, steps=300, device='cuda'):
    """Train model for specific gap and evaluate."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    model.train()
    data_idx = 0
    random.shuffle(train_dirs)

    for step in range(steps):
        if data_idx >= len(train_dirs):
            random.shuffle(train_dirs)
            data_idx = 0

        try:
            frames = load_frames_from_dir(train_dirs[data_idx], device)
            data_idx += 1

            if frames.shape[0] < gap + 1:
                continue

            patches = extract_patches(dino, frames)

            # Get pairs
            t = random.randint(0, frames.shape[0] - gap - 1)
            patch_in = patches[t:t+1]  # [1, N, D]
            patch_target = patches[t + gap:t + gap + 1]  # [1, N, D]

            optimizer.zero_grad()
            pred = model(patch_in)
            loss = F.mse_loss(pred, patch_target)
            loss.backward()
            optimizer.step()

        except Exception:
            data_idx += 1
            continue

    # Evaluate
    model.eval()
    copy_mse = []
    model_mse = []
    copy_cos = []
    model_cos = []

    for frame_dir in eval_dirs[:100]:
        try:
            frames = load_frames_from_dir(frame_dir, device)
            if frames.shape[0] < gap + 1:
                continue

            patches = extract_patches(dino, frames)

            with torch.no_grad():
                for t in range(frames.shape[0] - gap):
                    patch_in = patches[t:t+1]
                    patch_target = patches[t + gap:t + gap + 1]

                    # Copy baseline
                    copy_mse.append(F.mse_loss(patch_in, patch_target).item())
                    copy_cos.append(F.cosine_similarity(
                        patch_in.reshape(-1, 384),
                        patch_target.reshape(-1, 384), dim=-1
                    ).mean().item())

                    # Model prediction
                    pred = model(patch_in)
                    model_mse.append(F.mse_loss(pred, patch_target).item())
                    model_cos.append(F.cosine_similarity(
                        pred.reshape(-1, 384),
                        patch_target.reshape(-1, 384), dim=-1
                    ).mean().item())

        except Exception:
            continue

    return {
        'gap': gap,
        'copy_mse': np.mean(copy_mse),
        'model_mse': np.mean(model_mse),
        'copy_cos': np.mean(copy_cos),
        'model_cos': np.mean(model_cos),
        'improvement': (np.mean(copy_mse) - np.mean(model_mse)) / np.mean(copy_mse) * 100,
    }


def run_gap_comparison(data_dir, output_dir, gaps=[2, 4, 8, 12], device='cuda'):
    """Compare prediction difficulty across temporal gaps."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    random.shuffle(all_dirs)

    train_dirs = all_dirs[:1000]
    eval_dirs = all_dirs[1000:1200]

    dino = load_dino(device)

    results = []

    for gap in gaps:
        print(f"\n{'='*50}")
        print(f"Testing gap = {gap} frames ({gap/8:.2f}s)")
        print(f"{'='*50}")

        model = MLPPredictor()
        result = train_and_evaluate(model, dino, train_dirs, eval_dirs, gap, device=device)
        results.append(result)

        print(f"  Copy MSE:  {result['copy_mse']:.4f}")
        print(f"  Model MSE: {result['model_mse']:.4f}")
        print(f"  Improvement: {result['improvement']:+.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("GAP COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Gap':<8} {'Time':<10} {'Copy MSE':<12} {'Model MSE':<12} {'Improve':<12} {'Model Wins?':<12}")
    print("-" * 70)

    for r in results:
        wins = "YES ✓" if r['improvement'] > 0 else "NO"
        print(f"{r['gap']:<8} {r['gap']/8:.2f}s      {r['copy_mse']:<12.4f} {r['model_mse']:<12.4f} "
              f"{r['improvement']:+.1f}%       {wins}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    gaps_plot = [r['gap'] for r in results]
    times = [g/8 for g in gaps_plot]
    copy_mses = [r['copy_mse'] for r in results]
    model_mses = [r['model_mse'] for r in results]

    axes[0].plot(times, copy_mses, 'o-', label='Copy Baseline', linewidth=2, markersize=8)
    axes[0].plot(times, model_mses, 's-', label='MLP Predictor', linewidth=2, markersize=8)
    axes[0].set_xlabel('Temporal Gap (seconds)')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Prediction MSE vs Temporal Gap')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    improvements = [r['improvement'] for r in results]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1].bar(times, improvements, color=colors, alpha=0.7, width=0.1)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Temporal Gap (seconds)')
    axes[1].set_ylabel('Improvement over Copy (%)')
    axes[1].set_title('Model Improvement vs Gap')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'gap_comparison.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'gap_comparison.png'}")

    # Save results
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("Gap Comparison Results\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Gap':<8} {'Time':<10} {'Copy MSE':<12} {'Model MSE':<12} {'Improvement':<12}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['gap']:<8} {r['gap']/8:.2f}s      {r['copy_mse']:<12.4f} "
                    f"{r['model_mse']:<12.4f} {r['improvement']:+.1f}%\n")

        f.write("\n\nKEY INSIGHT:\n")
        f.write("-" * 40 + "\n")

        # Find crossover point
        positive_improvements = [r for r in results if r['improvement'] > 0]
        if positive_improvements:
            best = max(results, key=lambda r: r['improvement'])
            f.write(f"Best gap: {best['gap']} frames ({best['gap']/8:.2f}s)\n")
            f.write(f"Model beats copy by {best['improvement']:.1f}%\n")
        else:
            f.write("Copy baseline wins at ALL tested gaps!\n")
            f.write("Patch prediction from single frame is extremely hard.\n")
            f.write("Need conditioning (LLM hidden state) or attention guidance.\n")

    print(f"Saved: {output_dir / 'results.txt'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/d/projects/fVLM/data/precomputed/frames")
    parser.add_argument("--output_dir", type=str,
                        default="research/patch_prediction/results/03_gaps")
    parser.add_argument("--gaps", type=str, default="2,4,8,12",
                        help="Comma-separated gaps to test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaps = [int(g) for g in args.gaps.split(",")]

    results = run_gap_comparison(args.data_dir, args.output_dir, gaps=gaps, device=device)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
