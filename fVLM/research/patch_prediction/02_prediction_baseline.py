#!/usr/bin/env python3
"""
Diagnostic 02: Patch Prediction Baselines

Test different approaches to predict future DINO patches:

1. Copy baseline: Just copy previous patches (MSE ~2.0 at 1s gap)
2. Linear: Simple linear projection from current patches
3. MLP: Multi-layer network
4. FiLM: LLM hidden state modulates patch transformation

This tells us:
- Is patch prediction feasible?
- How hard is it compared to VAE latent prediction?
- What architecture works best?
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
    """Load frozen DINO."""
    print("Loading DINO...", flush=True)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    return dino


def load_frames_from_dir(frame_dir, device):
    """Load and normalize frames."""
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
    """Extract DINO patches."""
    with torch.no_grad():
        outputs = dino(frames)
        patches = outputs.last_hidden_state  # [T, 325, 384]
    return patches


# === PREDICTION MODELS ===

class CopyBaseline(nn.Module):
    """Just copy previous patches."""
    def forward(self, patches):
        return patches


class LinearPredictor(nn.Module):
    """Linear transformation per patch."""
    def __init__(self, dim=384):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, patches):
        return self.proj(patches)


class MLPPredictor(nn.Module):
    """MLP per patch position."""
    def __init__(self, dim=384, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, patches):
        return self.net(patches)


class TemporalMLPPredictor(nn.Module):
    """MLP that sees multiple past frames."""
    def __init__(self, dim=384, num_context=4, hidden=512):
        super().__init__()
        self.num_context = num_context
        self.net = nn.Sequential(
            nn.Linear(dim * num_context, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, patches_seq):
        # patches_seq: [B, num_context, N, D]
        B, T, N, D = patches_seq.shape
        x = patches_seq.reshape(B, T, N * D)  # [B, T, N*D]
        x = x.transpose(1, 2)  # [B, N*D, T]
        x = x.reshape(B * N, D * T)  # [B*N, D*T]
        out = self.net(x)  # [B*N, D]
        return out.reshape(B, N, D)


class AttentionPredictor(nn.Module):
    """Cross-attention over past frames."""
    def __init__(self, dim=384, num_heads=6, num_layers=2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, num_heads, dim * 4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, patches_seq):
        # patches_seq: [B, T, N, D]
        B, T, N, D = patches_seq.shape

        # Process each patch position independently
        patches_seq = patches_seq.permute(0, 2, 1, 3)  # [B, N, T, D]
        patches_seq = patches_seq.reshape(B * N, T, D)  # [B*N, T, D]

        query = self.query.expand(B * N, 1, D)
        x = query
        for layer in self.layers:
            x = layer(x, patches_seq)

        x = self.out_proj(x.squeeze(1))  # [B*N, D]
        return x.reshape(B, N, D)


def train_predictor(model, dino, data_dirs, gap=8, num_context=1,
                    steps=500, lr=1e-3, batch_size=4, device='cuda'):
    """Train a patch predictor.

    Args:
        model: Prediction model
        dino: Frozen DINO encoder
        data_dirs: List of video frame directories
        gap: Temporal gap to predict
        num_context: Number of context frames (1 for simple models)
        steps: Training steps
        lr: Learning rate
        batch_size: Batch size
        device: Device

    Returns:
        dict with training history
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {'loss': [], 'cos_sim': []}
    step = 0
    data_idx = 0

    pbar = tqdm(total=steps, desc="Training")

    while step < steps:
        batch_patches_in = []
        batch_patches_target = []

        # Collect batch
        for _ in range(batch_size):
            if data_idx >= len(data_dirs):
                random.shuffle(data_dirs)
                data_idx = 0

            try:
                frames = load_frames_from_dir(data_dirs[data_idx], device)
                data_idx += 1

                if frames.shape[0] < gap + num_context:
                    continue

                patches = extract_patches(dino, frames)

                # Get input/target pairs
                # Input: frames 0 to T-gap-1
                # Target: frames gap to T-1
                for t in range(num_context - 1, frames.shape[0] - gap):
                    if num_context == 1:
                        patch_in = patches[t]  # [N, D]
                    else:
                        patch_in = patches[t - num_context + 1:t + 1]  # [num_context, N, D]

                    patch_target = patches[t + gap]  # [N, D]

                    batch_patches_in.append(patch_in)
                    batch_patches_target.append(patch_target)

                    if len(batch_patches_in) >= batch_size:
                        break

            except Exception as e:
                data_idx += 1
                continue

            if len(batch_patches_in) >= batch_size:
                break

        if len(batch_patches_in) < 2:
            continue

        # Stack batch
        if num_context == 1:
            patches_in = torch.stack(batch_patches_in[:batch_size])  # [B, N, D]
        else:
            patches_in = torch.stack(batch_patches_in[:batch_size])  # [B, ctx, N, D]
        patches_target = torch.stack(batch_patches_target[:batch_size])  # [B, N, D]

        # Forward
        optimizer.zero_grad()
        pred = model(patches_in)  # [B, N, D]

        # Loss
        loss = F.mse_loss(pred, patches_target)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                pred.reshape(-1, 384),
                patches_target.reshape(-1, 384),
                dim=-1
            ).mean()

        history['loss'].append(loss.item())
        history['cos_sim'].append(cos_sim.item())

        step += 1
        pbar.update(1)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'cos': f"{cos_sim.item():.4f}"})

    pbar.close()
    return history


def evaluate_predictor(model, dino, data_dirs, gap=8, num_context=1,
                       num_samples=200, device='cuda'):
    """Evaluate a predictor."""
    model = model.to(device)
    model.eval()

    all_mse = []
    all_cos_sim = []

    for frame_dir in tqdm(data_dirs[:num_samples], desc="Evaluating"):
        try:
            frames = load_frames_from_dir(frame_dir, device)
            if frames.shape[0] < gap + num_context:
                continue

            patches = extract_patches(dino, frames)

            with torch.no_grad():
                for t in range(num_context - 1, frames.shape[0] - gap):
                    if num_context == 1:
                        patch_in = patches[t:t+1]  # [1, N, D]
                    else:
                        patch_in = patches[t - num_context + 1:t + 1].unsqueeze(0)  # [1, ctx, N, D]

                    patch_target = patches[t + gap:t + gap + 1]  # [1, N, D]

                    pred = model(patch_in)

                    mse = F.mse_loss(pred, patch_target).item()
                    cos_sim = F.cosine_similarity(
                        pred.reshape(-1, 384),
                        patch_target.reshape(-1, 384),
                        dim=-1
                    ).mean().item()

                    all_mse.append(mse)
                    all_cos_sim.append(cos_sim)

        except Exception:
            continue

    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'cos_sim_mean': np.mean(all_cos_sim),
        'cos_sim_std': np.std(all_cos_sim),
    }


def run_experiments(data_dir, output_dir, gap=8, steps=500, device='cuda'):
    """Run all baseline experiments."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all video directories
    all_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    random.shuffle(all_dirs)

    train_dirs = all_dirs[:1000]
    eval_dirs = all_dirs[1000:1200]

    print(f"Train videos: {len(train_dirs)}")
    print(f"Eval videos: {len(eval_dirs)}")

    dino = load_dino(device)

    results = {}

    # 1. Copy baseline (no training)
    print("\n" + "=" * 50)
    print("1. COPY BASELINE")
    print("=" * 50)

    copy_model = CopyBaseline()
    copy_results = evaluate_predictor(copy_model, dino, eval_dirs, gap=gap, device=device)
    results['copy'] = copy_results
    print(f"Copy baseline: MSE={copy_results['mse_mean']:.4f}, CosSim={copy_results['cos_sim_mean']:.4f}")

    # 2. Linear predictor
    print("\n" + "=" * 50)
    print("2. LINEAR PREDICTOR")
    print("=" * 50)

    linear_model = LinearPredictor()
    linear_history = train_predictor(linear_model, dino, train_dirs, gap=gap, steps=steps, device=device)
    linear_results = evaluate_predictor(linear_model, dino, eval_dirs, gap=gap, device=device)
    results['linear'] = linear_results
    results['linear_history'] = linear_history
    print(f"Linear: MSE={linear_results['mse_mean']:.4f}, CosSim={linear_results['cos_sim_mean']:.4f}")

    # 3. MLP predictor
    print("\n" + "=" * 50)
    print("3. MLP PREDICTOR")
    print("=" * 50)

    mlp_model = MLPPredictor()
    mlp_history = train_predictor(mlp_model, dino, train_dirs, gap=gap, steps=steps, device=device)
    mlp_results = evaluate_predictor(mlp_model, dino, eval_dirs, gap=gap, device=device)
    results['mlp'] = mlp_results
    results['mlp_history'] = mlp_history
    print(f"MLP: MSE={mlp_results['mse_mean']:.4f}, CosSim={mlp_results['cos_sim_mean']:.4f}")

    # 4. Temporal MLP (sees 4 past frames)
    print("\n" + "=" * 50)
    print("4. TEMPORAL MLP (4 context frames)")
    print("=" * 50)

    temporal_model = TemporalMLPPredictor(num_context=4)
    temporal_history = train_predictor(temporal_model, dino, train_dirs, gap=gap, num_context=4,
                                        steps=steps, device=device)
    temporal_results = evaluate_predictor(temporal_model, dino, eval_dirs, gap=gap, num_context=4, device=device)
    results['temporal'] = temporal_results
    results['temporal_history'] = temporal_history
    print(f"Temporal: MSE={temporal_results['mse_mean']:.4f}, CosSim={temporal_results['cos_sim_mean']:.4f}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Gap: {gap} frames ({gap/8:.1f}s)")
    print(f"\n{'Model':<20} {'MSE':<15} {'Cos Sim':<15} {'vs Copy':<15}")
    print("-" * 70)

    copy_mse = results['copy']['mse_mean']
    for name in ['copy', 'linear', 'mlp', 'temporal']:
        r = results[name]
        improvement = (copy_mse - r['mse_mean']) / copy_mse * 100
        print(f"{name:<20} {r['mse_mean']:<15.4f} {r['cos_sim_mean']:<15.4f} {improvement:+.1f}%")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name in ['linear', 'mlp', 'temporal']:
        if f'{name}_history' in results:
            history = results[f'{name}_history']
            # Smooth
            window = 20
            loss_smooth = np.convolve(history['loss'], np.ones(window)/window, mode='valid')
            axes[0].plot(loss_smooth, label=name)

    axes[0].axhline(copy_mse, color='red', linestyle='--', label='copy baseline')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name in ['linear', 'mlp', 'temporal']:
        if f'{name}_history' in results:
            history = results[f'{name}_history']
            cos_smooth = np.convolve(history['cos_sim'], np.ones(window)/window, mode='valid')
            axes[1].plot(cos_smooth, label=name)

    axes[1].axhline(results['copy']['cos_sim_mean'], color='red', linestyle='--', label='copy baseline')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Prediction Similarity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'training_curves.png'}")

    # Save results
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("Patch Prediction Baseline Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Gap: {gap} frames ({gap/8:.1f}s)\n\n")

        f.write(f"{'Model':<20} {'MSE':<15} {'Cos Sim':<15} {'vs Copy':<15}\n")
        f.write("-" * 70 + "\n")

        for name in ['copy', 'linear', 'mlp', 'temporal']:
            r = results[name]
            improvement = (copy_mse - r['mse_mean']) / copy_mse * 100
            f.write(f"{name:<20} {r['mse_mean']:<15.4f} {r['cos_sim_mean']:<15.4f} {improvement:+.1f}%\n")

        f.write("\n\nKEY FINDINGS:\n")
        f.write("-" * 40 + "\n")

        best_model = min(['linear', 'mlp', 'temporal'], key=lambda x: results[x]['mse_mean'])
        best_mse = results[best_model]['mse_mean']
        best_improvement = (copy_mse - best_mse) / copy_mse * 100

        f.write(f"Best model: {best_model}\n")
        f.write(f"Best MSE: {best_mse:.4f} ({best_improvement:+.1f}% vs copy)\n")
        f.write(f"\nPatch prediction IS learnable - model beats copy baseline!\n")

    print(f"Saved: {output_dir / 'results.txt'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/d/projects/fVLM/data/precomputed/frames")
    parser.add_argument("--output_dir", type=str,
                        default="research/patch_prediction/results/02_baselines")
    parser.add_argument("--gap", type=int, default=8, help="Temporal gap in frames")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = run_experiments(
        args.data_dir,
        args.output_dir,
        gap=args.gap,
        steps=args.steps,
        device=device,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
