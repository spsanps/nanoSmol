#!/usr/bin/env python3
"""
Ablation Experiments for Foveated VLM Fixes

Tests the following configurations to determine best path forward:
1. Baseline (current model)
2. Temperature = 0.1 (sharper attention)
3. Contrastive loss (push z_fine away from z_coarse)
4. Freeze DINO (preserve feature diversity)
5. Combined (all fixes together)

Each experiment runs for 500 steps to get initial signal.
GPU memory kept under 20GB.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import IterableDataset, DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from diffusers import AutoencoderKL
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import re
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str: str) -> int:
    """Parse ISO 8601 duration string."""
    try:
        match = re.match(r'PT(\d+)H(\d+)M(\d+)S', dur_str)
        if match:
            return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
        match = re.match(r'PT(\d+)M(\d+)S', dur_str)
        if match:
            return int(match[1]) * 60 + int(match[2])
        match = re.match(r'PT(\d+)S', dur_str)
        if match:
            return int(match[1])
    except:
        pass
    return 0


def download_video(url: str, timeout: int = 30) -> bytes:
    """Download video from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            return response.content
    except:
        pass
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames from video bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
        f.write(video_bytes)
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-i', f.name,
                '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                '-frames:v', str(num_frames * 4),
                '-q:v', '2',
                f'{tmpdir}/frame_%04d.jpg',
                '-y', '-loglevel', 'error'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                return None

            frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
            if len(frame_files) < num_frames:
                return None

            indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
            frames = []
            for idx in indices:
                img = Image.open(frame_files[idx]).convert('RGB')
                frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                frames.append(frame)
            return torch.stack(frames)


class SimpleStreamingDataset(IterableDataset):
    """Simple streaming dataset for ablation experiments."""

    def __init__(self, num_frames=8, frame_size=256, min_dur=5, max_dur=30):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.min_dur = min_dur
        self.max_dur = max_dur

    def __iter__(self):
        dataset = load_dataset(
            "TempoFunk/webvid-10M",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        for row in dataset:
            try:
                dur = parse_duration(row.get('duration', 'PT0S'))
                if not (self.min_dur <= dur <= self.max_dur):
                    continue

                url = row.get('contentUrl', '')
                if not url:
                    continue

                video_bytes = download_video(url)
                if video_bytes is None:
                    continue

                frames = extract_frames(video_bytes, self.num_frames, self.frame_size)
                if frames is None:
                    continue

                # Normalize frames
                frames = frames.float() / 255.0
                frames = (frames - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)

                yield {'frames': frames}

            except Exception:
                continue


def compute_feature_similarity(model, frames, device):
    """Compute cosine similarity between z_coarse and z_fine features."""
    B, T = frames.shape[:2]

    with torch.no_grad():
        frames_flat = frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = model.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Build per-frame caches
        all_caches = []
        if 'kv_cache' in cache_flat:
            num_layers = len(cache_flat['kv_cache'])
            for t in range(T):
                frame_kv_cache = []
                for layer_idx in range(num_layers):
                    layer_cache = cache_flat['kv_cache'][layer_idx]
                    K_all = layer_cache['K'].reshape(B, T, N, D)
                    V_all = layer_cache['V'].reshape(B, T, N, D)
                    frame_kv_cache.append({
                        'K': K_all[:, t], 'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': frame_kv_cache})
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Get coarse features
        q_static = model.q_static.expand(B, -1)
        z_coarse = torch.stack([model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)

        # Get fine features
        q_init = model.q_init.expand(B, -1)
        z_fine = torch.stack([model.encoder.query_attend(q_init, all_caches[t]) for t in range(T)], dim=1)

        # Compute similarity
        z_coarse_flat = z_coarse.reshape(-1, z_coarse.shape[-1]).float()
        z_fine_flat = z_fine.reshape(-1, z_fine.shape[-1]).float()
        cos_sim = F.cosine_similarity(z_coarse_flat, z_fine_flat, dim=-1).mean().item()

    return cos_sim


def run_experiment(config, num_steps=500, device='cuda'):
    """Run a single ablation experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'='*70}")
    print(f"Config: {config}")

    # Create model with config
    model = FoveatedVideoModel(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        dino_dim=384,
        llm_dim=576,
        query_dim=128,
        deep_query=True,
        attention_temperature=config.get('temperature', 1.0),
        lambda_contrastive=config.get('lambda_contrastive', 0.0),
        freeze_dino=config.get('freeze_dino', False),
    ).to(device)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_params:.1f}M, Trainable: {trainable_params:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=0.01
    )
    scaler = GradScaler()

    # Dataset
    dataset = SimpleStreamingDataset(num_frames=8, frame_size=256)
    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    # Training loop
    model.train()
    results = {
        'loss_fine': [],
        'loss_coarse': [],
        'ratio': [],
        'feature_sim': [],
    }

    data_iter = iter(loader)
    pbar = tqdm(range(num_steps), desc=config['name'])

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        except Exception as e:
            continue

        frames = batch['frames'].to(device)
        B, T = frames.shape[:2]

        # Compute VAE latents
        with torch.no_grad():
            frames_flat = frames.reshape(B * T, 3, 256, 256).half()
            latents_flat = vae.encode(frames_flat).latent_dist.sample() * 0.18215
            latents = latents_flat.reshape(B, T, 4, 32, 32).float()

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_embeds = model.get_empty_text_embeds(B).to(device)
            loss, loss_fine, loss_coarse = model(text_embeds, frames, latents)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        results['loss_fine'].append(loss_fine.item())
        results['loss_coarse'].append(loss_coarse.item())
        ratio = loss_coarse.item() / (loss_fine.item() + 1e-8)
        results['ratio'].append(ratio)

        # Compute feature similarity every 100 steps
        if step % 100 == 0:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                feat_sim = compute_feature_similarity(model, frames, device)
            results['feature_sim'].append(feat_sim)
            pbar.set_postfix({
                'fine': f'{loss_fine.item():.3f}',
                'coarse': f'{loss_coarse.item():.3f}',
                'ratio': f'{ratio:.3f}',
                'sim': f'{feat_sim:.3f}'
            })
        else:
            pbar.set_postfix({
                'fine': f'{loss_fine.item():.3f}',
                'coarse': f'{loss_coarse.item():.3f}',
                'ratio': f'{ratio:.3f}'
            })

    # Final metrics
    final_100 = slice(-100, None)
    summary = {
        'name': config['name'],
        'config': config,
        'avg_loss_fine': np.mean(results['loss_fine'][final_100]),
        'avg_loss_coarse': np.mean(results['loss_coarse'][final_100]),
        'avg_ratio': np.mean(results['ratio'][final_100]),
        'final_feature_sim': results['feature_sim'][-1] if results['feature_sim'] else None,
        'loss_fine_trend': results['loss_fine'],
        'ratio_trend': results['ratio'],
    }

    print(f"\nFinal Results ({config['name']}):")
    print(f"  Loss Fine:      {summary['avg_loss_fine']:.4f}")
    print(f"  Loss Coarse:    {summary['avg_loss_coarse']:.4f}")
    print(f"  Ratio:          {summary['avg_ratio']:.4f}")
    print(f"  Feature Sim:    {summary['final_feature_sim']:.4f}")

    # Clear GPU memory
    del model, vae, optimizer
    torch.cuda.empty_cache()

    return summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path('outputs/ablation_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ABLATION EXPERIMENTS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Define experiments
    experiments = [
        {
            'name': 'A_baseline',
            'temperature': 1.0,
            'lambda_contrastive': 0.0,
            'freeze_dino': False,
        },
        {
            'name': 'B_temp_0.1',
            'temperature': 0.1,
            'lambda_contrastive': 0.0,
            'freeze_dino': False,
        },
        {
            'name': 'C_contrastive',
            'temperature': 1.0,
            'lambda_contrastive': 0.1,
            'freeze_dino': False,
        },
        {
            'name': 'D_freeze_dino',
            'temperature': 1.0,
            'lambda_contrastive': 0.0,
            'freeze_dino': True,
        },
        {
            'name': 'E_combined',
            'temperature': 0.1,
            'lambda_contrastive': 0.1,
            'freeze_dino': True,
        },
    ]

    # Run experiments
    all_results = []
    for config in experiments:
        result = run_experiment(config, num_steps=500, device=device)
        all_results.append(result)

        # Save intermediate results
        with open(output_dir / 'results.json', 'w') as f:
            # Convert numpy arrays for JSON serialization
            serializable = []
            for r in all_results:
                r_copy = {k: v for k, v in r.items() if k not in ['loss_fine_trend', 'ratio_trend']}
                r_copy['loss_fine_trend'] = [float(x) for x in r['loss_fine_trend'][-50:]]
                r_copy['ratio_trend'] = [float(x) for x in r['ratio_trend'][-50:]]
                serializable.append(r_copy)
            json.dump(serializable, f, indent=2)

    # Generate report
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Experiment':<20} {'Fine':<10} {'Coarse':<10} {'Ratio':<10} {'Feat Sim':<10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['name']:<20} {r['avg_loss_fine']:<10.4f} {r['avg_loss_coarse']:<10.4f} "
              f"{r['avg_ratio']:<10.4f} {r['final_feature_sim'] or 0:<10.4f}")

    # Identify best configuration
    # Best = highest ratio (coarse/fine > 1 means fine is better)
    best_ratio = max(all_results, key=lambda x: x['avg_ratio'])
    # Best feature diversity = lowest similarity
    best_diversity = min(all_results, key=lambda x: x['final_feature_sim'] or 1.0)

    print(f"\nBest by Ratio:      {best_ratio['name']} (ratio={best_ratio['avg_ratio']:.4f})")
    print(f"Best by Diversity:  {best_diversity['name']} (sim={best_diversity['final_feature_sim']:.4f})")

    # Generate markdown report
    report = f"""# Ablation Experiment Results

**Date:** {datetime.now().isoformat()}
**Steps per experiment:** 500

## Summary Table

| Experiment | Loss Fine | Loss Coarse | Ratio | Feature Sim |
|------------|-----------|-------------|-------|-------------|
"""
    for r in all_results:
        report += f"| {r['name']} | {r['avg_loss_fine']:.4f} | {r['avg_loss_coarse']:.4f} | {r['avg_ratio']:.4f} | {r['final_feature_sim'] or 0:.4f} |\n"

    report += f"""
## Best Configurations

- **Best by Ratio:** {best_ratio['name']} (ratio={best_ratio['avg_ratio']:.4f})
- **Best by Feature Diversity:** {best_diversity['name']} (sim={best_diversity['final_feature_sim']:.4f})

## Interpretation

- **Ratio > 1.0** means loss_fine < loss_coarse (our goal!)
- **Lower Feature Sim** means z_fine and z_coarse are more different (better)

## Configurations Tested

1. **A_baseline**: Current model (temp=1.0, no contrastive, train DINO)
2. **B_temp_0.1**: Sharper attention (temp=0.1)
3. **C_contrastive**: Push features apart (lambda=0.1)
4. **D_freeze_dino**: Preserve pretrained feature diversity
5. **E_combined**: All fixes together

## Recommendations

Based on these results:
"""

    if best_ratio['avg_ratio'] > 1.05:
        report += f"- {best_ratio['name']} shows ratio > 1.05, indicating fine beats coarse!\n"
    else:
        report += "- No configuration achieved ratio > 1.05. May need longer training or different approach.\n"

    if best_diversity['final_feature_sim'] and best_diversity['final_feature_sim'] < 0.9:
        report += f"- {best_diversity['name']} reduced feature similarity to {best_diversity['final_feature_sim']:.2%}\n"
    else:
        report += "- Feature similarity remains high. Consider stronger contrastive loss.\n"

    report += "\n---\n*Generated by ablation_experiments.py*\n"

    with open(output_dir / 'ABLATION_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_dir / 'ABLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
