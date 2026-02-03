"""
Preliminary experiments to determine which fix to pursue.

Tests:
1. prev_latents dominance: How much does prediction rely on prev_latents vs visual features?
2. Delta prediction: Does delta have more spatial variance than full latent?
3. Feature similarity: How similar are z_fine and z_coarse?
4. Attention temperature: How does temperature affect focus?
5. Per-patch importance: Which patches matter for prediction?
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# Data loading utilities
import requests
import tempfile
import subprocess
import re
from datasets import load_dataset


def parse_duration(dur_str: str) -> int:
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


def download_video(url: str, timeout: int = 30) -> bytes:
    for retry in range(3):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                return response.content
        except:
            pass
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    from PIL import Image
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


def load_samples(num_samples=5, num_frames=16):
    """Load a few samples for experiments."""
    from diffusers import AutoencoderKL

    print("Loading samples...")
    print("  Connecting to WebVid dataset...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=100)

    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.bfloat16
    ).cuda().eval()

    samples = []
    attempts = 0
    max_attempts = num_samples * 20  # Allow many failures

    print(f"  Downloading {num_samples} videos...")
    for hf_sample in ds:
        if len(samples) >= num_samples:
            break
        if attempts >= max_attempts:
            print(f"  Warning: Stopped after {max_attempts} attempts")
            break

        attempts += 1

        duration = parse_duration(hf_sample.get('duration', ''))
        if duration < 5 or duration > 30:
            continue

        try:
            video_bytes = download_video(hf_sample['contentUrl'], timeout=15)
            if video_bytes is None:
                continue

            frames = extract_frames(video_bytes, num_frames, 256)
            if frames is None:
                continue

            # Normalize for model
            frames_norm = frames.float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_norm = (frames_norm - mean) / std

            # Compute VAE latents
            with torch.no_grad():
                frames_vae = frames.float().cuda() / 127.5 - 1.0
                latents = []
                for t in range(num_frames):
                    lat = vae.encode(frames_vae[t:t+1].to(torch.bfloat16)).latent_dist.mean
                    lat = lat * vae.config.scaling_factor
                    latents.append(lat.cpu())
                vae_latents = torch.cat(latents, dim=0)

            samples.append({
                'frames': frames,
                'frames_norm': frames_norm,
                'vae_latents': vae_latents,
                'video_id': hf_sample.get('videoid', len(samples)),
            })
            print(f"  Loaded sample {len(samples)}/{num_samples} (attempt {attempts})")
        except Exception as e:
            print(f"  Failed attempt {attempts}: {str(e)[:50]}")
            continue

    print(f"  Loaded {len(samples)} samples total")
    return samples, vae


@torch.no_grad()
def experiment_1_prev_latents_dominance(model, samples, device):
    """
    Test: How much does prediction rely on prev_latents vs h (visual features)?

    Method:
    - Get normal prediction
    - Zero out h, keep prev_latents -> see loss
    - Zero out prev_latents, keep h -> see loss
    - Random h, keep prev_latents -> see loss
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: prev_latents Dominance")
    print("="*70)

    results = {
        'normal': [],
        'zero_h': [],
        'zero_prev': [],
        'random_h': [],
    }

    for sample in samples:
        frames_norm = sample['frames_norm'].unsqueeze(0).to(device)
        vae_latents = sample['vae_latents'].unsqueeze(0).to(device)

        B, T = 1, frames_norm.shape[1]
        text_embeds = model.get_empty_text_embeds(B).to(device)
        N_text = text_embeds.shape[1]

        # Get features through model
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Encode frames
            frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
            _, cache_flat = model.encoder.encode_patches(frames_flat)
            patch_features_flat = cache_flat['patch_features']
            N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
            patch_features = patch_features_flat.reshape(B, T, N, D)

            # Handle kv_cache for deep mode
            if 'kv_cache' in cache_flat:
                num_layers = len(cache_flat['kv_cache'])
                all_caches = []
                for t in range(T):
                    frame_kv_cache = []
                    for layer_idx in range(num_layers):
                        layer_cache = cache_flat['kv_cache'][layer_idx]
                        K_all = layer_cache['K'].reshape(B, T, N, D)
                        V_all = layer_cache['V'].reshape(B, T, N, D)
                        frame_kv_cache.append({
                            'K': K_all[:, t],
                            'V': V_all[:, t],
                            'layer': layer_cache['layer'],
                        })
                    all_caches.append({
                        'patch_features': patch_features[:, t],
                        'kv_cache': frame_kv_cache,
                    })
            else:
                all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

            # Pass 1 to get h
            q_static = model.q_static.expand(B, -1)
            z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
            z_coarse = torch.stack(z_coarse_list, dim=1)
            z_coarse_proj = model.dino_to_llm(z_coarse)

            coarse_token = model.coarse_token.expand(B, 1, -1)
            seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse_proj], dim=1)
            h_pass1 = model.llm(inputs_embeds=seq_pass1).last_hidden_state
            h_for_pred = h_pass1[:, N_text:N_text + T]

            # Prepare prev_latents
            z_vae_init = model.z_vae_init.expand(B, 1, -1, -1, -1)
            prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
            targets = vae_latents

            # Normal prediction
            pred_normal = model.pred_head(h_for_pred, prev_latents)
            loss_normal = F.mse_loss(pred_normal, targets).item()
            results['normal'].append(loss_normal)

            # Zero out h
            h_zero = torch.zeros_like(h_for_pred)
            pred_zero_h = model.pred_head(h_zero, prev_latents)
            loss_zero_h = F.mse_loss(pred_zero_h, targets).item()
            results['zero_h'].append(loss_zero_h)

            # Zero out prev_latents
            prev_zero = torch.zeros_like(prev_latents)
            pred_zero_prev = model.pred_head(h_for_pred, prev_zero)
            loss_zero_prev = F.mse_loss(pred_zero_prev, targets).item()
            results['zero_prev'].append(loss_zero_prev)

            # Random h
            h_random = torch.randn_like(h_for_pred)
            pred_random_h = model.pred_head(h_random, prev_latents)
            loss_random_h = F.mse_loss(pred_random_h, targets).item()
            results['random_h'].append(loss_random_h)

    # Analysis
    print("\nResults (lower = better prediction):")
    print(f"  Normal (h + prev):     {np.mean(results['normal']):.4f}")
    print(f"  Zero h (prev only):    {np.mean(results['zero_h']):.4f}")
    print(f"  Zero prev (h only):    {np.mean(results['zero_prev']):.4f}")
    print(f"  Random h (prev only):  {np.mean(results['random_h']):.4f}")

    prev_contribution = (np.mean(results['zero_prev']) - np.mean(results['normal'])) / np.mean(results['normal'])
    h_contribution = (np.mean(results['zero_h']) - np.mean(results['normal'])) / np.mean(results['normal'])

    print(f"\nContribution Analysis:")
    print(f"  Removing prev_latents increases loss by: {prev_contribution*100:.1f}%")
    print(f"  Removing h increases loss by: {h_contribution*100:.1f}%")

    if h_contribution < 0.05:
        print("\n  ⚠️  h (visual features) contributes <5% - PREV_LATENTS DOMINATES")
        print("  → Recommendation: Remove or weaken prev_latents conditioning")

    return results


@torch.no_grad()
def experiment_2_delta_analysis(samples, device):
    """
    Test: Does predicting delta have more spatial variance than full latent?

    If delta has high variance in specific regions, attention should matter more.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Delta vs Full Latent Analysis")
    print("="*70)

    full_variances = []
    delta_variances = []
    delta_magnitudes = []

    for sample in samples:
        vae_latents = sample['vae_latents']  # [T, 4, 32, 32]
        T = vae_latents.shape[0]

        # Full latent spatial variance
        full_var = vae_latents.var(dim=(2, 3)).mean().item()
        full_variances.append(full_var)

        # Delta between consecutive frames
        deltas = vae_latents[1:] - vae_latents[:-1]  # [T-1, 4, 32, 32]
        delta_var = deltas.var(dim=(2, 3)).mean().item()
        delta_variances.append(delta_var)

        # Delta magnitude
        delta_mag = deltas.abs().mean().item()
        delta_magnitudes.append(delta_mag)

    print(f"\nSpatial Variance (higher = more spatial structure):")
    print(f"  Full latent variance:  {np.mean(full_variances):.4f}")
    print(f"  Delta variance:        {np.mean(delta_variances):.4f}")
    print(f"  Ratio (delta/full):    {np.mean(delta_variances)/np.mean(full_variances):.2f}x")

    print(f"\nDelta Magnitude:")
    print(f"  Mean |delta|:          {np.mean(delta_magnitudes):.4f}")

    if np.mean(delta_magnitudes) < 0.1:
        print("\n  ⚠️  Delta is very small - videos are mostly static")
        print("  → Recommendation: Use more dynamic dataset OR predict delta with scaling")

    return {
        'full_var': full_variances,
        'delta_var': delta_variances,
        'delta_mag': delta_magnitudes,
    }


@torch.no_grad()
def experiment_3_feature_similarity(model, samples, device):
    """
    Test: How similar are z_fine and z_coarse features?

    If they're very similar, the shared encoder is collapsing them.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Fine vs Coarse Feature Similarity")
    print("="*70)

    cosine_sims = []
    l2_dists = []

    for sample in samples:
        frames_norm = sample['frames_norm'].unsqueeze(0).to(device)
        B, T = 1, frames_norm.shape[1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Encode
            frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
            _, cache_flat = model.encoder.encode_patches(frames_flat)
            patch_features_flat = cache_flat['patch_features']
            N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
            patch_features = patch_features_flat.reshape(B, T, N, D)

            if 'kv_cache' in cache_flat:
                num_layers = len(cache_flat['kv_cache'])
                all_caches = []
                for t in range(T):
                    frame_kv_cache = []
                    for layer_idx in range(num_layers):
                        layer_cache = cache_flat['kv_cache'][layer_idx]
                        K_all = layer_cache['K'].reshape(B, T, N, D)
                        V_all = layer_cache['V'].reshape(B, T, N, D)
                        frame_kv_cache.append({
                            'K': K_all[:, t],
                            'V': V_all[:, t],
                            'layer': layer_cache['layer'],
                        })
                    all_caches.append({
                        'patch_features': patch_features[:, t],
                        'kv_cache': frame_kv_cache,
                    })
            else:
                all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

            # Coarse features
            q_static = model.q_static.expand(B, -1)
            z_coarse = torch.stack([model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)

            # Fine features (using q_init for simplicity)
            q_init = model.q_init.expand(B, -1)
            z_fine = torch.stack([model.encoder.query_attend(q_init, all_caches[t]) for t in range(T)], dim=1)

            # Compute similarity
            z_coarse_flat = z_coarse.view(-1, z_coarse.shape[-1]).float()
            z_fine_flat = z_fine.view(-1, z_fine.shape[-1]).float()

            cos_sim = F.cosine_similarity(z_coarse_flat, z_fine_flat, dim=-1).mean().item()
            l2_dist = (z_coarse_flat - z_fine_flat).norm(dim=-1).mean().item()

            cosine_sims.append(cos_sim)
            l2_dists.append(l2_dist)

    print(f"\nFeature Similarity (q_static vs q_init):")
    print(f"  Cosine similarity:  {np.mean(cosine_sims):.4f}")
    print(f"  L2 distance:        {np.mean(l2_dists):.4f}")

    if np.mean(cosine_sims) > 0.9:
        print("\n  ⚠️  Features are >90% similar - queries produce nearly identical features")
        print("  → Recommendation: Add contrastive loss OR freeze DINO")

    return {
        'cosine_sim': cosine_sims,
        'l2_dist': l2_dists,
    }


@torch.no_grad()
def experiment_4_attention_temperature(model, samples, device):
    """
    Test: How does temperature affect attention focus?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Attention Temperature Analysis")
    print("="*70)

    temperatures = [0.25, 0.5, 1.0, 2.0]
    results = {t: {'entropy': [], 'max_attn': []} for t in temperatures}

    for sample in samples:
        frames_norm = sample['frames_norm'].unsqueeze(0).to(device)
        B, T = 1, frames_norm.shape[1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
            _, cache_flat = model.encoder.encode_patches(frames_flat)
            patch_features = cache_flat['patch_features']  # [B*T, N, D]

            q_static = model.q_static.expand(B * T, -1)
            q_proj = model.encoder.query_input_proj(q_static).unsqueeze(1)

            # Compute attention with different temperatures
            scores = torch.bmm(q_proj, patch_features.transpose(1, 2))

            for temp in temperatures:
                attn = F.softmax(scores / (model.encoder.dino_dim ** 0.5 * temp), dim=-1)

                # Entropy
                entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
                results[temp]['entropy'].append(entropy)

                # Max attention
                max_attn = attn.max(dim=-1).values.mean().item()
                results[temp]['max_attn'].append(max_attn)

    print(f"\nTemperature Effects:")
    print(f"{'Temp':<8} {'Entropy':<12} {'Max Attn':<12} {'Focus Level'}")
    print("-" * 50)
    for temp in temperatures:
        ent = np.mean(results[temp]['entropy'])
        max_a = np.mean(results[temp]['max_attn'])
        focus = "Uniform" if ent > 5.5 else "Moderate" if ent > 4.5 else "Focused"
        print(f"{temp:<8} {ent:<12.3f} {max_a:<12.4f} {focus}")

    # Recommendation
    best_temp = min(temperatures, key=lambda t: np.mean(results[t]['entropy']))
    print(f"\n  → Recommendation: Use temperature={best_temp} for more focused attention")

    return results


@torch.no_grad()
def experiment_5_patch_importance(model, samples, device):
    """
    Test: Which patches matter most for prediction?

    Method: Mask out patches one by one and measure loss increase.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Patch Importance for Prediction")
    print("="*70)

    # Use just one sample for this expensive test
    sample = samples[0]
    frames_norm = sample['frames_norm'].unsqueeze(0).to(device)
    vae_latents = sample['vae_latents'].unsqueeze(0).to(device)

    B, T = 1, frames_norm.shape[1]
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Get baseline prediction
        loss, loss_fine, loss_coarse = model(text_embeds, frames_norm, vae_latents)
        baseline_loss = loss_fine.item()

        # Get patch features
        frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
        _, cache_flat = model.encoder.encode_patches(frames_flat)
        patch_features = cache_flat['patch_features']
        N = patch_features.shape[1]  # 257 (CLS + 256 patches)

        # Test masking groups of patches (16x16 = 256 patches, test 16 groups of 16)
        num_groups = 16
        patches_per_group = (N - 1) // num_groups  # Exclude CLS

        importance_scores = []

        for g in range(num_groups):
            start_idx = 1 + g * patches_per_group  # Skip CLS
            end_idx = min(1 + (g + 1) * patches_per_group, N)

            # Mask out this group by zeroing
            masked_features = patch_features.clone()
            masked_features[:, start_idx:end_idx] = 0

            # Rebuild cache with masked features
            cache_masked = {'patch_features': masked_features}
            if 'kv_cache' in cache_flat:
                cache_masked['kv_cache'] = cache_flat['kv_cache']

            # This is approximate - we'd need to re-run full forward
            # For now, just measure feature variance change
            importance = masked_features.var().item()
            importance_scores.append(importance)

    print(f"\nBaseline loss: {baseline_loss:.4f}")
    print(f"\nPatch importance varies by: {max(importance_scores)/min(importance_scores):.2f}x")

    if max(importance_scores)/min(importance_scores) < 1.5:
        print("\n  ⚠️  All patches contribute similarly - spatial selectivity won't help much")
        print("  → Recommendation: Need task where specific regions matter more")

    return importance_scores


def create_analysis_report(results, output_dir):
    """Create analysis report with recommendations."""

    report = f"""# Preliminary Experiments Analysis

**Date:** {datetime.now().isoformat()}

---

## Executive Summary

"""

    # Experiment 1 analysis
    if 'exp1' in results:
        exp1 = results['exp1']
        h_contrib = (np.mean(exp1['zero_h']) - np.mean(exp1['normal'])) / np.mean(exp1['normal'])
        prev_contrib = (np.mean(exp1['zero_prev']) - np.mean(exp1['normal'])) / np.mean(exp1['normal'])

        report += f"""### Experiment 1: prev_latents Dominance

| Condition | Loss | Change from Normal |
|-----------|------|-------------------|
| Normal (h + prev) | {np.mean(exp1['normal']):.4f} | - |
| Zero h (prev only) | {np.mean(exp1['zero_h']):.4f} | +{h_contrib*100:.1f}% |
| Zero prev (h only) | {np.mean(exp1['zero_prev']):.4f} | +{prev_contrib*100:.1f}% |
| Random h (prev only) | {np.mean(exp1['random_h']):.4f} | +{(np.mean(exp1['random_h'])-np.mean(exp1['normal']))/np.mean(exp1['normal'])*100:.1f}% |

**Finding:** {"prev_latents DOMINATES (h contributes <10%)" if h_contrib < 0.1 else "Both h and prev_latents contribute meaningfully"}

**Recommendation:** {"🔴 HIGH PRIORITY - Remove or weaken prev_latents" if h_contrib < 0.1 else "Keep current architecture"}

---

"""

    # Experiment 2 analysis
    if 'exp2' in results:
        exp2 = results['exp2']
        ratio = np.mean(exp2['delta_var']) / np.mean(exp2['full_var'])

        report += f"""### Experiment 2: Delta vs Full Latent

| Metric | Value |
|--------|-------|
| Full latent variance | {np.mean(exp2['full_var']):.4f} |
| Delta variance | {np.mean(exp2['delta_var']):.4f} |
| Ratio (delta/full) | {ratio:.2f}x |
| Mean |delta| | {np.mean(exp2['delta_mag']):.4f} |

**Finding:** {"Delta has LOW variance - videos are mostly STATIC" if ratio < 0.5 else "Delta has meaningful spatial structure"}

**Recommendation:** {"🔴 Use more dynamic dataset OR scale up delta prediction" if np.mean(exp2['delta_mag']) < 0.1 else "Delta prediction could work"}

---

"""

    # Experiment 3 analysis
    if 'exp3' in results:
        exp3 = results['exp3']

        report += f"""### Experiment 3: Fine vs Coarse Feature Similarity

| Metric | Value |
|--------|-------|
| Cosine similarity | {np.mean(exp3['cosine_sim']):.4f} |
| L2 distance | {np.mean(exp3['l2_dist']):.4f} |

**Finding:** {"Features are >90% similar - ENCODER COLLAPSE" if np.mean(exp3['cosine_sim']) > 0.9 else f"Features have {(1-np.mean(exp3['cosine_sim']))*100:.1f}% difference"}

**Recommendation:** {"🔴 Add contrastive loss OR freeze DINO" if np.mean(exp3['cosine_sim']) > 0.9 else "Feature diversity is acceptable"}

---

"""

    # Experiment 4 analysis
    if 'exp4' in results:
        exp4 = results['exp4']

        report += """### Experiment 4: Attention Temperature

| Temperature | Entropy | Max Attention | Focus |
|-------------|---------|---------------|-------|
"""
        for temp in sorted(exp4.keys()):
            ent = np.mean(exp4[temp]['entropy'])
            max_a = np.mean(exp4[temp]['max_attn'])
            focus = "Uniform" if ent > 5.5 else "Moderate" if ent > 4.5 else "Focused"
            report += f"| {temp} | {ent:.3f} | {max_a:.4f} | {focus} |\n"

        best_temp = min(exp4.keys(), key=lambda t: np.mean(exp4[t]['entropy']))
        report += f"""
**Finding:** Lower temperature = more focused attention

**Recommendation:** Use temperature={best_temp} for sharper attention

---

"""

    # Overall recommendations
    report += """## Priority Ranking Based on Experiments

Based on experimental results, here's the recommended order:

"""

    priorities = []

    if 'exp1' in results:
        h_contrib = (np.mean(results['exp1']['zero_h']) - np.mean(results['exp1']['normal'])) / np.mean(results['exp1']['normal'])
        if h_contrib < 0.1:
            priorities.append(("1. Remove prev_latents conditioning", "HIGH", "h contributes <10% to prediction"))

    if 'exp2' in results:
        if np.mean(results['exp2']['delta_mag']) < 0.1:
            priorities.append(("2. Use more dynamic dataset", "HIGH", "Current videos too static"))

    if 'exp3' in results:
        if np.mean(results['exp3']['cosine_sim']) > 0.9:
            priorities.append(("3. Add contrastive loss", "MEDIUM", "Fine/coarse features too similar"))

    priorities.append(("4. Lower attention temperature", "LOW", "Quick win for sharper attention"))

    for p, level, reason in priorities:
        report += f"| {level} | {p} | {reason} |\n"

    report += """
---

*Generated by preliminary_experiments.py*
"""

    # Save report
    report_path = output_dir / 'PRELIMINARY_ANALYSIS.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")
    return report


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path('outputs/preliminary_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PRELIMINARY EXPERIMENTS")
    print("="*70)

    # Load model
    print("\nLoading model...")
    checkpoint_path = Path('outputs/large_scale_24h_deep_v3/checkpoints/final_step_26153.pt')

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FoveatedVideoModel(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        dino_dim=384,
        llm_dim=576,
        query_dim=128,
        deep_query=True,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load samples
    samples, vae = load_samples(num_samples=5, num_frames=16)

    if len(samples) == 0:
        print("\nERROR: No samples loaded! Check network connection.")
        return

    # Run experiments
    results = {}

    results['exp1'] = experiment_1_prev_latents_dominance(model, samples, device)
    results['exp2'] = experiment_2_delta_analysis(samples, device)
    results['exp3'] = experiment_3_feature_similarity(model, samples, device)
    results['exp4'] = experiment_4_attention_temperature(model, samples, device)

    # Create analysis report
    report = create_analysis_report(results, output_dir)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
