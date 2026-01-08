"""
Quick preliminary experiments using synthetic data to test key hypotheses.
No network required - uses model's learned weights directly.
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


def create_synthetic_data(batch_size=2, num_frames=8, device='cuda'):
    """Create synthetic data for testing."""
    # Random frames (normalized like ImageNet)
    frames = torch.randn(batch_size, num_frames, 3, 256, 256).to(device)

    # Random VAE latents
    vae_latents = torch.randn(batch_size, num_frames, 4, 32, 32).to(device)

    return frames, vae_latents


@torch.no_grad()
def experiment_1_prev_latents_dominance(model, device):
    """
    Test: How much does prediction rely on prev_latents vs h (visual features)?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: prev_latents Dominance")
    print("="*70)

    frames, vae_latents = create_synthetic_data(batch_size=4, num_frames=8, device=device)
    B, T = frames.shape[:2]
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Encode frames
        frames_flat = frames.reshape(B * T, 3, 256, 256)
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
                        'K': K_all[:, t], 'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': frame_kv_cache})
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Pass 1 to get h
        q_static = model.q_static.expand(B, -1)
        z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
        z_coarse = torch.stack(z_coarse_list, dim=1)
        z_coarse_proj = model.dino_to_llm(z_coarse)

        coarse_token = model.coarse_token.expand(B, 1, -1)
        seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse_proj], dim=1)
        h_pass1 = model.llm.model(inputs_embeds=seq_pass1).last_hidden_state
        h_for_pred = h_pass1[:, N_text:N_text + T]

        # Prepare prev_latents
        z_vae_init = model.z_vae_init.expand(B, 1, -1, -1, -1)
        prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
        targets = vae_latents

        # Test different conditions
        results = {}

        # Normal prediction
        pred_normal = model.pred_head(h_for_pred, prev_latents)
        results['normal'] = F.mse_loss(pred_normal, targets).item()

        # Zero out h
        h_zero = torch.zeros_like(h_for_pred)
        pred_zero_h = model.pred_head(h_zero, prev_latents)
        results['zero_h'] = F.mse_loss(pred_zero_h, targets).item()

        # Zero out prev_latents
        prev_zero = torch.zeros_like(prev_latents)
        pred_zero_prev = model.pred_head(h_for_pred, prev_zero)
        results['zero_prev'] = F.mse_loss(pred_zero_prev, targets).item()

        # Random h
        h_random = torch.randn_like(h_for_pred)
        pred_random_h = model.pred_head(h_random, prev_latents)
        results['random_h'] = F.mse_loss(pred_random_h, targets).item()

        # Random prev_latents
        prev_random = torch.randn_like(prev_latents)
        pred_random_prev = model.pred_head(h_for_pred, prev_random)
        results['random_prev'] = F.mse_loss(pred_random_prev, targets).item()

    print("\nResults (lower = better match to target):")
    print(f"  Normal (h + prev):       {results['normal']:.4f}")
    print(f"  Zero h (prev only):      {results['zero_h']:.4f} ({(results['zero_h']-results['normal'])/results['normal']*100:+.1f}%)")
    print(f"  Zero prev (h only):      {results['zero_prev']:.4f} ({(results['zero_prev']-results['normal'])/results['normal']*100:+.1f}%)")
    print(f"  Random h (prev only):    {results['random_h']:.4f} ({(results['random_h']-results['normal'])/results['normal']*100:+.1f}%)")
    print(f"  Random prev (h only):    {results['random_prev']:.4f} ({(results['random_prev']-results['normal'])/results['normal']*100:+.1f}%)")

    h_impact = (results['zero_h'] - results['normal']) / results['normal']
    prev_impact = (results['zero_prev'] - results['normal']) / results['normal']

    print(f"\n  Impact Analysis:")
    print(f"    Zeroing h increases loss by: {h_impact*100:.1f}%")
    print(f"    Zeroing prev increases loss by: {prev_impact*100:.1f}%")

    if h_impact < prev_impact * 0.3:
        print(f"\n  ⚠️  h contributes {h_impact/prev_impact*100:.0f}% as much as prev_latents")
        print(f"  → prev_latents DOMINATES the prediction")
        print(f"  → RECOMMENDATION: Remove/weaken prev_latents conditioning")
    else:
        print(f"\n  ✓ Both h and prev_latents contribute meaningfully")

    return results


@torch.no_grad()
def experiment_2_feature_similarity(model, device):
    """
    Test: How similar are z_fine and z_coarse features?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Fine vs Coarse Feature Similarity")
    print("="*70)

    frames, _ = create_synthetic_data(batch_size=4, num_frames=8, device=device)
    B, T = frames.shape[:2]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        frames_flat = frames.reshape(B * T, 3, 256, 256)
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
                        'K': K_all[:, t], 'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': frame_kv_cache})
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # Coarse features (q_static)
        q_static = model.q_static.expand(B, -1)
        z_coarse = torch.stack([model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)

        # Fine features (q_init - baseline dynamic query)
        q_init = model.q_init.expand(B, -1)
        z_fine = torch.stack([model.encoder.query_attend(q_init, all_caches[t]) for t in range(T)], dim=1)

        # Random query
        q_random = torch.randn_like(q_static)
        z_random = torch.stack([model.encoder.query_attend(q_random, all_caches[t]) for t in range(T)], dim=1)

        # Compute similarities
        z_coarse_flat = z_coarse.view(-1, z_coarse.shape[-1]).float()
        z_fine_flat = z_fine.view(-1, z_fine.shape[-1]).float()
        z_random_flat = z_random.view(-1, z_random.shape[-1]).float()

        cos_static_init = F.cosine_similarity(z_coarse_flat, z_fine_flat, dim=-1).mean().item()
        cos_static_random = F.cosine_similarity(z_coarse_flat, z_random_flat, dim=-1).mean().item()
        cos_init_random = F.cosine_similarity(z_fine_flat, z_random_flat, dim=-1).mean().item()

        l2_static_init = (z_coarse_flat - z_fine_flat).norm(dim=-1).mean().item()
        l2_static_random = (z_coarse_flat - z_random_flat).norm(dim=-1).mean().item()

    print(f"\nCosine Similarity (1.0 = identical):")
    print(f"  q_static vs q_init:    {cos_static_init:.4f}")
    print(f"  q_static vs q_random:  {cos_static_random:.4f}")
    print(f"  q_init vs q_random:    {cos_init_random:.4f}")

    print(f"\nL2 Distance (0 = identical):")
    print(f"  q_static vs q_init:    {l2_static_init:.4f}")
    print(f"  q_static vs q_random:  {l2_static_random:.4f}")

    if cos_static_init > 0.95:
        print(f"\n  ⚠️  q_static and q_init produce >95% similar features!")
        print(f"  → Different queries are NOT producing different features")
        print(f"  → RECOMMENDATION: Add contrastive loss OR freeze DINO")
    elif cos_static_init > 0.8:
        print(f"\n  ⚠️  Features are {cos_static_init*100:.0f}% similar - moderate concern")
    else:
        print(f"\n  ✓ Features have meaningful {(1-cos_static_init)*100:.0f}% difference")

    return {
        'cos_static_init': cos_static_init,
        'cos_static_random': cos_static_random,
        'l2_static_init': l2_static_init,
    }


@torch.no_grad()
def experiment_3_attention_temperature(model, device):
    """
    Test: How does temperature affect attention distribution?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Attention Temperature Analysis")
    print("="*70)

    frames, _ = create_synthetic_data(batch_size=2, num_frames=4, device=device)
    B, T = frames.shape[:2]

    temperatures = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    results = {}

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        frames_flat = frames.reshape(B * T, 3, 256, 256)
        _, cache_flat = model.encoder.encode_patches(frames_flat)
        patch_features = cache_flat['patch_features']  # [B*T, N, D]

        q_static = model.q_static.expand(B * T, -1)
        q_proj = model.encoder.query_input_proj(q_static).unsqueeze(1)

        # Compute base attention scores
        scores = torch.bmm(q_proj, patch_features.transpose(1, 2))
        base_scale = model.encoder.dino_dim ** 0.5

        print(f"\nTemperature Effects:")
        print(f"{'Temp':<8} {'Entropy':<12} {'Max Attn':<12} {'Top-10%':<12} {'Focus'}")
        print("-" * 60)

        for temp in temperatures:
            attn = F.softmax(scores / (base_scale * temp), dim=-1)

            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
            max_attn = attn.max(dim=-1).values.mean().item()

            # Top 10% coverage
            sorted_attn, _ = attn.sort(dim=-1, descending=True)
            top_10_pct = sorted_attn[:, :, :26].sum(dim=-1).mean().item()  # ~10% of 257

            focus = "Uniform" if entropy > 5.5 else "Moderate" if entropy > 4.5 else "Focused" if entropy > 3.0 else "Sharp"

            results[temp] = {'entropy': entropy, 'max_attn': max_attn, 'top_10': top_10_pct}
            print(f"{temp:<8} {entropy:<12.3f} {max_attn:<12.4f} {top_10_pct:<12.2%} {focus}")

    best_temp = min(temperatures, key=lambda t: results[t]['entropy'])
    print(f"\n  → RECOMMENDATION: Use temperature={best_temp} for sharper attention")
    print(f"    (Current default is 1.0)")

    return results


@torch.no_grad()
def experiment_4_query_projection_analysis(model, device):
    """
    Test: How much do queries change after projection?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Query Projection Analysis")
    print("="*70)

    # Get the learned queries
    q_static = model.q_static.data
    q_init = model.q_init.data

    print(f"\nLearned Query Stats:")
    print(f"  q_static: mean={q_static.mean():.4f}, std={q_static.std():.4f}, norm={q_static.norm():.4f}")
    print(f"  q_init:   mean={q_init.mean():.4f}, std={q_init.std():.4f}, norm={q_init.norm():.4f}")

    # Project them
    q_static_proj = model.encoder.query_input_proj(q_static.to(device))
    q_init_proj = model.encoder.query_input_proj(q_init.to(device))

    print(f"\nAfter Projection:")
    print(f"  q_static_proj: mean={q_static_proj.mean():.4f}, std={q_static_proj.std():.4f}, norm={q_static_proj.norm():.4f}")
    print(f"  q_init_proj:   mean={q_init_proj.mean():.4f}, std={q_init_proj.std():.4f}, norm={q_init_proj.norm():.4f}")

    # Similarity before and after projection
    cos_before = F.cosine_similarity(q_static.flatten(), q_init.flatten(), dim=0).item()
    cos_after = F.cosine_similarity(q_static_proj.flatten(), q_init_proj.flatten(), dim=0).item()

    print(f"\nQuery Similarity:")
    print(f"  Before projection: {cos_before:.4f}")
    print(f"  After projection:  {cos_after:.4f}")

    if cos_after > cos_before + 0.1:
        print(f"\n  ⚠️  Projection INCREASES similarity by {(cos_after-cos_before)*100:.1f}%")
        print(f"  → Projection is collapsing query diversity")
    elif cos_after < cos_before - 0.1:
        print(f"\n  ✓ Projection PRESERVES/AMPLIFIES query differences")

    # Check projection weight characteristics
    W = model.encoder.query_input_proj.weight.data
    print(f"\nProjection Weight Stats:")
    print(f"  Shape: {W.shape}")
    print(f"  Rank (approx): {torch.linalg.matrix_rank(W.float()).item()}")
    print(f"  Condition number: {torch.linalg.cond(W.float()).item():.1f}")

    return {
        'cos_before': cos_before,
        'cos_after': cos_after,
        'q_static_norm': q_static.norm().item(),
        'q_init_norm': q_init.norm().item(),
    }


@torch.no_grad()
def experiment_5_pred_head_sensitivity(model, device):
    """
    Test: How sensitive is the prediction head to h vs prev_latents?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Prediction Head Sensitivity")
    print("="*70)

    B, T = 4, 8

    # Create base inputs
    h_base = torch.randn(B, T, 576, device=device, dtype=torch.bfloat16)
    prev_base = torch.randn(B, T, 4, 32, 32, device=device, dtype=torch.bfloat16)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_base = model.pred_head(h_base, prev_base)

        # Perturb h
        h_perturbed = h_base + 0.1 * torch.randn_like(h_base)
        pred_h_perturbed = model.pred_head(h_perturbed, prev_base)
        h_sensitivity = (pred_h_perturbed - pred_base).abs().mean().item()

        # Perturb prev_latents
        prev_perturbed = prev_base + 0.1 * torch.randn_like(prev_base)
        pred_prev_perturbed = model.pred_head(h_base, prev_perturbed)
        prev_sensitivity = (pred_prev_perturbed - pred_base).abs().mean().item()

        # Scale h
        h_scaled = h_base * 2
        pred_h_scaled = model.pred_head(h_scaled, prev_base)
        h_scale_effect = (pred_h_scaled - pred_base).abs().mean().item()

        # Scale prev
        prev_scaled = prev_base * 2
        pred_prev_scaled = model.pred_head(h_base, prev_scaled)
        prev_scale_effect = (pred_prev_scaled - pred_base).abs().mean().item()

    print(f"\nSensitivity to 10% perturbation:")
    print(f"  h perturbation effect:    {h_sensitivity:.4f}")
    print(f"  prev perturbation effect: {prev_sensitivity:.4f}")
    print(f"  Ratio (prev/h):           {prev_sensitivity/h_sensitivity:.2f}x")

    print(f"\nSensitivity to 2x scaling:")
    print(f"  h scaling effect:         {h_scale_effect:.4f}")
    print(f"  prev scaling effect:      {prev_scale_effect:.4f}")
    print(f"  Ratio (prev/h):           {prev_scale_effect/h_scale_effect:.2f}x")

    if prev_sensitivity > h_sensitivity * 3:
        print(f"\n  ⚠️  Prediction head is {prev_sensitivity/h_sensitivity:.1f}x more sensitive to prev_latents!")
        print(f"  → The model has learned to mostly copy prev_latents")
        print(f"  → RECOMMENDATION: Remove prev_latents or add noise")
    else:
        print(f"\n  ✓ Prediction head responds to both inputs")

    return {
        'h_sensitivity': h_sensitivity,
        'prev_sensitivity': prev_sensitivity,
        'ratio': prev_sensitivity / h_sensitivity,
    }


def create_report(results, output_dir):
    """Create analysis report."""
    report = f"""# Preliminary Experiments Analysis

**Date:** {datetime.now().isoformat()}

---

## Summary

"""

    # Experiment 1
    if 'exp1' in results:
        e = results['exp1']
        h_impact = (e['zero_h'] - e['normal']) / e['normal']
        prev_impact = (e['zero_prev'] - e['normal']) / e['normal']

        report += f"""### Experiment 1: prev_latents Dominance

| Condition | Loss | Change |
|-----------|------|--------|
| Normal | {e['normal']:.4f} | - |
| Zero h | {e['zero_h']:.4f} | +{h_impact*100:.1f}% |
| Zero prev | {e['zero_prev']:.4f} | +{prev_impact*100:.1f}% |

**Finding:** {"prev_latents DOMINATES - h contributes <30% as much" if h_impact < prev_impact * 0.3 else "Both contribute"}

**Priority:** {"🔴 HIGH - Remove prev_latents" if h_impact < prev_impact * 0.3 else "Low"}

---

"""

    # Experiment 2
    if 'exp2' in results:
        e = results['exp2']
        report += f"""### Experiment 2: Feature Similarity

| Query Pair | Cosine Similarity |
|------------|------------------|
| q_static vs q_init | {e['cos_static_init']:.4f} |
| q_static vs random | {e['cos_static_random']:.4f} |

**Finding:** {"Features >90% similar - queries don't differentiate!" if e['cos_static_init'] > 0.9 else f"Features are {(1-e['cos_static_init'])*100:.0f}% different"}

**Priority:** {"🔴 HIGH - Add contrastive loss" if e['cos_static_init'] > 0.9 else "Medium" if e['cos_static_init'] > 0.8 else "Low"}

---

"""

    # Experiment 3
    if 'exp3' in results:
        e = results['exp3']
        best = min(e.keys(), key=lambda t: e[t]['entropy'])
        report += f"""### Experiment 3: Temperature

| Temp | Entropy | Max Attn |
|------|---------|----------|
"""
        for t in sorted(e.keys()):
            report += f"| {t} | {e[t]['entropy']:.3f} | {e[t]['max_attn']:.4f} |\n"

        report += f"""
**Finding:** Best temperature = {best} (lowest entropy)

**Priority:** 🟡 MEDIUM - Easy win

---

"""

    # Experiment 5
    if 'exp5' in results:
        e = results['exp5']
        report += f"""### Experiment 5: Pred Head Sensitivity

| Input | Sensitivity |
|-------|-------------|
| h | {e['h_sensitivity']:.4f} |
| prev_latents | {e['prev_sensitivity']:.4f} |
| Ratio | {e['ratio']:.2f}x |

**Finding:** {"Model is {:.1f}x more sensitive to prev_latents!".format(e['ratio']) if e['ratio'] > 3 else "Balanced sensitivity"}

**Priority:** {"🔴 HIGH - Model copies prev_latents" if e['ratio'] > 3 else "Low"}

---

"""

    # Overall recommendations
    report += """## Priority Recommendations

Based on experiments:

"""

    priorities = []
    if 'exp1' in results:
        h_impact = (results['exp1']['zero_h'] - results['exp1']['normal']) / results['exp1']['normal']
        prev_impact = (results['exp1']['zero_prev'] - results['exp1']['normal']) / results['exp1']['normal']
        if h_impact < prev_impact * 0.3:
            priorities.append(("1. Remove/weaken prev_latents", "HIGH", "h contributes <30% vs prev"))

    if 'exp2' in results and results['exp2']['cos_static_init'] > 0.9:
        priorities.append(("2. Add contrastive loss", "HIGH", "Features >90% similar"))

    if 'exp5' in results and results['exp5']['ratio'] > 3:
        priorities.append(("3. Modify prediction head", "HIGH", f"prev {results['exp5']['ratio']:.1f}x more influential"))

    if 'exp3' in results:
        best = min(results['exp3'].keys(), key=lambda t: results['exp3'][t]['entropy'])
        if best != 1.0:
            priorities.append((f"4. Use temperature={best}", "MEDIUM", "Sharper attention"))

    for p, level, reason in priorities:
        report += f"| {level} | {p} | {reason} |\n"

    if not priorities:
        report += "No critical issues found!\n"

    report += "\n---\n*Generated by quick_experiments.py*\n"

    path = output_dir / 'PRELIMINARY_ANALYSIS.md'
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {path}")

    return report


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path('outputs/preliminary_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("QUICK PRELIMINARY EXPERIMENTS")
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

    # Run experiments
    results = {}

    results['exp1'] = experiment_1_prev_latents_dominance(model, device)
    results['exp2'] = experiment_2_feature_similarity(model, device)
    results['exp3'] = experiment_3_attention_temperature(model, device)
    results['exp4'] = experiment_4_query_projection_analysis(model, device)
    results['exp5'] = experiment_5_pred_head_sensitivity(model, device)

    # Create report
    create_report(results, output_dir)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
