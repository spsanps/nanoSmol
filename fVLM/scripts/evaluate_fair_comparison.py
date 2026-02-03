#!/usr/bin/env python3
"""
Fair comparison: Foveated (1 tok/frame) vs Baseline (16 tok/frame)

Both models use:
- Same vision encoder: DINOv2-small
- Same LLM: SmolLM2-135M-Instruct
- Same training data: 300 steps on train split
- Same training recipe: everything trainable, lr=3e-5

Only difference: 1 token per frame vs 16 tokens per frame
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
SPLIT_FILE = PROJECT_ROOT / "configs" / "data_split.json"
OUTPUT_DIR = Path("/mnt/d/projects/fVLM/outputs/evaluation_fair")

FOVEATED_CKPT = Path("/mnt/d/projects/fVLM/outputs/foveated_300step/checkpoints/latest.pt")
BASELINE_CKPT = Path("/mnt/d/projects/fVLM/outputs/baseline_vlm_300step/checkpoints/latest.pt")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NUM_FRAMES = 8


def get_val_shard_list():
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    return sorted(split["val_shards"])


def iter_val_samples(val_shards):
    for shard_name in val_shards:
        shard_path = SHARD_DIR / shard_name
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for s in shard['samples']:
                frames_raw = s['frames']
                frames_sub = subsample_frames(frames_raw, NUM_FRAMES)
                yield {
                    'frames_sub': frames_sub,
                    'frames_norm': normalize_frames(frames_sub),
                    'latents_sub': subsample_frames(s['latents'], NUM_FRAMES),
                    'caption': s['caption'],
                }
            del shard
        except Exception as e:
            print(f"  Skipping {shard_name}: {e}")


def subsample_frames(frames_raw, num_frames=NUM_FRAMES):
    T = frames_raw.shape[0]
    if T >= num_frames:
        indices = torch.linspace(0, T - 1, num_frames).long()
    else:
        indices = torch.arange(T)
    return frames_raw[indices]


def normalize_frames(frames_raw):
    frames = frames_raw.float() / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames


@torch.no_grad()
def eval_foveated(model, tokenizer, frames_normalized, latents, caption, use_fine, device):
    """Compute caption CE loss for foveated model."""
    frames = frames_normalized.unsqueeze(0).to(device)
    vae_latents = latents.unsqueeze(0).to(device)
    B, T = 1, frames.shape[1]

    tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
    caption_ids = tokens['input_ids'].to(device)
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]

    frames_flat = frames.reshape(B * T, 3, frames.shape[-2], frames.shape[-1])
    _, cache_flat = model.encoder.encode_patches(frames_flat)
    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = []
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        K_all, V_all, layers = [], [], []
        for li in range(num_layers):
            lc = cache_flat['kv_cache'][li]
            K_all.append(lc['K'].reshape(B, T, N, D))
            V_all.append(lc['V'].reshape(B, T, N, D))
            layers.append(lc['layer'])
        for t in range(T):
            kv = [{'K': K_all[li][:, t], 'V': V_all[li][:, t], 'layer': layers[li]}
                  for li in range(num_layers)]
            all_caches.append({'patch_features': patch_features[:, t], 'kv_cache': kv})
    else:
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    q_static = model.q_static.expand(B, -1)
    z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    if not use_fine:
        coarse_token = model.coarse_token.expand(B, -1, -1)
        seq = torch.cat([coarse_token, z_coarse_llm, caption_embeds], dim=1)
        outputs = model.llm.model(inputs_embeds=seq)
        logits = model.llm.lm_head(outputs.last_hidden_state)
        caption_logits = logits[:, 1+T:-1, :]
    else:
        no_text = model.no_text_token.expand(B, -1, -1)
        coarse_token = model.coarse_token.expand(B, -1, -1)
        fine_token = model.fine_token.expand(B, -1, -1)

        seq_q = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
        out_q = model.llm.model(inputs_embeds=seq_q)
        queries = model.llm_to_query(out_q.last_hidden_state[:, 2:])

        q_init = model.q_init.expand(B, -1).unsqueeze(1)
        current_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        for iteration in range(2):
            z_fine_list = [model.encoder.query_attend(current_q[:, t], all_caches[t]) for t in range(T)]
            z_fine = torch.stack(z_fine_list, dim=1)
            z_fine_llm = model.dino_to_llm(z_fine)
            z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

            if iteration < 1:
                seq_q2 = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
                out_q2 = model.llm.model(inputs_embeds=seq_q2)
                next_q = model.llm_to_query(out_q2.last_hidden_state[:, 2:])
                current_q = torch.cat([q_init, next_q[:, :-1]], dim=1)

        seq = torch.cat([fine_token, z_fine_llm, caption_embeds], dim=1)
        outputs = model.llm.model(inputs_embeds=seq)
        logits = model.llm.lm_head(outputs.last_hidden_state)
        caption_logits = logits[:, 1+T:-1, :]

    loss = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.size(-1)),
        caption_targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    mask = caption_targets.reshape(-1) != tokenizer.pad_token_id
    valid_losses = loss[mask]

    return valid_losses.mean().item(), len(valid_losses)


@torch.no_grad()
def eval_baseline(model, tokenizer, frames_normalized, caption, device):
    """Compute caption CE loss for baseline model (DINOv2 + PixelShuffle)."""
    frames = frames_normalized.unsqueeze(0).to(device)
    B, T = 1, frames.shape[1]

    # Resize to 224x224
    frames_flat = frames.reshape(B * T, 3, 256, 256)
    frames_resized = F.interpolate(frames_flat, size=(224, 224), mode='bilinear', align_corners=False)
    frames_resized = frames_resized.reshape(B, T, 3, 224, 224)

    tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
    caption_ids = tokens['input_ids'].to(device)
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]

    visual_features = model.encode_frames(frames_resized)
    num_tokens = visual_features.shape[2]
    visual_features = visual_features.reshape(B, T * num_tokens, model.llm_dim)

    visual_token = model.visual_token.expand(B, -1, -1)
    seq = torch.cat([visual_token, visual_features, caption_embeds], dim=1)

    outputs = model.llm.model(inputs_embeds=seq)
    logits = model.llm.lm_head(outputs.last_hidden_state)

    visual_len = 1 + T * num_tokens
    caption_logits = logits[:, visual_len:-1, :]

    per_token_loss = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.size(-1)),
        caption_targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    mask = caption_targets.reshape(-1) != tokenizer.pad_token_id
    valid_losses = per_token_loss[mask]

    return valid_losses.mean().item(), len(valid_losses)


@torch.no_grad()
def eval_blind(llm, tokenizer, caption, device):
    """Compute caption CE loss with no visual input."""
    tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)

    outputs = llm(input_ids=input_ids)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    mask = shift_labels.view(-1) != tokenizer.pad_token_id
    valid_losses = per_token_loss[mask]

    return valid_losses.mean().item(), len(valid_losses)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    val_shards = get_val_shard_list()
    print(f"Validation: {len(val_shards)} shards (~{len(val_shards)*200} samples)")

    results = {
        'caption_len': [],
        'foveated_fine': [],
        'foveated_coarse': [],
        'baseline': [],
        'blind': [],
    }

    from src.model.foveated_vlm import FoveatedVideoModel
    from src.model.baseline_vlm import BaselineVLM
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 1. Foveated model ---
    print("\n" + "=" * 60)
    print("Evaluating: Foveated VLM (1 token/frame)")
    print("=" * 60)

    fov_model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=False,
    ).to(device)

    ckpt = torch.load(FOVEATED_CKPT, map_location=device, weights_only=False)
    fov_model.load_state_dict(ckpt['model_state_dict'])
    fov_step = ckpt.get('step', '?')
    print(f"  Loaded checkpoint: step {fov_step}")
    del ckpt

    fov_model.eval()
    for s in tqdm(iter_val_samples(val_shards), desc="Foveated"):
        loss_fine, n_tok = eval_foveated(
            fov_model, tokenizer, s['frames_norm'], s['latents_sub'].float(),
            s['caption'], use_fine=True, device=device
        )
        loss_coarse, _ = eval_foveated(
            fov_model, tokenizer, s['frames_norm'], s['latents_sub'].float(),
            s['caption'], use_fine=False, device=device
        )
        results['foveated_fine'].append(loss_fine)
        results['foveated_coarse'].append(loss_coarse)
        results['caption_len'].append(n_tok)

    print(f"  Evaluated {len(results['foveated_fine'])} samples")
    del fov_model
    torch.cuda.empty_cache()

    # --- 2. Baseline model ---
    print("\n" + "=" * 60)
    print("Evaluating: Baseline VLM (16 tokens/frame)")
    print("=" * 60)

    baseline_model = BaselineVLM(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        pixel_shuffle_scale=4,
    ).to(device)

    ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt['model_state_dict'])
    baseline_step = ckpt.get('step', '?')
    print(f"  Loaded checkpoint: step {baseline_step}")
    del ckpt

    baseline_model.eval()
    for s in tqdm(iter_val_samples(val_shards), desc="Baseline"):
        loss, _ = eval_baseline(baseline_model, tokenizer, s['frames_norm'], s['caption'], device)
        results['baseline'].append(loss)

    print(f"  Evaluated {len(results['baseline'])} samples")
    del baseline_model
    torch.cuda.empty_cache()

    # --- 3. Blind baseline ---
    print("\n" + "=" * 60)
    print("Evaluating: Blind baseline (no vision)")
    print("=" * 60)

    blind_llm = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.bfloat16,
    ).to(device)
    blind_llm.eval()

    for s in tqdm(iter_val_samples(val_shards), desc="Blind"):
        loss, _ = eval_blind(blind_llm, tokenizer, s['caption'], device)
        results['blind'].append(loss)

    print(f"  Evaluated {len(results['blind'])} samples")
    del blind_llm
    torch.cuda.empty_cache()

    # ============================================================================
    # ANALYSIS
    # ============================================================================

    N = len(results['foveated_fine'])

    print("\n" + "=" * 80)
    print(f"FAIR COMPARISON RESULTS (N={N} validation samples)")
    print("=" * 80)

    model_names = ['foveated_fine', 'foveated_coarse', 'baseline', 'blind']
    display_names = ['Foveated fine (1 tok/frame)', 'Foveated coarse (1 tok/frame)',
                     'Baseline (16 tok/frame)', 'Blind (no vision)']

    summary = {}
    for name, display in zip(model_names, display_names):
        arr = np.array(results[name])
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            summary[name] = {'mean': float('nan'), 'std': float('nan'),
                             'se': float('nan'), 'ppl': float('nan'), 'n': 0}
            continue

        mean_loss = valid.mean()
        std_loss = valid.std()
        se = std_loss / np.sqrt(len(valid))
        ppl = np.exp(mean_loss)

        summary[name] = {
            'mean': float(mean_loss),
            'std': float(std_loss),
            'se': float(se),
            'ppl': float(ppl),
            'n': int(len(valid)),
        }

        print(f"  {display:35s} | Loss: {mean_loss:.4f} +/- {se:.4f} | PPL: {ppl:.1f}")

    # Visual contribution
    blind_mean = summary['blind']['mean']
    print("\nVisual Contribution (blind_loss - model_loss):")
    for name, display in zip(model_names[:-1], display_names[:-1]):
        if not np.isnan(summary[name]['mean']):
            contrib = blind_mean - summary[name]['mean']
            print(f"  {display:35s} | {contrib:+.4f} ({contrib/blind_mean*100:+.1f}%)")

    # Key comparison: foveated vs baseline
    print("\n" + "=" * 80)
    print("KEY RESULT: Foveated (1 tok) vs Baseline (16 tok)")
    print("=" * 80)

    fov_arr = np.array(results['foveated_fine'])
    base_arr = np.array(results['baseline'])
    valid = ~np.isnan(fov_arr) & ~np.isnan(base_arr)

    fov_mean = fov_arr[valid].mean()
    base_mean = base_arr[valid].mean()
    diff = fov_mean - base_mean

    t_stat, p_val = stats.ttest_rel(fov_arr[valid], base_arr[valid])
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    print(f"\n  Foveated (1 tok/frame):  Loss = {fov_mean:.4f}, PPL = {np.exp(fov_mean):.1f}")
    print(f"  Baseline (16 tok/frame): Loss = {base_mean:.4f}, PPL = {np.exp(base_mean):.1f}")
    print(f"\n  Difference: {diff:+.4f} nats ({diff/base_mean*100:+.1f}%)")
    print(f"  Paired t-test: t={t_stat:+.3f}, p={p_val:.6f} {sig}")

    efficiency_ratio = 16 / 1  # baseline uses 16x more tokens
    loss_ratio = fov_mean / base_mean
    print(f"\n  Token efficiency: Foveated uses {efficiency_ratio:.0f}x fewer tokens")
    print(f"  Loss ratio: {loss_ratio:.4f} (foveated/baseline)")

    if diff > 0:
        print(f"\n  Interpretation: Baseline is better by {diff:.4f} nats")
        print(f"  But foveated achieves {(1-loss_ratio)*100:.1f}% of baseline quality with 1/16 tokens")
    else:
        print(f"\n  Interpretation: Foveated is better by {-diff:.4f} nats!")
        print(f"  Foveated outperforms despite using 16x fewer tokens")

    # Save results
    summary_path = OUTPUT_DIR / "fair_comparison.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'val_samples': N,
            'num_frames': NUM_FRAMES,
            'foveated_checkpoint': str(FOVEATED_CKPT),
            'baseline_checkpoint': str(BASELINE_CKPT),
            'results': summary,
            'comparison': {
                'foveated_loss': float(fov_mean),
                'baseline_loss': float(base_mean),
                'difference': float(diff),
                't_stat': float(t_stat),
                'p_value': float(p_val),
            }
        }, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Markdown summary
    md_path = OUTPUT_DIR / "fair_comparison.md"
    with open(md_path, 'w') as f:
        f.write("# Fair Comparison: Foveated vs Baseline\n\n")
        f.write("Both models use:\n")
        f.write("- Same vision encoder: DINOv2-small\n")
        f.write("- Same LLM: SmolLM2-135M-Instruct\n")
        f.write("- Same training data: 300 steps on train split\n")
        f.write("- Same training recipe: everything trainable, lr=3e-5\n\n")
        f.write(f"Validation samples: {N} | Frames per video: {NUM_FRAMES}\n\n")
        f.write("| Model | Visual Tokens/Frame | Loss | SE | PPL |\n")
        f.write("|-------|---------------------|------|-----|-----|\n")
        f.write(f"| Foveated (fine) | 1 | {summary['foveated_fine']['mean']:.4f} | {summary['foveated_fine']['se']:.4f} | {summary['foveated_fine']['ppl']:.1f} |\n")
        f.write(f"| Foveated (coarse) | 1 | {summary['foveated_coarse']['mean']:.4f} | {summary['foveated_coarse']['se']:.4f} | {summary['foveated_coarse']['ppl']:.1f} |\n")
        f.write(f"| Baseline | 16 | {summary['baseline']['mean']:.4f} | {summary['baseline']['se']:.4f} | {summary['baseline']['ppl']:.1f} |\n")
        f.write(f"| Blind | 0 | {summary['blind']['mean']:.4f} | {summary['blind']['se']:.4f} | {summary['blind']['ppl']:.1f} |\n\n")
        f.write(f"**Key Result:** Foveated loss is {diff:+.4f} nats ({diff/base_mean*100:+.1f}%) compared to baseline\n")
        f.write(f"- Paired t-test: t={t_stat:+.3f}, p={p_val:.6f}\n")
        f.write(f"- Foveated uses 16x fewer visual tokens\n")
    print(f"Markdown saved to: {md_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
