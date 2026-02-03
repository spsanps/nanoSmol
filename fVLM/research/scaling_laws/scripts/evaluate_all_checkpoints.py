#!/usr/bin/env python3
"""
Evaluate all checkpoints from scaling experiments.

Collects validation loss, perplexity, and compute metrics for scaling law analysis.
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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
SPLIT_FILE = PROJECT_ROOT / "configs" / "data_split.json"
OUTPUT_BASE = Path("/mnt/d/projects/fVLM/outputs/scaling_study")
RESEARCH_DIR = Path(__file__).parent.parent

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NUM_FRAMES = 8
MAX_VAL_SAMPLES = 2000  # Limit for faster evaluation

# FLOPs constants (from estimate_flops.py)
FLOPS_PER_SAMPLE = {
    "foveated_opt": 144.7e9,    # 224px, 1 fine iter
    "baseline": 151.3e9,         # 224px, 16 tok/frame
    "foveated_orig": 202.7e9,   # 256px, 2 fine iter
}


# ============================================================================
# Data Loading
# ============================================================================

def get_val_shard_list():
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    return sorted(split["val_shards"])


def load_val_samples(val_shards, frame_size, max_samples=MAX_VAL_SAMPLES):
    """Load validation samples."""
    samples = []

    for shard_name in val_shards:
        if len(samples) >= max_samples:
            break

        shard_path = SHARD_DIR / shard_name
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for s in shard['samples']:
                if len(samples) >= max_samples:
                    break

                frames = s['frames']
                T = frames.shape[0]
                if T >= NUM_FRAMES:
                    indices = torch.linspace(0, T - 1, NUM_FRAMES).long()
                else:
                    indices = torch.arange(T)
                    pad = NUM_FRAMES - T
                    indices = torch.cat([indices, indices[-1:].expand(pad)])

                frames = frames[indices].float()

                # Resize if needed
                if frames.shape[-1] != frame_size:
                    frames = F.interpolate(
                        frames, size=(frame_size, frame_size),
                        mode='bilinear', align_corners=False
                    )

                # Normalize
                frames = frames / 255.0
                frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

                samples.append({
                    'frames': frames,
                    'latents': s['latents'][indices].float(),
                    'caption': s['caption'],
                })

            del shard
        except Exception as e:
            print(f"Skipping {shard_name}: {e}")

    return samples


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def eval_foveated(model, tokenizer, samples, fine_iterations, device):
    """Evaluate foveated model on validation samples."""
    model.eval()
    losses = []

    for s in tqdm(samples, desc="Evaluating foveated"):
        frames = s['frames'].unsqueeze(0).to(device)
        latents = s['latents'].unsqueeze(0).to(device)
        caption = s['caption']

        tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
        caption_ids = tokens['input_ids'].to(device)

        B, T = 1, frames.shape[1]

        # Encode frames
        frames_flat = frames.reshape(B * T, 3, frames.shape[-2], frames.shape[-1])
        _, cache_flat = model.encoder.encode_patches(frames_flat)
        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Build per-frame caches
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

        # Coarse features
        q_static = model.q_static.expand(B, -1)
        z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
        z_coarse = torch.stack(z_coarse_list, dim=1)
        z_coarse_llm = model.dino_to_llm(z_coarse)
        z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

        # Fine iterations
        if fine_iterations > 0:
            no_text = model.no_text_token.expand(B, -1, -1)
            coarse_token = model.coarse_token.expand(B, -1, -1)
            fine_token = model.fine_token.expand(B, -1, -1)

            seq_q = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
            out_q = model.llm.model(inputs_embeds=seq_q)
            queries = model.llm_to_query(out_q.last_hidden_state[:, 2:])

            q_init = model.q_init.expand(B, -1).unsqueeze(1)
            current_q = torch.cat([q_init, queries[:, :-1]], dim=1)

            for iteration in range(fine_iterations):
                z_fine_list = [model.encoder.query_attend(current_q[:, t], all_caches[t]) for t in range(T)]
                z_fine = torch.stack(z_fine_list, dim=1)
                z_fine_llm = model.dino_to_llm(z_fine)
                z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

                if iteration < fine_iterations - 1:
                    seq_q2 = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
                    out_q2 = model.llm.model(inputs_embeds=seq_q2)
                    next_q = model.llm_to_query(out_q2.last_hidden_state[:, 2:])
                    current_q = torch.cat([q_init, next_q[:, :-1]], dim=1)

            visual_features = z_fine_llm
            mode_token = fine_token
        else:
            visual_features = z_coarse_llm
            mode_token = model.coarse_token.expand(B, -1, -1)

        # Caption forward
        caption_embeds = model.llm.model.embed_tokens(caption_ids)
        seq = torch.cat([mode_token, visual_features, caption_embeds], dim=1)
        outputs = model.llm.model(inputs_embeds=seq)
        logits = model.llm.lm_head(outputs.last_hidden_state)

        caption_logits = logits[:, 1+T:-1, :]
        caption_targets = caption_ids[:, 1:]

        loss = F.cross_entropy(
            caption_logits.reshape(-1, caption_logits.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='none'
        )
        mask = caption_targets.reshape(-1) != tokenizer.pad_token_id
        valid_loss = loss[mask].mean().item()
        losses.append(valid_loss)

    return np.array(losses)


@torch.no_grad()
def eval_baseline(model, tokenizer, samples, device):
    """Evaluate baseline model on validation samples."""
    model.eval()
    losses = []

    for s in tqdm(samples, desc="Evaluating baseline"):
        frames = s['frames'].unsqueeze(0).to(device)
        caption = s['caption']

        tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
        caption_ids = tokens['input_ids'].to(device)

        B, T = 1, frames.shape[1]

        # Encode frames
        visual_features = model.encode_frames(frames)
        num_tokens = visual_features.shape[2]
        visual_features = visual_features.reshape(B, T * num_tokens, model.llm_dim)

        # Caption forward
        caption_embeds = model.llm.model.embed_tokens(caption_ids)
        visual_token = model.visual_token.expand(B, -1, -1)
        seq = torch.cat([visual_token, visual_features, caption_embeds], dim=1)

        outputs = model.llm.model(inputs_embeds=seq)
        logits = model.llm.lm_head(outputs.last_hidden_state)

        visual_len = 1 + T * num_tokens
        caption_logits = logits[:, visual_len:-1, :]
        caption_targets = caption_ids[:, 1:]

        loss = F.cross_entropy(
            caption_logits.reshape(-1, caption_logits.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='none'
        )
        mask = caption_targets.reshape(-1) != tokenizer.pad_token_id
        valid_loss = loss[mask].mean().item()
        losses.append(valid_loss)

    return np.array(losses)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("EVALUATE ALL CHECKPOINTS")
    print("=" * 80)

    device = torch.device("cuda")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get validation shards
    val_shards = get_val_shard_list()
    print(f"Validation shards: {len(val_shards)}")

    results = []

    # Find all experiments
    for exp_dir in sorted(OUTPUT_BASE.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith('.'):
            continue

        exp_name = exp_dir.name
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")

        # Determine config from experiment name
        if exp_name == "foveated_opt":
            model_type = "foveated"
            frame_size = 224
            fine_iterations = 1
        elif exp_name == "baseline":
            model_type = "baseline"
            frame_size = 224
            fine_iterations = 0
        elif exp_name == "foveated_orig":
            model_type = "foveated"
            frame_size = 256
            fine_iterations = 2
        else:
            print(f"  Unknown experiment: {exp_name}, skipping")
            continue

        # Load validation samples
        print(f"  Loading validation samples (frame_size={frame_size})...")
        samples = load_val_samples(val_shards, frame_size)
        print(f"  Loaded {len(samples)} samples")

        # Create model
        if model_type == "foveated":
            from src.model.foveated_vlm import FoveatedVideoModel
            model = FoveatedVideoModel(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                deep_query=True,
                freeze_dino=False,
            ).to(device)
        else:
            from src.model.baseline_vlm import BaselineVLM
            model = BaselineVLM(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                pixel_shuffle_scale=4,
            ).to(device)

        # Evaluate each checkpoint
        for ckpt_path in sorted(exp_dir.glob("step_*.pt")):
            step = int(ckpt_path.stem.split('_')[1])
            print(f"\n  Checkpoint: step {step}")

            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            # Evaluate
            if model_type == "foveated":
                losses = eval_foveated(model, tokenizer, samples, fine_iterations, device)
            else:
                losses = eval_baseline(model, tokenizer, samples, device)

            mean_loss = losses.mean()
            std_loss = losses.std()
            se_loss = std_loss / np.sqrt(len(losses))
            ppl = np.exp(mean_loss)

            # Calculate FLOPs
            flops_per_sample = FLOPS_PER_SAMPLE.get(exp_name, 150e9)
            batch_size = 16
            total_training_flops = step * batch_size * flops_per_sample * 3  # 3x for backward

            print(f"    Loss: {mean_loss:.4f} +/- {se_loss:.4f}, PPL: {ppl:.1f}")
            print(f"    Training FLOPs: {total_training_flops/1e15:.3f} PFLOPs")

            results.append({
                'experiment': exp_name,
                'model_type': model_type,
                'frame_size': frame_size,
                'fine_iterations': fine_iterations,
                'step': step,
                'val_loss': float(mean_loss),
                'val_loss_std': float(std_loss),
                'val_loss_se': float(se_loss),
                'perplexity': float(ppl),
                'n_samples': len(losses),
                'flops_per_sample': flops_per_sample,
                'total_training_flops': total_training_flops,
                'visual_tokens_per_frame': 1 if model_type == "foveated" else 16,
            })

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save results
    data_dir = RESEARCH_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    results_path = data_dir / "scaling_data.csv"
    import csv
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {results_path}")

    # Also save as JSON
    json_path = data_dir / "scaling_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
