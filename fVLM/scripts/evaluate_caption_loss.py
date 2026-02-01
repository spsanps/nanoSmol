#!/usr/bin/env python3
"""
Unified caption loss evaluation across multiple models.

Computes per-sample caption cross-entropy loss on the held-out validation set
for 5 model configurations:
  1. Foveated VLM (fine pathway)
  2. Foveated VLM (coarse pathway)
  3. SmolVLM2-256M-Video (pretrained, no fine-tuning)
  4. SmolVLM2-256M-Video (fine-tuned on train split)
  5. SmolLM2-135M blind baseline (no visual input)

Output: /mnt/d/projects/fVLM/outputs/evaluation/
"""

import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
SPLIT_FILE = PROJECT_ROOT / "configs" / "data_split.json"
OUTPUT_DIR = Path("/mnt/d/projects/fVLM/outputs/evaluation")

FOVEATED_CKPT = Path("/mnt/d/projects/fVLM/outputs/foveated_singleepoch/checkpoints/latest.pt")
SMOLVLM_CKPT = Path("/mnt/d/projects/fVLM/outputs/smolvlm_singleepoch/checkpoints/latest.pt")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NUM_FRAMES = 8


# ============================================================================
# DATA LOADING
# ============================================================================

def load_val_samples():
    """Load all validation samples into memory."""
    with open(SPLIT_FILE) as f:
        split = json.load(f)

    val_shards = split["val_shards"]
    print(f"Loading {len(val_shards)} val shards...")

    samples = []
    for shard_name in tqdm(val_shards, desc="Loading shards"):
        shard_path = SHARD_DIR / shard_name
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for s in shard['samples']:
                samples.append({
                    'frames_raw': s['frames'],   # [24, 3, 256, 256] uint8
                    'latents': s['latents'],      # [24, 4, 32, 32]
                    'caption': s['caption'],
                })
            del shard
        except Exception as e:
            print(f"  Skipping {shard_name}: {e}")

    print(f"Loaded {len(samples)} validation samples")
    return samples


def subsample_frames(frames_raw, num_frames=NUM_FRAMES):
    """Deterministic uniform temporal subsampling."""
    T = frames_raw.shape[0]
    if T >= num_frames:
        indices = torch.linspace(0, T - 1, num_frames).long()
    else:
        indices = torch.arange(T)
    return frames_raw[indices]


def normalize_frames(frames_raw):
    """uint8 [T, 3, 256, 256] -> ImageNet normalized float32."""
    frames = frames_raw.float() / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames


def frames_to_pil(frames_raw):
    """uint8 [T, 3, 256, 256] -> list of PIL images."""
    pil_frames = []
    for t in range(frames_raw.shape[0]):
        frame_np = frames_raw[t].permute(1, 2, 0).numpy()
        pil_frames.append(Image.fromarray(frame_np))
    return pil_frames


# ============================================================================
# MODEL EVALUATORS
# ============================================================================

@torch.no_grad()
def eval_foveated(model, tokenizer, frames_normalized, latents, caption, use_fine, device):
    """Compute caption CE loss for foveated model.

    Replicates the forward_multifine_joint loss computation on a single sample.
    """
    frames = frames_normalized.unsqueeze(0).to(device)  # [1, T, 3, 256, 256]
    vae_latents = latents.unsqueeze(0).to(device)
    B, T = 1, frames.shape[1]

    # Tokenize caption
    tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
    caption_ids = tokens['input_ids'].to(device)
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]  # shifted

    # Encode frames
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

    # Coarse features
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    if not use_fine:
        # Coarse pathway loss
        coarse_token = model.coarse_token.expand(B, -1, -1)
        seq = torch.cat([coarse_token, z_coarse_llm, caption_embeds], dim=1)
        outputs = model.llm.model(inputs_embeds=seq)
        logits = model.llm.lm_head(outputs.last_hidden_state)
        caption_logits = logits[:, 1+T:-1, :]
    else:
        # Fine pathway: coarse -> queries -> fine features
        no_text = model.no_text_token.expand(B, -1, -1)
        coarse_token = model.coarse_token.expand(B, -1, -1)
        fine_token = model.fine_token.expand(B, -1, -1)

        seq_q = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
        out_q = model.llm.model(inputs_embeds=seq_q)
        queries = model.llm_to_query(out_q.last_hidden_state[:, 2:])

        q_init = model.q_init.expand(B, -1).unsqueeze(1)
        current_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        # Multi-fine iterations (2 by default)
        for iteration in range(2):
            z_fine_list = [model.encoder.query_attend(current_q[:, t], all_caches[t]) for t in range(T)]
            z_fine = torch.stack(z_fine_list, dim=1)
            z_fine_llm = model.dino_to_llm(z_fine)
            z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

            if iteration < 1:  # Generate queries for next iteration
                seq_q2 = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
                out_q2 = model.llm.model(inputs_embeds=seq_q2)
                next_q = model.llm_to_query(out_q2.last_hidden_state[:, 2:])
                current_q = torch.cat([q_init, next_q[:, :-1]], dim=1)

        seq = torch.cat([fine_token, z_fine_llm, caption_embeds], dim=1)
        outputs = model.llm.model(inputs_embeds=seq)
        logits = model.llm.lm_head(outputs.last_hidden_state)
        caption_logits = logits[:, 1+T:-1, :]

    # Compute per-token CE loss (no reduction)
    loss = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.size(-1)),
        caption_targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    # Filter out padding
    mask = caption_targets.reshape(-1) != tokenizer.pad_token_id
    valid_losses = loss[mask]

    return valid_losses.mean().item(), len(valid_losses)


@torch.no_grad()
def eval_smolvlm(model, processor, pil_frames, caption, device):
    """Compute caption CE loss for SmolVLM2.

    Only counts loss on the caption (assistant) tokens, not the prompt.
    """
    # Build chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video:"},
            ] + [{"type": "image"} for _ in pil_frames],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": caption},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(text=text, images=pil_frames, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    # Create labels: mask everything except caption tokens
    labels = input_ids.clone()
    caption_tokens = processor.tokenizer(
        caption, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].squeeze(0)
    caption_len = len(caption_tokens)
    if caption_len < labels.shape[1]:
        labels[:, :-caption_len] = -100

    # Forward
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    )

    # The model returns mean loss over non-masked tokens
    # We also need token count for proper averaging
    # Recompute to get per-token loss
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='none'
    )
    mask = shift_labels.view(-1) != -100
    valid_losses = per_token_loss[mask]

    return valid_losses.mean().item(), len(valid_losses)


@torch.no_grad()
def eval_blind(llm, tokenizer, caption, device):
    """Compute caption CE loss with no visual input (language-only baseline)."""
    tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)

    outputs = llm(input_ids=input_ids)
    logits = outputs.logits

    # Shifted targets
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # Load validation data
    val_samples = load_val_samples()
    N = len(val_samples)

    # Pre-process frames for all samples (deterministic, shared across models)
    print("Pre-processing frames...")
    for s in val_samples:
        s['frames_sub'] = subsample_frames(s['frames_raw'], NUM_FRAMES)
        s['frames_norm'] = normalize_frames(s['frames_sub'])
        s['frames_pil'] = frames_to_pil(s['frames_sub'])
        s['latents_sub'] = subsample_frames(s['latents'], NUM_FRAMES)

    results = {
        'sample_id': list(range(N)),
        'caption_len': [],
        'foveated_fine': [],
        'foveated_coarse': [],
        'smolvlm_pretrained': [],
        'smolvlm_finetuned': [],
        'blind': [],
    }

    # --- 1. Foveated model (fine + coarse) ---
    print("\n" + "=" * 60)
    print("Evaluating: Foveated VLM (fine + coarse)")
    print("=" * 60)

    from src.model.foveated_vlm import FoveatedVideoModel
    from transformers import AutoTokenizer

    fov_model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=False,
    ).to(device)

    if FOVEATED_CKPT.exists():
        ckpt = torch.load(FOVEATED_CKPT, map_location=device, weights_only=False)
        fov_model.load_state_dict(ckpt['model_state_dict'])
        fov_step = ckpt.get('step', '?')
        print(f"  Loaded foveated checkpoint: step {fov_step}")
        del ckpt
    else:
        print(f"  WARNING: No foveated checkpoint at {FOVEATED_CKPT}")
        print(f"  Using randomly initialized model (results will be meaningless)")

    fov_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i, s in enumerate(tqdm(val_samples, desc="Foveated")):
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
        if i == 0:
            results['caption_len'].clear()
        results['caption_len'].append(n_tok)

    # Free VRAM
    del fov_model
    torch.cuda.empty_cache()

    # --- 2. Blind baseline ---
    print("\n" + "=" * 60)
    print("Evaluating: Blind baseline (SmolLM2-135M, no vision)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM

    blind_llm = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.bfloat16,
    ).to(device)
    blind_llm.eval()

    for s in tqdm(val_samples, desc="Blind"):
        loss, _ = eval_blind(blind_llm, tokenizer, s['caption'], device)
        results['blind'].append(loss)

    del blind_llm
    torch.cuda.empty_cache()

    # --- 3. SmolVLM pretrained ---
    print("\n" + "=" * 60)
    print("Evaluating: SmolVLM2-256M-Video (pretrained)")
    print("=" * 60)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    smol_model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    smol_processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        trust_remote_code=True,
    )
    smol_model.eval()

    for s in tqdm(val_samples, desc="SmolVLM pretrained"):
        try:
            loss, _ = eval_smolvlm(smol_model, smol_processor, s['frames_pil'], s['caption'], device)
            results['smolvlm_pretrained'].append(loss)
        except Exception as e:
            print(f"  Error: {e}")
            results['smolvlm_pretrained'].append(float('nan'))

    del smol_model
    torch.cuda.empty_cache()

    # --- 4. SmolVLM fine-tuned ---
    print("\n" + "=" * 60)
    print("Evaluating: SmolVLM2-256M-Video (fine-tuned)")
    print("=" * 60)

    smol_ft = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if SMOLVLM_CKPT.exists():
        ckpt = torch.load(SMOLVLM_CKPT, map_location=device, weights_only=False)
        smol_ft.load_state_dict(ckpt['model_state_dict'])
        smol_step = ckpt.get('step', '?')
        print(f"  Loaded SmolVLM checkpoint: step {smol_step}")
        del ckpt
    else:
        print(f"  WARNING: No SmolVLM checkpoint at {SMOLVLM_CKPT}")
        print(f"  Skipping fine-tuned evaluation")
        results['smolvlm_finetuned'] = [float('nan')] * N

    if SMOLVLM_CKPT.exists():
        smol_ft.eval()
        for s in tqdm(val_samples, desc="SmolVLM fine-tuned"):
            try:
                loss, _ = eval_smolvlm(smol_ft, smol_processor, s['frames_pil'], s['caption'], device)
                results['smolvlm_finetuned'].append(loss)
            except Exception as e:
                print(f"  Error: {e}")
                results['smolvlm_finetuned'].append(float('nan'))

    del smol_ft
    torch.cuda.empty_cache()

    # ============================================================================
    # ANALYSIS
    # ============================================================================

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Convert to numpy, filter NaN
    model_names = ['foveated_fine', 'foveated_coarse', 'smolvlm_pretrained',
                    'smolvlm_finetuned', 'blind']
    display_names = ['Foveated (fine)', 'Foveated (coarse)', 'SmolVLM2 (pretrained)',
                     'SmolVLM2 (fine-tuned)', 'Blind (no vision)']

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
            'ci95_low': float(mean_loss - 1.96 * se),
            'ci95_high': float(mean_loss + 1.96 * se),
        }

        print(f"  {display:30s} | Loss: {mean_loss:.4f} +/- {se:.4f} | PPL: {ppl:.1f} | n={len(valid)}")

    # Visual contribution
    print("\nVisual Contribution (blind_loss - model_loss):")
    blind_mean = summary['blind']['mean']
    for name, display in zip(model_names[:-1], display_names[:-1]):
        if not np.isnan(summary[name]['mean']):
            contrib = blind_mean - summary[name]['mean']
            print(f"  {display:30s} | {contrib:+.4f} ({contrib/blind_mean*100:+.1f}%)")

    # Paired t-tests
    print("\nPaired t-tests (p-values):")
    pairs = [
        ('foveated_fine', 'foveated_coarse', 'Fine vs Coarse'),
        ('foveated_fine', 'smolvlm_pretrained', 'Foveated vs SmolVLM pretrained'),
        ('foveated_fine', 'smolvlm_finetuned', 'Foveated vs SmolVLM fine-tuned'),
        ('foveated_fine', 'blind', 'Foveated vs Blind'),
        ('smolvlm_pretrained', 'smolvlm_finetuned', 'SmolVLM pretrained vs fine-tuned'),
    ]
    for a, b, label in pairs:
        arr_a = np.array(results[a])
        arr_b = np.array(results[b])
        valid = ~np.isnan(arr_a) & ~np.isnan(arr_b)
        if valid.sum() > 1:
            t_stat, p_val = stats.ttest_rel(arr_a[valid], arr_b[valid])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {label:45s} | t={t_stat:+.3f} p={p_val:.4f} {sig}")

    # Efficiency comparison
    print("\nEfficiency Comparison:")
    print(f"  {'Model':30s} | {'Vis Tokens/Frame':>18s}")
    print(f"  {'Foveated':30s} | {'1':>18s}")
    print(f"  {'SmolVLM2':30s} | {'64':>18s}")

    # Save results
    # Per-sample CSV
    import csv
    csv_path = OUTPUT_DIR / "caption_loss_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['sample_id', 'caption_len'] + model_names
        writer.writerow(header)
        for i in range(N):
            row = [i, results['caption_len'][i] if i < len(results['caption_len']) else 0]
            for name in model_names:
                row.append(results[name][i] if i < len(results[name]) else float('nan'))
            writer.writerow(row)
    print(f"\nPer-sample results saved to: {csv_path}")

    # Summary JSON
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'val_samples': N,
            'num_frames': NUM_FRAMES,
            'foveated_checkpoint': str(FOVEATED_CKPT),
            'smolvlm_checkpoint': str(SMOLVLM_CKPT),
            'results': summary,
        }, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Markdown table
    md_path = OUTPUT_DIR / "comparison_table.md"
    with open(md_path, 'w') as f:
        f.write("# Caption Loss Comparison\n\n")
        f.write(f"Validation samples: {N} | Frames per video: {NUM_FRAMES}\n\n")
        f.write("| Model | Loss (mean) | SE | PPL | Vis Tokens/Frame | Visual Contrib |\n")
        f.write("|-------|------------|-----|-----|-----------------|----------------|\n")
        for name, display in zip(model_names, display_names):
            s = summary[name]
            vt = "1" if "foveated" in name else ("64" if "smolvlm" in name else "0")
            vc = f"{blind_mean - s['mean']:+.4f}" if not np.isnan(s['mean']) and name != 'blind' else "-"
            f.write(f"| {display} | {s['mean']:.4f} | {s['se']:.4f} | {s['ppl']:.1f} | {vt} | {vc} |\n")
    print(f"Markdown table saved to: {md_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
