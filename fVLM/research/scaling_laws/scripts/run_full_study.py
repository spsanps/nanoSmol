#!/usr/bin/env python3
"""
Run complete scaling law study: train + evaluate + analyze.

Single script that:
1. Trains foveated and baseline at multiple step counts
2. Evaluates all checkpoints on validation set
3. Generates plots and analysis
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

STEP_COUNTS = [100, 300, 1000, 3000]  # Checkpoints to save/evaluate

CONFIGS = {
    "foveated_opt": {
        "model_type": "foveated",
        "frame_size": 224,
        "fine_iterations": 1,
        "flops_per_sample": 144.7e9,
        "visual_tokens_per_frame": 1,
    },
    "baseline": {
        "model_type": "baseline",
        "frame_size": 224,
        "fine_iterations": 0,
        "flops_per_sample": 151.3e9,
        "visual_tokens_per_frame": 16,
    },
}

COMMON = {
    "batch_size": 16,
    "learning_rate": 3e-5,
    "warmup_steps": 50,
    "num_frames": 8,
    "lambda_recon": 0.5,
}

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
SPLIT_FILE = PROJECT_ROOT / "configs" / "data_split.json"
OUTPUT_BASE = Path("/mnt/d/projects/fVLM/outputs/scaling_study")
RESEARCH_DIR = Path(__file__).parent.parent

MAX_VAL_SAMPLES = 1500  # Balance speed vs accuracy

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================================
# Dataset
# ============================================================================

class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir, shard_list, num_frames, frame_size):
        self.shard_dir = Path(shard_dir)
        self.shard_files = [self.shard_dir / s for s in shard_list if (self.shard_dir / s).exists()]
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __iter__(self):
        for shard_path in self.shard_files:
            try:
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                for sample in shard['samples']:
                    yield self._process(sample)
                del shard
            except Exception:
                continue

    def _process(self, data):
        frames = data['frames']
        latents = data['latents']
        caption = data['caption']

        T = frames.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            indices = list(range(T)) + [T-1] * (self.num_frames - T)
            indices = torch.tensor(indices)

        frames = frames[indices].float()
        latents = latents[indices].float()

        if frames.shape[-1] != self.frame_size:
            frames = F.interpolate(frames, size=(self.frame_size, self.frame_size), mode='bilinear', align_corners=False)

        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames, 'latents': latents, 'caption': caption}


def collate_fn(batch):
    return {
        'frames': torch.stack([b['frames'] for b in batch]),
        'latents': torch.stack([b['latents'] for b in batch]),
        'captions': [b['caption'] for b in batch],
    }


# ============================================================================
# Training
# ============================================================================

def train_model(model, model_type, tokenizer, dataloader, config, device, max_steps, checkpoint_steps, exp_name):
    """Train model and save checkpoints."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    exp_dir = OUTPUT_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_saved = []
    step = 0
    model.train()
    start_time = time.time()
    data_iter = iter(dataloader)

    pbar = tqdm(total=max_steps, desc=f"Training {exp_name}")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        frames = batch['frames'].to(device)
        latents = batch['latents'].to(device)
        captions = batch['captions']

        tokens = tokenizer(captions, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        caption_ids = tokens['input_ids']
        caption_mask = tokens['attention_mask']

        if model_type == "foveated":
            # Use forward_captioning which takes caption_ids directly
            # use_fine=True for fine iterations > 0
            use_fine = config['fine_iterations'] > 0
            loss = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=use_fine)
        else:
            # BaselineVLM.forward needs caption_embeds and caption_targets
            caption_embeds = model.llm.model.embed_tokens(caption_ids)
            caption_targets = caption_ids[:, 1:].clone()
            # Mask padding tokens in targets
            caption_targets[caption_targets == tokenizer.pad_token_id] = -100
            loss, _ = model.forward(frames, caption_embeds, caption_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Warmup
        if step <= config['warmup_steps']:
            for pg in optimizer.param_groups:
                pg['lr'] = config['learning_rate'] * (step / config['warmup_steps'])

        # Save checkpoint
        if step in checkpoint_steps:
            ckpt_path = exp_dir / f"step_{step:06d}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'loss': loss.item(),
                'elapsed': time.time() - start_time,
            }, ckpt_path)
            checkpoints_saved.append(step)
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'saved': step})

        pbar.update(1)

    pbar.close()
    return checkpoints_saved


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_checkpoint(model, model_type, tokenizer, samples, fine_iterations, device):
    """Evaluate a single checkpoint."""
    model.eval()
    losses = []

    for s in samples:
        frames = s['frames'].unsqueeze(0).to(device)
        caption = s['caption']

        tokens = tokenizer(caption, truncation=True, max_length=64, return_tensors='pt')
        caption_ids = tokens['input_ids'].to(device)
        caption_targets = caption_ids[:, 1:]

        B, T = 1, frames.shape[1]

        if model_type == "foveated":
            # Encode
            frames_flat = frames.reshape(B * T, 3, frames.shape[-2], frames.shape[-1])
            _, cache_flat = model.encoder.encode_patches(frames_flat)
            pf = cache_flat['patch_features']
            N, D = pf.shape[1], pf.shape[2]
            patch_features = pf.reshape(B, T, N, D)

            all_caches = []
            if 'kv_cache' in cache_flat:
                nl = len(cache_flat['kv_cache'])
                K_all = [cache_flat['kv_cache'][i]['K'].reshape(B,T,N,D) for i in range(nl)]
                V_all = [cache_flat['kv_cache'][i]['V'].reshape(B,T,N,D) for i in range(nl)]
                layers = [cache_flat['kv_cache'][i]['layer'] for i in range(nl)]
                for t in range(T):
                    kv = [{'K': K_all[i][:,t], 'V': V_all[i][:,t], 'layer': layers[i]} for i in range(nl)]
                    all_caches.append({'patch_features': patch_features[:,t], 'kv_cache': kv})
            else:
                all_caches = [{'patch_features': patch_features[:,t]} for t in range(T)]

            # Coarse
            q_static = model.q_static.expand(B, -1)
            z_coarse = torch.stack([model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)], dim=1)
            z_llm = model.dino_to_llm(z_coarse)
            z_llm = z_llm / (z_llm.std() + 1e-6) * model.visual_scale

            # Fine iterations
            if fine_iterations > 0:
                no_text = model.no_text_token.expand(B, -1, -1)
                coarse_tok = model.coarse_token.expand(B, -1, -1)
                fine_tok = model.fine_token.expand(B, -1, -1)

                seq_q = torch.cat([no_text, coarse_tok, z_llm], dim=1)
                out_q = model.llm.model(inputs_embeds=seq_q)
                queries = model.llm_to_query(out_q.last_hidden_state[:, 2:])
                q_init = model.q_init.expand(B, -1).unsqueeze(1)
                current_q = torch.cat([q_init, queries[:, :-1]], dim=1)

                for it in range(fine_iterations):
                    z_fine = torch.stack([model.encoder.query_attend(current_q[:,t], all_caches[t]) for t in range(T)], dim=1)
                    z_llm = model.dino_to_llm(z_fine)
                    z_llm = z_llm / (z_llm.std() + 1e-6) * model.visual_scale
                    if it < fine_iterations - 1:
                        seq_q2 = torch.cat([no_text, fine_tok, z_llm], dim=1)
                        out_q2 = model.llm.model(inputs_embeds=seq_q2)
                        next_q = model.llm_to_query(out_q2.last_hidden_state[:, 2:])
                        current_q = torch.cat([q_init, next_q[:, :-1]], dim=1)

                mode_tok = fine_tok
            else:
                mode_tok = model.coarse_token.expand(B, -1, -1)

            caption_embeds = model.llm.model.embed_tokens(caption_ids)
            seq = torch.cat([mode_tok, z_llm, caption_embeds], dim=1)
            outputs = model.llm.model(inputs_embeds=seq)
            logits = model.llm.lm_head(outputs.last_hidden_state)
            caption_logits = logits[:, 1+T:-1, :]

        else:  # baseline
            visual_features = model.encode_frames(frames)
            num_tok = visual_features.shape[2]
            visual_features = visual_features.reshape(B, T * num_tok, model.llm_dim)

            caption_embeds = model.llm.model.embed_tokens(caption_ids)
            visual_token = model.visual_token.expand(B, -1, -1)
            seq = torch.cat([visual_token, visual_features, caption_embeds], dim=1)

            outputs = model.llm.model(inputs_embeds=seq)
            logits = model.llm.lm_head(outputs.last_hidden_state)
            visual_len = 1 + T * num_tok
            caption_logits = logits[:, visual_len:-1, :]

        loss = F.cross_entropy(
            caption_logits.reshape(-1, caption_logits.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='none'
        )
        mask = caption_targets.reshape(-1) != tokenizer.pad_token_id
        losses.append(loss[mask].mean().item())

    return np.array(losses)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("SCALING LAW STUDY")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load data split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_shards = split['train_shards']
    val_shards = split['val_shards']

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    max_steps = max(STEP_COUNTS)

    # ========== TRAINING PHASE ==========
    for exp_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"TRAINING: {exp_name}")
        print(f"{'='*60}")

        # Create dataset
        dataset = ShardedDataset(
            SHARD_DIR, train_shards,
            COMMON['num_frames'], cfg['frame_size']
        )
        dataloader = DataLoader(dataset, batch_size=COMMON['batch_size'], collate_fn=collate_fn, num_workers=2)

        # Create model
        if cfg['model_type'] == "foveated":
            from src.model.foveated_vlm import FoveatedVideoModel
            model = FoveatedVideoModel(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                deep_query=True, freeze_dino=False,
            ).to(device)
        else:
            from src.model.baseline_vlm import BaselineVLM
            model = BaselineVLM(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                pixel_shuffle_scale=4,
            ).to(device)

        config = {**COMMON, **cfg}
        train_model(model, cfg['model_type'], tokenizer, dataloader, config, device, max_steps, set(STEP_COUNTS), exp_name)

        del model
        torch.cuda.empty_cache()

    # ========== EVALUATION PHASE ==========
    print(f"\n{'='*60}")
    print("EVALUATION PHASE")
    print(f"{'='*60}")

    for exp_name, cfg in CONFIGS.items():
        print(f"\nEvaluating: {exp_name}")

        # Load val samples
        val_dataset = ShardedDataset(SHARD_DIR, val_shards, COMMON['num_frames'], cfg['frame_size'])
        samples = []
        for s in val_dataset:
            samples.append(s)
            if len(samples) >= MAX_VAL_SAMPLES:
                break
        print(f"  Loaded {len(samples)} val samples")

        # Create model
        if cfg['model_type'] == "foveated":
            from src.model.foveated_vlm import FoveatedVideoModel
            model = FoveatedVideoModel(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                deep_query=True, freeze_dino=False,
            ).to(device)
        else:
            from src.model.baseline_vlm import BaselineVLM
            model = BaselineVLM(
                dino_model="facebook/dinov2-small",
                llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                pixel_shuffle_scale=4,
            ).to(device)

        exp_dir = OUTPUT_BASE / exp_name
        for ckpt_path in sorted(exp_dir.glob("step_*.pt")):
            step = int(ckpt_path.stem.split('_')[1])
            print(f"  Step {step}...", end=" ")

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            losses = evaluate_checkpoint(model, cfg['model_type'], tokenizer, samples, cfg['fine_iterations'], device)

            mean_loss = losses.mean()
            std_loss = losses.std()
            se_loss = std_loss / np.sqrt(len(losses))
            ppl = np.exp(mean_loss)
            total_flops = step * COMMON['batch_size'] * cfg['flops_per_sample'] * 3

            print(f"loss={mean_loss:.4f}, ppl={ppl:.1f}")

            all_results.append({
                'experiment': exp_name,
                'model_type': cfg['model_type'],
                'frame_size': cfg['frame_size'],
                'fine_iterations': cfg['fine_iterations'],
                'step': step,
                'val_loss': float(mean_loss),
                'val_loss_std': float(std_loss),
                'val_loss_se': float(se_loss),
                'perplexity': float(ppl),
                'n_samples': len(losses),
                'flops_per_sample': cfg['flops_per_sample'],
                'total_training_flops': total_flops,
                'visual_tokens_per_frame': cfg['visual_tokens_per_frame'],
            })

        del model
        torch.cuda.empty_cache()

    # ========== SAVE RESULTS ==========
    data_dir = RESEARCH_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "scaling_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    json_path = data_dir / "scaling_data.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")

    # ========== PRINT SUMMARY ==========
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Experiment':<20} {'Step':>6} {'Loss':>8} {'PPL':>8} {'FLOPs (PF)':>12}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: (x['experiment'], x['step'])):
        print(f"{r['experiment']:<20} {r['step']:>6} {r['val_loss']:>8.4f} {r['perplexity']:>8.1f} {r['total_training_flops']/1e15:>12.4f}")

    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Now run: python analyze_scaling_laws.py")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
