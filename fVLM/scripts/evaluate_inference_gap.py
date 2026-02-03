#!/usr/bin/env python3
"""
Evaluate Training vs Inference Gap

This script compares:
1. Training loss (parallel approximation - queries from coarse features)
2. Autoregressive inference loss (true inference - queries from fine features)
3. Coarse baseline loss (static query only)

This quantifies the gap between what we measure during training
and what the model actually achieves during inference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import argparse
import random
import numpy as np
from pathlib import Path

from model.foveated_vlm import FoveatedVideoModel


# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ShardedVideoDataset(IterableDataset):
    """Sharded dataset that supports shard whitelist for train/val split."""

    def __init__(self, shard_dir: str, num_frames: int = 8, shard_whitelist: list = None):
        self.shard_dir = Path(shard_dir)
        self.num_frames = num_frames

        all_shards = sorted(self.shard_dir.glob("shard_*.pt"))

        if shard_whitelist is not None:
            # Filter to only whitelisted shards
            whitelist_set = set(shard_whitelist)
            self.shard_files = [s for s in all_shards if s.name in whitelist_set]
        else:
            self.shard_files = all_shards

        self.num_shards = len(self.shard_files)
        print(f"Using {self.num_shards} shards")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard_indices = list(range(self.num_shards))

        if worker_info is not None:
            per_worker = len(shard_indices) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(shard_indices)
            shard_indices = shard_indices[start:end]

        for shard_idx in shard_indices:
            try:
                shard = torch.load(self.shard_files[shard_idx], map_location="cpu", weights_only=False)
                samples = shard['samples']
                del shard
            except Exception:
                continue

            for sample in samples:
                yield self._process(sample)

            del samples

    def _process(self, data):
        frames = data['frames']
        latents = data['latents']
        caption = data['caption']

        T_orig = frames.shape[0]
        if T_orig > self.num_frames:
            indices = np.linspace(0, T_orig - 1, self.num_frames, dtype=int)
            frames = frames[indices]
            latents = latents[indices]

        frames = frames.float() / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        return {
            'frames': frames,
            'latents': latents.float(),
            'caption': caption,
        }


def load_model(checkpoint_path: str, device: str = 'cuda',
               llm_model: str = None, dino_model: str = None) -> FoveatedVideoModel:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        llm_model: Override LLM model (auto-inferred if None)
        dino_model: Override DINO model (auto-inferred if None)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer model config from checkpoint or weights
    config = checkpoint.get('config', {})

    # Infer LLM model from weight shapes
    if llm_model is None:
        llm_dim_from_weights = state_dict.get('coarse_token', None)
        if llm_dim_from_weights is not None:
            llm_dim = llm_dim_from_weights.shape[-1]
            if llm_dim == 576:
                llm_model = 'HuggingFaceTB/SmolLM2-135M-Instruct'
            elif llm_dim == 960:
                llm_model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
            else:
                llm_model = config.get('llm_model', 'HuggingFaceTB/SmolLM2-135M-Instruct')
        else:
            llm_model = config.get('llm_model', 'HuggingFaceTB/SmolLM2-135M-Instruct')
        print(f"Inferred LLM model: {llm_model}")

    # Infer DINO model from weight shapes
    if dino_model is None:
        dino_dim_from_weights = state_dict.get('q_static', None)
        if dino_dim_from_weights is not None:
            dino_dim = dino_dim_from_weights.shape[-1]
            if dino_dim == 384:
                dino_model = 'facebook/dinov2-small'
            elif dino_dim == 768:
                dino_model = 'facebook/dinov2-base'
            else:
                dino_model = config.get('dino_model', 'facebook/dinov2-small')
        else:
            dino_model = config.get('dino_model', 'facebook/dinov2-small')
        print(f"Inferred DINO model: {dino_model}")

    # Get dimensions
    dino_dim = {'facebook/dinov2-small': 384, 'facebook/dinov2-base': 768}.get(dino_model, 384)
    llm_dim = {'HuggingFaceTB/SmolLM2-135M-Instruct': 576, 'HuggingFaceTB/SmolLM2-360M-Instruct': 960}.get(llm_model, 576)
    query_dim = config.get('query_dim', dino_dim)
    deep_query = config.get('deep_query', True)
    freeze_dino = config.get('freeze_dino', False)

    model = FoveatedVideoModel(
        dino_model=dino_model,
        llm_model=llm_model,
        dino_dim=dino_dim,
        llm_dim=llm_dim,
        query_dim=query_dim,
        deep_query=deep_query,
        freeze_dino=freeze_dino,
    ).to(device)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def evaluate_checkpoint(
    model: FoveatedVideoModel,
    dataloader: DataLoader,
    tokenizer,
    device: str = 'cuda',
    max_samples: int = None,
) -> dict:
    """
    Evaluate a model checkpoint with three loss types.

    Returns:
        dict with keys:
            - loss_train_fine: Training loss with fine path (parallel)
            - loss_train_coarse: Training loss with coarse path (static)
            - loss_autoregressive: True autoregressive inference loss
            - gap: loss_autoregressive - loss_train_fine
            - samples: number of samples evaluated
    """
    model.eval()

    total_train_fine = 0.0
    total_train_coarse = 0.0
    total_autoregressive = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and total_samples >= max_samples:
                break

            frames = batch['frames'].to(device)  # [B, T, 3, H, W]
            captions = batch['caption']  # List of strings

            B = frames.shape[0]

            # Tokenize captions
            caption_enc = tokenizer(
                captions,
                return_tensors='pt',
                padding='max_length',
                max_length=64,
                truncation=True,
            )
            caption_ids = caption_enc['input_ids'].to(device)
            caption_mask = caption_enc['attention_mask'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Training loss (parallel approximation)
                loss_fine = model.forward_captioning(
                    frames, caption_ids, caption_mask, use_fine=True
                )
                loss_coarse = model.forward_captioning(
                    frames, caption_ids, caption_mask, use_fine=False
                )

                # True autoregressive inference loss
                loss_auto = model.forward_autoregressive_captioning(
                    frames, caption_ids, caption_mask
                )

            total_train_fine += loss_fine.item() * B
            total_train_coarse += loss_coarse.item() * B
            total_autoregressive += loss_auto.item() * B
            total_samples += B

    # Compute averages
    avg_train_fine = total_train_fine / total_samples
    avg_train_coarse = total_train_coarse / total_samples
    avg_autoregressive = total_autoregressive / total_samples

    return {
        'loss_train_fine': avg_train_fine,
        'loss_train_coarse': avg_train_coarse,
        'loss_autoregressive': avg_autoregressive,
        'gap': avg_autoregressive - avg_train_fine,
        'gap_percent': (avg_autoregressive - avg_train_fine) / avg_train_fine * 100,
        'samples': total_samples,
        'ppl_train_fine': torch.exp(torch.tensor(avg_train_fine)).item(),
        'ppl_autoregressive': torch.exp(torch.tensor(avg_autoregressive)).item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate training vs inference gap')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='/mnt/d/projects/fVLM/data/frames_latents_sharded',
                        help='Path to precomputed shards')
    parser.add_argument('--split-file', type=str, default=None,
                        help='Path to data split JSON (uses val shards)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Evaluation batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')

    # Load validation data
    shard_whitelist = None
    if args.split_file:
        with open(args.split_file) as f:
            split = json.load(f)
        shard_whitelist = split.get('val_shards', None)
        print(f"Using {len(shard_whitelist)} validation shards from split file")

    dataset = ShardedVideoDataset(
        shard_dir=args.data_dir,
        num_frames=8,
        shard_whitelist=shard_whitelist,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_checkpoint(
        model, dataloader, tokenizer, device,
        max_samples=args.max_samples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples evaluated:        {results['samples']}")
    print(f"")
    print(f"Training loss (fine):     {results['loss_train_fine']:.4f}  (PPL: {results['ppl_train_fine']:.2f})")
    print(f"Training loss (coarse):   {results['loss_train_coarse']:.4f}")
    print(f"Autoregressive loss:      {results['loss_autoregressive']:.4f}  (PPL: {results['ppl_autoregressive']:.2f})")
    print(f"")
    print(f"Train/Inference Gap:      {results['gap']:.4f}  ({results['gap_percent']:.2f}%)")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
