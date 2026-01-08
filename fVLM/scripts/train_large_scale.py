#!/usr/bin/env python3
"""
Large-Scale Multi-Task Training

Designed for 24-hour training run on RTX 4090 (~20GB VRAM).
Streams data from WebVid-10M, mixes training modes:
  - Video-only reconstruction (self-supervised)
  - Text-conditioned reconstruction
  - Video captioning

Key features:
  - Streaming data (no 100GB download needed)
  - Dynamic mode mixing (curriculum learning)
  - Checkpoint resume support
  - Proper logging and monitoring
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
import threading
import queue
import random
import argparse
import json

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

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
    """Download video from URL with retries."""
    for retry in range(3):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                return response.content
        except:
            if retry < 2:
                time.sleep(0.5 * (retry + 1))
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


class LargeScaleDataset(IterableDataset):
    """
    Streaming dataset for large-scale multi-task training.

    Mode is selected per-batch (not per-sample) to allow proper batch processing.
    """

    def __init__(
        self,
        num_frames: int = 16,
        frame_size: int = 256,
        min_duration: int = 5,
        max_duration: int = 30,
        max_caption_tokens: int = 64,
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        seed: int = None,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_caption_tokens = max_caption_tokens
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stats = {
            'attempted': 0, 'success': 0, 'filtered': 0, 'failed': 0,
        }

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)

        # Retry loading dataset
        for retry in range(5):
            try:
                ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
                break
            except Exception as e:
                if retry < 4:
                    time.sleep(2 ** retry)
                else:
                    raise e

        # Shuffle for variety
        ds = ds.shuffle(seed=self.seed or 42, buffer_size=1000)

        for sample in ds:
            self.stats['attempted'] += 1

            # Filter by duration
            duration = parse_duration(sample.get('duration', ''))
            if duration < self.min_duration or duration > self.max_duration:
                self.stats['filtered'] += 1
                continue

            # Download video
            try:
                video_bytes = download_video(sample['contentUrl'])
                if video_bytes is None:
                    self.stats['failed'] += 1
                    continue
            except:
                self.stats['failed'] += 1
                continue

            # Extract frames
            try:
                frames_raw = extract_frames(video_bytes, self.num_frames, self.frame_size)
                if frames_raw is None:
                    self.stats['failed'] += 1
                    continue
            except:
                self.stats['failed'] += 1
                continue

            # Tokenize caption
            caption = sample.get('name', '')
            tokens = self.tokenizer(
                caption,
                max_length=self.max_caption_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.stats['success'] += 1

            yield {
                'frames_raw': frames_raw,  # [T, 3, H, W] uint8
                'caption_ids': tokens['input_ids'].squeeze(0),
                'caption_mask': tokens['attention_mask'].squeeze(0),
                'caption': caption,
            }


class PrefetchingDataset(IterableDataset):
    """Prefetch samples in background thread for better throughput."""

    def __init__(self, base_dataset, buffer_size: int = 32):
        self.base = base_dataset
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()

    def _producer(self):
        try:
            for sample in self.base:
                if self.stop_event.is_set():
                    break
                self.buffer.put(sample)
        finally:
            self.buffer.put(None)

    def __iter__(self):
        self.stop_event.clear()
        producer = threading.Thread(target=self._producer, daemon=True)
        producer.start()
        while True:
            sample = self.buffer.get()
            if sample is None:
                break
            yield sample

    def stop(self):
        self.stop_event.set()


def collate_fn(batch):
    """Collate batch."""
    return {
        'frames_raw': torch.stack([b['frames_raw'] for b in batch]),
        'caption_ids': torch.stack([b['caption_ids'] for b in batch]),
        'caption_mask': torch.stack([b['caption_mask'] for b in batch]),
        'captions': [b['caption'] for b in batch],
    }


def compute_vae_latents(frames_raw, vae, device):
    """Compute VAE latents from raw frames."""
    B, T, C, H, W = frames_raw.shape
    frames_flat = frames_raw.view(B * T, C, H, W).float().to(device)
    frames_vae = frames_flat / 255.0 * 2 - 1  # [-1, 1]

    with torch.no_grad():
        chunk_size = 16
        latent_chunks = []
        for i in range(0, frames_vae.shape[0], chunk_size):
            chunk = frames_vae[i:i+chunk_size].half()
            latent = vae.encode(chunk).latent_dist.sample() * 0.18215
            latent_chunks.append(latent)
        latents = torch.cat(latent_chunks, dim=0)

    return latents.view(B, T, 4, 32, 32).float()


def normalize_for_dino(frames_raw, device):
    """Normalize raw frames for DINO."""
    frames = frames_raw.float().to(device) / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames - mean) / std


def save_checkpoint(model, optimizer, scheduler, scaler, step, config, stats, output_dir, name='checkpoint'):
    """Save training checkpoint."""
    path = output_dir / 'checkpoints' / f'{name}_step_{step}.pt'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
        'stats': stats,
    }, path)
    print(f"\n  Saved: {path}")
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device='cuda'):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['step'], checkpoint.get('stats', {})


def main():
    parser = argparse.ArgumentParser(description='Large-Scale Multi-Task Training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/large_scale')
    parser.add_argument('--max_hours', type=float, default=24.0, help='Max training hours')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--mode_weights', type=str, default='0.6,0.2,0.2',
                       help='Weights for video-only,text-cond,captioning')
    parser.add_argument('--freeze_dino', action='store_true',
                       help='Freeze DINO backbone (ablation winner!)')
    args = parser.parse_args()

    # Parse mode weights
    mode_weights = tuple(float(x) for x in args.mode_weights.split(','))

    config = {
        'num_frames': 16,
        'frame_size': 256,
        'min_duration': 5,
        'max_duration': 30,
        'max_caption_tokens': 64,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'max_hours': args.max_hours,
        'grad_clip': 1.0,
        'lambda_caption': 0.1,
        'lambda_coarse': 0.5,
        'mode_weights': mode_weights,
        'log_every': 100,
        'save_every': 2000,
        'eval_every': 500,
        'output_dir': args.output_dir,
        'freeze_dino': args.freeze_dino,
    }

    print("=" * 70)
    print("Large-Scale Multi-Task Training")
    if args.freeze_dino:
        print(">>> DINO FROZEN (ablation winner!) <<<")
    print("=" * 70)
    print(f"Max duration: {config['max_hours']} hours")
    print(f"Batch: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Mode weights: video-only={mode_weights[0]:.1%}, text-cond={mode_weights[1]:.1%}, caption={mode_weights[2]:.1%}")
    print(f"Freeze DINO: {args.freeze_dino}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(config['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize W&B
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project="foveated-vlm-large",
            config=config,
            name=f"large_{datetime.now().strftime('%m%d_%H%M')}",
            resume='allow' if args.resume else None,
        )

    # Load VAE
    print("\nLoading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Create model
    print("Loading model...")
    model_cfg = {
        'dino_model': 'facebook/dinov2-small',
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_dim': 384,
        'llm_dim': 576,
        'query_dim': 128,
        'lambda_coarse': config['lambda_coarse'],
        'freeze_dino': args.freeze_dino,
    }
    model = FoveatedVideoModel(**model_cfg).to(device)

    if args.freeze_dino:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"DINO frozen! Trainable: {trainable/1e6:.1f}M, Frozen: {frozen/1e6:.1f}M")

    # Enable gradient checkpointing
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        # Cosine decay after warmup (assuming ~50K steps for 24hrs)
        progress = (step - config['warmup_steps']) / 50000
        return 0.1 + 0.9 * (1 + np.cos(np.pi * min(progress, 1))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_step, _ = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        print(f"Resumed at step {start_step}")

    # Dataset
    print("\nCreating streaming dataset...")
    base_dataset = LargeScaleDataset(
        num_frames=config['num_frames'],
        frame_size=config['frame_size'],
        min_duration=config['min_duration'],
        max_duration=config['max_duration'],
        max_caption_tokens=config['max_caption_tokens'],
        llm_model=model_cfg['llm_model'],
    )
    dataset = PrefetchingDataset(base_dataset, buffer_size=32)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        collate_fn=collate_fn
    )

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    model.train()
    global_step = start_step
    start_time = time.time()
    max_seconds = config['max_hours'] * 3600

    # Running stats
    running = {
        'loss': 0, 'recon': 0, 'caption': 0, 'fine': 0, 'coarse': 0,
        'mode_0': 0, 'mode_1': 0, 'mode_2': 0,
    }
    mode_weights = config['mode_weights']

    pbar = tqdm(desc="Training", initial=start_step)

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                print(f"\nReached max time ({config['max_hours']}h)")
                break

            for batch in dataloader:
                elapsed = time.time() - start_time
                if elapsed >= max_seconds:
                    break

                frames_raw = batch['frames_raw']
                caption_ids = batch['caption_ids'].to(device)
                caption_mask = batch['caption_mask'].to(device)

                # Compute VAE latents
                latents = compute_vae_latents(frames_raw, vae, device)
                frames = normalize_for_dino(frames_raw, device)

                # Sample mode for entire batch (not per-sample!)
                # This ensures proper batch processing like train_multitask.py
                mode = random.choices([0, 1, 2], weights=mode_weights)[0]
                running[f'mode_{mode}'] += 1

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if mode == 0:
                        # Video-only reconstruction (self-supervised)
                        text_embeds = model.get_empty_text_embeds(frames.shape[0])
                        loss_recon, loss_fine, loss_coarse = model(text_embeds, frames, latents)
                        loss = loss_recon
                        loss_caption = torch.tensor(0.0, device=device)

                    elif mode == 1:
                        # Text-conditioned reconstruction
                        text_embeds = model.get_text_embeds(caption_ids, caption_mask)
                        loss_recon, loss_fine, loss_coarse = model(text_embeds, frames, latents)
                        loss = loss_recon
                        loss_caption = torch.tensor(0.0, device=device)

                    else:  # mode == 2
                        # Video captioning
                        loss_caption = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)
                        loss = config['lambda_caption'] * loss_caption
                        loss_recon = torch.tensor(0.0, device=device)
                        loss_fine = torch.tensor(0.0, device=device)
                        loss_coarse = torch.tensor(0.0, device=device)

                    # Scale for gradient accumulation
                    loss = loss / config['grad_accum']

                scaler.scale(loss).backward()

                # Update running stats
                running['loss'] += loss.item() * config['grad_accum']
                if mode in [0, 1]:
                    running['recon'] += loss_recon.item()
                    running['fine'] += loss_fine.item()
                    running['coarse'] += loss_coarse.item()
                else:
                    running['caption'] += loss_caption.item()

                # Gradient step
                if (global_step + 1) % config['grad_accum'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                global_step += 1
                pbar.update(1)

                # Logging
                if global_step % config['log_every'] == 0:
                    n = config['log_every']
                    # Compute averages only over steps that had reconstruction
                    n_recon = running['mode_0'] + running['mode_1']
                    n_cap = running['mode_2']

                    avg_loss = running['loss'] / n
                    avg_fine = running['fine'] / max(n_recon, 1)
                    avg_coarse = running['coarse'] / max(n_recon, 1)
                    avg_recon = running['recon'] / max(n_recon, 1)
                    avg_caption = running['caption'] / max(n_cap, 1)
                    ratio = avg_coarse / (avg_fine + 1e-8) if avg_fine > 0 else 1.0

                    stats = base_dataset.stats
                    success_rate = stats['success'] / max(stats['attempted'], 1) * 100
                    hours_left = (max_seconds - elapsed) / 3600

                    pbar.set_postfix({
                        'L': f'{avg_loss:.3f}',
                        'fine': f'{avg_fine:.3f}',
                        'coarse': f'{avg_coarse:.3f}',
                        'r': f'{ratio:.2f}',
                        'h': f'{hours_left:.1f}',
                    })

                    if HAS_WANDB and not args.no_wandb:
                        wandb.log({
                            'loss_total': avg_loss,
                            'loss_recon': avg_recon,
                            'loss_caption': avg_caption,
                            'loss_fine': avg_fine,
                            'loss_coarse': avg_coarse,
                            'ratio': ratio,
                            'lr': scheduler.get_last_lr()[0],
                            'success_rate': success_rate,
                            'mode_video_only': running['mode_0'] / n,
                            'mode_text_cond': running['mode_1'] / n,
                            'mode_captioning': running['mode_2'] / n,
                            'hours_elapsed': elapsed / 3600,
                            'samples_seen': stats['success'],
                        }, step=global_step)

                    # Reset running stats
                    running = {k: 0 for k in running}

                # Eval
                if global_step % config['eval_every'] == 0:
                    model.eval()
                    with torch.no_grad():
                        sample_frames = normalize_for_dino(batch['frames_raw'][:1], device)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            gen_caption = model.generate_caption(
                                sample_frames, tokenizer,
                                max_new_tokens=30, temperature=0.7
                            )
                        print(f"\n  GT: {batch['captions'][0][:60]}...")
                        print(f"  Gen: {gen_caption[0][:60]}...")
                    model.train()

                # Save checkpoint
                if global_step % config['save_every'] == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, config, base_dataset.stats, output_dir
                    )

                # Periodic cache clear
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")

    finally:
        pbar.close()
        dataset.stop()

        # Final save
        final_path = save_checkpoint(
            model, optimizer, scheduler, scaler,
            global_step, config, base_dataset.stats, output_dir, name='final'
        )

        elapsed_hours = (time.time() - start_time) / 3600
        stats = base_dataset.stats

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Steps: {global_step}")
        print(f"Time: {elapsed_hours:.2f} hours")
        print(f"Samples seen: {stats['success']}")
        print(f"Success rate: {stats['success'] / max(stats['attempted'], 1) * 100:.1f}%")
        print(f"Final checkpoint: {final_path}")

        if HAS_WANDB and not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
