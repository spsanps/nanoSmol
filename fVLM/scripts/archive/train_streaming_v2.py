"""
Streaming Video-Text Training with Filtering

Key features:
- Streams from WebVid-10M (10M videos, never repeats)
- Filters videos by duration (no clipping)
- Text captions for guided attention
- Longer videos (32 frames)
- Efficient VAE computation with prefetching
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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str: str) -> int:
    """Parse duration string like 'PT00H00M11S' to seconds."""
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
    """Download video bytes."""
    response = requests.get(url, timeout=timeout, stream=True)
    if response.status_code == 200:
        return response.content
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames from video bytes using ffmpeg - uniform sampling."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
        f.write(video_bytes)
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract more frames than needed for uniform sampling
            cmd = [
                'ffmpeg', '-i', f.name,
                '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                '-frames:v', str(num_frames * 4),  # Extract extra for uniform sampling
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

            # Uniform sampling
            indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)

            frames = []
            for idx in indices:
                img = Image.open(frame_files[idx]).convert('RGB')
                frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                frames.append(frame)

            return torch.stack(frames)  # [T, 3, H, W] uint8


class StreamingVideoTextDataset(IterableDataset):
    """
    Streaming dataset from WebVid-10M with text captions.

    Features:
    - Filters by duration (no clipping)
    - Returns raw frames for VAE computation in training loop
    - Tokenizes captions
    """

    def __init__(
        self,
        num_frames: int = 32,
        frame_size: int = 256,
        min_duration: int = 10,  # Filter: minimum seconds
        max_duration: int = 30,  # Filter: maximum seconds
        max_text_tokens: int = 64,
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_text_tokens = max_text_tokens

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Stats
        self.stats = {
            'attempted': 0,
            'filtered_short': 0,
            'filtered_long': 0,
            'failed_download': 0,
            'failed_extract': 0,
            'success': 0,
        }

    def __iter__(self):
        # Load streaming dataset with retry
        for retry in range(5):
            try:
                ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
                break
            except Exception as e:
                if retry < 4:
                    time.sleep(2 ** retry)
                else:
                    raise e

        for sample in ds:
            self.stats['attempted'] += 1

            # Parse duration and FILTER (not clip)
            duration = parse_duration(sample.get('duration', ''))

            if duration < self.min_duration:
                self.stats['filtered_short'] += 1
                continue

            if duration > self.max_duration:
                self.stats['filtered_long'] += 1
                continue

            # Download video
            try:
                video_bytes = download_video(sample['contentUrl'])
                if video_bytes is None:
                    self.stats['failed_download'] += 1
                    continue
            except:
                self.stats['failed_download'] += 1
                continue

            # Extract frames (uniform sampling, not clipping)
            try:
                frames = extract_frames(video_bytes, self.num_frames, self.frame_size)
                if frames is None:
                    self.stats['failed_extract'] += 1
                    continue
            except:
                self.stats['failed_extract'] += 1
                continue

            # Tokenize caption
            caption = sample.get('name', '')
            tokens = self.tokenizer(
                caption,
                max_length=self.max_text_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.stats['success'] += 1

            yield {
                'frames_raw': frames,  # [T, 3, H, W] uint8
                'text_input_ids': tokens['input_ids'].squeeze(0),
                'text_attention_mask': tokens['attention_mask'].squeeze(0),
                'caption': caption,
                'video_id': str(sample.get('videoid', '')),
                'duration': duration,
            }


class PrefetchingDataset(IterableDataset):
    """Wraps dataset with background prefetching."""

    def __init__(self, base_dataset, buffer_size: int = 16):
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
            self.buffer.put(None)  # Signal end

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


def collate_streaming(batch):
    """Collate for streaming batches."""
    return {
        'frames_raw': torch.stack([b['frames_raw'] for b in batch]),
        'text_input_ids': torch.stack([b['text_input_ids'] for b in batch]),
        'text_attention_mask': torch.stack([b['text_attention_mask'] for b in batch]),
        'captions': [b['caption'] for b in batch],
        'video_ids': [b['video_id'] for b in batch],
        'durations': [b['duration'] for b in batch],
    }


def compute_vae_latents(frames_raw, vae, device):
    """Compute VAE latents efficiently."""
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


def main():
    # Config
    config = {
        'num_frames': 32,  # Longer videos
        'frame_size': 256,
        'min_duration': 10,  # Filter: at least 10 seconds
        'max_duration': 30,  # Filter: at most 30 seconds
        'max_text_tokens': 64,
        'batch_size': 2,
        'grad_accum': 8,  # Effective batch = 16
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'max_steps': 5000,  # ~1-2 hours
        'grad_clip': 1.0,
        'log_every': 50,
        'save_every': 1000,
        'output_dir': 'outputs/streaming_v2',
        'lambda_coarse': 1.0,
    }

    print("=" * 70)
    print("Streaming Video-Text Training")
    print("=" * 70)
    print(f"Frames per video: {config['num_frames']}")
    print(f"Duration filter: {config['min_duration']}-{config['max_duration']}s")
    print(f"Batch size: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"Max steps: {config['max_steps']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(config['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # W&B
    if HAS_WANDB:
        wandb.init(
            project="foveated-vlm",
            config=config,
            name=f"streaming_v2_{datetime.now().strftime('%m%d_%H%M')}",
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

    # Load model
    print("Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim=384,
        llm_dim=576,
        query_dim=384,
        lambda_coarse=config['lambda_coarse'],
    ).to(device)

    # Try to warm-start from Phase 2 checkpoint
    phase2_ckpt = Path('outputs/phase2/checkpoints/final.pt')
    if phase2_ckpt.exists():
        print(f"Loading checkpoint: {phase2_ckpt}")
        ckpt = torch.load(phase2_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Warm-started from step {ckpt.get('step', 'unknown')}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()

    # Create streaming dataset with prefetching
    print("\nCreating streaming dataset...")
    base_dataset = StreamingVideoTextDataset(
        num_frames=config['num_frames'],
        frame_size=config['frame_size'],
        min_duration=config['min_duration'],
        max_duration=config['max_duration'],
        max_text_tokens=config['max_text_tokens'],
    )
    dataset = PrefetchingDataset(base_dataset, buffer_size=16)

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        collate_fn=collate_streaming,
    )

    # Training
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    model.train()
    global_step = 0
    running_loss = 0
    running_fine = 0
    running_coarse = 0
    epoch = 0

    pbar = tqdm(total=config['max_steps'], desc="Training")

    while global_step < config['max_steps']:
        epoch += 1
        for batch in dataloader:
            if global_step >= config['max_steps']:
                break

            # Compute VAE latents
            frames_raw = batch['frames_raw']
            latents = compute_vae_latents(frames_raw, vae, device)

            # Normalize frames for DINO
            frames = normalize_for_dino(frames_raw, device)

            # Get text
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            text_embeds = model.get_text_embeds(text_ids, text_mask)

            # Forward
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, loss_fine, loss_coarse = model(text_embeds, frames, latents)
                loss = loss / config['grad_accum']

            # Backward
            scaler.scale(loss).backward()

            running_loss += loss.item() * config['grad_accum']
            running_fine += loss_fine.item()
            running_coarse += loss_coarse.item()

            # Optimizer step
            if (global_step + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            pbar.update(1)

            # Log
            if global_step % config['log_every'] == 0:
                avg_loss = running_loss / config['log_every']
                avg_fine = running_fine / config['log_every']
                avg_coarse = running_coarse / config['log_every']
                ratio = avg_coarse / (avg_fine + 1e-8)

                stats = base_dataset.stats
                success_rate = stats['success'] / max(stats['attempted'], 1) * 100

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'fine': f'{avg_fine:.4f}',
                    'ratio': f'{ratio:.3f}',
                    'success%': f'{success_rate:.0f}',
                })

                if HAS_WANDB:
                    wandb.log({
                        'loss': avg_loss,
                        'loss_fine': avg_fine,
                        'loss_coarse': avg_coarse,
                        'ratio': ratio,
                        'lr': scheduler.get_last_lr()[0],
                        'videos_attempted': stats['attempted'],
                        'videos_success': stats['success'],
                        'success_rate': success_rate,
                        'filtered_short': stats['filtered_short'],
                        'filtered_long': stats['filtered_long'],
                        'avg_duration': sum(batch['durations']) / len(batch['durations']),
                    }, step=global_step)

                running_loss = 0
                running_fine = 0
                running_coarse = 0

            # Save
            if global_step % config['save_every'] == 0:
                path = output_dir / 'checkpoints' / f'step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'stats': base_dataset.stats,
                }, path)
                print(f"\nSaved: {path}")

    pbar.close()
    dataset.stop()

    # Final save
    final_path = output_dir / 'checkpoints' / 'final.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': config,
        'stats': base_dataset.stats,
    }, final_path)

    # Print stats
    stats = base_dataset.stats
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final checkpoint: {final_path}")
    print(f"\nDataset stats:")
    print(f"  Videos attempted: {stats['attempted']}")
    print(f"  Videos used: {stats['success']}")
    print(f"  Filtered (too short): {stats['filtered_short']}")
    print(f"  Filtered (too long): {stats['filtered_long']}")
    print(f"  Failed downloads: {stats['failed_download']}")
    print(f"  Failed extractions: {stats['failed_extract']}")
    print(f"  Success rate: {stats['success'] / max(stats['attempted'], 1) * 100:.1f}%")

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
