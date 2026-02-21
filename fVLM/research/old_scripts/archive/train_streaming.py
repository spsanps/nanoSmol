"""
Streaming Training for Foveated VLM

Downloads and processes videos on-the-fly from WebVid-10M.
No disk storage needed. Never repeats samples (10M available).
No validation during training - just monitor train loss.

This is how large-scale pretraining works:
- Stream infinite data
- Train for X steps
- Evaluate separately after training
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import re
from datasets import load_dataset
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel


# Constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str):
    match = re.match(r'PT(\d+)H(\d+)M(\d+)S', dur_str)
    if match:
        return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
    return 0


def download_and_process(url, num_frames, frame_size, vae, device):
    """Download video, extract frames, compute latents. All in memory."""
    try:
        # Download
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        # Extract frames via temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    'ffmpeg', '-i', temp_path,
                    '-vf', f'scale={frame_size}:{frame_size}:force_original_aspect_ratio=increase,crop={frame_size}:{frame_size}',
                    '-frames:v', str(num_frames * 3),
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
                frames = torch.stack(frames)  # [T, 3, H, W] uint8

        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Compute latents
        frames_vae = frames.float().to(device) / 255.0 * 2 - 1
        with torch.no_grad():
            latents = []
            for i in range(0, frames_vae.shape[0], 4):
                batch = frames_vae[i:i+4]
                latent = vae.encode(batch).latent_dist.sample() * 0.18215
                latents.append(latent)
            latents = torch.cat(latents, dim=0).cpu()

        # Normalize frames for DINO
        frames_norm = frames.float() / 255.0
        frames_norm = (frames_norm - IMAGENET_MEAN) / IMAGENET_STD

        return {'frames': frames_norm, 'latents': latents}

    except Exception:
        return None


class StreamingDataLoader:
    """
    Streams from WebVid-10M with parallel prefetching.

    Architecture:
    - URL thread: streams URLs from HuggingFace
    - Download threads: download and extract frames (CPU)
    - Main thread: compute latents (GPU) and batch
    """

    def __init__(
        self,
        vae,
        batch_size=8,
        num_frames=16,
        frame_size=256,
        min_duration=8,
        max_duration=60,
        num_download_workers=8,
        prefetch_batches=4,
        device='cuda',
    ):
        self.vae = vae
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.num_workers = num_download_workers
        self.prefetch_batches = prefetch_batches
        self.device = device

        # Queues - large buffers to decouple download from training
        self.url_queue = queue.Queue(maxsize=5000)
        self.frames_queue = queue.Queue(maxsize=prefetch_batches * batch_size * 4)

        # Control
        self.stop_event = threading.Event()
        self.threads = []

        # Stats
        self.samples_yielded = 0
        self.download_failures = 0

    def _url_producer(self):
        """Stream URLs from WebVid."""
        ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
        for sample in ds:
            if self.stop_event.is_set():
                break
            duration = parse_duration(sample['duration'])
            if self.min_duration <= duration <= self.max_duration:
                self.url_queue.put(sample['contentUrl'])
        # Signal end
        for _ in range(self.num_workers):
            self.url_queue.put(None)

    def _download_worker(self):
        """Download and extract frames."""
        while not self.stop_event.is_set():
            try:
                url = self.url_queue.get(timeout=1)
            except queue.Empty:
                continue
            if url is None:
                break

            # Download and extract (no VAE yet - CPU only)
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    self.download_failures += 1
                    continue

                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name

                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        cmd = [
                            'ffmpeg', '-i', temp_path,
                            '-vf', f'scale={self.frame_size}:{self.frame_size}:force_original_aspect_ratio=increase,crop={self.frame_size}:{self.frame_size}',
                            '-frames:v', str(self.num_frames * 3),
                            '-q:v', '2',
                            f'{tmpdir}/frame_%04d.jpg',
                            '-y', '-loglevel', 'error'
                        ]
                        result = subprocess.run(cmd, capture_output=True, timeout=60)
                        if result.returncode != 0:
                            self.download_failures += 1
                            continue

                        frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                        if len(frame_files) < self.num_frames:
                            self.download_failures += 1
                            continue

                        indices = np.linspace(0, len(frame_files) - 1, self.num_frames).astype(int)
                        frames = []
                        for idx in indices:
                            img = Image.open(frame_files[idx]).convert('RGB')
                            frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                            frames.append(frame)
                        frames = torch.stack(frames)  # uint8

                        self.frames_queue.put(frames)
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            except Exception:
                self.download_failures += 1

    def start(self, warmup_batches=8):
        """Start background threads and wait for initial buffer."""
        self.stop_event.clear()

        # URL producer
        t = threading.Thread(target=self._url_producer, daemon=True)
        t.start()
        self.threads.append(t)

        # Download workers
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._download_worker, daemon=True)
            t.start()
            self.threads.append(t)

        # Warmup: wait for initial buffer to fill
        target = warmup_batches * self.batch_size
        print(f"Warming up: waiting for {target} videos in queue...")
        import time
        while self.frames_queue.qsize() < target:
            time.sleep(1)
            print(f"  Queue: {self.frames_queue.qsize()}/{target}", end='\r')
        print(f"\nQueue ready: {self.frames_queue.qsize()} videos buffered")

    def stop(self):
        """Stop all threads."""
        self.stop_event.set()

    def get_batch(self):
        """Get a batch, computing VAE latents on GPU."""
        frames_list = []
        while len(frames_list) < self.batch_size:
            try:
                frames = self.frames_queue.get(timeout=5)
                frames_list.append(frames)
            except queue.Empty:
                if all(not t.is_alive() for t in self.threads[1:]):  # download workers dead
                    break

        if not frames_list:
            return None

        # Stack and process batch
        frames_batch = torch.stack(frames_list)  # [B, T, 3, H, W] uint8

        # Compute latents on GPU (fp16 for speed)
        B, T = frames_batch.shape[:2]
        frames_flat = frames_batch.reshape(B * T, 3, self.frame_size, self.frame_size)
        frames_vae = frames_flat.half().to(self.device) / 255.0 * 2 - 1

        with torch.no_grad():
            latents = []
            for i in range(0, B * T, 32):  # Even larger batch VAE encoding with fp16
                batch = frames_vae[i:i+32]
                latent = self.vae.encode(batch).latent_dist.sample() * 0.18215
                latents.append(latent.float())  # Convert back to fp32 for training
            latents = torch.cat(latents, dim=0).cpu()
            latents = latents.reshape(B, T, 4, 32, 32)

        # Normalize frames for DINO
        frames_norm = frames_batch.float() / 255.0
        frames_norm = (frames_norm - IMAGENET_MEAN.unsqueeze(0)) / IMAGENET_STD.unsqueeze(0)

        self.samples_yielded += B

        return {
            'frames': frames_norm,
            'vae_latents': latents,
        }


def train(config_path):
    """Main training loop with streaming data."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Foveated VLM - Streaming Training")
    print("=" * 70)
    print(f"Data: WebVid-10M (streaming, no disk storage)")
    print(f"Samples: NEVER repeat (10M available)")

    # Output dir
    output_dir = Path(config['logging']['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # W&B
    if config['logging'].get('wandb_project'):
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=f"streaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # Load VAE for latent computation (fp16 for speed)
    print("\nLoading VAE (fp16)...")
    device = 'cuda'
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    # Create model
    print("Creating model...")
    model_cfg = config['model']
    model = FoveatedVideoModel(
        dino_model=model_cfg['dino_model'],
        llm_model=model_cfg['llm_model'],
        dino_dim=model_cfg['dino_dim'],
        llm_dim=model_cfg['llm_dim'],
        query_dim=model_cfg['query_dim'],
        lambda_coarse=model_cfg['lambda_coarse'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # Optimizer
    train_cfg = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )

    # LR scheduler
    def lr_lambda(step):
        warmup = train_cfg.get('warmup_steps', 500)
        if step < warmup:
            return step / warmup
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()

    # Create streaming dataloader
    print("\nStarting streaming dataloader...")
    dataloader = StreamingDataLoader(
        vae=vae,
        batch_size=train_cfg['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        num_download_workers=train_cfg.get('num_download_workers', 8),
        prefetch_batches=train_cfg.get('prefetch_batches', 4),
        device=device,
    )
    dataloader.start()

    # Training
    max_steps = train_cfg['max_steps']
    grad_accum = train_cfg.get('grad_accum', 1)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    log_every = config['logging'].get('log_every', 100)
    save_every = config['logging'].get('save_every', 5000)

    print(f"\nTraining for {max_steps} steps")
    print(f"Batch size: {train_cfg['batch_size']} x {grad_accum} = {train_cfg['batch_size'] * grad_accum}")
    print("=" * 70)

    model.train()
    global_step = 0
    running_loss = 0
    running_fine = 0
    running_coarse = 0

    pbar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        batch = dataloader.get_batch()
        if batch is None:
            print("Data exhausted!")
            break

        frames = batch['frames'].to(device)
        latents = batch['vae_latents'].to(device)
        text_embeds = model.get_empty_text_embeds(frames.shape[0]).to(device)

        # Forward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, loss_fine, loss_coarse = model(text_embeds, frames, latents)
            loss = loss / grad_accum

        # Backward
        scaler.scale(loss).backward()

        running_loss += loss.item() * grad_accum
        running_fine += loss_fine.item()
        running_coarse += loss_coarse.item()

        # Optimizer step
        if (global_step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        global_step += 1
        pbar.update(1)

        # Log
        if global_step % log_every == 0:
            avg_loss = running_loss / log_every
            avg_fine = running_fine / log_every
            avg_coarse = running_coarse / log_every
            ratio = avg_coarse / (avg_fine + 1e-8)

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'fine': f'{avg_fine:.4f}',
                'coarse': f'{avg_coarse:.4f}',
                'ratio': f'{ratio:.3f}',
            })

            if wandb.run:
                wandb.log({
                    'loss': avg_loss,
                    'loss_fine': avg_fine,
                    'loss_coarse': avg_coarse,
                    'ratio': ratio,
                    'lr': scheduler.get_last_lr()[0],
                    'samples': dataloader.samples_yielded,
                    'download_failures': dataloader.download_failures,
                }, step=global_step)

            running_loss = 0
            running_fine = 0
            running_coarse = 0

        # Save
        if global_step % save_every == 0:
            path = output_dir / 'checkpoints' / f'step_{global_step}.pt'
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, path)
            print(f"\nSaved: {path}")

    pbar.close()
    dataloader.stop()

    # Final save
    final_path = output_dir / 'checkpoints' / 'final.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Samples seen: {dataloader.samples_yielded}")
    print(f"Download failures: {dataloader.download_failures}")
    print(f"Checkpoint: {final_path}")
    print("=" * 70)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_streaming.yaml')
    args = parser.parse_args()
    train(args.config)
