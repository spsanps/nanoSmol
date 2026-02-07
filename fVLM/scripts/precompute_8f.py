#!/usr/bin/env python3
"""
Precompute 8-frame data for scaling law experiments.

Target: 5000+ samples (enough for 280 steps × 16 batch = 4480 samples < 1 epoch)

Output format (matching existing 8F shards):
- frames: [8, 3, 224, 224] as uint8
- latents: [8, 4, 28, 28] as bfloat16
- caption: str
"""

import sys
import os
import time
import json
import torch
import subprocess
import tempfile
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from diffusers import AutoencoderKL
from threading import Thread, Lock
from queue import Queue, Empty
import re
import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "output_dir": "/mnt/d/projects/fVLM/data/webvid_8f_5k",
    "num_frames": 8,
    "frame_size": 224,
    "min_duration": 3,   # Short videos fine for 8 frames
    "max_duration": 120,
    "prefetch_threads": 8,
    "prefetch_queue_size": 16,
    "download_timeout": 30,
    "target_videos": 5000,
    "samples_per_shard": 10,
}

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\nShutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_duration(dur_str: str) -> int:
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
        match = re.match(r'PT(\d+)M', dur_str)
        if match:
            return int(match[1]) * 60
    except:
        pass
    return 0


def download_video(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return response.content


def extract_frames_fast(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-i', temp_path,
                '-vf', f'scale={size}:{size}',
                '-r', '8',  # 8 fps for short videos
                '-q:v', '2',
                f'{tmpdir}/frame_%04d.jpg'
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
            if len(frame_files) < num_frames:
                raise ValueError(f"Only {len(frame_files)} frames, need {num_frames}")

            # Subsample uniformly
            indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)

            frames = []
            for idx in indices:
                img = Image.open(frame_files[idx]).convert('RGB')
                frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                frames.append(frame)

            return torch.stack(frames)  # [T, 3, H, W] uint8

    finally:
        os.unlink(temp_path)


class VAEEncoder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.bfloat16
        ).to(device)
        self.vae.eval()
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.bfloat16).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.bfloat16).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        # All 8 frames fit in one batch
        x = frames.to(self.device, dtype=torch.bfloat16) / 255.0
        x = (x - self.mean) / self.std
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents.cpu()


class VideoPrefetcher:
    def __init__(self, dataset_iter, config: dict):
        self.dataset_iter = dataset_iter
        self.config = config
        self.queue = Queue(maxsize=config["prefetch_queue_size"])
        self.stop_flag = False
        self.iter_lock = Lock()

        self.threads = []
        for _ in range(config["prefetch_threads"]):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        while not self.stop_flag:
            try:
                with self.iter_lock:
                    sample = next(self.dataset_iter)
            except StopIteration:
                break

            duration = parse_duration(sample.get('duration', 'PT0S'))
            if duration < self.config["min_duration"] or duration > self.config["max_duration"]:
                continue

            url = sample.get('contentUrl', '')
            if not url:
                continue

            try:
                video_bytes = download_video(url, self.config["download_timeout"])
                frames = extract_frames_fast(
                    video_bytes,
                    self.config["num_frames"],
                    self.config["frame_size"]
                )

                self.queue.put({
                    'video_id': sample.get('videoid', ''),
                    'caption': sample.get('name', ''),
                    'frames': frames,
                }, timeout=60)

            except Exception:
                continue

    def get(self, timeout: float = 60) -> dict:
        return self.queue.get(timeout=timeout)

    def stop(self):
        self.stop_flag = True


def main():
    global shutdown_requested

    output_dir = Path(CONFIG["output_dir"])
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRECOMPUTE 8-FRAME DATA")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Target: {CONFIG['target_videos']} videos")
    print(f"Frames: {CONFIG['num_frames']} @ {CONFIG['frame_size']}px")
    print("=" * 70)

    # Resume check
    existing_shards = list(shards_dir.glob("shard_*.pt"))
    start_count = 0
    for shard_path in existing_shards:
        shard = torch.load(shard_path, weights_only=False)
        start_count += len(shard.get('samples', [shard]))
    print(f"Resuming from {start_count} existing samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAEEncoder(device)

    print("Loading WebVid dataset...")
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    dataset_iter = iter(dataset)

    print("Starting prefetcher...")
    prefetcher = VideoPrefetcher(dataset_iter, CONFIG)

    start_time = time.time()
    success_count = start_count
    current_shard = []
    shard_idx = len(existing_shards)

    print("\nPrecomputing... (Ctrl+C to stop)")

    try:
        while not shutdown_requested and success_count < CONFIG["target_videos"]:
            try:
                sample = prefetcher.get(timeout=120)
            except Empty:
                print("Queue empty, waiting...")
                continue

            try:
                frames = sample['frames']
                latents = vae.encode(frames)

                current_shard.append({
                    'frames': frames.to(torch.uint8),
                    'latents': latents.to(torch.bfloat16),
                    'caption': sample['caption'],
                    'video_id': sample['video_id'],
                })

                success_count += 1

                # Save shard when full
                if len(current_shard) >= CONFIG["samples_per_shard"]:
                    shard_path = shards_dir / f"shard_{shard_idx:04d}.pt"
                    torch.save({'samples': current_shard}, shard_path)
                    print(f"Saved shard {shard_idx} ({CONFIG['samples_per_shard']} samples)")
                    current_shard = []
                    shard_idx += 1

                # Progress
                if success_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (success_count - start_count) / elapsed if elapsed > 0 else 0
                    remaining = CONFIG["target_videos"] - success_count
                    eta = remaining / rate if rate > 0 else float('inf')
                    print(f"Videos: {success_count}/{CONFIG['target_videos']} | "
                          f"Rate: {rate:.2f}/s | "
                          f"ETA: {timedelta(seconds=int(eta))}")

            except Exception as e:
                continue

    finally:
        prefetcher.stop()

        # Save remaining samples
        if current_shard:
            shard_path = shards_dir / f"shard_{shard_idx:04d}.pt"
            torch.save({'samples': current_shard}, shard_path)
            print(f"Saved final shard {shard_idx} ({len(current_shard)} samples)")

        print()
        print("=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Total: {success_count} videos")
        print(f"Shards: {shard_idx + 1}")
        print(f"Elapsed: {(time.time() - start_time) / 3600:.2f}h")
        print(f"Output: {shards_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
