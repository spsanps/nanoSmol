#!/usr/bin/env python3
"""
Precompute Frames + VAE Latents for Joint Multi-Fine Training.

This script precomputes:
- frames: [T, 3, 256, 256] as uint8 (for trainable DINO)
- latents: [T, 4, 32, 32] as bfloat16 (VAE targets)
- caption: str
- video_id: str (for tracking)

Features:
- Resumable: Checks existing files, continues from where it left off
- Robust: Handles network failures, corrupted videos gracefully
- Tracked: Detailed metadata, progress reporting, ETA
- Optimal: Multi-threaded prefetch, efficient storage

Storage estimate: ~1 MB/video (frames as uint8 + latents as bfloat16)
100K videos = ~100 GB

Usage:
    python precompute_frames_latents.py [--output_dir DIR] [--target_videos N]
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
from threading import Thread
from queue import Queue, Empty
import re
import signal
import atexit

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Output
    "output_dir": "/mnt/d/projects/fVLM/data/frames_latents_100k",

    # Video settings
    "num_frames": 24,
    "frame_size": 256,
    "min_duration": 10,  # seconds (relaxed from 20 for higher acceptance)
    "max_duration": 60,  # seconds

    # Performance - aggressive parallel downloads
    "prefetch_threads": 12,  # More parallel downloads
    "prefetch_queue_size": 32,  # Large queue to keep GPU saturated
    "download_timeout": 10,  # Quick timeout, skip slow URLs

    # Limits (0 = unlimited)
    "target_videos": 0,  # Run until stopped
    "max_hours": 0,      # Run until stopped

    # Checkpointing
    "save_metadata_interval": 100,  # Save metadata every N videos
}

# Global state for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\nShutdown requested, finishing current video...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
        match = re.match(r'PT(\d+)M', dur_str)
        if match:
            return int(match[1]) * 60
    except:
        pass
    return 0


def download_video(url: str, timeout: int = 30) -> bytes:
    """Download video bytes from URL."""
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return response.content


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames from video bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # Get video duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', temp_path],
            capture_output=True, timeout=10
        )
        duration = float(json.loads(probe.stdout)['format'].get('duration', 0))

        if duration < 1:
            raise ValueError("Video too short")

        # Calculate frame times
        times = np.linspace(0, duration - 0.1, num_frames)

        frames = []
        for t in times:
            cmd = [
                'ffmpeg', '-ss', str(t), '-i', temp_path,
                '-vframes', '1', '-f', 'image2pipe',
                '-vcodec', 'png', '-s', f'{size}x{size}', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0:
                raise ValueError(f"ffmpeg failed at t={t}")

            img = Image.open(tempfile.SpooledTemporaryFile())
            img = Image.frombytes('RGB', (size, size), result.stdout[-size*size*3:])

            # Actually decode properly
            from io import BytesIO
            img = Image.open(BytesIO(result.stdout)).convert('RGB').resize((size, size))

            frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [3, H, W]
            frames.append(frame)

        return torch.stack(frames)  # [T, 3, H, W]

    finally:
        os.unlink(temp_path)


def extract_frames_fast(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames using single ffmpeg call (faster)."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # Extract all frames at once
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-i', temp_path,
                '-vf', f'fps={num_frames}/$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {temp_path}),scale={size}:{size}',
                '-vsync', '0',
                f'{tmpdir}/frame_%04d.png'
            ]

            # Simpler approach: extract at fixed fps then subsample
            cmd = [
                'ffmpeg', '-i', temp_path,
                '-vf', f'scale={size}:{size}',
                '-r', '8',  # 8 fps
                '-q:v', '2',
                f'{tmpdir}/frame_%04d.jpg'
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            # Load frames
            frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
            if len(frame_files) < num_frames:
                raise ValueError(f"Only got {len(frame_files)} frames, need {num_frames}")

            # Subsample to exact number needed
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
    """VAE encoder for computing latents."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.bfloat16
        ).to(device)
        self.vae.eval()

        # ImageNet normalization (must be bfloat16 to match VAE)
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.bfloat16).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.bfloat16).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latents.

        Args:
            frames: [T, 3, H, W] uint8 tensor

        Returns:
            latents: [T, 4, H/8, W/8] bfloat16 tensor
        """
        # Normalize to [-1, 1]
        x = frames.to(self.device, dtype=torch.bfloat16) / 255.0
        x = (x - self.mean) / self.std

        # Encode
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents.cpu()


class VideoPrefetcher:
    """Prefetch videos in background threads."""

    def __init__(self, dataset_iter, config: dict):
        self.dataset_iter = dataset_iter
        self.config = config
        self.queue = Queue(maxsize=config["prefetch_queue_size"])
        self.stop_flag = False

        # Lock for thread-safe iterator access
        from threading import Lock
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

            # Filter by duration
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
                    'duration': duration,
                    'frames': frames,
                }, timeout=60)

            except Exception:
                # Skip failed videos silently
                continue

    def get(self, timeout: float = 30) -> dict:
        return self.queue.get(timeout=timeout)

    def stop(self):
        self.stop_flag = True


def load_metadata(output_dir: Path) -> dict:
    """Load existing metadata or create new."""
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {
        "config": CONFIG,
        "start_time": datetime.now().isoformat(),
        "samples": [],
        "stats": {
            "success": 0,
            "failed": 0,
            "filtered": 0,
        }
    }


def save_metadata(metadata: dict, output_dir: Path):
    """Save metadata to disk."""
    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_existing_count(output_dir: Path) -> int:
    """Fast count of existing files (no loading)."""
    features_dir = output_dir / "features"
    if not features_dir.exists():
        return 0
    # Just count files - much faster than loading each one
    return len(list(features_dir.glob("*.pt")))


def format_eta(seconds: float) -> str:
    """Format seconds to human readable."""
    if seconds < 0 or seconds > 1e9:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


def main():
    global shutdown_requested

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"])
    parser.add_argument("--target_videos", type=int, default=CONFIG["target_videos"])
    parser.add_argument("--max_hours", type=float, default=CONFIG["max_hours"])
    args = parser.parse_args()

    CONFIG["output_dir"] = args.output_dir
    CONFIG["target_videos"] = args.target_videos
    CONFIG["max_hours"] = args.max_hours

    output_dir = Path(CONFIG["output_dir"])
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRECOMPUTE: Frames + VAE Latents")
    print("=" * 70)
    print(f"Output: {output_dir}")
    target_str = 'unlimited' if CONFIG['target_videos'] == 0 else str(CONFIG['target_videos'])
    time_str = 'unlimited' if CONFIG['max_hours'] == 0 else f"{CONFIG['max_hours']}h"
    print(f"Target: {target_str} videos")
    print(f"Max time: {time_str}")
    print("=" * 70)

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free / (1024**3)
    print(f"Disk space: {free_gb:.1f} GB free")
    if free_gb < 50:
        print("WARNING: Less than 50 GB free, may run out of space!")
    print()

    # Load existing metadata
    metadata = load_metadata(output_dir)
    start_count = get_existing_count(output_dir)
    print(f"Resuming from {start_count} existing videos")

    # Initialize VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAEEncoder(device)

    # Load dataset
    print("Loading WebVid dataset...")
    dataset = load_dataset(
        "TempoFunk/webvid-10M",
        split="train",
        streaming=True
    )
    dataset_iter = iter(dataset)

    # Start prefetcher
    print("Starting video prefetcher...")
    prefetcher = VideoPrefetcher(dataset_iter, CONFIG)

    # Stats tracking
    start_time = time.time()
    success_count = start_count
    failed_count = 0
    recent_times = []

    print()
    print("Precomputing... (Ctrl+C to stop gracefully)")
    print()

    try:
        while not shutdown_requested:
            # Check limits
            if CONFIG["target_videos"] > 0 and success_count >= CONFIG["target_videos"]:
                print(f"\nReached target of {CONFIG['target_videos']} videos")
                break

            if CONFIG["max_hours"] > 0:
                elapsed = (time.time() - start_time) / 3600
                if elapsed >= CONFIG["max_hours"]:
                    print(f"\nReached time limit of {CONFIG['max_hours']}h")
                    break

            # Check disk space periodically
            if success_count % 1000 == 0 and success_count > start_count:
                _, _, free = shutil.disk_usage(output_dir)
                if free / (1024**3) < 10:
                    print("\nLess than 10 GB free, stopping...")
                    break

            # Get next video
            try:
                sample_start = time.time()
                sample = prefetcher.get(timeout=60)
            except Empty:
                print("Queue empty, waiting...")
                continue

            video_id = sample['video_id']

            try:
                # Encode frames to latents
                frames = sample['frames']  # [T, 3, H, W] uint8
                latents = vae.encode(frames)  # [T, 4, H/8, W/8] bfloat16

                # Save
                save_path = features_dir / f"{success_count:06d}.pt"
                torch.save({
                    'frames': frames.to(torch.uint8),
                    'latents': latents.to(torch.bfloat16),
                    'caption': sample['caption'],
                    'video_id': video_id,
                    'duration': sample['duration'],
                }, save_path)

                # Update tracking
                success_count += 1

                metadata['samples'].append({
                    'video_id': video_id,
                    'caption': sample['caption'][:100],
                    'duration': sample['duration'],
                })
                metadata['stats']['success'] = success_count

                # Timing
                sample_time = time.time() - sample_start
                recent_times.append(sample_time)
                if len(recent_times) > 100:
                    recent_times.pop(0)

                # Progress
                if success_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (success_count - start_count) / elapsed if elapsed > 0 else 0
                    avg_time = sum(recent_times) / len(recent_times) if recent_times else 0

                    # ETA to various targets
                    if rate > 0:
                        eta_100k = (100000 - success_count) / rate
                        eta_200k = (200000 - success_count) / rate
                    else:
                        eta_100k = eta_200k = float('inf')

                    # Disk usage
                    _, _, free = shutil.disk_usage(output_dir)
                    free_gb = free / (1024**3)

                    print(f"Videos: {success_count:,} | "
                          f"Rate: {rate:.1f}/s | "
                          f"Disk: {free_gb:.0f}GB free | "
                          f"ETA 100K: {format_eta(eta_100k)} | "
                          f"ETA 200K: {format_eta(eta_200k)}")

                # Save metadata periodically
                if success_count % CONFIG["save_metadata_interval"] == 0:
                    metadata['last_update'] = datetime.now().isoformat()
                    metadata['elapsed_hours'] = (time.time() - start_time) / 3600
                    save_metadata(metadata, output_dir)

            except Exception as e:
                failed_count += 1
                metadata['stats']['failed'] = failed_count
                continue

    finally:
        # Cleanup
        prefetcher.stop()

        # Save final metadata
        metadata['end_time'] = datetime.now().isoformat()
        metadata['elapsed_hours'] = (time.time() - start_time) / 3600
        metadata['stats']['success'] = success_count
        metadata['stats']['failed'] = failed_count
        save_metadata(metadata, output_dir)

        print()
        print("=" * 70)
        print("PRECOMPUTE COMPLETE")
        print("=" * 70)
        print(f"Total videos: {success_count:,}")
        print(f"Failed: {failed_count:,}")
        print(f"Elapsed: {(time.time() - start_time) / 3600:.2f}h")
        print(f"Output: {output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
