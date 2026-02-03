#!/usr/bin/env python3
"""
Precompute WebVid-10M dataset locally for faster training.

Downloads videos, extracts frames, computes VAE latents, saves to disk.
Optimized for longer videos (30s) with 16 frames.

Usage:
    python scripts/precompute_webvid.py --num_videos 100000 --num_workers 4

Output structure:
    /mnt/d/projects/fVLM/data/precomputed/
    ├── metadata.json           # Dataset info
    ├── frames/                  # JPEG frames
    │   ├── 000000/             # Video ID folders
    │   │   ├── frame_00.jpg
    │   │   ├── frame_01.jpg
    │   │   └── ...
    │   └── ...
    └── latents/                 # VAE latents (bfloat16)
        ├── 000000.pt           # [16, 4, 32, 32] tensor
        └── ...
"""

import sys
import os
import time
import json
import argparse
import subprocess
import tempfile
import threading
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import re

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from diffusers import AutoencoderKL
import requests

# Configuration
CONFIG = {
    "output_dir": "/mnt/d/projects/fVLM/data/precomputed",
    "num_frames": 16,
    "frame_size": 256,
    "min_duration": 20,  # Minimum video length in seconds
    "max_duration": 45,  # Maximum video length in seconds
    "target_duration": 30,  # Ideal video length
    "download_timeout": 30,
    "max_video_size_mb": 100,
    "vae_batch_size": 8,  # Process VAE in batches
}


def parse_duration(dur_str: str) -> int:
    """Parse ISO 8601 duration string to seconds."""
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


def download_video(url: str, timeout: int = 30, max_size_mb: int = 100) -> bytes:
    """Download video with timeout and size limit."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            return None

        content = b''
        max_bytes = max_size_mb * 1024 * 1024
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            content += chunk
            if len(content) > max_bytes:
                return None  # Video too large
        return content
    except Exception as e:
        return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int = 256) -> list:
    """Extract frames from video bytes using ffmpeg."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract more frames than needed, then sample uniformly
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 4),  # Extract extra for sampling
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

                # Sample uniformly
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frames.append(img)

                return frames
    except Exception as e:
        return None


def save_frames(frames: list, output_dir: Path, video_id: str):
    """Save frames as JPEG files."""
    frame_dir = output_dir / "frames" / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        frame_path = frame_dir / f"frame_{i:02d}.jpg"
        frame.save(frame_path, "JPEG", quality=95)


def compute_vae_latents(vae, frames: list, device: torch.device) -> torch.Tensor:
    """Compute VAE latents for a list of PIL images."""
    # Convert to tensor
    frame_tensors = []
    for frame in frames:
        arr = np.array(frame).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        tensor = tensor * 2 - 1  # Scale to [-1, 1] for VAE
        frame_tensors.append(tensor)

    frames_tensor = torch.stack(frame_tensors).to(device)  # [T, 3, H, W]

    with torch.no_grad():
        latents = vae.encode(frames_tensor).latent_dist.mean
        latents = latents * vae.config.scaling_factor

    return latents.to(torch.bfloat16).cpu()  # [T, 4, 32, 32]


def process_sample(sample: dict, vae, device, output_dir: Path, video_id: str) -> dict:
    """Process a single video sample."""
    url = sample.get('contentUrl')
    caption = sample.get('name', '')
    duration = parse_duration(sample.get('duration', ''))

    # Download
    video_bytes = download_video(
        url,
        timeout=CONFIG["download_timeout"],
        max_size_mb=CONFIG["max_video_size_mb"]
    )
    if video_bytes is None:
        return {"status": "download_failed"}

    # Extract frames
    frames = extract_frames(video_bytes, CONFIG["num_frames"], CONFIG["frame_size"])
    if frames is None:
        return {"status": "extract_failed"}

    # Save frames
    save_frames(frames, output_dir, video_id)

    # Compute VAE latents
    latents = compute_vae_latents(vae, frames, device)

    # Save latents
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    torch.save(latents, latent_dir / f"{video_id}.pt")

    return {
        "status": "success",
        "video_id": video_id,
        "caption": caption,
        "duration": duration,
        "num_frames": len(frames),
    }


class PrecomputeWorker:
    """Worker that processes videos from a queue."""

    def __init__(self, vae, device, output_dir: Path, result_queue: Queue):
        self.vae = vae
        self.device = device
        self.output_dir = output_dir
        self.result_queue = result_queue

    def process(self, sample: dict, video_id: str):
        try:
            result = process_sample(sample, self.vae, self.device, self.output_dir, video_id)
            self.result_queue.put(result)
        except Exception as e:
            self.result_queue.put({"status": "error", "error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="Precompute WebVid dataset")
    parser.add_argument("--num_videos", type=int, default=100000, help="Number of videos to process")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of download workers")
    parser.add_argument("--start_offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    args = parser.parse_args()

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WEBVID PRECOMPUTE")
    print("=" * 70)
    print(f"Target: {args.num_videos} videos")
    print(f"Frames per video: {CONFIG['num_frames']}")
    print(f"Video duration: {CONFIG['min_duration']}-{CONFIG['max_duration']}s")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")
    print("=" * 70)

    # Check existing progress
    existing_latents = set()
    if args.resume:
        latent_dir = output_dir / "latents"
        if latent_dir.exists():
            existing_latents = {p.stem for p in latent_dir.glob("*.pt")}
            print(f"Found {len(existing_latents)} existing latents, resuming...")

    # Load VAE
    print("\n[1/3] Loading VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  VAE loaded on {device}")

    # Load dataset
    print("\n[2/3] Loading WebVid-10M dataset...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
    print("  Dataset ready (streaming mode)")

    # Process videos
    print("\n[3/3] Processing videos...")

    metadata = {
        "config": CONFIG,
        "start_time": datetime.now().isoformat(),
        "samples": [],
    }

    result_queue = Queue()
    stats = {"success": 0, "failed": 0, "skipped": 0}

    video_idx = 0
    samples_seen = 0
    start_time = time.time()

    pbar = tqdm(total=args.num_videos, desc="Precomputing")

    # Use thread pool for downloads
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        pending_samples = []

        for sample in ds:
            samples_seen += 1

            # Skip offset
            if samples_seen <= args.start_offset:
                continue

            # Check duration filter
            duration = parse_duration(sample.get('duration', ''))
            if duration < CONFIG["min_duration"] or duration > CONFIG["max_duration"]:
                continue

            # Check URL and caption
            url = sample.get('contentUrl')
            caption = sample.get('name', '')
            if not url or not caption or len(caption) < 5:
                continue

            video_id = f"{video_idx:06d}"

            # Skip if already processed (resume mode)
            if video_id in existing_latents:
                stats["skipped"] += 1
                video_idx += 1
                pbar.update(1)
                if stats["success"] + stats["skipped"] >= args.num_videos:
                    break
                continue

            # Submit download job
            future = executor.submit(
                process_sample, sample, vae, device, output_dir, video_id
            )
            futures[future] = video_id
            video_idx += 1

            # Process completed futures
            done_futures = [f for f in futures if f.done()]
            for future in done_futures:
                vid = futures.pop(future)
                try:
                    result = future.result()
                    if result["status"] == "success":
                        stats["success"] += 1
                        metadata["samples"].append(result)
                        pbar.update(1)
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    stats["failed"] += 1

            # Check if done
            if stats["success"] + stats["skipped"] >= args.num_videos:
                break

            # Progress update
            if video_idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = (stats["success"] + stats["skipped"]) / elapsed if elapsed > 0 else 0
                eta = (args.num_videos - stats["success"] - stats["skipped"]) / rate if rate > 0 else 0
                pbar.set_postfix({
                    "ok": stats["success"],
                    "fail": stats["failed"],
                    "skip": stats["skipped"],
                    "rate": f"{rate:.1f}/s",
                    "ETA": f"{eta/3600:.1f}h"
                })

        # Wait for remaining futures
        for future in as_completed(futures):
            vid = futures[future]
            try:
                result = future.result()
                if result["status"] == "success":
                    stats["success"] += 1
                    metadata["samples"].append(result)
                    pbar.update(1)
                else:
                    stats["failed"] += 1
            except Exception as e:
                stats["failed"] += 1

    pbar.close()

    # Save metadata
    elapsed_hours = (time.time() - start_time) / 3600
    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_hours"] = elapsed_hours
    metadata["stats"] = stats

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PRECOMPUTE COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed_hours:.2f} hours")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (resume): {stats['skipped']}")
    print(f"Total samples: {stats['success'] + stats['skipped']}")
    print(f"Output: {output_dir}")

    # Estimate storage
    latent_dir = output_dir / "latents"
    frame_dir = output_dir / "frames"

    latent_size = sum(f.stat().st_size for f in latent_dir.glob("*.pt")) / 1e9 if latent_dir.exists() else 0
    frame_size = sum(f.stat().st_size for f in frame_dir.rglob("*.jpg")) / 1e9 if frame_dir.exists() else 0

    print(f"\nStorage used:")
    print(f"  Latents: {latent_size:.2f} GB")
    print(f"  Frames: {frame_size:.2f} GB")
    print(f"  Total: {latent_size + frame_size:.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
