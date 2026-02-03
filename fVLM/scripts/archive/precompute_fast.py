#!/usr/bin/env python3
"""
FAST precomputation for WebVid - optimized for speed.

Key optimizations vs precompute_webvid.py:
1. Skip saving frames (only save VAE latents) - saves 90% storage + I/O
2. Batch VAE encoding - better GPU utilization
3. Direct frame extraction (no 4x oversample) - faster ffmpeg
4. More download workers (16 default) - better parallelism

Default settings:
- 32 frames per video
- 30-60 second videos only (tighter FPS range: 0.53-1.07 fps)
- Latent shape: [32, 4, 32, 32] bfloat16

Storage estimate:
- ~262 KB per video (latents only)
- 100K videos = ~26 GB
- 50K videos = ~13 GB

Expected time: 10-15 hours for 100K videos

Usage:
    python scripts/precompute_fast.py --num_videos 100000 --num_workers 16
    python scripts/precompute_fast.py --num_videos 50000 --num_workers 16
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
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
import re

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from diffusers import AutoencoderKL
import requests


@dataclass
class VideoResult:
    """Result from processing a video."""
    video_id: str
    status: str  # "success", "download_failed", "extract_failed", "error"
    caption: str = ""
    duration: int = 0
    frames: Optional[List[Image.Image]] = None
    error: str = ""


# Shared VAE queue for batched processing
class VAEBatcher:
    """Batches VAE encoding across multiple videos for efficiency."""

    def __init__(self, vae, device, batch_size: int = 8):
        self.vae = vae
        self.device = device
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def encode_batch(self, all_frames: List[List[Image.Image]]) -> List[torch.Tensor]:
        """Encode multiple videos' frames in one batched call."""
        # Flatten all frames
        flat_frames = []
        video_lengths = []
        for frames in all_frames:
            video_lengths.append(len(frames))
            flat_frames.extend(frames)

        if not flat_frames:
            return [None] * len(all_frames)

        # Convert to tensor
        frame_tensors = []
        for frame in flat_frames:
            arr = np.array(frame).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensor = tensor * 2 - 1  # Scale to [-1, 1]
            frame_tensors.append(tensor)

        frames_tensor = torch.stack(frame_tensors).to(self.device)  # [total_frames, 3, H, W]

        # Encode in batches to avoid OOM
        all_latents = []
        with torch.no_grad():
            for i in range(0, len(frames_tensor), self.batch_size * 16):  # 16 frames per video max
                batch = frames_tensor[i:i + self.batch_size * 16]
                latents = self.vae.encode(batch).latent_dist.mean
                latents = latents * self.vae.config.scaling_factor
                all_latents.append(latents)

        all_latents = torch.cat(all_latents, dim=0).to(torch.bfloat16).cpu()

        # Split back by video
        results = []
        idx = 0
        for length in video_lengths:
            results.append(all_latents[idx:idx + length])
            idx += length

        return results


def parse_duration(dur_str: str) -> int:
    """Parse ISO 8601 duration string to seconds."""
    if not dur_str:
        return 0
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


def download_video(url: str, timeout: int = 20, max_size_mb: int = 50) -> Optional[bytes]:
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
                return None
        return content
    except:
        return None


def extract_frames_fast(video_bytes: bytes, num_frames: int, size: int = 256) -> Optional[List[Image.Image]]:
    """Extract frames directly at target count (no oversampling)."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Get video duration first
                probe_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', f.name
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                try:
                    duration = float(result.stdout.strip())
                except:
                    duration = 10.0  # Fallback

                # Calculate FPS to get exactly num_frames
                fps = num_frames / max(duration, 1.0)

                # Extract frames directly at calculated FPS
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'fps={fps:.4f},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames),
                    '-q:v', '5',  # Lower quality (faster) - we only need VAE latents
                    f'{tmpdir}/frame_%04d.jpg',
                    '-y', '-loglevel', 'error'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode != 0:
                    return None

                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    # Pad by repeating last frame
                    while len(frame_files) < num_frames:
                        frame_files.append(frame_files[-1])

                frames = []
                for fpath in frame_files[:num_frames]:
                    img = Image.open(fpath).convert('RGB')
                    frames.append(img)

                return frames
    except Exception as e:
        return None


def download_and_extract(sample: dict, video_id: str, num_frames: int, config: dict) -> VideoResult:
    """Download video and extract frames (CPU-only work)."""
    url = sample.get('contentUrl')
    caption = sample.get('name', '')
    duration = parse_duration(sample.get('duration', ''))

    # Download
    video_bytes = download_video(
        url,
        timeout=config["download_timeout"],
        max_size_mb=config["max_video_size_mb"]
    )
    if video_bytes is None:
        return VideoResult(video_id=video_id, status="download_failed")

    # Extract frames
    frames = extract_frames_fast(video_bytes, num_frames, config["frame_size"])
    if frames is None:
        return VideoResult(video_id=video_id, status="extract_failed")

    return VideoResult(
        video_id=video_id,
        status="pending_vae",  # Needs VAE encoding
        caption=caption,
        duration=duration,
        frames=frames
    )


def main():
    parser = argparse.ArgumentParser(description="Fast WebVid precomputation")
    parser.add_argument("--num_videos", type=int, default=100000, help="Number of videos")
    parser.add_argument("--num_workers", type=int, default=16, help="Download workers")
    parser.add_argument("--num_frames", type=int, default=32, help="Frames per video")
    parser.add_argument("--vae_batch", type=int, default=4, help="Videos to batch for VAE (lower for 32 frames)")
    parser.add_argument("--output_dir", type=str, default="/mnt/d/projects/fVLM/data/precomputed_32f")
    parser.add_argument("--min_duration", type=int, default=30, help="Min video duration (seconds)")
    parser.add_argument("--max_duration", type=int, default=60, help="Max video duration (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing")
    args = parser.parse_args()

    config = {
        "num_frames": args.num_frames,
        "frame_size": 256,
        "download_timeout": 20,
        "max_video_size_mb": 50,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("FAST WEBVID PRECOMPUTE")
    print("=" * 70)
    print(f"Target: {args.num_videos} videos")
    print(f"Frames: {args.num_frames} per video")
    print(f"Workers: {args.num_workers} (download) + 1 (VAE)")
    print(f"VAE batch: {args.vae_batch} videos")
    print(f"Duration filter: {args.min_duration}-{args.max_duration}s")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Check existing
    existing = set()
    if args.resume:
        existing = {p.stem for p in latent_dir.glob("*.pt")}
        print(f"Found {len(existing)} existing, resuming...")

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

    vae_batcher = VAEBatcher(vae, device, batch_size=args.vae_batch)
    print(f"  VAE ready on {device}")

    # Load dataset
    print("\n[2/3] Loading WebVid-10M...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Process
    print("\n[3/3] Processing...")

    stats = {"success": 0, "download_failed": 0, "extract_failed": 0, "filtered": 0, "skipped": 0}
    metadata_samples = []
    start_time = time.time()
    video_idx = 0

    pbar = tqdm(total=args.num_videos, desc="Precomputing")

    # Pending VAE queue
    pending_vae = []

    def flush_vae_batch():
        """Process pending VAE batch."""
        nonlocal pending_vae
        if not pending_vae:
            return

        # Extract frames list
        frames_list = [r.frames for r in pending_vae]

        # Batch encode
        latents_list = vae_batcher.encode_batch(frames_list)

        # Save results
        for result, latents in zip(pending_vae, latents_list):
            if latents is not None:
                torch.save(latents, latent_dir / f"{result.video_id}.pt")
                stats["success"] += 1
                metadata_samples.append({
                    "video_id": result.video_id,
                    "caption": result.caption,
                    "duration": result.duration,
                    "num_frames": len(result.frames),
                })
                pbar.update(1)
            else:
                stats["extract_failed"] += 1

            # Free memory
            result.frames = None

        pending_vae = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}

        for sample in ds:
            # Check done
            if stats["success"] + stats["skipped"] >= args.num_videos:
                break

            # Duration filter
            duration = parse_duration(sample.get('duration', ''))
            if duration < config["min_duration"] or duration > config["max_duration"]:
                stats["filtered"] += 1
                continue

            # URL/caption check
            url = sample.get('contentUrl')
            caption = sample.get('name', '')
            if not url or not caption or len(caption) < 5:
                stats["filtered"] += 1
                continue

            video_id = f"{video_idx:06d}"
            video_idx += 1

            # Skip existing
            if video_id in existing:
                stats["skipped"] += 1
                pbar.update(1)
                continue

            # Submit download job
            future = executor.submit(
                download_and_extract, sample, video_id, args.num_frames, config
            )
            futures[future] = video_id

            # Process completed downloads
            done = [f for f in futures if f.done()]
            for future in done:
                futures.pop(future)
                try:
                    result = future.result()
                    if result.status == "pending_vae":
                        pending_vae.append(result)
                        # Flush when batch is full
                        if len(pending_vae) >= args.vae_batch:
                            flush_vae_batch()
                    elif result.status == "download_failed":
                        stats["download_failed"] += 1
                    else:
                        stats["extract_failed"] += 1
                except Exception as e:
                    stats["extract_failed"] += 1

            # Progress
            if video_idx % 500 == 0:
                elapsed = time.time() - start_time
                rate = stats["success"] / elapsed if elapsed > 0 else 0
                eta = (args.num_videos - stats["success"] - stats["skipped"]) / rate if rate > 0 else 0
                pbar.set_postfix({
                    "ok": stats["success"],
                    "fail": stats["download_failed"] + stats["extract_failed"],
                    "rate": f"{rate:.1f}/s",
                    "ETA": f"{eta/3600:.1f}h"
                })

        # Wait for remaining
        for future in as_completed(futures):
            try:
                result = future.result()
                if result.status == "pending_vae":
                    pending_vae.append(result)
                elif result.status == "download_failed":
                    stats["download_failed"] += 1
                else:
                    stats["extract_failed"] += 1
            except:
                stats["extract_failed"] += 1

        # Final VAE flush
        flush_vae_batch()

    pbar.close()

    # Save metadata
    elapsed_hours = (time.time() - start_time) / 3600
    metadata = {
        "config": config,
        "num_frames": args.num_frames,
        "elapsed_hours": elapsed_hours,
        "stats": stats,
        "samples": metadata_samples,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed_hours:.2f} hours")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['download_failed'] + stats['extract_failed']}")
    print(f"Rate: {stats['success'] / (elapsed_hours * 3600):.2f} videos/sec")

    # Storage
    latent_size = sum(f.stat().st_size for f in latent_dir.glob("*.pt")) / 1e9
    print(f"Storage: {latent_size:.2f} GB (latents only)")
    print("=" * 70)


if __name__ == "__main__":
    main()
