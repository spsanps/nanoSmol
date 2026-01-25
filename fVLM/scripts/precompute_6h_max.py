#!/usr/bin/env python3
"""
Maximum throughput precomputation - 6 hour run.

Optimizations:
1. 20-60s duration filter (31% hit rate vs 10.8%)
2. 24 frames per video (0.4-1.2 fps range)
3. Large VAE batch (process 64+ frames at once)
4. Producer-consumer pipeline with queues
5. 16 download workers

Target: 50-80K videos in 6 hours

Usage:
    python scripts/precompute_6h_max.py
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import threading
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
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


# ============================================================
# CONFIGURATION - OPTIMIZED FOR 6H MAX THROUGHPUT
# ============================================================
CONFIG = {
    # Video selection
    "min_duration": 20,
    "max_duration": 60,

    # Frame extraction
    "num_frames": 24,
    "frame_size": 256,

    # Download
    "num_workers": 12,  # Reduced to avoid overwhelming system
    "download_timeout": 15,  # Fail fast
    "max_video_size_mb": 50,

    # VAE batching - reduced to avoid OOM
    "vae_batch_frames": 48,  # Process 48 frames at once (2 videos × 24 frames)

    # Queues
    "download_queue_size": 100,
    "vae_queue_size": 32,  # Videos waiting for VAE

    # Output (v2 format: includes both frames and latents)
    "output_dir": "/mnt/d/projects/fVLM/data/precomputed_24f_v2",

    # Runtime
    "max_hours": 1.0,  # Changed to 1 hour for resume
}


def parse_duration(dur_str: str) -> int:
    if not dur_str:
        return 0
    try:
        m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur_str)
        if m:
            h, mins, s = m.groups()
            return int(h or 0) * 3600 + int(mins or 0) * 60 + int(s or 0)
    except:
        pass
    return 0


def download_video(url: str, timeout: int = 15) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        chunks = []
        size = 0
        for chunk in r.iter_content(chunk_size=512 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size > 50 * 1024 * 1024:
                return None
        return b''.join(chunks)
    except:
        return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int = 256) -> Optional[List[np.ndarray]]:
    """Extract frames, return as numpy arrays (faster than PIL)."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Get duration
                probe = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', f.name],
                    capture_output=True, text=True, timeout=5
                )
                try:
                    duration = float(probe.stdout.strip())
                except:
                    duration = 30.0

                fps = num_frames / max(duration, 1.0)

                # Extract with ffmpeg - use pipe for speed
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'fps={fps:.4f},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames),
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    'pipe:1', '-loglevel', 'error'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)

                if result.returncode != 0 or len(result.stdout) == 0:
                    return None

                # Parse raw video output
                raw = np.frombuffer(result.stdout, dtype=np.uint8)
                expected_size = num_frames * size * size * 3

                if len(raw) < expected_size:
                    # Pad with last frame if short
                    actual_frames = len(raw) // (size * size * 3)
                    if actual_frames < num_frames // 2:
                        return None
                    raw = np.pad(raw, (0, expected_size - len(raw)), mode='edge')

                frames = raw[:expected_size].reshape(num_frames, size, size, 3)
                return frames

    except Exception as e:
        return None


@dataclass
class VideoData:
    video_id: str
    caption: str
    duration: int
    frames: np.ndarray  # [T, H, W, 3] uint8


class VAEProcessor:
    """Batched VAE processing for maximum GPU utilization."""

    def __init__(self, device, batch_frames: int = 96):
        self.device = device
        self.batch_frames = batch_frames

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16  # Use fp16 for speed
        ).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Warmup
        with torch.no_grad():
            dummy = torch.randn(4, 3, 256, 256, device=device, dtype=torch.float16)
            _ = self.vae.encode(dummy).latent_dist.mean
        torch.cuda.synchronize()
        print(f"VAE ready, batch_frames={batch_frames}")

    def encode_videos(self, videos: List[VideoData]) -> List[torch.Tensor]:
        """Encode multiple videos in efficient batches."""
        if not videos:
            return []

        # Stack all frames
        all_frames = []
        frame_counts = []
        for v in videos:
            all_frames.append(v.frames)
            frame_counts.append(len(v.frames))

        all_frames = np.concatenate(all_frames, axis=0)  # [total_frames, H, W, 3]

        # Convert to tensor
        frames_tensor = torch.from_numpy(all_frames).permute(0, 3, 1, 2)  # [N, 3, H, W]
        frames_tensor = frames_tensor.to(self.device, dtype=torch.float16)
        frames_tensor = frames_tensor / 255.0 * 2 - 1  # Scale to [-1, 1]

        # Encode in batches
        all_latents = []
        with torch.no_grad():
            for i in range(0, len(frames_tensor), self.batch_frames):
                batch = frames_tensor[i:i + self.batch_frames]
                latents = self.vae.encode(batch).latent_dist.mean
                latents = latents * self.vae.config.scaling_factor
                all_latents.append(latents.cpu())

        all_latents = torch.cat(all_latents, dim=0).to(torch.bfloat16)

        # Split by video
        results = []
        idx = 0
        for count in frame_counts:
            results.append(all_latents[idx:idx + count])
            idx += count

        return results


def download_worker(sample_queue: Queue, result_queue: Queue, config: dict, stop_event: threading.Event):
    """Worker that downloads and extracts frames."""
    while not stop_event.is_set():
        try:
            item = sample_queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            break

        video_id, sample = item
        url = sample.get('contentUrl', '')
        caption = sample.get('name', '')
        duration = parse_duration(sample.get('duration', ''))

        # Download
        video_bytes = download_video(url, config["download_timeout"])
        if video_bytes is None:
            result_queue.put(("failed", video_id, None))
            continue

        # Extract frames
        frames = extract_frames(video_bytes, config["num_frames"], config["frame_size"])
        if frames is None:
            result_queue.put(("failed", video_id, None))
            continue

        result_queue.put(("success", video_id, VideoData(
            video_id=video_id,
            caption=caption,
            duration=duration,
            frames=frames
        )))


def main():
    config = CONFIG.copy()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("6-HOUR MAXIMUM THROUGHPUT PRECOMPUTE")
    print("=" * 70)
    print(f"Duration filter: {config['min_duration']}-{config['max_duration']}s")
    print(f"Frames per video: {config['num_frames']}")
    print(f"Download workers: {config['num_workers']}")
    print(f"VAE batch frames: {config['vae_batch_frames']}")
    print(f"Output: {output_dir}")
    print(f"Max runtime: {config['max_hours']} hours")
    print("=" * 70)

    # Check existing - find max ID and stream position to continue from
    existing = {p.stem for p in latent_dir.glob("*.pt")}
    metadata_path = output_dir / "metadata.json"
    stream_position = 0

    if existing and metadata_path.exists():
        max_existing_id = max(int(x) for x in existing)
        # Load stream position from metadata
        try:
            with open(metadata_path) as f:
                old_meta = json.load(f)
                stream_position = old_meta.get("stream_position", 0)
        except:
            # Estimate stream position from video count (31% filter rate)
            stream_position = int(len(existing) / 0.31)
        print(f"Found {len(existing)} existing files, max ID: {max_existing_id}")
        print(f"Resuming from stream position: {stream_position}")
        start_video_idx = max_existing_id + 1
    else:
        start_video_idx = 0
        print("Starting fresh")

    # Initialize
    device = torch.device("cuda")
    vae_processor = VAEProcessor(device, config["vae_batch_frames"])

    # Load dataset
    print("\nLoading WebVid-10M (streaming)...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Skip to resume position
    if stream_position > 0:
        print(f"Skipping {stream_position} samples to resume position...")
        ds = ds.skip(stream_position)
        print("Skip complete, starting processing...")

    # Queues
    sample_queue = Queue(maxsize=config["download_queue_size"])
    result_queue = Queue(maxsize=config["vae_queue_size"] * 2)
    stop_event = threading.Event()

    # Start download workers
    workers = []
    for _ in range(config["num_workers"]):
        t = threading.Thread(target=download_worker, args=(sample_queue, result_queue, config, stop_event))
        t.daemon = True
        t.start()
        workers.append(t)

    # Stats
    stats = {"success": 0, "failed": 0, "filtered": 0, "skipped": 0, "samples_seen": 0}
    metadata_list = []
    start_time = time.time()
    max_seconds = config["max_hours"] * 3600

    video_idx = start_video_idx  # Continue from max existing ID + 1
    pending_vae = []
    vae_batch_size = 8  # Videos per VAE batch

    pbar = tqdm(desc="Precomputing", unit="videos")

    # Producer thread - feeds samples to download queue
    def producer():
        nonlocal video_idx
        for sample in ds:
            stats["samples_seen"] += 1

            if stop_event.is_set():
                break

            # Duration filter
            duration = parse_duration(sample.get('duration', ''))
            if duration < config["min_duration"] or duration > config["max_duration"]:
                stats["filtered"] += 1
                continue

            # URL/caption check
            url = sample.get('contentUrl', '')
            caption = sample.get('name', '')
            if not url or not caption or len(caption) < 5:
                stats["filtered"] += 1
                continue

            vid = f"{video_idx:06d}"
            video_idx += 1

            # Skip existing
            if vid in existing:
                stats["skipped"] += 1
                pbar.update(1)
                continue

            try:
                sample_queue.put((vid, sample), timeout=5)
            except:
                break

        # Signal workers to stop
        for _ in range(config["num_workers"]):
            sample_queue.put(None)

    producer_thread = threading.Thread(target=producer)
    producer_thread.daemon = True
    producer_thread.start()

    # Main loop - process results and run VAE
    last_save = time.time()
    last_log = time.time()

    try:
        while True:
            elapsed = time.time() - start_time

            # Time limit
            if elapsed > max_seconds:
                print("\n\nTime limit reached!")
                break

            # Get results from download workers
            try:
                status, vid, data = result_queue.get(timeout=0.1)

                if status == "success":
                    pending_vae.append(data)
                else:
                    stats["failed"] += 1

            except Empty:
                # No results, check if we should process VAE batch
                pass

            # Process VAE batch when full or periodically
            if len(pending_vae) >= vae_batch_size or (len(pending_vae) > 0 and time.time() - last_save > 30):
                batch = pending_vae[:vae_batch_size]
                pending_vae = pending_vae[vae_batch_size:]

                # Encode
                latents_list = vae_processor.encode_videos(batch)

                # Save (frames + latents for efficient training)
                for video_data, latents in zip(batch, latents_list):
                    # Convert frames to tensor and normalize for DINO
                    frames_np = video_data.frames  # [T, H, W, 3] uint8
                    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]

                    # Save dict with both frames and latents
                    save_data = {
                        "frames": frames_tensor.to(torch.bfloat16),  # [T, 3, 256, 256]
                        "latents": latents,  # [T, 4, 32, 32]
                    }
                    torch.save(save_data, latent_dir / f"{video_data.video_id}.pt")

                    metadata_list.append({
                        "video_id": video_data.video_id,
                        "caption": video_data.caption,
                        "duration": video_data.duration,
                        "num_frames": len(latents),
                    })
                    stats["success"] += 1
                    pbar.update(1)

                last_save = time.time()

            # Progress logging
            if time.time() - last_log > 30:
                rate = stats["success"] / elapsed if elapsed > 0 else 0
                remaining = max_seconds - elapsed
                projected = stats["success"] + rate * remaining

                pbar.set_postfix({
                    "ok": stats["success"],
                    "fail": stats["failed"],
                    "rate": f"{rate:.1f}/s",
                    "proj": f"{projected/1000:.0f}K",
                    "eta": f"{remaining/3600:.1f}h"
                })
                last_log = time.time()

            # Check if producer is done and queues empty
            if not producer_thread.is_alive() and sample_queue.empty() and result_queue.empty() and len(pending_vae) == 0:
                print("\n\nAll samples processed!")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        stop_event.set()

        # Process remaining VAE batch
        if pending_vae:
            print(f"\nProcessing {len(pending_vae)} remaining videos...")
            latents_list = vae_processor.encode_videos(pending_vae)
            for video_data, latents in zip(pending_vae, latents_list):
                # Convert frames to tensor and normalize for DINO
                frames_np = video_data.frames  # [T, H, W, 3] uint8
                frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]

                # Save dict with both frames and latents
                save_data = {
                    "frames": frames_tensor.to(torch.bfloat16),  # [T, 3, 256, 256]
                    "latents": latents,  # [T, 4, 32, 32]
                }
                torch.save(save_data, latent_dir / f"{video_data.video_id}.pt")

                metadata_list.append({
                    "video_id": video_data.video_id,
                    "caption": video_data.caption,
                    "duration": video_data.duration,
                    "num_frames": len(latents),
                })
                stats["success"] += 1

        pbar.close()

    # Save metadata with stream position for resume
    elapsed_hours = (time.time() - start_time) / 3600
    total_stream_position = stream_position + stats["samples_seen"]
    metadata = {
        "config": config,
        "elapsed_hours": elapsed_hours,
        "stats": stats,
        "stream_position": total_stream_position,  # For resume
        "samples": metadata_list,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Runtime: {elapsed_hours:.2f} hours")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Filtered: {stats['filtered']}")
    print(f"Rate: {stats['success'] / elapsed_hours:.0f} videos/hour")

    # Storage
    latent_size = sum(f.stat().st_size for f in latent_dir.glob("*.pt")) / 1e9
    print(f"Storage: {latent_size:.2f} GB")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
