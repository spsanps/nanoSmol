#!/usr/bin/env python3
"""
Precompute DINO features + VAE latents for efficient training.

Saves:
- DINO patch features: [T, 197, 384] (CLS + patches)
- VAE latents: [T, 4, 32, 32]
- Caption: str

Size: ~4 MB per video (vs 11 MB for raw frames)
Training: No DINO forward pass needed, just query-attention

Usage:
    python scripts/precompute_dino.py
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
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
import re

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from diffusers import AutoencoderKL
from transformers import AutoModel
import requests


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Video selection
    "min_duration": 20,
    "max_duration": 60,

    # Frame extraction
    "num_frames": 24,
    "frame_size": 256,

    # Download
    "num_workers": 12,
    "download_timeout": 15,
    "max_video_size_mb": 50,

    # Processing batch sizes
    "dino_batch_frames": 48,
    "vae_batch_frames": 48,

    # Queues
    "download_queue_size": 100,
    "process_queue_size": 32,

    # Output
    "output_dir": "/mnt/d/projects/fVLM/data/precomputed_dino_100k",

    # Runtime
    "max_hours": 6.0,
    "target_videos": 100000,
}

# DINO normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str: str) -> int:
    """Parse ISO 8601 duration to seconds."""
    try:
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur_str)
        if match:
            h, m, s = match.groups()
            return int(h or 0) * 3600 + int(m or 0) * 60 + int(s or 0)
    except:
        pass
    return 0


def download_video(url: str, timeout: int = 15) -> Optional[bytes]:
    """Download video with timeout."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            content = b''
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 50 * 1024 * 1024:
                    break
            return content
    except:
        pass
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int = 256) -> Optional[np.ndarray]:
    """Extract frames using ffmpeg."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 3),
                    '-q:v', '2',
                    f'{tmpdir}/frame_%04d.jpg',
                    '-y', '-loglevel', 'error'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode != 0:
                    return None

                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    return None

                # Sample evenly
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frames.append(np.array(img))

                return np.stack(frames)  # [T, H, W, 3]
    except:
        return None


@dataclass
class VideoData:
    video_id: str
    caption: str
    duration: int
    frames: np.ndarray  # [T, H, W, 3] uint8


class FeatureProcessor:
    """Process frames through DINO and VAE."""

    def __init__(self, device="cuda", dino_batch=48, vae_batch=48):
        self.device = device
        self.dino_batch = dino_batch
        self.vae_batch = vae_batch

        # Load DINO
        print("Loading DINO...")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
        self.dino.eval()

        # Load VAE
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        ).to(device)
        self.vae.eval()

        # Warmup
        with torch.no_grad():
            dummy = torch.randn(2, 3, 256, 256, device=device)
            _ = self.dino(dummy).last_hidden_state
            dummy_vae = torch.randn(2, 3, 256, 256, device=device, dtype=torch.float16)
            _ = self.vae.encode(dummy_vae).latent_dist.mean
        torch.cuda.synchronize()
        print("Models ready")

    def process_videos(self, videos: List[VideoData]) -> List[dict]:
        """Process videos through DINO and VAE."""
        if not videos:
            return []

        results = []

        for video in videos:
            frames = video.frames  # [T, H, W, 3] uint8

            # Convert to tensor
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]

            # DINO normalization
            frames_dino = (frames_tensor - IMAGENET_MEAN) / IMAGENET_STD
            frames_dino = frames_dino.to(self.device)

            # VAE normalization ([-1, 1])
            frames_vae = (frames_tensor * 2 - 1).to(self.device, dtype=torch.float16)

            # Process DINO in batches
            dino_features = []
            with torch.no_grad():
                for i in range(0, len(frames_dino), self.dino_batch):
                    batch = frames_dino[i:i + self.dino_batch]
                    # Get all hidden states including CLS
                    out = self.dino(batch).last_hidden_state  # [B, 197, 384]
                    dino_features.append(out.cpu())
            dino_features = torch.cat(dino_features, dim=0)  # [T, 197, 384]

            # Process VAE in batches
            vae_latents = []
            with torch.no_grad():
                for i in range(0, len(frames_vae), self.vae_batch):
                    batch = frames_vae[i:i + self.vae_batch]
                    latents = self.vae.encode(batch).latent_dist.mean
                    latents = latents * self.vae.config.scaling_factor
                    vae_latents.append(latents.cpu())
            vae_latents = torch.cat(vae_latents, dim=0)  # [T, 4, 32, 32]

            results.append({
                "dino_features": dino_features.to(torch.bfloat16),  # [T, 197, 384]
                "latents": vae_latents.to(torch.bfloat16),  # [T, 4, 32, 32]
                "caption": video.caption,
            })

        return results


def download_worker(sample_queue: Queue, result_queue: Queue, config: dict, stop_event: threading.Event, workers_done: list):
    """Worker that downloads and extracts frames."""
    while not stop_event.is_set():
        try:
            item = sample_queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            workers_done.append(1)  # Signal this worker is done
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
    feature_dir = output_dir / "features"
    feature_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("DINO + VAE PRECOMPUTE")
    print("=" * 70)
    print(f"Duration filter: {config['min_duration']}-{config['max_duration']}s")
    print(f"Frames per video: {config['num_frames']}")
    print(f"Download workers: {config['num_workers']}")
    print(f"Target videos: {config['target_videos']}")
    print(f"Output: {output_dir}")
    print(f"Max runtime: {config['max_hours']} hours")
    print("=" * 70)

    # Check existing files
    existing = {p.stem for p in feature_dir.glob("*.pt")}
    metadata_path = output_dir / "metadata.json"
    stream_position = 0

    if existing and metadata_path.exists():
        max_existing_id = max(int(x) for x in existing)
        try:
            with open(metadata_path) as f:
                old_meta = json.load(f)
                stream_position = old_meta.get("stream_position", 0)
        except:
            stream_position = int(len(existing) / 0.31)
        print(f"Found {len(existing)} existing files, max ID: {max_existing_id}")
        print(f"Resuming from stream position: {stream_position}")
        start_video_idx = max_existing_id + 1
    else:
        start_video_idx = 0
        print("Starting fresh")

    # Check if we've reached target
    if len(existing) >= config["target_videos"]:
        print(f"Already have {len(existing)} videos (target: {config['target_videos']})")
        return

    # Load processor
    processor = FeatureProcessor(
        dino_batch=config["dino_batch_frames"],
        vae_batch=config["vae_batch_frames"]
    )

    # Setup queues and workers
    sample_queue = Queue(maxsize=config["download_queue_size"])
    result_queue = Queue(maxsize=config["process_queue_size"])
    stop_event = threading.Event()
    workers_done = []  # Track how many workers have finished

    workers = []
    for _ in range(config["num_workers"]):
        w = threading.Thread(target=download_worker, args=(sample_queue, result_queue, config, stop_event, workers_done))
        w.daemon = True
        w.start()
        workers.append(w)

    # Load dataset
    print("\nLoading WebVid-10M (streaming)...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Skip to resume position
    if stream_position > 0:
        print(f"Skipping {stream_position} samples to resume position...")
        ds = ds.skip(stream_position)
        print("Skip complete, starting processing...")

    # Stats
    stats = {"success": 0, "failed": 0, "filtered": 0, "samples_seen": 0}
    metadata_list = []
    start_time = time.time()
    max_seconds = config["max_hours"] * 3600
    video_idx = start_video_idx
    pending_process = []
    process_batch_size = 4

    # Progress bar
    remaining_target = config["target_videos"] - len(existing)
    pbar = tqdm(total=remaining_target, desc="Precomputing", unit="videos")

    # Producer thread
    def producer():
        nonlocal video_idx
        for sample in ds:
            if stop_event.is_set():
                break

            stats["samples_seen"] += 1

            # Check target
            if len(existing) + stats["success"] >= config["target_videos"]:
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
                continue

            # Keep trying to put in queue (with stop check)
            while not stop_event.is_set():
                try:
                    sample_queue.put((vid, sample), timeout=5)
                    break
                except:
                    continue

        # Signal workers to stop
        for _ in range(config["num_workers"]):
            sample_queue.put(None)

    producer_thread = threading.Thread(target=producer)
    producer_thread.daemon = True
    producer_thread.start()

    # Main loop
    last_log = time.time()
    no_progress_count = 0  # Track consecutive empty polls

    try:
        while True:
            elapsed = time.time() - start_time

            # Check limits
            if elapsed > max_seconds:
                print("\n\nTime limit reached!")
                break

            if len(existing) + stats["success"] >= config["target_videos"]:
                print("\n\nTarget reached!")
                break

            # Get results from download workers
            try:
                status, video_id, video_data = result_queue.get(timeout=2)
                no_progress_count = 0  # Reset on any result
                if status == "failed":
                    stats["failed"] += 1
                elif status == "success":
                    pending_process.append(video_data)
            except Empty:
                no_progress_count += 1

            # Process batch when full
            if len(pending_process) >= process_batch_size or (len(pending_process) > 0 and time.time() - last_log > 30):
                batch = pending_process[:process_batch_size]
                pending_process = pending_process[process_batch_size:]

                # Process through DINO + VAE
                results = processor.process_videos(batch)

                # Save
                for video_data, result in zip(batch, results):
                    save_data = {
                        "dino_features": result["dino_features"],  # [T, 197, 384]
                        "latents": result["latents"],  # [T, 4, 32, 32]
                        "caption": result["caption"],
                    }
                    torch.save(save_data, feature_dir / f"{video_data.video_id}.pt")
                    metadata_list.append({
                        "video_id": video_data.video_id,
                        "caption": video_data.caption,
                        "duration": video_data.duration,
                        "num_frames": len(result["latents"]),
                    })
                    stats["success"] += 1
                    pbar.update(1)

            # Progress logging
            if time.time() - last_log > 30:
                rate = stats["success"] / elapsed if elapsed > 0 else 0
                remaining_videos = config["target_videos"] - len(existing) - stats["success"]
                eta_seconds = remaining_videos / rate if rate > 0 else 0

                pbar.set_postfix({
                    "ok": stats["success"],
                    "fail": stats["failed"],
                    "rate": f"{rate:.1f}/s",
                    "eta": f"{eta_seconds/3600:.1f}h"
                })
                last_log = time.time()

            # Check if done - all workers finished AND no progress for 10 consecutive polls
            if len(workers_done) >= config["num_workers"] and result_queue.empty() and len(pending_process) == 0 and no_progress_count >= 5:
                print("\n\nAll samples processed!")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        stop_event.set()

        # Process remaining
        if pending_process:
            print(f"\nProcessing {len(pending_process)} remaining videos...")
            results = processor.process_videos(pending_process)
            for video_data, result in zip(pending_process, results):
                save_data = {
                    "dino_features": result["dino_features"],
                    "latents": result["latents"],
                    "caption": result["caption"],
                }
                torch.save(save_data, feature_dir / f"{video_data.video_id}.pt")
                metadata_list.append({
                    "video_id": video_data.video_id,
                    "caption": video_data.caption,
                    "duration": video_data.duration,
                    "num_frames": len(result["latents"]),
                })
                stats["success"] += 1

        pbar.close()

    # Save metadata
    total_stream_position = stream_position + stats["samples_seen"]
    elapsed_hours = (time.time() - start_time) / 3600
    metadata = {
        "config": config,
        "elapsed_hours": elapsed_hours,
        "stats": stats,
        "stream_position": total_stream_position,
        "samples": metadata_list,
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    total_videos = len(existing) + stats["success"]
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Runtime: {elapsed_hours:.2f} hours")
    print(f"New videos: {stats['success']}")
    print(f"Total videos: {total_videos}")
    print(f"Failed: {stats['failed']}")
    print(f"Filtered: {stats['filtered']}")
    print(f"Rate: {stats['success'] / elapsed_hours:.0f} videos/hour")

    # Check storage
    import shutil
    total_size = sum(f.stat().st_size for f in feature_dir.glob("*.pt"))
    print(f"Storage: {total_size / 1e9:.2f} GB")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
