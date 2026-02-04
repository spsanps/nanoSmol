#!/usr/bin/env python3
"""
Download long videos (>=60s) from WebVid for 64-frame experiments.
Extracts 64 frames per video, computes VAE latents, and shards the data.
"""

import os
import sys
import torch
import requests
import subprocess
import json
import tempfile
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from diffusers import AutoencoderKL

# Config
NUM_FRAMES = 64
FRAME_SIZE = 224  # Match training
MIN_DURATION = 45  # seconds (45s @ 1.5fps = 67 frames)
TARGET_VIDEOS = 500  # Total videos to download (smaller for quick test)
SAMPLES_PER_SHARD = 25
OUTPUT_DIR = Path('/mnt/d/projects/fVLM/data/webvid_64frames')

import re

def parse_iso_duration(duration_str):
    """Parse ISO 8601 duration like PT00H00M11S to seconds."""
    if not duration_str:
        return 0
    match = re.match(r'PT(\d+)H(\d+)M(\d+)S', duration_str)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    return 0


def get_video_duration(url: str, timeout: int = 10) -> float:
    """Get video duration using ffprobe without downloading."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0


def download_video(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download video from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception:
        pass
    return False


def extract_frames_ffmpeg(video_path: Path, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames uniformly from video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First get video info
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets', '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0', str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        # Extract all frames first, then sample
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
            '-q:v', '2',
            f'{tmpdir}/frame_%06d.jpg',
            '-y', '-loglevel', 'error'
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            return None

        # Load frames
        frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
        if len(frame_files) < num_frames:
            return None

        # Sample uniformly
        indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)

        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert('RGB')
            frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [3, H, W]
            frames.append(frame)

        return torch.stack(frames)  # [T, 3, H, W]


def compute_vae_latents(vae, frames: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """Compute VAE latents for frames."""
    frames = frames.float() / 255.0 * 2 - 1
    frames = frames.to(device)

    with torch.no_grad():
        latents = []
        for i in range(frames.shape[0]):
            frame = frames[i:i+1]
            latent = vae.encode(frame).latent_dist.sample()
            latent = latent * 0.18215
            latents.append(latent)

        return torch.cat(latents, dim=0).cpu()


def process_single_video(video_id: str, url: str, caption: str, vae, device: str) -> dict:
    """Download and process a single video."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Download
        if not download_video(url, temp_path):
            return None

        # Extract frames
        frames = extract_frames_ffmpeg(temp_path, NUM_FRAMES, FRAME_SIZE)
        if frames is None:
            return None

        # Compute latents
        latents = compute_vae_latents(vae, frames, device)

        return {
            'video_id': video_id,
            'frames': frames,  # [64, 3, 224, 224] uint8
            'latents': latents,  # [64, 4, 28, 28]
            'caption': caption,
        }

    except Exception as e:
        return None

    finally:
        temp_path.unlink(missing_ok=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shards_dir = OUTPUT_DIR / 'shards'
    shards_dir.mkdir(exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Load WebVid dataset (streaming)
    print("Loading WebVid dataset (streaming)...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Process videos
    print(f"Downloading {TARGET_VIDEOS} videos with duration >= {MIN_DURATION}s...")
    print(f"Extracting {NUM_FRAMES} frames per video at {FRAME_SIZE}x{FRAME_SIZE}")

    samples = []
    shard_idx = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    pbar = tqdm(total=TARGET_VIDEOS, desc="Processing")

    for sample in ds:
        if success_count >= TARGET_VIDEOS:
            break

        video_id = str(sample['videoid'])
        url = sample['contentUrl']
        caption = sample['name']
        duration_str = sample.get('duration', '')

        # Parse ISO 8601 duration (e.g., PT00H01M30S)
        duration = parse_iso_duration(duration_str)

        # Filter by duration
        if duration < MIN_DURATION:
            skip_count += 1
            continue

        # Process video
        result = process_single_video(video_id, url, caption, vae, device)

        if result is not None:
            samples.append(result)
            success_count += 1
            pbar.update(1)

            # Save shard when full
            if len(samples) >= SAMPLES_PER_SHARD:
                shard_path = shards_dir / f'shard_{shard_idx:04d}.pt'
                torch.save({
                    'samples': [{
                        'video_id': s['video_id'],
                        'frames': s['frames'],
                        'latents': s['latents'],
                        'caption': s['caption'],
                    } for s in samples]
                }, shard_path)
                print(f"\nSaved {shard_path.name} ({len(samples)} samples)")
                samples = []
                shard_idx += 1
        else:
            fail_count += 1

        pbar.set_postfix({
            'success': success_count,
            'skipped': skip_count,
            'failed': fail_count,
        })

    pbar.close()

    # Save remaining samples
    if samples:
        shard_path = shards_dir / f'shard_{shard_idx:04d}.pt'
        torch.save({
            'samples': [{
                'video_id': s['video_id'],
                'frames': s['frames'],
                'latents': s['latents'],
                'caption': s['caption'],
            } for s in samples]
        }, shard_path)
        print(f"Saved {shard_path.name} ({len(samples)} samples)")

    # Save metadata
    metadata = {
        'num_frames': NUM_FRAMES,
        'frame_size': FRAME_SIZE,
        'min_duration': MIN_DURATION,
        'total_videos': success_count,
        'num_shards': shard_idx + 1,
        'samples_per_shard': SAMPLES_PER_SHARD,
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Success: {success_count}")
    print(f"  Skipped (too short): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Shards: {shard_idx + 1}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
