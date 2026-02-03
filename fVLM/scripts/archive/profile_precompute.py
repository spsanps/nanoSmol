#!/usr/bin/env python3
"""
Profile WebVid dataset and precompute pipeline before committing to full run.

Checks:
1. Duration distribution - how many videos in each duration bucket?
2. Download speed - actual throughput
3. ffmpeg extraction speed - for 32 frames
4. VAE encoding speed - batched
5. Storage path verification

Usage:
    python scripts/profile_precompute.py
"""

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
import re

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from diffusers import AutoencoderKL
import requests
from tqdm import tqdm


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


def profile_duration_distribution(num_samples: int = 10000):
    """Sample dataset to get duration distribution."""
    print("\n" + "=" * 60)
    print("1. DURATION DISTRIBUTION ANALYSIS")
    print("=" * 60)

    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    buckets = defaultdict(int)
    bucket_ranges = [
        (0, 5, "0-5s"),
        (5, 10, "5-10s"),
        (10, 20, "10-20s"),
        (20, 30, "20-30s"),
        (30, 45, "30-45s"),
        (45, 60, "45-60s"),
        (60, 120, "60-120s"),
        (120, 999, "120s+"),
    ]

    valid_urls = 0
    valid_captions = 0

    print(f"Sampling {num_samples} videos...")
    for i, sample in enumerate(tqdm(ds, total=num_samples)):
        if i >= num_samples:
            break

        duration = parse_duration(sample.get('duration', ''))
        url = sample.get('contentUrl', '')
        caption = sample.get('name', '')

        if url:
            valid_urls += 1
        if caption and len(caption) >= 5:
            valid_captions += 1

        for low, high, label in bucket_ranges:
            if low <= duration < high:
                buckets[label] += 1
                break

    print("\nDuration distribution:")
    print("-" * 40)
    total_30_60 = 0
    for low, high, label in bucket_ranges:
        count = buckets[label]
        pct = count / num_samples * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:10s}: {count:5d} ({pct:5.1f}%) {bar}")
        if 30 <= low < 60 or (low < 30 and high > 30):
            if label in ["30-45s", "45-60s"]:
                total_30_60 += count

    print("-" * 40)
    print(f"  30-60s total: {total_30_60} ({total_30_60/num_samples*100:.1f}%)")
    print(f"  Valid URLs: {valid_urls/num_samples*100:.1f}%")
    print(f"  Valid captions: {valid_captions/num_samples*100:.1f}%")

    # Estimate for 100K target
    success_rate_30_60 = total_30_60 / num_samples
    if success_rate_30_60 > 0:
        videos_needed = int(100000 / success_rate_30_60)
        print(f"\n  To get 100K 30-60s videos, need to scan ~{videos_needed:,} videos")
        print(f"  WebVid-10M has 10M videos, so this is {'feasible' if videos_needed < 10_000_000 else 'NOT feasible'}")
    else:
        print("\n  WARNING: No 30-60s videos found in sample!")

    return buckets, total_30_60 / num_samples


def profile_download_speed(num_samples: int = 20):
    """Test actual download speeds."""
    print("\n" + "=" * 60)
    print("2. DOWNLOAD SPEED PROFILING")
    print("=" * 60)

    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    times = []
    sizes = []
    failures = 0

    print(f"Testing {num_samples} downloads...")
    for i, sample in enumerate(ds):
        if len(times) >= num_samples:
            break

        url = sample.get('contentUrl', '')
        if not url:
            continue

        try:
            start = time.time()
            response = requests.get(url, timeout=20, stream=True)
            if response.status_code != 200:
                failures += 1
                continue

            content = b''
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                content += chunk
                if len(content) > 50 * 1024 * 1024:  # 50MB limit
                    break

            elapsed = time.time() - start
            times.append(elapsed)
            sizes.append(len(content) / 1024 / 1024)  # MB
            print(f"  [{len(times):2d}] {elapsed:.2f}s, {sizes[-1]:.1f}MB")

        except Exception as e:
            failures += 1

    if times:
        print(f"\nDownload stats (n={len(times)}):")
        print(f"  Mean time: {np.mean(times):.2f}s")
        print(f"  Median time: {np.median(times):.2f}s")
        print(f"  Mean size: {np.mean(sizes):.1f}MB")
        print(f"  Throughput: {np.mean(sizes)/np.mean(times):.1f} MB/s")
        print(f"  Failure rate: {failures/(len(times)+failures)*100:.1f}%")

    return np.mean(times) if times else None


def profile_ffmpeg_speed(num_frames: int = 32):
    """Test ffmpeg extraction speed."""
    print("\n" + "=" * 60)
    print(f"3. FFMPEG EXTRACTION SPEED ({num_frames} frames)")
    print("=" * 60)

    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    times = []

    print(f"Testing extraction on 10 videos...")
    for i, sample in enumerate(ds):
        if len(times) >= 10:
            break

        url = sample.get('contentUrl', '')
        duration = parse_duration(sample.get('duration', ''))

        # Only test 30-60s videos
        if not url or duration < 30 or duration > 60:
            continue

        try:
            # Download
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                continue

            video_bytes = response.content

            # Extract frames
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
                f.write(video_bytes)
                f.flush()

                with tempfile.TemporaryDirectory() as tmpdir:
                    fps = num_frames / max(duration, 1.0)

                    start = time.time()
                    cmd = [
                        'ffmpeg', '-i', f.name,
                        '-vf', f'fps={fps:.4f},scale=256:256:force_original_aspect_ratio=increase,crop=256:256',
                        '-frames:v', str(num_frames),
                        '-q:v', '5',
                        f'{tmpdir}/frame_%04d.jpg',
                        '-y', '-loglevel', 'error'
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=60)
                    elapsed = time.time() - start

                    if result.returncode == 0:
                        frame_count = len(list(Path(tmpdir).glob('frame_*.jpg')))
                        times.append(elapsed)
                        print(f"  [{len(times):2d}] {elapsed:.2f}s, {frame_count} frames, {duration}s video")
        except Exception as e:
            print(f"  Failed: {e}")

    if times:
        print(f"\nffmpeg stats (n={len(times)}):")
        print(f"  Mean time: {np.mean(times):.2f}s")
        print(f"  Median time: {np.median(times):.2f}s")

    return np.mean(times) if times else None


def profile_vae_speed(num_frames: int = 32, num_batches: int = 5):
    """Test VAE encoding speed."""
    print("\n" + "=" * 60)
    print(f"4. VAE ENCODING SPEED ({num_frames} frames)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("  WARNING: No GPU available, skipping VAE profiling")
        return None

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Create dummy frames
    dummy_frames = torch.randn(num_frames, 3, 256, 256).to(device)

    # Warmup
    with torch.no_grad():
        _ = vae.encode(dummy_frames[:4]).latent_dist.mean
    torch.cuda.synchronize()

    times = []
    print(f"Testing {num_batches} batches of {num_frames} frames...")

    for i in range(num_batches):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            latents = vae.encode(dummy_frames).latent_dist.mean
            latents = latents * vae.config.scaling_factor

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{i+1}] {elapsed:.3f}s, peak mem: {mem:.2f}GB")

    print(f"\nVAE stats:")
    print(f"  Mean time: {np.mean(times):.3f}s for {num_frames} frames")
    print(f"  Per-frame: {np.mean(times)/num_frames*1000:.1f}ms")
    print(f"  Latent shape: [{num_frames}, 4, 32, 32]")
    print(f"  Latent size: {num_frames * 4 * 32 * 32 * 2 / 1024:.1f}KB (bfloat16)")

    return np.mean(times)


def check_storage():
    """Check storage paths."""
    print("\n" + "=" * 60)
    print("5. STORAGE CHECK")
    print("=" * 60)

    paths = [
        "/mnt/d/projects/fVLM/data",
        "/mnt/d/projects/fVLM/data/precomputed_32f",
        "/mnt/d",
        "/mnt/c/Users/sanps/Desktop/Projects/dino/nanoSmolLM/fVLM/data",
    ]

    for path in paths:
        p = Path(path)
        exists = p.exists()

        if exists or p.parent.exists():
            # Get disk usage
            check_path = p if exists else p.parent
            try:
                import shutil
                total, used, free = shutil.disk_usage(check_path)
                print(f"  {path}")
                print(f"    Exists: {exists}")
                print(f"    Free space: {free/1e9:.1f} GB")
                print(f"    Total space: {total/1e9:.1f} GB")
            except:
                print(f"  {path}: exists={exists}, couldn't get disk usage")
        else:
            print(f"  {path}: NOT FOUND")


def main():
    print("=" * 60)
    print("PRECOMPUTE PIPELINE PROFILING")
    print("=" * 60)

    # 1. Duration distribution
    buckets, rate_30_60 = profile_duration_distribution(num_samples=10000)

    # 2. Download speed
    avg_download = profile_download_speed(num_samples=15)

    # 3. ffmpeg speed
    avg_ffmpeg = profile_ffmpeg_speed(num_frames=32)

    # 4. VAE speed
    avg_vae = profile_vae_speed(num_frames=32)

    # 5. Storage
    check_storage()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    print(f"\n30-60s video availability: {rate_30_60*100:.1f}%")

    if avg_download and avg_ffmpeg and avg_vae:
        total_per_video = avg_download + avg_ffmpeg + avg_vae
        print(f"\nPer-video time breakdown:")
        print(f"  Download: {avg_download:.2f}s ({avg_download/total_per_video*100:.0f}%)")
        print(f"  ffmpeg:   {avg_ffmpeg:.2f}s ({avg_ffmpeg/total_per_video*100:.0f}%)")
        print(f"  VAE:      {avg_vae:.2f}s ({avg_vae/total_per_video*100:.0f}%)")
        print(f"  TOTAL:    {total_per_video:.2f}s per video (serial)")

        # With parallelism
        # Download can be parallelized with workers
        # ffmpeg runs per-video (parallel with download)
        # VAE is GPU-bound (serial, but can batch)

        # Estimate with 16 download workers
        # Bottleneck becomes max(download/16, ffmpeg, vae)
        parallel_bottleneck = max(avg_download/16, avg_ffmpeg, avg_vae)

        print(f"\nWith 16 download workers:")
        print(f"  Effective per-video: ~{parallel_bottleneck:.2f}s")
        print(f"  100K videos: ~{100000 * parallel_bottleneck / 3600:.1f} hours")
        print(f"  50K videos:  ~{50000 * parallel_bottleneck / 3600:.1f} hours")

        # Account for failure rate and filtering
        effective_rate = rate_30_60 * 0.7  # Assume 30% download failures
        print(f"\nAccounting for filtering + failures (~{effective_rate*100:.0f}% success):")
        adjusted_time_100k = 100000 / effective_rate * parallel_bottleneck / 3600
        print(f"  100K videos: ~{adjusted_time_100k:.1f} hours")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
