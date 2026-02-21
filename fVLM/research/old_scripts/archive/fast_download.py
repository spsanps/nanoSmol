#!/usr/bin/env python
"""
Fast parallel WebVid downloader using multiple processes.

Downloads videos in parallel using multiprocessing for maximum throughput.
Processes VAE latents in batches on GPU.

Target: 50K+ videos for proper training without repeats.
"""

import torch
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
import re
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from diffusers import AutoencoderKL
import argparse
import os


def parse_duration(dur_str: str) -> int:
    match = re.match(r'PT(\d+)H(\d+)M(\d+)S', dur_str)
    if match:
        h, m, s = int(match[1]), int(match[2]), int(match[3])
        return h * 3600 + m * 60 + s
    return 0


def download_single(args):
    """Download and extract frames for a single video."""
    video_id, url, caption, num_frames, frame_size, output_dir = args

    frames_path = output_dir / 'frames' / f'{video_id}.pt'
    if frames_path.exists():
        return {'video_id': video_id, 'status': 'exists', 'caption': caption}

    try:
        # Download
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return {'video_id': video_id, 'status': 'download_failed'}

        # Write to temp and extract
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
                    return {'video_id': video_id, 'status': 'extract_failed'}

                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    return {'video_id': video_id, 'status': 'not_enough_frames'}

                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                    frames.append(frame)

                frames = torch.stack(frames)  # [T, 3, H, W] uint8
                torch.save(frames, frames_path)

                return {'video_id': video_id, 'status': 'downloaded', 'caption': caption}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        return {'video_id': video_id, 'status': 'error', 'error': str(e)}


def compute_latents_batch(frames_dir, latents_dir, vae, device, batch_size=16):
    """Compute VAE latents for all downloaded frames."""
    frames_files = list(frames_dir.glob('*.pt'))

    # Find frames without latents
    to_process = []
    for fp in frames_files:
        lp = latents_dir / fp.name
        if not lp.exists():
            to_process.append(fp)

    if not to_process:
        return 0

    print(f"Computing latents for {len(to_process)} videos...")

    count = 0
    for fp in tqdm(to_process, desc="VAE encoding"):
        frames = torch.load(fp, weights_only=True)  # [T, 3, H, W] uint8

        # Normalize for VAE
        frames_vae = frames.float() / 255.0 * 2 - 1
        frames_vae = frames_vae.to(device)

        with torch.no_grad():
            latents = []
            for i in range(0, frames_vae.shape[0], 4):
                batch = frames_vae[i:i+4]
                latent = vae.encode(batch).latent_dist.sample()
                latent = latent * 0.18215
                latents.append(latent)
            latents = torch.cat(latents, dim=0).cpu()

        torch.save(latents, latents_dir / fp.name)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/webvid_large')
    parser.add_argument('--num_videos', type=int, default=50000)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--frame_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--min_duration', type=int, default=8)
    parser.add_argument('--max_duration', type=int, default=60)
    parser.add_argument('--latent_batch_interval', type=int, default=1000,
                        help='Compute latents every N downloads')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / 'frames'
    latents_dir = output_dir / 'latents'
    frames_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = len(list(frames_dir.glob('*.pt')))
    print(f"Found {existing} existing videos")

    if existing >= args.num_videos:
        print("Already have enough videos!")
        return

    # Load VAE for latent computation
    print("Loading VAE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Load dataset
    print("Loading WebVid dataset...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Prepare download tasks
    print(f"Preparing download tasks (target: {args.num_videos})...")

    tasks = []
    captions = {}
    filtered = 0

    for sample in ds:
        video_id = str(sample['videoid'])

        # Skip if already downloaded
        if (frames_dir / f'{video_id}.pt').exists():
            continue

        # Check duration
        duration = parse_duration(sample['duration'])
        if duration < args.min_duration or duration > args.max_duration:
            filtered += 1
            continue

        tasks.append((
            video_id,
            sample['contentUrl'],
            sample['name'],
            args.num_frames,
            args.frame_size,
            output_dir,
        ))

        if len(tasks) + existing >= args.num_videos * 2:  # Buffer for failures
            break

    print(f"Prepared {len(tasks)} download tasks ({filtered} filtered by duration)")

    # Download with multiprocessing
    print(f"Downloading with {args.num_workers} workers...")

    success = 0
    failed = 0
    downloaded = existing

    with Pool(args.num_workers) as pool:
        pbar = tqdm(total=args.num_videos - existing, desc="Downloading")

        for result in pool.imap_unordered(download_single, tasks):
            if result['status'] in ['downloaded', 'exists']:
                success += 1
                downloaded += 1
                captions[result['video_id']] = result.get('caption', '')
                pbar.update(1)

                # Periodically compute latents
                if downloaded % args.latent_batch_interval == 0:
                    pbar.set_description("Computing latents...")
                    compute_latents_batch(frames_dir, latents_dir, vae, device)
                    pbar.set_description("Downloading")

            else:
                failed += 1

            pbar.set_postfix({
                'success': success,
                'failed': failed,
                'total': downloaded,
            })

            if downloaded >= args.num_videos:
                break

        pbar.close()

    # Final latent computation
    print("\nFinal latent computation...")
    compute_latents_batch(frames_dir, latents_dir, vae, device)

    # Save captions
    caption_path = output_dir / 'captions.json'
    if caption_path.exists():
        with open(caption_path) as f:
            existing_captions = json.load(f)
        captions.update(existing_captions)

    with open(caption_path, 'w') as f:
        json.dump(captions, f, indent=2)

    print(f"\nDone!")
    print(f"  Downloaded: {success}")
    print(f"  Failed: {failed}")
    print(f"  Total videos: {len(list(frames_dir.glob('*.pt')))}")
    print(f"  Total latents: {len(list(latents_dir.glob('*.pt')))}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
