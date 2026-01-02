"""
WebVid Video Downloader

Downloads videos from WebVid-10M dataset URLs in parallel.
Processes them into our training format (frames + VAE latents).
"""

import os
import sys
import torch
import requests
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from datasets import load_dataset
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image
import io
import tempfile
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_video(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a video from URL."""
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


def extract_frames_ffmpeg(video_path: Path, num_frames: int = 16, size: int = 256) -> torch.Tensor:
    """Extract frames from video using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
            '-frames:v', str(num_frames * 2),  # Extract more, sample later
            '-q:v', '2',
            f'{tmpdir}/frame_%04d.jpg',
            '-y', '-loglevel', 'error'
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return None

        # Load frames
        frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
        if len(frame_files) < num_frames:
            return None

        # Sample evenly
        indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)

        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert('RGB')
            frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [3, H, W]
            frames.append(frame)

        return torch.stack(frames)  # [T, 3, H, W]


def compute_vae_latents(vae, frames: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """Compute VAE latents for frames."""
    # Normalize to [-1, 1]
    frames = frames.float() / 255.0 * 2 - 1
    frames = frames.to(device)

    with torch.no_grad():
        latents = []
        for i in range(frames.shape[0]):
            frame = frames[i:i+1]  # [1, 3, H, W]
            latent = vae.encode(frame).latent_dist.sample()
            latent = latent * 0.18215  # Scale factor
            latents.append(latent)

        return torch.cat(latents, dim=0).cpu()  # [T, 4, 32, 32]


def process_video(
    video_id: str,
    url: str,
    caption: str,
    output_dir: Path,
    vae,
    device: str,
    num_frames: int = 16,
    frame_size: int = 256,
) -> dict:
    """Download and process a single video."""
    frames_dir = output_dir / 'frames'
    latents_dir = output_dir / 'latents'

    # Check if already processed
    frames_path = frames_dir / f'{video_id}.pt'
    latents_path = latents_dir / f'{video_id}.pt'

    if frames_path.exists() and latents_path.exists():
        return {'video_id': video_id, 'status': 'exists', 'caption': caption}

    # Download video to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = Path(f.name)

    try:
        if not download_video(url, temp_path):
            return {'video_id': video_id, 'status': 'download_failed'}

        # Extract frames
        frames = extract_frames_ffmpeg(temp_path, num_frames, frame_size)
        if frames is None:
            return {'video_id': video_id, 'status': 'extract_failed'}

        # Compute latents
        latents = compute_vae_latents(vae, frames, device)

        # Save
        torch.save(frames, frames_path)
        torch.save(latents, latents_path)

        return {'video_id': video_id, 'status': 'success', 'caption': caption}

    finally:
        temp_path.unlink(missing_ok=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/webvid')
    parser.add_argument('--num_videos', type=int, default=10000)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--frame_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / 'frames').mkdir(parents=True, exist_ok=True)
    (output_dir / 'latents').mkdir(parents=True, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(args.device)
    vae.eval()

    # Load WebVid dataset
    print("Loading WebVid dataset...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Process videos
    print(f"Downloading and processing {args.num_videos} videos...")

    success_count = 0
    fail_count = 0
    captions = {}

    pbar = tqdm(total=args.num_videos, desc="Processing")

    for i, sample in enumerate(ds):
        if success_count >= args.num_videos:
            break

        video_id = str(sample['videoid'])
        url = sample['contentUrl']
        caption = sample['name']

        result = process_video(
            video_id=video_id,
            url=url,
            caption=caption,
            output_dir=output_dir,
            vae=vae,
            device=args.device,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
        )

        if result['status'] == 'success' or result['status'] == 'exists':
            success_count += 1
            captions[video_id] = caption
            pbar.update(1)
        else:
            fail_count += 1

        pbar.set_postfix({
            'success': success_count,
            'failed': fail_count,
            'rate': f"{success_count/(success_count+fail_count+1e-6)*100:.1f}%"
        })

    pbar.close()

    # Save captions
    with open(output_dir / 'captions.json', 'w') as f:
        json.dump(captions, f, indent=2)

    print(f"\nDone! Downloaded {success_count} videos, {fail_count} failed")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
