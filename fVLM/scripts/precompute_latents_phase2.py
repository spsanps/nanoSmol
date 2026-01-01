"""
Precompute VAE latents for Phase 2 videos.

Processes all videos in data/videos, computes VAE latents with 32 frames.
"""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from pathlib import Path
from tqdm import tqdm
import decord
import numpy as np
import argparse

decord.bridge.set_bridge('torch')


def sample_frames_uniform(video_path, num_frames, frame_size=256):
    """Uniformly sample frames from video."""
    try:
        vr = decord.VideoReader(str(video_path))
        total_frames = len(vr)

        if total_frames < num_frames:
            indices = list(range(total_frames))
            indices += [total_frames - 1] * (num_frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = vr.get_batch(indices)  # [T, H, W, C] uint8
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        # Resize
        if frames.shape[2] != frame_size or frames.shape[3] != frame_size:
            frames = F.interpolate(
                frames.float(),
                size=(frame_size, frame_size),
                mode='bilinear',
                align_corners=False
            ).byte()

        return frames
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None


def compute_latents(frames, vae, device):
    """Compute VAE latents for frames."""
    # frames: [T, 3, H, W] uint8
    frames_vae = frames.float().to(device) / 255.0 * 2 - 1

    with torch.no_grad():
        latents = []
        # Process in chunks to avoid OOM
        chunk_size = 4
        for i in range(0, frames_vae.shape[0], chunk_size):
            batch = frames_vae[i:i+chunk_size]
            latent = vae.encode(batch).latent_dist.sample() * 0.18215
            latents.append(latent.cpu())

        latents = torch.cat(latents, dim=0)

    return latents  # [T, 4, 32, 32]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='data/videos')
    parser.add_argument('--output_dir', type=str, default='data/latents_phase2')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--frame_size', type=int, default=256)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Find all videos
    video_paths = list(video_dir.glob("**/*.mp4"))
    video_paths += list(video_dir.glob("**/*.avi"))
    video_paths += list(video_dir.glob("**/*.mkv"))

    print(f"Found {len(video_paths)} videos")

    # Process
    success = 0
    failed = 0

    for video_path in tqdm(video_paths, desc="Computing latents"):
        output_path = output_dir / f"{video_path.stem}.pt"

        if output_path.exists():
            success += 1
            continue

        # Sample frames
        frames = sample_frames_uniform(video_path, args.num_frames, args.frame_size)
        if frames is None:
            failed += 1
            continue

        # Compute latents
        try:
            latents = compute_latents(frames, vae, device)
            torch.save(latents, output_path)
            success += 1
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
            failed += 1

    print(f"\nDone!")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
