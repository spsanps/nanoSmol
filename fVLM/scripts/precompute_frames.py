"""
Pre-decode video frames to eliminate H264 decoding bottleneck during training.

Saves frames as uint8 tensors - normalize to ImageNet stats on load.
Expected storage: ~1.28 GB for 813 videos × 8 frames
Expected speedup: 20-100x on data loading
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import decord
import torchvision.transforms.functional as TF

decord.bridge.set_bridge('torch')

# Config
VIDEO_DIR = Path("data/videos")
FRAMES_DIR = Path("data/frames")
NUM_FRAMES = 8
FRAME_SIZE = 256


def sample_frames_uint8(video_path: str, num_frames: int = 8, target_size: int = 256) -> torch.Tensor:
    """
    Sample frames from video and return as uint8 tensor (no normalization).

    Returns:
        frames: [T, 3, H, W] uint8 tensor in [0, 255] range
    """
    try:
        vr = decord.VideoReader(str(video_path))
        total_frames = len(vr)

        if total_frames < num_frames:
            # Repeat frames if video too short
            indices = list(range(total_frames)) * (num_frames // total_frames + 1)
            indices = indices[:num_frames]
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        frames = vr.get_batch(indices)  # [T, H, W, C] uint8

        # Resize if needed
        if frames.shape[1] != target_size or frames.shape[2] != target_size:
            frames_resized = []
            for i in range(frames.shape[0]):
                frame = frames[i].permute(2, 0, 1)  # [C, H, W]
                frame = TF.resize(frame, [target_size, target_size], antialias=True)
                frames_resized.append(frame)
            frames = torch.stack(frames_resized)  # [T, C, H, W]
        else:
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        return frames.to(torch.uint8)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def main():
    print("=" * 60)
    print("Pre-decoding Video Frames")
    print("=" * 60)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all videos that have VAE latents (our working set)
    latent_dir = Path("data/latents")
    latent_files = list(latent_dir.glob("*.pt"))
    print(f"\nFound {len(latent_files)} videos with VAE latents")

    # Build video path mapping
    print("Scanning video directory...")
    video_map = {}
    for video_path in VIDEO_DIR.glob("**/*.mp4"):
        video_map[video_path.stem] = video_path
    print(f"Found {len(video_map)} videos")

    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    for latent_path in tqdm(latent_files, desc="Pre-decoding"):
        video_id = latent_path.stem
        output_path = FRAMES_DIR / f"{video_id}.pt"

        # Skip if already processed
        if output_path.exists():
            skipped += 1
            continue

        # Find video
        if video_id not in video_map:
            failed += 1
            continue

        video_path = video_map[video_id]

        # Decode and save
        frames = sample_frames_uint8(str(video_path), NUM_FRAMES, FRAME_SIZE)

        if frames is not None:
            torch.save(frames, output_path)
            processed += 1
            total_bytes += frames.numel()
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("Pre-decoding Complete!")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total storage: {total_bytes / 1e9:.2f} GB")
    print(f"Output directory: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
