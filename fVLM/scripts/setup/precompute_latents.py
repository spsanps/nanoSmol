"""
Precompute VAE latents for all videos.

This is run ONCE before training to avoid VAE forward passes during training.
Saves ~10x training speed.

Output: data/latents/{video_id}.pt containing [T, 4, 32, 32] tensor
"""

import torch
import sys
from pathlib import Path
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.sampling import sample_frames


# Configuration
INPUT_DIR = Path("data/videos")
OUTPUT_DIR = Path("data/latents")
NUM_FRAMES = 8
FRAME_SIZE = 256
BATCH_PROCESS = 4  # Process 4 frames at a time to save memory


def encode_video(video_path: Path, vae: AutoencoderKL) -> torch.Tensor:
    """
    Encode video frames to VAE latents.

    Args:
        video_path: Path to video file
        vae: Frozen VAE encoder

    Returns:
        latents: [T, 4, 32, 32] tensor (scaled)
    """
    # Sample frames for VAE (needs [-1, 1] range)
    frames = sample_frames(
        str(video_path),
        num_frames=NUM_FRAMES,
        target_size=FRAME_SIZE,
        normalize_for='vae'
    )  # [T, 3, 256, 256]

    frames = frames.to(dtype=torch.bfloat16, device='cuda')

    # Encode in batches to save memory
    latents = []
    for i in range(0, frames.shape[0], BATCH_PROCESS):
        batch = frames[i:i+BATCH_PROCESS]  # [B, 3, 256, 256]

        # Encode to latent
        with torch.no_grad():
            lat = vae.encode(batch).latent_dist.mean  # [B, 4, 32, 32]
            lat = lat * vae.config.scaling_factor  # Apply scaling
            latents.append(lat.cpu())

    latents = torch.cat(latents, dim=0)  # [T, 4, 32, 32]
    return latents


def main():
    print("=" * 70)
    print("VAE Latent Preprocessing")
    print("=" * 70)

    # Load frozen VAE
    print("\n📦 Loading Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.bfloat16
    ).cuda().eval()
    print(f"   ✓ VAE loaded (scaling_factor: {vae.config.scaling_factor})")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Find all videos
    video_paths = sorted(list(INPUT_DIR.glob("**/*.mp4")))
    print(f"\n📂 Found {len(video_paths)} videos")

    # Check how many already processed
    existing = set(p.stem for p in OUTPUT_DIR.glob("*.pt"))
    to_process = [v for v in video_paths if v.stem not in existing]

    if len(existing) > 0:
        print(f"   ⏭️  Skipping {len(existing)} already processed videos")

    if len(to_process) == 0:
        print("\n✓ All videos already processed!")
        return

    print(f"   🔄 Processing {len(to_process)} videos...")

    # Process videos
    success = 0
    failed = 0
    failed_videos = []

    for video_path in tqdm(to_process, desc="Encoding videos"):
        output_path = OUTPUT_DIR / f"{video_path.stem}.pt"

        try:
            latents = encode_video(video_path, vae)
            torch.save(latents, output_path)
            success += 1

        except Exception as e:
            failed += 1
            failed_videos.append((video_path.stem, str(e)))
            # Don't print individual errors, will summarize at end

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Successfully processed: {success} videos")
    if failed > 0:
        print(f"✗ Failed: {failed} videos")
        print(f"\nFailed videos (first 10):")
        for video_id, error in failed_videos[:10]:
            print(f"  - {video_id}: {error[:50]}...")

    print(f"\n📊 Storage info:")
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.pt"))
    print(f"   Total latent files: {len(list(OUTPUT_DIR.glob('*.pt')))}")
    print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"   Average per video: {total_size / len(list(OUTPUT_DIR.glob('*.pt'))) / 1024:.1f} KB")

    print("\n✓ Preprocessing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
