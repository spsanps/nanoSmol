#!/usr/bin/env python3
"""Precompute VAE latents for all frame files."""

import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device, dtype=torch.float32)
    vae.eval()

    # Directories with frame files that need latent conversion
    frame_dirs = [
        ("data/frames", "data/frames_latents"),
        ("data/webvid/frames", "data/webvid/latents"),
        ("data/webvid_large/frames", "data/webvid_large/latents"),
        ("data/webvid_test/frames", "data/webvid_test/latents"),
    ]

    total_processed = 0

    for src_dir, dst_dir in frame_dirs:
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)

        if not src_path.exists():
            print(f"Skipping {src_dir} (not found)")
            continue

        dst_path.mkdir(parents=True, exist_ok=True)

        # Get all frame files
        frame_files = sorted(list(src_path.glob("*.pt")))

        if len(frame_files) == 0:
            print(f"Skipping {src_dir} (no files)")
            continue

        # Check if already latents
        sample = torch.load(frame_files[0], weights_only=True)
        if sample.shape[1] == 4:
            print(f"Skipping {src_dir} (already latents)")
            continue

        print(f"\nProcessing {src_dir} -> {dst_dir}")
        print(f"  {len(frame_files)} files to process")

        for frame_file in tqdm(frame_files, desc=f"Encoding {src_dir}"):
            dst_file = dst_path / frame_file.name

            # Skip if already exists
            if dst_file.exists():
                continue

            # Load frames
            frames = torch.load(frame_file, weights_only=True)  # (T, 3, 256, 256) uint8

            # Normalize
            frames = frames.float() / 255.0  # [0, 1]
            frames = frames * 2 - 1  # [-1, 1]

            # Encode with VAE
            with torch.no_grad():
                frames = frames.to(device, dtype=torch.float32)
                latents = vae.encode(frames).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(torch.bfloat16).cpu()

            # Save
            torch.save(latents, dst_file)
            total_processed += 1

    print(f"\nDone! Processed {total_processed} files")

if __name__ == "__main__":
    main()
