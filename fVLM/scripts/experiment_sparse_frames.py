#!/usr/bin/env python3
"""
Sparse Frame Reconstruction Experiment

Hypothesis: Foveated attention helps reconstruction when frames have
significant temporal gaps (more motion/change between frames).

Key changes from original:
- Sample frames with large temporal gaps (e.g., 1-2 fps from 30fps video)
- Predict frame at t+N where N is large (e.g., 2-5 seconds later)
- This forces the model to understand motion/dynamics, not just copy
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel
from diffusers import AutoencoderKL

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_vae(device):
    """Load frozen SD-VAE for encoding targets."""
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def encode_to_vae_latent(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode frames to VAE latents.

    Args:
        vae: SD-VAE model
        frames: [B, C, H, W] in [0, 1] range

    Returns:
        latents: [B, 4, 32, 32]
    """
    # VAE expects [-1, 1] range
    frames_scaled = frames * 2.0 - 1.0

    with torch.no_grad():
        latent_dist = vae.encode(frames_scaled).latent_dist
        latents = latent_dist.mean * 0.18215  # SD scaling factor

    return latents


def parse_duration(dur_str: str) -> int:
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


def download_video(url: str, timeout: int = 15) -> bytes:
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


def extract_sparse_frames(video_bytes: bytes, num_frames: int, size: int,
                          target_fps: float = 1.0) -> tuple:
    """Extract frames with sparse temporal sampling.

    Args:
        video_bytes: Raw video bytes
        num_frames: Number of context frames to extract
        size: Frame size (square)
        target_fps: Target FPS for sampling (lower = more sparse)

    Returns:
        (context_frames, target_frame) or (None, None) if failed
        context_frames: [num_frames, C, H, W] tensor
        target_frame: [C, H, W] tensor (the frame to predict)
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            # Get video info
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration,r_frame_rate',
                '-of', 'csv=p=0',
                f.name
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            if probe_result.returncode != 0:
                return None, None

            parts = probe_result.stdout.strip().split(',')
            if len(parts) < 2:
                return None, None

            # Parse frame rate
            fps_str = parts[0]
            if '/' in fps_str:
                num, den = fps_str.split('/')
                video_fps = float(num) / float(den)
            else:
                video_fps = float(fps_str)

            # Parse duration
            try:
                duration = float(parts[1])
            except:
                duration = 10.0  # Default

            # Calculate frame sampling
            # We want num_frames + 1 frames (context + target)
            total_frames_needed = num_frames + 1

            # Minimum time span needed (in seconds)
            min_span = total_frames_needed / target_fps

            if duration < min_span:
                return None, None

            # Sample times evenly across the video
            # Leave some margin at start/end
            margin = 0.5  # seconds
            usable_duration = duration - 2 * margin

            if usable_duration < min_span:
                return None, None

            # Generate timestamps for sparse sampling
            frame_times = np.linspace(margin, duration - margin, total_frames_needed)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract all needed frames at once
                frames = []
                for i, t in enumerate(frame_times):
                    cmd = [
                        'ffmpeg', '-ss', str(t), '-i', f.name,
                        '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                        '-frames:v', '1',
                        '-q:v', '2',
                        f'{tmpdir}/frame_{i:04d}.jpg',
                        '-y', '-loglevel', 'error'
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=10)

                    frame_path = Path(tmpdir) / f'frame_{i:04d}.jpg'
                    if not frame_path.exists():
                        return None, None

                    img = Image.open(frame_path).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)

                if len(frames) != total_frames_needed:
                    return None, None

                # Split into context and target
                context_frames = torch.stack(frames[:-1])  # [num_frames, C, H, W]
                target_frame = frames[-1]  # [C, H, W]

                return context_frames, target_frame

    except Exception as e:
        return None, None


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    if frames.dim() == 4:  # [T, C, H, W]
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    return (frames - mean) / std


def webvid_sparse_generator(num_frames=8, frame_size=256, target_fps=1.0,
                            min_duration=10, max_duration=30):
    """Generator for WebVid samples with sparse frame extraction.

    Args:
        num_frames: Number of context frames
        frame_size: Frame size
        target_fps: Target FPS (1.0 = 1 frame per second)
        min_duration: Minimum video duration (ensures enough temporal span)
        max_duration: Maximum video duration
    """
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    for sample in ds:
        try:
            duration = parse_duration(sample.get('duration', ''))

            # Need longer videos for sparse sampling
            if duration < min_duration or duration > max_duration:
                continue

            url = sample.get('contentUrl')
            caption = sample.get('name', '')

            if not url or not caption:
                continue

            video_bytes = download_video(url)
            if video_bytes is None:
                continue

            context_frames, target_frame = extract_sparse_frames(
                video_bytes, num_frames, frame_size, target_fps
            )

            if context_frames is None:
                continue

            yield context_frames, target_frame, caption, duration

        except Exception:
            continue


def main(
    steps: int = 1000,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 3e-5,
    num_frames: int = 8,
    target_fps: float = 1.0,  # 1 frame per second = very sparse
    min_duration: int = 10,
    use_wandb: bool = True,
):
    """Run sparse frame reconstruction experiment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SPARSE FRAME RECONSTRUCTION EXPERIMENT")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Steps: {steps}")
    print(f"Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"Frames: {num_frames} context + 1 target")
    print(f"Target FPS: {target_fps} (frame every {1/target_fps:.1f}s)")
    print(f"Min video duration: {min_duration}s")
    print("=" * 70)

    if use_wandb:
        import wandb
        wandb.init(
            project="foveated-vlm-sparse",
            config={
                "steps": steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "lr": lr,
                "num_frames": num_frames,
                "target_fps": target_fps,
                "min_duration": min_duration,
                "experiment": "sparse_reconstruction",
            }
        )

    # Load model
    print("\nLoading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=False,  # Trainable for reconstruction
    ).to(device)

    # Load VAE for encoding targets
    print("Loading VAE...")
    vae = load_vae(device)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Data generator
    data_gen = webvid_sparse_generator(
        num_frames=num_frames,
        target_fps=target_fps,
        min_duration=min_duration,
    )

    # Training loop
    model.train()
    step = 0
    accum_loss_fine = 0.0
    accum_loss_coarse = 0.0
    accum_count = 0

    ratio_history = []

    print("\nStarting training...")
    print("(Looking for longer videos with sparse frames)")

    while step < steps:
        optimizer.zero_grad()

        batch_loss_fine = 0.0
        batch_loss_coarse = 0.0
        valid_samples = 0

        for _ in range(grad_accum):
            try:
                context_frames, target_frame, caption, duration = next(data_gen)
            except StopIteration:
                print("Data exhausted, restarting...")
                data_gen = webvid_sparse_generator(
                    num_frames=num_frames,
                    target_fps=target_fps,
                    min_duration=min_duration,
                )
                context_frames, target_frame, caption, duration = next(data_gen)

            # Move to device and normalize
            context_frames = context_frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
            target_frame = target_frame.unsqueeze(0).to(device)  # [1, C, H, W]

            context_norm = normalize_frames(context_frames)
            target_norm = normalize_frames(target_frame)

            # Compute target VAE latent (what we want to predict)
            # target_frame is [1, C, H, W] in [0,1] range
            target_latent = encode_to_vae_latent(vae, target_frame)  # [1, 4, 32, 32]

            # Also encode context frames to VAE latents for the model
            T = context_frames.shape[1]
            context_latents = encode_to_vae_latent(
                vae, context_frames.squeeze(0)  # [T, C, H, W]
            )  # [T, 4, 32, 32]

            # Add batch dimension and append target as last frame for full sequence
            # The model predicts frame t+1 from frames 0..t
            all_frames = torch.cat([
                context_norm.squeeze(0),  # [T, C, H, W]
                target_norm  # [1, C, H, W]
            ], dim=0).unsqueeze(0)  # [1, T+1, C, H, W]

            all_latents = torch.cat([
                context_latents,  # [T, 4, 32, 32]
                target_latent  # [1, 4, 32, 32]
            ], dim=0).unsqueeze(0)  # [1, T+1, 4, 32, 32]

            # Get empty text embeds (self-supervised mode)
            text_embeds = model.get_empty_text_embeds(1)  # [1, 1, llm_dim]

            # Use model forward pass - returns (combined_loss, loss_fine, loss_coarse)
            loss, loss_fine, loss_coarse = model(text_embeds, all_frames, all_latents)

            # Scale for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            batch_loss_fine += loss_fine.item()
            batch_loss_coarse += loss_coarse.item()
            valid_samples += 1

        if valid_samples > 0:
            optimizer.step()

            avg_fine = batch_loss_fine / valid_samples
            avg_coarse = batch_loss_coarse / valid_samples
            ratio = avg_coarse / avg_fine if avg_fine > 0 else 1.0

            accum_loss_fine += avg_fine
            accum_loss_coarse += avg_coarse
            accum_count += 1
            ratio_history.append(ratio)

            step += 1

            if step % 25 == 0:
                avg_ratio = accum_loss_coarse / accum_loss_fine if accum_loss_fine > 0 else 1.0
                status = "FINE_BETTER" if avg_ratio > 1.0 else "COARSE_BETTER"

                print(f"Step {step:4d} | fine={avg_fine:.4f} coarse={avg_coarse:.4f} | "
                      f"ratio={ratio:.4f} ({status})")

                if use_wandb:
                    wandb.log({
                        "loss_fine": avg_fine,
                        "loss_coarse": avg_coarse,
                        "ratio": ratio,
                        "ratio_avg": avg_ratio,
                        "step": step,
                    })

                accum_loss_fine = 0.0
                accum_loss_coarse = 0.0
                accum_count = 0

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    avg_ratio = sum(ratio_history) / len(ratio_history) if ratio_history else 1.0
    ratio_above_1 = sum(1 for r in ratio_history if r > 1.0) / len(ratio_history) * 100

    print(f"Average ratio: {avg_ratio:.4f}")
    print(f"Steps with ratio > 1.0: {ratio_above_1:.1f}%")
    print(f"Peak ratio: {max(ratio_history):.4f}")
    print(f"Min ratio: {min(ratio_history):.4f}")

    if avg_ratio > 1.05:
        print("\n✓ HYPOTHESIS VALIDATED: Sparse frames benefit from foveated attention!")
    elif avg_ratio > 1.0:
        print("\n~ MARGINAL: Slight benefit, may need more training")
    else:
        print("\n✗ NOT VALIDATED: Even sparse frames don't benefit")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--target_fps", type=float, default=1.0,
                       help="Target FPS for sampling (lower = more sparse)")
    parser.add_argument("--min_duration", type=int, default=10,
                       help="Minimum video duration in seconds")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    main(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        min_duration=args.min_duration,
        use_wandb=not args.no_wandb,
    )
