#!/usr/bin/env python3
"""Quick profiling script to test memory and throughput before 8h run."""

import sys
import time
import torch
from torch.amp import GradScaler, autocast
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def extract_frames(video_bytes: bytes, num_frames: int, size: int = 256) -> torch.Tensor:
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
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)
                return torch.stack(frames)
    except:
        return None


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


@torch.no_grad()
def compute_vae_latents(vae, frames, device):
    B, T, C, H, W = frames.shape
    frames_vae = frames * 2 - 1
    frames_flat = frames_vae.reshape(B * T, C, H, W).to(device)
    latents_flat = vae.encode(frames_flat).latent_dist.sample()
    latents_flat = latents_flat * vae.config.scaling_factor
    latents = latents_flat.reshape(B, T, 4, H // 8, W // 8)
    return latents


def profile(batch_size=3, num_steps=5):
    """Profile training step with given batch size."""
    device = torch.device("cuda")

    print("=" * 60)
    print(f"PROFILING: batch_size={batch_size}, {num_steps} steps")
    print("=" * 60)

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n[1/4] Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    # Enable gradient checkpointing
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    model.train()

    model_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Model loaded: {model_mem:.2f} GB")

    print("[2/4] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    vae_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VAE loaded: {vae_mem:.2f} GB (delta: {vae_mem - model_mem:.2f} GB)")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resume from checkpoint
    checkpoint_path = Path("outputs/joint_recon_caption/checkpoints/step_008000.pt")
    if checkpoint_path.exists():
        print(f"[3/4] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Checkpoint loaded")
    else:
        print("[3/4] No checkpoint found, using random weights")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = GradScaler('cuda')

    print("[4/4] Getting sample data...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    samples = []
    for sample in ds:
        if len(samples) >= batch_size * num_steps:
            break
        try:
            duration = parse_duration(sample.get('duration', ''))
            if duration == 0 or duration > 15:
                continue
            url = sample.get('contentUrl')
            caption = sample.get('name', '')
            if not url or not caption or len(caption) < 5:
                continue
            video_bytes = download_video(url)
            if video_bytes is None:
                continue
            frames = extract_frames(video_bytes, 8)
            if frames is None:
                continue
            samples.append((frames, caption))
            print(f"  Collected {len(samples)}/{batch_size * num_steps} samples", end='\r')
        except:
            continue

    print(f"\n  Got {len(samples)} samples")

    if len(samples) < batch_size:
        print("ERROR: Not enough samples!")
        return

    # Run profiling
    print("\n" + "=" * 60)
    print("RUNNING FORWARD/BACKWARD PASSES")
    print("=" * 60)

    step_times = []
    peak_mems = []

    for step in range(num_steps):
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        # Get batch
        batch_idx = step * batch_size
        batch_frames = [samples[batch_idx + i][0] for i in range(batch_size)]
        batch_captions = [samples[batch_idx + i][1] for i in range(batch_size)]

        frames_raw = torch.stack(batch_frames)
        frames_norm = normalize_frames(frames_raw.to(device))
        vae_latents = compute_vae_latents(vae, frames_raw, device)

        tokens = tokenizer(
            batch_captions,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        caption_ids = tokens['input_ids'].to(device)
        caption_mask = tokens['attention_mask'].to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Captioning
            loss_cap_fine = model.forward_captioning(
                frames_norm, caption_ids, caption_mask, use_fine=True
            )
            loss_cap_coarse = model.forward_captioning(
                frames_norm, caption_ids, caption_mask, use_fine=False
            )

            # Reconstruction
            text_embeds = model.get_empty_text_embeds(batch_size)
            _, loss_rec_fine, loss_rec_coarse = model(
                text_embeds, frames_norm, vae_latents
            )

            loss = loss_cap_fine + 0.5 * loss_rec_fine

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        step_times.append(step_time)
        peak_mems.append(peak_mem)

        cap_ratio = loss_cap_coarse.item() / loss_cap_fine.item()
        rec_ratio = loss_rec_coarse.item() / loss_rec_fine.item()

        print(f"  Step {step+1}/{num_steps}: {step_time:.2f}s | "
              f"Peak: {peak_mem:.2f} GB | "
              f"cap_r={cap_ratio:.3f} rec_r={rec_ratio:.3f}")

    # Summary
    avg_time = np.mean(step_times[1:])  # Skip first step (warmup)
    max_mem = max(peak_mems)

    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Average step time: {avg_time:.2f}s")
    print(f"Peak memory: {max_mem:.2f} GB")
    print(f"GPU headroom: {24 - max_mem:.2f} GB")

    # Estimate 8-hour capacity
    steps_per_hour = 3600 / avg_time
    steps_8h = steps_per_hour * 7.5  # Leave 30 min buffer

    print(f"\nEstimated throughput:")
    print(f"  Steps per hour: {steps_per_hour:.0f}")
    print(f"  Steps in 7.5 hours: {steps_8h:.0f}")
    print(f"  Starting from step 8000: final step ~{8000 + steps_8h:.0f}")

    if max_mem > 22:
        print(f"\nWARNING: Memory usage too high ({max_mem:.1f} GB)!")
        print("  Recommend reducing batch_size to 2")
    elif max_mem < 18:
        print(f"\nNOTE: Memory headroom available ({24 - max_mem:.1f} GB)")
        print("  Could potentially increase batch_size")
    else:
        print(f"\nMemory usage is optimal ({max_mem:.1f} GB)")

    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    profile(batch_size=args.batch_size, num_steps=args.steps)
