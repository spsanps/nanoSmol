#!/usr/bin/env python3
"""
Profile training bottlenecks to understand scaling limitations.

Measures time spent in:
1. Network download
2. ffmpeg frame extraction
3. VAE latent encoding (GPU)
4. Model forward pass (GPU)
5. Backward pass (GPU)
"""

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
import json

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


def timed_download(url: str, timeout: int = 20) -> tuple:
    """Download video and return (bytes, time_taken)."""
    start = time.time()
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            content = b''
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 50 * 1024 * 1024:
                    break
            return content, time.time() - start
    except:
        pass
    return None, time.time() - start


def timed_extract_frames(video_bytes: bytes, num_frames: int, size: int = 256) -> tuple:
    """Extract frames and return (tensor, time_taken)."""
    start = time.time()
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
                    return None, time.time() - start
                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    return None, time.time() - start
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)
                return torch.stack(frames), time.time() - start
    except:
        return None, time.time() - start


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


@torch.no_grad()
def timed_vae_encode(vae, frames, device) -> tuple:
    """Encode with VAE and return (latents, time_taken)."""
    torch.cuda.synchronize()
    start = time.time()

    B, T, C, H, W = frames.shape
    frames_vae = frames * 2 - 1
    frames_flat = frames_vae.reshape(B * T, C, H, W).to(device)
    latents_flat = vae.encode(frames_flat).latent_dist.sample()
    latents_flat = latents_flat * vae.config.scaling_factor
    latents = latents_flat.reshape(B, T, 4, H // 8, W // 8)

    torch.cuda.synchronize()
    return latents, time.time() - start


def profile_bottlenecks(num_samples=20, batch_size=8, num_frames=8):
    """Profile each stage of the training pipeline."""

    device = torch.device("cuda")

    print("=" * 70)
    print("BOTTLENECK PROFILING")
    print("=" * 70)
    print(f"Target: {num_samples} successful samples, batch_size={batch_size}")
    print()

    # Load models
    print("[1/3] Loading models...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
    model.train()

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = GradScaler('cuda')

    model_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Models loaded: {model_mem:.2f} GB VRAM")

    # Profiling data collection
    print("\n[2/3] Collecting samples and timing each stage...")

    download_times = []
    extract_times = []
    vae_times = []
    forward_times = []
    backward_times = []

    samples = []
    failed = 0

    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    for sample in ds:
        if len(samples) >= num_samples:
            break

        duration = parse_duration(sample.get('duration', ''))
        if duration == 0 or duration > 15:
            continue

        url = sample.get('contentUrl')
        caption = sample.get('name', '')
        if not url or not caption or len(caption) < 5:
            continue

        # Time download
        video_bytes, dl_time = timed_download(url)
        if video_bytes is None:
            failed += 1
            continue
        download_times.append(dl_time)

        # Time extraction
        frames, extract_time = timed_extract_frames(video_bytes, num_frames)
        if frames is None:
            failed += 1
            continue
        extract_times.append(extract_time)

        samples.append((frames, caption))
        print(f"  Sample {len(samples)}/{num_samples}: dl={dl_time:.2f}s, extract={extract_time:.2f}s", end='\r')

    print(f"\n  Collected {len(samples)} samples ({failed} failed)")

    if len(samples) < batch_size:
        print("ERROR: Not enough samples!")
        return

    # Time GPU operations
    print("\n[3/3] Profiling GPU operations...")

    num_batches = len(samples) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch_frames = [samples[start_idx + i][0] for i in range(batch_size)]
        batch_captions = [samples[start_idx + i][1] for i in range(batch_size)]

        frames_raw = torch.stack(batch_frames)

        # Time VAE encoding
        vae_latents, vae_time = timed_vae_encode(vae, frames_raw, device)
        vae_times.append(vae_time)

        # Prepare for model
        frames_norm = normalize_frames(frames_raw.to(device))
        tokens = tokenizer(batch_captions, padding=True, truncation=True, max_length=64, return_tensors='pt')
        caption_ids = tokens['input_ids'].to(device)
        caption_mask = tokens['attention_mask'].to(device)

        # Time forward pass
        torch.cuda.synchronize()
        fwd_start = time.time()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_cap_fine = model.forward_captioning(frames_norm, caption_ids, caption_mask, use_fine=True)
            loss_cap_coarse = model.forward_captioning(frames_norm, caption_ids, caption_mask, use_fine=False)
            text_embeds = model.get_empty_text_embeds(batch_size)
            _, loss_rec_fine, loss_rec_coarse = model(text_embeds, frames_norm, vae_latents)
            loss = loss_cap_fine + 0.5 * loss_rec_fine

        torch.cuda.synchronize()
        forward_times.append(time.time() - fwd_start)

        # Time backward pass
        torch.cuda.synchronize()
        bwd_start = time.time()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        backward_times.append(time.time() - bwd_start)

        print(f"  Batch {batch_idx+1}/{num_batches}: vae={vae_time:.2f}s, fwd={forward_times[-1]:.2f}s, bwd={backward_times[-1]:.2f}s")

    # Summary
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS RESULTS")
    print("=" * 70)

    avg_download = np.mean(download_times)
    avg_extract = np.mean(extract_times)
    avg_vae = np.mean(vae_times)
    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)

    # Per-sample times (for batch of 8)
    avg_download_per_sample = avg_download
    avg_extract_per_sample = avg_extract
    avg_vae_per_sample = avg_vae / batch_size
    avg_forward_per_sample = avg_forward / batch_size
    avg_backward_per_sample = avg_backward / batch_size

    print(f"\nPer-SAMPLE timing (batch_size={batch_size}):")
    print(f"  Network download:    {avg_download_per_sample:.3f}s")
    print(f"  ffmpeg extraction:   {avg_extract_per_sample:.3f}s")
    print(f"  VAE encoding:        {avg_vae_per_sample:.3f}s")
    print(f"  Model forward:       {avg_forward_per_sample:.3f}s")
    print(f"  Model backward:      {avg_backward_per_sample:.3f}s")

    total_per_sample = avg_download_per_sample + avg_extract_per_sample + avg_vae_per_sample + avg_forward_per_sample + avg_backward_per_sample

    print(f"\n  TOTAL per sample:    {total_per_sample:.3f}s")

    print(f"\nPer-BATCH timing (batch_size={batch_size}):")
    print(f"  Data loading (serial): {(avg_download + avg_extract) * batch_size:.2f}s")
    print(f"  VAE encoding (GPU):    {avg_vae:.2f}s")
    print(f"  Model forward (GPU):   {avg_forward:.2f}s")
    print(f"  Model backward (GPU):  {avg_backward:.2f}s")

    gpu_time = avg_vae + avg_forward + avg_backward
    data_time = (avg_download + avg_extract) * batch_size

    print(f"\n  GPU time per batch:    {gpu_time:.2f}s")
    print(f"  Data time per batch:   {data_time:.2f}s (SERIAL!)")

    print(f"\nBottleneck analysis:")
    if data_time > gpu_time:
        ratio = data_time / gpu_time
        print(f"  DATA LOADING IS {ratio:.1f}x SLOWER THAN GPU!")
        print(f"  GPU utilization estimate: {gpu_time / (gpu_time + data_time) * 100:.0f}%")
        print(f"\n  To saturate GPU, need {int(np.ceil(ratio))}+ parallel data loaders")
    else:
        print(f"  GPU is the bottleneck (good!)")

    print(f"\nPeak GPU memory: {peak_mem:.2f} GB / 24 GB")
    print(f"  Headroom: {24 - peak_mem:.2f} GB")

    # Scaling projections
    print("\n" + "=" * 70)
    print("SCALING PROJECTIONS")
    print("=" * 70)

    # Current setup (1 thread, serial)
    current_samples_per_hour = 3600 / total_per_sample
    current_steps_per_hour = current_samples_per_hour / batch_size  # No grad accum in this calc

    print(f"\nCurrent (1 thread, serial data loading):")
    print(f"  Samples/hour: {current_samples_per_hour:.0f}")
    print(f"  Steps/hour (BS={batch_size}): {current_steps_per_hour:.0f}")

    # Optimal (parallel data loading)
    optimal_time = gpu_time / batch_size + 0.1  # Small overhead
    optimal_samples_per_hour = 3600 / optimal_time
    optimal_steps_per_hour = optimal_samples_per_hour / batch_size

    print(f"\nOptimal (fully parallel data loading):")
    print(f"  Samples/hour: {optimal_samples_per_hour:.0f}")
    print(f"  Steps/hour (BS={batch_size}): {optimal_steps_per_hour:.0f}")
    print(f"  Speedup: {optimal_samples_per_hour / current_samples_per_hour:.1f}x")

    # With local data
    local_time = avg_extract_per_sample + avg_vae_per_sample + avg_forward_per_sample + avg_backward_per_sample
    local_samples_per_hour = 3600 / local_time

    print(f"\nWith local data (no network):")
    print(f"  Per-sample time: {local_time:.3f}s (vs {total_per_sample:.3f}s)")
    print(f"  Samples/hour: {local_samples_per_hour:.0f}")
    print(f"  Speedup: {local_samples_per_hour / current_samples_per_hour:.1f}x")

    # Save results
    results = {
        "per_sample": {
            "download": avg_download_per_sample,
            "extract": avg_extract_per_sample,
            "vae": avg_vae_per_sample,
            "forward": avg_forward_per_sample,
            "backward": avg_backward_per_sample,
            "total": total_per_sample,
        },
        "per_batch": {
            "data_loading": data_time,
            "gpu_total": gpu_time,
        },
        "bottleneck": "data_loading" if data_time > gpu_time else "gpu",
        "gpu_utilization_pct": gpu_time / (gpu_time + data_time) * 100,
        "peak_memory_gb": peak_mem,
        "scaling": {
            "current_samples_per_hour": current_samples_per_hour,
            "optimal_samples_per_hour": optimal_samples_per_hour,
            "local_samples_per_hour": local_samples_per_hour,
        }
    }

    output_path = Path("outputs/bottleneck_profile.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    profile_bottlenecks(args.samples, args.batch_size, args.num_frames)
