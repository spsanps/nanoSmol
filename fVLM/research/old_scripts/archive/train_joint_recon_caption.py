#!/usr/bin/env python3
"""
Joint Reconstruction + Captioning Training.

THESIS:
The captioning task teaches the model WHERE to look (semantically relevant regions).
This learned attention pattern should ALSO help reconstruction.

Combined loss: loss = loss_caption + lambda_recon * loss_reconstruction

We track BOTH:
- Caption ratio: loss_caption_coarse / loss_caption_fine (should be > 1.0)
- Reconstruction ratio: loss_recon_coarse / loss_recon_fine (hypothesis: should also become > 1.0)
"""

import sys
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

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


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
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


def webvid_generator(num_frames=8, frame_size=256, max_duration=15):
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
    for sample in ds:
        try:
            duration = parse_duration(sample.get('duration', ''))
            if duration == 0 or duration > max_duration:
                continue
            url = sample.get('contentUrl')
            caption = sample.get('name', '')
            if not url or not caption or len(caption) < 5:
                continue
            video_bytes = download_video(url)
            if video_bytes is None:
                continue
            frames = extract_frames(video_bytes, num_frames, frame_size)
            if frames is None:
                continue
            yield frames, caption
        except:
            continue


@torch.no_grad()
def compute_vae_latents(vae, frames, device):
    """Compute VAE latents for frames. frames: [B, T, C, H, W] in [0,1] range."""
    B, T, C, H, W = frames.shape
    # VAE expects [-1, 1] range
    frames_vae = frames * 2 - 1
    frames_flat = frames_vae.reshape(B * T, C, H, W).to(device)

    latents_flat = vae.encode(frames_flat).latent_dist.sample()
    latents_flat = latents_flat * vae.config.scaling_factor

    # Reshape back to [B, T, 4, H/8, W/8]
    latents = latents_flat.reshape(B, T, 4, H // 8, W // 8)
    return latents


def save_checkpoint(model, optimizer, scaler, step, metrics, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_dir / 'latest.pt')
    if step % 1000 == 0:
        torch.save(checkpoint, save_dir / f'step_{step:06d}.pt')
    print(f"Checkpoint saved at step {step}", flush=True)


def run_training(
    num_steps: int = 10000,
    batch_size: int = 2,
    grad_accum: int = 4,
    num_frames: int = 8,
    learning_rate: float = 3e-5,
    lambda_recon: float = 0.5,
    warmup_steps: int = 100,
    log_interval: int = 25,
    eval_interval: int = 200,
    save_interval: int = 500,
    output_dir: str = "outputs/joint_recon_caption",
    resume: str = None,
    use_wandb: bool = True,
):
    """Run joint reconstruction + captioning training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JOINT RECONSTRUCTION + CAPTIONING TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Steps: {num_steps}")
    print(f"Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"Lambda recon: {lambda_recon}")
    print("=" * 70)

    # Initialize wandb
    if use_wandb and HAS_WANDB:
        wandb.init(
            project="foveated-vlm-joint",
            name=f"joint_{datetime.now().strftime('%m%d_%H%M')}",
            config={
                "num_steps": num_steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "num_frames": num_frames,
                "learning_rate": learning_rate,
                "lambda_recon": lambda_recon,
                "warmup_steps": warmup_steps,
            }
        )

    # Load models
    print("\nLoading model...", flush=True)
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)
    model.train()

    print("Loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda')

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_step = 0
    if resume and Path(resume).exists():
        print(f"Resuming from {resume}", flush=True)
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed at step {start_step}", flush=True)

    data_gen = webvid_generator(num_frames=num_frames)

    # Tracking
    step = start_step
    accum_caption_fine = 0.0
    accum_caption_coarse = 0.0
    accum_recon_fine = 0.0
    accum_recon_coarse = 0.0
    accum_count = 0
    samples_ok = 0
    samples_fail = 0
    caption_ratios = []
    recon_ratios = []

    print(f"\nStarting training...", flush=True)
    pbar = tqdm(total=num_steps, initial=start_step, desc="Training")
    optimizer.zero_grad()

    while step < num_steps:
        batch_frames = []
        batch_captions = []

        while len(batch_frames) < batch_size:
            try:
                frames, caption = next(data_gen)
                batch_frames.append(frames)
                batch_captions.append(caption)
                samples_ok += 1
            except StopIteration:
                data_gen = webvid_generator(num_frames=num_frames)
                samples_fail += 1
                continue

        # Prepare batch
        frames_raw = torch.stack(batch_frames)  # [B, T, C, H, W] in [0,1]
        frames_norm = normalize_frames(frames_raw.to(device))  # ImageNet normalized

        # Compute VAE latents for reconstruction
        vae_latents = compute_vae_latents(vae, frames_raw, device)  # [B, T, 4, 32, 32]

        # Tokenize captions
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
            # === CAPTIONING LOSSES ===
            loss_caption_fine = model.forward_captioning(
                frames_norm, caption_ids, caption_mask, use_fine=True
            )
            loss_caption_coarse = model.forward_captioning(
                frames_norm, caption_ids, caption_mask, use_fine=False
            )

            # === RECONSTRUCTION LOSSES ===
            # Get text embeddings (empty for reconstruction)
            text_embeds = model.get_empty_text_embeds(frames_norm.shape[0])

            # Forward pass returns (loss, loss_fine, loss_coarse)
            _, loss_recon_fine, loss_recon_coarse = model(
                text_embeds, frames_norm, vae_latents
            )

            # === COMBINED LOSS ===
            # Train on fine losses, but track coarse for comparison
            loss = (loss_caption_fine + lambda_recon * loss_recon_fine) / grad_accum

        scaler.scale(loss).backward()

        # Accumulate metrics
        accum_caption_fine += loss_caption_fine.item()
        accum_caption_coarse += loss_caption_coarse.item()
        accum_recon_fine += loss_recon_fine.item()
        accum_recon_coarse += loss_recon_coarse.item()
        accum_count += 1

        if accum_count >= grad_accum:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # Compute averages
            avg_cap_fine = accum_caption_fine / accum_count
            avg_cap_coarse = accum_caption_coarse / accum_count
            avg_rec_fine = accum_recon_fine / accum_count
            avg_rec_coarse = accum_recon_coarse / accum_count

            # Compute ratios
            cap_ratio = avg_cap_coarse / avg_cap_fine if avg_cap_fine > 0 else 1.0
            rec_ratio = avg_rec_coarse / avg_rec_fine if avg_rec_fine > 0 else 1.0

            caption_ratios.append(cap_ratio)
            recon_ratios.append(rec_ratio)

            step += 1
            pbar.update(1)

            if step % log_interval == 0:
                recent_cap_ratio = np.mean(caption_ratios[-50:]) if caption_ratios else 1.0
                recent_rec_ratio = np.mean(recon_ratios[-50:]) if recon_ratios else 1.0
                current_lr = scheduler.get_last_lr()[0]

                log_dict = {
                    "caption/loss_fine": avg_cap_fine,
                    "caption/loss_coarse": avg_cap_coarse,
                    "caption/ratio": cap_ratio,
                    "caption/ratio_avg50": recent_cap_ratio,
                    "recon/loss_fine": avg_rec_fine,
                    "recon/loss_coarse": avg_rec_coarse,
                    "recon/ratio": rec_ratio,
                    "recon/ratio_avg50": recent_rec_ratio,
                    "learning_rate": current_lr,
                }

                if use_wandb and HAS_WANDB:
                    wandb.log(log_dict, step=step)

                cap_status = "CAP_FINE_BETTER" if cap_ratio > 1.02 else "CAP_SAME"
                rec_status = "REC_FINE_BETTER" if rec_ratio > 1.02 else "REC_SAME"

                tqdm.write(
                    f"Step {step:5d} | "
                    f"cap: fine={avg_cap_fine:.3f} coarse={avg_cap_coarse:.3f} ratio={cap_ratio:.4f} ({cap_status}) | "
                    f"rec: fine={avg_rec_fine:.3f} coarse={avg_rec_coarse:.3f} ratio={rec_ratio:.4f} ({rec_status})"
                )

            # Reset accumulators
            accum_caption_fine = 0.0
            accum_caption_coarse = 0.0
            accum_recon_fine = 0.0
            accum_recon_coarse = 0.0
            accum_count = 0

            # Evaluation
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        eval_frames, eval_caption = next(data_gen)
                        eval_frames_norm = normalize_frames(eval_frames.unsqueeze(0).to(device))

                        caption_fine = model.generate_caption(
                            eval_frames_norm, tokenizer, max_new_tokens=40, use_fine=True
                        )[0]
                        caption_coarse = model.generate_caption(
                            eval_frames_norm, tokenizer, max_new_tokens=40, use_fine=False
                        )[0]

                        tqdm.write(f"\n--- Sample at step {step} ---")
                        tqdm.write(f"GT: {eval_caption}")
                        tqdm.write(f"Fine: {caption_fine}")
                        tqdm.write(f"Coarse: {caption_coarse}")
                        tqdm.write("")

                    except Exception as e:
                        tqdm.write(f"Eval error: {e}")
                model.train()

            # Save checkpoint
            if step % save_interval == 0:
                metrics = {
                    'caption_ratios': caption_ratios[-100:],
                    'recon_ratios': recon_ratios[-100:],
                    'avg_caption_ratio': np.mean(caption_ratios[-100:]) if caption_ratios else 1.0,
                    'avg_recon_ratio': np.mean(recon_ratios[-100:]) if recon_ratios else 1.0,
                }
                save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints")

    pbar.close()

    # Final checkpoint
    metrics = {
        'caption_ratios': caption_ratios[-100:],
        'recon_ratios': recon_ratios[-100:],
        'avg_caption_ratio': np.mean(caption_ratios[-100:]) if caption_ratios else 1.0,
        'avg_recon_ratio': np.mean(recon_ratios[-100:]) if recon_ratios else 1.0,
    }
    save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    final_cap_ratio = np.mean(caption_ratios[-100:]) if caption_ratios else 1.0
    final_rec_ratio = np.mean(recon_ratios[-100:]) if recon_ratios else 1.0

    print(f"Final Caption Ratio (last 100): {final_cap_ratio:.4f}")
    print(f"Final Reconstruction Ratio (last 100): {final_rec_ratio:.4f}")

    if final_cap_ratio > 1.05:
        print("CAPTION: Fine queries BETTER than coarse (expected)")
    else:
        print("CAPTION: No significant difference")

    if final_rec_ratio > 1.02:
        print("RECONSTRUCTION: Fine queries BETTER than coarse (THESIS VALIDATED!)")
    else:
        print("RECONSTRUCTION: No improvement (thesis not validated)")

    if use_wandb and HAS_WANDB:
        wandb.log({
            "final_caption_ratio": final_cap_ratio,
            "final_recon_ratio": final_rec_ratio,
        })
        wandb.finish()

    return final_cap_ratio, final_rec_ratio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs/joint_recon_caption")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    run_training(
        num_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_frames=args.num_frames,
        learning_rate=args.lr,
        lambda_recon=args.lambda_recon,
        warmup_steps=args.warmup,
        output_dir=args.output_dir,
        resume=args.resume,
        use_wandb=not args.no_wandb,
    )
