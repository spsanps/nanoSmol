#!/usr/bin/env python3
"""
Captioning-Only Experiment: Test if foveated attention helps semantic tasks.

Hypothesis: Unlike reconstruction (global task), captioning requires
semantic understanding which might benefit from foveated attention.

Key metrics:
- loss_fine: Cross-entropy loss with dynamic queries
- loss_coarse: Cross-entropy loss with static query
- ratio: loss_coarse / loss_fine (want > 1.0)
"""

import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import random
import time

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def download_video(url: str, timeout: int = 30) -> bytes:
    """Download video from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            return response.content
    except:
        pass
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames from video bytes using ffmpeg."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(video_bytes)
            f.flush()

            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 4),
                    '-q:v', '2',
                    f'{tmpdir}/frame_%04d.jpg',
                    '-y', '-loglevel', 'error'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=60)
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
    """Apply ImageNet normalization."""
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


def extract_caption(conversations):
    """Extract caption from LLaVA-Video conversation format."""
    for conv in conversations:
        if conv['from'] == 'gpt':
            return conv['value']
    return None


def streaming_data_generator(num_frames=8, frame_size=256, max_caption_len=128):
    """Generate (frames, caption) pairs from LLaVA-Video streaming."""
    ds = load_dataset(
        'lmms-lab/LLaVA-Video-178K',
        '0_30_s_academic_v0_1',
        split='caption',
        streaming=True
    )

    for sample in ds:
        try:
            video_url = sample['video']
            caption = extract_caption(sample['conversations'])

            if not video_url or not caption:
                continue

            # Download and extract frames
            video_bytes = download_video(video_url)
            if video_bytes is None:
                continue

            frames = extract_frames(video_bytes, num_frames, frame_size)
            if frames is None:
                continue

            # Truncate caption if needed
            if len(caption) > max_caption_len * 4:  # Rough char estimate
                caption = caption[:max_caption_len * 4]

            yield frames, caption

        except Exception as e:
            continue


def run_captioning_experiment(
    num_steps: int = 2000,
    batch_size: int = 2,
    grad_accum: int = 4,
    num_frames: int = 8,
    learning_rate: float = 3e-5,
    log_interval: int = 50,
    eval_interval: int = 200,
    use_wandb: bool = True,
):
    """Run captioning-only training experiment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 70)

    # Initialize wandb
    if use_wandb and HAS_WANDB:
        wandb.init(
            project="foveated-vlm-captioning",
            name=f"caption_exp_{datetime.now().strftime('%H%M')}",
            config={
                "num_steps": num_steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "num_frames": num_frames,
                "learning_rate": learning_rate,
                "task": "captioning_only",
            }
        )

    # Load model
    print("Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,  # Freeze DINO as established
    ).to(device)
    model.train()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Data generator
    data_gen = streaming_data_generator(num_frames=num_frames)

    # Tracking
    step = 0
    accum_loss_fine = 0.0
    accum_loss_coarse = 0.0
    accum_count = 0
    samples_processed = 0

    # For ratio tracking
    all_ratios = []

    print(f"\nStarting captioning experiment...")
    print(f"  Steps: {num_steps}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print("=" * 70)

    pbar = tqdm(total=num_steps, desc="Training")

    optimizer.zero_grad()

    while step < num_steps:
        # Collect batch
        batch_frames = []
        batch_captions = []

        while len(batch_frames) < batch_size:
            try:
                frames, caption = next(data_gen)
                batch_frames.append(frames)
                batch_captions.append(caption)
            except StopIteration:
                data_gen = streaming_data_generator(num_frames=num_frames)
                continue

        # Stack frames
        frames = torch.stack(batch_frames).to(device)  # [B, T, 3, H, W]
        frames = normalize_frames(frames)

        # Tokenize captions
        tokens = tokenizer(
            batch_captions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        caption_ids = tokens['input_ids'].to(device)
        caption_mask = tokens['attention_mask'].to(device)

        # Forward pass for BOTH fine and coarse
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Fine pass (dynamic queries)
            loss_fine = model.forward_captioning(
                frames, caption_ids, caption_mask, use_fine=True
            )

            # Coarse pass (static query)
            loss_coarse = model.forward_captioning(
                frames, caption_ids, caption_mask, use_fine=False
            )

            # Train on fine loss (want dynamic queries to be better)
            loss = loss_fine / grad_accum

        # Backward
        scaler.scale(loss).backward()

        # Track losses
        accum_loss_fine += loss_fine.item()
        accum_loss_coarse += loss_coarse.item()
        accum_count += 1
        samples_processed += batch_size

        # Optimizer step
        if accum_count >= grad_accum:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Calculate metrics
            avg_fine = accum_loss_fine / accum_count
            avg_coarse = accum_loss_coarse / accum_count
            ratio = avg_coarse / avg_fine if avg_fine > 0 else 1.0
            all_ratios.append(ratio)

            step += 1
            pbar.update(1)

            # Log
            if step % log_interval == 0:
                recent_ratio = np.mean(all_ratios[-20:]) if all_ratios else 1.0

                log_dict = {
                    "loss_fine": avg_fine,
                    "loss_coarse": avg_coarse,
                    "ratio": ratio,
                    "ratio_avg20": recent_ratio,
                    "samples": samples_processed,
                }

                if use_wandb and HAS_WANDB:
                    wandb.log(log_dict, step=step)

                status = "BETTER" if ratio > 1.01 else ("WORSE" if ratio < 0.99 else "SAME")
                tqdm.write(
                    f"Step {step:5d} | fine={avg_fine:.4f} coarse={avg_coarse:.4f} | "
                    f"ratio={ratio:.4f} ({status}) | avg20={recent_ratio:.4f}"
                )

            # Reset accumulators
            accum_loss_fine = 0.0
            accum_loss_coarse = 0.0
            accum_count = 0

            # Evaluation: generate sample captions
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        # Get a fresh sample
                        eval_frames, eval_caption = next(data_gen)
                        eval_frames = eval_frames.unsqueeze(0).to(device)
                        eval_frames = normalize_frames(eval_frames)

                        # Generate with fine and coarse
                        caption_fine = model.generate_caption(
                            eval_frames, tokenizer, max_new_tokens=50, use_fine=True
                        )[0]
                        caption_coarse = model.generate_caption(
                            eval_frames, tokenizer, max_new_tokens=50, use_fine=False
                        )[0]

                        tqdm.write(f"\n--- Sample at step {step} ---")
                        tqdm.write(f"Ground truth: {eval_caption[:200]}...")
                        tqdm.write(f"Fine caption: {caption_fine[:200]}")
                        tqdm.write(f"Coarse caption: {caption_coarse[:200]}")
                        tqdm.write("")

                        if use_wandb and HAS_WANDB:
                            wandb.log({
                                "sample_gt": eval_caption[:500],
                                "sample_fine": caption_fine,
                                "sample_coarse": caption_coarse,
                            }, step=step)

                    except Exception as e:
                        tqdm.write(f"Eval error: {e}")

                model.train()

    pbar.close()

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    final_ratios = all_ratios[-100:] if len(all_ratios) >= 100 else all_ratios
    avg_ratio = np.mean(final_ratios)

    print(f"Final average ratio (last 100): {avg_ratio:.4f}")

    if avg_ratio > 1.05:
        print("RESULT: Fine queries BETTER than coarse for captioning!")
    elif avg_ratio < 0.95:
        print("RESULT: Coarse queries BETTER than fine (unexpected)")
    else:
        print("RESULT: No significant difference (ratio ~ 1.0)")

    if use_wandb and HAS_WANDB:
        wandb.log({"final_ratio": avg_ratio})
        wandb.finish()

    return avg_ratio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    run_captioning_experiment(
        num_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_frames=args.num_frames,
        learning_rate=args.lr,
        use_wandb=not args.no_wandb,
    )
