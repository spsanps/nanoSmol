#!/usr/bin/env python3
"""
Scaled Captioning Training with WebVid-10M.

Trains foveated VLM for video captioning with:
- Checkpoint saving
- Attention visualization logging
- Caption quality evaluation
- Learning rate scheduling
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
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import re
import json
import matplotlib.pyplot as plt

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
    """Parse PT00H00M11S format to seconds."""
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
    """Download video with short timeout."""
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
    """Extract frames using ffmpeg."""
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
    """Apply ImageNet normalization."""
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


def denormalize_frames(frames: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return frames * std + mean


def webvid_generator(num_frames=8, frame_size=256, max_duration=15):
    """Generate (frames, caption) pairs from WebVid-10M."""
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

        except Exception:
            continue


def visualize_attention(model, frames, caption, tokenizer, save_path, step, device):
    """Create attention visualization for a sample."""
    model.eval()

    try:
        with torch.no_grad():
            frames_norm = normalize_frames(frames.unsqueeze(0).to(device))

            # Get attention weights (if model supports it)
            if hasattr(model.encoder, 'get_attention_weights'):
                attn_fine = model.encoder.get_attention_weights(frames_norm, use_fine=True)
                attn_coarse = model.encoder.get_attention_weights(frames_norm, use_fine=False)
            else:
                # Fallback: use forward pass outputs
                attn_fine = attn_coarse = None

            # Generate captions
            caption_fine = model.generate_caption(
                frames_norm, tokenizer, max_new_tokens=40, use_fine=True
            )[0]
            caption_coarse = model.generate_caption(
                frames_norm, tokenizer, max_new_tokens=40, use_fine=False
            )[0]

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Show frames in first row
        # Note: frames are already in [0,1] range from extract_frames, no denormalization needed
        for i, ax in enumerate(axes[0]):
            if i < frames.shape[0]:
                img = frames[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                ax.imshow(img)
                ax.set_title(f'Frame {i+1}')
            ax.axis('off')

        # Show attention or info in second row
        axes[1, 0].text(0.5, 0.5, f'GT:\n{caption[:100]}',
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Ground Truth')

        axes[1, 1].text(0.5, 0.5, f'Fine:\n{caption_fine[:100]}',
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Fine Query Caption')

        axes[1, 2].text(0.5, 0.5, f'Coarse:\n{caption_coarse[:100]}',
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Coarse Query Caption')

        axes[1, 3].text(0.5, 0.5, f'Step: {step}',
                       ha='center', va='center', fontsize=14)
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'gt': caption,
            'fine': caption_fine,
            'coarse': caption_coarse,
        }

    except Exception as e:
        print(f"Visualization error: {e}")
        return None
    finally:
        model.train()


def save_checkpoint(model, optimizer, scaler, step, metrics, save_dir):
    """Save training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
    }

    # Save latest
    torch.save(checkpoint, save_dir / 'latest.pt')

    # Save periodic checkpoint
    if step % 1000 == 0:
        torch.save(checkpoint, save_dir / f'step_{step:06d}.pt')

    print(f"Checkpoint saved at step {step}")


def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['step'], checkpoint.get('metrics', {})


def run_training(
    num_steps: int = 5000,
    batch_size: int = 2,
    grad_accum: int = 4,
    num_frames: int = 8,
    learning_rate: float = 3e-5,
    warmup_steps: int = 100,
    log_interval: int = 25,
    eval_interval: int = 200,
    save_interval: int = 500,
    viz_interval: int = 500,
    output_dir: str = "outputs/captioning_scaled",
    resume: str = None,
    use_wandb: bool = True,
):
    """Run scaled captioning training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    run_name = f"caption_scaled_{datetime.now().strftime('%m%d_%H%M')}"

    if use_wandb and HAS_WANDB:
        wandb.init(
            project="foveated-vlm-captioning",
            name=run_name,
            config={
                "num_steps": num_steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "num_frames": num_frames,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "dataset": "webvid-10M",
            }
        )

    print("Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda')

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_step = 0
    if resume and Path(resume).exists():
        print(f"Resuming from {resume}")
        start_step, _ = load_checkpoint(model, optimizer, scaler, resume)
        print(f"Resumed at step {start_step}")

    data_gen = webvid_generator(num_frames=num_frames)

    step = start_step
    accum_loss_fine = 0.0
    accum_loss_coarse = 0.0
    accum_count = 0
    samples_ok = 0
    samples_fail = 0
    all_ratios = []
    eval_samples = []

    print(f"\nStarting scaled captioning training...")
    print(f"  Total steps: {num_steps}")
    print(f"  Starting from: {start_step}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Warmup steps: {warmup_steps}")
    print("=" * 70)

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
                if samples_fail > 100 and samples_ok == 0:
                    print("Too many failures, aborting")
                    return
                continue

        frames = torch.stack(batch_frames).to(device)
        frames = normalize_frames(frames)

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
            loss_fine = model.forward_captioning(
                frames, caption_ids, caption_mask, use_fine=True
            )
            loss_coarse = model.forward_captioning(
                frames, caption_ids, caption_mask, use_fine=False
            )
            loss = loss_fine / grad_accum

        scaler.scale(loss).backward()

        accum_loss_fine += loss_fine.item()
        accum_loss_coarse += loss_coarse.item()
        accum_count += 1

        if accum_count >= grad_accum:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            avg_fine = accum_loss_fine / accum_count
            avg_coarse = accum_loss_coarse / accum_count
            ratio = avg_coarse / avg_fine if avg_fine > 0 else 1.0
            all_ratios.append(ratio)

            step += 1
            pbar.update(1)

            if step % log_interval == 0:
                recent_ratio = np.mean(all_ratios[-50:]) if all_ratios else 1.0
                success_rate = samples_ok / (samples_ok + samples_fail) * 100 if (samples_ok + samples_fail) > 0 else 0
                current_lr = scheduler.get_last_lr()[0]

                log_dict = {
                    "loss_fine": avg_fine,
                    "loss_coarse": avg_coarse,
                    "ratio": ratio,
                    "ratio_avg50": recent_ratio,
                    "success_rate": success_rate,
                    "learning_rate": current_lr,
                }

                if use_wandb and HAS_WANDB:
                    wandb.log(log_dict, step=step)

                status = "FINE_BETTER" if ratio > 1.02 else ("COARSE_BETTER" if ratio < 0.98 else "SAME")
                tqdm.write(
                    f"Step {step:5d} | fine={avg_fine:.3f} coarse={avg_coarse:.3f} | "
                    f"ratio={ratio:.4f} ({status}) | lr={current_lr:.2e}"
                )

            accum_loss_fine = 0.0
            accum_loss_coarse = 0.0
            accum_count = 0

            # Evaluation with caption generation
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        eval_frames, eval_caption = next(data_gen)
                        eval_frames_norm = eval_frames.unsqueeze(0).to(device)
                        eval_frames_norm = normalize_frames(eval_frames_norm)

                        caption_fine = model.generate_caption(
                            eval_frames_norm, tokenizer, max_new_tokens=40, use_fine=True
                        )[0]
                        caption_coarse = model.generate_caption(
                            eval_frames_norm, tokenizer, max_new_tokens=40, use_fine=False
                        )[0]

                        eval_samples.append({
                            'step': step,
                            'gt': eval_caption,
                            'fine': caption_fine,
                            'coarse': caption_coarse,
                        })

                        tqdm.write(f"\n--- Sample at step {step} ---")
                        tqdm.write(f"GT: {eval_caption}")
                        tqdm.write(f"Fine: {caption_fine}")
                        tqdm.write(f"Coarse: {caption_coarse}")
                        tqdm.write("")

                        if use_wandb and HAS_WANDB:
                            wandb.log({
                                "eval/gt_caption": eval_caption,
                                "eval/fine_caption": caption_fine,
                                "eval/coarse_caption": caption_coarse,
                            }, step=step)

                    except Exception as e:
                        tqdm.write(f"Eval error: {e}")
                model.train()

            # Visualization
            if step % viz_interval == 0:
                try:
                    eval_frames, eval_caption = next(data_gen)
                    viz_path = viz_dir / f"step_{step:06d}.png"
                    viz_result = visualize_attention(
                        model, eval_frames, eval_caption, tokenizer,
                        viz_path, step, device
                    )
                    if viz_result and use_wandb and HAS_WANDB:
                        wandb.log({
                            "visualization": wandb.Image(str(viz_path)),
                        }, step=step)
                except Exception as e:
                    tqdm.write(f"Viz error: {e}")

            # Save checkpoint
            if step % save_interval == 0:
                metrics = {
                    'all_ratios': all_ratios[-100:],
                    'avg_ratio': np.mean(all_ratios[-100:]) if all_ratios else 1.0,
                    'samples_ok': samples_ok,
                    'samples_fail': samples_fail,
                }
                save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints")

    pbar.close()

    # Final checkpoint
    metrics = {
        'all_ratios': all_ratios[-100:],
        'avg_ratio': np.mean(all_ratios[-100:]) if all_ratios else 1.0,
        'samples_ok': samples_ok,
        'samples_fail': samples_fail,
    }
    save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints")

    # Save evaluation samples
    with open(output_dir / "eval_samples.json", 'w') as f:
        json.dump(eval_samples, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    final_ratios = all_ratios[-100:] if len(all_ratios) >= 100 else all_ratios
    avg_ratio = np.mean(final_ratios)

    print(f"Final average ratio (last 100): {avg_ratio:.4f}")
    print(f"Total samples processed: {samples_ok}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Visualizations saved to: {viz_dir}")

    if avg_ratio > 1.05:
        print("RESULT: Fine queries consistently BETTER than coarse!")
    elif avg_ratio < 0.95:
        print("RESULT: Coarse queries better (unexpected)")
    else:
        print("RESULT: No significant difference")

    if use_wandb and HAS_WANDB:
        wandb.log({"final_ratio": avg_ratio})
        wandb.finish()

    return avg_ratio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs/captioning_scaled")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    run_training(
        num_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_frames=args.num_frames,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        output_dir=args.output_dir,
        resume=args.resume,
        use_wandb=not args.no_wandb,
    )
