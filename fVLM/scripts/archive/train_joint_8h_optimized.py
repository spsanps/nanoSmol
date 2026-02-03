#!/usr/bin/env python3
"""
8-Hour Optimized Joint Reconstruction + Captioning Training.

OPTIMIZATIONS for maximum GPU utilization:
1. Reduced logging/eval/checkpoint frequency
2. Larger batch size (3 vs 2)
3. Time-based stopping with buffer for final checkpoint
4. Async data prefetching
5. No visualizations during training (post-analysis only)
6. Gradient checkpointing enabled

EXPERIMENT: Joint training teaches reconstruction through semantic understanding.
Previous result (8K steps): Both ratios 1.07-1.33
Goal: Scale to 25K+ steps and observe if ratios continue improving.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
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
import threading
from queue import Queue, Empty
from collections import deque

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

# ============================================================================
# CONFIGURATION - Optimized for 8-hour run
# ============================================================================

CONFIG = {
    # Training - optimized for 8 hours
    "max_hours": 7.5,  # Stop 30 min early for final checkpoint/cleanup
    "target_steps": 50000,  # High target - will hit time limit first
    "batch_size": 8,  # Profiled: 15.7 GB peak, ~6 samples/s
    "grad_accum": 3,  # Effective batch = 24
    "num_frames": 8,
    "learning_rate": 3e-5,
    "lambda_recon": 0.5,  # Joint loss: caption + 0.5 * reconstruction
    "warmup_steps": 100,

    # Logging (reduced frequency for speed)
    "log_interval": 100,
    "eval_interval": 1000,
    "save_interval": 2000,

    # Data loading - larger prefetch to avoid CPU bottleneck
    "prefetch_size": 24,  # Large buffer to handle network variability
    "download_timeout": 20,  # Seconds
    "max_video_duration": 15,  # Seconds
    "num_prefetch_threads": 2,  # Multiple threads for data loading

    # Checkpointing
    "resume_checkpoint": "outputs/joint_recon_caption/checkpoints/step_008000.pt",
    "output_dir": "outputs/joint_8h_scaled",
}


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


def download_video(url: str, timeout: int = 20) -> bytes:
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


class AsyncDataLoader:
    """Async data loader with multi-threaded prefetching for maximum throughput."""

    def __init__(self, num_frames=8, prefetch_size=24, max_duration=15, num_threads=2):
        self.num_frames = num_frames
        self.prefetch_size = prefetch_size
        self.max_duration = max_duration
        self.num_threads = num_threads
        self.queue = Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.stats = {"ok": 0, "fail": 0, "skipped": 0}
        self.lock = threading.Lock()

        # Start multiple prefetch threads
        self.threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=self._prefetch_loop, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)
        print(f"  Started {num_threads} data loading threads")

    def _prefetch_loop(self, thread_id):
        """Prefetch loop - each thread streams from a different position in dataset."""
        # Use thread_id to skip different number of samples for diversity
        ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

        # Skip some samples based on thread_id to avoid overlap
        skip_count = thread_id * 10000
        skipped = 0

        for sample in ds:
            if self.stop_event.is_set():
                break

            # Skip initial samples for thread diversity
            if skipped < skip_count:
                skipped += 1
                continue

            try:
                duration = parse_duration(sample.get('duration', ''))
                if duration == 0 or duration > self.max_duration:
                    with self.lock:
                        self.stats["skipped"] += 1
                    continue

                url = sample.get('contentUrl')
                caption = sample.get('name', '')
                if not url or not caption or len(caption) < 5:
                    with self.lock:
                        self.stats["skipped"] += 1
                    continue

                video_bytes = download_video(url)
                if video_bytes is None:
                    with self.lock:
                        self.stats["fail"] += 1
                    continue

                frames = extract_frames(video_bytes, self.num_frames)
                if frames is None:
                    with self.lock:
                        self.stats["fail"] += 1
                    continue

                # Put in queue (blocks if full)
                self.queue.put((frames, caption), timeout=60)
                with self.lock:
                    self.stats["ok"] += 1

            except Exception as e:
                with self.lock:
                    self.stats["fail"] += 1
                continue

    def get_batch(self, batch_size):
        """Get a batch of samples. Blocks until batch is ready."""
        frames_list = []
        captions_list = []
        while len(frames_list) < batch_size:
            try:
                frames, caption = self.queue.get(timeout=120)  # Longer timeout
                frames_list.append(frames)
                captions_list.append(caption)
            except Empty:
                # Queue empty - data loading is bottleneck
                print(f"WARNING: Data queue empty, waiting... (queue size: {self.queue.qsize()})")
                time.sleep(1)
        return torch.stack(frames_list), captions_list

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)

    def get_stats(self):
        with self.lock:
            return dict(self.stats)


@torch.no_grad()
def compute_vae_latents(vae, frames, device):
    B, T, C, H, W = frames.shape
    frames_vae = frames * 2 - 1
    frames_flat = frames_vae.reshape(B * T, C, H, W).to(device)
    latents_flat = vae.encode(frames_flat).latent_dist.sample()
    latents_flat = latents_flat * vae.config.scaling_factor
    latents = latents_flat.reshape(B, T, 4, H // 8, W // 8)
    return latents


def save_checkpoint(model, optimizer, scaler, step, metrics, save_dir, force_save=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, save_dir / 'latest.pt')
    if step % 2000 == 0 or force_save:
        torch.save(checkpoint, save_dir / f'step_{step:06d}.pt')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved: step {step}", flush=True)


def run_training():
    """Run optimized 8-hour joint training."""

    start_time = time.time()
    max_duration = CONFIG["max_hours"] * 3600  # Convert to seconds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("8-HOUR OPTIMIZED JOINT TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max duration: {CONFIG['max_hours']} hours")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Batch: {CONFIG['batch_size']} x {CONFIG['grad_accum']} = {CONFIG['batch_size'] * CONFIG['grad_accum']}")
    print(f"Target steps: {CONFIG['target_steps']}")
    print(f"Resume from: {CONFIG['resume_checkpoint']}")
    print("=" * 80)

    # Initialize wandb
    if HAS_WANDB:
        wandb.init(
            project="foveated-vlm-joint",
            name=f"joint_8h_{datetime.now().strftime('%m%d_%H%M')}",
            config=CONFIG,
        )

    # Load model
    print("\n[1/4] Loading model...", flush=True)
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    model.train()

    # Load VAE
    print("[2/4] Loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = GradScaler('cuda')

    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / CONFIG["warmup_steps"]
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_step = 0
    resume_path = Path(CONFIG["resume_checkpoint"])
    if resume_path.exists():
        print(f"[3/4] Resuming from {resume_path}...", flush=True)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Don't load optimizer state - fresh optimizer for this run
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']
        print(f"  Resumed at step {start_step}")
    else:
        print(f"[3/4] No checkpoint found at {resume_path}, starting fresh")

    # Start async data loader
    print("[4/4] Starting async data loader...", flush=True)
    data_loader = AsyncDataLoader(
        num_frames=CONFIG["num_frames"],
        prefetch_size=CONFIG["prefetch_size"],
        max_duration=CONFIG["max_video_duration"],
        num_threads=CONFIG.get("num_prefetch_threads", 2),
    )
    print(f"  Waiting for prefetch buffer to fill (target: {CONFIG['prefetch_size']})...")
    time.sleep(10)  # Wait for prefetch buffer to fill
    print(f"  Queue size: {data_loader.queue.qsize()}")

    # Tracking
    step = start_step
    accum_cap_fine = 0.0
    accum_cap_coarse = 0.0
    accum_rec_fine = 0.0
    accum_rec_coarse = 0.0
    accum_count = 0

    # Rolling metrics (for averaging)
    cap_ratios = deque(maxlen=100)
    rec_ratios = deque(maxlen=100)
    step_times = deque(maxlen=50)

    print(f"\n{'='*80}")
    print("TRAINING STARTED")
    print(f"{'='*80}\n", flush=True)

    pbar = tqdm(total=CONFIG["target_steps"], initial=start_step, desc="Training")
    optimizer.zero_grad()

    last_step_time = time.time()

    try:
        while step < CONFIG["target_steps"]:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                print(f"\n[TIME LIMIT] {CONFIG['max_hours']} hours reached, stopping...")
                break

            # Get batch
            frames_raw, captions = data_loader.get_batch(CONFIG["batch_size"])
            frames_norm = normalize_frames(frames_raw.to(device))

            # Compute VAE latents
            vae_latents = compute_vae_latents(vae, frames_raw, device)

            # Tokenize captions
            tokens = tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            caption_ids = tokens['input_ids'].to(device)
            caption_mask = tokens['attention_mask'].to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Captioning losses
                loss_cap_fine = model.forward_captioning(
                    frames_norm, caption_ids, caption_mask, use_fine=True
                )
                loss_cap_coarse = model.forward_captioning(
                    frames_norm, caption_ids, caption_mask, use_fine=False
                )

                # Reconstruction losses
                text_embeds = model.get_empty_text_embeds(frames_norm.shape[0])
                _, loss_rec_fine, loss_rec_coarse = model(
                    text_embeds, frames_norm, vae_latents
                )

                # Combined loss (train on fine only)
                loss = (loss_cap_fine + CONFIG["lambda_recon"] * loss_rec_fine) / CONFIG["grad_accum"]

            scaler.scale(loss).backward()

            # Accumulate metrics
            accum_cap_fine += loss_cap_fine.item()
            accum_cap_coarse += loss_cap_coarse.item()
            accum_rec_fine += loss_rec_fine.item()
            accum_rec_coarse += loss_rec_coarse.item()
            accum_count += 1

            if accum_count >= CONFIG["grad_accum"]:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Compute averages
                avg_cap_fine = accum_cap_fine / accum_count
                avg_cap_coarse = accum_cap_coarse / accum_count
                avg_rec_fine = accum_rec_fine / accum_count
                avg_rec_coarse = accum_rec_coarse / accum_count

                # Compute ratios
                cap_ratio = avg_cap_coarse / avg_cap_fine if avg_cap_fine > 0 else 1.0
                rec_ratio = avg_rec_coarse / avg_rec_fine if avg_rec_fine > 0 else 1.0

                cap_ratios.append(cap_ratio)
                rec_ratios.append(rec_ratio)

                # Track step time
                now = time.time()
                step_times.append(now - last_step_time)
                last_step_time = now

                step += 1
                pbar.update(1)

                # Logging
                if step % CONFIG["log_interval"] == 0:
                    avg_cap_ratio = np.mean(list(cap_ratios))
                    avg_rec_ratio = np.mean(list(rec_ratios))
                    avg_step_time = np.mean(list(step_times))

                    # Estimate remaining time
                    remaining_steps = CONFIG["target_steps"] - step
                    remaining_time_steps = remaining_steps * avg_step_time
                    remaining_time_limit = max_duration - (time.time() - start_time)
                    remaining_time = min(remaining_time_steps, remaining_time_limit)
                    eta = datetime.now() + timedelta(seconds=remaining_time)

                    # Data loader stats
                    dl_stats = data_loader.get_stats()
                    success_rate = dl_stats["ok"] / (dl_stats["ok"] + dl_stats["fail"]) * 100 if (dl_stats["ok"] + dl_stats["fail"]) > 0 else 0

                    log_dict = {
                        "caption/loss_fine": avg_cap_fine,
                        "caption/loss_coarse": avg_cap_coarse,
                        "caption/ratio": cap_ratio,
                        "caption/ratio_avg100": avg_cap_ratio,
                        "recon/loss_fine": avg_rec_fine,
                        "recon/loss_coarse": avg_rec_coarse,
                        "recon/ratio": rec_ratio,
                        "recon/ratio_avg100": avg_rec_ratio,
                        "perf/step_time": avg_step_time,
                        "perf/success_rate": success_rate,
                        "perf/queue_size": data_loader.queue.qsize(),
                    }

                    if HAS_WANDB:
                        wandb.log(log_dict, step=step)

                    cap_status = "CAP+" if cap_ratio > 1.02 else "cap="
                    rec_status = "REC+" if rec_ratio > 1.02 else "rec="

                    tqdm.write(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Step {step:6d} | "
                        f"cap: {avg_cap_fine:.3f}/{avg_cap_coarse:.3f} r={cap_ratio:.3f} ({cap_status}) | "
                        f"rec: {avg_rec_fine:.3f}/{avg_rec_coarse:.3f} r={rec_ratio:.3f} ({rec_status}) | "
                        f"{avg_step_time:.2f}s/step | ETA: {eta.strftime('%H:%M')}"
                    )

                # Reset accumulators
                accum_cap_fine = 0.0
                accum_cap_coarse = 0.0
                accum_rec_fine = 0.0
                accum_rec_coarse = 0.0
                accum_count = 0

                # Evaluation (sample captions)
                if step % CONFIG["eval_interval"] == 0:
                    model.eval()
                    with torch.no_grad():
                        try:
                            eval_frames, eval_caption = data_loader.queue.get(timeout=10)
                            eval_frames_norm = normalize_frames(eval_frames.unsqueeze(0).to(device))

                            caption_fine = model.generate_caption(
                                eval_frames_norm, tokenizer, max_new_tokens=40, use_fine=True
                            )[0]
                            caption_coarse = model.generate_caption(
                                eval_frames_norm, tokenizer, max_new_tokens=40, use_fine=False
                            )[0]

                            tqdm.write(f"\n{'='*60}")
                            tqdm.write(f"EVAL at step {step}")
                            tqdm.write(f"GT:     {eval_caption[:80]}")
                            tqdm.write(f"FINE:   {caption_fine[:80]}")
                            tqdm.write(f"COARSE: {caption_coarse[:80]}")
                            tqdm.write(f"{'='*60}\n")

                        except Exception as e:
                            tqdm.write(f"Eval error: {e}")
                    model.train()

                # Checkpoint
                if step % CONFIG["save_interval"] == 0:
                    metrics = {
                        'cap_ratios': list(cap_ratios),
                        'rec_ratios': list(rec_ratios),
                        'avg_cap_ratio': np.mean(list(cap_ratios)),
                        'avg_rec_ratio': np.mean(list(rec_ratios)),
                        'elapsed_hours': (time.time() - start_time) / 3600,
                    }
                    save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint...")

    finally:
        # Stop data loader
        data_loader.stop()
        pbar.close()

        # Final checkpoint
        elapsed_hours = (time.time() - start_time) / 3600
        final_cap_ratio = np.mean(list(cap_ratios)) if cap_ratios else 1.0
        final_rec_ratio = np.mean(list(rec_ratios)) if rec_ratios else 1.0

        metrics = {
            'cap_ratios': list(cap_ratios),
            'rec_ratios': list(rec_ratios),
            'final_cap_ratio': final_cap_ratio,
            'final_rec_ratio': final_rec_ratio,
            'elapsed_hours': elapsed_hours,
            'total_steps': step,
        }
        save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints", force_save=True)

        # Save summary
        summary = {
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_hours': elapsed_hours,
            'total_steps': step,
            'steps_from_resume': step - start_step,
            'final_cap_ratio': final_cap_ratio,
            'final_rec_ratio': final_rec_ratio,
            'config': CONFIG,
        }
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Duration: {elapsed_hours:.2f} hours")
        print(f"Total steps: {step} (started from {start_step})")
        print(f"Steps completed: {step - start_step}")
        print(f"Final Caption Ratio (avg100): {final_cap_ratio:.4f}")
        print(f"Final Recon Ratio (avg100): {final_rec_ratio:.4f}")
        print(f"Checkpoints: {output_dir / 'checkpoints'}")
        print(f"Summary: {output_dir / 'training_summary.json'}")

        if final_cap_ratio > 1.05:
            print("\nCAPTIONING: Fine queries BETTER than coarse (EXPECTED)")
        if final_rec_ratio > 1.02:
            print("RECONSTRUCTION: Fine queries BETTER than coarse (THESIS VALIDATED!)")

        if HAS_WANDB:
            wandb.log({
                "final/cap_ratio": final_cap_ratio,
                "final/rec_ratio": final_rec_ratio,
                "final/total_steps": step,
                "final/elapsed_hours": elapsed_hours,
            })
            wandb.finish()

        print("=" * 80)


if __name__ == "__main__":
    # Check for GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        sys.exit(1)

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_training()
