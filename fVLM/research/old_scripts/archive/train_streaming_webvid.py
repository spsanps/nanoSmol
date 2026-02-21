#!/usr/bin/env python3
"""
Streaming training from WebVid-10M.

Downloads and processes videos on-the-fly for maximum data coverage.
Target: 4 hours of training with as many samples as possible.
"""

import argparse
import os
import sys
import time
import tempfile
import queue
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import requests
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.smolvlm_video import SmolVLMVideo

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available, using PIL fallback")


def download_video(url: str, timeout: int = 10) -> bytes | None:
    """Download video from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None


def extract_frames_cv2(video_bytes: bytes, num_frames: int = 8, size: int = 256) -> torch.Tensor | None:
    """Extract frames from video bytes using OpenCV."""
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name

        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            cap.release()
            os.unlink(temp_path)
            return None

        # Sample frame indices uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB, resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size))
                frames.append(frame)

        cap.release()
        os.unlink(temp_path)

        if len(frames) < num_frames:
            return None

        # Stack and normalize to [-1, 1]
        frames = np.stack(frames)  # (T, H, W, 3)
        frames = frames.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        frames = torch.from_numpy(frames).float() / 127.5 - 1.0

        return frames

    except Exception as e:
        return None


class WebVidStreamingDataset(IterableDataset):
    """Streaming dataset that downloads and processes WebVid on-the-fly."""

    def __init__(
        self,
        num_frames: int = 8,
        frame_size: int = 256,
        buffer_size: int = 100,
        num_download_workers: int = 8,
        max_samples: int = None,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.buffer_size = buffer_size
        self.num_download_workers = num_download_workers
        self.max_samples = max_samples

        # Load streaming dataset
        print("Loading WebVid-10M metadata (streaming)...")
        self.dataset = load_dataset(
            'TempoFunk/webvid-10M',
            split='train',
            streaming=True
        )
        print("Dataset loaded!")

    def __iter__(self):
        buffer = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()
        samples_produced = [0]  # Use list for mutability in closure

        def producer():
            """Download and process videos in background."""
            with ThreadPoolExecutor(max_workers=self.num_download_workers) as executor:
                futures = {}
                ds_iter = iter(self.dataset)

                while not stop_event.is_set():
                    # Submit new downloads
                    while len(futures) < self.num_download_workers * 2:
                        try:
                            sample = next(ds_iter)
                            url = sample['contentUrl']
                            caption = sample['name']
                            future = executor.submit(self._process_sample, url, caption)
                            futures[future] = (url, caption)
                        except StopIteration:
                            break

                    if not futures:
                        break

                    # Collect completed futures
                    done = []
                    for future in list(futures.keys()):
                        if future.done():
                            done.append(future)

                    for future in done:
                        try:
                            result = future.result()
                            if result is not None:
                                if self.max_samples and samples_produced[0] >= self.max_samples:
                                    stop_event.set()
                                    break
                                buffer.put(result, timeout=1)
                                samples_produced[0] += 1
                        except Exception:
                            pass
                        del futures[future]

                    time.sleep(0.01)

            buffer.put(None)  # Signal end

        def _process_sample(self, url: str, caption: str):
            """Download and process a single sample."""
            video_bytes = download_video(url)
            if video_bytes is None:
                return None

            frames = extract_frames_cv2(video_bytes, self.num_frames, self.frame_size)
            if frames is None:
                return None

            return {"frames": frames, "caption": caption}

        self._process_sample = _process_sample

        # Start producer thread
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # Yield from buffer
        while True:
            try:
                item = buffer.get(timeout=30)
                if item is None:
                    break
                yield item
            except queue.Empty:
                break

        stop_event.set()


class PreloadedWebVidDataset(IterableDataset):
    """Preload samples into memory for faster iteration."""

    def __init__(
        self,
        num_samples: int = 10000,
        num_frames: int = 8,
        frame_size: int = 256,
        num_workers: int = 16,
        cache_dir: str = None,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.samples = []
        self._preload()

    def _preload(self):
        """Preload samples from WebVid."""
        print(f"Preloading {self.num_samples} samples from WebVid-10M...")

        # Check cache first
        if self.cache_dir and self.cache_dir.exists():
            cache_file = self.cache_dir / f"webvid_cache_{self.num_samples}.pt"
            if cache_file.exists():
                print(f"Loading from cache: {cache_file}")
                self.samples = torch.load(cache_file, weights_only=False)
                print(f"Loaded {len(self.samples)} cached samples")
                return

        ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

        success = 0
        failed = 0
        pbar = tqdm(total=self.num_samples, desc="Downloading")

        def process_sample(item):
            url = item['contentUrl']
            caption = item['name']

            video_bytes = download_video(url, timeout=15)
            if video_bytes is None:
                return None

            frames = extract_frames_cv2(video_bytes, self.num_frames, self.frame_size)
            if frames is None:
                return None

            return {"frames": frames, "caption": caption}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            ds_iter = iter(ds)

            # Submit initial batch
            for _ in range(self.num_workers * 4):
                try:
                    item = next(ds_iter)
                    futures.append(executor.submit(process_sample, item))
                except StopIteration:
                    break

            while success < self.num_samples and futures:
                # Wait for any future to complete
                done_futures = []
                for f in futures:
                    if f.done():
                        done_futures.append(f)

                for f in done_futures:
                    futures.remove(f)
                    try:
                        result = f.result()
                        if result is not None:
                            self.samples.append(result)
                            success += 1
                            pbar.update(1)
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

                    # Submit new job
                    if success + len(futures) < self.num_samples + 100:
                        try:
                            item = next(ds_iter)
                            futures.append(executor.submit(process_sample, item))
                        except StopIteration:
                            pass

                if not done_futures:
                    time.sleep(0.1)

        pbar.close()
        print(f"Preloaded {success} samples ({failed} failed)")

        # Save cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"webvid_cache_{self.num_samples}.pt"
            torch.save(self.samples, cache_file)
            print(f"Saved cache: {cache_file}")

    def __iter__(self):
        # Shuffle and yield
        indices = torch.randperm(len(self.samples)).tolist()
        for idx in indices:
            yield self.samples[idx]

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"frames": frames, "captions": captions}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"STREAMING WEBVID TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Target: {args.num_samples:,} samples")
    print(f"Max time: {args.max_hours} hours")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print(f"\n=== Loading Model ===")
    model = SmolVLMVideo(
        model_name=args.model_name,
        freeze_vision=True,
        gradient_checkpointing=True,
    )
    model = model.to(device)
    model.vae.to(device, dtype=torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.1f}M")

    # Create dataset
    print(f"\n=== Loading Dataset ===")

    if args.preload:
        dataset = PreloadedWebVidDataset(
            num_samples=args.num_samples,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            num_workers=args.download_workers,
            cache_dir=args.cache_dir,
        )
    else:
        dataset = WebVidStreamingDataset(
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            buffer_size=args.buffer_size,
            num_download_workers=args.download_workers,
            max_samples=args.num_samples,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Streaming handles its own workers
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if HAS_WANDB and args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # Training
    print(f"\n=== Starting Training ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    start_time = time.time()
    max_time = args.max_hours * 3600
    global_step = 0
    total_samples = 0
    losses = []
    best_loss = float("inf")

    model.train()
    pbar = tqdm(desc="Training")

    for batch in dataloader:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_time:
            print(f"\nTime limit reached ({args.max_hours}h)")
            break

        # Move to device
        frames = batch["frames"].to(device, dtype=torch.bfloat16)  # (B, T, 3, H, W)
        B, T = frames.shape[:2]

        # Encode frames to latents with VAE
        with torch.no_grad():
            flat_frames = frames.view(B * T, 3, args.frame_size, args.frame_size).float()
            # Normalize for VAE (already -1 to 1)
            latents = model.vae.encode(flat_frames).latent_dist.sample()
            latents = latents * 0.18215  # VAE scaling factor
            latents = latents.view(B, T, 4, args.frame_size // 8, args.frame_size // 8)

        # Split into context and target
        context_latents = latents[:, :-1]  # (B, T-1, 4, H, W)
        target_latents = latents[:, -1]    # (B, 4, H, W)

        # Decode context for VLM input
        with torch.no_grad():
            T_ctx = T - 1
            flat_ctx = context_latents.view(B * T_ctx, 4, args.frame_size // 8, args.frame_size // 8).float()
            pixel_values = model.vae.decode(flat_ctx / 0.18215).sample
            pixel_values = pixel_values.view(B, T_ctx, 3, args.frame_size, args.frame_size)
            pixel_values = pixel_values.to(torch.bfloat16)

        # Forward
        pred_out = model.forward_predict(pixel_values, target_latents.to(torch.bfloat16))
        loss = pred_out["loss"]

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        total_samples += B
        losses.append(loss.item())

        # Logging
        if global_step % args.log_every == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            mem = torch.cuda.max_memory_allocated() / 1e9
            throughput = total_samples / elapsed

            log_dict = {
                "loss": loss.item(),
                "avg_loss": avg_loss,
                "samples": total_samples,
                "throughput": throughput,
                "memory_gb": mem,
            }

            if HAS_WANDB and args.wandb:
                wandb.log(log_dict)

            remaining = max_time - elapsed
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "samples": f"{total_samples:,}",
                "spd": f"{throughput:.1f}/s",
                "eta": f"{remaining/3600:.1f}h",
            })

        pbar.update(1)

        # Save checkpoint
        if global_step % args.save_every == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "step": global_step,
                    "samples": total_samples,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss,
                }, checkpoint_dir / "best.pt")

    pbar.close()

    # Final save
    total_time = time.time() - start_time
    final_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else 0

    torch.save({
        "step": global_step,
        "samples": total_samples,
        "model_state_dict": model.state_dict(),
        "loss": final_loss,
        "training_time": total_time,
    }, checkpoint_dir / "final.pt")

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {total_time/3600:.2f} hours")
    print(f"Samples: {total_samples:,}")
    print(f"Throughput: {total_samples/total_time:.1f} samples/sec")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    # Save summary
    summary = {
        "training_time_hours": total_time / 3600,
        "total_samples": total_samples,
        "throughput": total_samples / total_time,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "config": vars(args),
    }

    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if HAS_WANDB and args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--preload", action="store_true", help="Preload all samples")
    parser.add_argument("--cache_dir", type=str, default="data/webvid_cache")
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--download_workers", type=int, default=16)

    # Model
    parser.add_argument("--model_name", type=str,
                        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")

    # Training
    parser.add_argument("--max_hours", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/webvid_streaming")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="smolvlm-video")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"webvid_{args.num_samples//1000}k_{args.max_hours}h"

    train(args)


if __name__ == "__main__":
    main()
