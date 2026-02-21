#!/usr/bin/env python3
"""
Joint Reconstruction + Captioning with Multi-Fine Iterations (8h scaled run).

NEW EXPERIMENT: Combines two successful approaches:
1. Joint training (caption + reconstruction) - teaches WHERE to look
2. Multi-fine iterations (coarse → fine₁ → fine₂) - closes train-test gap

Architecture per batch:
  frames → coarse (q_static) → z°
        → fine₁ (query from z°) → z₁
        → fine₂ (query from z₁) → z₂ (final)

Loss:
  loss = loss_caption_fine₂ + λ * loss_recon_fine₂

We track all iterations to see progressive improvement.
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
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Training
    "max_hours": 7.5,
    "target_steps": 50000,
    "batch_size": 8,
    "grad_accum": 3,  # Effective batch = 24
    "num_frames": 8,
    "learning_rate": 3e-5,
    "lambda_recon": 0.5,
    "warmup_steps": 100,

    # Multi-fine iterations
    "fine_iterations": 2,  # coarse → fine₁ → fine₂

    # Logging
    "log_interval": 100,
    "eval_interval": 1000,
    "save_interval": 2000,

    # Data loading
    "prefetch_size": 24,
    "download_timeout": 20,
    "max_video_duration": 15,
    "num_prefetch_threads": 1,  # Single thread to avoid dataset loading race

    # Checkpointing - start fresh (new experiment)
    "resume_checkpoint": None,
    "output_dir": "outputs/joint_multifine_8h",
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
    """Async data loader with multi-threaded prefetching."""

    def __init__(self, num_frames=8, prefetch_size=24, max_duration=15, num_threads=2):
        self.num_frames = num_frames
        self.prefetch_size = prefetch_size
        self.max_duration = max_duration
        self.queue = Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.stats = {"ok": 0, "fail": 0}
        self.lock = threading.Lock()

        self.threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=self._prefetch_loop, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _prefetch_loop(self, thread_id):
        ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
        skip_count = thread_id * 10000
        skipped = 0

        for sample in ds:
            if self.stop_event.is_set():
                break
            if skipped < skip_count:
                skipped += 1
                continue
            try:
                duration = parse_duration(sample.get('duration', ''))
                if duration == 0 or duration > self.max_duration:
                    continue
                url = sample.get('contentUrl')
                caption = sample.get('name', '')
                if not url or not caption or len(caption) < 5:
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
                self.queue.put((frames, caption), timeout=60)
                with self.lock:
                    self.stats["ok"] += 1
            except:
                with self.lock:
                    self.stats["fail"] += 1

    def get_batch(self, batch_size):
        frames_list, captions_list = [], []
        while len(frames_list) < batch_size:
            try:
                frames, caption = self.queue.get(timeout=120)
                frames_list.append(frames)
                captions_list.append(caption)
            except Empty:
                time.sleep(1)
        return torch.stack(frames_list), captions_list

    def stop(self):
        self.stop_event.set()

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
    return latents_flat.reshape(B, T, 4, H // 8, W // 8)


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


def forward_multifine_joint(model, frames, caption_ids, caption_mask, vae_latents, tokenizer, num_iterations=2):
    """
    Multi-fine iteration forward pass for BOTH captioning and reconstruction.
    Returns losses for coarse and each fine iteration for both tasks.

    Architecture:
      coarse (q_static) → z° → LLM → queries₁
      fine₁ (queries₁)  → z₁ → LLM → queries₂
      fine₂ (queries₂)  → z₂ (final)
    """
    B, T = frames.shape[:2]
    device = frames.device

    # Encode all frames with DINO
    frames_flat = frames.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Build per-frame caches
    all_caches = []
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        for t in range(T):
            frame_kv_cache = []
            for layer_idx in range(num_layers):
                layer_cache = cache_flat['kv_cache'][layer_idx]
                K_all = layer_cache['K'].reshape(B, T, N, D)
                V_all = layer_cache['V'].reshape(B, T, N, D)
                frame_kv_cache.append({
                    'K': K_all[:, t],
                    'V': V_all[:, t],
                    'layer': layer_cache['layer'],
                })
            all_caches.append({
                'patch_features': patch_features[:, t],
                'kv_cache': frame_kv_cache,
            })
    else:
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    # Prepare embeddings
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]
    text_embeds = model.get_empty_text_embeds(B)
    N_text = text_embeds.shape[1]

    # prev_latents for reconstruction conditioning
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)

    # === Pass 0: Coarse (baseline) ===
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    coarse_token = model.coarse_token.expand(B, -1, -1)
    fine_token = model.fine_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)

    # --- Coarse caption loss ---
    seq_cap_coarse = torch.cat([coarse_token, z_coarse_llm, caption_embeds], dim=1)
    outputs_cap_coarse = model.llm.model(inputs_embeds=seq_cap_coarse)
    logits_cap_coarse = model.llm.lm_head(outputs_cap_coarse.last_hidden_state)
    caption_logits_coarse = logits_cap_coarse[:, 1+T:-1, :]
    loss_cap_coarse = F.cross_entropy(
        caption_logits_coarse.reshape(-1, caption_logits_coarse.size(-1)),
        caption_targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )

    # --- Coarse reconstruction loss ---
    # Sequence: [text_embeds, coarse_token, z_coarse]
    seq_rec_coarse = torch.cat([text_embeds, coarse_token, z_coarse_llm], dim=1)
    outputs_rec_coarse = model.llm.model(inputs_embeds=seq_rec_coarse)
    # h positions for prediction: after text_embeds, at z_coarse positions
    h_coarse_for_pred = outputs_rec_coarse.last_hidden_state[:, N_text:N_text + T]
    pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)
    loss_rec_coarse = F.mse_loss(pred_coarse, vae_latents)

    # --- Generate first queries from coarse ---
    seq_query0 = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_query0 = model.llm.model(inputs_embeds=seq_query0)
    queries = model.llm_to_query(outputs_query0.last_hidden_state[:, 2:])

    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    current_queries = torch.cat([q_init, queries[:, :-1]], dim=1)

    # === Fine iterations ===
    cap_losses = []
    rec_losses = []

    for iteration in range(num_iterations):
        # Extract fine features
        z_fine_list = [model.encoder.query_attend(current_queries[:, t], all_caches[t]) for t in range(T)]
        z_fine = torch.stack(z_fine_list, dim=1)
        z_fine_llm = model.dino_to_llm(z_fine)
        z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

        # --- Caption loss for this iteration ---
        seq_cap_fine = torch.cat([fine_token, z_fine_llm, caption_embeds], dim=1)
        outputs_cap_fine = model.llm.model(inputs_embeds=seq_cap_fine)
        logits_cap_fine = model.llm.lm_head(outputs_cap_fine.last_hidden_state)
        caption_logits_fine = logits_cap_fine[:, 1+T:-1, :]
        loss_cap_iter = F.cross_entropy(
            caption_logits_fine.reshape(-1, caption_logits_fine.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        cap_losses.append(loss_cap_iter)

        # --- Reconstruction loss for this iteration ---
        seq_rec_fine = torch.cat([text_embeds, fine_token, z_fine_llm], dim=1)
        outputs_rec_fine = model.llm.model(inputs_embeds=seq_rec_fine)
        h_fine_for_pred = outputs_rec_fine.last_hidden_state[:, N_text:N_text + T]
        pred_fine = model.pred_head(h_fine_for_pred, prev_latents)
        loss_rec_iter = F.mse_loss(pred_fine, vae_latents)
        rec_losses.append(loss_rec_iter)

        # Generate next queries (for next iteration)
        if iteration < num_iterations - 1:
            seq_query = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
            outputs_query = model.llm.model(inputs_embeds=seq_query)
            next_queries = model.llm_to_query(outputs_query.last_hidden_state[:, 2:])
            current_queries = torch.cat([q_init, next_queries[:, :-1]], dim=1)

    return loss_cap_coarse, cap_losses, loss_rec_coarse, rec_losses


def run_training():
    """Run joint + multi-fine training."""

    start_time = time.time()
    max_duration = CONFIG["max_hours"] * 3600
    num_fine_iters = CONFIG["fine_iterations"]

    device = torch.device("cuda")
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("JOINT + MULTI-FINE TRAINING (NEW EXPERIMENT)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fine iterations: {num_fine_iters} (coarse → fine₁ → fine₂)")
    print(f"Batch: {CONFIG['batch_size']} x {CONFIG['grad_accum']} = {CONFIG['batch_size'] * CONFIG['grad_accum']}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    if HAS_WANDB:
        wandb.init(
            project="foveated-vlm-joint",
            name=f"joint_multifine_{datetime.now().strftime('%m%d_%H%M')}",
            config=CONFIG,
        )

    # Load models
    print("\n[1/4] Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
    model.train()

    print("[2/4] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = GradScaler('cuda')

    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / CONFIG["warmup_steps"]
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume if checkpoint exists
    start_step = 0
    if CONFIG["resume_checkpoint"] and Path(CONFIG["resume_checkpoint"]).exists():
        print(f"[3/4] Resuming from {CONFIG['resume_checkpoint']}...")
        ckpt = torch.load(CONFIG["resume_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt['step']
    else:
        print("[3/4] Starting fresh (new experiment)")

    print("[4/4] Starting data loader...")
    data_loader = AsyncDataLoader(
        num_frames=CONFIG["num_frames"],
        prefetch_size=CONFIG["prefetch_size"],
        max_duration=CONFIG["max_video_duration"],
        num_threads=CONFIG["num_prefetch_threads"],
    )
    time.sleep(10)
    print(f"  Queue size: {data_loader.queue.qsize()}")

    # Tracking
    step = start_step
    accum_count = 0
    accum_cap_coarse = 0.0
    accum_cap_iters = [0.0] * num_fine_iters
    accum_rec_coarse = 0.0
    accum_rec_iters = [0.0] * num_fine_iters

    cap_ratios = deque(maxlen=100)
    rec_ratios = deque(maxlen=100)
    step_times = deque(maxlen=50)

    print(f"\n{'='*80}")
    print("TRAINING STARTED")
    print(f"{'='*80}\n")

    pbar = tqdm(total=CONFIG["target_steps"], initial=start_step, desc="Training")
    optimizer.zero_grad()
    last_step_time = time.time()

    try:
        while step < CONFIG["target_steps"]:
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                print(f"\n[TIME LIMIT] {CONFIG['max_hours']} hours reached")
                break

            frames_raw, captions = data_loader.get_batch(CONFIG["batch_size"])
            frames_norm = normalize_frames(frames_raw.to(device))
            vae_latents = compute_vae_latents(vae, frames_raw, device)

            tokens = tokenizer(
                captions, padding=True, truncation=True,
                max_length=64, return_tensors='pt'
            )
            caption_ids = tokens['input_ids'].to(device)
            caption_mask = tokens['attention_mask'].to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Joint multi-fine forward (both tasks together)
                cap_coarse, cap_iter_losses, rec_coarse, rec_iter_losses = forward_multifine_joint(
                    model, frames_norm, caption_ids, caption_mask, vae_latents, tokenizer, num_fine_iters
                )

                # Train on final fine iteration
                loss_cap = cap_iter_losses[-1]
                loss_rec = rec_iter_losses[-1]
                loss = (loss_cap + CONFIG["lambda_recon"] * loss_rec) / CONFIG["grad_accum"]

            scaler.scale(loss).backward()

            # Accumulate metrics
            accum_cap_coarse += cap_coarse.item()
            for i, l in enumerate(cap_iter_losses):
                accum_cap_iters[i] += l.item()
            accum_rec_coarse += rec_coarse.item()
            for i, l in enumerate(rec_iter_losses):
                accum_rec_iters[i] += l.item()
            accum_count += 1

            if accum_count >= CONFIG["grad_accum"]:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Compute averages
                avg_cap_coarse = accum_cap_coarse / accum_count
                avg_cap_iters = [x / accum_count for x in accum_cap_iters]
                avg_rec_coarse = accum_rec_coarse / accum_count
                avg_rec_iters = [x / accum_count for x in accum_rec_iters]

                # Final fine vs coarse ratio
                cap_ratio = avg_cap_coarse / avg_cap_iters[-1] if avg_cap_iters[-1] > 0 else 1.0
                rec_ratio = avg_rec_coarse / avg_rec_iters[-1] if avg_rec_iters[-1] > 0 else 1.0

                cap_ratios.append(cap_ratio)
                rec_ratios.append(rec_ratio)

                now = time.time()
                step_times.append(now - last_step_time)
                last_step_time = now

                step += 1
                pbar.update(1)

                if step % CONFIG["log_interval"] == 0:
                    avg_step_time = np.mean(list(step_times))
                    remaining = min(
                        (CONFIG["target_steps"] - step) * avg_step_time,
                        max_duration - (time.time() - start_time)
                    )
                    eta = datetime.now() + timedelta(seconds=remaining)

                    # Build iteration progression string
                    cap_prog = " → ".join([f"{l:.3f}" for l in avg_cap_iters])
                    rec_prog = " → ".join([f"{l:.3f}" for l in avg_rec_iters])

                    log_dict = {
                        "caption/loss_coarse": avg_cap_coarse,
                        "caption/loss_fine": avg_cap_iters[-1],
                        "caption/ratio": cap_ratio,
                        "recon/loss_coarse": avg_rec_coarse,
                        "recon/loss_fine": avg_rec_iters[-1],
                        "recon/ratio": rec_ratio,
                    }
                    for i, l in enumerate(avg_cap_iters):
                        log_dict[f"caption/iter_{i+1}"] = l
                    for i, l in enumerate(avg_rec_iters):
                        log_dict[f"recon/iter_{i+1}"] = l

                    if HAS_WANDB:
                        wandb.log(log_dict, step=step)

                    tqdm.write(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Step {step:6d} | "
                        f"cap: {avg_cap_coarse:.3f} → [{cap_prog}] r={cap_ratio:.3f} | "
                        f"rec: {avg_rec_coarse:.3f} → [{rec_prog}] r={rec_ratio:.3f} | "
                        f"{avg_step_time:.2f}s | ETA: {eta.strftime('%H:%M')}"
                    )

                # Reset accumulators
                accum_cap_coarse = 0.0
                accum_cap_iters = [0.0] * num_fine_iters
                accum_rec_coarse = 0.0
                accum_rec_iters = [0.0] * num_fine_iters
                accum_count = 0

                # Checkpoint
                if step % CONFIG["save_interval"] == 0:
                    metrics = {
                        'cap_ratio': np.mean(list(cap_ratios)),
                        'rec_ratio': np.mean(list(rec_ratios)),
                        'elapsed_hours': (time.time() - start_time) / 3600,
                    }
                    save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")

    finally:
        data_loader.stop()
        pbar.close()

        elapsed_hours = (time.time() - start_time) / 3600
        final_cap_ratio = np.mean(list(cap_ratios)) if cap_ratios else 1.0
        final_rec_ratio = np.mean(list(rec_ratios)) if rec_ratios else 1.0

        metrics = {
            'final_cap_ratio': final_cap_ratio,
            'final_rec_ratio': final_rec_ratio,
            'elapsed_hours': elapsed_hours,
            'total_steps': step,
        }
        save_checkpoint(model, optimizer, scaler, step, metrics, output_dir / "checkpoints", force_save=True)

        summary = {
            'experiment': 'joint_multifine',
            'fine_iterations': num_fine_iters,
            'elapsed_hours': elapsed_hours,
            'total_steps': step,
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
        print(f"Steps: {step}")
        print(f"Final Caption Ratio: {final_cap_ratio:.4f}")
        print(f"Final Recon Ratio: {final_rec_ratio:.4f}")
        print("=" * 80)

        if HAS_WANDB:
            wandb.log({
                "final/cap_ratio": final_cap_ratio,
                "final/rec_ratio": final_rec_ratio,
            })
            wandb.finish()


if __name__ == "__main__":
    run_training()
