"""
Multi-task Training: Reconstruction + Captioning

Trains both:
1. Video reconstruction (VAE latent prediction) - original task
2. Video captioning (text generation) - new task

Combined loss: loss_total = loss_recon + lambda_caption * loss_caption
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import IterableDataset, DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from diffusers import AutoencoderKL
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import re
import time
import threading
import queue

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


def download_video(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout, stream=True)
    if response.status_code == 200:
        return response.content
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
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
                frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                frames.append(frame)
            return torch.stack(frames)


class MultiTaskDataset(IterableDataset):
    """Streaming dataset for multi-task training (reconstruction + captioning)."""

    def __init__(
        self,
        num_frames: int = 16,
        frame_size: int = 256,
        min_duration: int = 5,
        max_duration: int = 30,
        max_caption_tokens: int = 64,
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_caption_tokens = max_caption_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stats = {'attempted': 0, 'success': 0, 'filtered': 0, 'failed': 0}

    def __iter__(self):
        for retry in range(5):
            try:
                ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
                break
            except Exception as e:
                if retry < 4:
                    time.sleep(2 ** retry)
                else:
                    raise e

        for sample in ds:
            self.stats['attempted'] += 1

            duration = parse_duration(sample.get('duration', ''))
            if duration < self.min_duration or duration > self.max_duration:
                self.stats['filtered'] += 1
                continue

            try:
                video_bytes = download_video(sample['contentUrl'])
                if video_bytes is None:
                    self.stats['failed'] += 1
                    continue
            except:
                self.stats['failed'] += 1
                continue

            try:
                frames_raw = extract_frames(video_bytes, self.num_frames, self.frame_size)
                if frames_raw is None:
                    self.stats['failed'] += 1
                    continue
            except:
                self.stats['failed'] += 1
                continue

            # Tokenize caption
            caption = sample.get('name', '')
            tokens = self.tokenizer(
                caption,
                max_length=self.max_caption_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.stats['success'] += 1

            yield {
                'frames_raw': frames_raw,  # [T, 3, H, W] uint8 - for VAE
                'caption_ids': tokens['input_ids'].squeeze(0),
                'caption_mask': tokens['attention_mask'].squeeze(0),
                'caption': caption,
            }


class PrefetchingDataset(IterableDataset):
    def __init__(self, base_dataset, buffer_size: int = 16):
        self.base = base_dataset
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()

    def _producer(self):
        try:
            for sample in self.base:
                if self.stop_event.is_set():
                    break
                self.buffer.put(sample)
        finally:
            self.buffer.put(None)

    def __iter__(self):
        self.stop_event.clear()
        producer = threading.Thread(target=self._producer, daemon=True)
        producer.start()
        while True:
            sample = self.buffer.get()
            if sample is None:
                break
            yield sample

    def stop(self):
        self.stop_event.set()


def collate_fn(batch):
    return {
        'frames_raw': torch.stack([b['frames_raw'] for b in batch]),
        'caption_ids': torch.stack([b['caption_ids'] for b in batch]),
        'caption_mask': torch.stack([b['caption_mask'] for b in batch]),
        'captions': [b['caption'] for b in batch],
    }


def compute_vae_latents(frames_raw, vae, device):
    """Compute VAE latents from raw frames."""
    B, T, C, H, W = frames_raw.shape
    frames_flat = frames_raw.view(B * T, C, H, W).float().to(device)
    frames_vae = frames_flat / 255.0 * 2 - 1  # [-1, 1]

    with torch.no_grad():
        chunk_size = 16
        latent_chunks = []
        for i in range(0, frames_vae.shape[0], chunk_size):
            chunk = frames_vae[i:i+chunk_size].half()
            latent = vae.encode(chunk).latent_dist.sample() * 0.18215
            latent_chunks.append(latent)
        latents = torch.cat(latent_chunks, dim=0)

    return latents.view(B, T, 4, 32, 32).float()


def normalize_for_dino(frames_raw, device):
    """Normalize raw frames for DINO."""
    frames = frames_raw.float().to(device) / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames - mean) / std


def main():
    config = {
        'num_frames': 16,
        'frame_size': 256,
        'min_duration': 5,
        'max_duration': 30,
        'max_caption_tokens': 64,
        'batch_size': 2,  # Reduced for memory
        'grad_accum': 8,  # Effective batch = 16
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'max_steps': 15000,  # ~8 hours at ~2s/step
        'grad_clip': 1.0,
        'lambda_caption': 0.1,  # Weight for caption loss (start low, recon is primary)
        'log_every': 100,
        'save_every': 1500,
        'eval_every': 500,
        'output_dir': 'outputs/multitask',
    }

    print("=" * 70)
    print("Multi-Task Training: Reconstruction + Captioning")
    print("=" * 70)
    print(f"Frames: {config['num_frames']}")
    print(f"Batch: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"Lambda caption: {config['lambda_caption']}")
    print(f"Max steps: {config['max_steps']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(config['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    if HAS_WANDB:
        wandb.init(
            project="foveated-vlm",
            config=config,
            name=f"multitask_{datetime.now().strftime('%m%d_%H%M')}",
        )

    # Load VAE for reconstruction
    print("\nLoading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Create model with BASE weights (preserves LLM language capabilities)
    print("Loading model with base weights (fresh SmolLM2)...")
    model_cfg = {
        'dino_model': 'facebook/dinov2-small',
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_dim': 384,
        'llm_dim': 576,
        'query_dim': 128,
        'lambda_coarse': 0.5,
    }
    model = FoveatedVideoModel(
        dino_model=model_cfg['dino_model'],
        llm_model=model_cfg['llm_model'],
        dino_dim=model_cfg['dino_dim'],
        llm_dim=model_cfg['llm_dim'],
        query_dim=model_cfg['query_dim'],
        lambda_coarse=model_cfg['lambda_coarse'],
    ).to(device)
    print("Using base model weights (SmolLM2 language capabilities preserved)")

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for LLM")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()

    # Dataset
    print("\nCreating dataset...")
    base_dataset = MultiTaskDataset(
        num_frames=config['num_frames'],
        frame_size=config['frame_size'],
        min_duration=config['min_duration'],
        max_duration=config['max_duration'],
        max_caption_tokens=config['max_caption_tokens'],
        llm_model=model_cfg['llm_model'],
    )
    dataset = PrefetchingDataset(base_dataset, buffer_size=16)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=0, collate_fn=collate_fn)

    # Training
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    model.train()
    global_step = 0
    running_loss = 0
    running_recon = 0
    running_caption = 0
    running_fine = 0
    running_coarse = 0

    pbar = tqdm(total=config['max_steps'], desc="Training")

    while global_step < config['max_steps']:
        for batch in dataloader:
            if global_step >= config['max_steps']:
                break

            frames_raw = batch['frames_raw']
            caption_ids = batch['caption_ids'].to(device)
            caption_mask = batch['caption_mask'].to(device)

            # Compute VAE latents for reconstruction
            latents = compute_vae_latents(frames_raw, vae, device)

            # Normalize frames for DINO
            frames = normalize_for_dino(frames_raw, device)

            # Get text embeddings (for reconstruction conditioning)
            text_embeds = model.get_text_embeds(caption_ids, caption_mask)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Task 1: Reconstruction loss
                loss_recon, loss_fine, loss_coarse = model(text_embeds, frames, latents)

                # Task 2: Captioning loss
                loss_caption = model.forward_captioning(frames, caption_ids, caption_mask, use_fine=True)

                # Combined loss
                loss = loss_recon + config['lambda_caption'] * loss_caption
                loss = loss / config['grad_accum']

            scaler.scale(loss).backward()

            running_loss += loss.item() * config['grad_accum']
            running_recon += loss_recon.item()
            running_caption += loss_caption.item()
            running_fine += loss_fine.item()
            running_coarse += loss_coarse.item()

            if (global_step + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            pbar.update(1)

            if global_step % config['log_every'] == 0:
                n = config['log_every']
                avg_loss = running_loss / n
                avg_recon = running_recon / n
                avg_caption = running_caption / n
                avg_fine = running_fine / n
                avg_coarse = running_coarse / n
                ratio = avg_coarse / (avg_fine + 1e-8)
                stats = base_dataset.stats
                success_rate = stats['success'] / max(stats['attempted'], 1) * 100

                pbar.set_postfix({
                    'recon': f'{avg_recon:.4f}',
                    'cap': f'{avg_caption:.2f}',
                    'ratio': f'{ratio:.3f}',
                    'suc%': f'{success_rate:.0f}',
                })

                if HAS_WANDB:
                    wandb.log({
                        'loss_total': avg_loss,
                        'loss_recon': avg_recon,
                        'loss_caption': avg_caption,
                        'loss_fine': avg_fine,
                        'loss_coarse': avg_coarse,
                        'ratio': ratio,
                        'caption_perplexity': np.exp(min(avg_caption, 10)),  # Cap to avoid overflow
                        'lr': scheduler.get_last_lr()[0],
                        'success_rate': success_rate,
                    }, step=global_step)

                running_loss = 0
                running_recon = 0
                running_caption = 0
                running_fine = 0
                running_coarse = 0

            # Eval: generate sample captions
            if global_step % config['eval_every'] == 0:
                model.eval()
                with torch.no_grad():
                    sample_frames = normalize_for_dino(batch['frames_raw'][:1], device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        gen_caption = model.generate_caption(sample_frames, tokenizer, max_new_tokens=30, temperature=0.7)
                    print(f"\n  GT: {batch['captions'][0][:80]}...")
                    print(f"  Gen: {gen_caption[0][:80]}...")
                model.train()

            if global_step % config['save_every'] == 0:
                path = output_dir / 'checkpoints' / f'step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'stats': base_dataset.stats,
                }, path)
                print(f"\nSaved: {path}")

            # Clear CUDA cache periodically to prevent memory buildup
            if global_step % 50 == 0:
                torch.cuda.empty_cache()

    pbar.close()
    dataset.stop()

    # Final save
    final_path = output_dir / 'checkpoints' / 'final.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': config,
        'stats': base_dataset.stats,
    }, final_path)

    print("\n" + "=" * 70)
    print("Multi-Task Training Complete!")
    print(f"Checkpoint: {final_path}")
    print("=" * 70)

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
