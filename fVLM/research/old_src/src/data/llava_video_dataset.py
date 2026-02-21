"""
LLaVA-Video dataset for Phase 2 training.

Features:
- Text captions for guided attention
- Action-rich videos (0-30s academic subset)
- Uniform temporal sampling for consistent FPS
- On-the-fly VAE latent computation (streaming mode)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import json
import decord
import numpy as np
from transformers import AutoTokenizer

# ImageNet normalization for DINO
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class LLaVAVideoDataset(Dataset):
    """
    LLaVA-Video-178K dataset for text-conditioned video prediction.

    Each sample contains:
    - frames: [T, 3, H, W] normalized for DINO
    - vae_latents: [T, 4, 32, 32] computed on-the-fly or precomputed
    - text_input_ids: [max_len] tokenized caption
    - text_attention_mask: [max_len]
    """

    def __init__(
        self,
        video_dir,
        caption_json,
        num_frames=32,
        frame_size=256,
        max_text_tokens=64,
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        min_duration=5,
        max_duration=30,
        latent_dir=None,  # Optional: if None, compute on-the-fly
        vae_model=None,   # Shared VAE for on-the-fly computation
        vae_device='cuda',
    ):
        self.video_dir = Path(video_dir)
        self.latent_dir = Path(latent_dir) if latent_dir else None
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_text_tokens = max_text_tokens
        self.vae_model = vae_model
        self.vae_device = vae_device

        # Load captions
        with open(caption_json) as f:
            data = json.load(f)

        # Build sample list: filter by duration, check video exists
        self.samples = []
        for item in data:
            video_id = item['video']
            video_path = self.video_dir / video_id

            if not video_path.exists():
                continue

            # If using precomputed latents, check they exist
            if self.latent_dir:
                latent_path = self.latent_dir / f"{Path(video_id).stem}.pt"
                if not latent_path.exists():
                    continue
            else:
                latent_path = None

            # Check duration (skip filter if duration not in metadata)
            duration = item.get('duration')
            if duration is not None:
                if duration < min_duration or duration > max_duration:
                    continue

            # Extract caption from conversations (GPT response) or direct caption field
            caption = item.get('caption', '')
            if not caption and 'conversations' in item:
                # LLaVA format: conversations[1] is GPT response
                for conv in item['conversations']:
                    if conv.get('from') == 'gpt':
                        caption = conv.get('value', '')
                        break

            self.samples.append({
                'video_id': video_id,
                'video_path': video_path,
                'latent_path': latent_path,
                'caption': caption,
                'duration': duration,
            })

        print(f"Loaded {len(self.samples)} videos from LLaVA-Video")
        if self.latent_dir:
            print(f"  Mode: Precomputed latents from {self.latent_dir}")
        else:
            print(f"  Mode: On-the-fly VAE computation")

        # Tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up decord
        decord.bridge.set_bridge('torch')

    def __len__(self):
        return len(self.samples)

    def sample_frames_uniform(self, video_path, num_frames):
        """
        Uniformly sample frames from video.

        Ensures consistent temporal spacing regardless of video length.
        """
        try:
            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)

            if total_frames < num_frames:
                # Video too short, repeat last frame
                indices = list(range(total_frames))
                indices += [total_frames - 1] * (num_frames - total_frames)
            else:
                # Uniform sampling
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = vr.get_batch(indices)  # [T, H, W, C] uint8
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

            # Resize to target size
            if frames.shape[2] != self.frame_size or frames.shape[3] != self.frame_size:
                frames = F.interpolate(
                    frames.float(),
                    size=(self.frame_size, self.frame_size),
                    mode='bilinear',
                    align_corners=False
                ).byte()

            return frames
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return black frames as fallback
            return torch.zeros(num_frames, 3, self.frame_size, self.frame_size, dtype=torch.uint8)

    def normalize_for_dino(self, frames):
        """Normalize frames for DINO (ImageNet stats)."""
        # frames: [T, 3, H, W] uint8
        frames_norm = frames.float() / 255.0
        frames_norm = (frames_norm - IMAGENET_MEAN) / IMAGENET_STD
        return frames_norm

    def compute_vae_latents(self, frames):
        """Compute VAE latents on-the-fly for given frames."""
        # frames: [T, 3, H, W] uint8
        if self.vae_model is None:
            raise ValueError("VAE model required for on-the-fly latent computation")

        # Normalize for VAE: [0, 255] -> [-1, 1]
        frames_vae = frames.float().to(self.vae_device) / 255.0 * 2 - 1

        with torch.no_grad():
            latents = []
            chunk_size = 4  # Process in chunks to avoid OOM
            for i in range(0, frames_vae.shape[0], chunk_size):
                batch = frames_vae[i:i+chunk_size]
                latent = self.vae_model.encode(batch).latent_dist.sample() * 0.18215
                latents.append(latent.cpu())
            latents = torch.cat(latents, dim=0)

        return latents  # [T, 4, 32, 32]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load frames
        frames = self.sample_frames_uniform(sample['video_path'], self.num_frames)
        frames_norm = self.normalize_for_dino(frames)

        # Get VAE latents (precomputed only - streaming defers to training loop)
        if sample['latent_path'] is not None:
            # Load precomputed latents
            vae_latents = torch.load(sample['latent_path'], weights_only=True)
            # Handle latent length mismatch
            if vae_latents.shape[0] != self.num_frames:
                vae_latents = F.interpolate(
                    vae_latents.unsqueeze(0),
                    size=(self.num_frames, 32, 32),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)
        elif self.vae_model is None:
            # Streaming mode: return raw frames, VAE computed in training loop
            vae_latents = frames  # Raw uint8 frames for VAE (computed later)
        else:
            # Compute on-the-fly (only if VAE passed and num_workers=0)
            vae_latents = self.compute_vae_latents(frames)

        # Tokenize caption
        tokens = self.tokenizer(
            sample['caption'],
            max_length=self.max_text_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'frames': frames_norm,  # [T, 3, 256, 256]
            'vae_latents': vae_latents,  # [T, 4, 32, 32] or [T, 3, H, W] raw
            'frames_raw': frames,  # [T, 3, H, W] uint8 for VAE
            'text_input_ids': tokens['input_ids'].squeeze(0),  # [max_len]
            'text_attention_mask': tokens['attention_mask'].squeeze(0),  # [max_len]
            'caption': sample['caption'],  # For logging
            'video_id': sample['video_id'],
        }


def collate_fn(batch):
    """Custom collate to handle variable-length sequences."""
    return {
        'frames': torch.stack([item['frames'] for item in batch]),
        'vae_latents': torch.stack([item['vae_latents'] for item in batch]),
        'frames_raw': torch.stack([item['frames_raw'] for item in batch]),
        'text_input_ids': torch.stack([item['text_input_ids'] for item in batch]),
        'text_attention_mask': torch.stack([item['text_attention_mask'] for item in batch]),
        'captions': [item['caption'] for item in batch],
        'video_ids': [item['video_id'] for item in batch],
    }
