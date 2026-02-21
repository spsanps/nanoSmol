#!/usr/bin/env python3
"""
Visualization Script for Foveated VLM Captioning.

Creates:
1. Attention map visualizations comparing fine vs coarse queries
2. Caption quality comparisons
3. Animated attention sequences
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
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


def denormalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return frames * std + mean


def get_attention_maps(model, frames, device, use_fine=True):
    """Extract attention maps from the model.

    Returns attention maps of shape [B, T, num_heads, H, W] or similar.
    """
    model.eval()
    B, T, C, H, W = frames.shape

    with torch.no_grad():
        frames_flat = frames.view(B * T, C, H, W)

        # Get DINO features
        dino_features = model.encoder.dino(frames_flat)

        # Get patch tokens (excluding CLS)
        if hasattr(dino_features, 'last_hidden_state'):
            patch_tokens = dino_features.last_hidden_state[:, 1:]  # [B*T, N, D]
        else:
            patch_tokens = dino_features[:, 1:]

        N = patch_tokens.shape[1]
        D = patch_tokens.shape[2]

        # Reshape to [B, T, N, D]
        patch_tokens = patch_tokens.view(B, T, N, D)

        # Get query
        if use_fine:
            # Get LLM-generated query
            if hasattr(model, 'q_init'):
                q_init = model.q_init.expand(B, -1)
            else:
                q_init = model.encoder.q_init.expand(B, -1)

            # Simple pass through LLM projection
            if hasattr(model.encoder, 'query_input_proj'):
                query_embed = model.encoder.query_input_proj(q_init)
            else:
                query_embed = q_init
        else:
            # Use static query
            if hasattr(model, 'q_static'):
                q_static = model.q_static.expand(B, -1)
            else:
                q_static = model.encoder.q_static.expand(B, -1)

            if hasattr(model.encoder, 'query_input_proj'):
                query_embed = model.encoder.query_input_proj(q_static)
            else:
                query_embed = q_static

        # Compute attention: query vs patches
        # query_embed: [B, D]
        # patch_tokens: [B, T, N, D]
        query_embed = query_embed.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]

        # Simple dot-product attention
        attn_logits = (patch_tokens * query_embed).sum(-1)  # [B, T, N]
        attn_weights = F.softmax(attn_logits / np.sqrt(D), dim=-1)  # [B, T, N]

        # Reshape to spatial
        grid_size = int(np.sqrt(N))
        attn_maps = attn_weights.view(B, T, grid_size, grid_size)

        return attn_maps.cpu().numpy()


def visualize_sample(model, frames, caption, tokenizer, save_path, device):
    """Create comprehensive visualization for a single sample."""
    model.eval()

    frames_norm = normalize_frames(frames.unsqueeze(0).to(device))

    with torch.no_grad():
        # Generate captions
        caption_fine = model.generate_caption(
            frames_norm, tokenizer, max_new_tokens=50, use_fine=True
        )[0]
        caption_coarse = model.generate_caption(
            frames_norm, tokenizer, max_new_tokens=50, use_fine=False
        )[0]

        # Get attention maps
        attn_fine = get_attention_maps(model, frames_norm, device, use_fine=True)
        attn_coarse = get_attention_maps(model, frames_norm, device, use_fine=False)

    num_frames = frames.shape[0]
    fig = plt.figure(figsize=(20, 12))

    # Grid: 4 rows
    # Row 1: Original frames
    # Row 2: Fine attention overlays
    # Row 3: Coarse attention overlays
    # Row 4: Captions

    frames_vis = frames.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()

    for i in range(min(num_frames, 8)):
        # Original frame
        ax1 = plt.subplot(4, 8, i + 1)
        ax1.imshow(frames_vis[i])
        ax1.set_title(f'Frame {i+1}', fontsize=10)
        ax1.axis('off')

        # Fine attention overlay
        ax2 = plt.subplot(4, 8, 8 + i + 1)
        ax2.imshow(frames_vis[i])
        attn_upsampled = np.array(Image.fromarray((attn_fine[0, i] * 255).astype(np.uint8)).resize(
            (frames_vis[i].shape[1], frames_vis[i].shape[0]), Image.BILINEAR)) / 255.0
        ax2.imshow(attn_upsampled, cmap='hot', alpha=0.5)
        if i == 0:
            ax2.set_ylabel('Fine Query', fontsize=12)
        ax2.axis('off')

        # Coarse attention overlay
        ax3 = plt.subplot(4, 8, 16 + i + 1)
        ax3.imshow(frames_vis[i])
        attn_upsampled = np.array(Image.fromarray((attn_coarse[0, i] * 255).astype(np.uint8)).resize(
            (frames_vis[i].shape[1], frames_vis[i].shape[0]), Image.BILINEAR)) / 255.0
        ax3.imshow(attn_upsampled, cmap='hot', alpha=0.5)
        if i == 0:
            ax3.set_ylabel('Coarse Query', fontsize=12)
        ax3.axis('off')

    # Captions row
    ax_gt = plt.subplot(4, 3, 10)
    ax_gt.text(0.5, 0.5, f'Ground Truth:\n{caption}',
               ha='center', va='center', wrap=True, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax_gt.axis('off')

    ax_fine = plt.subplot(4, 3, 11)
    ax_fine.text(0.5, 0.5, f'Fine Query:\n{caption_fine}',
                 ha='center', va='center', wrap=True, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax_fine.axis('off')

    ax_coarse = plt.subplot(4, 3, 12)
    ax_coarse.text(0.5, 0.5, f'Coarse Query:\n{caption_coarse}',
                   ha='center', va='center', wrap=True, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax_coarse.axis('off')

    plt.suptitle('Foveated VLM: Fine vs Coarse Query Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'gt': caption,
        'fine': caption_fine,
        'coarse': caption_coarse,
    }


def create_attention_animation(model, frames, tokenizer, save_path, device):
    """Create animated GIF showing attention over time."""
    model.eval()

    frames_norm = normalize_frames(frames.unsqueeze(0).to(device))

    with torch.no_grad():
        attn_fine = get_attention_maps(model, frames_norm, device, use_fine=True)
        attn_coarse = get_attention_maps(model, frames_norm, device, use_fine=False)

    frames_vis = frames.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    num_frames = frames.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def update(frame_idx):
        for ax in axes:
            ax.clear()

        # Original frame
        axes[0].imshow(frames_vis[frame_idx])
        axes[0].set_title(f'Frame {frame_idx+1}/{num_frames}', fontsize=12)
        axes[0].axis('off')

        # Fine attention
        axes[1].imshow(frames_vis[frame_idx])
        attn_up = np.array(Image.fromarray((attn_fine[0, frame_idx] * 255).astype(np.uint8)).resize(
            (frames_vis[frame_idx].shape[1], frames_vis[frame_idx].shape[0]), Image.BILINEAR)) / 255.0
        axes[1].imshow(attn_up, cmap='hot', alpha=0.6)
        axes[1].set_title('Fine Query Attention', fontsize=12)
        axes[1].axis('off')

        # Coarse attention
        axes[2].imshow(frames_vis[frame_idx])
        attn_up = np.array(Image.fromarray((attn_coarse[0, frame_idx] * 255).astype(np.uint8)).resize(
            (frames_vis[frame_idx].shape[1], frames_vis[frame_idx].shape[0]), Image.BILINEAR)) / 255.0
        axes[2].imshow(attn_up, cmap='hot', alpha=0.6)
        axes[2].set_title('Coarse Query Attention', fontsize=12)
        axes[2].axis('off')

        return axes

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, blit=False)
    anim.save(save_path, writer='pillow', fps=2)
    plt.close()
    print(f"Animation saved to {save_path}")


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
        except Exception:
            continue


def main(
    checkpoint_path: str = None,
    output_dir: str = "outputs/visualizations",
    num_samples: int = 10,
    create_animations: bool = True,
):
    """Generate visualizations from trained model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load model
    print("Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate samples
    print(f"\nGenerating {num_samples} visualizations...")
    data_gen = webvid_generator()

    results = []
    for i in range(num_samples):
        try:
            frames, caption = next(data_gen)

            # Static visualization
            viz_path = output_dir / f"sample_{i+1:03d}.png"
            result = visualize_sample(model, frames, caption, tokenizer, viz_path, device)
            result['sample_id'] = i + 1
            results.append(result)
            print(f"  Sample {i+1}: {viz_path}")

            # Animation (first 3 samples only)
            if create_animations and i < 3:
                anim_path = output_dir / f"attention_anim_{i+1:03d}.gif"
                create_attention_animation(model, frames, tokenizer, anim_path, device)

        except StopIteration:
            print(f"Ran out of samples at {i+1}")
            break
        except Exception as e:
            print(f"Error on sample {i+1}: {e}")
            continue

    # Save results summary
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Visualizations complete!")
    print(f"Saved to: {output_dir}")
    print(f"Total samples: {len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--no_animations", action="store_true")
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        create_animations=not args.no_animations,
    )
