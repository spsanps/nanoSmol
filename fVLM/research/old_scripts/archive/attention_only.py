"""
Generate ONLY attention maps for diverse videos.
Simplified version with better error handling.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
import subprocess
import tempfile
import re
import time
from datasets import load_dataset
import imageio
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# Constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str):
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


def download_video(url, num_frames=16, frame_size=256, retry=3):
    """Download a video with retry logic."""
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                continue

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    cmd = [
                        'ffmpeg', '-i', temp_path,
                        '-vf', f'scale={frame_size}:{frame_size}:force_original_aspect_ratio=increase,crop={frame_size}:{frame_size}',
                        '-frames:v', str(num_frames * 3),
                        '-q:v', '2',
                        f'{tmpdir}/frame_%04d.jpg',
                        '-y', '-loglevel', 'error'
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=60)
                    if result.returncode != 0:
                        continue

                    frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                    if len(frame_files) < num_frames:
                        continue

                    indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                    frames = []
                    for idx in indices:
                        img = Image.open(frame_files[idx]).convert('RGB')
                        frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                        frames.append(frame)
                    return torch.stack(frames)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2)
                continue
            print(f"Download error after {retry} attempts: {e}")
    return None


def get_diverse_videos(num_videos=20, num_frames=16, frame_size=256):
    """Download diverse test videos with better error handling."""
    print(f"Downloading {num_videos} diverse test videos...")

    # Try with timeout increase
    import datasets
    datasets.config.HF_DATASETS_OFFLINE = False

    max_retries = 3
    for retry in range(max_retries):
        try:
            ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
            break
        except Exception as e:
            if retry < max_retries - 1:
                print(f"  Dataset load failed (attempt {retry+1}/{max_retries}), retrying...")
                time.sleep(5)
            else:
                raise e

    target_keywords = [
        'person', 'animal', 'nature', 'city', 'ocean',
        'dance', 'sport', 'car', 'food', 'abstract',
        'sky', 'building', 'water', 'tree', 'bird'
    ]

    videos = []
    category_counts = {}
    tried = 0
    max_tries = num_videos * 20  # More attempts

    for sample in ds:
        if len(videos) >= num_videos:
            break

        if tried > max_tries:
            print(f"  Reached max tries ({max_tries}), got {len(videos)} videos")
            break

        duration = parse_duration(sample.get('duration', ''))
        if duration < 8 or duration > 60:
            continue

        caption = sample.get('name', '').lower()
        if not caption:
            continue

        # Try to get diverse categories
        category = None
        for keyword in target_keywords:
            if keyword in caption:
                category = keyword
                break

        # Allow more per category
        if category and category_counts.get(category, 0) >= 3:
            continue

        tried += 1

        # Download with retry
        frames = download_video(sample['contentUrl'], num_frames, frame_size)
        if frames is not None:
            videos.append({
                'frames': frames,
                'caption': sample['name'],
                'video_id': sample.get('videoid', f'video_{len(videos)}'),
                'category': category or 'other'
            })
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
            print(f"  [{len(videos)}/{num_videos}] {category or 'other'}: {sample['name'][:50]}...")

    return videos


def normalize_for_dino(frames):
    """Normalize frames for DINO."""
    frames_norm = frames.float() / 255.0
    mean = IMAGENET_MEAN.to(frames_norm.device)
    std = IMAGENET_STD.to(frames_norm.device)
    return (frames_norm - mean) / std


@torch.no_grad()
def get_attention_maps(model, tokenizer, frames, caption, device):
    """Extract attention maps from both coarse and fine passes."""
    B = 1
    T = frames.shape[0]

    # Normalize frames
    raw_frames = normalize_for_dino(frames).unsqueeze(0).to(device)

    # Tokenize caption
    tokens = tokenizer(caption, max_length=64, padding='max_length',
                      truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    text_embeds = model.llm.model.embed_tokens(input_ids)
    N_text = text_embeds.shape[1]

    # Encode with DINO
    frames_flat = raw_frames.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)
    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    # === Pass 1: Coarse (with attention capture) ===
    q_static = model.q_static.expand(B, -1)
    attn_coarse_list = []
    z_coarse_list = []

    for t in range(T):
        # Manually compute attention (mimic query_attend_shallow)
        q_embed = model.encoder.query_input_proj(q_static)  # [B, D]
        q_embed = q_embed.unsqueeze(1)  # [B, 1, D]
        patches = all_caches[t]['patch_features']  # [B, N, D]

        # Attention computation
        attn_scores = torch.bmm(q_embed, patches.transpose(1, 2))  # [B, 1, N]
        attn_weights = F.softmax(attn_scores / (model.encoder.dino_dim ** 0.5), dim=-1)

        # Save attention
        attn_coarse_list.append(attn_weights.squeeze(0).squeeze(0))  # [N]

        # Compute attended features
        z_t = torch.bmm(attn_weights, patches).squeeze(1)  # [B, D]
        z_t = model.encoder.query_output_proj(z_t)  # [B, D_out]
        z_coarse_list.append(z_t)

    attn_coarse = torch.stack(attn_coarse_list, dim=0)  # [T, N]
    z_coarse = torch.stack(z_coarse_list, dim=1)  # [B, T, D]
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    # Get queries
    coarse_token = model.coarse_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state
    queries = model.llm_to_query(h_pass1[:, N_text + 1:])  # [B, T, D]

    # === Pass 2: Fine (with attention capture) ===
    q_init = model.q_init.expand(B, -1)
    shifted_q_list = [q_init] + [queries[:, t] for t in range(T-1)]
    shifted_q = torch.stack(shifted_q_list, dim=1)  # [B, T, D]

    attn_fine_list = []

    for t in range(T):
        q_t = shifted_q[:, t]  # [B, D]

        # Manually compute attention
        q_embed = model.encoder.query_input_proj(q_t)  # [B, D]
        q_embed = q_embed.unsqueeze(1)  # [B, 1, D]
        patches = all_caches[t]['patch_features']  # [B, N, D]

        # Attention computation
        attn_scores = torch.bmm(q_embed, patches.transpose(1, 2))  # [B, 1, N]
        attn_weights = F.softmax(attn_scores / (model.encoder.dino_dim ** 0.5), dim=-1)

        # Save attention
        attn_fine_list.append(attn_weights.squeeze(0).squeeze(0))  # [N]

    attn_fine = torch.stack(attn_fine_list, dim=0)  # [T, N]

    return attn_coarse.cpu(), attn_fine.cpu()


def create_attention_overlay(frame, attention, alpha=0.6):
    """Create heatmap overlay."""
    H, W = frame.shape[1], frame.shape[2]

    # Skip CLS token (first token) - only visualize patch attention
    attention_patches = attention[1:]  # Skip CLS
    grid_size = int(np.sqrt(attention_patches.shape[0]))
    attn_map = attention_patches.reshape(grid_size, grid_size).numpy()
    attn_map = Image.fromarray((attn_map * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    attn_map = np.array(attn_map) / 255.0
    cmap = plt.cm.jet
    heatmap = cmap(attn_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    frame_np = frame.permute(1, 2, 0).numpy()
    blended = (alpha * heatmap + (1 - alpha) * frame_np).astype(np.uint8)
    return torch.from_numpy(blended).permute(2, 0, 1)


def create_attention_grid(frames, attn_coarse, attn_fine, caption, output_path):
    """Create attention comparison grid."""
    T = min(16, frames.shape[0])
    fig, axes = plt.subplots(T, 4, figsize=(16, 4 * T))
    if T == 1:
        axes = axes.reshape(1, -1)

    for t in range(T):
        # Frame
        axes[t, 0].imshow(frames[t].permute(1, 2, 0).numpy())
        axes[t, 0].set_title(f'Frame t={t}')
        axes[t, 0].axis('off')

        # Coarse
        coarse_overlay = create_attention_overlay(frames[t], attn_coarse[t])
        axes[t, 1].imshow(coarse_overlay.permute(1, 2, 0).numpy())
        axes[t, 1].set_title(f'Coarse t={t}')
        axes[t, 1].axis('off')

        # Fine
        fine_overlay = create_attention_overlay(frames[t], attn_fine[t])
        axes[t, 2].imshow(fine_overlay.permute(1, 2, 0).numpy())
        axes[t, 2].set_title(f'Fine t={t}')
        axes[t, 2].axis('off')

        # Difference (skip CLS token)
        diff = attn_fine[t] - attn_coarse[t]
        diff_patches = diff[1:]  # Skip CLS
        grid_size = int(np.sqrt(diff_patches.shape[0]))
        diff_map = diff_patches.reshape(grid_size, grid_size).numpy()
        im = axes[t, 3].imshow(diff_map, cmap='RdBu', vmin=-diff.abs().max(), vmax=diff.abs().max())
        axes[t, 3].set_title(f'Difference t={t}')
        axes[t, 3].axis('off')

    plt.suptitle(f'"{caption[:80]}"', fontsize=10, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_videos = 20

    output_dir = Path('outputs/attention_full')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    checkpoint_path = Path('outputs/phase2/checkpoints/final.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    model_cfg = config['model']

    model = FoveatedVideoModel(
        dino_model=model_cfg['dino_model'],
        llm_model=model_cfg['llm_model'],
        dino_dim=model_cfg['dino_dim'],
        llm_dim=model_cfg['llm_dim'],
        query_dim=model_cfg['query_dim'],
        lambda_coarse=model_cfg['lambda_coarse'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Download videos
    print()
    videos = get_diverse_videos(num_videos, num_frames)
    print(f"\nGot {len(videos)} videos\n")

    print("=" * 80)
    print("GENERATING ATTENTION MAPS")
    print("=" * 80)

    for i, video in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {video['category'].upper()}: {video['caption'][:60]}")

        frames = video['frames']
        caption = video['caption']

        # Get attention maps
        attn_coarse, attn_fine = get_attention_maps(
            model, tokenizer, frames, caption, device
        )

        # Save visualization
        create_attention_grid(
            frames, attn_coarse, attn_fine, caption,
            output_dir / f'attention_grid_{i:02d}.png'
        )

    print(f"\n{'=' * 80}")
    print(f"Saved {len(videos)} attention visualizations to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
