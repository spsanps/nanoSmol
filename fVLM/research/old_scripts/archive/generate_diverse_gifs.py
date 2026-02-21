#!/usr/bin/env python3
"""
Generate diverse attention GIFs with fine vs coarse comparison.

Creates 4-panel GIFs showing:
1. Original frame
2. Coarse attention (static query)
3. Fine attention (dynamic query)
4. Difference (Fine - Coarse)
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imageio
import requests
import subprocess
import tempfile
import re
from datasets import load_dataset

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


def download_video(url: str, timeout: int = 20) -> bytes:
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            content = b''
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 100 * 1024 * 1024:
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
                # Extract more frames than needed for smooth sampling
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 4),  # Extract extra frames
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

                # Sample evenly across the video
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)
                return torch.stack(frames)
    except Exception as e:
        return None


def normalize_for_dino(frames, device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames.to(device) - mean) / std


@torch.no_grad()
def extract_attention_maps(model, frames, device):
    """
    Extract attention maps using simple dot-product attention.
    Works with both shallow and deep query modes.
    """
    model.eval()
    B, T, C, H, W = 1, frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]

    frames_norm = normalize_for_dino(frames, device).unsqueeze(0)  # [1, T, C, H, W]
    frames_flat = frames_norm.view(B * T, C, H, W)

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

    # Get coarse query (static)
    q_static = model.q_static.expand(B, -1)
    if hasattr(model.encoder, 'query_input_proj'):
        q_static_proj = model.encoder.query_input_proj(q_static)
    else:
        q_static_proj = q_static

    # Get fine query (dynamic - use q_init as approximation)
    if hasattr(model, 'q_init'):
        q_init = model.q_init.expand(B, -1)
    else:
        q_init = model.encoder.q_init.expand(B, -1)

    if hasattr(model.encoder, 'query_input_proj'):
        q_init_proj = model.encoder.query_input_proj(q_init)
    else:
        q_init_proj = q_init

    # Compute attention for each frame
    coarse_attn_list = []
    fine_attn_list = []

    for t in range(T):
        pf = patch_tokens[:, t]  # [B, N, D]

        # Coarse attention: q_static vs patches
        q_c = q_static_proj.unsqueeze(1)  # [B, 1, D]
        attn_logits_c = torch.bmm(q_c, pf.transpose(1, 2)) / np.sqrt(D)  # [B, 1, N]
        attn_c = F.softmax(attn_logits_c, dim=-1).squeeze(1)  # [B, N]
        coarse_attn_list.append(attn_c)

        # Fine attention: q_init vs patches
        q_f = q_init_proj.unsqueeze(1)  # [B, 1, D]
        attn_logits_f = torch.bmm(q_f, pf.transpose(1, 2)) / np.sqrt(D)  # [B, 1, N]
        attn_f = F.softmax(attn_logits_f, dim=-1).squeeze(1)  # [B, N]
        fine_attn_list.append(attn_f)

    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)  # [T, N]
    fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)  # [T, N]

    grid_size = int(np.sqrt(N))

    return coarse_attn.reshape(T, grid_size, grid_size).cpu(), fine_attn.reshape(T, grid_size, grid_size).cpu()


def create_attention_overlay(frame, attn_map, alpha=0.5):
    H, W = frame.shape[1], frame.shape[2]
    attn_up = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    heatmap = (plt.cm.hot(attn_norm)[:, :, :3] * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

    return ((1 - alpha) * frame_np + alpha * heatmap).astype(np.uint8)


def create_diff_overlay(frame, diff_map):
    H, W = frame.shape[1], frame.shape[2]
    diff_up = F.interpolate(
        diff_map.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # Normalize diff to [-1, 1] then to [0, 1] for colormap
    diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
    diff_norm = (diff_up / diff_max + 1) / 2  # Now in [0, 1], 0.5 = no diff

    # Blue = coarse stronger, Red = fine stronger
    diff_overlay = (plt.cm.RdBu_r(diff_norm)[:, :, :3] * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

    # Blend with original
    alpha = 0.6
    return ((1 - alpha) * frame_np + alpha * diff_overlay).astype(np.uint8)


def create_comparison_gif(frames, coarse_attn, fine_attn, output_path, caption="", fps=8):
    """Create 4-panel comparison GIF with original, coarse, fine, and diff."""
    T = frames.shape[0]
    gif_frames = []

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = font

    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        frame_np = frame.permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

        coarse_overlay = create_attention_overlay(frame, coarse_attn[t])
        fine_overlay = create_attention_overlay(frame, fine_attn[t])

        # Difference map
        diff = fine_attn[t] - coarse_attn[t]
        diff_overlay = create_diff_overlay(frame, diff)

        # Create 2x2 layout
        header_h = 35
        footer_h = 25
        total_h = H * 2 + header_h + footer_h
        total_w = W * 2

        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        combined[:] = [40, 40, 40]  # Dark gray background

        # Place images
        combined[header_h:header_h+H, :W] = frame_np
        combined[header_h:header_h+H, W:] = coarse_overlay
        combined[header_h+H:header_h+2*H, :W] = fine_overlay
        combined[header_h+H:header_h+2*H, W:] = diff_overlay

        # Add labels
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)

        # Header with frame number
        draw.text((10, 8), f"Frame {t+1}/{T}", fill=(255, 255, 255), font=font)

        # Panel labels
        draw.text((W//2 - 30, header_h + 5), "Original", fill=(255, 255, 255), font=small_font)
        draw.text((W + W//2 - 40, header_h + 5), "Coarse (static)", fill=(100, 150, 255), font=small_font)
        draw.text((W//2 - 40, header_h + H + 5), "Fine (dynamic)", fill=(255, 150, 100), font=small_font)
        draw.text((W + W//2 - 50, header_h + H + 5), "Diff (Fine-Coarse)", fill=(150, 255, 150), font=small_font)

        # Footer with caption (truncated)
        if caption:
            cap_short = caption[:80] + "..." if len(caption) > 80 else caption
            draw.text((10, total_h - footer_h + 5), f"GT: {cap_short}", fill=(200, 200, 200), font=small_font)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def webvid_generator(num_frames=16, frame_size=256, min_duration=8, max_duration=30):
    """Stream longer videos from WebVid."""
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    for sample in ds:
        try:
            duration = parse_duration(sample.get('duration', ''))
            if duration < min_duration or duration > max_duration:
                continue

            url = sample.get('contentUrl')
            caption = sample.get('name', '')
            if not url or not caption:
                continue

            video_bytes = download_video(url)
            if video_bytes is None:
                continue

            frames = extract_frames(video_bytes, num_frames, frame_size)
            if frames is None:
                continue

            yield frames, caption, duration

        except Exception:
            continue


def main(
    checkpoint_path: str = None,
    output_dir: str = "outputs/diverse_gifs_10k",
    num_samples: int = 64,
    num_frames: int = 16,
    min_duration: int = 8,
    fps: int = 6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"GENERATING DIVERSE ATTENTION GIFS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"Frames per video: {num_frames}")
    print(f"Min duration: {min_duration}s")
    print(f"FPS: {fps}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 'unknown')
        print(f"Loaded checkpoint from step {step}")
    else:
        print("WARNING: No checkpoint provided, using random weights!")

    model.eval()

    # Generate GIFs
    print(f"\nGenerating {num_samples} GIFs...")
    print("(Looking for videos {}-30s long)".format(min_duration))

    data_gen = webvid_generator(
        num_frames=num_frames,
        frame_size=256,
        min_duration=min_duration,
        max_duration=30
    )

    completed = 0
    attempted = 0

    while completed < num_samples:
        try:
            frames, caption, duration = next(data_gen)
            attempted += 1

            # Extract attention maps
            coarse_attn, fine_attn = extract_attention_maps(model, frames, device)

            # Create GIF
            gif_path = output_dir / f"attn_{completed+1:03d}.gif"
            create_comparison_gif(frames, coarse_attn, fine_attn, gif_path, caption, fps=fps)

            completed += 1
            print(f"  [{completed:3d}/{num_samples}] {duration}s video: {gif_path.name} - '{caption[:50]}...'", flush=True)

        except StopIteration:
            print(f"Ran out of videos at {completed}")
            break
        except Exception as e:
            if attempted % 10 == 0:
                print(f"  (Attempted {attempted}, completed {completed}, error: {str(e)[:50]})")
            continue

    print("\n" + "=" * 70)
    print(f"COMPLETE!")
    print(f"Generated: {completed} GIFs")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="/mnt/c/Users/sanps/Desktop/Projects/dino/nanoSmolLM/outputs/captioning_scaled/checkpoints/step_010000.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/diverse_gifs_10k")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--min_duration", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        min_duration=args.min_duration,
        fps=args.fps,
    )
