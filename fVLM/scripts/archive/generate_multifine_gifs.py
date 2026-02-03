#!/usr/bin/env python3
"""
Generate paper-quality GIFs for multi-fine model showing progressive attention refinement.

Layout (2x3 grid):
- Top row: Original | Coarse (static) | Difference (coarse vs fine₂)
- Bottom row: Fine iter₁ | Fine iter₂ (EMPHASIZED) | Iteration diff (iter₁ vs iter₂)

Uses joint multi-fine checkpoint for best results.
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
from transformers import AutoTokenizer

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


def normalize_for_dino(frames, device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames.to(device) - mean) / std


@torch.no_grad()
def extract_multifine_attention(model, frames, device, fine_iterations=2):
    """
    Extract attention maps for coarse and multiple fine iterations.

    Returns:
        coarse_attn: [T, grid, grid] - static query attention
        fine_attn_list: list of [T, grid, grid] - one per fine iteration
    """
    model.eval()
    B = 1
    T = frames.shape[0]

    frames_norm = normalize_for_dino(frames, device).unsqueeze(0)
    frames_flat = frames_norm.view(B * T, 3, 256, 256)

    # Get DINO features
    dino_out = model.encoder.dino(frames_flat)
    if hasattr(dino_out, 'last_hidden_state'):
        all_patch_tokens = dino_out.last_hidden_state
    else:
        all_patch_tokens = dino_out

    patch_tokens = all_patch_tokens[:, 1:]  # Remove CLS
    N, D = patch_tokens.shape[1], patch_tokens.shape[2]
    patch_tokens = patch_tokens.view(B, T, N, D)

    grid_size = int(np.sqrt(N))

    # === COARSE: Same static query for all frames ===
    q_static = model.q_static.expand(B, -1)
    if hasattr(model.encoder, 'query_input_proj'):
        q_static_proj = model.encoder.query_input_proj(q_static)
    else:
        q_static_proj = q_static

    coarse_attn_list = []
    coarse_features = []
    for t in range(T):
        pf = patch_tokens[:, t]
        q = q_static_proj.unsqueeze(1)
        attn_logits = torch.bmm(q, pf.transpose(1, 2)) / np.sqrt(D)
        attn = F.softmax(attn_logits, dim=-1).squeeze(1)
        coarse_attn_list.append(attn)
        z_t = torch.bmm(attn.unsqueeze(1), pf).squeeze(1)
        coarse_features.append(z_t)

    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)

    # === FINE ITERATIONS: Autoregressive ===
    fine_attn_iterations = []

    # Start with coarse features for first iteration query generation
    prev_features = coarse_features

    for iteration in range(fine_iterations):
        fine_attn_list = []
        fine_features = []

        q_current = model.q_init.expand(B, -1)

        for t in range(T):
            pf = patch_tokens[:, t]

            if hasattr(model.encoder, 'query_input_proj'):
                q_proj = model.encoder.query_input_proj(q_current)
            else:
                q_proj = q_current

            q = q_proj.unsqueeze(1)
            attn_logits = torch.bmm(q, pf.transpose(1, 2)) / np.sqrt(D)
            attn = F.softmax(attn_logits, dim=-1).squeeze(1)
            fine_attn_list.append(attn)

            z_t = torch.bmm(attn.unsqueeze(1), pf).squeeze(1)
            fine_features.append(z_t)

            # Generate next query from features seen so far
            if t < T - 1:
                # Use features from current iteration (autoregressive within iteration)
                z_so_far = torch.stack(fine_features, dim=1)
                z_llm = model.dino_to_llm(z_so_far)
                z_llm = z_llm / (z_llm.std() + 1e-6) * model.visual_scale

                no_text = model.no_text_token.expand(B, -1, -1)
                fine_token = model.fine_token.expand(B, -1, -1)
                seq = torch.cat([no_text, fine_token, z_llm], dim=1)

                outputs = model.llm.model(inputs_embeds=seq)
                h = outputs.last_hidden_state
                h_last = h[:, -1, :]
                q_current = model.llm_to_query(h_last)

        fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)
        fine_attn_iterations.append(fine_attn.reshape(T, grid_size, grid_size).cpu())

        # Use this iteration's features as basis for next iteration
        prev_features = fine_features

    return (coarse_attn.reshape(T, grid_size, grid_size).cpu(),
            fine_attn_iterations)


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


def create_diff_overlay(frame, diff_map, colormap='RdBu_r'):
    H, W = frame.shape[1], frame.shape[2]
    diff_up = F.interpolate(
        diff_map.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
    diff_norm = (diff_up / diff_max + 1) / 2

    cmap = plt.cm.RdBu_r if colormap == 'RdBu_r' else plt.cm.PiYG
    diff_overlay = (cmap(diff_norm)[:, :, :3] * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

    alpha = 0.6
    return ((1 - alpha) * frame_np + alpha * diff_overlay).astype(np.uint8)


def create_multifine_gif(frames, coarse_attn, fine_attn_list, output_path,
                         gt_caption="", fine_caption="", coarse_caption="", fps=6):
    """
    Create paper-quality 6-panel GIF showing multi-fine progression.

    Layout:
    - Row 1: Original | Coarse | Diff (coarse vs fine₂)
    - Row 2: Fine₁ | Fine₂ (emphasized) | Iter diff (fine₁ vs fine₂)
    """
    T = frames.shape[0]
    gif_frames = []

    fine1_attn = fine_attn_list[0]
    fine2_attn = fine_attn_list[1] if len(fine_attn_list) > 1 else fine1_attn

    # Try to load fonts
    try:
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_regular = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_bold = ImageFont.load_default()
        font_regular = font_bold
        font_small = font_bold

    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        frame_np = frame.permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

        coarse_overlay = create_attention_overlay(frame, coarse_attn[t])
        fine1_overlay = create_attention_overlay(frame, fine1_attn[t])
        fine2_overlay = create_attention_overlay(frame, fine2_attn[t])

        # Differences
        diff_coarse_fine2 = fine2_attn[t] - coarse_attn[t]  # Red = fine2 attends more
        diff_fine1_fine2 = fine2_attn[t] - fine1_attn[t]    # Red = fine2 attends more than fine1

        diff_coarse_overlay = create_diff_overlay(frame, diff_coarse_fine2, 'RdBu_r')
        diff_iter_overlay = create_diff_overlay(frame, diff_fine1_fine2, 'PiYG')

        # Layout dimensions
        border = 3
        header_h = 30
        caption_h = 55
        gap = 3

        panel_w = W
        panel_h = H

        total_w = panel_w * 3 + gap * 2
        total_h = header_h + panel_h * 2 + gap + caption_h

        # Create canvas
        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        combined[:] = [25, 25, 30]

        # Row 1: Original | Coarse | Diff
        y1 = header_h
        combined[y1:y1+H, 0:W] = frame_np
        combined[y1:y1+H, W+gap:2*W+gap] = coarse_overlay
        combined[y1:y1+H, 2*W+2*gap:] = diff_coarse_overlay

        # Row 2: Fine1 | Fine2 (emphasized) | Iter diff
        y2 = header_h + H + gap
        combined[y2:y2+H, 0:W] = fine1_overlay

        # Emphasize Fine2 with border
        x_fine2 = W + gap
        combined[y2-border:y2+H+border, x_fine2-border:x_fine2+W+border] = [255, 140, 60]
        combined[y2:y2+H, x_fine2:x_fine2+W] = fine2_overlay

        combined[y2:y2+H, 2*W+2*gap:] = diff_iter_overlay

        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)

        # Header
        draw.text((10, 6), f"Frame {t+1}/{T}", fill=(255, 255, 255), font=font_bold)
        draw.text((total_w//2 - 80, 8), "Multi-Fine Attention Progression",
                  fill=(180, 180, 160), font=font_regular)

        # Panel labels - Row 1
        draw.text((3, y1 + 2), "Original", fill=(200, 200, 200), font=font_small)
        draw.text((W + gap + 3, y1 + 2), "Coarse (static)", fill=(120, 150, 200), font=font_small)
        draw.text((2*W + 2*gap + 3, y1 + 2), "Diff (red=fine2)", fill=(200, 150, 150), font=font_small)

        # Panel labels - Row 2
        draw.text((3, y2 + 2), "Fine iter1", fill=(200, 180, 120), font=font_small)
        draw.text((W + gap + 3, y2 + 2), "FINE iter2", fill=(255, 180, 80), font=font_bold)
        draw.text((2*W + 2*gap + 3, y2 + 2), "Iter diff", fill=(150, 200, 150), font=font_small)

        # Captions at bottom
        y_cap = y2 + H + 5
        if gt_caption:
            gt_short = gt_caption[:80] + "..." if len(gt_caption) > 80 else gt_caption
            draw.text((5, y_cap), f"GT: {gt_short}", fill=(180, 180, 180), font=font_small)
        if fine_caption:
            fine_short = fine_caption[:80] + "..." if len(fine_caption) > 80 else fine_caption
            draw.text((5, y_cap + 16), f"Fine: {fine_short}", fill=(255, 180, 100), font=font_small)
        if coarse_caption:
            coarse_short = coarse_caption[:80] + "..." if len(coarse_caption) > 80 else coarse_caption
            draw.text((5, y_cap + 32), f"Coarse: {coarse_short}", fill=(140, 170, 220), font=font_small)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def local_data_generator(data_dir="data/webvid/frames", num_frames=16):
    """Load frames from local .pt files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")

    frame_files = sorted(data_path.glob("*.pt"))
    print(f"Found {len(frame_files)} local frame files")

    for frame_file in frame_files:
        try:
            frames = torch.load(frame_file, weights_only=True)
            # Frames are [T, 3, 256, 256] tensors
            if frames.shape[0] >= num_frames:
                # Subsample if needed
                indices = torch.linspace(0, frames.shape[0]-1, num_frames).long()
                frames = frames[indices]
            # Use filename as caption placeholder
            video_id = frame_file.stem
            caption = f"Video {video_id}"
            yield frames, caption, 10  # duration placeholder
        except Exception as e:
            print(f"Error loading {frame_file}: {e}")
            continue


def webvid_generator(num_frames=16, frame_size=256, min_duration=8, max_duration=30):
    """Stream from WebVid-10M (fallback if local data not available)."""
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
        except:
            continue


def main(
    checkpoint_path: str = None,
    output_dir: str = "outputs/paper_gifs_multifine",
    num_samples: int = 64,
    num_frames: int = 16,
    fine_iterations: int = 2,
    min_duration: int = 8,
    fps: int = 6,
    use_local: bool = True,
    local_data_dir: str = "data/webvid/frames",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-FINE GIFS: Progressive Attention Visualization")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"Fine iterations: {fine_iterations}")
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
        print(f"Loaded from step {checkpoint.get('step', 'unknown')}")

    model.eval()

    # Load tokenizer for caption generation
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nGenerating {num_samples} multi-fine GIFs...")

    if use_local:
        print(f"Using local data from: {local_data_dir}")
        data_gen = local_data_generator(data_dir=local_data_dir, num_frames=num_frames)
    else:
        print("Streaming from WebVid-10M...")
        data_gen = webvid_generator(num_frames=num_frames, min_duration=min_duration)

    completed = 0
    for frames, gt_caption, duration in data_gen:
        if completed >= num_samples:
            break

        try:
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            # Extract multi-fine attention maps
            coarse_attn, fine_attn_list = extract_multifine_attention(
                model, frames, device, fine_iterations=fine_iterations
            )

            # Generate captions (use final fine iteration)
            fine_caption = model.generate_caption(frames_norm, tokenizer, max_new_tokens=30, use_fine=True)[0]
            coarse_caption = model.generate_caption(frames_norm, tokenizer, max_new_tokens=30, use_fine=False)[0]

            # Create GIF
            gif_path = output_dir / f"multifine_{completed+1:03d}.gif"
            create_multifine_gif(
                frames, coarse_attn, fine_attn_list, gif_path,
                gt_caption=gt_caption,
                fine_caption=fine_caption,
                coarse_caption=coarse_caption,
                fps=fps
            )

            completed += 1
            print(f"\n[{completed:3d}/{num_samples}] {duration}s video -> {gif_path.name}")
            print(f"  GT: {gt_caption[:70]}...")
            print(f"  Fine: {fine_caption[:70]}...")
            print(f"  Coarse: {coarse_caption[:70]}...")

        except Exception as e:
            print(f"  Error: {str(e)[:60]}")
            continue

    print("\n" + "=" * 70)
    print(f"COMPLETE! Generated {completed} multi-fine GIFs")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/joint_multifine_8h/checkpoints/latest.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/paper_gifs_multifine")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--fine_iterations", type=int, default=2)
    parser.add_argument("--min_duration", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--use_local", action="store_true", default=True,
                        help="Use local data instead of streaming")
    parser.add_argument("--no_local", dest="use_local", action="store_false",
                        help="Stream from WebVid-10M instead of using local data")
    parser.add_argument("--local_data_dir", type=str, default="data/webvid/frames",
                        help="Directory containing local .pt frame files")
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        fine_iterations=args.fine_iterations,
        min_duration=args.min_duration,
        fps=args.fps,
        use_local=args.use_local,
        local_data_dir=args.local_data_dir,
    )
