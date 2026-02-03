#!/usr/bin/env python3
"""
Generate attention GIFs showing TRUE autoregressive fine vs static coarse.

THE THESIS: Fine queries are generated autoregressively from FINE features.
- Coarse: Same q_static for all frames (parallel, no temporal info)
- Fine: q_t depends on LLM seeing z_1...z_{t-1} (sequential, temporal info)

This script generates GIFs showing:
1. Original frames
2. Coarse attention (static query - same for all frames)
3. Fine attention (autoregressive - each query depends on previous fine features)
4. Difference (Fine - Coarse)
5. Generated captions from both paths
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
def extract_autoregressive_attention(model, frames, device):
    """
    Extract attention maps using TRUE autoregressive fine generation.

    Coarse: Same q_static for all frames
    Fine: q_t depends on LLM seeing fine features z_1...z_{t-1}
    """
    model.eval()
    B = 1
    T = frames.shape[0]

    frames_norm = normalize_for_dino(frames, device).unsqueeze(0)
    frames_flat = frames_norm.view(B * T, 3, 256, 256)

    # Get DINO features (we'll reuse these)
    dino_out = model.encoder.dino(frames_flat)
    if hasattr(dino_out, 'last_hidden_state'):
        all_patch_tokens = dino_out.last_hidden_state  # [B*T, N+1, D]
    else:
        all_patch_tokens = dino_out

    patch_tokens = all_patch_tokens[:, 1:]  # Remove CLS, [B*T, N, D]
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
    for t in range(T):
        pf = patch_tokens[:, t]  # [B, N, D]
        q = q_static_proj.unsqueeze(1)  # [B, 1, D]
        attn_logits = torch.bmm(q, pf.transpose(1, 2)) / np.sqrt(D)
        attn = F.softmax(attn_logits, dim=-1).squeeze(1)  # [B, N]
        coarse_attn_list.append(attn)

    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)  # [T, N]

    # === FINE: Autoregressive - each query depends on previous fine features ===
    # This is the KEY difference: queries are generated from FINE features, not coarse

    fine_attn_list = []
    z_fine_list = []

    # Start with q_init for first frame
    q_current = model.q_init.expand(B, -1)  # [B, query_dim]

    for t in range(T):
        pf = patch_tokens[:, t]  # [B, N, D]

        # Project current query
        if hasattr(model.encoder, 'query_input_proj'):
            q_proj = model.encoder.query_input_proj(q_current)
        else:
            q_proj = q_current

        # Compute attention
        q = q_proj.unsqueeze(1)  # [B, 1, D]
        attn_logits = torch.bmm(q, pf.transpose(1, 2)) / np.sqrt(D)
        attn = F.softmax(attn_logits, dim=-1).squeeze(1)  # [B, N]
        fine_attn_list.append(attn)

        # Extract feature using attention
        z_t = torch.bmm(attn.unsqueeze(1), pf).squeeze(1)  # [B, D]
        z_fine_list.append(z_t)

        # Generate next query from LLM based on fine features so far
        if t < T - 1:
            z_so_far = torch.stack(z_fine_list, dim=1)  # [B, t+1, D]
            z_llm = model.dino_to_llm(z_so_far)  # [B, t+1, llm_dim]
            z_llm = z_llm / (z_llm.std() + 1e-6) * model.visual_scale

            # Build sequence for LLM
            no_text = model.no_text_token.expand(B, -1, -1)
            fine_token = model.fine_token.expand(B, -1, -1)
            seq = torch.cat([no_text, fine_token, z_llm], dim=1)

            # Get next query from LLM
            outputs = model.llm.model(inputs_embeds=seq)
            h = outputs.last_hidden_state
            h_last = h[:, -1, :]  # Last position predicts next query
            q_current = model.llm_to_query(h_last)  # [B, query_dim]

    fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)  # [T, N]

    return (coarse_attn.reshape(T, grid_size, grid_size).cpu(),
            fine_attn.reshape(T, grid_size, grid_size).cpu())


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

    diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
    diff_norm = (diff_up / diff_max + 1) / 2

    diff_overlay = (plt.cm.RdBu_r(diff_norm)[:, :, :3] * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

    alpha = 0.6
    return ((1 - alpha) * frame_np + alpha * diff_overlay).astype(np.uint8)


def create_comparison_gif(frames, coarse_attn, fine_attn, output_path,
                          gt_caption="", fine_caption="", coarse_caption="", fps=6):
    """Create 4-panel comparison GIF with captions."""
    T = frames.shape[0]
    gif_frames = []

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
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

        diff = fine_attn[t] - coarse_attn[t]
        diff_overlay = create_diff_overlay(frame, diff)

        # Create 2x2 layout with caption area
        header_h = 30
        caption_h = 80
        total_h = H * 2 + header_h + caption_h
        total_w = W * 2

        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        combined[:] = [30, 30, 30]

        # Place images
        combined[header_h:header_h+H, :W] = frame_np
        combined[header_h:header_h+H, W:] = coarse_overlay
        combined[header_h+H:header_h+2*H, :W] = fine_overlay
        combined[header_h+H:header_h+2*H, W:] = diff_overlay

        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)

        # Header
        draw.text((10, 8), f"Frame {t+1}/{T}", fill=(255, 255, 255), font=font)
        draw.text((W + 10, 8), "AUTOREGRESSIVE FINE vs STATIC COARSE", fill=(200, 200, 100), font=small_font)

        # Panel labels
        draw.text((5, header_h + 3), "Original", fill=(255, 255, 255), font=small_font)
        draw.text((W + 5, header_h + 3), "Coarse (static q)", fill=(100, 150, 255), font=small_font)
        draw.text((5, header_h + H + 3), "Fine (autoregressive)", fill=(255, 150, 100), font=small_font)
        draw.text((W + 5, header_h + H + 3), "Diff (red=fine)", fill=(150, 255, 150), font=small_font)

        # Captions at bottom
        y_cap = header_h + 2*H + 5
        if gt_caption:
            gt_short = gt_caption[:50] + "..." if len(gt_caption) > 50 else gt_caption
            draw.text((5, y_cap), f"GT: {gt_short}", fill=(200, 200, 200), font=small_font)
        if fine_caption:
            fine_short = fine_caption[:50] + "..." if len(fine_caption) > 50 else fine_caption
            draw.text((5, y_cap + 25), f"Fine: {fine_short}", fill=(255, 180, 100), font=small_font)
        if coarse_caption:
            coarse_short = coarse_caption[:50] + "..." if len(coarse_caption) > 50 else coarse_caption
            draw.text((5, y_cap + 50), f"Coarse: {coarse_short}", fill=(100, 180, 255), font=small_font)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def webvid_generator(num_frames=16, frame_size=256, min_duration=8, max_duration=30):
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
    output_dir: str = "outputs/autoregressive_gifs",
    num_samples: int = 64,
    num_frames: int = 16,
    min_duration: int = 8,
    fps: int = 6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUTOREGRESSIVE FINE vs STATIC COARSE VISUALIZATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
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

    print(f"\nGenerating {num_samples} GIFs with autoregressive attention + captions...")

    data_gen = webvid_generator(num_frames=num_frames, min_duration=min_duration)

    completed = 0
    for frames, gt_caption, duration in data_gen:
        if completed >= num_samples:
            break

        try:
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            # Extract autoregressive attention maps
            coarse_attn, fine_attn = extract_autoregressive_attention(model, frames, device)

            # Generate captions
            fine_caption = model.generate_caption(frames_norm, tokenizer, max_new_tokens=30, use_fine=True)[0]
            coarse_caption = model.generate_caption(frames_norm, tokenizer, max_new_tokens=30, use_fine=False)[0]

            # Create GIF
            gif_path = output_dir / f"autoreg_{completed+1:03d}.gif"
            create_comparison_gif(
                frames, coarse_attn, fine_attn, gif_path,
                gt_caption=gt_caption,
                fine_caption=fine_caption,
                coarse_caption=coarse_caption,
                fps=fps
            )

            completed += 1
            print(f"  [{completed:3d}/{num_samples}] {duration}s: {gif_path.name}", flush=True)
            print(f"      GT: {gt_caption[:60]}...")
            print(f"      Fine: {fine_caption[:60]}...")
            print(f"      Coarse: {coarse_caption[:60]}...")

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
            continue

    print("\n" + "=" * 70)
    print(f"COMPLETE! Generated {completed} GIFs")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="/mnt/c/Users/sanps/Desktop/Projects/dino/nanoSmolLM/outputs/captioning_scaled/checkpoints/step_010000.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/autoregressive_gifs")
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
