#!/usr/bin/env python3
"""
Generate smooth GIFs by streaming fresh videos with many frames.
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
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def download_video_many_frames(url, num_frames=48, size=256):
    """Download video and extract many frames."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None, None

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract more frames than needed
                cmd = [
                    'ffmpeg', '-i', temp_path,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 2),
                    '-q:v', '2',
                    f'{tmpdir}/frame_%04d.jpg',
                    '-y', '-loglevel', 'error'
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)

                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    return None, None

                # Sample evenly
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                    frames.append(frame)
                return torch.stack(frames), len(frame_files)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"  Download error: {e}")
        return None, None


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def get_attention_weights(query, patch_features, dino_dim):
    q_embed = query.unsqueeze(1)
    attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
    return torch.softmax(attn_scores / (dino_dim ** 0.5), dim=-1).squeeze(1)


@torch.no_grad()
def extract_attention_maps(model, frames, device):
    B, T = 1, frames.shape[0]
    frames_norm = normalize_for_dino(frames, device).unsqueeze(0)
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    q_static = model.q_static.expand(B, -1)
    q_static_proj = model.encoder.query_input_proj(q_static)

    coarse_attn_list, z_coarse_list = [], []
    for t in range(T):
        attn = get_attention_weights(q_static_proj, all_caches[t]['patch_features'], model.encoder.dino_dim)
        coarse_attn_list.append(attn[:, 1:])
        z_coarse_list.append(model.encoder.query_attend(q_static, all_caches[t]))

    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    seq_pass1 = torch.cat([text_embeds, model.coarse_token.expand(B, -1, -1), z_coarse], dim=1)
    h_pass1 = model.llm.model(inputs_embeds=seq_pass1).last_hidden_state
    queries = model.llm_to_query(h_pass1[:, N_text + 1:])

    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    fine_attn_list = []
    for t in range(T):
        q_t_proj = model.encoder.query_input_proj(shifted_q[:, t])
        attn = get_attention_weights(q_t_proj, all_caches[t]['patch_features'], model.encoder.dino_dim)
        fine_attn_list.append(attn[:, 1:])

    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)
    fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)

    grid_size = int(coarse_attn.shape[1] ** 0.5)
    return coarse_attn.reshape(T, grid_size, grid_size).cpu(), fine_attn.reshape(T, grid_size, grid_size).cpu()


def create_attention_overlay(frame, attn_map, alpha=0.5):
    H, W = frame.shape[1], frame.shape[2]
    attn_up = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    heatmap = (plt.cm.jet(attn_norm)[:, :, :3] * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    if frame_np.max() > 1:
        frame_np = frame_np.astype(np.uint8)
    else:
        frame_np = (frame_np * 255).astype(np.uint8)

    return ((1 - alpha) * frame_np + alpha * heatmap).astype(np.uint8)


def create_smooth_gif(frames, coarse_attn, fine_attn, output_path, caption, fps=12):
    """Create smooth comparison GIF."""
    T = frames.shape[0]
    gif_frames = []

    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() > 1:
            frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = (frame_np * 255).astype(np.uint8)

        coarse_overlay = create_attention_overlay(frame, coarse_attn[t])
        fine_overlay = create_attention_overlay(frame, fine_attn[t])

        # Difference
        diff = fine_attn[t] - coarse_attn[t]
        diff_up = F.interpolate(
            diff.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
        diff_overlay = (plt.cm.RdBu_r((diff_up / diff_max + 1) / 2)[:, :, :3] * 255).astype(np.uint8)

        # Layout
        header_h, footer_h = 25, 35
        combined = np.zeros((H + header_h + footer_h, W * 4, 3), dtype=np.uint8)
        combined[header_h:header_h+H, :W] = frame_np
        combined[header_h:header_h+H, W:2*W] = coarse_overlay
        combined[header_h:header_h+H, 2*W:3*W] = fine_overlay
        combined[header_h:header_h+H, 3*W:] = diff_overlay

        # Colors
        combined[:header_h, :W] = [40, 40, 40]
        combined[:header_h, W:2*W] = [30, 30, 100]
        combined[:header_h, 2*W:3*W] = [100, 30, 30]
        combined[:header_h, 3*W:] = [30, 80, 30]
        combined[-footer_h:, :] = [25, 25, 25]

        # Text
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = font_small = ImageFont.load_default()

        labels = [f"Original ({t+1}/{T})", "Coarse (static)", "Fine (dynamic)", "Difference"]
        for i, label in enumerate(labels):
            draw.text((i * W + 8, 6), label, fill=(255, 255, 255), font=font)

        # Caption
        cap_short = caption[:120] + "..." if len(caption) > 120 else caption
        draw.text((8, H + header_h + 8), cap_short, fill=(180, 180, 180), font=font_small)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 32  # Many frames for smooth motion
    num_examples = 64
    skip_first = 500  # Skip more to get diverse samples
    fps = 10

    output_dir = Path('outputs/diverse_64')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Generating Smooth GIFs ({num_frames} frames @ {fps} fps)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load('outputs/multitask/checkpoints/final.pt', map_location=device, weights_only=False)

    model = FoveatedVideoModel(
        dino_model='facebook/dinov2-small',
        llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
        dino_dim=384, llm_dim=576, query_dim=128, lambda_coarse=1.0,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Stream WebVid
    print(f"\nStreaming WebVid (skipping first {skip_first})...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    videos = []
    skipped = 0
    for sample in ds:
        if skipped < skip_first:
            skipped += 1
            continue
        if len(videos) >= num_examples:
            break

        frames, total = download_video_many_frames(sample['contentUrl'], num_frames)
        if frames is not None:
            videos.append({
                'frames': frames,
                'caption': sample.get('name', 'Unknown'),
                'video_id': sample.get('videoid', f'vid_{len(videos)}'),
                'total_frames': total,
            })
            print(f"  [{len(videos)}/{num_examples}] {sample.get('name', '')[:50]}... ({total} frames)")

    print(f"\nGot {len(videos)} videos")

    # Generate GIFs
    for i, video in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {video['video_id']}: {video['caption'][:40]}...")

        frames = video['frames']
        coarse_attn, fine_attn = extract_attention_maps(model, frames, device)

        # Comparison GIF
        gif_path = output_dir / f'comparison_{i:02d}_{video["video_id"]}.gif'
        create_smooth_gif(frames, coarse_attn, fine_attn, gif_path, video['caption'], fps=fps)
        print(f"  Saved: {gif_path.name} ({frames.shape[0]} frames)")

        # Individual GIFs
        coarse_frames = [create_attention_overlay(frames[t], coarse_attn[t]) for t in range(len(frames))]
        fine_frames_list = [create_attention_overlay(frames[t], fine_attn[t]) for t in range(len(frames))]

        imageio.mimsave(output_dir / f'coarse_{i:02d}.gif', coarse_frames, fps=fps, loop=0)
        imageio.mimsave(output_dir / f'fine_{i:02d}.gif', fine_frames_list, fps=fps, loop=0)

    print(f"\n{'='*60}")
    print(f"Done! GIFs saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
