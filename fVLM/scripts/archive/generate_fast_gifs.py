#!/usr/bin/env python3
"""
Generate fast-motion GIFs with more frames for better visualization.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def get_attention_weights(query, patch_features, dino_dim):
    q_embed = query.unsqueeze(1)
    attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
    attn_weights = torch.softmax(attn_scores / (dino_dim ** 0.5), dim=-1)
    return attn_weights.squeeze(1)


@torch.no_grad()
def extract_attention_maps(model, frames, device):
    B = 1
    T = frames.shape[0]

    frames_norm = normalize_for_dino(frames, device).unsqueeze(0)
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    # Pass 1: Coarse
    q_static = model.q_static.expand(B, -1)
    q_static_proj = model.encoder.query_input_proj(q_static)

    coarse_attn_list = []
    z_coarse_list = []

    for t in range(T):
        pf = all_caches[t]['patch_features']
        attn = get_attention_weights(q_static_proj, pf, model.encoder.dino_dim)
        coarse_attn_list.append(attn[:, 1:])
        z_coarse_list.append(model.encoder.query_attend(q_static, all_caches[t]))

    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    coarse_token = model.coarse_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state

    queries = model.llm_to_query(h_pass1[:, N_text + 1:])

    # Pass 2: Fine
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    fine_attn_list = []
    for t in range(T):
        q_t = shifted_q[:, t]
        q_t_proj = model.encoder.query_input_proj(q_t)
        attn = get_attention_weights(q_t_proj, all_caches[t]['patch_features'], model.encoder.dino_dim)
        fine_attn_list.append(attn[:, 1:])

    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)
    fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)

    n_patches = coarse_attn.shape[1]
    grid_size = int(n_patches ** 0.5)

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
        frame_np = (frame_np / 255.0 * 255).astype(np.uint8)
    else:
        frame_np = (frame_np * 255).astype(np.uint8)

    return ((1 - alpha) * frame_np + alpha * heatmap).astype(np.uint8)


def create_fast_gif(frames, coarse_attn, fine_attn, output_path, video_id, fps=12):
    """Create fast side-by-side GIF."""
    T = frames.shape[0]
    gif_frames = []

    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() > 1:
            frame_np = (frame_np / 255.0 * 255).astype(np.uint8)
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
        diff_norm = (diff_up / diff_max + 1) / 2
        diff_overlay = (plt.cm.RdBu_r(diff_norm)[:, :, :3] * 255).astype(np.uint8)

        # Combine 4 panels
        header_h = 30
        combined = np.zeros((H + header_h, W * 4, 3), dtype=np.uint8)
        combined[header_h:, :W] = frame_np
        combined[header_h:, W:2*W] = coarse_overlay
        combined[header_h:, 2*W:3*W] = fine_overlay
        combined[header_h:, 3*W:] = diff_overlay

        # Headers
        combined[:header_h, :W] = [50, 50, 50]
        combined[:header_h, W:2*W] = [30, 30, 120]
        combined[:header_h, 2*W:3*W] = [120, 30, 30]
        combined[:header_h, 3*W:] = [30, 100, 30]

        # Add text
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()

        labels = [f"Frame {t}/{T-1}", "Coarse (static)", "Fine (dynamic)", "Diff (F-C)"]
        for i, label in enumerate(labels):
            draw.text((i * W + 10, 8), label, fill=(255, 255, 255), font=font)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 32  # More frames for smoother motion
    num_examples = 6
    skip_first = 1000
    fps = 10  # Fast playback

    output_dir = Path('outputs/fast_gifs')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Generating Fast GIFs ({num_frames} frames @ {fps} fps)")
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

    # Load frames
    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))[skip_first:]

    for i, frame_file in enumerate(frame_files[:num_examples]):
        video_id = frame_file.stem
        print(f"\n[{i+1}/{num_examples}] Video {video_id}")

        try:
            frames = torch.load(frame_file, weights_only=True)
            total_frames = frames.shape[0]

            if total_frames < num_frames:
                print(f"  Only {total_frames} frames, using all")
                sampled_frames = frames
            else:
                indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
                sampled_frames = frames[indices]

            print(f"  Using {sampled_frames.shape[0]} frames")

            # Extract attention
            coarse_attn, fine_attn = extract_attention_maps(model, sampled_frames, device)

            # Create GIF
            gif_path = output_dir / f'comparison_{i:02d}_{video_id}.gif'
            create_fast_gif(sampled_frames, coarse_attn, fine_attn, gif_path, video_id, fps=fps)
            print(f"  Saved: {gif_path.name}")

            # Also create individual attention GIFs
            coarse_gif = output_dir / f'coarse_{i:02d}_{video_id}.gif'
            fine_gif = output_dir / f'fine_{i:02d}_{video_id}.gif'

            coarse_frames = [create_attention_overlay(sampled_frames[t], coarse_attn[t]) for t in range(len(sampled_frames))]
            fine_frames_list = [create_attention_overlay(sampled_frames[t], fine_attn[t]) for t in range(len(sampled_frames))]

            imageio.mimsave(coarse_gif, coarse_frames, fps=fps, loop=0)
            imageio.mimsave(fine_gif, fine_frames_list, fps=fps, loop=0)
            print(f"  Saved: {coarse_gif.name}, {fine_gif.name}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"Done! GIFs saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
