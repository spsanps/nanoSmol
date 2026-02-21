#!/usr/bin/env python3
"""
Generate attention map GIFs for temporal density analysis.
Shows how attention patterns change at different video speeds.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
import imageio
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def get_attention_weights(model, frames_norm, device):
    """
    Extract attention weights for coarse and fine queries.
    Returns attention maps for each frame.
    """
    B, T, C, H, W = frames_norm.shape

    with torch.no_grad():
        # Process each frame
        all_coarse_attn = []
        all_fine_attn = []

        for t in range(T):
            frame = frames_norm[:, t]  # [1, C, H, W]

            # Get patch features
            patch_features, cache = model.encoder.encode_patches(frame)
            # patch_features: [1, N+1, D] where N+1 includes CLS token

            # Coarse query attention
            q_coarse = model.q_static.expand(1, -1)  # [1, query_dim]
            q_embed_coarse = model.encoder.query_input_proj(q_coarse)  # [1, dino_dim]
            q_embed_coarse = q_embed_coarse.unsqueeze(1)  # [1, 1, D]

            attn_scores_coarse = torch.bmm(q_embed_coarse, patch_features.transpose(1, 2))
            attn_coarse = torch.softmax(attn_scores_coarse / (model.encoder.dino_dim ** 0.5), dim=-1)
            all_coarse_attn.append(attn_coarse[0, 0, 1:].cpu())  # Skip CLS token

        # Get fine queries from LLM
        # First, get coarse visual features for all frames
        coarse_features = []
        for t in range(T):
            frame = frames_norm[:, t]
            patch_features, cache = model.encoder.encode_patches(frame)
            q_coarse = model.q_static.expand(1, -1)
            z_coarse = model.encoder.query_attend(q_coarse, cache)
            coarse_features.append(z_coarse)

        coarse_features = torch.stack(coarse_features, dim=1)  # [1, T, dino_dim]
        coarse_proj = model.dino_to_llm(coarse_features) * model.visual_scale  # [1, T, llm_dim]

        # Build sequence for LLM
        coarse_token = model.coarse_token.expand(1, 1, -1)
        text_embeds = model.get_empty_text_embeds(1)
        seq = torch.cat([text_embeds, coarse_token, coarse_proj], dim=1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            llm_out = model.llm(inputs_embeds=seq, output_hidden_states=True)

        # Get hidden states for query prediction
        hidden = llm_out.hidden_states[-1]  # [1, seq_len, llm_dim]
        N_text = text_embeds.shape[1]
        h_for_queries = hidden[:, N_text + 1: N_text + 1 + T]  # [1, T, llm_dim]

        # Generate fine queries
        fine_queries = model.llm_to_query(h_for_queries.float())  # [1, T, query_dim]

        # Now compute fine attention for each frame
        for t in range(T):
            frame = frames_norm[:, t]
            patch_features, cache = model.encoder.encode_patches(frame)

            q_fine = fine_queries[:, t]  # [1, query_dim]
            q_embed_fine = model.encoder.query_input_proj(q_fine)
            q_embed_fine = q_embed_fine.unsqueeze(1)

            attn_scores_fine = torch.bmm(q_embed_fine, patch_features.transpose(1, 2))
            attn_fine = torch.softmax(attn_scores_fine / (model.encoder.dino_dim ** 0.5), dim=-1)
            all_fine_attn.append(attn_fine[0, 0, 1:].cpu())  # Skip CLS token

        return all_coarse_attn, all_fine_attn


def create_attention_frame(frame_img, attn_coarse, attn_fine, frame_idx, config_name, caption_fine=""):
    """Create a single frame for the attention GIF."""
    # Frame size
    frame_size = 256
    attn_size = 128
    total_width = frame_size + attn_size * 2 + 30
    total_height = frame_size + 60

    img = Image.new('RGB', (total_width, total_height), color=(25, 25, 25))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font = font_small = ImageFont.load_default()

    # Title
    draw.text((10, 5), f"{config_name} - Frame {frame_idx+1}", fill=(255, 255, 255), font=font)

    # Original frame
    if isinstance(frame_img, torch.Tensor):
        frame_np = frame_img.permute(1, 2, 0).numpy()
        if frame_np.max() > 1:
            frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = (frame_np * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)
    else:
        frame_pil = frame_img

    frame_pil = frame_pil.resize((frame_size, frame_size))
    img.paste(frame_pil, (10, 25))
    draw.text((10, 25 + frame_size + 2), "Original", fill=(200, 200, 200), font=font_small)

    # Coarse attention heatmap
    attn_coarse_np = attn_coarse.numpy()
    grid_size = int(np.sqrt(len(attn_coarse_np)))
    attn_coarse_2d = attn_coarse_np.reshape(grid_size, grid_size)

    # Normalize to 0-255
    attn_coarse_norm = (attn_coarse_2d - attn_coarse_2d.min()) / (attn_coarse_2d.max() - attn_coarse_2d.min() + 1e-8)
    attn_coarse_rgb = plt_colormap(attn_coarse_norm)
    attn_coarse_pil = Image.fromarray(attn_coarse_rgb).resize((attn_size, attn_size), Image.NEAREST)

    x_coarse = frame_size + 20
    img.paste(attn_coarse_pil, (x_coarse, 25))
    draw.text((x_coarse, 25 + attn_size + 2), "Coarse", fill=(100, 100, 255), font=font_small)

    # Fine attention heatmap
    attn_fine_np = attn_fine.numpy()
    attn_fine_2d = attn_fine_np.reshape(grid_size, grid_size)
    attn_fine_norm = (attn_fine_2d - attn_fine_2d.min()) / (attn_fine_2d.max() - attn_fine_2d.min() + 1e-8)
    attn_fine_rgb = plt_colormap(attn_fine_norm)
    attn_fine_pil = Image.fromarray(attn_fine_rgb).resize((attn_size, attn_size), Image.NEAREST)

    x_fine = x_coarse + attn_size + 10
    img.paste(attn_fine_pil, (x_fine, 25))
    draw.text((x_fine, 25 + attn_size + 2), "Fine", fill=(255, 100, 100), font=font_small)

    # Overlay attention on original frame (blended)
    frame_overlay = create_overlay(frame_pil, attn_fine_2d)
    img.paste(frame_overlay, (x_fine, 25 + attn_size + 20))
    draw.text((x_fine, 25 + attn_size + 20 + attn_size + 2), "Overlay", fill=(255, 200, 100), font=font_small)

    # Caption (truncated)
    if caption_fine:
        caption_short = caption_fine[:80].replace('\n', ' ')
        draw.text((10, total_height - 20), f"Caption: {caption_short}...", fill=(200, 200, 200), font=font_small)

    return img


def plt_colormap(arr):
    """Simple jet-like colormap."""
    # Normalize to 0-1
    arr = np.clip(arr, 0, 1)

    # Create RGB array
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)

    # Blue -> Cyan -> Green -> Yellow -> Red
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if v < 0.25:
                r, g, b = 0, int(v * 4 * 255), 255
            elif v < 0.5:
                r, g, b = 0, 255, int((0.5 - v) * 4 * 255)
            elif v < 0.75:
                r, g, b = int((v - 0.5) * 4 * 255), 255, 0
            else:
                r, g, b = 255, int((1 - v) * 4 * 255), 0
            rgb[i, j] = [r, g, b]

    return rgb


def create_overlay(frame_pil, attn_2d):
    """Create attention overlay on frame."""
    frame_np = np.array(frame_pil.resize((128, 128)))

    # Resize attention to frame size
    attn_resized = np.array(Image.fromarray((attn_2d * 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR)) / 255.0

    # Create heatmap
    heatmap = plt_colormap(attn_resized)

    # Blend
    alpha = 0.4
    overlay = (frame_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


def get_sampling_configs():
    """Define temporal sampling configurations."""
    return {
        '4_sparse': {
            'indices': [0, 5, 10, 15],
            'description': '4 frames (fast)',
            'color': (255, 100, 100),
        },
        '8_uniform': {
            'indices': [0, 2, 4, 6, 9, 11, 13, 15],
            'description': '8 frames (medium)',
            'color': (255, 200, 100),
        },
        '16_dense': {
            'indices': list(range(16)),
            'description': '16 frames (slow)',
            'color': (100, 255, 100),
        },
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_examples = 8
    skip_first = 1500

    configs = get_sampling_configs()

    output_dir = Path('outputs/temporal_attention_gifs')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Temporal Attention GIF Generator")
    print("=" * 70)
    print(f"Configurations: {list(configs.keys())}")
    print(f"Samples: {num_examples}")

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

    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load frames
    print(f"\nLoading frames (skipping first {skip_first})...")
    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))[skip_first:skip_first + num_examples * 2]

    processed = 0

    for frame_file in frame_files:
        if processed >= num_examples:
            break

        video_id = frame_file.stem

        try:
            all_frames = torch.load(frame_file, weights_only=True)
            if all_frames.shape[0] < 16:
                continue

            print(f"\n[{processed + 1}/{num_examples}] Processing {video_id}")

            for config_name, config in configs.items():
                indices = config['indices']
                sampled_frames = all_frames[indices]

                # Normalize for DINO
                frames_norm = normalize_for_dino(sampled_frames, device).unsqueeze(0)

                # Get attention weights
                coarse_attns, fine_attns = get_attention_weights(model, frames_norm, device)

                # Generate caption
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        caption_fine = model.generate_caption(
                            frames_norm, tokenizer, max_new_tokens=40, temperature=0.5, use_fine=True
                        )[0].strip()

                # Create GIF frames
                gif_frames = []
                for t in range(len(indices)):
                    frame_img = create_attention_frame(
                        sampled_frames[t],
                        coarse_attns[t],
                        fine_attns[t],
                        t,
                        config['description'],
                        caption_fine if t == 0 else ""
                    )
                    gif_frames.append(np.array(frame_img))

                # Save GIF
                gif_path = output_dir / f'{processed:02d}_{video_id}_{config_name}.gif'
                imageio.mimsave(gif_path, gif_frames, fps=3, loop=0)

                print(f"  {config_name}: saved GIF with {len(indices)} frames")

            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Create summary
    summary_path = output_dir / 'README.md'
    with open(summary_path, 'w') as f:
        f.write("# Temporal Attention GIFs\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Videos:** {processed}\n\n")

        f.write("## What These Show\n\n")
        f.write("Each GIF compares attention patterns at different video 'speeds':\n\n")
        f.write("- **4 frames (fast):** Sparse sampling, appears like fast-forwarded video\n")
        f.write("- **8 frames (medium):** Moderate sampling\n")
        f.write("- **16 frames (slow):** Dense sampling, full temporal detail\n\n")

        f.write("## Visualization Components\n\n")
        f.write("Each frame shows:\n")
        f.write("1. **Original Frame:** The video frame\n")
        f.write("2. **Coarse Attention:** Static query attention (same for all configs)\n")
        f.write("3. **Fine Attention:** Dynamic query attention (changes with LLM)\n")
        f.write("4. **Overlay:** Fine attention overlaid on original frame\n\n")

        f.write("## Key Questions\n\n")
        f.write("- Does fine attention focus on different regions at different speeds?\n")
        f.write("- Does attention track motion differently when video is 'faster'?\n")
        f.write("- Are the attention patterns meaningful or random?\n\n")

        f.write("## Files\n\n")
        for i in range(processed):
            f.write(f"### Video {i+1}\n")
            for config_name in configs.keys():
                f.write(f"- `{i:02d}_*_{config_name}.gif`\n")
            f.write("\n")

    print(f"\n{'='*70}")
    print(f"Generated {processed * len(configs)} GIFs")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
