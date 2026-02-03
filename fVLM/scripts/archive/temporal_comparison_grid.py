#!/usr/bin/env python3
"""
Create static comparison grids showing attention at different temporal densities.
Each grid shows the same video at 4/8/16 frames with attention maps.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
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
    """Extract attention weights for coarse and fine queries."""
    B, T, C, H, W = frames_norm.shape

    with torch.no_grad():
        all_coarse_attn = []
        all_fine_attn = []

        # Get patch features and coarse attention for each frame
        for t in range(T):
            frame = frames_norm[:, t]
            patch_features, cache = model.encoder.encode_patches(frame)

            q_coarse = model.q_static.expand(1, -1)
            q_embed_coarse = model.encoder.query_input_proj(q_coarse).unsqueeze(1)

            attn_scores_coarse = torch.bmm(q_embed_coarse, patch_features.transpose(1, 2))
            attn_coarse = torch.softmax(attn_scores_coarse / (model.encoder.dino_dim ** 0.5), dim=-1)
            all_coarse_attn.append(attn_coarse[0, 0, 1:].cpu())

        # Get fine queries from LLM
        coarse_features = []
        for t in range(T):
            frame = frames_norm[:, t]
            patch_features, cache = model.encoder.encode_patches(frame)
            q_coarse = model.q_static.expand(1, -1)
            z_coarse = model.encoder.query_attend(q_coarse, cache)
            coarse_features.append(z_coarse)

        coarse_features = torch.stack(coarse_features, dim=1)
        coarse_proj = model.dino_to_llm(coarse_features) * model.visual_scale

        coarse_token = model.coarse_token.expand(1, 1, -1)
        text_embeds = model.get_empty_text_embeds(1)
        seq = torch.cat([text_embeds, coarse_token, coarse_proj], dim=1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            llm_out = model.llm(inputs_embeds=seq, output_hidden_states=True)

        hidden = llm_out.hidden_states[-1]
        N_text = text_embeds.shape[1]
        h_for_queries = hidden[:, N_text + 1: N_text + 1 + T]
        fine_queries = model.llm_to_query(h_for_queries.float())

        for t in range(T):
            frame = frames_norm[:, t]
            patch_features, cache = model.encoder.encode_patches(frame)

            q_fine = fine_queries[:, t]
            q_embed_fine = model.encoder.query_input_proj(q_fine).unsqueeze(1)

            attn_scores_fine = torch.bmm(q_embed_fine, patch_features.transpose(1, 2))
            attn_fine = torch.softmax(attn_scores_fine / (model.encoder.dino_dim ** 0.5), dim=-1)
            all_fine_attn.append(attn_fine[0, 0, 1:].cpu())

        return all_coarse_attn, all_fine_attn


def jet_colormap(arr):
    """Simple jet-like colormap."""
    arr = np.clip(arr, 0, 1)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)

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


def create_overlay(frame_pil, attn_2d, size=64):
    """Create attention overlay on frame."""
    frame_np = np.array(frame_pil.resize((size, size)))
    attn_resized = np.array(Image.fromarray((attn_2d * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    heatmap = jet_colormap(attn_resized)
    alpha = 0.5
    overlay = (frame_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def create_comparison_grid(video_id, all_frames, config_results, output_path):
    """
    Create a grid comparing attention patterns across temporal densities.

    Layout:
    - Rows: Different speeds (4 sparse, 8 uniform, 16 dense)
    - Columns: Selected frames with Fine attention overlay
    """
    configs = list(config_results.keys())
    n_configs = len(configs)

    # Show 4 key frames for each config
    n_show_frames = 4

    frame_size = 80
    label_width = 120
    padding = 5
    header_height = 40
    row_height = frame_size + 40

    total_width = label_width + (n_show_frames * 2) * (frame_size + padding) + 20
    total_height = header_height + n_configs * row_height + 80

    img = Image.new('RGB', (total_width, total_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_title = font_label = font_small = ImageFont.load_default()

    # Title
    draw.text((10, 10), f"Temporal Attention Comparison: {video_id}", fill=(255, 255, 255), font=font_title)

    # Column headers
    x = label_width
    draw.text((x, header_height - 15), "Original Frames", fill=(150, 150, 150), font=font_small)
    x += n_show_frames * (frame_size + padding)
    draw.text((x, header_height - 15), "Fine Attention Overlay", fill=(255, 150, 150), font=font_small)

    y = header_height

    config_labels = {
        '4_sparse': ('4 frames\n(fast)', (255, 100, 100)),
        '8_uniform': ('8 frames\n(medium)', (255, 200, 100)),
        '16_dense': ('16 frames\n(slow)', (100, 255, 100)),
    }

    for config_name in configs:
        result = config_results[config_name]
        sampled_frames = result['frames']
        fine_attns = result['fine_attns']
        caption = result['caption']

        label, color = config_labels.get(config_name, (config_name, (255, 255, 255)))

        # Row label
        draw.rectangle([(5, y + 5), (label_width - 5, y + 35)], fill=(40, 40, 40))
        draw.text((10, y + 10), label, fill=color, font=font_label)

        # Select frames to show (evenly spaced)
        n_frames = len(sampled_frames)
        show_indices = np.linspace(0, n_frames - 1, n_show_frames).astype(int)

        x = label_width

        # Original frames
        for i, idx in enumerate(show_indices):
            frame = sampled_frames[idx].permute(1, 2, 0).numpy()
            if frame.max() > 1:
                frame = frame.astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)

            frame_pil = Image.fromarray(frame).resize((frame_size, frame_size))
            img.paste(frame_pil, (x, y + 5))
            draw.text((x + 2, y + frame_size + 7), f"t{idx+1}", fill=(100, 100, 100), font=font_small)
            x += frame_size + padding

        # Fine attention overlays
        for i, idx in enumerate(show_indices):
            frame = sampled_frames[idx].permute(1, 2, 0).numpy()
            if frame.max() > 1:
                frame = frame.astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)

            frame_pil = Image.fromarray(frame)

            attn = fine_attns[idx].numpy()
            grid_size = int(np.sqrt(len(attn)))
            attn_2d = attn.reshape(grid_size, grid_size)
            attn_norm = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

            overlay = create_overlay(frame_pil, attn_norm, frame_size)
            img.paste(overlay, (x, y + 5))

            # Max attention value
            draw.text((x + 2, y + frame_size + 7), f"{attn.max():.3f}", fill=(255, 150, 150), font=font_small)
            x += frame_size + padding

        y += row_height

    # Captions section
    y += 10
    draw.line([(10, y), (total_width - 10, y)], fill=(60, 60, 60), width=1)
    y += 5

    draw.text((10, y), "Captions:", fill=(200, 200, 200), font=font_label)
    y += 18

    for config_name in configs:
        caption = config_results[config_name]['caption'][:100].replace('\n', ' ')
        label, color = config_labels.get(config_name, (config_name, (255, 255, 255)))
        draw.text((10, y), f"{config_name}:", fill=color, font=font_small)
        draw.text((80, y), caption + "...", fill=(200, 200, 200), font=font_small)
        y += 14

    img.save(output_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_examples = 10
    skip_first = 1500

    configs = {
        '4_sparse': [0, 5, 10, 15],
        '8_uniform': [0, 2, 4, 6, 9, 11, 13, 15],
        '16_dense': list(range(16)),
    }

    output_dir = Path('outputs/temporal_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Temporal Comparison Grid Generator")
    print("=" * 70)

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
    print(f"\nLoading frames...")
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

            config_results = {}

            for config_name, indices in configs.items():
                sampled_frames = all_frames[indices]
                frames_norm = normalize_for_dino(sampled_frames, device).unsqueeze(0)

                coarse_attns, fine_attns = get_attention_weights(model, frames_norm, device)

                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        caption = model.generate_caption(
                            frames_norm, tokenizer, max_new_tokens=40, temperature=0.5, use_fine=True
                        )[0].strip()

                config_results[config_name] = {
                    'frames': sampled_frames,
                    'fine_attns': fine_attns,
                    'coarse_attns': coarse_attns,
                    'caption': caption,
                }

                # Compute attention stats
                max_attn = max(a.max().item() for a in fine_attns)
                print(f"  {config_name}: max_attn={max_attn:.4f}")

            # Create comparison grid
            grid_path = output_dir / f'{processed:02d}_{video_id}_comparison.png'
            create_comparison_grid(video_id, all_frames, config_results, grid_path)

            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Generated {processed} comparison grids")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
