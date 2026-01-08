#!/usr/bin/env python3
"""
Generate GIFs with captions overlaid for visual comparison.
Shows video frames with Fine and Coarse captions displayed.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
import imageio
import textwrap

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def wrap_text(text, width=50):
    """Wrap text to fit in image."""
    return '\n'.join(textwrap.wrap(text, width=width))


def create_caption_gif(frames, caption_fine, caption_coarse, output_path, video_id, fps=6):
    """Create GIF with video and captions side by side."""
    T = frames.shape[0]
    H, W = frames.shape[2], frames.shape[3]

    # Layout: video on left, captions on right
    caption_width = 400
    total_width = W + caption_width
    total_height = H

    gif_frames = []

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_text = font_title

    for t in range(T):
        # Create frame
        img = Image.new('RGB', (total_width, total_height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Add video frame
        frame = frames[t].permute(1, 2, 0).numpy()
        if frame.max() > 1:
            frame = (frame / 255.0 * 255).astype(np.uint8)
        else:
            frame = (frame * 255).astype(np.uint8)

        frame_img = Image.fromarray(frame)
        img.paste(frame_img, (0, 0))

        # Add frame counter on video
        draw.rectangle([(5, 5), (80, 25)], fill=(0, 0, 0, 180))
        draw.text((10, 7), f"Frame {t+1}/{T}", fill=(255, 255, 255), font=font_title)

        # Caption area
        x_offset = W + 10
        y_offset = 10

        # Video ID
        draw.text((x_offset, y_offset), f"Video: {video_id}", fill=(200, 200, 200), font=font_title)
        y_offset += 25

        # Fine caption
        draw.rectangle([(x_offset - 5, y_offset), (x_offset + caption_width - 15, y_offset + 20)], fill=(100, 40, 40))
        draw.text((x_offset, y_offset + 3), "FINE (dynamic):", fill=(255, 255, 255), font=font_title)
        y_offset += 25

        wrapped_fine = wrap_text(caption_fine[:200], width=45)
        draw.text((x_offset, y_offset), wrapped_fine, fill=(255, 200, 200), font=font_text)
        y_offset += len(wrapped_fine.split('\n')) * 16 + 20

        # Coarse caption
        draw.rectangle([(x_offset - 5, y_offset), (x_offset + caption_width - 15, y_offset + 20)], fill=(40, 40, 100))
        draw.text((x_offset, y_offset + 3), "COARSE (static):", fill=(255, 255, 255), font=font_title)
        y_offset += 25

        wrapped_coarse = wrap_text(caption_coarse[:200], width=45)
        draw.text((x_offset, y_offset), wrapped_coarse, fill=(200, 200, 255), font=font_text)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 30
    skip_first = 3000  # Different samples

    output_dir = Path('outputs/caption_gifs')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Caption GIFs")
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

    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load frames
    print(f"\nLoading frames (skipping first {skip_first})...")
    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))[skip_first:skip_first + num_examples * 2]

    generated = 0
    for frame_file in frame_files:
        if generated >= num_examples:
            break

        video_id = frame_file.stem

        try:
            frames = torch.load(frame_file, weights_only=True)
            if frames.shape[0] < num_frames:
                continue

            # Sample frames
            indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
            frames = frames[indices]

            # Generate captions
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    caption_fine = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=80, temperature=0.5, use_fine=True
                    )[0].strip()

                    caption_coarse = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=80, temperature=0.5, use_fine=False
                    )[0].strip()

            # Create GIF
            gif_path = output_dir / f'{generated:02d}_{video_id}.gif'
            create_caption_gif(frames, caption_fine, caption_coarse, gif_path, video_id)

            generated += 1
            print(f"[{generated}/{num_examples}] {video_id}")
            print(f"  Fine: {caption_fine[:60]}...")
            print(f"  Coarse: {caption_coarse[:60]}...")

        except Exception as e:
            print(f"  Error with {video_id}: {e}")

    print(f"\n{'='*60}")
    print(f"Generated {generated} caption GIFs in {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
