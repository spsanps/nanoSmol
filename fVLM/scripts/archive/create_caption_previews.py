#!/usr/bin/env python3
"""Create static preview images from caption GIFs."""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
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


def wrap_text(text, width=40):
    return '\n'.join(textwrap.wrap(text, width=width))


def create_preview(frames, caption_fine, caption_coarse, output_path, video_id):
    """Create 2x4 grid preview with captions."""
    T = min(8, frames.shape[0])
    indices = np.linspace(0, frames.shape[0] - 1, T).astype(int)

    H, W = 128, 128  # Smaller frames
    cols, rows = 4, 2
    padding = 5
    caption_height = 120

    total_width = cols * W + (cols + 1) * padding
    total_height = rows * H + (rows + 1) * padding + caption_height

    img = Image.new('RGB', (total_width, total_height), color=(25, 25, 25))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_title = font_text = ImageFont.load_default()

    # Draw frames
    for i, idx in enumerate(indices):
        row, col = i // cols, i % cols
        x = padding + col * (W + padding)
        y = padding + row * (H + padding)

        frame = frames[idx].permute(1, 2, 0).numpy()
        if frame.max() > 1:
            frame = (frame).astype(np.uint8)
        else:
            frame = (frame * 255).astype(np.uint8)

        frame_img = Image.fromarray(frame).resize((W, H))
        img.paste(frame_img, (x, y))

    # Caption area
    y_cap = rows * H + (rows + 1) * padding + 5

    # Fine caption
    draw.text((padding, y_cap), f"FINE: ", fill=(255, 100, 100), font=font_title)
    fine_wrapped = wrap_text(caption_fine[:150], width=70)
    draw.text((padding + 40, y_cap), fine_wrapped, fill=(255, 200, 200), font=font_text)

    # Coarse caption
    y_cap += 55
    draw.text((padding, y_cap), f"COARSE: ", fill=(100, 100, 255), font=font_title)
    coarse_wrapped = wrap_text(caption_coarse[:150], width=70)
    draw.text((padding + 55, y_cap), coarse_wrapped, fill=(200, 200, 255), font=font_text)

    img.save(output_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 12
    skip_first = 4000

    output_dir = Path('outputs/caption_previews')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating caption previews...")

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

            indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
            frames = frames[indices]

            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    caption_fine = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=True
                    )[0].strip()

                    caption_coarse = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=False
                    )[0].strip()

            preview_path = output_dir / f'{generated:02d}_{video_id}.png'
            create_preview(frames, caption_fine, caption_coarse, preview_path, video_id)

            generated += 1
            print(f"[{generated}/{num_examples}] {video_id}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nPreviews saved to {output_dir}")


if __name__ == "__main__":
    main()
