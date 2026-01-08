#!/usr/bin/env python3
"""
Create caption comparison images with ground truth from WebVid.
Shows: Video frames + Ground Truth + Fine caption + Coarse caption
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
from datasets import load_dataset
import requests
import subprocess
import tempfile
import textwrap

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def download_video(url, num_frames=16, size=256):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(response.content)
            temp_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    'ffmpeg', '-i', temp_path,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 2),
                    '-q:v', '2', f'{tmpdir}/frame_%04d.jpg',
                    '-y', '-loglevel', 'error'
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)
                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    return None
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
        return None


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def wrap_text(text, width=55):
    return '\n'.join(textwrap.wrap(text, width=width))


def create_comparison_image(frames, gt_caption, caption_fine, caption_coarse, output_path, video_id):
    """Create comparison image with GT and generated captions."""
    T = min(8, frames.shape[0])
    indices = np.linspace(0, frames.shape[0] - 1, T).astype(int)

    H, W = 128, 128
    cols, rows = 4, 2
    padding = 5
    caption_height = 180

    total_width = cols * W + (cols + 1) * padding
    total_height = rows * H + (rows + 1) * padding + caption_height

    img = Image.new('RGB', (total_width, total_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_title = font_text = ImageFont.load_default()

    # Draw video ID
    draw.text((padding, 2), f"Video: {video_id}", fill=(150, 150, 150), font=font_text)

    # Draw frames
    for i, idx in enumerate(indices):
        row, col = i // cols, i % cols
        x = padding + col * (W + padding)
        y = padding + 12 + row * (H + padding)

        frame = frames[idx].permute(1, 2, 0).numpy()
        if frame.max() > 1:
            frame = frame.astype(np.uint8)
        else:
            frame = (frame * 255).astype(np.uint8)

        frame_img = Image.fromarray(frame).resize((W, H))
        img.paste(frame_img, (x, y))

    # Caption area
    y_cap = rows * H + (rows + 1) * padding + 18

    # Ground Truth caption (GREEN)
    draw.rectangle([(padding - 2, y_cap - 2), (total_width - padding, y_cap + 14)], fill=(20, 60, 20))
    draw.text((padding, y_cap), "GROUND TRUTH:", fill=(100, 255, 100), font=font_title)
    y_cap += 16
    gt_wrapped = wrap_text(gt_caption[:180], width=60)
    draw.text((padding, y_cap), gt_wrapped, fill=(150, 255, 150), font=font_text)
    y_cap += len(gt_wrapped.split('\n')) * 12 + 8

    # Fine caption (RED)
    draw.rectangle([(padding - 2, y_cap - 2), (total_width - padding, y_cap + 14)], fill=(60, 20, 20))
    draw.text((padding, y_cap), "FINE (dynamic):", fill=(255, 100, 100), font=font_title)
    y_cap += 16
    fine_wrapped = wrap_text(caption_fine[:180], width=60)
    draw.text((padding, y_cap), fine_wrapped, fill=(255, 180, 180), font=font_text)
    y_cap += len(fine_wrapped.split('\n')) * 12 + 8

    # Coarse caption (BLUE)
    draw.rectangle([(padding - 2, y_cap - 2), (total_width - padding, y_cap + 14)], fill=(20, 20, 60))
    draw.text((padding, y_cap), "COARSE (static):", fill=(100, 100, 255), font=font_title)
    y_cap += 16
    coarse_wrapped = wrap_text(caption_coarse[:180], width=60)
    draw.text((padding, y_cap), coarse_wrapped, fill=(180, 180, 255), font=font_text)

    img.save(output_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 15
    skip_first = 800  # Skip to get diverse examples

    output_dir = Path('outputs/caption_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Caption Comparison with Ground Truth")
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

    # Stream WebVid
    print(f"\nStreaming WebVid (skipping first {skip_first})...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    generated = 0
    skipped = 0

    for sample in ds:
        if skipped < skip_first:
            skipped += 1
            if skipped % 200 == 0:
                print(f"  Skipped {skipped}/{skip_first}...")
            continue

        if generated >= num_examples:
            break

        frames = download_video(sample['contentUrl'], num_frames)
        if frames is None:
            continue

        gt_caption = sample.get('name', 'Unknown')
        video_id = sample.get('videoid', 'unknown')

        # Generate captions
        frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                caption_fine = model.generate_caption(
                    frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=True
                )[0].strip()

                caption_coarse = model.generate_caption(
                    frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=False
                )[0].strip()

        # Create comparison image
        img_path = output_dir / f'{generated:02d}_{video_id}.png'
        create_comparison_image(frames, gt_caption, caption_fine, caption_coarse, img_path, video_id)

        generated += 1
        print(f"\n[{generated}/{num_examples}] Video {video_id}")
        print(f"  GT: {gt_caption[:70]}...")
        print(f"  Fine: {caption_fine[:70]}...")
        print(f"  Coarse: {caption_coarse[:70]}...")

    print(f"\n{'='*60}")
    print(f"Generated {generated} comparison images in {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
