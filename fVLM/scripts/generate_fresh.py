#!/usr/bin/env python3
"""
Generate captions on fresh WebVid examples using the multitask-trained model.
Skips early examples to test on truly unseen videos.
"""

import torch
import sys
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
import requests
import subprocess
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def download_video(url, num_frames=16, size=256):
    """Download and extract frames from video."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
            f.write(response.content)
            f.flush()

            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    'ffmpeg', '-i', f.name,
                    '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                    '-frames:v', str(num_frames * 3),
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
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                    frames.append(frame)
                return torch.stack(frames)
    except Exception as e:
        return None


def normalize_frames(frames, device):
    """Normalize for DINO."""
    frames = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    frames = (frames - mean) / std
    return frames


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 15
    skip_first = 500  # Skip first N examples to get fresh ones

    print("=" * 70)
    print("Fresh Caption Generation - Multitask Model")
    print("=" * 70)
    print(f"Skipping first {skip_first} videos to get unseen examples")
    print(f"Generating captions for {num_examples} videos")
    print()

    # Load model
    print("Loading multitask-trained model...")
    checkpoint_path = Path('outputs/multitask/checkpoints/final.pt')

    if not checkpoint_path.exists():
        print(f"ERROR: {checkpoint_path} not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Model config (hardcoded - matches train_multitask.py)
    model_cfg = {
        'dino_model': 'facebook/dinov2-small',
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_dim': 384,
        'llm_dim': 576,
        'query_dim': 128,  # Was 128 in training
        'lambda_coarse': 1.0,
    }

    model = FoveatedVideoModel(
        dino_model=model_cfg['dino_model'],
        llm_model=model_cfg['llm_model'],
        dino_dim=model_cfg['dino_dim'],
        llm_dim=model_cfg['llm_dim'],
        query_dim=model_cfg['query_dim'],
        lambda_coarse=model_cfg['lambda_coarse'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use local precomputed frames (from webvid_large)
    print(f"\nLoading local frames (skipping first {skip_first})...")

    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))
    print(f"  Found {len(frame_files)} precomputed frame files")

    # Skip first N and sample from the rest
    available = frame_files[skip_first:]
    print(f"  Using videos from index {skip_first}+")

    videos = []
    for frame_file in available[:num_examples * 2]:  # Try more in case some fail
        if len(videos) >= num_examples:
            break

        try:
            frames = torch.load(frame_file, weights_only=True)  # (T, 3, H, W)
            if frames.shape[0] >= num_frames:
                # Sample num_frames evenly
                indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
                frames = frames[indices]
            else:
                continue  # Skip if not enough frames

            video_id = frame_file.stem
            videos.append({
                'frames': frames,
                'caption': f'Video {video_id} (no caption available - local data)',
                'videoid': video_id,
            })
            print(f"  [{len(videos)}/{num_examples}] Video {video_id}")
        except Exception as e:
            print(f"  Error loading {frame_file.name}: {e}")
            continue

    print(f"\nGot {len(videos)} fresh test videos")

    # Generate captions
    print("\n" + "=" * 70)
    print("Generating captions (comparing fine vs coarse)...")
    print("=" * 70)

    results = []

    for i, video in enumerate(videos):
        frames = normalize_frames(video['frames'], device).unsqueeze(0).to(device)

        print(f"\n{'='*60}")
        print(f"Video {i+1}: {video['videoid']}")
        print(f"{'='*60}")
        print(f"Ground Truth: {video['caption']}")
        print()

        result = {
            'video_id': video['videoid'],
            'ground_truth': video['caption'],
            'generations': {}
        }

        # Generate with fine encoding (dynamic queries) - different temperatures
        for temp in [0.3, 0.7]:
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    captions = model.generate_caption(
                        frames,
                        tokenizer,
                        max_new_tokens=60,
                        temperature=temp,
                        use_fine=True
                    )
            gen = captions[0].strip()
            result['generations'][f'fine_t{temp}'] = gen
            print(f"Fine (T={temp}): {gen[:100]}")

        # Generate with coarse encoding (static queries)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                captions_coarse = model.generate_caption(
                    frames,
                    tokenizer,
                    max_new_tokens=60,
                    temperature=0.5,
                    use_fine=False
                )
        gen_coarse = captions_coarse[0].strip()
        result['generations']['coarse_t0.5'] = gen_coarse
        print(f"Coarse (T=0.5): {gen_coarse[:100]}")

        results.append(result)

    # Save results
    output_dir = Path('outputs/fresh_generation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save markdown
    md_path = output_dir / 'results.md'
    with open(md_path, 'w') as f:
        f.write("# Fresh Caption Generation Results\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Model:** multitask checkpoint (step {checkpoint['step']})\n")
        f.write(f"**Videos:** {len(results)} (skipped first {skip_first})\n\n")

        for i, r in enumerate(results):
            f.write(f"## Video {i+1}: `{r['video_id']}`\n\n")
            f.write(f"**Ground Truth:** {r['ground_truth']}\n\n")
            f.write("| Mode | Temperature | Generated Caption |\n")
            f.write("|------|-------------|-------------------|\n")
            for mode, gen in r['generations'].items():
                f.write(f"| {mode.split('_')[0]} | {mode.split('_t')[1] if '_t' in mode else 'N/A'} | {gen[:80]}... |\n")
            f.write("\n")

    print(f"\n{'='*70}")
    print(f"Results saved to:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
