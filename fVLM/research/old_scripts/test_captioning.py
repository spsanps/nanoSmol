"""
Test video captioning capability.

The model wasn't trained for captioning yet, but let's see what it outputs
when we use the LLM's language head after video features.
"""

import torch
import sys
from pathlib import Path
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
        print(f"Error: {e}")
        return None


def normalize_frames(frames):
    """Normalize for DINO."""
    frames = frames.float() / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16

    print("=" * 70)
    print("Testing Video Captioning")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    checkpoint_path = Path('outputs/phase2/checkpoints/final.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    model_cfg = config['model']

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

    # Get test videos - try local first, then WebVid with retry
    print("\nGetting test videos...")

    videos = []

    # Try local videos first
    local_video_dir = Path('data/videos')
    if local_video_dir.exists():
        import decord
        decord.bridge.set_bridge('torch')
        local_vids = list(local_video_dir.glob('**/*.mp4'))[:5]
        for vp in local_vids:
            try:
                vr = decord.VideoReader(str(vp))
                indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
                frames = vr.get_batch(indices).permute(0, 3, 1, 2)
                if frames.shape[2] != 256 or frames.shape[3] != 256:
                    frames = torch.nn.functional.interpolate(
                        frames.float(), size=(256, 256), mode='bilinear'
                    ).byte()
                videos.append({'frames': frames, 'caption': vp.stem})
                print(f"  [{len(videos)}/5] Local: {vp.name[:50]}...")
            except Exception as e:
                continue
            if len(videos) >= 5:
                break

    # If no local videos, try WebVid with retry
    if len(videos) < 5:
        import time
        for retry in range(3):
            try:
                ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
                for sample in ds:
                    if len(videos) >= 5:
                        break
                    frames = download_video(sample['contentUrl'], num_frames)
                    if frames is not None:
                        videos.append({
                            'frames': frames,
                            'caption': sample.get('name', 'Unknown'),
                        })
                        print(f"  [{len(videos)}/5] {sample.get('name', '')[:60]}...")
                break
            except Exception as e:
                print(f"  Retry {retry+1}/3: {e}")
                time.sleep(2 ** retry)

    print(f"\nGot {len(videos)} test videos")

    # Generate captions
    print("\n" + "=" * 70)
    print("Generating captions...")
    print("=" * 70)

    for i, video in enumerate(videos):
        frames = normalize_frames(video['frames']).unsqueeze(0).to(device)

        print(f"\n--- Video {i+1} ---")
        print(f"Ground truth: {video['caption']}")

        # Try different temperatures
        for temp in [0.3, 0.7, 1.0]:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                captions = model.generate_caption(
                    frames,
                    tokenizer,
                    max_new_tokens=50,
                    temperature=temp,
                    use_fine=True
                )
            print(f"Generated (T={temp}): {captions[0][:100]}")

        # Also try coarse (static) encoding
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            captions_coarse = model.generate_caption(
                frames,
                tokenizer,
                max_new_tokens=50,
                temperature=0.7,
                use_fine=False
            )
        print(f"Generated (coarse): {captions_coarse[0][:100]}")

    print("\n" + "=" * 70)
    print("Note: Model wasn't trained for captioning - these are just")
    print("raw LLM outputs after video features. Need captioning training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
