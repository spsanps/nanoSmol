#!/usr/bin/env python3
"""
Evaluate caption quality on diverse examples.
Compare generated captions to ground truth.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import requests
import subprocess
import tempfile
from PIL import Image
from datetime import datetime

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
    except:
        return None


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def score_caption(generated, ground_truth):
    """Simple word overlap score."""
    gen_words = set(generated.lower().split())
    gt_words = set(ground_truth.lower().split())

    # Remove common filler words
    stopwords = {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'on', 'at', 'to', 'and', 'with', 'for'}
    gen_words -= stopwords
    gt_words -= stopwords

    if not gt_words:
        return 0.0

    overlap = len(gen_words & gt_words)
    precision = overlap / len(gen_words) if gen_words else 0
    recall = overlap / len(gt_words) if gt_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 30
    skip_first = 500

    print("=" * 70)
    print("Caption Quality Evaluation")
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

    # Stream WebVid
    print(f"\nStreaming WebVid (skipping first {skip_first})...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    results = []
    skipped = 0

    for sample in ds:
        if skipped < skip_first:
            skipped += 1
            continue
        if len(results) >= num_examples:
            break

        frames = download_video(sample['contentUrl'], num_frames)
        if frames is None:
            continue

        ground_truth = sample.get('name', 'Unknown')
        video_id = sample.get('videoid', 'unknown')

        # Generate caption
        frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                caption_fine = model.generate_caption(
                    frames_norm, tokenizer, max_new_tokens=50, temperature=0.5, use_fine=True
                )[0].strip()

                caption_coarse = model.generate_caption(
                    frames_norm, tokenizer, max_new_tokens=50, temperature=0.5, use_fine=False
                )[0].strip()

        # Score
        score_fine = score_caption(caption_fine, ground_truth)
        score_coarse = score_caption(caption_coarse, ground_truth)

        results.append({
            'video_id': video_id,
            'ground_truth': ground_truth,
            'caption_fine': caption_fine,
            'caption_coarse': caption_coarse,
            'score_fine': score_fine,
            'score_coarse': score_coarse,
        })

        print(f"\n[{len(results)}/{num_examples}] Video {video_id}")
        print(f"  GT: {ground_truth[:70]}...")
        print(f"  Fine ({score_fine:.2f}): {caption_fine[:70]}...")
        print(f"  Coarse ({score_coarse:.2f}): {caption_coarse[:70]}...")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    avg_fine = np.mean([r['score_fine'] for r in results])
    avg_coarse = np.mean([r['score_coarse'] for r in results])

    print(f"\nAverage F1 Score:")
    print(f"  Fine (dynamic):  {avg_fine:.3f}")
    print(f"  Coarse (static): {avg_coarse:.3f}")
    print(f"  Improvement:     {(avg_fine - avg_coarse) / (avg_coarse + 1e-8) * 100:.1f}%")

    # Detailed breakdown
    fine_wins = sum(1 for r in results if r['score_fine'] > r['score_coarse'])
    coarse_wins = sum(1 for r in results if r['score_coarse'] > r['score_fine'])
    ties = sum(1 for r in results if r['score_fine'] == r['score_coarse'])

    print(f"\nHead-to-head:")
    print(f"  Fine wins:   {fine_wins} ({fine_wins/len(results)*100:.0f}%)")
    print(f"  Coarse wins: {coarse_wins} ({coarse_wins/len(results)*100:.0f}%)")
    print(f"  Ties:        {ties} ({ties/len(results)*100:.0f}%)")

    # Quality buckets
    good_fine = sum(1 for r in results if r['score_fine'] >= 0.2)
    good_coarse = sum(1 for r in results if r['score_coarse'] >= 0.2)

    print(f"\nGood captions (F1 >= 0.2):")
    print(f"  Fine:   {good_fine}/{len(results)} ({good_fine/len(results)*100:.0f}%)")
    print(f"  Coarse: {good_coarse}/{len(results)} ({good_coarse/len(results)*100:.0f}%)")

    # Save detailed results
    output_dir = Path('outputs/caption_eval')
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / 'RESULTS.md'
    with open(md_path, 'w') as f:
        f.write("# Caption Quality Evaluation\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Samples:** {len(results)}\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Fine | Coarse |\n")
        f.write(f"|--------|------|--------|\n")
        f.write(f"| Avg F1 | {avg_fine:.3f} | {avg_coarse:.3f} |\n")
        f.write(f"| Wins | {fine_wins} | {coarse_wins} |\n")
        f.write(f"| Good (F1>=0.2) | {good_fine} | {good_coarse} |\n\n")

        f.write("## Detailed Results\n\n")
        for i, r in enumerate(results):
            winner = "Fine" if r['score_fine'] > r['score_coarse'] else "Coarse" if r['score_coarse'] > r['score_fine'] else "Tie"
            f.write(f"### {i+1}. Video {r['video_id']} ({winner})\n\n")
            f.write(f"**Ground Truth:** {r['ground_truth']}\n\n")
            f.write(f"**Fine (F1={r['score_fine']:.2f}):** {r['caption_fine']}\n\n")
            f.write(f"**Coarse (F1={r['score_coarse']:.2f}):** {r['caption_coarse']}\n\n")
            f.write("---\n\n")

    print(f"\nDetailed results saved to {md_path}")


if __name__ == "__main__":
    main()
