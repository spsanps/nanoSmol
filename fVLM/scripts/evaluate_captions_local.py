#!/usr/bin/env python3
"""
Evaluate caption quality using local precomputed frames.
Since we don't have ground truth for local data, we'll:
1. Compare fine vs coarse caption diversity
2. Check for repetition/quality issues
3. Measure semantic coherence
"""

import torch
import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dino(frames, device):
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames_norm - mean) / std


def analyze_caption_quality(caption):
    """Analyze caption quality metrics."""
    words = caption.lower().split()

    # Repetition score (lower is better)
    if len(words) < 2:
        repetition = 0
    else:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        repetition = 1 - (unique_bigrams / len(bigrams)) if bigrams else 0

    # Resolution spam (mentions of 4k, fps, etc.)
    tech_terms = ['4k', '1080p', '720p', 'fps', '60fps', '30fps', 'hd', 'uhd', '2160', '3840', '2560']
    tech_count = sum(1 for word in words if any(t in word for t in tech_terms))
    tech_ratio = tech_count / len(words) if words else 0

    # Sentence completeness (ends with period, has verb-like structure)
    has_ending = caption.strip().endswith(('.', '!', '?'))

    # Length
    length = len(words)

    # Diversity (unique words / total words)
    diversity = len(set(words)) / len(words) if words else 0

    return {
        'repetition': repetition,
        'tech_spam': tech_ratio,
        'has_ending': has_ending,
        'length': length,
        'diversity': diversity,
    }


def compute_caption_similarity(cap1, cap2):
    """Compute word overlap between two captions."""
    words1 = set(cap1.lower().split())
    words2 = set(cap2.lower().split())

    stopwords = {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'on', 'at', 'to', 'and', 'with', 'for', 'by'}
    words1 -= stopwords
    words2 -= stopwords

    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    jaccard = overlap / len(words1 | words2)
    return jaccard


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 50
    skip_first = 2000  # Use different samples than before

    print("=" * 70)
    print("Caption Quality Analysis (Local Data)")
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

    # Load local frames
    print(f"\nLoading local frames (skipping first {skip_first})...")
    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))[skip_first:skip_first + num_examples]

    results = []

    for i, frame_file in enumerate(frame_files):
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
                        frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=True
                    )[0].strip()

                    caption_coarse = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=False
                    )[0].strip()

            # Analyze quality
            quality_fine = analyze_caption_quality(caption_fine)
            quality_coarse = analyze_caption_quality(caption_coarse)
            similarity = compute_caption_similarity(caption_fine, caption_coarse)

            results.append({
                'video_id': video_id,
                'caption_fine': caption_fine,
                'caption_coarse': caption_coarse,
                'quality_fine': quality_fine,
                'quality_coarse': quality_coarse,
                'fine_coarse_similarity': similarity,
            })

            print(f"\n[{i+1}/{len(frame_files)}] Video {video_id}")
            print(f"  Fine: {caption_fine[:80]}...")
            print(f"  Coarse: {caption_coarse[:80]}...")
            print(f"  Similarity: {similarity:.2f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("CAPTION QUALITY SUMMARY")
    print("=" * 70)

    # Aggregate metrics
    avg_metrics = {
        'fine': {k: np.mean([r['quality_fine'][k] for r in results]) for k in results[0]['quality_fine']},
        'coarse': {k: np.mean([r['quality_coarse'][k] for r in results]) for k in results[0]['quality_coarse']},
    }
    avg_similarity = np.mean([r['fine_coarse_similarity'] for r in results])

    print(f"\n{'Metric':<20} {'Fine':<12} {'Coarse':<12} {'Better':<10}")
    print("-" * 54)

    metrics_to_show = [
        ('Repetition', 'repetition', 'lower'),
        ('Tech Spam', 'tech_spam', 'lower'),
        ('Diversity', 'diversity', 'higher'),
        ('Length', 'length', 'moderate'),
    ]

    for name, key, prefer in metrics_to_show:
        fine_val = avg_metrics['fine'][key]
        coarse_val = avg_metrics['coarse'][key]

        if prefer == 'lower':
            better = 'Fine' if fine_val < coarse_val else 'Coarse' if coarse_val < fine_val else 'Tie'
        elif prefer == 'higher':
            better = 'Fine' if fine_val > coarse_val else 'Coarse' if coarse_val > fine_val else 'Tie'
        else:
            better = '-'

        print(f"{name:<20} {fine_val:<12.3f} {coarse_val:<12.3f} {better:<10}")

    print(f"\nFine-Coarse Similarity: {avg_similarity:.3f}")
    print(f"  (Lower = more diverse captions between modes)")

    # Categorize caption themes
    print("\n" + "=" * 70)
    print("CAPTION THEMES DETECTED")
    print("=" * 70)

    themes = {
        'aerial/drone': ['aerial', 'drone', 'flying', 'sky', 'above'],
        'nature': ['forest', 'mountain', 'river', 'lake', 'tree', 'flower', 'nature'],
        'urban': ['city', 'street', 'building', 'urban', 'downtown'],
        'people': ['man', 'woman', 'girl', 'boy', 'person', 'people'],
        'abstract': ['abstract', 'background', 'animation', 'loop', 'seamless'],
        'water': ['ocean', 'sea', 'water', 'beach', 'wave'],
    }

    theme_counts = {theme: {'fine': 0, 'coarse': 0} for theme in themes}

    for r in results:
        for theme, keywords in themes.items():
            if any(kw in r['caption_fine'].lower() for kw in keywords):
                theme_counts[theme]['fine'] += 1
            if any(kw in r['caption_coarse'].lower() for kw in keywords):
                theme_counts[theme]['coarse'] += 1

    print(f"\n{'Theme':<15} {'Fine':<10} {'Coarse':<10}")
    print("-" * 35)
    for theme, counts in theme_counts.items():
        print(f"{theme:<15} {counts['fine']:<10} {counts['coarse']:<10}")

    # Sample outputs
    print("\n" + "=" * 70)
    print("SAMPLE CAPTIONS")
    print("=" * 70)

    for r in results[:10]:
        print(f"\nVideo {r['video_id']}:")
        print(f"  Fine: {r['caption_fine'][:100]}")
        print(f"  Coarse: {r['caption_coarse'][:100]}")

    # Save results
    output_dir = Path('outputs/caption_eval')
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / 'QUALITY_ANALYSIS.md'
    with open(md_path, 'w') as f:
        f.write("# Caption Quality Analysis\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Samples:** {len(results)}\n\n")

        f.write("## Quality Metrics\n\n")
        f.write(f"| Metric | Fine | Coarse |\n")
        f.write(f"|--------|------|--------|\n")
        for name, key, _ in metrics_to_show:
            f.write(f"| {name} | {avg_metrics['fine'][key]:.3f} | {avg_metrics['coarse'][key]:.3f} |\n")
        f.write(f"| Fine-Coarse Similarity | {avg_similarity:.3f} | - |\n\n")

        f.write("## All Captions\n\n")
        for i, r in enumerate(results):
            f.write(f"### {i+1}. Video {r['video_id']}\n\n")
            f.write(f"**Fine:** {r['caption_fine']}\n\n")
            f.write(f"**Coarse:** {r['caption_coarse']}\n\n")
            f.write(f"*Similarity: {r['fine_coarse_similarity']:.2f}*\n\n")
            f.write("---\n\n")

    print(f"\nResults saved to {md_path}")


if __name__ == "__main__":
    main()
