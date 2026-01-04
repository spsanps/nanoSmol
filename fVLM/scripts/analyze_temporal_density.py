#!/usr/bin/env python3
"""
Analyze model behavior at different temporal densities.
Tests how captions change with varying numbers of input frames.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
import textwrap
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


def get_sampling_configs():
    """Define different temporal sampling configurations."""
    return {
        '4_sparse': {
            'indices': [0, 5, 10, 15],
            'description': '4 frames (sparse)',
            'color': (255, 100, 100),
        },
        '8_uniform': {
            'indices': [0, 2, 4, 6, 9, 11, 13, 15],
            'description': '8 frames (uniform)',
            'color': (255, 200, 100),
        },
        '16_dense': {
            'indices': list(range(16)),
            'description': '16 frames (dense)',
            'color': (100, 255, 100),
        },
        'first_half': {
            'indices': list(range(8)),
            'description': 'First 8 frames',
            'color': (100, 200, 255),
        },
        'second_half': {
            'indices': list(range(8, 16)),
            'description': 'Last 8 frames',
            'color': (200, 100, 255),
        },
    }


def word_overlap(cap1, cap2):
    """Compute word overlap between two captions."""
    stopwords = {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'on', 'at', 'to', 'and', 'with', 'for', 'by', '.', ','}
    words1 = set(cap1.lower().split()) - stopwords
    words2 = set(cap2.lower().split()) - stopwords
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def create_comparison_image(video_id, all_frames, config_results, output_path):
    """Create comprehensive comparison image."""
    configs = get_sampling_configs()
    n_configs = len(configs)

    # Layout
    img_width = 1000
    row_height = 140
    header_height = 50
    total_height = header_height + n_configs * row_height

    img = Image.new('RGB', (img_width, total_height), color=(25, 25, 25))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_title = font_text = font_small = ImageFont.load_default()

    # Header
    draw.text((10, 10), f"Temporal Density Analysis: {video_id}", fill=(255, 255, 255), font=font_title)
    draw.text((10, 30), "How captions change with different frame sampling", fill=(150, 150, 150), font=font_small)

    y = header_height

    for config_name, config in configs.items():
        result = config_results[config_name]

        # Config label
        draw.rectangle([(5, y + 5), (130, y + 25)], fill=tuple(c // 3 for c in config['color']))
        draw.text((10, y + 7), config['description'], fill=config['color'], font=font_text)

        # Sample frames visualization
        indices = config['indices']
        frame_w = 35
        x_start = 140
        for i, idx in enumerate(indices[:8]):
            fx = x_start + i * (frame_w + 2)
            fy = y + 10

            frame = all_frames[idx].permute(1, 2, 0).numpy()
            if frame.max() > 1:
                frame = frame.astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)

            frame_img = Image.fromarray(frame).resize((frame_w, frame_w))
            img.paste(frame_img, (fx, fy))

        # More frames indicator
        if len(indices) > 8:
            draw.text((x_start + 8 * (frame_w + 2), y + 25), f"+{len(indices)-8}", fill=(100, 100, 100), font=font_small)

        # Fine Caption
        caption_fine = result['caption_fine'][:180].replace('\n', ' ')
        wrapped = textwrap.wrap(caption_fine, width=55)
        caption_text = '\n'.join(wrapped[:3])
        draw.text((450, y + 5), "Fine:", fill=(255, 150, 150), font=font_text)
        draw.text((490, y + 5), caption_text, fill=(255, 220, 220), font=font_small)

        # Coarse Caption (smaller)
        caption_coarse = result['caption_coarse'][:100].replace('\n', ' ')
        draw.text((450, y + 55), "Coarse:", fill=(150, 150, 255), font=font_small)
        draw.text((500, y + 55), caption_coarse[:60] + "...", fill=(200, 200, 255), font=font_small)

        # Overlap score
        overlap = result.get('overlap_with_dense', 0)
        draw.text((450, y + 75), f"Overlap with 16-frame: {overlap:.2f}", fill=(150, 150, 150), font=font_small)

        # Separator
        y += row_height
        draw.line([(10, y - 5), (img_width - 10, y - 5)], fill=(50, 50, 50), width=1)

    img.save(output_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_examples = 15
    skip_first = 1500

    configs = get_sampling_configs()

    output_dir = Path('outputs/temporal_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Temporal Density Analysis")
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

    all_results = []
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

            for config_name, config in configs.items():
                indices = config['indices']
                sampled_frames = all_frames[indices]

                # Normalize and process
                frames_norm = normalize_for_dino(sampled_frames, device).unsqueeze(0)

                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        caption_fine = model.generate_caption(
                            frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=True
                        )[0].strip()

                        caption_coarse = model.generate_caption(
                            frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=False
                        )[0].strip()

                config_results[config_name] = {
                    'indices': indices,
                    'caption_fine': caption_fine,
                    'caption_coarse': caption_coarse,
                }

                print(f"  {config_name}: {caption_fine[:50]}...")

            # Compute overlaps with dense (16-frame) baseline
            dense_caption = config_results['16_dense']['caption_fine']
            for config_name in configs.keys():
                overlap = word_overlap(dense_caption, config_results[config_name]['caption_fine'])
                config_results[config_name]['overlap_with_dense'] = overlap

            # Create comparison image
            img_path = output_dir / f'{processed:02d}_{video_id}_temporal.png'
            create_comparison_image(video_id, all_frames, config_results, img_path)

            all_results.append({
                'video_id': video_id,
                'config_results': config_results,
            })

            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Write summary report
    report_path = output_dir / 'TEMPORAL_ANALYSIS.md'
    with open(report_path, 'w') as f:
        f.write("# Temporal Density Analysis\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Videos Analyzed:** {len(all_results)}\n\n")

        f.write("## Question Being Answered\n\n")
        f.write("**Does changing video speed (frame sampling rate) affect model captions?**\n\n")
        f.write("- If captions stay similar: Model NOT using temporal information effectively\n")
        f.write("- If captions change significantly: Model IS sensitive to video speed/content\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Configuration | Frames | Avg Overlap with 16-frame |\n")
        f.write("|---------------|--------|---------------------------|\n")

        for config_name, config in configs.items():
            overlaps = [r['config_results'][config_name]['overlap_with_dense'] for r in all_results]
            n_frames = len(config['indices'])
            f.write(f"| {config['description']} | {n_frames} | {np.mean(overlaps):.3f} |\n")

        # Calculate key metrics
        sparse_overlap = np.mean([r['config_results']['4_sparse']['overlap_with_dense'] for r in all_results])
        first_half_overlap = np.mean([r['config_results']['first_half']['overlap_with_dense'] for r in all_results])
        second_half_overlap = np.mean([r['config_results']['second_half']['overlap_with_dense'] for r in all_results])

        f.write("\n## Key Findings\n\n")

        f.write("### 1. Frame Count Sensitivity\n\n")
        if sparse_overlap > 0.5:
            f.write(f"- **4 frames vs 16 frames overlap: {sparse_overlap:.2f}** (HIGH)\n")
            f.write("- **Interpretation:** Captions are STABLE despite 4x fewer frames\n")
            f.write("- **Implication:** Model is NOT fully utilizing temporal information\n\n")
        else:
            f.write(f"- **4 frames vs 16 frames overlap: {sparse_overlap:.2f}** (LOW)\n")
            f.write("- **Interpretation:** Captions CHANGE significantly with fewer frames\n")
            f.write("- **Implication:** Model IS sensitive to temporal density\n\n")

        f.write("### 2. First Half vs Second Half\n\n")
        f.write(f"- First half overlap with full video: {first_half_overlap:.2f}\n")
        f.write(f"- Second half overlap with full video: {second_half_overlap:.2f}\n")

        if abs(first_half_overlap - second_half_overlap) < 0.1:
            f.write("- **Interpretation:** Both halves contribute equally to caption\n\n")
        elif first_half_overlap > second_half_overlap:
            f.write("- **Interpretation:** First half dominates the caption\n\n")
        else:
            f.write("- **Interpretation:** Second half dominates the caption\n\n")

        f.write("### 3. Fine vs Coarse Consistency\n\n")
        fine_coarse_overlaps = []
        for r in all_results:
            for config_name in configs.keys():
                cap_fine = r['config_results'][config_name]['caption_fine']
                cap_coarse = r['config_results'][config_name]['caption_coarse']
                fine_coarse_overlaps.append(word_overlap(cap_fine, cap_coarse))

        f.write(f"- Average Fine-Coarse overlap: {np.mean(fine_coarse_overlaps):.2f}\n")
        if np.mean(fine_coarse_overlaps) < 0.3:
            f.write("- **Interpretation:** Fine and Coarse produce DIFFERENT captions\n")
        else:
            f.write("- **Interpretation:** Fine and Coarse produce SIMILAR captions\n")

        f.write("\n## Detailed Results by Video\n\n")

        for result in all_results:
            f.write(f"### Video: {result['video_id']}\n\n")
            f.write("| Config | Overlap | Fine Caption |\n")
            f.write("|--------|---------|-------------|\n")
            for config_name, config in configs.items():
                cr = result['config_results'][config_name]
                caption = cr['caption_fine'][:70].replace('\n', ' ').replace('|', '/')
                overlap = cr['overlap_with_dense']
                f.write(f"| {config['description']} | {overlap:.2f} | {caption}... |\n")
            f.write("\n")

        f.write("## Conclusion for Management Report\n\n")

        if sparse_overlap > 0.5:
            f.write("**Caption stability across frame counts indicates the model is NOT utilizing temporal dynamics effectively.**\n\n")
            f.write("The model produces similar captions whether it sees 4 frames or 16 frames, suggesting:\n")
            f.write("1. Temporal information is being collapsed/ignored\n")
            f.write("2. The caption is based on aggregate visual features, not temporal understanding\n")
            f.write("3. This aligns with the `loss_fine ≈ loss_coarse` finding - dynamic attention isn't helping\n")
        else:
            f.write("**Captions DO change with frame count, indicating some temporal sensitivity.**\n\n")
            f.write("However, the semantic accuracy of these captions should be evaluated against ground truth.\n")

    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Images: {processed} comparison images generated")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
