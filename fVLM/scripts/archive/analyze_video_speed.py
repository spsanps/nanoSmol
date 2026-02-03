#!/usr/bin/env python3
"""
Analyze model behavior at different video speeds (temporal sampling rates).
Tests how attention and captions change when we sample frames at different rates.
"""

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


def sample_at_speed(all_frames, num_frames, speed_factor):
    """
    Sample frames at different speeds.
    speed_factor > 1: faster video (skip more frames)
    speed_factor < 1: slower video (sample closer frames)
    speed_factor = 1: normal sampling
    """
    total = all_frames.shape[0]

    if speed_factor >= 1:
        # Faster: sample from wider range, skip frames
        effective_range = min(total, int(total * speed_factor))
        start = 0
        end = effective_range
    else:
        # Slower: sample from narrower range, denser sampling
        effective_range = max(num_frames, int(total * speed_factor))
        # Center the window
        center = total // 2
        start = max(0, center - effective_range // 2)
        end = min(total, start + effective_range)

    indices = np.linspace(start, end - 1, num_frames).astype(int)
    return all_frames[indices], indices


def compute_attention_stats(model, frames_norm, device):
    """Compute attention statistics for fine and coarse passes."""
    with torch.no_grad():
        # Get patch features
        patch_features = model.encoder.get_patch_features(frames_norm[0])

        # Coarse attention
        q_coarse = model.q_coarse.expand(frames_norm.shape[1], -1)
        attn_coarse = model.encoder.compute_attention(q_coarse, patch_features)

        # Fine attention (from coarse features)
        coarse_visual = model.encoder.extract_features(patch_features, q_coarse)
        coarse_proj = model.visual_projection(coarse_visual)
        fine_token = model.fine_start_token.expand(1, 1, -1)
        coarse_seq = torch.cat([fine_token, coarse_proj.unsqueeze(0)], dim=1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            llm_out = model.llm(inputs_embeds=coarse_seq, output_hidden_states=True)

        hidden = llm_out.hidden_states[-1][0, 1:]
        q_fine = model.query_head(hidden)
        attn_fine = model.encoder.compute_attention(q_fine, patch_features)

        # Stats
        coarse_entropy = -(attn_coarse * (attn_coarse + 1e-10).log()).sum(-1).mean().item()
        fine_entropy = -(attn_fine * (attn_fine + 1e-10).log()).sum(-1).mean().item()
        coarse_max = attn_coarse.max(-1).values.mean().item()
        fine_max = attn_fine.max(-1).values.mean().item()

        return {
            'coarse_entropy': coarse_entropy,
            'fine_entropy': fine_entropy,
            'coarse_max': coarse_max,
            'fine_max': fine_max,
            'focus_ratio': fine_max / (coarse_max + 1e-8),
            'attn_fine': attn_fine.cpu(),
            'attn_coarse': attn_coarse.cpu(),
        }


def create_speed_comparison_image(video_id, speed_results, output_path):
    """Create comparison image showing results at different speeds."""
    speeds = list(speed_results.keys())
    n_speeds = len(speeds)

    # Layout
    img_width = 900
    row_height = 200
    header_height = 40
    total_height = header_height + n_speeds * row_height

    img = Image.new('RGB', (img_width, total_height), color=(25, 25, 25))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_title = font_text = font_small = ImageFont.load_default()

    # Header
    draw.text((10, 10), f"Video Speed Analysis: {video_id}", fill=(255, 255, 255), font=font_title)

    # Column headers
    cols = ["Speed", "Sample Frames", "Fine Caption", "Attn Stats"]
    col_widths = [80, 300, 350, 170]
    col_x = [10]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    y = header_height

    for speed in speeds:
        result = speed_results[speed]

        # Speed label
        speed_label = f"{speed}x"
        if speed < 1:
            speed_label += "\n(slow)"
        elif speed > 1:
            speed_label += "\n(fast)"
        else:
            speed_label += "\n(normal)"

        color = (100, 255, 100) if speed == 1.0 else (255, 200, 100) if speed > 1 else (100, 200, 255)
        draw.text((col_x[0], y + 10), speed_label, fill=color, font=font_title)

        # Sample frames visualization
        frames = result['sampled_frames']
        frame_w = 35
        for i, idx in enumerate(result['indices'][:8]):
            fx = col_x[1] + i * (frame_w + 2)
            fy = y + 5

            frame = frames[i].permute(1, 2, 0).numpy()
            if frame.max() > 1:
                frame = frame.astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)

            frame_img = Image.fromarray(frame).resize((frame_w, frame_w))
            img.paste(frame_img, (fx, fy))
            draw.text((fx, fy + frame_w + 2), f"f{idx}", fill=(150, 150, 150), font=font_small)

        # Fine caption
        caption = result['caption_fine'][:250]
        wrapped = textwrap.wrap(caption, width=45)
        caption_text = '\n'.join(wrapped[:6])
        draw.text((col_x[2], y + 5), caption_text, fill=(255, 180, 180), font=font_small)

        # Attention stats
        stats = result['stats']
        stats_text = (
            f"Fine entropy: {stats['fine_entropy']:.3f}\n"
            f"Coarse entropy: {stats['coarse_entropy']:.3f}\n"
            f"Fine max: {stats['fine_max']:.4f}\n"
            f"Coarse max: {stats['coarse_max']:.4f}\n"
            f"Focus ratio: {stats['focus_ratio']:.1f}x"
        )
        draw.text((col_x[3], y + 5), stats_text, fill=(200, 200, 255), font=font_small)

        # Separator
        y += row_height
        draw.line([(10, y - 5), (img_width - 10, y - 5)], fill=(60, 60, 60), width=1)

    img.save(output_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 10
    skip_first = 500

    # Speed factors to test
    speed_factors = [0.25, 0.5, 1.0, 2.0, 4.0]

    output_dir = Path('outputs/speed_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Video Speed Analysis")
    print("=" * 70)
    print(f"Speed factors: {speed_factors}")
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

    # Load frames (need videos with many frames)
    print(f"\nLoading frames (skipping first {skip_first})...")
    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))

    # Filter for videos with enough frames
    valid_files = []
    for f in frame_files[skip_first:]:
        try:
            frames = torch.load(f, weights_only=True)
            if frames.shape[0] >= 32:  # Need enough frames for speed variation
                valid_files.append(f)
            if len(valid_files) >= num_examples:
                break
        except:
            pass

    print(f"Found {len(valid_files)} videos with 32+ frames")

    all_results = []

    for vid_idx, frame_file in enumerate(valid_files):
        video_id = frame_file.stem
        print(f"\n[{vid_idx + 1}/{len(valid_files)}] Processing {video_id}")

        all_frames = torch.load(frame_file, weights_only=True)
        print(f"  Total frames: {all_frames.shape[0]}")

        speed_results = {}

        for speed in speed_factors:
            # Sample at this speed
            sampled_frames, indices = sample_at_speed(all_frames, num_frames, speed)

            # Normalize and process
            frames_norm = normalize_for_dino(sampled_frames, device).unsqueeze(0)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Generate captions
                    caption_fine = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=True
                    )[0].strip()

                    caption_coarse = model.generate_caption(
                        frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=False
                    )[0].strip()

            # Compute attention stats
            stats = compute_attention_stats(model, frames_norm, device)

            speed_results[speed] = {
                'sampled_frames': sampled_frames,
                'indices': indices.tolist(),
                'caption_fine': caption_fine,
                'caption_coarse': caption_coarse,
                'stats': stats,
            }

            print(f"  Speed {speed}x: frames {indices[0]}-{indices[-1]}, "
                  f"focus ratio: {stats['focus_ratio']:.1f}x")

        # Create comparison image
        img_path = output_dir / f'{vid_idx:02d}_{video_id}_speed.png'
        create_speed_comparison_image(video_id, speed_results, img_path)

        all_results.append({
            'video_id': video_id,
            'total_frames': all_frames.shape[0],
            'speed_results': {
                speed: {
                    'indices': r['indices'],
                    'caption_fine': r['caption_fine'],
                    'caption_coarse': r['caption_coarse'],
                    'focus_ratio': r['stats']['focus_ratio'],
                    'fine_entropy': r['stats']['fine_entropy'],
                    'coarse_entropy': r['stats']['coarse_entropy'],
                }
                for speed, r in speed_results.items()
            }
        })

    # Write summary report
    report_path = output_dir / 'SPEED_ANALYSIS.md'
    with open(report_path, 'w') as f:
        f.write("# Video Speed Analysis\n\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"**Videos Analyzed:** {len(all_results)}\n")
        f.write(f"**Speed Factors:** {speed_factors}\n\n")

        f.write("## Summary Statistics\n\n")

        # Aggregate by speed
        f.write("| Speed | Avg Focus Ratio | Avg Fine Entropy | Caption Variation |\n")
        f.write("|-------|-----------------|------------------|-------------------|\n")

        for speed in speed_factors:
            focus_ratios = [r['speed_results'][speed]['focus_ratio'] for r in all_results]
            entropies = [r['speed_results'][speed]['fine_entropy'] for r in all_results]
            avg_focus = np.mean(focus_ratios)
            avg_entropy = np.mean(entropies)

            f.write(f"| {speed}x | {avg_focus:.1f}x | {avg_entropy:.3f} | See below |\n")

        f.write("\n## Caption Consistency Analysis\n\n")
        f.write("How much do captions change with video speed?\n\n")

        # Measure caption similarity across speeds for each video
        for result in all_results:
            f.write(f"### Video: {result['video_id']}\n\n")

            f.write("| Speed | Frame Range | Fine Caption |\n")
            f.write("|-------|-------------|-------------|\n")

            for speed in speed_factors:
                sr = result['speed_results'][speed]
                indices = sr['indices']
                caption = sr['caption_fine'][:80].replace('\n', ' ')
                f.write(f"| {speed}x | {indices[0]}-{indices[-1]} | {caption}... |\n")

            f.write("\n")

        f.write("## Key Findings\n\n")

        # Analyze patterns
        normal_focus = np.mean([r['speed_results'][1.0]['focus_ratio'] for r in all_results])
        slow_focus = np.mean([r['speed_results'][0.25]['focus_ratio'] for r in all_results])
        fast_focus = np.mean([r['speed_results'][4.0]['focus_ratio'] for r in all_results])

        f.write(f"1. **Focus Ratio by Speed:**\n")
        f.write(f"   - Slow (0.25x): {slow_focus:.1f}x\n")
        f.write(f"   - Normal (1.0x): {normal_focus:.1f}x\n")
        f.write(f"   - Fast (4.0x): {fast_focus:.1f}x\n\n")

        if fast_focus > normal_focus * 1.2:
            f.write("   **Finding:** Model focuses MORE when video is faster (more change between frames)\n\n")
        elif slow_focus > normal_focus * 1.2:
            f.write("   **Finding:** Model focuses MORE when video is slower (finer details visible)\n\n")
        else:
            f.write("   **Finding:** Focus ratio relatively stable across speeds\n\n")

        f.write("2. **Caption Stability:**\n")
        # Check if captions are similar across speeds
        similar_count = 0
        for result in all_results:
            captions = [result['speed_results'][s]['caption_fine'][:50] for s in speed_factors]
            # Check word overlap
            words_sets = [set(c.lower().split()) for c in captions]
            base_words = words_sets[2]  # 1.0x speed
            overlaps = [len(base_words & ws) / max(len(base_words), 1) for ws in words_sets]
            if np.mean(overlaps) > 0.5:
                similar_count += 1

        f.write(f"   - {similar_count}/{len(all_results)} videos have consistent captions across speeds\n")
        f.write(f"   - {len(all_results) - similar_count}/{len(all_results)} videos show caption variation\n\n")

        f.write("## Interpretation\n\n")
        f.write("- **If captions are stable across speeds:** Model is NOT using temporal information effectively\n")
        f.write("- **If captions change with speed:** Model IS sensitive to motion/temporal dynamics\n")
        f.write("- **If focus ratio increases with speed:** Model attends more carefully to rapid changes\n")

    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
