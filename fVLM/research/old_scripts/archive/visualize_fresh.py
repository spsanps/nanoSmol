#!/usr/bin/env python3
"""
Comprehensive visualization for fresh samples using the multitask-trained model.

Generates:
1. Attention grids (coarse vs fine)
2. Comparison GIFs
3. Attention statistics
4. Caption comparison (fine vs coarse)
5. Detailed analysis markdown
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# Constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dino(frames, device):
    """Normalize frames for DINO (from uint8)."""
    frames_norm = frames.to(device).float() / 255.0
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    frames_norm = (frames_norm - mean) / std
    return frames_norm


def get_attention_weights(query, patch_features, dino_dim):
    """Compute attention weights for a query over patch features."""
    q_embed = query.unsqueeze(1)  # [B, 1, D]
    attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))  # [B, 1, N+1]
    attn_weights = torch.softmax(attn_scores / (dino_dim ** 0.5), dim=-1)
    return attn_weights.squeeze(1)  # [B, N+1]


@torch.no_grad()
def extract_attention_maps(model, frames, device):
    """Extract attention maps for both coarse and fine queries."""
    B = 1
    T = frames.shape[0]

    frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = []
    for t in range(T):
        all_caches.append({'patch_features': patch_features[:, t]})

    # Pass 1: Coarse with static query
    q_static = model.q_static.expand(B, -1)
    q_static_proj = model.encoder.query_input_proj(q_static)

    coarse_attn_list = []
    z_coarse_list = []

    for t in range(T):
        pf = all_caches[t]['patch_features']
        attn = get_attention_weights(q_static_proj, pf, model.encoder.dino_dim)
        coarse_attn_list.append(attn[:, 1:])
        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)

    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    coarse_token = model.coarse_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state

    h_for_queries = h_pass1[:, N_text + 1:]
    queries = model.llm_to_query(h_for_queries)

    # Pass 2: Fine with dynamic queries
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    fine_attn_list = []

    for t in range(T):
        q_t = shifted_q[:, t]
        q_t_proj = model.encoder.query_input_proj(q_t)
        pf = all_caches[t]['patch_features']
        attn = get_attention_weights(q_t_proj, pf, model.encoder.dino_dim)
        fine_attn_list.append(attn[:, 1:])

    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)
    fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)

    n_patches = coarse_attn.shape[1]
    grid_size = int(n_patches ** 0.5)

    coarse_attn = coarse_attn.reshape(T, grid_size, grid_size)
    fine_attn = fine_attn.reshape(T, grid_size, grid_size)

    return coarse_attn.cpu(), fine_attn.cpu()


def create_attention_overlay(frame, attn_map, alpha=0.5):
    """Overlay attention heatmap on frame."""
    H, W = frame.shape[1], frame.shape[2]

    attn_up = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Normalize relative to see structure
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    cmap = plt.cm.jet
    heatmap = cmap(attn_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    if frame_np.max() > 1:
        frame_np = frame_np / 255.0
    frame_np = (frame_np * 255).astype(np.uint8)

    overlaid = (1 - alpha) * frame_np + alpha * heatmap
    overlaid = overlaid.astype(np.uint8)

    return overlaid


def create_full_grid(frames, coarse_attn, fine_attn, caption_fine, caption_coarse, output_path, video_id):
    """Create comprehensive grid with frames, attention, and captions."""
    T = min(8, frames.shape[0])  # Show 8 frames
    indices = np.linspace(0, frames.shape[0] - 1, T).astype(int)

    fig = plt.figure(figsize=(20, 14))

    # Create grid spec for layout
    gs = fig.add_gridspec(4, T, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.05)

    for i, t in enumerate(indices):
        frame = frames[t]
        if frame.max() > 1:
            frame_display = frame.float() / 255.0
        else:
            frame_display = frame.float()

        # Row 1: Original frames
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(frame_display.permute(1, 2, 0).numpy())
        ax1.set_title(f'Frame {t}', fontsize=9)
        ax1.axis('off')

        # Row 2: Coarse attention
        ax2 = fig.add_subplot(gs[1, i])
        coarse_overlay = create_attention_overlay(frame, coarse_attn[t], alpha=0.5)
        ax2.imshow(coarse_overlay)
        if i == 0:
            ax2.set_ylabel('Coarse\n(static)', fontsize=10, color='blue')
        ax2.axis('off')

        # Row 3: Fine attention
        ax3 = fig.add_subplot(gs[2, i])
        fine_overlay = create_attention_overlay(frame, fine_attn[t], alpha=0.5)
        ax3.imshow(fine_overlay)
        if i == 0:
            ax3.set_ylabel('Fine\n(dynamic)', fontsize=10, color='red')
        ax3.axis('off')

    # Row 4: Captions
    ax_cap = fig.add_subplot(gs[3, :])
    ax_cap.axis('off')

    caption_text = f"""Video: {video_id}

FINE Caption (T=0.5):
{caption_fine[:200]}{'...' if len(caption_fine) > 200 else ''}

COARSE Caption (T=0.5):
{caption_coarse[:200]}{'...' if len(caption_coarse) > 200 else ''}"""

    ax_cap.text(0.02, 0.95, caption_text, transform=ax_cap.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'Video {video_id} - Attention Analysis', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_comparison_gif(frames, coarse_attn, fine_attn, output_path, video_id, fps=8):
    """Create side-by-side comparison GIF with fast playback."""
    T = frames.shape[0]

    gif_frames = []
    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        # Original frame
        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() > 1:
            frame_np = (frame_np / 255.0 * 255).astype(np.uint8)
        else:
            frame_np = (frame_np * 255).astype(np.uint8)

        coarse_overlay = create_attention_overlay(frame, coarse_attn[t])
        fine_overlay = create_attention_overlay(frame, fine_attn[t])

        # Difference heatmap
        diff = fine_attn[t] - coarse_attn[t]
        diff_up = F.interpolate(
            diff.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
        diff_norm = (diff_up / diff_max + 1) / 2

        cmap = plt.cm.RdBu_r
        diff_colored = cmap(diff_norm)[:, :, :3]
        diff_overlay = (diff_colored * 255).astype(np.uint8)

        # Combine panels
        header_h = 35
        combined = np.zeros((H + header_h, W * 4, 3), dtype=np.uint8)
        combined[header_h:, :W] = frame_np
        combined[header_h:, W:2*W] = coarse_overlay
        combined[header_h:, 2*W:3*W] = fine_overlay
        combined[header_h:, 3*W:] = diff_overlay

        # Header colors
        combined[:header_h, :W] = [60, 60, 60]
        combined[:header_h, W:2*W] = [40, 40, 150]
        combined[:header_h, 2*W:3*W] = [150, 40, 40]
        combined[:header_h, 3*W:] = [40, 120, 40]

        # Add text
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        labels = [f"Frame {t}", "Coarse (static)", "Fine (dynamic)", "Difference"]
        for i, label in enumerate(labels):
            x = i * W + W // 2 - len(label) * 4
            draw.text((x, 10), label, fill=(255, 255, 255), font=font)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def plot_attention_stats(coarse_attn, fine_attn, output_path, video_id):
    """Plot attention statistics."""
    T = coarse_attn.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Attention Statistics - Video {video_id}', fontsize=12, fontweight='bold')

    # Entropy
    coarse_entropy = []
    fine_entropy = []
    for t in range(T):
        c = coarse_attn[t].flatten()
        f = fine_attn[t].flatten()
        c = c / c.sum()
        f = f / f.sum()
        coarse_entropy.append(-(c * torch.log(c + 1e-10)).sum().item())
        fine_entropy.append(-(f * torch.log(f + 1e-10)).sum().item())

    axes[0, 0].plot(coarse_entropy, 'b-o', label='Coarse', linewidth=2)
    axes[0, 0].plot(fine_entropy, 'r-o', label='Fine', linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Attention Entropy (lower = more focused)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Max attention
    coarse_max = [coarse_attn[t].max().item() for t in range(T)]
    fine_max = [fine_attn[t].max().item() for t in range(T)]

    axes[0, 1].plot(coarse_max, 'b-o', label='Coarse', linewidth=2)
    axes[0, 1].plot(fine_max, 'r-o', label='Fine', linewidth=2)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Max Attention')
    axes[0, 1].set_title('Peak Attention Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Mean difference heatmap
    diff = (fine_attn - coarse_attn).mean(dim=0)
    im = axes[1, 0].imshow(diff.numpy(), cmap='RdBu_r', vmin=-0.005, vmax=0.005)
    axes[1, 0].set_title('Mean(Fine - Coarse)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])

    # Ratio over time
    ratio = []
    for t in range(T):
        c_max = coarse_attn[t].max().item()
        f_max = fine_attn[t].max().item()
        ratio.append(f_max / (c_max + 1e-8))

    axes[1, 1].plot(ratio, 'g-o', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Fine/Coarse Ratio')
    axes[1, 1].set_title('Peak Attention Ratio (>1 = fine more focused)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 10
    skip_first = 1000  # Skip more to get truly fresh samples

    output_dir = Path('outputs/fresh_detailed')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fresh Detailed Visualization - Multitask Model")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    checkpoint_path = Path('outputs/multitask/checkpoints/final.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = {
        'dino_model': 'facebook/dinov2-small',
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_dim': 384,
        'llm_dim': 576,
        'query_dim': 128,
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

    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load local frames
    print(f"\nLoading local frames (skipping first {skip_first})...")
    frames_dir = Path('data/webvid_large/frames')
    frame_files = sorted(list(frames_dir.glob('*.pt')))
    available = frame_files[skip_first:]

    results = []

    for i, frame_file in enumerate(available[:num_examples]):
        video_id = frame_file.stem
        print(f"\n{'='*60}")
        print(f"[{i+1}/{num_examples}] Video {video_id}")
        print(f"{'='*60}")

        try:
            frames = torch.load(frame_file, weights_only=True)
            if frames.shape[0] < num_frames:
                print(f"  Skipping - not enough frames ({frames.shape[0]})")
                continue

            # Sample frames evenly
            indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
            frames = frames[indices]

            # Extract attention maps
            print("  Extracting attention maps...")
            coarse_attn, fine_attn = extract_attention_maps(model, frames, device)

            # Generate captions
            print("  Generating captions...")
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                caption_fine = model.generate_caption(
                    frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=True
                )[0].strip()

                caption_coarse = model.generate_caption(
                    frames_norm, tokenizer, max_new_tokens=60, temperature=0.5, use_fine=False
                )[0].strip()

            print(f"  Fine: {caption_fine[:80]}...")
            print(f"  Coarse: {caption_coarse[:80]}...")

            # Create visualizations
            print("  Creating visualizations...")

            # 1. Full grid
            create_full_grid(
                frames, coarse_attn, fine_attn,
                caption_fine, caption_coarse,
                output_dir / f'grid_{i:02d}_{video_id}.png',
                video_id
            )

            # 2. Comparison GIF
            create_comparison_gif(
                frames, coarse_attn, fine_attn,
                output_dir / f'comparison_{i:02d}_{video_id}.gif',
                video_id
            )

            # 3. Stats
            plot_attention_stats(
                coarse_attn, fine_attn,
                output_dir / f'stats_{i:02d}_{video_id}.png',
                video_id
            )

            # Compute metrics
            coarse_entropy = -(coarse_attn.flatten() * torch.log(coarse_attn.flatten() + 1e-10)).mean().item()
            fine_entropy = -(fine_attn.flatten() * torch.log(fine_attn.flatten() + 1e-10)).mean().item()

            results.append({
                'video_id': video_id,
                'caption_fine': caption_fine,
                'caption_coarse': caption_coarse,
                'coarse_entropy': coarse_entropy,
                'fine_entropy': fine_entropy,
                'coarse_max': coarse_attn.max().item(),
                'fine_max': fine_attn.max().item(),
            })

            print(f"  Entropy - Coarse: {coarse_entropy:.3f}, Fine: {fine_entropy:.3f}")
            print(f"  Max Attn - Coarse: {coarse_attn.max():.4f}, Fine: {fine_attn.max():.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save summary markdown
    md_path = output_dir / 'ANALYSIS.md'
    with open(md_path, 'w') as f:
        f.write("# Fresh Sample Detailed Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Model:** Multitask checkpoint (step {checkpoint['step']})\n")
        f.write(f"**Samples:** {len(results)} videos (skipped first {skip_first})\n\n")

        f.write("## Summary Metrics\n\n")
        f.write("| Video | Coarse Entropy | Fine Entropy | Coarse Max | Fine Max | Ratio |\n")
        f.write("|-------|----------------|--------------|------------|----------|-------|\n")

        for r in results:
            ratio = r['fine_max'] / (r['coarse_max'] + 1e-8)
            f.write(f"| {r['video_id']} | {r['coarse_entropy']:.3f} | {r['fine_entropy']:.3f} | ")
            f.write(f"{r['coarse_max']:.4f} | {r['fine_max']:.4f} | {ratio:.3f} |\n")

        f.write("\n## Caption Comparison\n\n")
        for i, r in enumerate(results):
            f.write(f"### Video {i+1}: {r['video_id']}\n\n")
            f.write(f"**Fine Caption:** {r['caption_fine']}\n\n")
            f.write(f"**Coarse Caption:** {r['caption_coarse']}\n\n")
            f.write("---\n\n")

        # Overall analysis
        avg_coarse_entropy = np.mean([r['coarse_entropy'] for r in results])
        avg_fine_entropy = np.mean([r['fine_entropy'] for r in results])
        avg_ratio = np.mean([r['fine_max'] / (r['coarse_max'] + 1e-8) for r in results])

        f.write("## Key Findings\n\n")
        f.write(f"- **Average Coarse Entropy:** {avg_coarse_entropy:.3f}\n")
        f.write(f"- **Average Fine Entropy:** {avg_fine_entropy:.3f}\n")
        f.write(f"- **Average Fine/Coarse Ratio:** {avg_ratio:.3f}\n\n")

        if avg_ratio > 1.05:
            f.write("✅ Fine queries show more focused attention than coarse\n")
        elif avg_ratio < 0.95:
            f.write("⚠️ Coarse queries show more focused attention than fine\n")
        else:
            f.write("➡️ Fine and coarse queries show similar attention focus\n")

    print(f"\n{'='*70}")
    print(f"Results saved to {output_dir}")
    print(f"  - grid_XX_ID.png: Full visualization grids")
    print(f"  - comparison_XX_ID.gif: Side-by-side animations")
    print(f"  - stats_XX_ID.png: Attention statistics")
    print(f"  - ANALYSIS.md: Summary report")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
