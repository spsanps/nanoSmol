"""
Visualize where fine (dynamic) queries attend in teacher forcing mode.

Shows:
1. Attention heatmaps overlaid on frames
2. Comparison of static (coarse) vs dynamic (fine) attention
3. How attention shifts over time
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
import subprocess
import tempfile
import re
from datasets import load_dataset
from diffusers import AutoencoderKL
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# Constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str):
    match = re.match(r'PT(\d+)H(\d+)M(\d+)S', dur_str)
    if match:
        return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
    return 0


def download_video(url, num_frames=16, frame_size=256):
    """Download a video and extract frames."""
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
                    '-vf', f'scale={frame_size}:{frame_size}:force_original_aspect_ratio=increase,crop={frame_size}:{frame_size}',
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
        finally:
            Path(temp_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"Download error: {e}")
        return None


def get_test_videos(num_videos=8, num_frames=16, frame_size=256):
    """Download test videos from WebVid."""
    print(f"Downloading {num_videos} test videos...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    videos = []
    tried = 0
    for sample in ds:
        if len(videos) >= num_videos:
            break

        duration = parse_duration(sample['duration'])
        if duration < 8 or duration > 60:
            continue

        tried += 1
        frames = download_video(sample['contentUrl'], num_frames, frame_size)
        if frames is not None:
            videos.append({
                'frames': frames,
                'caption': sample['name'],
                'video_id': sample['videoid']
            })
            print(f"  Downloaded {len(videos)}/{num_videos}")

        if tried > num_videos * 5:
            break

    return videos


def normalize_for_dino(frames):
    """Normalize frames for DINO (from uint8)."""
    frames_norm = frames.float() / 255.0
    mean = IMAGENET_MEAN.to(frames_norm.device)
    std = IMAGENET_STD.to(frames_norm.device)
    frames_norm = (frames_norm - mean) / std
    return frames_norm


def get_attention_weights(query, patch_features, dino_dim):
    """
    Compute attention weights for a query over patch features.

    Args:
        query: [B, D] query vector (already projected to dino_dim)
        patch_features: [B, N+1, D] patch features (CLS + 256 patches)
        dino_dim: dimension for scaling

    Returns:
        attn_weights: [B, N+1] attention weights
    """
    q_embed = query.unsqueeze(1)  # [B, 1, D]
    attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))  # [B, 1, N+1]
    attn_weights = torch.softmax(attn_scores / (dino_dim ** 0.5), dim=-1)
    return attn_weights.squeeze(1)  # [B, N+1]


@torch.no_grad()
def extract_attention_maps(model, frames, device):
    """
    Extract attention maps for both coarse (static) and fine (dynamic) queries.

    Returns:
        coarse_attn: [T, 16, 16] attention maps for static query
        fine_attn: [T, 16, 16] attention maps for dynamic queries
        queries_coarse: [T, query_dim] the static query (same for all frames)
        queries_fine: [T, query_dim] the dynamic queries per frame
    """
    B = 1
    T = frames.shape[0]

    # Normalize for model
    frames_norm = normalize_for_dino(frames).unsqueeze(0).to(device)

    # Get text embeddings
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    # Encode all frames
    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']  # [B*T, N+1, D]
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)  # [B, T, N+1, D]

    # Create caches
    all_caches = []
    for t in range(T):
        all_caches.append({'patch_features': patch_features[:, t]})

    # ========== Pass 1: Coarse with static query ==========
    q_static = model.q_static.expand(B, -1)  # [B, query_dim]
    q_static_proj = model.encoder.query_input_proj(q_static)  # [B, dino_dim]

    coarse_attn_list = []
    z_coarse_list = []

    for t in range(T):
        pf = all_caches[t]['patch_features']  # [B, N+1, D]
        attn = get_attention_weights(q_static_proj, pf, model.encoder.dino_dim)
        coarse_attn_list.append(attn[:, 1:])  # Skip CLS, keep 256 patches

        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)

    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    # LLM forward for Pass 1 to get dynamic queries
    coarse_token = model.coarse_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state

    h_for_queries = h_pass1[:, N_text + 1:]  # [B, T, llm_dim]
    queries = model.llm_to_query(h_for_queries)  # [B, T, query_dim]

    # ========== Pass 2: Fine with dynamic queries ==========
    q_init = model.q_init.expand(B, -1).unsqueeze(1)  # [B, 1, query_dim]
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)  # [B, T, query_dim]

    fine_attn_list = []

    for t in range(T):
        q_t = shifted_q[:, t]  # [B, query_dim]
        q_t_proj = model.encoder.query_input_proj(q_t)  # [B, dino_dim]

        pf = all_caches[t]['patch_features']  # [B, N+1, D]
        attn = get_attention_weights(q_t_proj, pf, model.encoder.dino_dim)
        fine_attn_list.append(attn[:, 1:])  # Skip CLS

    # Stack attention maps
    coarse_attn = torch.stack(coarse_attn_list, dim=1).squeeze(0)  # [T, N_patches]
    fine_attn = torch.stack(fine_attn_list, dim=1).squeeze(0)  # [T, N_patches]

    # DINO patch size is 14, so 256/14 ≈ 18 -> 18x18 = 324 patches
    # But actual count may vary, so compute dynamically
    n_patches = coarse_attn.shape[1]
    grid_size = int(n_patches ** 0.5)
    print(f"  Patches: {n_patches} = {grid_size}x{grid_size}")

    coarse_attn = coarse_attn.reshape(T, grid_size, grid_size)
    fine_attn = fine_attn.reshape(T, grid_size, grid_size)

    return coarse_attn.cpu(), fine_attn.cpu(), q_static.cpu(), shifted_q.squeeze(0).cpu()


def create_attention_overlay(frame, attn_map, alpha=0.4, normalize_mode='absolute'):
    """
    Overlay attention heatmap on frame.

    Args:
        frame: [3, H, W] uint8 tensor
        attn_map: [H_grid, W_grid] attention weights (should sum to 1)
        alpha: transparency of overlay
        normalize_mode:
            'absolute' - use fixed scale based on uniform baseline
            'relative' - stretch to min-max (old behavior)
            'deviation' - show deviation from uniform

    Returns:
        overlaid: [H, W, 3] uint8 numpy array
    """
    H, W = frame.shape[1], frame.shape[2]
    n_patches = attn_map.numel()
    uniform_val = 1.0 / n_patches  # Expected value if uniform

    # Upsample attention to frame size
    attn_up = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    if normalize_mode == 'absolute':
        # Scale relative to uniform: 0 = no attention, 1 = 3x uniform attention
        attn_norm = attn_up / (3 * uniform_val)
        attn_norm = np.clip(attn_norm, 0, 1)
    elif normalize_mode == 'deviation':
        # Show deviation from uniform: red = above average, blue = below
        deviation = (attn_up - uniform_val) / uniform_val  # Relative deviation
        attn_norm = (deviation + 1) / 2  # Map [-1, 1] to [0, 1]
        attn_norm = np.clip(attn_norm, 0, 1)
    else:  # relative
        attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    # Create heatmap - use perceptually uniform colormaps
    if normalize_mode == 'deviation':
        cmap = plt.cm.coolwarm  # Blue = below avg, Red = above avg
    else:
        cmap = plt.cm.viridis  # Dark = low attention, Bright = high attention
    heatmap = cmap(attn_norm)[:, :, :3]  # RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend with frame
    frame_np = frame.permute(1, 2, 0).numpy()
    overlaid = (1 - alpha) * frame_np + alpha * heatmap
    overlaid = overlaid.astype(np.uint8)

    return overlaid, attn_norm


def create_attention_grid(frames, coarse_attn, fine_attn, output_path):
    """
    Create a grid showing: Frame | Coarse Attention | Fine Attention | Difference
    for each timestep.
    """
    T = frames.shape[0]

    fig, axes = plt.subplots(T, 4, figsize=(16, 4 * T))

    for t in range(T):
        frame = frames[t]
        coarse = coarse_attn[t]
        fine = fine_attn[t]

        # Original frame
        axes[t, 0].imshow(frame.permute(1, 2, 0).numpy())
        axes[t, 0].set_title(f'Frame {t}')
        axes[t, 0].axis('off')

        # Coarse attention overlay (relative normalization to see structure)
        coarse_overlay, _ = create_attention_overlay(frame, coarse, normalize_mode='relative')
        axes[t, 1].imshow(coarse_overlay)
        axes[t, 1].set_title(f'Coarse (static) t={t}')
        axes[t, 1].axis('off')

        # Fine attention overlay (relative normalization)
        fine_overlay, _ = create_attention_overlay(frame, fine, normalize_mode='relative')
        axes[t, 2].imshow(fine_overlay)
        axes[t, 2].set_title(f'Fine (dynamic) t={t}')
        axes[t, 2].axis('off')

        # Difference: Fine - Coarse (raw heatmap, no frame overlay for clarity)
        diff = fine - coarse
        H, W = frame.shape[1], frame.shape[2]
        diff_up = F.interpolate(
            diff.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Normalize to [-1, 1] range for better visibility
        diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
        diff_norm = diff_up / diff_max

        # Use diverging colormap: blue = fine attends less, red = fine attends more
        im = axes[t, 3].imshow(diff_norm, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[t, 3].set_title(f'Fine - Coarse (normalized) t={t}')
        axes[t, 3].axis('off')

        # Add colorbar for first frame only
        if t == 0:
            cbar = plt.colorbar(im, ax=axes[t, 3], fraction=0.046)
            cbar.set_label('Attention Difference')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_attention_gif(frames, attn_maps, output_path, title="Attention", fps=4, normalize_mode='relative'):
    """Create a GIF with attention overlay."""
    T = frames.shape[0]

    gif_frames = []
    for t in range(T):
        overlay, _ = create_attention_overlay(frames[t], attn_maps[t], normalize_mode=normalize_mode)
        gif_frames.append(overlay)

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def create_comparison_gif(frames, coarse_attn, fine_attn, output_path, fps=4):
    """Create side-by-side GIF: Original | Coarse | Fine | Diff"""
    T = frames.shape[0]

    gif_frames = []
    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        coarse_overlay, _ = create_attention_overlay(frame, coarse_attn[t], normalize_mode='relative')
        fine_overlay, _ = create_attention_overlay(frame, fine_attn[t], normalize_mode='relative')

        # Create normalized difference heatmap
        diff = fine_attn[t] - coarse_attn[t]
        diff_up = F.interpolate(
            diff.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Normalize to [-1, 1] for better visibility
        diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
        diff_norm = diff_up / diff_max

        # Create colored heatmap (blue=negative, red=positive)
        cmap = plt.cm.RdBu_r
        diff_colored = cmap((diff_norm + 1) / 2)[:, :, :3]  # Map [-1,1] to [0,1]
        diff_overlay = (diff_colored * 255).astype(np.uint8)

        # Add labels
        label_h = 25

        combined = np.zeros((H + label_h, W * 4, 3), dtype=np.uint8)
        combined[label_h:, :W] = frame.permute(1, 2, 0).numpy()
        combined[label_h:, W:2*W] = coarse_overlay
        combined[label_h:, 2*W:3*W] = fine_overlay
        combined[label_h:, 3*W:] = diff_overlay

        # Add colored headers
        combined[:label_h, :W] = [80, 80, 80]        # Gray for original
        combined[:label_h, W:2*W] = [50, 50, 150]    # Blue for coarse
        combined[:label_h, 2*W:3*W] = [150, 50, 50]  # Red for fine
        combined[:label_h, 3*W:] = [50, 150, 50]     # Green for diff

        gif_frames.append(combined)

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def plot_attention_stats(coarse_attn, fine_attn, output_path):
    """Plot attention statistics over time."""
    T = coarse_attn.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Entropy over time (higher = more distributed attention)
    coarse_entropy = []
    fine_entropy = []
    for t in range(T):
        c = coarse_attn[t].flatten()
        f = fine_attn[t].flatten()
        c = c / c.sum()
        f = f / f.sum()
        coarse_entropy.append(-(c * torch.log(c + 1e-10)).sum().item())
        fine_entropy.append(-(f * torch.log(f + 1e-10)).sum().item())

    axes[0, 0].plot(coarse_entropy, 'b-o', label='Coarse (static)')
    axes[0, 0].plot(fine_entropy, 'r-o', label='Fine (dynamic)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Attention Entropy (higher = more spread)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Max attention value over time
    coarse_max = [coarse_attn[t].max().item() for t in range(T)]
    fine_max = [fine_attn[t].max().item() for t in range(T)]

    axes[0, 1].plot(coarse_max, 'b-o', label='Coarse (static)')
    axes[0, 1].plot(fine_max, 'r-o', label='Fine (dynamic)')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Max Attention')
    axes[0, 1].set_title('Peak Attention (higher = more focused)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Attention difference heatmap
    diff = (fine_attn - coarse_attn).mean(dim=0)  # Average over time
    im = axes[1, 0].imshow(diff.numpy(), cmap='RdBu', vmin=-0.01, vmax=0.01)
    axes[1, 0].set_title('Mean(Fine - Coarse) Attention')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])

    # Attention movement (center of mass)
    coarse_cx, coarse_cy = [], []
    fine_cx, fine_cy = [], []
    grid_size = coarse_attn.shape[1]  # Dynamic grid size

    for t in range(T):
        # Create coordinate grids
        y_coords = torch.arange(grid_size).float()
        x_coords = torch.arange(grid_size).float()

        c = coarse_attn[t]
        f = fine_attn[t]

        coarse_cy.append((c.sum(dim=1) * y_coords).sum().item() / c.sum().item())
        coarse_cx.append((c.sum(dim=0) * x_coords).sum().item() / c.sum().item())
        fine_cy.append((f.sum(dim=1) * y_coords).sum().item() / f.sum().item())
        fine_cx.append((f.sum(dim=0) * x_coords).sum().item() / f.sum().item())

    axes[1, 1].plot(coarse_cx, coarse_cy, 'b-o', label='Coarse', markersize=8)
    axes[1, 1].plot(fine_cx, fine_cy, 'r-o', label='Fine', markersize=8)
    axes[1, 1].set_xlabel('X center of mass')
    axes[1, 1].set_ylabel('Y center of mass')
    axes[1, 1].set_title('Attention Center Movement')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, grid_size - 1)
    axes[1, 1].set_ylim(0, grid_size - 1)
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True)

    # Mark start and end
    axes[1, 1].scatter([coarse_cx[0]], [coarse_cy[0]], c='blue', s=100, marker='s', zorder=5)
    axes[1, 1].scatter([fine_cx[0]], [fine_cy[0]], c='red', s=100, marker='s', zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 10  # More examples for diversity

    output_dir = Path('outputs/attention')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Download test videos
    videos = get_test_videos(num_examples, num_frames)
    print(f"\nGot {len(videos)} test videos")

    # Process each video
    print("\nExtracting attention maps...")
    for i, video in enumerate(videos):
        print(f"\nVideo {i}: {video['caption'][:50]}...")
        frames = video['frames']

        # Extract attention maps
        coarse_attn, fine_attn, q_static, q_fine = extract_attention_maps(
            model, frames, device
        )

        # Create visualizations
        # 1. Grid of all frames with attention
        create_attention_grid(
            frames, coarse_attn, fine_attn,
            output_dir / f'grid_{i:02d}.png'
        )

        # 2. Comparison GIF
        create_comparison_gif(
            frames, coarse_attn, fine_attn,
            output_dir / f'comparison_{i:02d}.gif'
        )

        # 3. Individual attention GIFs
        create_attention_gif(
            frames, coarse_attn,
            output_dir / f'coarse_{i:02d}.gif',
            title='Coarse'
        )
        create_attention_gif(
            frames, fine_attn,
            output_dir / f'fine_{i:02d}.gif',
            title='Fine'
        )

        # 4. Attention statistics
        plot_attention_stats(
            coarse_attn, fine_attn,
            output_dir / f'stats_{i:02d}.png'
        )

        # Print summary
        coarse_entropy = -(coarse_attn.flatten() * torch.log(coarse_attn.flatten() + 1e-10)).mean()
        fine_entropy = -(fine_attn.flatten() * torch.log(fine_attn.flatten() + 1e-10)).mean()

        print(f"  Coarse attention entropy: {coarse_entropy:.3f}")
        print(f"  Fine attention entropy:   {fine_entropy:.3f}")
        print(f"  Coarse max attention:     {coarse_attn.max():.3f}")
        print(f"  Fine max attention:       {fine_attn.max():.3f}")

    print(f"\nSaved visualizations to {output_dir}")
    print("\nFiles per video:")
    print("  - grid_XX.png: All frames with coarse/fine attention overlay")
    print("  - comparison_XX.gif: Side-by-side animation (Original | Coarse | Fine)")
    print("  - coarse_XX.gif: Coarse attention animation")
    print("  - fine_XX.gif: Fine attention animation")
    print("  - stats_XX.png: Attention statistics plots")


if __name__ == "__main__":
    main()
