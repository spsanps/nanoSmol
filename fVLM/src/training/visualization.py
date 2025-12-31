"""
Attention visualization utilities for Foveated VLM.

Visualize where the model is looking (attention patterns) to verify
that dynamic queries produce focused, motion-tracking attention.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.sampling import denormalize_frames


def get_attention_maps(model, frames, return_queries=False):
    """
    Extract attention weights from both static and dynamic queries.

    Args:
        model: FoveatedVideoModel
        frames: [B, T, 3, 256, 256] video frames (ImageNet normalized)
        return_queries: Whether to return query predictions

    Returns:
        attn_static: [B, T, N] attention weights from static query
        attn_dynamic: [B, T, N] attention weights from dynamic queries
        queries: [B, T, query_dim] predicted queries (if return_queries=True)
    """
    model.eval()
    B, T = frames.shape[:2]

    with torch.no_grad():
        text_embeds = model.get_empty_text_embeds(B).to(frames.device)

        # Encode all frames
        all_caches = []
        for t in range(T):
            _, cache = model.encoder.encode_patches(frames[:, t])
            all_caches.append(cache)

        # === Pass 1: Static query attention ===
        q_static = model.q_static.expand(B, -1)

        attn_static_list = []
        for t in range(T):
            # Get patch features
            patch_features = all_caches[t]['patch_features']  # [B, N, D]

            # Compute attention (simplified - actual is inside encoder)
            q_embed = model.encoder.query_input_proj(q_static).unsqueeze(1)  # [B, 1, D]
            attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))  # [B, 1, N]
            attn_weights = torch.softmax(attn_scores / (model.dino_dim ** 0.5), dim=-1)

            attn_static_list.append(attn_weights.squeeze(1))  # [B, N]

        attn_static = torch.stack(attn_static_list, dim=1)  # [B, T, N]

        # === Get query predictions from Pass 1 ===
        z_coarse_list = []
        for t in range(T):
            z_t = model.encoder.query_attend(q_static, all_caches[t])
            z_coarse_list.append(z_t)
        z_coarse = torch.stack(z_coarse_list, dim=1)
        z_coarse = model.dino_to_llm(z_coarse)

        coarse_token = model.coarse_token.expand(B, -1, -1)
        seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
        outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
        h_pass1 = outputs_pass1.last_hidden_state

        N_text = text_embeds.shape[1]
        h_for_queries = h_pass1[:, N_text + 1:]
        queries = model.llm_to_query(h_for_queries)  # [B, T, query_dim]

        # === Pass 2: Dynamic query attention ===
        q_init = model.q_init.expand(B, -1).unsqueeze(1)
        shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

        attn_dynamic_list = []
        for t in range(T):
            patch_features = all_caches[t]['patch_features']

            q_embed = model.encoder.query_input_proj(shifted_q[:, t]).unsqueeze(1)
            attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
            attn_weights = torch.softmax(attn_scores / (model.dino_dim ** 0.5), dim=-1)

            attn_dynamic_list.append(attn_weights.squeeze(1))

        attn_dynamic = torch.stack(attn_dynamic_list, dim=1)  # [B, T, N]

    if return_queries:
        return attn_static, attn_dynamic, queries
    return attn_static, attn_dynamic


def visualize_attention_video(
    frames,
    attn_static,
    attn_dynamic,
    save_path,
    video_id="",
    patch_size=16,
):
    """
    Visualize attention patterns over a video sequence.

    Args:
        frames: [T, 3, H, W] video frames (normalized)
        attn_static: [T, N] static query attention weights
        attn_dynamic: [T, N] dynamic query attention weights
        save_path: Where to save the visualization
        video_id: Video identifier for title
        patch_size: Patch size (16 for DINOv2-S)
    """
    T = frames.shape[0]
    H, W = frames.shape[2], frames.shape[3]
    grid_h, grid_w = H // patch_size, W // patch_size

    # Denormalize frames for visualization
    frames_vis = denormalize_frames(frames, normalize_mode='dino')
    frames_vis = frames_vis.cpu().numpy().transpose(0, 2, 3, 1)  # [T, H, W, 3]

    # Reshape attention to 2D grids
    attn_static_2d = attn_static.cpu().numpy().reshape(T, grid_h, grid_w)
    attn_dynamic_2d = attn_dynamic.cpu().numpy().reshape(T, grid_h, grid_w)

    # Create figure
    fig, axes = plt.subplots(3, T, figsize=(3 * T, 9))
    if T == 1:
        axes = axes[:, None]

    for t in range(T):
        # Row 1: Original frames
        axes[0, t].imshow(frames_vis[t])
        axes[0, t].set_title(f'Frame {t+1}')
        axes[0, t].axis('off')

        # Row 2: Static attention overlay
        axes[1, t].imshow(frames_vis[t])
        attn_overlay = axes[1, t].imshow(
            attn_static_2d[t],
            alpha=0.6,
            cmap='hot',
            interpolation='bilinear',
            extent=[0, W, H, 0]
        )
        axes[1, t].set_title(f'Static Attention')
        axes[1, t].axis('off')

        # Row 3: Dynamic attention overlay
        axes[2, t].imshow(frames_vis[t])
        axes[2, t].imshow(
            attn_dynamic_2d[t],
            alpha=0.6,
            cmap='hot',
            interpolation='bilinear',
            extent=[0, W, H, 0]
        )
        axes[2, t].set_title(f'Dynamic Attention')
        axes[2, t].axis('off')

    # Add colorbars
    fig.colorbar(attn_overlay, ax=axes[1, :].ravel().tolist(), label='Attention Weight', shrink=0.8)
    fig.colorbar(attn_overlay, ax=axes[2, :].ravel().tolist(), label='Attention Weight', shrink=0.8)

    plt.suptitle(f'Attention Patterns: {video_id}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_attention_entropy(attn_weights):
    """
    Compute entropy of attention distribution.
    Lower entropy = more focused attention.

    Args:
        attn_weights: [B, T, N] attention weights

    Returns:
        entropy: [B, T] entropy values
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
    return entropy


def analyze_attention_focus(attn_static, attn_dynamic):
    """
    Analyze how focused the attention is.

    Returns:
        dict with entropy stats and comparison
    """
    entropy_static = compute_attention_entropy(attn_static)
    entropy_dynamic = compute_attention_entropy(attn_dynamic)

    return {
        'entropy_static_mean': entropy_static.mean().item(),
        'entropy_static_std': entropy_static.std().item(),
        'entropy_dynamic_mean': entropy_dynamic.mean().item(),
        'entropy_dynamic_std': entropy_dynamic.std().item(),
        'entropy_reduction': (entropy_static.mean() - entropy_dynamic.mean()).item(),
        'is_more_focused': entropy_dynamic.mean() < entropy_static.mean(),
    }


if __name__ == "__main__":
    # Test visualization
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.model.foveated_vlm import FoveatedVideoModel
    from src.data.dataset import create_dataloader

    print("=" * 70)
    print("Testing Attention Visualization")
    print("=" * 70)

    # Load model
    print("\n📦 Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
    ).cuda()
    model.eval()

    # Load a test batch
    print("\n📦 Loading test data...")
    try:
        dataloader = create_dataloader(
            video_dir="data/videos",
            latent_dir="data/latents",
            batch_size=1,
            num_workers=0,
            shuffle=False,
            num_frames=4,  # Small for testing
        )
        batch = next(iter(dataloader))
    except ValueError:
        print("⚠️  Waiting for VAE preprocessing to complete...")
        sys.exit(0)

    frames = batch['frames'].cuda()  # [1, 4, 3, 256, 256]
    video_id = batch['video_id'][0]

    # Get attention maps
    print(f"\n🔄 Computing attention maps for {video_id}...")
    attn_static, attn_dynamic = get_attention_maps(model, frames)

    print(f"   Static attention shape: {attn_static.shape}")
    print(f"   Dynamic attention shape: {attn_dynamic.shape}")

    # Analyze focus
    print("\n📊 Analyzing attention focus...")
    stats = analyze_attention_focus(attn_static, attn_dynamic)
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Visualize
    print("\n🎨 Creating visualization...")
    save_path = Path("outputs/visualizations") / f"attention_{video_id}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_attention_video(
        frames[0],  # First sample
        attn_static[0],
        attn_dynamic[0],
        save_path=save_path,
        video_id=video_id,
    )

    print(f"   ✓ Saved to: {save_path}")
    print("\n" + "=" * 70)
    print("✓ Visualization test complete!")
    print("=" * 70)
