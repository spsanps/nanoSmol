#!/usr/bin/env python3
"""
Comprehensive evaluation of the 24h trained model.

Generates:
1. Per-frame loss analysis (especially last frames)
2. Attention maps comparison (coarse vs fine)
3. Generated captions
4. Comparison visualizations
5. Detailed analysis report
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
from diffusers import AutoencoderKL
import imageio
import requests
import tempfile
import subprocess
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel


def parse_duration(dur_str: str) -> int:
    """Parse ISO 8601 duration string."""
    try:
        match = re.match(r'PT(\d+)H(\d+)M(\d+)S', dur_str)
        if match:
            return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
        match = re.match(r'PT(\d+)M(\d+)S', dur_str)
        if match:
            return int(match[1]) * 60 + int(match[2])
        match = re.match(r'PT(\d+)S', dur_str)
        if match:
            return int(match[1])
    except:
        pass
    return 0


def download_video(url: str, timeout: int = 30) -> bytes:
    """Download video from URL with retries."""
    for retry in range(3):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                return response.content
        except:
            if retry < 2:
                import time
                time.sleep(0.5 * (retry + 1))
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    """Extract frames from video bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as f:
        f.write(video_bytes)
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-i', f.name,
                '-vf', f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
                '-frames:v', str(num_frames * 4),
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


@torch.no_grad()
def compute_per_frame_losses(model, vae, frames_norm, device):
    """Compute per-frame reconstruction losses for both coarse and fine."""
    B, T = 1, frames_norm.shape[1]

    # Compute VAE latents
    latents_list = []
    for t in range(T):
        frame_t = F.interpolate(frames_norm[:, t], size=(256, 256), mode='bilinear')
        latent = vae.encode(frame_t).latent_dist.mean * 0.18215
        latents_list.append(latent)
    vae_latents = torch.stack(latents_list, dim=1)  # [B, T, 4, 32, 32]

    # Get text embeddings
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    # Encode frames
    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Handle kv_cache for deep mode
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        all_caches = []
        for t in range(T):
            frame_kv_cache = []
            for layer_idx in range(num_layers):
                layer_cache = cache_flat['kv_cache'][layer_idx]
                K_all = layer_cache['K'].reshape(B, T, N, D)
                V_all = layer_cache['V'].reshape(B, T, N, D)
                frame_kv_cache.append({
                    'K': K_all[:, t],
                    'V': V_all[:, t],
                    'layer': layer_cache['layer'],
                })
            all_caches.append({
                'patch_features': patch_features[:, t],
                'kv_cache': frame_kv_cache,
            })
    else:
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    # === Pass 1: Coarse with q_static ===
    q_static = model.q_static.expand(B, -1)

    z_coarse_list = []
    for t in range(T):
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

    # Coarse predictions
    h_coarse_for_pred = h_pass1[:, N_text:N_text + T]
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
    target_latents = vae_latents

    pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)

    # Per-frame coarse loss
    coarse_losses = []
    for t in range(T):
        loss_t = F.mse_loss(pred_coarse[:, t], target_latents[:, t])
        coarse_losses.append(loss_t.item())

    # === Pass 2: Fine with dynamic queries ===
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    z_focused_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(shifted_q[:, t], all_caches[t])
        z_focused_list.append(z_t)
    z_focused = torch.stack(z_focused_list, dim=1)
    z_focused = model.dino_to_llm(z_focused)
    z_focused = z_focused / (z_focused.std() + 1e-6) * model.visual_scale

    fine_token = model.fine_token.expand(B, -1, -1)
    seq_pass2 = torch.cat([text_embeds, fine_token, z_focused], dim=1)
    outputs_pass2 = model.llm.model(inputs_embeds=seq_pass2)
    h_pass2 = outputs_pass2.last_hidden_state

    h_fine_for_pred = h_pass2[:, N_text:N_text + T]
    pred_fine = model.pred_head(h_fine_for_pred, prev_latents)

    # Per-frame fine loss
    fine_losses = []
    for t in range(T):
        loss_t = F.mse_loss(pred_fine[:, t], target_latents[:, t])
        fine_losses.append(loss_t.item())

    return coarse_losses, fine_losses, pred_coarse, pred_fine, target_latents


@torch.no_grad()
def extract_attention_maps(model, frames_norm, device):
    """Extract attention maps for both coarse and fine queries."""
    B, T = 1, frames_norm.shape[1]

    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Handle kv_cache for deep mode
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
        all_caches = []
        for t in range(T):
            frame_kv_cache = []
            for layer_idx in range(num_layers):
                layer_cache = cache_flat['kv_cache'][layer_idx]
                K_all = layer_cache['K'].reshape(B, T, N, D)
                V_all = layer_cache['V'].reshape(B, T, N, D)
                frame_kv_cache.append({
                    'K': K_all[:, t],
                    'V': V_all[:, t],
                    'layer': layer_cache['layer'],
                })
            all_caches.append({
                'patch_features': patch_features[:, t],
                'kv_cache': frame_kv_cache,
            })
    else:
        # Shallow mode
        all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

    # Pass 1: Coarse
    q_static = model.q_static.expand(B, -1)
    q_static_proj = model.encoder.query_input_proj(q_static)

    coarse_attn_list = []
    z_coarse_list = []

    for t in range(T):
        pf = all_caches[t]['patch_features']
        # Compute attention weights for visualization
        q_embed = q_static_proj.unsqueeze(1)
        attn_scores = torch.bmm(q_embed, pf.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (model.encoder.dino_dim ** 0.5), dim=-1)
        coarse_attn_list.append(attn_weights.squeeze(1)[:, 1:])  # Skip CLS

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

    # Pass 2: Fine
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    fine_attn_list = []
    for t in range(T):
        q_t = shifted_q[:, t]
        q_t_proj = model.encoder.query_input_proj(q_t)
        pf = all_caches[t]['patch_features']

        q_embed = q_t_proj.unsqueeze(1)
        attn_scores = torch.bmm(q_embed, pf.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (model.encoder.dino_dim ** 0.5), dim=-1)
        fine_attn_list.append(attn_weights.squeeze(1)[:, 1:])

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

    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    cmap = plt.cm.jet
    heatmap = cmap(attn_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    frame_np = frame.permute(1, 2, 0).numpy()
    if frame_np.max() > 1:
        frame_np = frame_np / 255.0
    frame_np = (frame_np * 255).astype(np.uint8)

    overlaid = (1 - alpha) * frame_np + alpha * heatmap
    return overlaid.astype(np.uint8)


def decode_latent(vae, latent, device):
    """Decode VAE latent to image."""
    latent = latent.unsqueeze(0).to(device) / 0.18215
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    decoded = (decoded.clamp(-1, 1) + 1) / 2
    return decoded.squeeze(0).cpu()


def create_comparison_figure(frames, coarse_attn, fine_attn, coarse_losses, fine_losses,
                            pred_coarse, pred_fine, targets, vae, device,
                            caption_fine, caption_coarse, output_path, video_id):
    """Create comprehensive comparison figure."""
    T = min(8, frames.shape[0])
    indices = np.linspace(0, frames.shape[0] - 1, T).astype(int)

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(6, T, height_ratios=[1, 1, 1, 1, 0.8, 0.8], hspace=0.3, wspace=0.05)

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
        coarse_overlay = create_attention_overlay(frame, coarse_attn[t])
        ax2.imshow(coarse_overlay)
        if i == 0:
            ax2.set_ylabel('Coarse Attn', fontsize=10, color='blue')
        ax2.axis('off')

        # Row 3: Fine attention
        ax3 = fig.add_subplot(gs[2, i])
        fine_overlay = create_attention_overlay(frame, fine_attn[t])
        ax3.imshow(fine_overlay)
        if i == 0:
            ax3.set_ylabel('Fine Attn', fontsize=10, color='red')
        ax3.axis('off')

        # Row 4: Reconstructions (decode a few frames)
        if i < 4:  # Only decode first 4 to save time
            ax4 = fig.add_subplot(gs[3, i])
            try:
                pred_img = decode_latent(vae, pred_fine[0, t], device)
                ax4.imshow(pred_img.permute(1, 2, 0).numpy())
            except:
                ax4.imshow(np.zeros((256, 256, 3)))
            if i == 0:
                ax4.set_ylabel('Pred (Fine)', fontsize=10)
            ax4.axis('off')

    # Row 5: Per-frame loss plot
    ax_loss = fig.add_subplot(gs[4, :])
    x = list(range(len(coarse_losses)))
    ax_loss.plot(x, coarse_losses, 'b-o', label='Coarse Loss', linewidth=2, markersize=4)
    ax_loss.plot(x, fine_losses, 'r-o', label='Fine Loss', linewidth=2, markersize=4)
    ax_loss.set_xlabel('Frame')
    ax_loss.set_ylabel('MSE Loss')
    ax_loss.set_title('Per-Frame Reconstruction Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Highlight last frames
    last_n = 4
    ax_loss.axvspan(len(coarse_losses) - last_n, len(coarse_losses), alpha=0.2, color='yellow', label='Last 4 frames')

    # Row 6: Captions and stats
    ax_cap = fig.add_subplot(gs[5, :])
    ax_cap.axis('off')

    avg_coarse = np.mean(coarse_losses)
    avg_fine = np.mean(fine_losses)
    last_coarse = np.mean(coarse_losses[-4:])
    last_fine = np.mean(fine_losses[-4:])

    stats_text = f"""Video: {video_id}

LOSSES:
  Overall - Coarse: {avg_coarse:.4f}, Fine: {avg_fine:.4f}, Δ: {avg_coarse - avg_fine:.4f} ({'Fine wins' if avg_fine < avg_coarse else 'Coarse wins'})
  Last 4  - Coarse: {last_coarse:.4f}, Fine: {last_fine:.4f}, Δ: {last_coarse - last_fine:.4f} ({'Fine wins' if last_fine < last_coarse else 'Coarse wins'})

CAPTIONS:
  Fine:   {caption_fine[:150]}{'...' if len(caption_fine) > 150 else ''}
  Coarse: {caption_coarse[:150]}{'...' if len(caption_coarse) > 150 else ''}"""

    ax_cap.text(0.02, 0.95, stats_text, transform=ax_cap.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'24h Model Evaluation - Video {video_id}', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_comparison_gif(frames, coarse_attn, fine_attn, output_path, fps=8):
    """Create side-by-side comparison GIF with difference map."""
    T = frames.shape[0]
    gif_frames = []

    for t in range(T):
        frame = frames[t]
        H, W = frame.shape[1], frame.shape[2]

        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() > 1:
            frame_np = (frame_np / 255.0 * 255).astype(np.uint8)
        else:
            frame_np = (frame_np * 255).astype(np.uint8)

        coarse_overlay = create_attention_overlay(frame, coarse_attn[t])
        fine_overlay = create_attention_overlay(frame, fine_attn[t])

        # Create difference heatmap (Fine - Coarse)
        diff = fine_attn[t] - coarse_attn[t]
        diff_up = F.interpolate(
            diff.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Normalize to [-1, 1] range for colormap
        diff_max = max(abs(diff_up.min()), abs(diff_up.max())) + 1e-8
        diff_norm = (diff_up / diff_max + 1) / 2  # Map to [0, 1]

        # RdBu colormap: Blue = coarse stronger, Red = fine stronger
        cmap = plt.cm.RdBu_r
        diff_colored = cmap(diff_norm)[:, :, :3]
        diff_overlay = (diff_colored * 255).astype(np.uint8)

        # Combine: Original | Coarse | Fine | Difference
        combined = np.concatenate([frame_np, coarse_overlay, fine_overlay, diff_overlay], axis=1)

        # Add header with colored backgrounds
        header_h = 35
        header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
        header[:, :W] = [60, 60, 60]        # Gray for original
        header[:, W:2*W] = [40, 40, 150]    # Blue for coarse
        header[:, 2*W:3*W] = [150, 40, 40]  # Red for fine
        header[:, 3*W:] = [40, 120, 40]     # Green for difference

        combined = np.concatenate([header, combined], axis=0)

        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        labels = [f"Frame {t}", "Coarse (static)", "Fine (dynamic)", "Diff (F-C)"]
        for i, label in enumerate(labels):
            x = i * W + W // 2 - len(label) * 4
            draw.text((x, 10), label, fill=(255, 255, 255), font=font)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def plot_attention_stats(coarse_attn, fine_attn, coarse_losses, fine_losses, output_path, video_id):
    """Plot detailed attention statistics."""
    T = coarse_attn.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Attention & Loss Statistics - Video {video_id}', fontsize=14, fontweight='bold')

    # 1. Attention Entropy over time
    coarse_entropy = []
    fine_entropy = []
    for t in range(T):
        c = coarse_attn[t].flatten()
        f = fine_attn[t].flatten()
        c = c / c.sum()
        f = f / f.sum()
        coarse_entropy.append(-(c * torch.log(c + 1e-10)).sum().item())
        fine_entropy.append(-(f * torch.log(f + 1e-10)).sum().item())

    axes[0, 0].plot(coarse_entropy, 'b-o', label='Coarse', linewidth=2, markersize=4)
    axes[0, 0].plot(fine_entropy, 'r-o', label='Fine', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Attention Entropy (lower = more focused)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Max attention over time
    coarse_max = [coarse_attn[t].max().item() for t in range(T)]
    fine_max = [fine_attn[t].max().item() for t in range(T)]

    axes[0, 1].plot(coarse_max, 'b-o', label='Coarse', linewidth=2, markersize=4)
    axes[0, 1].plot(fine_max, 'r-o', label='Fine', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Max Attention')
    axes[0, 1].set_title('Peak Attention Value (higher = more focused)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Fine/Coarse attention ratio
    ratio = [f_max / (c_max + 1e-8) for c_max, f_max in zip(coarse_max, fine_max)]

    axes[0, 2].plot(ratio, 'g-o', linewidth=2, markersize=4)
    axes[0, 2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal')
    axes[0, 2].set_xlabel('Frame')
    axes[0, 2].set_ylabel('Fine/Coarse Ratio')
    axes[0, 2].set_title('Attention Focus Ratio (>1 = fine more focused)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Per-frame losses
    x = list(range(len(coarse_losses)))
    axes[1, 0].plot(x, coarse_losses, 'b-o', label='Coarse', linewidth=2, markersize=4)
    axes[1, 0].plot(x, fine_losses, 'r-o', label='Fine', linewidth=2, markersize=4)
    axes[1, 0].axvspan(len(x) - 4, len(x), alpha=0.2, color='yellow')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].set_title('Per-Frame Reconstruction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Loss difference over time
    loss_diff = [c - f for c, f in zip(coarse_losses, fine_losses)]
    colors = ['green' if d > 0 else 'red' for d in loss_diff]
    axes[1, 1].bar(x, loss_diff, color=colors, alpha=0.7)
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Coarse - Fine')
    axes[1, 1].set_title('Loss Difference (green = fine better)')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Mean attention difference heatmap
    diff = (fine_attn - coarse_attn).mean(dim=0)
    im = axes[1, 2].imshow(diff.numpy(), cmap='RdBu_r', vmin=-0.01, vmax=0.01)
    axes[1, 2].set_title('Mean(Fine - Coarse) Attention')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 20

    output_dir = Path('outputs/eval_24h')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("24h Model Evaluation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    checkpoint_path = Path('outputs/large_scale_24h_deep_v3/checkpoints/final_step_26153.pt')

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = {
        'dino_model': 'facebook/dinov2-small',
        'llm_model': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'dino_dim': 384,
        'llm_dim': 576,
        'query_dim': 128,
        'lambda_coarse': 0.5,
        'deep_query': True,
    }

    model = FoveatedVideoModel(
        dino_model=model_cfg['dino_model'],
        llm_model=model_cfg['llm_model'],
        dino_dim=model_cfg['dino_dim'],
        llm_dim=model_cfg['llm_dim'],
        query_dim=model_cfg['query_dim'],
        lambda_coarse=model_cfg['lambda_coarse'],
        deep_query=model_cfg['deep_query'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load samples from streaming dataset
    print("\nLoading samples from WebVid streaming...")
    from datasets import load_dataset

    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=100)

    results = []
    all_coarse_losses = []
    all_fine_losses = []

    sample_idx = 0
    for hf_sample in ds:
        if sample_idx >= num_examples:
            break

        try:
            # Filter by duration
            duration = parse_duration(hf_sample.get('duration', ''))
            if duration < 5 or duration > 30:
                continue

            # Download video
            video_bytes = download_video(hf_sample['contentUrl'])
            if video_bytes is None:
                continue

            # Extract frames
            frames = extract_frames(video_bytes, num_frames, 256)
            if frames is None:
                continue

            video_id = hf_sample.get('videoid', f'sample_{sample_idx}')

            print(f"\n{'='*60}")
            print(f"[{sample_idx+1}/{num_examples}] Video {video_id}")
            print(f"{'='*60}")

            # frames already has the correct number from extract_frames
            # Normalize for model
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            # Compute per-frame losses
            print("  Computing per-frame losses...")
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                coarse_losses, fine_losses, pred_coarse, pred_fine, targets = compute_per_frame_losses(
                    model, vae, frames_norm, device
                )

            all_coarse_losses.append(coarse_losses)
            all_fine_losses.append(fine_losses)

            avg_coarse = np.mean(coarse_losses)
            avg_fine = np.mean(fine_losses)
            last_coarse = np.mean(coarse_losses[-4:])
            last_fine = np.mean(fine_losses[-4:])

            print(f"  Overall - Coarse: {avg_coarse:.4f}, Fine: {avg_fine:.4f}")
            print(f"  Last 4  - Coarse: {last_coarse:.4f}, Fine: {last_fine:.4f}")

            # Extract attention maps
            print("  Extracting attention maps...")
            coarse_attn, fine_attn = extract_attention_maps(model, frames_norm, device)

            # Generate captions
            print("  Generating captions...")
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

            create_comparison_figure(
                frames, coarse_attn, fine_attn, coarse_losses, fine_losses,
                pred_coarse, pred_fine, targets, vae, device,
                caption_fine, caption_coarse,
                output_dir / f'eval_{sample_idx:02d}_{video_id}.png',
                video_id
            )

            create_comparison_gif(
                frames, coarse_attn, fine_attn,
                output_dir / f'gif_{sample_idx:02d}_{video_id}.gif'
            )

            # Attention statistics plot
            plot_attention_stats(
                coarse_attn, fine_attn, coarse_losses, fine_losses,
                output_dir / f'stats_{sample_idx:02d}_{video_id}.png',
                video_id
            )

            # Compute entropy stats
            coarse_entropy = []
            fine_entropy = []
            for t in range(coarse_attn.shape[0]):
                c = coarse_attn[t].flatten()
                f = fine_attn[t].flatten()
                c = c / c.sum()
                f = f / f.sum()
                coarse_entropy.append(-(c * torch.log(c + 1e-10)).sum().item())
                fine_entropy.append(-(f * torch.log(f + 1e-10)).sum().item())

            results.append({
                'video_id': video_id,
                'avg_coarse': avg_coarse,
                'avg_fine': avg_fine,
                'last4_coarse': last_coarse,
                'last4_fine': last_fine,
                'caption_fine': caption_fine,
                'caption_coarse': caption_coarse,
                'coarse_losses': coarse_losses,
                'fine_losses': fine_losses,
                'avg_coarse_entropy': np.mean(coarse_entropy),
                'avg_fine_entropy': np.mean(fine_entropy),
                'coarse_max_attn': coarse_attn.max().item(),
                'fine_max_attn': fine_attn.max().item(),
            })

            print(f"  Entropy - Coarse: {np.mean(coarse_entropy):.3f}, Fine: {np.mean(fine_entropy):.3f}")
            print(f"  Max Attn - Coarse: {coarse_attn.max():.4f}, Fine: {fine_attn.max():.4f}")

            sample_idx += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary report
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)

    md_path = output_dir / 'EVALUATION_REPORT.md'
    with open(md_path, 'w') as f:
        f.write("# 24h Model Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Checkpoint:** final_step_26153.pt\n")
        f.write(f"**Samples Evaluated:** {len(results)}\n\n")

        # Overall statistics
        all_avg_coarse = [r['avg_coarse'] for r in results]
        all_avg_fine = [r['avg_fine'] for r in results]
        all_last_coarse = [r['last4_coarse'] for r in results]
        all_last_fine = [r['last4_fine'] for r in results]

        f.write("## Overall Statistics\n\n")
        f.write("| Metric | Coarse | Fine | Δ (Coarse - Fine) | Winner |\n")
        f.write("|--------|--------|------|-------------------|--------|\n")
        f.write(f"| Avg Loss (all frames) | {np.mean(all_avg_coarse):.4f} | {np.mean(all_avg_fine):.4f} | {np.mean(all_avg_coarse) - np.mean(all_avg_fine):.4f} | {'Fine' if np.mean(all_avg_fine) < np.mean(all_avg_coarse) else 'Coarse'} |\n")
        f.write(f"| Avg Loss (last 4) | {np.mean(all_last_coarse):.4f} | {np.mean(all_last_fine):.4f} | {np.mean(all_last_coarse) - np.mean(all_last_fine):.4f} | {'Fine' if np.mean(all_last_fine) < np.mean(all_last_coarse) else 'Coarse'} |\n\n")

        # Per-frame analysis
        f.write("## Per-Frame Loss Analysis\n\n")
        f.write("Averaged across all samples:\n\n")
        f.write("| Frame | Coarse | Fine | Δ | Winner |\n")
        f.write("|-------|--------|------|---|--------|\n")

        avg_per_frame_coarse = np.mean(all_coarse_losses, axis=0)
        avg_per_frame_fine = np.mean(all_fine_losses, axis=0)

        for t in range(len(avg_per_frame_coarse)):
            delta = avg_per_frame_coarse[t] - avg_per_frame_fine[t]
            winner = 'Fine' if avg_per_frame_fine[t] < avg_per_frame_coarse[t] else 'Coarse'
            marker = '**' if t >= len(avg_per_frame_coarse) - 4 else ''
            f.write(f"| {marker}Frame {t}{marker} | {avg_per_frame_coarse[t]:.4f} | {avg_per_frame_fine[t]:.4f} | {delta:.4f} | {winner} |\n")

        f.write("\n*Bold = last 4 frames*\n\n")

        # Count wins
        fine_wins_overall = sum(1 for r in results if r['avg_fine'] < r['avg_coarse'])
        fine_wins_last4 = sum(1 for r in results if r['last4_fine'] < r['last4_coarse'])

        f.write("## Win Statistics\n\n")
        f.write(f"- Fine wins overall: {fine_wins_overall}/{len(results)} ({100*fine_wins_overall/len(results):.1f}%)\n")
        f.write(f"- Fine wins on last 4 frames: {fine_wins_last4}/{len(results)} ({100*fine_wins_last4/len(results):.1f}%)\n\n")

        # Attention Statistics
        f.write("## Attention Statistics\n\n")
        f.write("| # | Video | Coarse Entropy | Fine Entropy | Coarse Max | Fine Max | Focus Ratio |\n")
        f.write("|---|-------|----------------|--------------|------------|----------|-------------|\n")

        for i, r in enumerate(results):
            vid = str(r['video_id'])[:15]
            ratio = r['fine_max_attn'] / (r['coarse_max_attn'] + 1e-8)
            f.write(f"| {i+1} | {vid} | {r['avg_coarse_entropy']:.3f} | {r['avg_fine_entropy']:.3f} | ")
            f.write(f"{r['coarse_max_attn']:.4f} | {r['fine_max_attn']:.4f} | {ratio:.3f} |\n")

        avg_coarse_entropy = np.mean([r['avg_coarse_entropy'] for r in results])
        avg_fine_entropy = np.mean([r['avg_fine_entropy'] for r in results])
        avg_ratio = np.mean([r['fine_max_attn'] / (r['coarse_max_attn'] + 1e-8) for r in results])

        f.write(f"\n**Averages:**\n")
        f.write(f"- Coarse Entropy: {avg_coarse_entropy:.3f}\n")
        f.write(f"- Fine Entropy: {avg_fine_entropy:.3f}\n")
        f.write(f"- Focus Ratio (Fine/Coarse): {avg_ratio:.3f}\n\n")

        if avg_fine_entropy < avg_coarse_entropy - 0.1:
            f.write("✅ Fine queries show more focused attention (lower entropy)\n\n")
        elif avg_fine_entropy > avg_coarse_entropy + 0.1:
            f.write("⚠️ Coarse queries show more focused attention (lower entropy)\n\n")
        else:
            f.write("➡️ Fine and coarse show similar attention focus\n\n")

        # Per-sample details
        f.write("## Per-Sample Results\n\n")
        f.write("| # | Video | Coarse | Fine | Δ | Last4 Coarse | Last4 Fine | Δ |\n")
        f.write("|---|-------|--------|------|---|--------------|------------|---|\n")

        for i, r in enumerate(results):
            delta1 = r['avg_coarse'] - r['avg_fine']
            delta2 = r['last4_coarse'] - r['last4_fine']
            vid = str(r['video_id'])[:15]
            f.write(f"| {i+1} | {vid} | {r['avg_coarse']:.4f} | {r['avg_fine']:.4f} | {delta1:+.4f} | {r['last4_coarse']:.4f} | {r['last4_fine']:.4f} | {delta2:+.4f} |\n")

        f.write("\n## Captions\n\n")
        for i, r in enumerate(results):
            f.write(f"### Sample {i+1}: {r['video_id']}\n\n")
            f.write(f"**Fine:** {r['caption_fine']}\n\n")
            f.write(f"**Coarse:** {r['caption_coarse']}\n\n")
            f.write("---\n\n")

        # Key findings
        f.write("## Key Findings\n\n")

        overall_delta = np.mean(all_avg_coarse) - np.mean(all_avg_fine)
        last4_delta = np.mean(all_last_coarse) - np.mean(all_last_fine)

        if overall_delta > 0.001:
            f.write(f"- ✅ Fine queries achieve {overall_delta:.4f} lower loss overall\n")
        elif overall_delta < -0.001:
            f.write(f"- ❌ Coarse queries achieve {-overall_delta:.4f} lower loss overall\n")
        else:
            f.write(f"- ➡️ Fine and coarse losses are essentially identical (Δ={overall_delta:.4f})\n")

        if last4_delta > 0.001:
            f.write(f"- ✅ Fine queries achieve {last4_delta:.4f} lower loss on last 4 frames\n")
        elif last4_delta < -0.001:
            f.write(f"- ❌ Coarse queries achieve {-last4_delta:.4f} lower loss on last 4 frames\n")
        else:
            f.write(f"- ➡️ Last 4 frame losses are essentially identical (Δ={last4_delta:.4f})\n")

    print(f"\nResults saved to {output_dir}")
    print(f"  - eval_XX_ID.png: Comprehensive visualizations")
    print(f"  - gif_XX_ID.gif: Attention comparison animations (with diff)")
    print(f"  - stats_XX_ID.png: Attention & loss statistics")
    print(f"  - EVALUATION_REPORT.md: Detailed analysis")


if __name__ == "__main__":
    main()
