"""
Generate video samples showing model predictions.

Two modes:
1. Teacher Forcing: Use ground truth frames to guide attention, predict reconstructions
2. Autoregressive: Generate frame-by-frame using predicted queries
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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


def get_test_videos(num_videos=5, num_frames=16, frame_size=256):
    """Download diverse test videos from WebVid."""
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
            print(f"  Downloaded {len(videos)}/{num_videos}: {sample['name'][:60]}...")

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


def denormalize_from_vae(latents):
    """Convert VAE latents back to pixel space."""
    # Latents are in [-1, 1] range after VAE decode
    pixels = (latents + 1.0) / 2.0  # [0, 1]
    pixels = torch.clamp(pixels, 0, 1)
    pixels = (pixels * 255).byte()
    return pixels


@torch.no_grad()
def teacher_forcing_reconstruction(model, vae, frames, device):
    """
    Teacher forcing mode: Use ground truth frames to guide attention.
    Returns reconstructed frames.
    """
    B = 1
    T = frames.shape[0]

    # Normalize and encode frames
    frames_norm = normalize_for_dino(frames).unsqueeze(0).to(device)
    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)

    # Encode to DINO features
    _, cache_flat = model.encoder.encode_patches(frames_flat)
    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Encode to VAE latents (ground truth)
    frames_vae = frames.float().unsqueeze(0).to(device) / 255.0 * 2 - 1
    frames_vae_flat = frames_vae.reshape(B * T, 3, 256, 256)
    latents_true = vae.encode(frames_vae_flat.half()).latent_dist.sample() * 0.18215
    latents_true = latents_true.view(B, T, 4, latents_true.shape[-2], latents_true.shape[-1])

    # Get text embeddings (empty for visualization)
    text_embeds = model.get_empty_text_embeds(B).to(device)

    # Create caches
    all_caches = []
    for t in range(T):
        all_caches.append({'patch_features': patch_features[:, t]})

    # Two-pass forward (teacher forcing)
    loss, outputs = model(
        all_caches,
        latents_true.float(),
        text_embeds,
        return_predictions=True
    )

    # Decode predictions
    preds_coarse = outputs['predictions_coarse']  # [B, T, 4, H, W]
    preds_fine = outputs['predictions_fine']

    # Decode both predictions
    preds_coarse_flat = preds_coarse.reshape(B * T, 4, preds_coarse.shape[-2], preds_coarse.shape[-1])
    preds_fine_flat = preds_fine.reshape(B * T, 4, preds_fine.shape[-2], preds_fine.shape[-1])

    frames_coarse = vae.decode((preds_coarse_flat / 0.18215).half()).sample
    frames_fine = vae.decode((preds_fine_flat / 0.18215).half()).sample

    frames_coarse = denormalize_from_vae(frames_coarse.float())
    frames_fine = denormalize_from_vae(frames_fine.float())

    frames_coarse = frames_coarse.reshape(B, T, 3, 256, 256).squeeze(0)
    frames_fine = frames_fine.reshape(B, T, 3, 256, 256).squeeze(0)

    return frames_coarse, frames_fine, loss.item()


@torch.no_grad()
def autoregressive_generation(model, vae, frames, device, num_pred_frames=8):
    """
    Autoregressive mode: Generate future frames one by one.
    Uses first half of frames as context, predicts second half.
    """
    B = 1
    T = frames.shape[0]
    context_frames = T // 2  # Use first half as context

    # Normalize and encode context frames
    frames_norm = normalize_for_dino(frames[:context_frames]).unsqueeze(0).to(device)
    frames_flat = frames_norm.reshape(B * context_frames, 3, 256, 256)

    # Encode context to DINO features
    _, cache_flat = model.encoder.encode_patches(frames_flat)
    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, context_frames, N, D)

    # Get text embeddings
    text_embeds = model.get_empty_text_embeds(B).to(device)

    # Initialize: Use context frames to get initial queries
    # Encode context frames to VAE
    frames_vae = frames[:context_frames].float().unsqueeze(0).to(device) / 255.0 * 2 - 1
    frames_vae_flat = frames_vae.reshape(B * context_frames, 3, 256, 256)
    latents_ctx = vae.encode(frames_vae_flat.half()).latent_dist.sample() * 0.18215
    latents_ctx = latents_ctx.view(B, context_frames, 4, latents_ctx.shape[-2], latents_ctx.shape[-1])

    # Create caches for context
    ctx_caches = []
    for t in range(context_frames):
        ctx_caches.append({'patch_features': patch_features[:, t]})

    # Get queries from context (Pass 1 + Pass 2)
    # Pass 1: Coarse
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(context_frames):
        z_t = model.encoder.query_attend(q_static, ctx_caches[t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    # LLM forward
    coarse_token = model.coarse_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state
    N_text = text_embeds.shape[1]
    h_for_queries = h_pass1[:, N_text + 1:]
    queries_ctx = model.llm_to_query(h_for_queries)  # [B, context_frames, query_dim]

    # Now generate future frames autoregressively
    predicted_frames = []
    current_query = queries_ctx[:, -1]  # Start from last context query

    for i in range(num_pred_frames):
        # Use current query to attend and predict next frame
        # Since we don't have future DINO features, we'll use the model in a
        # simplified mode: predict next latent from query alone
        # This is a limitation - in a true autoregressive setup, we'd:
        # 1. Decode predicted latent to frame
        # 2. Encode frame to DINO features
        # 3. Use those features for next step

        # For now, let's use the last context frame's features as a proxy
        # (This is a simplified version - a full implementation would decode->encode)
        last_cache = ctx_caches[-1]

        # Attend with current query
        z_pred = model.encoder.query_attend(current_query, last_cache)
        z_pred = model.dino_to_llm(z_pred.unsqueeze(1))
        z_pred = z_pred / (z_pred.std() + 1e-6) * model.visual_scale

        # Predict latent
        fine_token = model.fine_token.expand(B, -1, -1)
        seq_pass2 = torch.cat([text_embeds, fine_token, z_pred], dim=1)
        outputs_pass2 = model.llm(inputs_embeds=seq_pass2)
        logits = outputs_pass2.logits[:, N_text + 1:]

        # Project to latent space
        pred_latent = model.logits_to_latent(logits)  # [B, 1, latent_tokens, latent_dim]

        # Reshape to spatial
        pred_latent_spatial = model.reshape_to_spatial(pred_latent)  # [B, 1, 4, H, W]

        # Decode to frame
        pred_frame = vae.decode((pred_latent_spatial.squeeze(1) / 0.18215).half()).sample
        pred_frame = denormalize_from_vae(pred_frame.float())

        predicted_frames.append(pred_frame.squeeze(0))

        # Update query for next iteration (simplified - reuse projection)
        # In full version, would encode pred_frame and get new query
        current_query = model.llm_to_query(outputs_pass2.last_hidden_state[:, N_text + 1:])[:, 0]

    predicted_frames = torch.stack(predicted_frames, dim=0)  # [num_pred, 3, H, W]

    return predicted_frames


def create_comparison_grid(frames_gt, frames_coarse, frames_fine, output_path, title=""):
    """Create a grid comparing GT | Coarse | Fine reconstructions."""
    T = frames_gt.shape[0]

    fig, axes = plt.subplots(T, 3, figsize=(12, 4 * T))
    if T == 1:
        axes = axes.reshape(1, -1)

    for t in range(T):
        # Ground truth
        axes[t, 0].imshow(frames_gt[t].permute(1, 2, 0).cpu().numpy())
        axes[t, 0].set_title(f'Ground Truth t={t}')
        axes[t, 0].axis('off')

        # Coarse reconstruction
        axes[t, 1].imshow(frames_coarse[t].permute(1, 2, 0).cpu().numpy())
        axes[t, 1].set_title(f'Coarse Pred t={t}')
        axes[t, 1].axis('off')

        # Fine reconstruction
        axes[t, 2].imshow(frames_fine[t].permute(1, 2, 0).cpu().numpy())
        axes[t, 2].set_title(f'Fine Pred t={t}')
        axes[t, 2].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_gif(frames_gt, frames_pred, output_path, fps=4, label_gt="GT", label_pred="Pred"):
    """Create side-by-side GIF."""
    T = min(frames_gt.shape[0], frames_pred.shape[0])

    gif_frames = []
    label_h = 30
    H, W = 256, 256

    for t in range(T):
        combined = np.zeros((H + label_h, W * 2, 3), dtype=np.uint8)

        # Add frames
        combined[label_h:, :W] = frames_gt[t].permute(1, 2, 0).cpu().numpy()
        combined[label_h:, W:] = frames_pred[t].permute(1, 2, 0).cpu().numpy()

        # Add headers
        combined[:label_h, :W] = [80, 120, 80]    # Green
        combined[:label_h, W:] = [120, 80, 80]    # Red

        gif_frames.append(combined)

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 5  # Number of videos to test

    output_dir = Path('outputs/generation')
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

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    # Download test videos
    videos = get_test_videos(num_examples, num_frames)
    print(f"\nGot {len(videos)} test videos\n")

    # Process each video
    print("=" * 70)
    print("TEACHER FORCING MODE (using ground truth frames)")
    print("=" * 70)

    for i, video in enumerate(videos):
        print(f"\nVideo {i}: {video['caption'][:60]}...")
        frames = video['frames']

        # Teacher forcing reconstruction
        frames_coarse, frames_fine, loss = teacher_forcing_reconstruction(
            model, vae, frames, device
        )

        print(f"  Loss: {loss:.4f}")

        # Save comparison grid
        create_comparison_grid(
            frames, frames_coarse, frames_fine,
            output_dir / f'teacher_forcing_grid_{i:02d}.png',
            title=f"Teacher Forcing: {video['caption'][:60]}"
        )

        # Save GIFs
        create_comparison_gif(
            frames, frames_fine,
            output_dir / f'teacher_forcing_{i:02d}.gif',
            label_gt="Ground Truth",
            label_pred="Fine Prediction"
        )

    print(f"\n{'=' * 70}")
    print("AUTOREGRESSIVE MODE (generating future frames)")
    print("=" * 70)
    print("Note: Simplified autoregressive - uses context to predict future\n")

    for i, video in enumerate(videos):
        print(f"\nVideo {i}: {video['caption'][:60]}...")
        frames = video['frames']

        # Autoregressive generation
        try:
            predicted_frames = autoregressive_generation(
                model, vae, frames, device, num_pred_frames=8
            )

            # Compare predicted with ground truth second half
            gt_future = frames[8:]  # Second half

            create_comparison_grid(
                gt_future, predicted_frames, predicted_frames,
                output_dir / f'autoregressive_grid_{i:02d}.png',
                title=f"Autoregressive: {video['caption'][:60]}"
            )

            create_comparison_gif(
                gt_future, predicted_frames,
                output_dir / f'autoregressive_{i:02d}.gif',
                label_gt="Ground Truth",
                label_pred="Generated"
            )

            print(f"  Generated {predicted_frames.shape[0]} frames")
        except Exception as e:
            print(f"  Autoregressive failed: {e}")
            continue

    print(f"\n{'=' * 70}")
    print(f"Saved all visualizations to {output_dir}")
    print("=" * 70)
    print("\nFiles:")
    print("  - teacher_forcing_grid_XX.png: GT | Coarse | Fine comparison")
    print("  - teacher_forcing_XX.gif: GT vs Fine animation")
    print("  - autoregressive_grid_XX.png: Future frame predictions")
    print("  - autoregressive_XX.gif: GT vs Generated animation")


if __name__ == "__main__":
    main()
