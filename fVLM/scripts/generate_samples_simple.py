"""
Simplified generation script showing teacher forcing reconstructions.
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
    pixels = (latents + 1.0) / 2.0
    pixels = torch.clamp(pixels, 0, 1)
    pixels = (pixels * 255).byte()
    return pixels


@torch.no_grad()
def get_reconstructions(model, vae, frames, device):
    """
    Get model reconstructions via manual forward pass.
    Returns both coarse and fine predictions.
    """
    B = 1
    T = frames.shape[0]

    # Normalize frames for DINO
    raw_frames = normalize_for_dino(frames).unsqueeze(0).to(device)

    # Encode to VAE latents
    frames_vae = frames.float().unsqueeze(0).to(device) / 255.0 * 2 - 1
    frames_vae_flat = frames_vae.reshape(B * T, 3, 256, 256)
    vae_latents = vae.encode(frames_vae_flat.half()).latent_dist.sample() * 0.18215
    vae_latents = vae_latents.view(B, T, 4, vae_latents.shape[-2], vae_latents.shape[-1])

    # Get text embeddings
    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    # === Encode frames with DINO ===
    frames_flat = raw_frames.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    all_caches = []
    for t in range(T):
        all_caches.append({'patch_features': patch_features[:, t]})

    # === Pass 1: Coarse ===
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

    # Get coarse predictions
    h_coarse_for_pred = h_pass1[:, N_text:N_text + T]
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents_coarse = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
    pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents_coarse)

    # === Pass 2: Fine ===
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    z_fine_list = []
    for t in range(T):
        q_t = shifted_q[:, t]
        z_t = model.encoder.query_attend(q_t, all_caches[t])
        z_fine_list.append(z_t)
    z_fine = torch.stack(z_fine_list, dim=1)
    z_fine = model.dino_to_llm(z_fine)
    z_fine = z_fine / (z_fine.std() + 1e-6) * model.visual_scale

    fine_token = model.fine_token.expand(B, -1, -1)
    seq_pass2 = torch.cat([text_embeds, fine_token, z_fine], dim=1)

    outputs_pass2 = model.llm.model(inputs_embeds=seq_pass2)
    h_pass2 = outputs_pass2.last_hidden_state

    h_fine_for_pred = h_pass2[:, N_text:N_text + T]
    prev_latents_fine = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
    pred_fine = model.pred_head(h_fine_for_pred, prev_latents_fine)

    # Decode predictions
    pred_coarse_flat = pred_coarse.reshape(B * T, 4, pred_coarse.shape[-2], pred_coarse.shape[-1])
    pred_fine_flat = pred_fine.reshape(B * T, 4, pred_fine.shape[-2], pred_fine.shape[-1])

    frames_coarse = vae.decode((pred_coarse_flat / 0.18215).half()).sample
    frames_fine = vae.decode((pred_fine_flat / 0.18215).half()).sample

    frames_coarse = denormalize_from_vae(frames_coarse.float())
    frames_fine = denormalize_from_vae(frames_fine.float())

    frames_coarse = frames_coarse.reshape(B, T, 3, 256, 256).squeeze(0)
    frames_fine = frames_fine.reshape(B, T, 3, 256, 256).squeeze(0)

    # Compute loss
    loss_coarse = F.mse_loss(pred_coarse, vae_latents.float())
    loss_fine = F.mse_loss(pred_fine, vae_latents.float())

    return frames_coarse, frames_fine, loss_coarse.item(), loss_fine.item()


def create_comparison_grid(frames_gt, frames_coarse, frames_fine, output_path, title="", loss_coarse=0, loss_fine=0):
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
        axes[t, 1].set_title(f'Coarse (loss={loss_coarse:.4f}) t={t}')
        axes[t, 1].axis('off')

        # Fine reconstruction
        axes[t, 2].imshow(frames_fine[t].permute(1, 2, 0).cpu().numpy())
        axes[t, 2].set_title(f'Fine (loss={loss_fine:.4f}) t={t}')
        axes[t, 2].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_gif(frames_list, labels, output_path, fps=4):
    """Create side-by-side GIF for multiple predictions."""
    num_versions = len(frames_list)
    T = min([f.shape[0] for f in frames_list])

    gif_frames = []
    label_h = 30
    H, W = 256, 256

    for t in range(T):
        combined = np.zeros((H + label_h, W * num_versions, 3), dtype=np.uint8)

        for i, (frames, label) in enumerate(zip(frames_list, labels)):
            combined[label_h:, i*W:(i+1)*W] = frames[t].permute(1, 2, 0).cpu().numpy()

            # Add colored header
            if i == 0:
                combined[:label_h, i*W:(i+1)*W] = [80, 120, 80]  # Green for GT
            elif i == 1:
                combined[:label_h, i*W:(i+1)*W] = [80, 80, 120]  # Blue for coarse
            else:
                combined[:label_h, i*W:(i+1)*W] = [120, 80, 80]  # Red for fine

        gif_frames.append(combined)

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 5

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

    print("=" * 70)
    print("TEACHER FORCING RECONSTRUCTIONS")
    print("=" * 70)

    for i, video in enumerate(videos):
        print(f"\nVideo {i}: {video['caption'][:60]}...")
        frames = video['frames']

        # Get reconstructions
        frames_coarse, frames_fine, loss_coarse, loss_fine = get_reconstructions(
            model, vae, frames, device
        )

        print(f"  Coarse loss: {loss_coarse:.4f}")
        print(f"  Fine loss:   {loss_fine:.4f}")

        # Save comparison grid
        create_comparison_grid(
            frames, frames_coarse, frames_fine,
            output_dir / f'reconstruction_grid_{i:02d}.png',
            title=f"{video['caption'][:70]}",
            loss_coarse=loss_coarse,
            loss_fine=loss_fine
        )

        # Save GIF
        create_comparison_gif(
            [frames, frames_coarse, frames_fine],
            ["GT", "Coarse", "Fine"],
            output_dir / f'reconstruction_{i:02d}.gif'
        )

    print(f"\n{'=' * 70}")
    print(f"Saved all visualizations to {output_dir}")
    print("=" * 70)
    print("\nFiles:")
    print("  - reconstruction_grid_XX.png: GT | Coarse | Fine comparison")
    print("  - reconstruction_XX.gif: Side-by-side animation")


if __name__ == "__main__":
    main()
