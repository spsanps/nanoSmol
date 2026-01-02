"""
Generate video predictions in two modes:
1. Teacher forcing: Use all GT frames, predict next-frame latents
2. Autoregressive: Use only first frame GT, predict rest autoregressively

Downloads test videos from WebVid-10M for inference.
"""

import torch
import torch.nn.functional as F
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
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
                return torch.stack(frames)  # [T, 3, H, W] uint8
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
            print("Too many failures, stopping")
            break

    return videos


def compute_latents(frames, vae, device):
    """Compute VAE latents for frames."""
    # frames: [T, 3, H, W] uint8
    frames_vae = frames.float().to(device) / 255.0 * 2 - 1

    with torch.no_grad():
        latents = []
        for i in range(0, frames_vae.shape[0], 4):
            batch = frames_vae[i:i+4]
            latent = vae.encode(batch).latent_dist.sample() * 0.18215
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
    return latents  # [T, 4, 32, 32]


def decode_latents(latents, vae, device):
    """Decode VAE latents to images."""
    # latents: [T, 4, 32, 32]
    latents = latents.float().to(device) / 0.18215

    with torch.no_grad():
        images = []
        for i in range(0, latents.shape[0], 4):
            batch = latents[i:i+4]
            image = vae.decode(batch).sample
            images.append(image)
        images = torch.cat(images, dim=0)

    # Convert to uint8 [T, 3, H, W]
    images = (images.clamp(-1, 1) + 1) / 2 * 255
    return images.byte()


def normalize_for_dino(frames):
    """Normalize frames for DINO (from uint8)."""
    # frames: [T, 3, H, W] uint8
    frames_norm = frames.float() / 255.0
    mean = IMAGENET_MEAN.to(frames_norm.device)
    std = IMAGENET_STD.to(frames_norm.device)
    frames_norm = (frames_norm - mean) / std
    return frames_norm


@torch.no_grad()
def generate_teacher_forcing(model, frames, vae, device):
    """
    Generate predictions using teacher forcing (all GT frames).
    Returns predicted latents for frames 1 to T.
    """
    B = 1
    T = frames.shape[0]

    # Normalize for model
    frames_norm = normalize_for_dino(frames).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    # Compute GT latents
    gt_latents = compute_latents(frames, vae, device).unsqueeze(0)  # [1, T, 4, 32, 32]

    # Get text embeddings
    text_embeds = model.get_empty_text_embeds(B).to(device)

    # Encode all frames
    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Create caches
    all_caches = []
    for t in range(T):
        all_caches.append({'patch_features': patch_features[:, t]})

    # Pass 1: Coarse
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse = model.dino_to_llm(z_coarse)
    z_coarse = z_coarse / (z_coarse.std() + 1e-6) * model.visual_scale

    coarse_token = model.coarse_token.expand(B, -1, -1)
    N_text = text_embeds.shape[1]
    seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)

    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state

    h_for_queries = h_pass1[:, N_text + 1:]
    queries = model.llm_to_query(h_for_queries)

    # Pass 2: Fine with shifted queries
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

    # Predict latents
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, gt_latents[:, :-1]], dim=1)

    pred_latents = model.pred_head(h_fine_for_pred, prev_latents)  # [1, T, 4, 32, 32]

    return pred_latents.squeeze(0), gt_latents.squeeze(0)


@torch.no_grad()
def generate_autoregressive(model, frames, vae, device):
    """
    Generate predictions autoregressively (only first frame is GT).
    Predicts frame t+1 from frames 0..t, then uses predicted frame.
    """
    B = 1
    T = frames.shape[0]

    # Start with first GT frame
    gt_frame_0 = frames[0:1]  # [1, 3, H, W]
    gt_latent_0 = compute_latents(gt_frame_0, vae, device)  # [1, 4, 32, 32]

    # Store predictions
    pred_frames = [gt_frame_0.to(device)]  # First frame is GT
    pred_latents = [gt_latent_0]

    text_embeds = model.get_empty_text_embeds(B).to(device)
    N_text = text_embeds.shape[1]

    # Autoregressive generation
    for t in range(1, T):
        # Current frames (mix of GT frame 0 and predicted frames 1..t-1)
        current_frames = torch.stack([f.squeeze(0) for f in pred_frames], dim=0)  # [t, 3, H, W]
        current_latents = torch.cat(pred_latents, dim=0)  # [t, 4, 32, 32]

        # Normalize for DINO
        frames_norm = normalize_for_dino(current_frames).unsqueeze(0).to(device)  # [1, t, 3, H, W]

        # Encode current frames
        frames_flat = frames_norm.reshape(t, 3, 256, 256)
        _, cache_flat = model.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.unsqueeze(0)  # [1, t, N, D]

        all_caches = []
        for ti in range(t):
            all_caches.append({'patch_features': patch_features[:, ti]})

        # Pass 1: Coarse
        q_static = model.q_static.expand(B, -1)
        z_coarse_list = []
        for ti in range(t):
            z_ti = model.encoder.query_attend(q_static, all_caches[ti])
            z_coarse_list.append(z_ti)
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
        if t > 1:
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)
        else:
            shifted_q = q_init

        z_focused_list = []
        for ti in range(t):
            z_ti = model.encoder.query_attend(shifted_q[:, ti], all_caches[ti])
            z_focused_list.append(z_ti)
        z_focused = torch.stack(z_focused_list, dim=1)
        z_focused = model.dino_to_llm(z_focused)
        z_focused = z_focused / (z_focused.std() + 1e-6) * model.visual_scale

        fine_token = model.fine_token.expand(B, -1, -1)
        seq_pass2 = torch.cat([text_embeds, fine_token, z_focused], dim=1)

        outputs_pass2 = model.llm.model(inputs_embeds=seq_pass2)
        h_pass2 = outputs_pass2.last_hidden_state

        # Get hidden state for last frame (predict next)
        h_last = h_pass2[:, N_text + t - 1:N_text + t]  # [1, 1, llm_dim]

        # Previous latent for conditioning
        prev_latent = current_latents[-1:].unsqueeze(0)  # [1, 1, 4, 32, 32]

        # Predict next latent
        pred_latent = model.pred_head(h_last, prev_latent)  # [1, 1, 4, 32, 32]
        pred_latent = pred_latent.squeeze(1)  # [1, 4, 32, 32]

        # Decode to frame
        pred_frame = decode_latents(pred_latent, vae, device)  # [1, 3, H, W]

        pred_frames.append(pred_frame)
        pred_latents.append(pred_latent)

    # Stack all predictions
    all_pred_frames = torch.cat(pred_frames, dim=0)  # [T, 3, H, W]
    all_pred_latents = torch.cat(pred_latents, dim=0)  # [T, 4, 32, 32]

    # Also get GT latents for comparison
    gt_latents = compute_latents(frames, vae, device)

    return all_pred_latents, gt_latents, all_pred_frames


def create_comparison_gif(gt_frames, pred_frames_tf, pred_frames_ar, output_path, fps=4):
    """Create a side-by-side comparison GIF: GT | Teacher Forcing | Autoregressive"""
    T = gt_frames.shape[0]

    comparison_frames = []
    for t in range(T):
        gt = gt_frames[t].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        tf = pred_frames_tf[t].permute(1, 2, 0).cpu().numpy()
        ar = pred_frames_ar[t].permute(1, 2, 0).cpu().numpy()

        # Add labels
        label_height = 20
        gt_labeled = np.zeros((gt.shape[0] + label_height, gt.shape[1], 3), dtype=np.uint8)
        tf_labeled = np.zeros_like(gt_labeled)
        ar_labeled = np.zeros_like(gt_labeled)

        gt_labeled[label_height:] = gt
        tf_labeled[label_height:] = tf
        ar_labeled[label_height:] = ar

        # Combine horizontally
        combined = np.concatenate([gt_labeled, tf_labeled, ar_labeled], axis=1)
        comparison_frames.append(combined)

    imageio.mimsave(output_path, comparison_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 8

    output_dir = Path('outputs/generation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Load model
    print("Loading model...")
    checkpoint_path = Path('outputs/streaming/checkpoints/final.pt')
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

    # Generate for each video
    print("\nGenerating predictions...")
    for i, video in enumerate(tqdm(videos)):
        frames = video['frames']  # [T, 3, H, W] uint8

        # Teacher forcing
        pred_latents_tf, gt_latents = generate_teacher_forcing(model, frames, vae, device)
        pred_frames_tf = decode_latents(pred_latents_tf, vae, device)

        # Autoregressive
        pred_latents_ar, _, pred_frames_ar = generate_autoregressive(model, frames, vae, device)

        # Create comparison GIF
        create_comparison_gif(
            frames.to(device),
            pred_frames_tf,
            pred_frames_ar,
            output_dir / f'comparison_{i:02d}.gif'
        )

        # Also save individual GIFs
        # GT
        gt_np = [f.permute(1, 2, 0).cpu().numpy() for f in frames]
        imageio.mimsave(output_dir / f'gt_{i:02d}.gif', gt_np, fps=4, loop=0)

        # Teacher forcing
        tf_np = [f.permute(1, 2, 0).cpu().numpy() for f in pred_frames_tf]
        imageio.mimsave(output_dir / f'teacher_forcing_{i:02d}.gif', tf_np, fps=4, loop=0)

        # Autoregressive
        ar_np = [f.permute(1, 2, 0).cpu().numpy() for f in pred_frames_ar]
        imageio.mimsave(output_dir / f'autoregressive_{i:02d}.gif', ar_np, fps=4, loop=0)

        # Compute MSE
        mse_tf = F.mse_loss(pred_latents_tf, gt_latents).item()
        mse_ar = F.mse_loss(pred_latents_ar, gt_latents).item()

        print(f"\nVideo {i}: {video['caption'][:50]}...")
        print(f"  MSE Teacher Forcing: {mse_tf:.4f}")
        print(f"  MSE Autoregressive:  {mse_ar:.4f}")

    print(f"\nSaved to {output_dir}")
    print("Files:")
    print("  - comparison_XX.gif: Side-by-side (GT | TF | AR)")
    print("  - gt_XX.gif: Ground truth")
    print("  - teacher_forcing_XX.gif: Predictions with GT conditioning")
    print("  - autoregressive_XX.gif: Predictions with only first frame GT")


if __name__ == "__main__":
    main()
