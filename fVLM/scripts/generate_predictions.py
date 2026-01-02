"""
Inference script for Foveated VLM

Loads trained model and generates next-frame predictions from video input.
Decodes VAE latents to visualize what the model predicts.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.foveated_vlm import FoveatedVideoModel
from diffusers import AutoencoderKL
from torchvision import transforms


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained FoveatedVideoModel from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = FoveatedVideoModel(
        dino_model=model_config.get('dino_model', 'facebook/dinov2-small'),
        llm_model=model_config.get('llm_model', 'HuggingFaceTB/SmolLM2-135M-Instruct'),
        dino_dim=model_config.get('dino_dim', 384),
        llm_dim=model_config.get('llm_dim', 576),
        query_dim=model_config.get('query_dim', 384),
        lambda_coarse=model_config.get('lambda_coarse', 1.0),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    return model


def load_vae(device: str = "cuda"):
    """Load Stable Diffusion VAE for decoding latents."""
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    return vae


def load_video_frames(frames_dir: str, video_name: str, num_frames: int = 8):
    """Load pre-decoded frames from disk."""
    frames_path = Path(frames_dir) / f"{video_name}.pt"
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames not found: {frames_path}")

    frames_tensor = torch.load(frames_path)  # [T, 3, H, W] uint8

    # Select frames
    total_frames = frames_tensor.shape[0]
    if total_frames < num_frames:
        raise ValueError(f"Video has only {total_frames} frames, need {num_frames}")

    # Sample evenly
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    frames = frames_tensor[indices]  # [num_frames, 3, H, W]

    # Normalize to [-1, 1] for VAE / ImageNet normalization
    frames = frames.float() / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames_normalized = (frames - mean) / std

    return frames_normalized, frames  # Return both normalized and original


def load_latents(latent_dir: str, video_name: str, num_frames: int = 8):
    """Load precomputed VAE latents."""
    latent_path = Path(latent_dir) / f"{video_name}.pt"
    if not latent_path.exists():
        raise FileNotFoundError(f"Latents not found: {latent_path}")

    latents = torch.load(latent_path)  # [T, 4, 32, 32]

    total_frames = latents.shape[0]
    if total_frames < num_frames:
        raise ValueError(f"Latents have only {total_frames} frames, need {num_frames}")

    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    return latents[indices]


def decode_latents(vae, latents):
    """Decode VAE latents to images."""
    # VAE expects latents scaled by 0.18215
    latents = latents.float() / 0.18215

    with torch.no_grad():
        images = vae.decode(latents).sample

    # Clamp and convert to [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def predict_next_frames(model, frames, vae_latents, device):
    """Run model forward to get predicted latents."""
    B = 1
    frames = frames.unsqueeze(0).to(device)  # [1, T, 3, H, W]
    vae_latents = vae_latents.unsqueeze(0).to(device)  # [1, T, 4, 32, 32]

    text_embeds = model.get_empty_text_embeds(B).to(device)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # We need to access intermediate predictions
            # Run the forward pass manually to extract predictions
            T = frames.shape[1]
            N_text = text_embeds.shape[1]

            # Encode frames
            frames_flat = frames.reshape(B * T, 3, 256, 256)
            _, cache_flat = model.encoder.encode_patches(frames_flat)

            patch_features_flat = cache_flat['patch_features']
            N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
            patch_features = patch_features_flat.reshape(B, T, N, D)

            all_caches = []
            for t in range(T):
                all_caches.append({'patch_features': patch_features[:, t]})

            # Pass 1: Static query
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

            # Pass 1 predictions (coarse)
            h_coarse_for_pred = h_pass1[:, N_text:N_text + T]
            z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
            prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
            pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)

            # Get dynamic queries
            h_for_queries = h_pass1[:, N_text + 1:]
            queries = model.llm_to_query(h_for_queries)
            q_init = model.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

            # Pass 2: Dynamic queries
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

            # Pass 2 predictions (fine)
            h_fine_for_pred = h_pass2[:, N_text:N_text + T]
            pred_fine = model.pred_head(h_fine_for_pred, prev_latents)

    return pred_fine.squeeze(0).float(), pred_coarse.squeeze(0).float()


def create_comparison_grid(original, gt_decoded, pred_fine_decoded, pred_coarse_decoded, output_path):
    """Create a comparison grid showing original, ground truth, and predictions."""
    T = original.shape[0]

    # Convert tensors to numpy
    def to_numpy(tensor):
        return (tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    original_np = to_numpy(original)
    gt_np = to_numpy(gt_decoded)
    pred_fine_np = to_numpy(pred_fine_decoded)
    pred_coarse_np = to_numpy(pred_coarse_decoded)

    # Create grid: 4 rows (original, GT decoded, fine pred, coarse pred) x T columns
    H, W = original_np.shape[1], original_np.shape[2]
    grid = np.zeros((4 * H + 30, T * W, 3), dtype=np.uint8)

    # Add labels
    from PIL import ImageDraw, ImageFont

    for t in range(T):
        grid[0:H, t*W:(t+1)*W] = original_np[t]
        grid[H:2*H, t*W:(t+1)*W] = gt_np[t]
        grid[2*H:3*H, t*W:(t+1)*W] = pred_fine_np[t]
        grid[3*H:4*H, t*W:(t+1)*W] = pred_coarse_np[t]

    # Convert to PIL and add labels
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)

    # Add row labels
    labels = ["Input Frames", "GT (VAE decoded)", "Pred Fine (dynamic Q)", "Pred Coarse (static Q)"]
    for i, label in enumerate(labels):
        draw.text((5, i * H + 5), label, fill=(255, 255, 255))

    img.save(output_path)
    print(f"Saved comparison grid: {output_path}")

    return img


def create_gif(frames_list, output_path, fps=4):
    """Create animated GIF from frames."""
    images = []
    for frame in frames_list:
        if isinstance(frame, torch.Tensor):
            frame = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(frame))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,
        loop=0
    )
    print(f"Saved GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/final.pt')
    parser.add_argument('--frames_dir', type=str, default='data/frames')
    parser.add_argument('--latent_dir', type=str, default='data/latents')
    parser.add_argument('--video_name', type=str, default=None, help='Video name (without extension)')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions')
    parser.add_argument('--num_frames', type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and VAE
    model = load_model(args.checkpoint, device)
    vae = load_vae(device)

    # Find a video if not specified
    if args.video_name is None:
        frames_files = list(Path(args.frames_dir).glob("*.pt"))
        if not frames_files:
            raise ValueError(f"No frame files found in {args.frames_dir}")
        args.video_name = frames_files[0].stem
        print(f"Using video: {args.video_name}")

    # Load data
    print(f"\nLoading video: {args.video_name}")
    frames_normalized, frames_original = load_video_frames(args.frames_dir, args.video_name, args.num_frames)
    vae_latents = load_latents(args.latent_dir, args.video_name, args.num_frames)

    print(f"Frames shape: {frames_normalized.shape}")
    print(f"Latents shape: {vae_latents.shape}")

    # Run prediction
    print("\nRunning model inference...")
    pred_fine, pred_coarse = predict_next_frames(model, frames_normalized, vae_latents, device)

    # Decode latents
    print("\nDecoding latents with VAE...")
    with torch.no_grad():
        gt_decoded = decode_latents(vae, vae_latents.to(device))
        pred_fine_decoded = decode_latents(vae, pred_fine.to(device))
        pred_coarse_decoded = decode_latents(vae, pred_coarse.to(device))

    # Compute metrics
    mse_fine = F.mse_loss(pred_fine, vae_latents.to(device)).item()
    mse_coarse = F.mse_loss(pred_coarse, vae_latents.to(device)).item()

    print(f"\n{'='*50}")
    print("Prediction Quality (MSE in latent space):")
    print(f"  Fine (dynamic query):  {mse_fine:.4f}")
    print(f"  Coarse (static query): {mse_coarse:.4f}")
    print(f"  Ratio (coarse/fine):   {mse_coarse/mse_fine:.4f}")
    print(f"{'='*50}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Comparison grid
    grid_path = output_dir / f"{args.video_name}_comparison.png"
    create_comparison_grid(
        frames_original,
        gt_decoded,
        pred_fine_decoded,
        pred_coarse_decoded,
        grid_path
    )

    # Individual GIFs
    create_gif(list(frames_original), output_dir / f"{args.video_name}_input.gif")
    create_gif(list(gt_decoded), output_dir / f"{args.video_name}_gt.gif")
    create_gif(list(pred_fine_decoded), output_dir / f"{args.video_name}_pred_fine.gif")
    create_gif(list(pred_coarse_decoded), output_dir / f"{args.video_name}_pred_coarse.gif")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
