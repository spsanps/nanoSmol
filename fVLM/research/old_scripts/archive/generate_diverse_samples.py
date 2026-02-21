"""
Generate diverse samples with text conditioning visualization.
Shows how text captions influence the model's predictions.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import requests
import subprocess
import tempfile
import re
from datasets import load_dataset
from diffusers import AutoencoderKL
import imageio
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# Constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_duration(dur_str):
    match = re.match(r'PT(\d+)H(\d+)M(\d+)S', dur_str)
    if match:
        return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
    match = re.match(r'PT(\d+)M(\d+)S', dur_str)
    if match:
        return int(match[1]) * 60 + int(match[2])
    match = re.match(r'PT(\d+)S', dur_str)
    if match:
        return int(match[1])
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


def get_diverse_videos(num_videos=10, num_frames=16, frame_size=256):
    """Download diverse test videos from WebVid."""
    print(f"Downloading {num_videos} diverse test videos...")
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    # Target diverse content categories
    target_keywords = [
        'person', 'animal', 'nature', 'city', 'ocean',
        'dance', 'sport', 'car', 'food', 'abstract'
    ]

    videos = []
    category_counts = {}
    tried = 0

    for sample in ds:
        if len(videos) >= num_videos:
            break

        duration = parse_duration(sample['duration'])
        if duration < 8 or duration > 60:
            continue

        caption = sample['name'].lower()

        # Try to get diverse categories
        category = None
        for keyword in target_keywords:
            if keyword in caption:
                category = keyword
                break

        # Skip if we already have too many from this category
        if category and category_counts.get(category, 0) >= 2:
            continue

        tried += 1
        frames = download_video(sample['contentUrl'], num_frames, frame_size)
        if frames is not None:
            videos.append({
                'frames': frames,
                'caption': sample['name'],
                'video_id': sample['videoid'],
                'category': category or 'other'
            })
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
            print(f"  [{len(videos)}/{num_videos}] {category or 'other'}: {sample['name'][:50]}...")

        if tried > num_videos * 10:
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
def get_reconstructions(model, tokenizer, vae, frames, caption, device):
    """
    Get model reconstructions with text conditioning.
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

    # Tokenize caption for text conditioning
    tokens = tokenizer(
        caption,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)

    # Get text embeddings from LLM
    text_embeds = model.llm.model.embed_tokens(input_ids)
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


def add_text_to_image(img_array, text, position='top', color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add text overlay to image array."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position
    if position == 'top':
        x = (img.width - text_w) // 2
        y = 5
    else:
        x = (img.width - text_w) // 2
        y = img.height - text_h - 5

    # Draw background rectangle
    padding = 5
    draw.rectangle(
        [(x - padding, y - padding), (x + text_w + padding, y + text_h + padding)],
        fill=bg_color
    )

    # Draw text
    draw.text((x, y), text, fill=color, font=font)

    return np.array(img)


def create_comparison_grid(frames_gt, frames_coarse, frames_fine, caption, output_path, loss_coarse=0, loss_fine=0):
    """Create a grid comparing GT | Coarse | Fine with caption."""
    T = min(16, frames_gt.shape[0])  # Show max 16 frames

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
        axes[t, 1].set_title(f'Coarse t={t}')
        axes[t, 1].axis('off')

        # Fine reconstruction
        axes[t, 2].imshow(frames_fine[t].permute(1, 2, 0).cpu().numpy())
        axes[t, 2].set_title(f'Fine t={t}')
        axes[t, 2].axis('off')

    # Title with caption and losses
    title = f'Caption: "{caption}"\nCoarse Loss: {loss_coarse:.4f} | Fine Loss: {loss_fine:.4f}'
    plt.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_gif(frames_gt, frames_coarse, frames_fine, caption, output_path, fps=4):
    """Create side-by-side GIF with caption overlay."""
    T = min(16, frames_gt.shape[0])

    gif_frames = []
    label_h = 40
    H, W = 256, 256

    for t in range(T):
        # Create canvas
        combined = np.zeros((H + label_h * 2, W * 3, 3), dtype=np.uint8)

        # Add frames
        combined[label_h:-label_h, 0:W] = frames_gt[t].permute(1, 2, 0).cpu().numpy()
        combined[label_h:-label_h, W:2*W] = frames_coarse[t].permute(1, 2, 0).cpu().numpy()
        combined[label_h:-label_h, 2*W:3*W] = frames_fine[t].permute(1, 2, 0).cpu().numpy()

        # Add colored headers
        combined[:label_h, 0:W] = [80, 120, 80]  # Green for GT
        combined[:label_h, W:2*W] = [80, 80, 120]  # Blue for coarse
        combined[:label_h, 2*W:3*W] = [120, 80, 80]  # Red for fine

        # Add caption at bottom
        combined[-label_h:, :] = [40, 40, 40]  # Dark background

        # Convert to PIL to add text
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = font

        # Draw labels
        draw.text((W//2 - 30, 10), "Ground Truth", fill=(255, 255, 255), font=font_small)
        draw.text((W + W//2 - 20, 10), "Coarse", fill=(255, 255, 255), font=font_small)
        draw.text((2*W + W//2 - 15, 10), "Fine", fill=(255, 255, 255), font=font_small)

        # Draw caption (truncate if needed)
        caption_short = caption[:80] + "..." if len(caption) > 80 else caption
        draw.text((10, H + label_h + 10), f"Caption: {caption_short}", fill=(255, 255, 255), font=font_small)

        gif_frames.append(np.array(img))

    imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 16
    num_examples = 20  # More diverse examples

    output_dir = Path('outputs/generation_diverse')
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

    # Load tokenizer for text conditioning
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['llm_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    # Download diverse test videos
    videos = get_diverse_videos(num_examples, num_frames)
    print(f"\nGot {len(videos)} diverse test videos\n")

    print("=" * 80)
    print("TEXT-CONDITIONED VIDEO RECONSTRUCTION")
    print("=" * 80)
    print(f"Model was trained with text captions to guide attention.")
    print(f"Showing diverse video categories with their captions.\n")

    results = []

    for i, video in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] Category: {video['category'].upper()}")
        print(f"  Caption: \"{video['caption']}\"")

        frames = video['frames']
        caption = video['caption']

        # Get reconstructions with text conditioning
        frames_coarse, frames_fine, loss_coarse, loss_fine = get_reconstructions(
            model, tokenizer, vae, frames, caption, device
        )

        print(f"  Coarse loss: {loss_coarse:.4f}")
        print(f"  Fine loss:   {loss_fine:.4f}")
        print(f"  Improvement: {((loss_coarse - loss_fine) / loss_coarse * 100):.1f}%")

        results.append({
            'category': video['category'],
            'caption': caption,
            'loss_coarse': loss_coarse,
            'loss_fine': loss_fine,
        })

        # Save comparison grid
        create_comparison_grid(
            frames, frames_coarse, frames_fine, caption,
            output_dir / f'reconstruction_grid_{i:02d}.png',
            loss_coarse=loss_coarse,
            loss_fine=loss_fine
        )

        # Save GIF with caption
        create_comparison_gif(
            frames, frames_coarse, frames_fine, caption,
            output_dir / f'reconstruction_{i:02d}.gif'
        )

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)

    avg_loss_coarse = np.mean([r['loss_coarse'] for r in results])
    avg_loss_fine = np.mean([r['loss_fine'] for r in results])
    avg_improvement = (avg_loss_coarse - avg_loss_fine) / avg_loss_coarse * 100

    print(f"\nAverage Coarse Loss: {avg_loss_coarse:.4f}")
    print(f"Average Fine Loss:   {avg_loss_fine:.4f}")
    print(f"Average Improvement: {avg_improvement:.1f}%")

    print(f"\nPer-category results:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cat_results in categories.items():
        avg_cat_loss = np.mean([r['loss_fine'] for r in cat_results])
        print(f"  {cat:12s}: {avg_cat_loss:.4f} avg loss ({len(cat_results)} samples)")

    print(f"\n{'=' * 80}")
    print(f"Saved all visualizations to {output_dir}/")
    print("=" * 80)
    print("\nFiles:")
    print("  - reconstruction_grid_XX.png: GT | Coarse | Fine comparison")
    print("  - reconstruction_XX.gif: Side-by-side animation with caption")
    print("\nNote: Text captions guide the model's attention during reconstruction.")


if __name__ == "__main__":
    main()
