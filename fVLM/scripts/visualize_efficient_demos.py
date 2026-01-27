#!/usr/bin/env python3
"""
Generate demos from the efficient joint-trained model.

Creates:
1. 64 samples with generated vs real captions
2. Attention visualization GIFs showing where model looks
3. Autoregressive generation (fine path only, no coarse)
"""

import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the model from training script
from scripts.train_dino_efficient import (
    EfficientFoveatedModel,
    PrecomputedDINODataset,
    CONFIG,
)


def load_model_and_tokenizer(checkpoint_path: str, device: str = "cuda"):
    """Load trained model and tokenizer."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = EfficientFoveatedModel(
        dino_dim=CONFIG["dino_dim"],
        llm_dim=CONFIG["llm_dim"],
        query_dim=CONFIG["query_dim"],
        lambda_coarse=CONFIG["lambda_coarse"],
        lambda_recon=CONFIG.get("lambda_recon", 0.1),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded from step {checkpoint['step']}")
    return model, tokenizer


def get_attention_weights(model, query, patch_features):
    """Get attention weights for visualization."""
    import math
    q = model.query_proj(query)  # [B, dino_dim]
    q = q.unsqueeze(1)  # [B, 1, dino_dim]

    attn_scores = torch.bmm(q, patch_features.transpose(1, 2))  # [B, 1, N]
    attn_weights = F.softmax(attn_scores / math.sqrt(model.dino_dim), dim=-1)

    return attn_weights.squeeze(1)  # [B, N]


def attention_to_heatmap(attn_weights, grid_size=18, img_size=256):
    """Convert attention weights to heatmap overlay."""
    # attn_weights: [N] where N = 325 (1 CLS + 18x18 patches)
    # Skip CLS token, reshape to grid
    patch_attn = attn_weights[1:].reshape(grid_size, grid_size)

    # Normalize to 0-1
    patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)

    # Resize to image size
    patch_attn_np = patch_attn.cpu().numpy()
    heatmap = Image.fromarray((patch_attn_np * 255).astype(np.uint8), mode='L')
    heatmap = heatmap.resize((img_size, img_size), Image.BILINEAR)

    # Convert to RGB colormap (red = high attention)
    heatmap_np = np.array(heatmap)
    rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    rgb[:, :, 0] = heatmap_np  # Red channel
    rgb[:, :, 2] = 255 - heatmap_np  # Blue channel (inverse)

    return Image.fromarray(rgb)


@torch.no_grad()
def generate_caption(model, tokenizer, dino_features, device, max_length=32):
    """Generate caption autoregressively using fine path only."""
    B, T, N, D = dino_features.shape

    # First do a coarse pass to get initial queries
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(T):
        z_t = model.query_attend(q_static, dino_features[:, t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # Get initial queries from coarse pass
    coarse_tok = model.coarse_token.expand(B, -1, -1)
    seq_coarse = torch.cat([coarse_tok, z_coarse_llm], dim=1)
    outputs_coarse = model.llm.model(inputs_embeds=seq_coarse)
    h_coarse = outputs_coarse.last_hidden_state
    current_queries = model.llm_to_query(h_coarse[:, 1:])

    # Now do fine pass with autoregressive queries
    q_init = model.q_init.expand(B, -1)

    z_fine_list = []
    attn_weights_list = []

    for t in range(T):
        if t == 0:
            q_t = q_init
        else:
            q_t = current_queries[:, t-1]

        # Get attention weights for visualization
        attn = get_attention_weights(model, q_t, dino_features[:, t])
        attn_weights_list.append(attn)

        z_t = model.query_attend(q_t, dino_features[:, t])
        z_fine_list.append(z_t)

    z_fine = torch.stack(z_fine_list, dim=1)  # [B, T, dino_dim]
    z_fine_llm = model.dino_to_llm(z_fine)
    z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

    # Get queries for next iteration (fine iteration 2)
    fine_tok = model.fine_token.expand(B, -1, -1)
    seq = torch.cat([fine_tok, z_fine_llm], dim=1)
    outputs = model.llm.model(inputs_embeds=seq)
    h = outputs.last_hidden_state
    current_queries = model.llm_to_query(h[:, 1:])

    # Now generate caption autoregressively
    # Start with the visual sequence
    generated_ids = []

    # Get BOS token
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    current_token = torch.tensor([[bos_id]], device=device).expand(B, -1)

    for _ in range(max_length):
        # Embed current token
        token_embed = model.llm.model.embed_tokens(current_token)

        # Concatenate with visual features
        full_seq = torch.cat([fine_tok, z_fine_llm, token_embed], dim=1)

        # Get next token prediction
        outputs = model.llm.model(inputs_embeds=full_seq)
        logits = model.llm.lm_head(outputs.last_hidden_state[:, -1:, :])

        # Sample next token (greedy)
        next_token = logits.argmax(dim=-1)
        generated_ids.append(next_token)

        # Update for next iteration
        current_token = torch.cat([current_token, next_token], dim=1)

        # Stop if EOS
        if (next_token == tokenizer.eos_token_id).all():
            break

    # Decode
    generated_ids = torch.cat(generated_ids, dim=1)
    captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return captions, attn_weights_list


def create_attention_gif(attn_weights_list, output_path, grid_size=18, img_size=256, fps=4):
    """Create GIF showing attention over frames."""
    frames = []

    for t, attn in enumerate(attn_weights_list):
        # Create heatmap
        heatmap = attention_to_heatmap(attn[0], grid_size, img_size)

        # Add frame number
        draw = ImageDraw.Draw(heatmap)
        draw.text((5, 5), f"Frame {t}", fill=(255, 255, 255))

        frames.append(np.array(heatmap))

    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def create_sample_visualization(
    sample_idx,
    real_caption,
    generated_caption,
    attn_weights_list,
    output_dir,
):
    """Create visualization for a single sample."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create attention GIF
    gif_path = output_dir / f"sample_{sample_idx:03d}_attention.gif"
    create_attention_gif(attn_weights_list, str(gif_path))

    # Create summary image
    summary_path = output_dir / f"sample_{sample_idx:03d}_summary.png"

    # Create a simple text summary image
    img = Image.new('RGB', (800, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"Sample {sample_idx}", fill=(0, 0, 0), font=font)
    draw.text((10, 40), f"Real: {real_caption[:100]}", fill=(0, 100, 0), font=font)
    draw.text((10, 80), f"Generated: {generated_caption[:100]}", fill=(0, 0, 100), font=font)

    img.save(summary_path)

    return gif_path, summary_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="outputs/dino_efficient_joint/checkpoint_latest.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/efficient_demos")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # Load dataset
    dataset = PrecomputedDINODataset(CONFIG["data_dir"], num_frames=CONFIG["num_frames"])

    # Generate demos
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Sample random indices
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)

    print(f"Generating {len(indices)} demos...")

    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]

        dino_features = sample["dino_features"].unsqueeze(0).to(device)
        real_caption = sample["caption"]

        # Generate caption
        generated_captions, attn_weights = generate_caption(
            model, tokenizer, dino_features, device
        )
        generated_caption = generated_captions[0]

        # Create visualization
        gif_path, summary_path = create_sample_visualization(
            i, real_caption, generated_caption, attn_weights, output_dir
        )

        results.append({
            "idx": int(idx),
            "real_caption": real_caption,
            "generated_caption": generated_caption,
            "gif_path": str(gif_path),
        })

    # Save results summary
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create markdown summary
    with open(output_dir / "DEMO_SUMMARY.md", "w") as f:
        f.write("# Efficient Joint Training Demos\n\n")
        f.write(f"Generated {len(results)} samples from checkpoint: {args.checkpoint}\n\n")
        f.write("## Samples\n\n")
        f.write("| # | Real Caption | Generated Caption |\n")
        f.write("|---|--------------|-------------------|\n")
        for r in results[:20]:  # First 20 in table
            real = r["real_caption"][:50] + "..." if len(r["real_caption"]) > 50 else r["real_caption"]
            gen = r["generated_caption"][:50] + "..." if len(r["generated_caption"]) > 50 else r["generated_caption"]
            f.write(f"| {r['idx']} | {real} | {gen} |\n")

        f.write("\n\n## Attention GIFs\n\n")
        for r in results[:10]:
            f.write(f"### Sample {r['idx']}\n")
            f.write(f"- Real: {r['real_caption']}\n")
            f.write(f"- Generated: {r['generated_caption']}\n")
            f.write(f"- ![Attention]({Path(r['gif_path']).name})\n\n")

    print(f"\nDemos saved to {output_dir}")
    print(f"Summary: {output_dir / 'DEMO_SUMMARY.md'}")


if __name__ == "__main__":
    main()
