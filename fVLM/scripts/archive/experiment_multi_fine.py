#!/usr/bin/env python3
"""
Experiment: Multi-Fine Iteration Training

Architecture:
  Pass 0 (Coarse): q_static → z_coarse → LLM → queries_1
  Pass 1 (Fine 1): queries_1 → z_fine_1 → LLM → queries_2
  Pass 2 (Fine 2): queries_2 → z_fine_2 → LLM → queries_3
  Pass 3 (Fine 3): queries_3 → z_fine_3 (final prediction)

Hypothesis: Training with multiple fine iterations will:
1. Teach the model to generate queries from fine features (not just coarse)
2. Close the train-test gap discovered in autoregressive inference
3. Potentially improve the fine/coarse ratio further

Loss options:
- final_only: Loss only on final fine pass
- all_passes: Loss on all passes (weighted)
- progressive: Increasing weight on later passes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
import requests
import subprocess
import tempfile
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dino(frames, device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (frames.to(device) - mean) / std


def parse_duration(dur_str: str) -> int:
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


def download_video(url: str, timeout: int = 15) -> bytes:
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            content = b''
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 50 * 1024 * 1024:
                    break
            return content
    except:
        pass
    return None


def extract_frames(video_bytes: bytes, num_frames: int, size: int) -> torch.Tensor:
    try:
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
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode != 0:
                    return None
                frame_files = sorted(Path(tmpdir).glob('frame_*.jpg'))
                if len(frame_files) < num_frames:
                    return None
                indices = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
                frames = []
                from PIL import Image
                for idx in indices:
                    img = Image.open(frame_files[idx]).convert('RGB')
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)
                return torch.stack(frames)
    except:
        return None


def forward_multi_fine_captioning(
    model,
    frames_norm,
    caption_ids,
    caption_mask,
    tokenizer,
    num_fine_iterations: int = 3,
    loss_mode: str = "final_only"  # "final_only", "all_passes", "progressive"
):
    """
    Forward pass with multiple fine iterations.

    Coarse → Fine_1 → Fine_2 → ... → Fine_N

    Each fine pass generates queries from the previous fine features.
    """
    B, T = frames_norm.shape[:2]
    device = frames_norm.device

    # Encode all frames with DINO
    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Build per-frame caches
    all_caches = []
    if 'kv_cache' in cache_flat:
        num_layers = len(cache_flat['kv_cache'])
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

    # Get caption embeddings
    caption_embeds = model.llm.model.embed_tokens(caption_ids)
    caption_targets = caption_ids[:, 1:]

    losses = []

    # === Pass 0: Coarse (baseline) ===
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = [model.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # Coarse caption loss (for comparison)
    coarse_token = model.coarse_token.expand(B, -1, -1)
    seq_coarse = torch.cat([coarse_token, z_coarse_llm, caption_embeds], dim=1)
    outputs_coarse = model.llm.model(inputs_embeds=seq_coarse)
    logits_coarse = model.llm.lm_head(outputs_coarse.last_hidden_state)
    caption_logits_coarse = logits_coarse[:, 1+T:-1, :]
    loss_coarse = F.cross_entropy(
        caption_logits_coarse.reshape(-1, caption_logits_coarse.size(-1)),
        caption_targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )

    # Generate first queries from coarse
    no_text = model.no_text_token.expand(B, -1, -1)
    seq_pass0 = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_pass0 = model.llm.model(inputs_embeds=seq_pass0)
    h_pass0 = outputs_pass0.last_hidden_state
    queries = model.llm_to_query(h_pass0[:, 2:])  # [B, T, query_dim]

    # Shift queries for first fine pass
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    current_queries = torch.cat([q_init, queries[:, :-1]], dim=1)  # [B, T, query_dim]

    fine_token = model.fine_token.expand(B, -1, -1)

    # === Fine iterations ===
    for iteration in range(num_fine_iterations):
        # Extract fine features with current queries
        z_fine_list = [model.encoder.query_attend(current_queries[:, t], all_caches[t]) for t in range(T)]
        z_fine = torch.stack(z_fine_list, dim=1)
        z_fine_llm = model.dino_to_llm(z_fine)
        z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

        # Compute caption loss for this iteration
        seq_fine = torch.cat([fine_token, z_fine_llm, caption_embeds], dim=1)
        outputs_fine = model.llm.model(inputs_embeds=seq_fine)
        logits_fine = model.llm.lm_head(outputs_fine.last_hidden_state)
        caption_logits_fine = logits_fine[:, 1+T:-1, :]
        loss_fine_iter = F.cross_entropy(
            caption_logits_fine.reshape(-1, caption_logits_fine.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        losses.append(loss_fine_iter)

        # Generate next queries from fine features (for next iteration)
        if iteration < num_fine_iterations - 1:
            seq_query = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
            outputs_query = model.llm.model(inputs_embeds=seq_query)
            h_query = outputs_query.last_hidden_state
            next_queries = model.llm_to_query(h_query[:, 2:])
            current_queries = torch.cat([q_init, next_queries[:, :-1]], dim=1)

    # Compute final loss based on mode
    if loss_mode == "final_only":
        loss_fine = losses[-1]
    elif loss_mode == "all_passes":
        loss_fine = sum(losses) / len(losses)
    elif loss_mode == "progressive":
        # Weight later passes more heavily: [0.1, 0.2, 0.3, 0.4] for 4 passes
        weights = [(i + 1) / sum(range(1, len(losses) + 1)) for i in range(len(losses))]
        loss_fine = sum(w * l for w, l in zip(weights, losses))
    else:
        loss_fine = losses[-1]

    return loss_fine, loss_coarse, losses


def webvid_generator(num_frames=8, frame_size=256, min_duration=8, max_duration=20):
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)
    for sample in ds:
        try:
            duration = parse_duration(sample.get('duration', ''))
            if duration < min_duration or duration > max_duration:
                continue
            url = sample.get('contentUrl')
            caption = sample.get('name', '')
            if not url or not caption or len(caption) < 10:
                continue
            video_bytes = download_video(url)
            if video_bytes is None:
                continue
            frames = extract_frames(video_bytes, num_frames, frame_size)
            if frames is None:
                continue
            yield frames, caption
        except:
            continue


def main(
    num_fine_iterations: int = 3,
    loss_mode: str = "progressive",
    num_steps: int = 1000,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 3e-5,
    num_frames: int = 8,
    checkpoint_path: str = None,
    output_dir: str = "outputs/multi_fine",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="foveated-vlm-multi-fine",
        name=f"fine_iter_{num_fine_iterations}_{loss_mode}",
        config={
            "num_fine_iterations": num_fine_iterations,
            "loss_mode": loss_mode,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "num_frames": num_frames,
        }
    )

    print("=" * 70)
    print(f"MULTI-FINE ITERATION EXPERIMENT")
    print(f"  Fine iterations: {num_fine_iterations}")
    print(f"  Loss mode: {loss_mode}")
    print(f"  Steps: {num_steps}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    data_gen = webvid_generator(num_frames=num_frames)

    step = 0
    accum_loss = 0
    accum_fine = 0
    accum_coarse = 0
    accum_count = 0

    pbar = tqdm(total=num_steps, desc="Training")

    for frames, caption in data_gen:
        if step >= num_steps:
            break

        try:
            # Prepare inputs
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            caption_encoded = tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=32
            )
            caption_ids = caption_encoded['input_ids'].to(device)
            caption_mask = caption_encoded['attention_mask'].to(device)

            # Forward pass
            loss_fine, loss_coarse, iter_losses = forward_multi_fine_captioning(
                model, frames_norm, caption_ids, caption_mask, tokenizer,
                num_fine_iterations=num_fine_iterations,
                loss_mode=loss_mode
            )

            # Combined loss (fine + small coarse regularization)
            loss = loss_fine + 0.1 * loss_coarse
            loss = loss / grad_accum
            loss.backward()

            accum_loss += loss.item() * grad_accum
            accum_fine += loss_fine.item()
            accum_coarse += loss_coarse.item()
            accum_count += 1

            # Gradient accumulation step
            if accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                pbar.update(1)

                # Compute ratio
                avg_fine = accum_fine / accum_count
                avg_coarse = accum_coarse / accum_count
                ratio = avg_coarse / avg_fine if avg_fine > 0 else 1.0

                # Log
                wandb.log({
                    "loss": accum_loss / accum_count,
                    "loss_fine": avg_fine,
                    "loss_coarse": avg_coarse,
                    "ratio": ratio,
                    "step": step,
                })

                if step % 25 == 0:
                    iter_loss_str = " → ".join([f"{l.item():.3f}" for l in iter_losses])
                    tqdm.write(f"Step {step:4d} | fine={avg_fine:.3f} coarse={avg_coarse:.3f} ratio={ratio:.3f} | iters: {iter_loss_str}")

                accum_loss = 0
                accum_fine = 0
                accum_coarse = 0
                accum_count = 0

                # Save checkpoint
                if step % 500 == 0:
                    ckpt_path = output_dir / f"step_{step:06d}.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    tqdm.write(f"  Saved checkpoint: {ckpt_path}")

        except Exception as e:
            continue

    pbar.close()

    # Save final checkpoint
    final_path = output_dir / "final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
    }, final_path)

    print("\n" + "=" * 70)
    print(f"COMPLETE! Trained for {step} steps")
    print(f"Output: {output_dir}")
    print("=" * 70)

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fine_iterations", type=int, default=3)
    parser.add_argument("--loss_mode", type=str, default="progressive",
                        choices=["final_only", "all_passes", "progressive"])
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/multi_fine")
    args = parser.parse_args()

    main(
        num_fine_iterations=args.num_fine_iterations,
        loss_mode=args.loss_mode,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        num_frames=args.num_frames,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
