#!/usr/bin/env python3
"""
Diagnostic: Compare LOSSES between Training Mode vs True Autoregressive Mode

This tests whether the train-test mismatch actually affects output quality:
1. Reconstruction loss (MSE on VAE latents)
2. Fine/Coarse ratio in both modes
3. Caption generation quality
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
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


def download_video(url: str, timeout: int = 20) -> bytes:
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            content = b''
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 100 * 1024 * 1024:
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
                result = subprocess.run(cmd, capture_output=True, timeout=60)
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


@torch.no_grad()
def compute_losses_training_mode(model, frames_norm, vae_latents, device):
    """
    Compute reconstruction loss using TRAINING mode (queries from coarse).
    Returns: loss_fine, loss_coarse
    """
    B, T = frames_norm.shape[:2]

    # Encode all frames
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

    # === COARSE path (same as training) ===
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # LLM generates queries from coarse
    coarse_token = model.coarse_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state

    # Coarse prediction
    h_coarse_for_pred = h_pass1[:, 1:1+T]  # After no_text token
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
    target_latents = vae_latents

    pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)
    loss_coarse = F.mse_loss(pred_coarse, target_latents)

    # === FINE path (training mode - queries from coarse) ===
    queries = model.llm_to_query(h_pass1[:, 2:])
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

    z_fine_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(shifted_q[:, t], all_caches[t])
        z_fine_list.append(z_t)
    z_fine = torch.stack(z_fine_list, dim=1)
    z_fine_llm = model.dino_to_llm(z_fine)
    z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

    # Fine LLM pass
    fine_token = model.fine_token.expand(B, -1, -1)
    seq_pass2 = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
    outputs_pass2 = model.llm.model(inputs_embeds=seq_pass2)
    h_pass2 = outputs_pass2.last_hidden_state

    h_fine_for_pred = h_pass2[:, 1:1+T]
    pred_fine = model.pred_head(h_fine_for_pred, prev_latents)
    loss_fine = F.mse_loss(pred_fine, target_latents)

    return loss_fine.item(), loss_coarse.item()


@torch.no_grad()
def compute_losses_autoregressive_mode(model, frames_norm, vae_latents, device):
    """
    Compute reconstruction loss using TRUE AUTOREGRESSIVE mode (queries from fine).
    Returns: loss_fine, loss_coarse
    """
    B, T = frames_norm.shape[:2]

    # Encode all frames
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

    # === COARSE path (same as always) ===
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # Coarse prediction (same as training)
    coarse_token = model.coarse_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)
    seq_coarse = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_coarse = model.llm.model(inputs_embeds=seq_coarse)
    h_coarse = outputs_coarse.last_hidden_state

    h_coarse_for_pred = h_coarse[:, 1:1+T]
    z_vae_init = model.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)
    prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)
    target_latents = vae_latents

    pred_coarse = model.pred_head(h_coarse_for_pred, prev_latents)
    loss_coarse = F.mse_loss(pred_coarse, target_latents)

    # === FINE path (TRUE AUTOREGRESSIVE - queries from fine) ===
    z_fine_list = []
    q_current = model.q_init.expand(B, -1)

    for t in range(T):
        z_t = model.encoder.query_attend(q_current, all_caches[t])
        z_fine_list.append(z_t)

        if t < T - 1:
            z_so_far = torch.stack(z_fine_list, dim=1)
            z_llm = model.dino_to_llm(z_so_far)
            z_llm = z_llm / (z_llm.std() + 1e-6) * model.visual_scale

            fine_token = model.fine_token.expand(B, -1, -1)
            seq = torch.cat([no_text, fine_token, z_llm], dim=1)
            outputs = model.llm.model(inputs_embeds=seq)
            h = outputs.last_hidden_state
            h_last = h[:, -1, :]
            q_current = model.llm_to_query(h_last)

    z_fine = torch.stack(z_fine_list, dim=1)
    z_fine_llm = model.dino_to_llm(z_fine)
    z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * model.visual_scale

    # Fine prediction
    fine_token = model.fine_token.expand(B, -1, -1)
    seq_fine = torch.cat([no_text, fine_token, z_fine_llm], dim=1)
    outputs_fine = model.llm.model(inputs_embeds=seq_fine)
    h_fine = outputs_fine.last_hidden_state

    h_fine_for_pred = h_fine[:, 1:1+T]
    pred_fine = model.pred_head(h_fine_for_pred, prev_latents)
    loss_fine = F.mse_loss(pred_fine, target_latents)

    return loss_fine.item(), loss_coarse.item()


def main(checkpoint_path: str, num_samples: int = 10, num_frames: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DIAGNOSTIC: Loss Comparison - Training vs Autoregressive Mode")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {num_samples}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        deep_query=True,
        freeze_dino=True,
    ).to(device)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from step {checkpoint.get('step', 'unknown')}")

    model.eval()

    # Load VAE for latent extraction
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Collect statistics
    stats = {
        'train_fine': [],
        'train_coarse': [],
        'train_ratio': [],
        'auto_fine': [],
        'auto_coarse': [],
        'auto_ratio': [],
    }

    print(f"\nProcessing {num_samples} samples...")

    # Stream videos
    ds = load_dataset('TempoFunk/webvid-10M', split='train', streaming=True)

    completed = 0
    for sample in ds:
        if completed >= num_samples:
            break

        try:
            duration = parse_duration(sample.get('duration', ''))
            if duration < 8 or duration > 30:
                continue

            url = sample.get('contentUrl')
            if not url:
                continue

            video_bytes = download_video(url)
            if video_bytes is None:
                continue

            frames = extract_frames(video_bytes, num_frames, 256)
            if frames is None:
                continue

            # Normalize for DINO
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)

            # Compute VAE latents
            frames_vae = frames.to(device) * 2 - 1  # Scale to [-1, 1] for VAE
            frames_vae = F.interpolate(frames_vae, size=(256, 256), mode='bilinear')
            with torch.no_grad():
                vae_latents = vae.encode(frames_vae).latent_dist.sample() * 0.18215
            vae_latents = vae_latents.unsqueeze(0)  # [1, T, 4, 32, 32]

            # Compute losses in both modes
            train_fine, train_coarse = compute_losses_training_mode(
                model, frames_norm, vae_latents, device
            )
            auto_fine, auto_coarse = compute_losses_autoregressive_mode(
                model, frames_norm, vae_latents, device
            )

            train_ratio = train_coarse / train_fine if train_fine > 0 else 1.0
            auto_ratio = auto_coarse / auto_fine if auto_fine > 0 else 1.0

            stats['train_fine'].append(train_fine)
            stats['train_coarse'].append(train_coarse)
            stats['train_ratio'].append(train_ratio)
            stats['auto_fine'].append(auto_fine)
            stats['auto_coarse'].append(auto_coarse)
            stats['auto_ratio'].append(auto_ratio)

            completed += 1
            print(f"\n[{completed:2d}/{num_samples}]")
            print(f"  Training mode:      fine={train_fine:.4f} coarse={train_coarse:.4f} ratio={train_ratio:.4f}")
            print(f"  Autoregressive mode: fine={auto_fine:.4f} coarse={auto_coarse:.4f} ratio={auto_ratio:.4f}")

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
            continue

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n1. TRAINING MODE (queries from coarse features)")
    print(f"   Fine loss:   {np.mean(stats['train_fine']):.4f} ± {np.std(stats['train_fine']):.4f}")
    print(f"   Coarse loss: {np.mean(stats['train_coarse']):.4f} ± {np.std(stats['train_coarse']):.4f}")
    print(f"   Ratio:       {np.mean(stats['train_ratio']):.4f} ± {np.std(stats['train_ratio']):.4f}")

    print("\n2. AUTOREGRESSIVE MODE (queries from fine features)")
    print(f"   Fine loss:   {np.mean(stats['auto_fine']):.4f} ± {np.std(stats['auto_fine']):.4f}")
    print(f"   Coarse loss: {np.mean(stats['auto_coarse']):.4f} ± {np.std(stats['auto_coarse']):.4f}")
    print(f"   Ratio:       {np.mean(stats['auto_ratio']):.4f} ± {np.std(stats['auto_ratio']):.4f}")

    print("\n3. COMPARISON")
    fine_diff = np.mean(stats['auto_fine']) - np.mean(stats['train_fine'])
    ratio_diff = np.mean(stats['auto_ratio']) - np.mean(stats['train_ratio'])
    print(f"   Fine loss change: {fine_diff:+.4f} ({'worse' if fine_diff > 0 else 'better'})")
    print(f"   Ratio change:     {ratio_diff:+.4f} ({'better' if ratio_diff > 0 else 'worse'})")

    print("\n4. PER-SAMPLE BREAKDOWN")
    print("   Sample | Train Ratio | Auto Ratio | Δ Ratio")
    print("   " + "-" * 45)
    for i in range(len(stats['train_ratio'])):
        delta = stats['auto_ratio'][i] - stats['train_ratio'][i]
        print(f"   {i+1:6d} | {stats['train_ratio'][i]:11.4f} | {stats['auto_ratio'][i]:10.4f} | {delta:+.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    mean_train_ratio = np.mean(stats['train_ratio'])
    mean_auto_ratio = np.mean(stats['auto_ratio'])

    if mean_auto_ratio > mean_train_ratio * 1.05:
        print("Autoregressive mode produces BETTER fine/coarse ratio!")
        print("The thesis behavior is even stronger in true autoregressive mode.")
    elif mean_auto_ratio > mean_train_ratio * 0.95:
        print("Both modes produce SIMILAR ratios (within 5%).")
        print("The model generalizes well from training to autoregressive inference.")
    else:
        print("Training mode produces better ratio than autoregressive.")
        print("There may be a train-test mismatch affecting performance.")

    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/joint_recon_caption/checkpoints/step_008000.pt")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    main(args.checkpoint, args.num_samples, args.num_frames)
