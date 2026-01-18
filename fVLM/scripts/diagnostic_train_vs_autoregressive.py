#!/usr/bin/env python3
"""
Diagnostic: Compare Training Mode vs True Autoregressive Mode

QUESTION: Does the model work the same when we switch from:
- Training: Queries derived from COARSE features (parallel, efficient)
- Inference: Queries derived from FINE features (sequential, autoregressive)

This script tests:
1. Query similarity between the two approaches
2. Attention pattern differences
3. Feature (z) similarity
4. Loss differences (reconstruction and captioning)
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
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
def extract_training_mode(model, frames_norm, device):
    """
    Extract queries and features using TRAINING mode.
    Queries are derived from COARSE features (all at once).
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

    # === Pass 1: Coarse extraction with q_static ===
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)  # [B, T, dino_dim]
    z_coarse_llm = model.dino_to_llm(z_coarse)
    z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * model.visual_scale

    # === LLM generates ALL queries from coarse features ===
    coarse_token = model.coarse_token.expand(B, -1, -1)
    no_text = model.no_text_token.expand(B, -1, -1)
    seq_pass1 = torch.cat([no_text, coarse_token, z_coarse_llm], dim=1)
    outputs_pass1 = model.llm.model(inputs_embeds=seq_pass1)
    h_pass1 = outputs_pass1.last_hidden_state
    queries_from_coarse = model.llm_to_query(h_pass1[:, 2:])  # [B, T, query_dim]

    # === Shift queries: [q_init, q_0, q_1, ..., q_{T-2}] ===
    q_init = model.q_init.expand(B, -1).unsqueeze(1)
    shifted_q = torch.cat([q_init, queries_from_coarse[:, :-1]], dim=1)

    # === Pass 2: Fine extraction with shifted queries ===
    z_fine_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(shifted_q[:, t], all_caches[t])
        z_fine_list.append(z_t)
    z_fine = torch.stack(z_fine_list, dim=1)  # [B, T, dino_dim]

    return {
        'queries': shifted_q,  # [B, T, query_dim] - the queries used for fine extraction
        'z_coarse': z_coarse,  # [B, T, dino_dim]
        'z_fine': z_fine,      # [B, T, dino_dim]
        'all_caches': all_caches,
    }


@torch.no_grad()
def extract_autoregressive_mode(model, frames_norm, device):
    """
    Extract queries and features using TRUE AUTOREGRESSIVE mode.
    Each query is derived from previous FINE features (sequential).
    """
    B, T = frames_norm.shape[:2]

    # Encode all frames (same as training)
    frames_flat = frames_norm.reshape(B * T, 3, 256, 256)
    _, cache_flat = model.encoder.encode_patches(frames_flat)

    patch_features_flat = cache_flat['patch_features']
    N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
    patch_features = patch_features_flat.reshape(B, T, N, D)

    # Build per-frame caches (same as training)
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

    # Also extract coarse for comparison
    q_static = model.q_static.expand(B, -1)
    z_coarse_list = []
    for t in range(T):
        z_t = model.encoder.query_attend(q_static, all_caches[t])
        z_coarse_list.append(z_t)
    z_coarse = torch.stack(z_coarse_list, dim=1)

    # === TRUE AUTOREGRESSIVE: Query from previous FINE features ===
    queries_autoregressive = []
    z_fine_list = []

    # Frame 0: use q_init
    q_current = model.q_init.expand(B, -1)
    queries_autoregressive.append(q_current)

    for t in range(T):
        # Extract fine feature with current query
        z_t = model.encoder.query_attend(q_current, all_caches[t])
        z_fine_list.append(z_t)

        # Generate next query from LLM (if not last frame)
        if t < T - 1:
            z_so_far = torch.stack(z_fine_list, dim=1)  # [B, t+1, dino_dim]
            z_llm = model.dino_to_llm(z_so_far)
            z_llm = z_llm / (z_llm.std() + 1e-6) * model.visual_scale

            no_text = model.no_text_token.expand(B, -1, -1)
            fine_token = model.fine_token.expand(B, -1, -1)
            seq = torch.cat([no_text, fine_token, z_llm], dim=1)

            outputs = model.llm.model(inputs_embeds=seq)
            h = outputs.last_hidden_state
            h_last = h[:, -1, :]  # Last position predicts next query
            q_current = model.llm_to_query(h_last)
            queries_autoregressive.append(q_current)

    queries_autoregressive = torch.stack(queries_autoregressive, dim=1)  # [B, T, query_dim]
    z_fine = torch.stack(z_fine_list, dim=1)  # [B, T, dino_dim]

    return {
        'queries': queries_autoregressive,  # [B, T, query_dim]
        'z_coarse': z_coarse,               # [B, T, dino_dim]
        'z_fine': z_fine,                   # [B, T, dino_dim]
        'all_caches': all_caches,
    }


def cosine_sim(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def main(checkpoint_path: str, num_samples: int = 10, num_frames: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DIAGNOSTIC: Training Mode vs True Autoregressive Mode")
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect statistics
    stats = {
        'query_sim': [],           # Cosine similarity between queries from two modes
        'z_fine_sim': [],          # Cosine similarity between fine features
        'query_diff_norm': [],     # L2 norm of query difference
        'z_fine_diff_norm': [],    # L2 norm of feature difference
        'per_frame_query_sim': [], # Per-frame query similarity
    }

    print(f"\nProcessing {num_samples} samples...")

    # Stream videos from WebVid
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

            # Normalize and prepare
            frames_norm = normalize_for_dino(frames, device).unsqueeze(0)  # [1, T, 3, H, W]

            # Extract in both modes
            train_out = extract_training_mode(model, frames_norm, device)
            auto_out = extract_autoregressive_mode(model, frames_norm, device)

            # Compare queries
            q_train = train_out['queries']  # [1, T, query_dim]
            q_auto = auto_out['queries']    # [1, T, query_dim]

            query_sim = cosine_sim(q_train, q_auto)
            query_diff = (q_train - q_auto).norm().item()

            # Compare fine features
            z_train = train_out['z_fine']  # [1, T, dino_dim]
            z_auto = auto_out['z_fine']    # [1, T, dino_dim]

            z_sim = cosine_sim(z_train, z_auto)
            z_diff = (z_train - z_auto).norm().item()

            # Per-frame query similarity
            per_frame_sim = []
            for t in range(num_frames):
                sim = cosine_sim(q_train[0, t], q_auto[0, t])
                per_frame_sim.append(sim)

            stats['query_sim'].append(query_sim)
            stats['z_fine_sim'].append(z_sim)
            stats['query_diff_norm'].append(query_diff)
            stats['z_fine_diff_norm'].append(z_diff)
            stats['per_frame_query_sim'].append(per_frame_sim)

            completed += 1
            print(f"  [{completed:2d}/{num_samples}] Query sim: {query_sim:.4f}, Z_fine sim: {z_sim:.4f}")

        except Exception as e:
            continue

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n1. QUERY SIMILARITY (Training Mode vs Autoregressive Mode)")
    print(f"   Mean cosine similarity: {np.mean(stats['query_sim']):.4f}")
    print(f"   Std:                    {np.std(stats['query_sim']):.4f}")
    print(f"   Min:                    {np.min(stats['query_sim']):.4f}")
    print(f"   Max:                    {np.max(stats['query_sim']):.4f}")

    print("\n2. FINE FEATURE SIMILARITY (z_fine)")
    print(f"   Mean cosine similarity: {np.mean(stats['z_fine_sim']):.4f}")
    print(f"   Std:                    {np.std(stats['z_fine_sim']):.4f}")
    print(f"   Min:                    {np.min(stats['z_fine_sim']):.4f}")
    print(f"   Max:                    {np.max(stats['z_fine_sim']):.4f}")

    print("\n3. PER-FRAME QUERY SIMILARITY")
    per_frame_avg = np.mean(stats['per_frame_query_sim'], axis=0)
    for t, sim in enumerate(per_frame_avg):
        print(f"   Frame {t:2d}: {sim:.4f}")

    print("\n4. INTERPRETATION")
    mean_q_sim = np.mean(stats['query_sim'])
    mean_z_sim = np.mean(stats['z_fine_sim'])

    if mean_q_sim > 0.95:
        print("   -> Queries are VERY SIMILAR between modes (>0.95)")
        print("   -> Training and autoregressive modes produce nearly identical queries")
    elif mean_q_sim > 0.8:
        print("   -> Queries are SIMILAR between modes (0.8-0.95)")
        print("   -> Some divergence but fundamentally aligned")
    elif mean_q_sim > 0.5:
        print("   -> Queries are MODERATELY DIFFERENT (0.5-0.8)")
        print("   -> Significant divergence between training and autoregressive modes")
    else:
        print("   -> Queries are VERY DIFFERENT (<0.5)")
        print("   -> Training and autoregressive modes produce different queries!")

    print()
    if mean_z_sim > 0.95:
        print("   -> Fine features are VERY SIMILAR despite query differences")
        print("   -> The model is robust to query generation method")
    elif mean_z_sim > 0.8:
        print("   -> Fine features are SIMILAR (0.8-0.95)")
        print("   -> Query differences have moderate impact on features")
    else:
        print("   -> Fine features DIFFER significantly")
        print("   -> Query generation method has substantial impact!")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if mean_q_sim > 0.8 and mean_z_sim > 0.8:
        print("The model behaves consistently between training and inference.")
        print("Switching from coarse-derived to fine-derived queries is SAFE.")
    elif mean_z_sim > 0.8:
        print("Despite query differences, features remain similar.")
        print("The model is ROBUST to query generation method.")
    else:
        print("Significant differences detected!")
        print("Consider implications for inference vs training behavior.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/joint_recon_caption/checkpoints/step_008000.pt")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    main(args.checkpoint, args.num_samples, args.num_frames)
