"""
Attention heatmap saver + entropy computation for foveated VLM.

Saves raw attention weights from the encoder cross-attention for fixed
eval videos. Used for paper figures (heatmap visualisation) and the
attention_entropy metric logged every eval step.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch


def compute_attention_entropy(attn_weights_list: List[torch.Tensor]) -> float:
    """
    Compute mean attention entropy across all layers and heads.

    Parameters
    ----------
    attn_weights_list : list of [B, H, 1, N+1] tensors
        One tensor per DINO layer, from encoder.query_attend(return_attention=True).

    Returns
    -------
    float
        Mean entropy in nats, averaged over batch, heads, and layers.
        Lower entropy = sharper (more selective) attention.
    """
    total_entropy = 0.0
    count = 0
    for w in attn_weights_list:
        # w: [B, H, 1, N+1] -> squeeze query dim -> [B, H, N+1]
        p = w.squeeze(2).float()
        # Shannon entropy: -sum(p * log(p)), clamped to avoid log(0)
        log_p = torch.log(p.clamp(min=1e-10))
        entropy = -(p * log_p).sum(dim=-1)  # [B, H]
        total_entropy += entropy.mean().item()
        count += 1
    return total_entropy / max(count, 1)


def save_attention_maps(
    attn_weights_list: List[torch.Tensor],
    save_dir: str,
    step: int,
    sample_idx: int = 0,
    frame_idx: int = 0,
    prefix: str = "attn",
):
    """
    Save raw attention weights to disk for later visualisation.

    Saves a .pt file containing the full attention weight stack and a
    .txt summary with per-layer entropy and top-5 patch indices.

    Parameters
    ----------
    attn_weights_list : list of [B, H, 1, N+1] tensors
        Attention weights from encoder.query_attend(return_attention=True).
    save_dir : str
        Directory to write files.
    step : int
        Training step (for naming).
    sample_idx : int
        Which sample in the batch to save (default 0).
    frame_idx : int
        Which frame this attention map belongs to.
    prefix : str
        Filename prefix.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Stack all layers: [L, H, N+1]
    stacked = torch.stack(
        [w[sample_idx, :, 0, :].cpu() for w in attn_weights_list],
        dim=0,
    )  # [L, H, N+1]

    fname = f"{prefix}_step{step:07d}_frame{frame_idx:03d}"
    torch.save(stacked, save_dir / f"{fname}.pt")

    # Summary text
    with open(save_dir / f"{fname}.txt", "w") as f:
        f.write(f"step={step}, frame={frame_idx}\n")
        f.write(f"shape: {list(stacked.shape)} (layers, heads, patches+CLS)\n\n")
        for li in range(stacked.shape[0]):
            # Average across heads for this layer
            avg = stacked[li].mean(dim=0)  # [N+1]
            entropy = -(avg * torch.log(avg.clamp(min=1e-10))).sum().item()
            top5 = avg.topk(5)
            f.write(
                f"Layer {li:2d}: entropy={entropy:.3f}, "
                f"top5_patches={top5.indices.tolist()}, "
                f"top5_weights={[f'{v:.4f}' for v in top5.values.tolist()]}\n"
            )


def extract_attention_for_eval(
    model,
    frames: torch.Tensor,
    save_dir: str,
    step: int,
    num_frames_to_save: int = 4,
) -> float:
    """
    Run the foveated encoder on frames and extract attention maps + entropy.

    This performs the coarse pass (q_static) and saves attention maps for the
    first few frames. Returns the mean attention entropy.

    Parameters
    ----------
    model : FoveatedVLM
        The (unwrapped) model.
    frames : [B, T, 3, H, W]
        Video frames.
    save_dir : str
        Directory to save attention map files.
    step : int
        Current training step.
    num_frames_to_save : int
        Number of frames to save attention maps for.

    Returns
    -------
    float
        Mean attention entropy across saved frames.
    """
    B, T = frames.shape[:2]
    device = frames.device

    # Encode all frames (DINO patches + KV cache)
    per_frame_caches = model.encoder._encode_all_frames_raw(frames) \
        if hasattr(model.encoder, '_encode_all_frames_raw') \
        else model._encode_all_frames(frames)

    q_static = model.q_static.expand(B, -1)  # [B, qd]

    all_entropies = []
    frames_to_save = min(T, num_frames_to_save)

    for t in range(frames_to_save):
        z, attn_weights = model.encoder.query_attend(
            q_static, per_frame_caches[t], return_attention=True,
        )
        entropy = compute_attention_entropy(attn_weights)
        all_entropies.append(entropy)

        # Save raw maps for first sample in batch
        save_attention_maps(
            attn_weights, save_dir, step,
            sample_idx=0, frame_idx=t,
        )

    # For remaining frames, just compute entropy (no save)
    for t in range(frames_to_save, T):
        z, attn_weights = model.encoder.query_attend(
            q_static, per_frame_caches[t], return_attention=True,
        )
        all_entropies.append(compute_attention_entropy(attn_weights))

    return sum(all_entropies) / max(len(all_entropies), 1)
