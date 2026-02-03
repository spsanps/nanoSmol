#!/usr/bin/env python3
"""
Estimate FLOPs comparison between Foveated VLM and Baseline VLM.

Compares computational cost for:
- Vision encoder (DINOv2-small)
- Connector (query attention vs PixelShuffle+MLP)
- LLM forward pass (SmolLM2-135M)
"""

import math

# ============================================================================
# Model dimensions
# ============================================================================

# DINOv2-small
DINO_DIM = 384
DINO_HEADS = 6
DINO_LAYERS = 12
DINO_MLP_RATIO = 4
PATCH_SIZE = 14

# SmolLM2-135M
LLM_DIM = 576
LLM_HEADS = 9
LLM_LAYERS = 30
LLM_MLP_RATIO = 4  # Approximate
LLM_VOCAB = 49152

# Input specs
NUM_FRAMES = 8
FRAME_SIZE_FOVEATED = 256  # Foveated uses 256x256
FRAME_SIZE_BASELINE = 224  # Baseline uses 224x224
CAPTION_TOKENS = 32  # Average caption length


def flops_linear(in_dim, out_dim, batch=1):
    """FLOPs for linear layer: 2 * in * out (multiply-add)"""
    return 2 * batch * in_dim * out_dim


def flops_attention(seq_len, dim, num_heads):
    """FLOPs for multi-head attention.

    QKV projection: 3 * 2 * seq * dim * dim
    Attention scores: 2 * seq * seq * dim
    Attention @ V: 2 * seq * seq * dim
    Output projection: 2 * seq * dim * dim
    """
    qkv = 3 * flops_linear(dim, dim, seq_len)
    scores = 2 * seq_len * seq_len * dim
    attn_v = 2 * seq_len * seq_len * dim
    out_proj = flops_linear(dim, dim, seq_len)
    return qkv + scores + attn_v + out_proj


def flops_mlp(seq_len, dim, mlp_ratio=4):
    """FLOPs for MLP block (2 linear layers with expansion)."""
    hidden = int(dim * mlp_ratio)
    up = flops_linear(dim, hidden, seq_len)
    down = flops_linear(hidden, dim, seq_len)
    return up + down


def flops_transformer_layer(seq_len, dim, num_heads, mlp_ratio=4):
    """FLOPs for one transformer layer."""
    attn = flops_attention(seq_len, dim, num_heads)
    mlp = flops_mlp(seq_len, dim, mlp_ratio)
    # LayerNorm: negligible compared to attention/MLP
    return attn + mlp


def flops_dino_encoder(frame_size):
    """FLOPs for DINOv2-small encoding one frame."""
    num_patches = (frame_size // PATCH_SIZE) ** 2
    seq_len = num_patches + 1  # +1 for CLS token

    # Patch embedding (conv)
    patch_embed = 2 * 3 * PATCH_SIZE * PATCH_SIZE * DINO_DIM * num_patches

    # Transformer layers
    transformer = sum(
        flops_transformer_layer(seq_len, DINO_DIM, DINO_HEADS, DINO_MLP_RATIO)
        for _ in range(DINO_LAYERS)
    )

    return patch_embed + transformer


def flops_query_attention(num_patches, dim, num_layers=12):
    """FLOPs for foveated query attention (cross-attention).

    Query: 1 token attending to num_patches tokens.
    Done at multiple DINO layers for deep_query=True.
    """
    # For each layer: Q projection (1 token), K/V projection (num_patches tokens)
    # Attention: 1 query attending to num_patches keys
    total = 0
    for _ in range(num_layers):
        q_proj = flops_linear(dim, dim, 1)
        kv_proj = 2 * flops_linear(dim, dim, num_patches)
        # Attention scores: 1 * num_patches * dim (for dot product)
        scores = 2 * 1 * num_patches * dim
        # Weighted sum: 1 * num_patches * dim
        weighted = 2 * 1 * num_patches * dim
        out_proj = flops_linear(dim, dim, 1)
        total += q_proj + kv_proj + scores + weighted + out_proj
    return total


def flops_pixelshuffle_mlp(num_input_patches, input_dim, output_dim, scale=4):
    """FLOPs for PixelShuffle + MLP connector.

    PixelShuffle is just reshape (0 FLOPs).
    MLP: 2 linear layers with expansion.
    """
    num_output_patches = num_input_patches // (scale * scale)
    shuffle_dim = input_dim * scale * scale  # After pixel shuffle

    # 2-layer MLP with SiLU
    hidden_dim = output_dim * 4
    up = flops_linear(shuffle_dim, hidden_dim, num_output_patches)
    down = flops_linear(hidden_dim, output_dim, num_output_patches)

    return up + down


def flops_llm_forward(seq_len):
    """FLOPs for SmolLM2-135M forward pass."""
    # Embedding lookup: negligible

    # Transformer layers
    transformer = sum(
        flops_transformer_layer(seq_len, LLM_DIM, LLM_HEADS, LLM_MLP_RATIO)
        for _ in range(LLM_LAYERS)
    )

    # LM head
    lm_head = flops_linear(LLM_DIM, LLM_VOCAB, seq_len)

    return transformer + lm_head


def estimate_foveated_flops():
    """Total FLOPs for Foveated VLM (1 token/frame)."""

    # DINOv2 encoder: all frames
    num_patches_256 = (FRAME_SIZE_FOVEATED // PATCH_SIZE) ** 2  # 18*18 = 324
    dino_flops = NUM_FRAMES * flops_dino_encoder(FRAME_SIZE_FOVEATED)

    # Query attention (coarse + fine iterations)
    # Coarse: 1 pass with q_static
    # Fine: 2 iterations with learned queries
    query_flops = NUM_FRAMES * 3 * flops_query_attention(num_patches_256, DINO_DIM)

    # Projection layers (dino_to_llm, llm_to_query)
    proj_flops = NUM_FRAMES * 2 * flops_linear(DINO_DIM, LLM_DIM, 1)

    # LLM forward passes:
    # - Query generation pass: 1 + 8 tokens (mode + coarse visual)
    # - Fine iteration 1: 1 + 8 tokens
    # - Fine iteration 2: 1 + 8 tokens
    # - Caption pass: 1 + 8 + 32 tokens (mode + visual + caption)
    visual_tokens = NUM_FRAMES * 1  # 1 token per frame
    llm_query_pass = flops_llm_forward(1 + visual_tokens)  # Coarse query gen
    llm_fine_pass1 = flops_llm_forward(1 + visual_tokens)  # Fine iter 1
    llm_fine_pass2 = flops_llm_forward(1 + visual_tokens)  # Fine iter 2
    llm_caption_pass = flops_llm_forward(1 + visual_tokens + CAPTION_TOKENS)

    llm_flops = llm_query_pass + llm_fine_pass1 + llm_fine_pass2 + llm_caption_pass

    return {
        'dino': dino_flops,
        'connector': query_flops + proj_flops,
        'llm': llm_flops,
        'total': dino_flops + query_flops + proj_flops + llm_flops,
    }


def estimate_baseline_flops():
    """Total FLOPs for Baseline VLM (16 tokens/frame)."""

    # DINOv2 encoder: all frames at 224x224
    num_patches_224 = (FRAME_SIZE_BASELINE // PATCH_SIZE) ** 2  # 16*16 = 256
    dino_flops = NUM_FRAMES * flops_dino_encoder(FRAME_SIZE_BASELINE)

    # PixelShuffle + MLP connector
    connector_flops = NUM_FRAMES * flops_pixelshuffle_mlp(
        num_patches_224, DINO_DIM, LLM_DIM, scale=4
    )

    # LLM forward pass: 1 + 128 + 32 tokens (mode + visual + caption)
    visual_tokens = NUM_FRAMES * 16  # 16 tokens per frame after PixelShuffle 4x
    llm_flops = flops_llm_forward(1 + visual_tokens + CAPTION_TOKENS)

    return {
        'dino': dino_flops,
        'connector': connector_flops,
        'llm': llm_flops,
        'total': dino_flops + connector_flops + llm_flops,
    }


def format_flops(flops):
    """Format FLOPs in human-readable form."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def estimate_training_flops():
    """Estimate training FLOPs (forward + backward)."""
    fov_inf = estimate_foveated_flops()
    base_inf = estimate_baseline_flops()

    # Backward pass is roughly 2x forward
    BACKWARD_MULTIPLIER = 2

    # Foveated training adds:
    # - Reconstruction prediction head (FiLM-conditioned)
    # - VAE latent comparison (negligible)
    # Prediction head: MLP that predicts 4*32*32 = 4096 dim latent
    LATENT_DIM = 4 * 32 * 32  # VAE latent size
    pred_head_flops = NUM_FRAMES * (
        flops_linear(LLM_DIM, LLM_DIM * 2, 1) +  # FiLM modulation
        flops_linear(LLM_DIM, LATENT_DIM, 1)     # Prediction
    )

    fov_train = {
        'forward': fov_inf['total'],
        'backward': fov_inf['total'] * BACKWARD_MULTIPLIER,
        'prediction_head': pred_head_flops * 3,  # forward + backward
        'total': fov_inf['total'] * (1 + BACKWARD_MULTIPLIER) + pred_head_flops * 3,
    }

    base_train = {
        'forward': base_inf['total'],
        'backward': base_inf['total'] * BACKWARD_MULTIPLIER,
        'prediction_head': 0,
        'total': base_inf['total'] * (1 + BACKWARD_MULTIPLIER),
    }

    return fov_train, base_train, fov_inf, base_inf


def main():
    print("=" * 70)
    print("FLOPs Comparison: Foveated VLM vs Baseline VLM")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Frames per video: {NUM_FRAMES}")
    print(f"  Caption tokens: {CAPTION_TOKENS}")
    print(f"  Foveated frame size: {FRAME_SIZE_FOVEATED}x{FRAME_SIZE_FOVEATED}")
    print(f"  Baseline frame size: {FRAME_SIZE_BASELINE}x{FRAME_SIZE_BASELINE}")

    fov = estimate_foveated_flops()
    base = estimate_baseline_flops()

    print(f"\n{'Component':<20} {'Foveated':>15} {'Baseline':>15} {'Ratio':>10}")
    print("-" * 60)

    for key in ['dino', 'connector', 'llm', 'total']:
        f_val = fov[key]
        b_val = base[key]
        ratio = f_val / b_val if b_val > 0 else 0
        print(f"{key.upper():<20} {format_flops(f_val):>15} {format_flops(b_val):>15} {ratio:>9.2f}x")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_ratio = fov['total'] / base['total']
    llm_ratio = fov['llm'] / base['llm']

    print(f"\nTotal FLOPs:")
    print(f"  Foveated: {format_flops(fov['total'])}")
    print(f"  Baseline: {format_flops(base['total'])}")
    print(f"  Foveated/Baseline ratio: {total_ratio:.2f}x")

    print(f"\nLLM FLOPs (main difference):")
    print(f"  Foveated: {format_flops(fov['llm'])} ({fov['llm']/fov['total']*100:.1f}% of total)")
    print(f"  Baseline: {format_flops(base['llm'])} ({base['llm']/base['total']*100:.1f}% of total)")
    print(f"  Foveated/Baseline ratio: {llm_ratio:.2f}x")

    print(f"\nSequence lengths (LLM):")
    fov_seq = 1 + NUM_FRAMES * 1 + CAPTION_TOKENS  # Final caption pass
    base_seq = 1 + NUM_FRAMES * 16 + CAPTION_TOKENS
    print(f"  Foveated caption pass: {fov_seq} tokens")
    print(f"  Baseline caption pass: {base_seq} tokens")
    print(f"  Baseline has {base_seq/fov_seq:.1f}x longer sequence")

    # Attention FLOPs scale quadratically
    attn_ratio = (base_seq ** 2) / (fov_seq ** 2)
    print(f"  Attention FLOPs ratio: {attn_ratio:.1f}x (quadratic in seq len)")

    print(f"\nQuality vs Efficiency:")
    print(f"  Loss difference: +0.067 nats (+1.7%)")
    print(f"  FLOPs savings: {(1 - total_ratio) * 100:.1f}%")
    print(f"  LLM FLOPs savings: {(1 - llm_ratio) * 100:.1f}%")

    # Quality per FLOP
    # Baseline: 3.98 loss at X FLOPs
    # Foveated: 4.05 loss at Y FLOPs
    fov_loss = 4.0478
    base_loss = 3.9810

    # "Quality" = blind_loss - model_loss (visual contribution)
    blind_loss = 6.0513
    fov_quality = blind_loss - fov_loss
    base_quality = blind_loss - base_loss

    fov_quality_per_gflop = fov_quality / (fov['total'] / 1e9)
    base_quality_per_gflop = base_quality / (base['total'] / 1e9)

    print(f"\nEfficiency metric (visual contribution per GFLOPs):")
    print(f"  Foveated: {fov_quality_per_gflop:.4f}")
    print(f"  Baseline: {base_quality_per_gflop:.4f}")
    print(f"  Foveated is {fov_quality_per_gflop/base_quality_per_gflop:.2f}x more efficient")

    # Training comparison
    fov_train, base_train, _, _ = estimate_training_flops()

    print("\n" + "=" * 70)
    print("TRAINING FLOPs (forward + backward)")
    print("=" * 70)

    print(f"\n{'Component':<20} {'Foveated':>15} {'Baseline':>15}")
    print("-" * 50)
    print(f"{'Forward':<20} {format_flops(fov_train['forward']):>15} {format_flops(base_train['forward']):>15}")
    print(f"{'Backward (~2x)':<20} {format_flops(fov_train['backward']):>15} {format_flops(base_train['backward']):>15}")
    print(f"{'Prediction head':<20} {format_flops(fov_train['prediction_head']):>15} {format_flops(base_train['prediction_head']):>15}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {format_flops(fov_train['total']):>15} {format_flops(base_train['total']):>15}")

    train_ratio = fov_train['total'] / base_train['total']
    print(f"\nTraining FLOPs ratio: {train_ratio:.2f}x (foveated/baseline)")

    # Scaling analysis
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: More frames")
    print("=" * 70)

    for nf in [8, 16, 32, 64]:
        fov_tokens = 1 + nf * 1 + CAPTION_TOKENS
        base_tokens = 1 + nf * 16 + CAPTION_TOKENS

        # LLM attention scales O(n^2)
        fov_attn = fov_tokens ** 2
        base_attn = base_tokens ** 2

        print(f"\n{nf} frames:")
        print(f"  Foveated: {fov_tokens} tokens, attention ∝ {fov_attn:,}")
        print(f"  Baseline: {base_tokens} tokens, attention ∝ {base_attn:,}")
        print(f"  LLM attention ratio: {base_attn/fov_attn:.1f}x more for baseline")


if __name__ == "__main__":
    main()
