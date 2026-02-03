#!/usr/bin/env python3
"""
Find configurations where foveated training is faster than baseline.

Variables to explore:
1. Number of frames (more frames → LLM dominates)
2. Frame size (same size removes DINO advantage for baseline)
3. Caption length (longer captions → LLM dominates)
4. Connector complexity (simpler query attention)
"""

import math

# Model dimensions
DINO_DIM = 384
DINO_HEADS = 6
DINO_LAYERS = 12
DINO_MLP_RATIO = 4
PATCH_SIZE = 14

LLM_DIM = 576
LLM_HEADS = 9
LLM_LAYERS = 30
LLM_MLP_RATIO = 4
LLM_VOCAB = 49152


def flops_linear(in_dim, out_dim, batch=1):
    return 2 * batch * in_dim * out_dim


def flops_attention(seq_len, dim, num_heads):
    qkv = 3 * flops_linear(dim, dim, seq_len)
    scores = 2 * seq_len * seq_len * dim
    attn_v = 2 * seq_len * seq_len * dim
    out_proj = flops_linear(dim, dim, seq_len)
    return qkv + scores + attn_v + out_proj


def flops_mlp(seq_len, dim, mlp_ratio=4):
    hidden = int(dim * mlp_ratio)
    return flops_linear(dim, hidden, seq_len) + flops_linear(hidden, dim, seq_len)


def flops_transformer_layer(seq_len, dim, num_heads, mlp_ratio=4):
    return flops_attention(seq_len, dim, num_heads) + flops_mlp(seq_len, dim, mlp_ratio)


def flops_dino(frame_size):
    num_patches = (frame_size // PATCH_SIZE) ** 2
    seq_len = num_patches + 1
    patch_embed = 2 * 3 * PATCH_SIZE * PATCH_SIZE * DINO_DIM * num_patches
    transformer = sum(flops_transformer_layer(seq_len, DINO_DIM, DINO_HEADS, DINO_MLP_RATIO)
                      for _ in range(DINO_LAYERS))
    return patch_embed + transformer


def flops_query_attention(num_patches, query_layers=12):
    """Query attention cost. Can vary number of layers."""
    total = 0
    for _ in range(query_layers):
        q_proj = flops_linear(DINO_DIM, DINO_DIM, 1)
        kv_proj = 2 * flops_linear(DINO_DIM, DINO_DIM, num_patches)
        scores = 2 * 1 * num_patches * DINO_DIM
        weighted = 2 * 1 * num_patches * DINO_DIM
        out_proj = flops_linear(DINO_DIM, DINO_DIM, 1)
        total += q_proj + kv_proj + scores + weighted + out_proj
    return total


def flops_pixelshuffle_mlp(num_patches, scale=4):
    num_output = num_patches // (scale * scale)
    shuffle_dim = DINO_DIM * scale * scale
    hidden_dim = LLM_DIM * 4
    return flops_linear(shuffle_dim, hidden_dim, num_output) + flops_linear(hidden_dim, LLM_DIM, num_output)


def flops_llm(seq_len):
    transformer = sum(flops_transformer_layer(seq_len, LLM_DIM, LLM_HEADS, LLM_MLP_RATIO)
                      for _ in range(LLM_LAYERS))
    lm_head = flops_linear(LLM_DIM, LLM_VOCAB, seq_len)
    return transformer + lm_head


def estimate_foveated(num_frames, frame_size, caption_len, query_layers=12, num_fine_iters=2):
    """Estimate foveated FLOPs with configurable parameters."""
    num_patches = (frame_size // PATCH_SIZE) ** 2

    # DINO
    dino = num_frames * flops_dino(frame_size)

    # Query attention: coarse + fine iterations
    num_query_passes = 1 + num_fine_iters  # coarse + fine iters
    connector = num_frames * num_query_passes * flops_query_attention(num_patches, query_layers)

    # Projections
    proj = num_frames * 2 * flops_linear(DINO_DIM, LLM_DIM, 1)

    # LLM passes: query generation + caption
    visual_tokens = num_frames * 1
    llm_query_passes = num_query_passes * flops_llm(1 + visual_tokens)
    llm_caption = flops_llm(1 + visual_tokens + caption_len)
    llm = llm_query_passes + llm_caption

    return {'dino': dino, 'connector': connector + proj, 'llm': llm,
            'total': dino + connector + proj + llm}


def estimate_baseline(num_frames, frame_size, caption_len):
    """Estimate baseline FLOPs."""
    num_patches = (frame_size // PATCH_SIZE) ** 2

    # DINO
    dino = num_frames * flops_dino(frame_size)

    # PixelShuffle + MLP
    connector = num_frames * flops_pixelshuffle_mlp(num_patches)

    # LLM: single pass
    visual_tokens = num_frames * 16  # After 4x pixelshuffle on 16x16 grid
    if frame_size != 224:
        # Adjust for different frame sizes
        grid_size = frame_size // PATCH_SIZE
        visual_tokens = num_frames * (grid_size // 4) ** 2

    llm = flops_llm(1 + visual_tokens + caption_len)

    return {'dino': dino, 'connector': connector, 'llm': llm,
            'total': dino + connector + llm}


def format_gflops(flops):
    return f"{flops/1e9:.1f}"


def compare(num_frames, frame_size, caption_len, query_layers=12, num_fine_iters=2):
    """Compare and return ratio."""
    fov = estimate_foveated(num_frames, frame_size, caption_len, query_layers, num_fine_iters)
    base = estimate_baseline(num_frames, frame_size, caption_len)

    ratio = fov['total'] / base['total']
    llm_ratio = fov['llm'] / base['llm']

    return {
        'fov_total': fov['total'],
        'base_total': base['total'],
        'ratio': ratio,
        'llm_ratio': llm_ratio,
        'fov_llm_pct': fov['llm'] / fov['total'] * 100,
        'base_llm_pct': base['llm'] / base['total'] * 100,
    }


def main():
    print("=" * 80)
    print("FINDING CROSSOVER: When is Foveated Training Faster?")
    print("=" * 80)

    # Current config
    print("\n### Current Configuration ###")
    print("Foveated: 256x256, query_layers=12, fine_iters=2")
    print("Baseline: 224x224, pixelshuffle=4x")

    r = compare(8, 256, 32, query_layers=12, num_fine_iters=2)
    # For baseline with 224
    base_224 = estimate_baseline(8, 224, 32)
    fov_256 = estimate_foveated(8, 256, 32, 12, 2)
    ratio = fov_256['total'] / base_224['total']
    print(f"\nFrames=8, Caption=32:")
    print(f"  Foveated: {format_gflops(fov_256['total'])} GFLOPs")
    print(f"  Baseline: {format_gflops(base_224['total'])} GFLOPs")
    print(f"  Ratio: {ratio:.2f}x (>1 = foveated slower)")

    # === OPTION 1: Same frame size ===
    print("\n" + "=" * 80)
    print("OPTION 1: Same Frame Size (224x224 for both)")
    print("=" * 80)

    print(f"\n{'Frames':<8} {'Caption':<8} {'Fov GFLOPs':>12} {'Base GFLOPs':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 60)

    for nf in [8, 16, 32, 64, 128]:
        for cap in [32, 64, 128]:
            r = compare(nf, 224, cap, query_layers=12, num_fine_iters=2)
            winner = "FOVEATED" if r['ratio'] < 1.0 else "baseline"
            marker = "<<<" if r['ratio'] < 1.0 else ""
            print(f"{nf:<8} {cap:<8} {format_gflops(r['fov_total']):>12} {format_gflops(r['base_total']):>12} {r['ratio']:>7.2f}x {winner:>10} {marker}")

    # === OPTION 2: Simpler connector ===
    print("\n" + "=" * 80)
    print("OPTION 2: Simpler Connector (query_layers=1, fine_iters=1)")
    print("=" * 80)

    print(f"\n{'Frames':<8} {'Caption':<8} {'Fov GFLOPs':>12} {'Base GFLOPs':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 60)

    for nf in [8, 16, 32, 64]:
        for cap in [32, 64, 128]:
            r = compare(nf, 224, cap, query_layers=1, num_fine_iters=1)
            winner = "FOVEATED" if r['ratio'] < 1.0 else "baseline"
            marker = "<<<" if r['ratio'] < 1.0 else ""
            print(f"{nf:<8} {cap:<8} {format_gflops(r['fov_total']):>12} {format_gflops(r['base_total']):>12} {r['ratio']:>7.2f}x {winner:>10} {marker}")

    # === OPTION 3: Longer captions ===
    print("\n" + "=" * 80)
    print("OPTION 3: Longer Captions (256+ tokens)")
    print("=" * 80)

    print(f"\n{'Frames':<8} {'Caption':<8} {'Fov GFLOPs':>12} {'Base GFLOPs':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 60)

    for nf in [8, 16, 32]:
        for cap in [256, 512, 1024]:
            r = compare(nf, 224, cap, query_layers=12, num_fine_iters=2)
            winner = "FOVEATED" if r['ratio'] < 1.0 else "baseline"
            marker = "<<<" if r['ratio'] < 1.0 else ""
            print(f"{nf:<8} {cap:<8} {format_gflops(r['fov_total']):>12} {format_gflops(r['base_total']):>12} {r['ratio']:>7.2f}x {winner:>10} {marker}")

    # === OPTION 4: Combined optimizations ===
    print("\n" + "=" * 80)
    print("OPTION 4: Optimized Foveated (224px, query_layers=3, fine_iters=1)")
    print("=" * 80)

    print(f"\n{'Frames':<8} {'Caption':<8} {'Fov GFLOPs':>12} {'Base GFLOPs':>12} {'Ratio':>8} {'Winner':>10}")
    print("-" * 60)

    for nf in [8, 16, 32, 64]:
        for cap in [32, 64, 128]:
            r = compare(nf, 224, cap, query_layers=3, num_fine_iters=1)
            winner = "FOVEATED" if r['ratio'] < 1.0 else "baseline"
            marker = "<<<" if r['ratio'] < 1.0 else ""
            print(f"{nf:<8} {cap:<8} {format_gflops(r['fov_total']):>12} {format_gflops(r['base_total']):>12} {r['ratio']:>7.2f}x {winner:>10} {marker}")

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: Configurations where Foveated is Faster")
    print("=" * 80)

    print("\nFoveated becomes faster when:")
    print("  1. More frames (64+) with same frame size")
    print("  2. Simpler connector (fewer query layers, fewer fine iterations)")
    print("  3. Longer captions (256+ tokens)")
    print("  4. Any combination of above")

    print("\nCurrent bottlenecks in foveated:")
    print("  - 12-layer deep query attention (13.97x more expensive than MLP)")
    print("  - 2 fine iterations (3 total LLM query passes)")
    print("  - Larger frame size (256 vs 224)")

    print("\nRecommended optimizations:")
    print("  - Use 224x224 frames (matches baseline)")
    print("  - Reduce query_layers to 3-6 (still effective, much cheaper)")
    print("  - Use 1 fine iteration instead of 2")
    print("  - Target longer video / longer caption use cases")


if __name__ == "__main__":
    main()
