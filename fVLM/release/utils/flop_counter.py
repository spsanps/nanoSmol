"""
Rough FLOP estimation for throughput reporting.

Provides order-of-magnitude FLOP counts for the foveated VLM forward pass.
Not exact -- ignores attention softmax, layer norms, biases, etc.  Good
enough for comparing configurations and tracking training throughput.
"""


def estimate_flops_per_sample(
    num_frames: int = 8,
    dino_dim: int = 384,
    dino_layers: int = 12,
    num_patches: int = 256,
    llm_dim: int = 576,
    llm_layers: int = 30,
    seq_len: int = 128,
    num_passes: int = 2,
) -> int:
    """
    Estimate FLOPs for one sample through the foveated VLM.

    Parameters
    ----------
    num_frames : int
        Number of video frames.
    dino_dim : int
        Hidden dimension of DINOv2.
    dino_layers : int
        Number of DINOv2 transformer layers.
    num_patches : int
        Number of patch tokens per frame (256 for 224x224 with patch_size=14).
    llm_dim : int
        Hidden dimension of the LLM (576 for SmolLM2-135M).
    llm_layers : int
        Number of LLM transformer layers.
    seq_len : int
        Text sequence length.
    num_passes : int
        Number of LLM forward passes (2 for coarse+fine training).

    Returns
    -------
    int
        Approximate total FLOPs (multiply by 3 for backward pass estimate).
    """
    N = num_patches + 1  # +1 for CLS token

    # DINO encoder: self-attention + FFN per layer, per frame
    # Self-attention: 4 * N * D^2 (Q,K,V projections + output)
    #                 + 2 * N^2 * D (attention matmul)
    dino_attn = (4 * N * dino_dim**2 + 2 * N**2 * dino_dim) * dino_layers
    # FFN: 2 * N * D * 4D (two linear layers with 4x expansion)
    dino_ffn = 2 * N * dino_dim * (4 * dino_dim) * dino_layers
    dino_per_frame = dino_attn + dino_ffn
    dino_total = dino_per_frame * num_frames

    # Query cross-attention: 1 query token attending to N patches, per layer
    # Much cheaper: 4 * 1 * D^2 + 2 * 1 * N * D per layer
    query_per_frame = (4 * dino_dim**2 + 2 * N * dino_dim) * dino_layers
    # Two queries per frame in training (coarse + fine)
    query_total = query_per_frame * num_frames * num_passes

    # Projection: dino_dim -> llm_dim per frame
    proj_total = num_frames * dino_dim * llm_dim * num_passes

    # LLM: T visual tokens + S text tokens
    total_seq = num_frames + seq_len
    llm_attn = (4 * total_seq * llm_dim**2 + 2 * total_seq**2 * llm_dim) * llm_layers
    llm_ffn = 2 * total_seq * llm_dim * (4 * llm_dim) * llm_layers
    llm_total = (llm_attn + llm_ffn) * num_passes

    return dino_total + query_total + proj_total + llm_total
