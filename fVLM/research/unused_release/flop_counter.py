"""
Rough FLOP estimation for throughput reporting.

Provides order-of-magnitude FLOP counts for the foveated VLM forward pass.
Not exact -- ignores attention softmax, layer norms, biases, etc.  Good
enough for comparing configurations and tracking training throughput.

Also provides ``estimate_flops_from_config(cfg)`` to compute FLOPs from a
training YAML config dict, and ``compute_samples_for_budget()`` for
iso-FLOP scaling grid computations.
"""

# Known LLM configurations for SmolLM2 family
LLM_CONFIGS = {
    "SmolLM2-135M":  {"dim": 576,  "layers": 30, "params": 135_000_000},
    "SmolLM2-360M":  {"dim": 960,  "layers": 32, "params": 360_000_000},
    "SmolLM2-1.7B":  {"dim": 2048, "layers": 24, "params": 1_700_000_000},
}

DINO_CONFIGS = {
    "dinov2-small": {"dim": 384, "layers": 12, "patches": 256, "params": 22_000_000},
    "dinov2-base":  {"dim": 768, "layers": 12, "patches": 256, "params": 86_000_000},
}


def _resolve_llm(name: str) -> dict:
    """Match an LLM path/name to a known config."""
    for key, cfg in LLM_CONFIGS.items():
        if key.lower() in name.lower():
            return cfg
    return LLM_CONFIGS["SmolLM2-135M"]


def _resolve_dino(name: str) -> dict:
    """Match a DINO path/name to a known config."""
    for key, cfg in DINO_CONFIGS.items():
        if key.lower() in name.lower():
            return cfg
    return DINO_CONFIGS["dinov2-small"]


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


def estimate_flops_from_config(cfg: dict, avg_frames: int = 8, avg_seq_len: int = 128) -> int:
    """
    Estimate FLOPs per sample from a training config dict.

    Automatically resolves LLM/DINO dimensions from model paths.
    Handles both foveated (1 tok/frame, 2 passes) and multi-token
    (N tok/frame, 1 pass) architectures.

    Parameters
    ----------
    cfg : dict
        Training config (as loaded from YAML).
    avg_frames : int
        Average number of frames per sample.
    avg_seq_len : int
        Average text sequence length.

    Returns
    -------
    int
        Approximate FLOPs per sample (forward only; multiply by 3 for training).
    """
    llm_cfg = _resolve_llm(cfg.get("model", {}).get("llm", "SmolLM2-135M"))
    dino_cfg = _resolve_dino(cfg.get("model", {}).get("dino", "dinov2-small"))

    is_multi_token = cfg.get("model", {}).get("multi_token", False)
    tokens_per_frame = cfg.get("model", {}).get("tokens_per_frame", 16) if is_multi_token else 1
    # Multi-token uses 1 LLM pass (no coarse/fine); foveated uses 2
    num_passes = 1 if is_multi_token else 2

    if is_multi_token:
        # Multi-token: no query cross-attention, but LLM sees T*tokens_per_frame visual tokens
        N = dino_cfg["patches"] + 1
        dino_attn = (4 * N * dino_cfg["dim"]**2 + 2 * N**2 * dino_cfg["dim"]) * dino_cfg["layers"]
        dino_ffn = 2 * N * dino_cfg["dim"] * (4 * dino_cfg["dim"]) * dino_cfg["layers"]
        dino_total = (dino_attn + dino_ffn) * avg_frames

        proj_total = avg_frames * tokens_per_frame * dino_cfg["dim"] * llm_cfg["dim"]

        total_seq = avg_frames * tokens_per_frame + avg_seq_len
        llm_attn = (4 * total_seq * llm_cfg["dim"]**2 + 2 * total_seq**2 * llm_cfg["dim"]) * llm_cfg["layers"]
        llm_ffn = 2 * total_seq * llm_cfg["dim"] * (4 * llm_cfg["dim"]) * llm_cfg["layers"]

        return dino_total + proj_total + llm_attn + llm_ffn
    else:
        return estimate_flops_per_sample(
            num_frames=avg_frames,
            dino_dim=dino_cfg["dim"],
            dino_layers=dino_cfg["layers"],
            num_patches=dino_cfg["patches"],
            llm_dim=llm_cfg["dim"],
            llm_layers=llm_cfg["layers"],
            seq_len=avg_seq_len,
            num_passes=num_passes,
        )


def compute_samples_for_budget(
    flop_budget: float,
    cfg: dict,
    avg_frames: int = 8,
    avg_seq_len: int = 128,
) -> int:
    """
    Compute how many training samples fit in a given FLOP budget.

    Training FLOPs ≈ 3 × forward FLOPs × num_samples (forward + backward).

    Parameters
    ----------
    flop_budget : float
        Total FLOP budget (e.g. 1.6e16).
    cfg : dict
        Training config dict.
    avg_frames, avg_seq_len : int
        Average sample dimensions.

    Returns
    -------
    int
        Number of training samples that fit in the budget.
    """
    fwd_flops = estimate_flops_from_config(cfg, avg_frames, avg_seq_len)
    train_flops_per_sample = 3 * fwd_flops
    return max(1, int(flop_budget / train_flops_per_sample))
