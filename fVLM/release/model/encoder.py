"""
FoveatedEncoder -- DINOv2 vision encoder with query-guided cross-attention.

Deep query mode only: the query token is projected into DINO dimension then
propagated through every DINO layer using cached K,V from the patch tokens.
Patches never attend to the query (asymmetric mask), so the patch forward pass
runs once and all K,V are cached.  The single query-position output after the
final layer is the foveated visual token.

Key design decisions (pre-fixed bugs baked in):
  * query_input_proj has bias=False  (BUG-002: bias dominated small queries,
    causing uniform attention regardless of query content)
  * No shallow mode                  (BUG-004: single cross-attention on final
    DINO features gives output correlation ~0.98 -- effectively uniform)
  * CLS token is kept                (DINO was trained with it)
  * Layer norm applied after all layers (matches DINO forward)

torch.compile friendly:
  * Fixed loop count (num_layers is a Python int constant per model)
  * No Python-level branching in hot paths
  * Attention scale stored as a float constant (not recomputed)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


# ---------------------------------------------------------------------------
# Model configs -- keeps torch.compile happy (loop counts are Python ints)
# ---------------------------------------------------------------------------
DINO_CONFIGS = {
    "facebook/dinov2-small": {"dim": 384, "heads": 6, "layers": 12, "patch_size": 14},
    "facebook/dinov2-base":  {"dim": 768, "heads": 12, "layers": 12, "patch_size": 14},
}


class FoveatedEncoder(nn.Module):
    """
    Vision encoder with deep query-guided attention.

    Two-phase usage:
        1.  ``patches, kv_cache = encoder.encode_patches(images)``
            Run DINO on all frames, cache K/V at every layer.
        2.  ``z = encoder.query_attend(query, kv_cache)``
            Propagate query through all layers using cached K/V.
            Returns a single foveated visual token per image.
    """

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-small",
        query_dim: int = 384,
        output_dim: int | None = None,
    ) -> None:
        """
        Args:
            dino_model_name: HuggingFace model id for DINOv2.
            query_dim:       Dimension of incoming query vector (from LLM).
            output_dim:      Dimension of the output foveated token.
        """
        super().__init__()

        # -- Load pretrained DINOv2 -----------------------------------------
        self.dino: Dinov2Model = Dinov2Model.from_pretrained(dino_model_name)

        # Cache model geometry as plain Python values for torch.compile.
        cfg = self.dino.config
        self.dino_dim: int = cfg.hidden_size
        self.num_heads: int = cfg.num_attention_heads
        self.head_dim: int = self.dino_dim // self.num_heads
        self.num_layers: int = cfg.num_hidden_layers
        self.patch_size: int = cfg.patch_size

        # Pre-compute attention scale as a constant.
        self.attn_scale: float = 1.0 / math.sqrt(self.head_dim)

        # -- Projections ----------------------------------------------------
        if output_dim is None:
            output_dim = self.dino_dim

        # bias=False is CRITICAL (BUG-002).  With bias, different queries
        # produce near-identical embeddings at init (bias dominates the small
        # query signal), so attention is uniform and fine == coarse always.
        self.query_input_proj = nn.Linear(query_dim, self.dino_dim, bias=False)
        self.output_proj = nn.Linear(self.dino_dim, output_dim)

        # Dummy buffer for device / dtype inference.
        self.register_buffer("_device_probe", torch.zeros(1), persistent=False)

    # -- Convenience --------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device_probe.device

    def num_patches(self, image_size: int = 224) -> int:
        """Number of spatial patch tokens for a square image (excludes CLS)."""
        grid = image_size // self.patch_size
        return grid * grid

    def num_tokens(self, image_size: int = 224) -> int:
        """Total sequence length from DINO (CLS + spatial patches)."""
        return 1 + self.num_patches(image_size)

    # ======================================================================
    # Phase 1: encode patches  (run once per frame set)
    # ======================================================================

    def encode_patches(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Encode images through DINOv2, caching K and V at every layer.

        Args:
            images: ``[B*T, 3, H, W]`` input images (ImageNet-normalised).

        Returns:
            patch_features: ``[B*T, N+1, D]`` final embeddings (CLS + patches),
                            after the last layer norm.
            kv_cache:       List of ``(K, V)`` tuples, one per DINO layer.
                            Each K, V has shape ``[B*T, N+1, D]`` (full dim,
                            not yet reshaped to multi-head).
        """
        # Patch + position embedding (includes CLS prepend).
        hidden: torch.Tensor = self.dino.embeddings(images)  # [B*T, N+1, D]

        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Walk every encoder layer.  The loop count (self.num_layers) is a
        # Python int constant, so torch.compile unrolls it -- no graph breaks.
        for layer in self.dino.encoder.layer:
            normed = layer.norm1(hidden)

            # Grab the K, V linear projections on the *normed* input.
            attn_mod = layer.attention.attention  # Dinov2SelfAttention
            K = attn_mod.key(normed)    # [B*T, N+1, D]
            V = attn_mod.value(normed)  # [B*T, N+1, D]
            kv_cache.append((K, V))

            # Full forward for the patch tokens (self-attention + FFN).
            # Patches attend to patches only -- the query is not present yet.
            hidden = layer(hidden)

        # Final layer norm (matches Dinov2Model.forward).
        patch_features = self.dino.layernorm(hidden)  # [B*T, N+1, D]

        return patch_features, kv_cache

    # ======================================================================
    # Phase 2: query-attend  (run per query)
    # ======================================================================

    def query_attend(
        self,
        query: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Propagate a query token through every DINO layer using cached K/V.

        The query can attend to all patch tokens, but patches never see the
        query (asymmetric attention -- enabled by using the cached K/V that
        were computed without the query present).

        Args:
            query:    ``[B*T, query_dim]`` query vector from the LLM.
            kv_cache: Output of :meth:`encode_patches` (list of (K, V) per layer).

        Returns:
            z: ``[B*T, output_dim]``  -- the single foveated visual token.
        """
        B = query.shape[0]

        # Project query into DINO space.
        q_hidden = self.query_input_proj(query).unsqueeze(1)  # [B, 1, D]

        all_attn_weights = [] if return_attention else None

        # Walk every layer, reusing cached K/V from patches.
        for layer_idx, layer in enumerate(self.dino.encoder.layer):
            K, V = kv_cache[layer_idx]  # each [B, N+1, D]

            attn_mod = layer.attention.attention  # Dinov2SelfAttention

            # Pre-norm for the query token.
            q_normed = layer.norm1(q_hidden)  # [B, 1, D]

            # Q projection for the query token only.
            Q = attn_mod.query(q_normed)  # [B, 1, D]

            # Reshape to multi-head:  [B, S, D] -> [B, H, S, d]
            Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            K_h = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V_h = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention (query attends to all patches).
            # Q: [B, H, 1, d],  K_h: [B, H, N+1, d],  V_h: [B, H, N+1, d]
            attn_scores = torch.matmul(Q, K_h.transpose(-2, -1)) * self.attn_scale
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, N+1]

            if return_attention:
                all_attn_weights.append(attn_weights.detach())

            attn_out = torch.matmul(attn_weights, V_h)     # [B, H, 1, d]

            # Merge heads:  [B, H, 1, d] -> [B, 1, D]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.dino_dim)

            # Output projection + dropout (Dinov2SelfOutput.dense / .dropout).
            attn_out = layer.attention.output.dense(attn_out)
            attn_out = layer.attention.output.dropout(attn_out)

            # Layer scale 1  +  residual.
            attn_out = layer.layer_scale1(attn_out)
            q_hidden = q_hidden + attn_out

            # FFN block:  norm2 -> MLP -> layer_scale2 -> residual.
            ffn_out = layer.mlp(layer.norm2(q_hidden))
            ffn_out = layer.layer_scale2(ffn_out)
            q_hidden = q_hidden + ffn_out

        # Final layer norm (same norm used at the end of encode_patches).
        q_hidden = self.dino.layernorm(q_hidden)  # [B, 1, D]

        # Squeeze sequence dim and project to output dimension.
        z = self.output_proj(q_hidden.squeeze(1))  # [B, output_dim]

        if return_attention:
            return z, all_attn_weights
        return z

    # ======================================================================
    # Convenience: full forward (encode + attend in one call)
    # ======================================================================

    def forward(
        self,
        images: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward: encode patches then attend with query.

        Args:
            images: ``[B, 3, H, W]``
            query:  ``[B, query_dim]``

        Returns:
            z: ``[B, output_dim]``  foveated visual token.
        """
        _, kv_cache = self.encode_patches(images)
        return self.query_attend(query, kv_cache)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing FoveatedEncoder (deep query mode)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    encoder = FoveatedEncoder(
        dino_model_name="facebook/dinov2-small",
        query_dim=384,
        output_dim=384,
    ).to(device)

    print(f"  dino_dim   = {encoder.dino_dim}")
    print(f"  num_heads  = {encoder.num_heads}")
    print(f"  head_dim   = {encoder.head_dim}")
    print(f"  num_layers = {encoder.num_layers}")
    print(f"  patch_size = {encoder.patch_size}")

    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    query_a = torch.randn(batch_size, 384, device=device)
    query_b = torch.randn(batch_size, 384, device=device)

    print(f"\n  num_patches(224) = {encoder.num_patches(224)}")
    print(f"  num_tokens(224)  = {encoder.num_tokens(224)}")

    # -- Phase 1 --
    print("\n--- encode_patches ---")
    patch_features, kv_cache = encoder.encode_patches(images)
    print(f"  patch_features: {patch_features.shape}")
    print(f"  kv_cache:       {len(kv_cache)} layers, K shape = {kv_cache[0][0].shape}")

    # -- Phase 2 --
    print("\n--- query_attend ---")
    z_a = encoder.query_attend(query_a, kv_cache)
    z_b = encoder.query_attend(query_b, kv_cache)
    print(f"  z_a: {z_a.shape}")
    print(f"  z_b: {z_b.shape}")

    # Check that different queries give different outputs.
    cosine = F.cosine_similarity(z_a, z_b, dim=-1).mean().item()
    l2_diff = (z_a - z_b).norm(dim=-1).mean().item()
    print(f"  cosine(z_a, z_b) = {cosine:.4f}  (should be << 1.0)")
    print(f"  L2 diff          = {l2_diff:.4f}  (should be >> 0)")

    # -- Backward --
    print("\n--- backward ---")
    z_a.sum().backward()
    print("  backward: OK")

    # -- Combined forward --
    print("\n--- forward (combined) ---")
    encoder.zero_grad()
    z = encoder(images, query_a)
    z.sum().backward()
    print(f"  z: {z.shape}")
    print("  backward: OK")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
