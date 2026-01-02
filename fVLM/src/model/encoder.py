"""
Foveated Encoder: DINO + Query Mechanism

Implements query-guided attention via asymmetric masking in the vision encoder.
Supports two modes:
1. Shallow (default): Single cross-attention on final DINO features
2. Deep: Query propagates through all 12 DINO layers with KV caching

Key innovation: query can attend to patches, but patches cannot attend to query.
This enables KV caching for efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Tuple
import math


class FoveatedEncoder(nn.Module):
    """
    Vision encoder with query-guided attention mechanism.

    Architecture:
        1. DINO encodes image to patch tokens
        2. Query attends to patches (shallow or deep mode)
        3. Patches don't see query (enables KV caching)
    """

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-small",
        query_dim: int = 384,
        output_dim: int = 384,
        deep_query: bool = False,  # Set True for deep query injection
    ):
        """
        Args:
            dino_model_name: HuggingFace model ID for DINOv2
            query_dim: Dimension of query vector from LLM
            output_dim: Dimension of output foveated token
            deep_query: If True, use deep query injection through all layers
        """
        super().__init__()

        self.deep_query = deep_query

        # Load pretrained DINOv2 (will be fine-tuned)
        self.dino = AutoModel.from_pretrained(dino_model_name)
        self.dino_dim = self.dino.config.hidden_size  # 384 for ViT-S
        self.num_heads = self.dino.config.num_attention_heads  # 6 for ViT-S
        self.head_dim = self.dino_dim // self.num_heads  # 64

        # Projections for query
        self.query_input_proj = nn.Linear(query_dim, self.dino_dim)
        self.query_output_proj = nn.Linear(self.dino_dim, output_dim)

        # For device inference
        self.register_buffer("_dummy", torch.zeros(1))

    @property
    def device(self):
        return self._dummy.device

    def encode_patches(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Encode images to patch tokens.

        For shallow mode: runs full DINO forward
        For deep mode: caches K,V at each layer

        Args:
            images: [B, 3, H, W] input images (ImageNet normalized)

        Returns:
            patch_features: [B, N+1, D] final embeddings (CLS + patches)
            cache: Dict with cached activations for query pass
        """
        if self.deep_query:
            return self._encode_patches_deep(images)
        else:
            return self._encode_patches_shallow(images)

    def _encode_patches_shallow(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Shallow mode: run full DINO forward, cache final features."""
        # Get patch embeddings from DINO
        outputs = self.dino(images, output_hidden_states=True)
        patch_features = outputs.last_hidden_state  # [B, N+1, D] (includes CLS)

        # Keep CLS token - DINO was trained with it
        cache = {
            'patch_features': patch_features,  # [B, N+1, D]
        }

        return patch_features, cache

    def _encode_patches_deep(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Deep mode: cache K,V at each layer for query pass."""
        # Get patch embeddings through DINO's embedding layer
        embeddings = self.dino.embeddings(images)  # [B, N+1, D] includes CLS

        # Keep CLS token - DINO was trained with it
        hidden_states = embeddings  # [B, N+1, D]

        # Cache K, V at each layer for query pass
        kv_cache = []

        # Run through each DINO encoder layer
        for layer in self.dino.encoder.layer:
            # Pre-norm
            normed = layer.norm1(hidden_states)

            # Get K, V projections
            attn = layer.attention.attention
            K = attn.key(normed)    # [B, N+1, D]
            V = attn.value(normed)  # [B, N+1, D]

            # Cache K, V for query pass
            kv_cache.append({
                'K': K,
                'V': V,
                'layer': layer,
            })

            # Run full layer forward (returns tensor directly)
            hidden_states = layer(hidden_states)

        # Final layer norm
        patch_features = self.dino.layernorm(hidden_states)  # [B, N+1, D]

        cache = {
            'patch_features': patch_features,
            'kv_cache': kv_cache,
        }

        return patch_features, cache

    def query_attend(
        self,
        query: torch.Tensor,
        cache: dict
    ) -> torch.Tensor:
        """
        Use query to attend over cached patch features.

        Args:
            query: [B, D_q] query vector from LLM
            cache: Cached features from encode_patches

        Returns:
            z: [B, D_out] foveated visual token
        """
        if self.deep_query:
            return self._query_attend_deep(query, cache)
        else:
            return self._query_attend_shallow(query, cache)

    def _query_attend_shallow(self, query: torch.Tensor, cache: dict) -> torch.Tensor:
        """Shallow mode: single cross-attention on final features."""
        # Project query to DINO dimension
        q_embed = self.query_input_proj(query)  # [B, D_dino]

        # Get cached patch features (includes CLS)
        patch_features = cache['patch_features']  # [B, N+1, D_dino]

        # Cross-attention: query attends to all tokens (CLS + patches)
        q_embed = q_embed.unsqueeze(1)  # [B, 1, D]
        attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))  # [B, 1, N+1]
        attn_weights = torch.softmax(attn_scores / (self.dino_dim ** 0.5), dim=-1)

        # Attended features
        z = torch.bmm(attn_weights, patch_features)  # [B, 1, D]
        z = z.squeeze(1)  # [B, D]

        # Project to output dimension
        z = self.query_output_proj(z)  # [B, D_out]

        return z

    def _query_attend_deep(self, query: torch.Tensor, cache: dict) -> torch.Tensor:
        """Deep mode: query propagates through all layers with cached K,V."""
        kv_cache = cache['kv_cache']
        B = query.shape[0]

        # Project query to DINO dimension
        q_hidden = self.query_input_proj(query)  # [B, D]
        q_hidden = q_hidden.unsqueeze(1)  # [B, 1, D]

        # Propagate query through each layer
        for layer_cache in kv_cache:
            K = layer_cache['K']  # [B, N+1, D]
            V = layer_cache['V']  # [B, N+1, D]
            layer = layer_cache['layer']

            # Get attention module
            attn = layer.attention.attention

            # Pre-norm for query
            q_normed = layer.norm1(q_hidden)

            # Compute Q for query token
            Q = attn.query(q_normed)  # [B, 1, D]

            # Reshape for multi-head attention
            def reshape_for_heads(x):
                b, n, _ = x.shape
                return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

            Q = reshape_for_heads(Q)  # [B, num_heads, 1, head_dim]
            K_heads = reshape_for_heads(K)  # [B, num_heads, N+1, head_dim]
            V_heads = reshape_for_heads(V)  # [B, num_heads, N+1, head_dim]

            # Scaled dot-product attention
            scale = math.sqrt(self.head_dim)
            attn_scores = torch.matmul(Q, K_heads.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Weighted sum of values
            attn_output = torch.matmul(attn_weights, V_heads)  # [B, heads, 1, head_dim]

            # Reshape back: [B, heads, 1, head_dim] -> [B, 1, D]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(B, 1, self.dino_dim)

            # Output projection (part of attention)
            attn_output = layer.attention.output.dense(attn_output)
            attn_output = layer.attention.output.dropout(attn_output)

            # Layer scale and residual
            if hasattr(layer, 'layer_scale1'):
                attn_output = layer.layer_scale1(attn_output)

            q_hidden = q_hidden + attn_output

            # FFN block
            q_normed2 = layer.norm2(q_hidden)
            ffn_output = layer.mlp(q_normed2)

            if hasattr(layer, 'layer_scale2'):
                ffn_output = layer.layer_scale2(ffn_output)

            q_hidden = q_hidden + ffn_output

        # Final layer norm
        q_hidden = self.dino.layernorm(q_hidden)

        # Remove sequence dimension and project to output
        z = q_hidden.squeeze(1)  # [B, D]
        z = self.query_output_proj(z)  # [B, D_out]

        return z

    def forward(
        self,
        images: torch.Tensor,
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass: encode patches and attend with query.

        Args:
            images: [B, 3, H, W] input images
            query: [B, D_q] query vector

        Returns:
            z: [B, D_out] foveated visual token
        """
        _, cache = self.encode_patches(images)
        z = self.query_attend(query, cache)
        return z


if __name__ == "__main__":
    # Test FoveatedEncoder in both modes
    print("=" * 60)
    print("Testing FoveatedEncoder")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    for mode, deep in [("Shallow", False), ("Deep", True)]:
        print(f"\n--- Testing {mode} Mode ---")
        encoder = FoveatedEncoder(
            dino_model_name="facebook/dinov2-small",
            query_dim=384,
            output_dim=384,
            deep_query=deep,
        ).to(device)

        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256).to(device)
        query = torch.randn(batch_size, 384).to(device)

        patch_features, cache = encoder.encode_patches(images)
        print(f"   Patch features: {patch_features.shape}")

        z = encoder.query_attend(query, cache)
        print(f"   Output: {z.shape}")

        z.sum().backward()
        print(f"   Backward: OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
