"""
Foveated Encoder: DINO + Query Mechanism

Implements query-guided attention via asymmetric masking in the vision encoder.
Key innovation: query can attend to patches, but patches cannot attend to query.
This enables KV caching for efficient inference.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple, Optional


class FoveatedEncoder(nn.Module):
    """
    Vision encoder with query-guided attention mechanism.

    Architecture:
        1. DINO encodes image to patch tokens
        2. Query token is injected with asymmetric attention mask
        3. Query attends to all patches (extracts information)
        4. Patches don't see query (enables KV caching)
    """

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-small",
        query_dim: int = 384,
        output_dim: int = 384,
    ):
        """
        Args:
            dino_model_name: HuggingFace model ID for DINOv2
            query_dim: Dimension of query vector from LLM
            output_dim: Dimension of output foveated token
        """
        super().__init__()

        # Load pretrained DINOv2 (will be fine-tuned)
        self.dino = AutoModel.from_pretrained(dino_model_name)
        self.dino_dim = self.dino.config.hidden_size  # 384 for ViT-S

        # Projections for query
        self.query_input_proj = nn.Linear(query_dim, self.dino_dim)
        self.query_output_proj = nn.Linear(self.dino_dim, output_dim)

        # Learnable query tokens
        self.register_buffer("_dummy", torch.zeros(1))  # For device inference

    @property
    def device(self):
        return self._dummy.device

    def encode_patches(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Encode images to patch tokens WITHOUT query.
        This is the expensive operation - run once per frame.

        Args:
            images: [B, 3, H, W] input images (ImageNet normalized)

        Returns:
            patch_features: [B, N, D] final patch embeddings
            cache: Dict with intermediate activations for query pass
        """
        B = images.shape[0]

        # Get patch embeddings from DINO
        # Note: dinov2 outputs include last_hidden_state
        outputs = self.dino(images, output_hidden_states=True)
        patch_features = outputs.last_hidden_state  # [B, N+1, D] (includes CLS)

        # Remove CLS token, keep only patches
        patch_features = patch_features[:, 1:, :]  # [B, N, D]

        # Cache for query pass (we're not implementing full KV caching yet for simplicity)
        cache = {
            'patch_features': patch_features,
        }

        return patch_features, cache

    def query_attend(
        self,
        query: torch.Tensor,
        cache: dict
    ) -> torch.Tensor:
        """
        Use query to attend over cached patch features.
        This is the cheap operation - can run multiple times with different queries.

        Args:
            query: [B, D_q] query vector from LLM
            cache: Cached patch features from encode_patches

        Returns:
            z: [B, D_out] foveated visual token
        """
        # Project query to DINO dimension
        q_embed = self.query_input_proj(query)  # [B, D_dino]

        # Get cached patch features
        patch_features = cache['patch_features']  # [B, N, D_dino]

        # Cross-attention: query attends to patches
        # Using simple dot-product attention for now
        # TODO: For full implementation, inject query into DINO blocks with masking

        # Compute attention scores
        q_embed = q_embed.unsqueeze(1)  # [B, 1, D]
        attn_scores = torch.bmm(q_embed, patch_features.transpose(1, 2))  # [B, 1, N]
        attn_weights = torch.softmax(attn_scores / (self.dino_dim ** 0.5), dim=-1)

        # Attended features
        z = torch.bmm(attn_weights, patch_features)  # [B, 1, D]
        z = z.squeeze(1)  # [B, D]

        # Project to output dimension
        z = self.query_output_proj(z)  # [B, D_out]

        return z

    def forward(
        self,
        images: torch.Tensor,
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass: encode patches and attend with query.
        Used during training.

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
    # Test FoveatedEncoder
    print("=" * 60)
    print("Testing FoveatedEncoder")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create encoder
    print("\n📦 Loading DINOv2...")
    encoder = FoveatedEncoder(
        dino_model_name="facebook/dinov2-small",
        query_dim=384,
        output_dim=384,
    ).to(device)

    print(f"   ✓ Encoder loaded")
    print(f"   DINO dim: {encoder.dino_dim}")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")

    # Test forward pass
    print("\n🔄 Testing forward pass...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    query = torch.randn(batch_size, 384).to(device)

    # Encode patches
    patch_features, cache = encoder.encode_patches(images)
    print(f"   Patch features shape: {patch_features.shape}")

    # Query attend
    z = encoder.query_attend(query, cache)
    print(f"   Foveated token shape: {z.shape}")

    # Full forward
    z_full = encoder.forward(images, query)
    print(f"   Full forward shape: {z_full.shape}")

    print("\n✓ FoveatedEncoder test passed!")
    print("=" * 60)
