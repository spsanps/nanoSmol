"""
Prediction Head with FiLM Conditioning

Predicts next-frame VAE latent given:
- h_t: LLM hidden state (semantic understanding)
- z_vae_t: Previous frame's VAE latent (spatial structure)

Uses FiLM to modulate transformation of previous latent.
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    FiLM-style prediction head.

    The LLM output (h) modulates how to transform the previous latent (z_vae_prev)
    into the next latent prediction.
    """

    def __init__(self, h_dim: int = 576, latent_channels: int = 4):
        """
        Args:
            h_dim: LLM hidden dimension
            latent_channels: VAE latent channels (4 for SD-VAE)
        """
        super().__init__()

        # FiLM parameters from h
        self.h_to_film = nn.Sequential(
            nn.Linear(h_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 2)  # gamma and beta
        )

        # Encoder for VAE latent
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 1, 1),
        )

        # Decoder back to latent
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, 3, 1, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        z_vae_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next latent using FiLM conditioning.

        Args:
            h: [B, T, H] or [B, H] LLM hidden states
            z_vae_prev: [B, T, C, H, W] or [B, C, H, W] previous latents

        Returns:
            pred: [B, T, C, H, W] or [B, C, H, W] predicted latents
        """
        # Handle batched time dimension
        if h.dim() == 3:
            B, T, D = h.shape
            h = h.reshape(B * T, D)
            z_vae_prev = z_vae_prev.reshape(B * T, *z_vae_prev.shape[2:])
            batched = True
        else:
            batched = False
            B = h.shape[0]

        # Get FiLM parameters
        film = self.h_to_film(h)  # [B*T, 512]
        gamma, beta = film.chunk(2, dim=-1)  # [B*T, 256] each

        # Encode previous latent
        feat = self.encoder(z_vae_prev)  # [B*T, 256, 32, 32]

        # Apply FiLM modulation
        gamma = gamma.view(-1, 256, 1, 1)
        beta = beta.view(-1, 256, 1, 1)
        feat = gamma * feat + beta

        # Decode to prediction
        pred = self.decoder(feat)  # [B*T, 4, 32, 32]

        if batched:
            pred = pred.reshape(B, T, *pred.shape[1:])

        return pred


if __name__ == "__main__":
    # Test PredictionHead
    print("=" * 60)
    print("Testing PredictionHead")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create prediction head
    pred_head = PredictionHead(h_dim=576, latent_channels=4).to(device)
    print(f"\n📦 PredictionHead created")
    print(f"   Parameters: {sum(p.numel() for p in pred_head.parameters()) / 1e6:.1f}M")

    # Test single sample
    print("\n🔄 Testing single sample...")
    h = torch.randn(2, 576).to(device)
    z_prev = torch.randn(2, 4, 32, 32).to(device)
    pred = pred_head(h, z_prev)
    print(f"   Input h shape: {h.shape}")
    print(f"   Input z_prev shape: {z_prev.shape}")
    print(f"   Output pred shape: {pred.shape}")

    # Test batched time
    print("\n🔄 Testing batched time...")
    h_batch = torch.randn(2, 8, 576).to(device)  # [B, T, H]
    z_batch = torch.randn(2, 8, 4, 32, 32).to(device)  # [B, T, C, H, W]
    pred_batch = pred_head(h_batch, z_batch)
    print(f"   Input h shape: {h_batch.shape}")
    print(f"   Input z_prev shape: {z_batch.shape}")
    print(f"   Output pred shape: {pred_batch.shape}")

    print("\n✓ PredictionHead test passed!")
    print("=" * 60)
