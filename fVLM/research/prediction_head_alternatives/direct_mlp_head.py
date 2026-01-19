"""
Direct MLP Prediction Head (Experiment 1)

Replaces FiLM with a simple MLP that predicts next latent directly.
No spatial structure preservation - pure MLP.
"""

import torch
import torch.nn as nn


class DirectMLPHead(nn.Module):
    """
    Direct MLP prediction - no spatial modulation.

    Concatenates h and flattened prev_latent, then uses MLP to predict next latent.
    """

    def __init__(self, h_dim: int = 576, latent_channels: int = 4, latent_size: int = 32):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        latent_flat = latent_channels * latent_size * latent_size  # 4*32*32 = 4096

        # MLP: h + flattened latent -> predicted latent
        input_dim = h_dim + latent_flat  # 576 + 4096 = 4672

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, latent_flat),
        )

    def forward(self, h: torch.Tensor, z_vae_prev: torch.Tensor) -> torch.Tensor:
        """
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

        # Flatten previous latent
        z_flat = z_vae_prev.view(z_vae_prev.shape[0], -1)  # [B*T, 4096]

        # Concatenate h and z_flat
        combined = torch.cat([h, z_flat], dim=-1)  # [B*T, 4672]

        # MLP prediction
        pred_flat = self.mlp(combined)  # [B*T, 4096]

        # Reshape to latent shape
        pred = pred_flat.view(-1, self.latent_channels, self.latent_size, self.latent_size)

        if batched:
            pred = pred.reshape(B, T, *pred.shape[1:])

        return pred


if __name__ == "__main__":
    print("Testing DirectMLPHead...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = DirectMLPHead().to(device)

    # Test batched
    h = torch.randn(2, 8, 576).to(device)
    z_prev = torch.randn(2, 8, 4, 32, 32).to(device)
    pred = head(h, z_prev)

    print(f"Input h: {h.shape}")
    print(f"Input z_prev: {z_prev.shape}")
    print(f"Output pred: {pred.shape}")
    print(f"Parameters: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")
    print("Test passed!")
