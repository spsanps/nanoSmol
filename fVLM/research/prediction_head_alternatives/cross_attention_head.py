"""
Cross-Attention Prediction Head (Experiment 2)

LLM hidden state queries over spatial positions of previous latent.
Allows LLM to selectively attend to different spatial regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionHead(nn.Module):
    """
    Cross-attention based prediction head.

    LLM hidden state (h) becomes query that attends over spatial positions
    of the previous latent. This allows position-specific information flow.
    """

    def __init__(
        self,
        h_dim: int = 576,
        latent_channels: int = 4,
        latent_size: int = 32,
        hidden_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        num_positions = latent_size * latent_size  # 1024

        # Encode previous latent to spatial tokens
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 3, 1, 1),
        )

        # Learnable position embeddings for spatial positions
        self.pos_embed = nn.Parameter(torch.randn(1, num_positions, hidden_dim) * 0.02)

        # Project h to queries (one query per output position)
        # h will be used to generate position-specific queries
        self.h_to_query_context = nn.Linear(h_dim, hidden_dim)

        # Position-specific query generation
        # Combines h context with position embedding to create queries
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Key and Value projections (from encoded latent)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decoder back to latent
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, 3, 1, 1),
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

        BT = h.shape[0]
        num_positions = self.latent_size * self.latent_size

        # Encode previous latent to spatial features
        feat = self.latent_encoder(z_vae_prev)  # [BT, hidden_dim, 32, 32]
        feat_flat = feat.flatten(2).transpose(1, 2)  # [BT, 1024, hidden_dim]

        # Add position embeddings to features
        feat_flat = feat_flat + self.pos_embed

        # Create position-specific queries using h
        h_context = self.h_to_query_context(h)  # [BT, hidden_dim]
        h_context = h_context.unsqueeze(1).expand(-1, num_positions, -1)  # [BT, 1024, hidden_dim]

        # Combine h context with position embeddings to create queries
        query_input = torch.cat([h_context, self.pos_embed.expand(BT, -1, -1)], dim=-1)  # [BT, 1024, hidden_dim*2]
        queries = self.query_mlp(query_input)  # [BT, 1024, hidden_dim]

        # Keys and values from encoded latent
        keys = self.key_proj(feat_flat)  # [BT, 1024, hidden_dim]
        values = self.value_proj(feat_flat)  # [BT, 1024, hidden_dim]

        # Multi-head attention
        # Reshape for multi-head: [BT, num_heads, num_positions, head_dim]
        queries = queries.view(BT, num_positions, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(BT, num_positions, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(BT, num_positions, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # [BT, num_heads, 1024, 1024]
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, values)  # [BT, num_heads, 1024, head_dim]
        out = out.transpose(1, 2).contiguous().view(BT, num_positions, self.hidden_dim)  # [BT, 1024, hidden_dim]

        # Output projection
        out = self.out_proj(out)  # [BT, 1024, hidden_dim]

        # Reshape to spatial and decode
        out = out.transpose(1, 2).view(BT, self.hidden_dim, self.latent_size, self.latent_size)
        pred = self.decoder(out)  # [BT, 4, 32, 32]

        if batched:
            pred = pred.reshape(B, T, *pred.shape[1:])

        return pred


if __name__ == "__main__":
    print("Testing CrossAttentionHead...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = CrossAttentionHead().to(device)

    # Test batched
    h = torch.randn(2, 8, 576).to(device)
    z_prev = torch.randn(2, 8, 4, 32, 32).to(device)
    pred = head(h, z_prev)

    print(f"Input h: {h.shape}")
    print(f"Input z_prev: {z_prev.shape}")
    print(f"Output pred: {pred.shape}")
    print(f"Parameters: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")
    print("Test passed!")
