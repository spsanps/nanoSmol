"""
Foveated Vision-Language Model

Main model implementing two-pass parallel architecture:
- Pass 1: Query planning with static query (coarse)
- Pass 2: Focused extraction with dynamic queries (fine)

Both passes predict next-frame VAE latents for supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Tuple
import sys
from pathlib import Path

# Handle both package and standalone imports
try:
    from .encoder import FoveatedEncoder
    from .prediction import PredictionHead
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from encoder import FoveatedEncoder
    from prediction import PredictionHead


class FoveatedVideoModel(nn.Module):
    """
    Two-pass foveated VLM for video understanding.

    Architecture:
        1. Static query extracts coarse features from all frames
        2. LLM processes coarse features, predicts where to look
        3. Dynamic queries extract focused features
        4. LLM processes focused features, predicts next frame latents
    """

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
        lambda_coarse: float = 1.0,
    ):
        """
        Args:
            dino_model: HuggingFace model ID for vision encoder
            llm_model: HuggingFace model ID for LLM
            dino_dim: DINO embedding dimension
            llm_dim: LLM hidden dimension
            query_dim: Query vector dimension
            lambda_coarse: Weight for auxiliary coarse loss
        """
        super().__init__()

        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.query_dim = query_dim
        self.lambda_coarse = lambda_coarse

        # Vision encoder
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_model,
            query_dim=query_dim,
            output_dim=dino_dim,
        )

        # Core LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm.config.use_cache = False  # Disable KV cache during training

        # Projections
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.llm_to_query = nn.Linear(llm_dim, query_dim)

        # Prediction head (shared between passes)
        self.pred_head = PredictionHead(h_dim=llm_dim, latent_channels=4)

        # Learned parameters (careful initialization)
        self.q_static = nn.Parameter(torch.randn(1, query_dim) * 0.02)
        self.q_init = nn.Parameter(torch.randn(1, query_dim) * 0.02)
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Mode tokens (to signal coarse vs fine pass)
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim))
        self.fine_token = nn.Parameter(torch.randn(1, 1, llm_dim))

    def get_empty_text_embeds(self, batch_size: int) -> torch.Tensor:
        """
        Get empty text embeddings for Phase 1 (self-supervised).

        Returns a single dummy token per sample.
        """
        device = self.q_static.device
        # Single dummy embedding per sample
        return torch.zeros(batch_size, 1, self.llm_dim, device=device)

    def forward(
        self,
        text_embeds: torch.Tensor,
        raw_frames: torch.Tensor,
        vae_latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-pass forward with auxiliary loss.

        Args:
            text_embeds: [B, N_text, llm_dim] pre-embedded text tokens
            raw_frames: [B, T, 3, 256, 256] video frames (ImageNet normalized)
            vae_latents: [B, T, 4, 32, 32] precomputed VAE latents (targets)

        Returns:
            loss: combined loss
            loss_fine: Pass 2 reconstruction loss (main)
            loss_coarse: Pass 1 reconstruction loss (auxiliary)
        """
        B, T = raw_frames.shape[:2]
        N_text = text_embeds.shape[1]

        # === Encode all frames with DINO, cache features ===
        # Batch process all frames at once for efficiency
        frames_flat = raw_frames.reshape(B * T, 3, 256, 256)  # [B*T, 3, 256, 256]
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        # Reshape cache back to [B, T, N, D]
        patch_features_flat = cache_flat['patch_features']  # [B*T, N, D]
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)  # [B, T, N, D]

        # Create per-frame caches for query_attend
        all_caches = []
        for t in range(T):
            all_caches.append({'patch_features': patch_features[:, t]})  # [B, N, D]

        # === Pass 1: Query Planning with q_static ===
        q_static = self.q_static.expand(B, -1)  # [B, query_dim]

        z_coarse_list = []
        for t in range(T):
            z_t = self.encoder.query_attend(q_static, all_caches[t])
            z_coarse_list.append(z_t)
        z_coarse = torch.stack(z_coarse_list, dim=1)  # [B, T, dino_dim]
        z_coarse = self.dino_to_llm(z_coarse)  # [B, T, llm_dim]

        # Build Pass 1 sequence: [text, <coarse>, z°_1, ..., z°_T]
        coarse_token = self.coarse_token.expand(B, -1, -1)
        seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)

        # LLM forward (causal)
        outputs_pass1 = self.llm.model(inputs_embeds=seq_pass1)
        h_pass1 = outputs_pass1.last_hidden_state  # [B, N_text + 1 + T, llm_dim]

        # Extract query predictions from positions after <coarse> token
        h_for_queries = h_pass1[:, N_text + 1:]  # [B, T, llm_dim]
        queries = self.llm_to_query(h_for_queries)  # [B, T, query_dim]

        # === Auxiliary loss on Pass 1 ===
        h_coarse_for_pred = h_pass1[:, N_text:N_text + T]  # [B, T, llm_dim]

        # Conditioning latents: [z_vae_init, z_vae_1, ..., z_vae_{T-1}]
        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)  # [B, 1, 4, 32, 32]
        prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)  # [B, T, 4, 32, 32]

        # Targets: [z_vae_1, z_vae_2, ..., z_vae_T]
        target_latents = vae_latents  # [B, T, 4, 32, 32]

        # Coarse prediction (shared head)
        pred_coarse = self.pred_head(h_coarse_for_pred, prev_latents)
        loss_coarse = F.mse_loss(pred_coarse, target_latents)

        # === Shift queries: q_t used for frame_{t+1} ===
        q_init = self.q_init.expand(B, -1).unsqueeze(1)  # [B, 1, query_dim]
        shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)  # [B, T, query_dim]

        # === Pass 2: Focused Extraction with dynamic queries ===
        z_focused_list = []
        for t in range(T):
            z_t = self.encoder.query_attend(shifted_q[:, t], all_caches[t])
            z_focused_list.append(z_t)
        z_focused = torch.stack(z_focused_list, dim=1)  # [B, T, dino_dim]
        z_focused = self.dino_to_llm(z_focused)  # [B, T, llm_dim]

        # Build Pass 2 sequence: [text, <fine>, z_1, ..., z_T]
        fine_token = self.fine_token.expand(B, -1, -1)
        seq_pass2 = torch.cat([text_embeds, fine_token, z_focused], dim=1)

        # LLM forward (causal)
        outputs_pass2 = self.llm.model(inputs_embeds=seq_pass2)
        h_pass2 = outputs_pass2.last_hidden_state  # [B, N_text + 1 + T, llm_dim]

        # === Main loss on Pass 2 ===
        h_fine_for_pred = h_pass2[:, N_text:N_text + T]  # [B, T, llm_dim]

        # Fine prediction (same shared head)
        pred_fine = self.pred_head(h_fine_for_pred, prev_latents)
        loss_fine = F.mse_loss(pred_fine, target_latents)

        # === Combined loss ===
        loss = loss_fine + self.lambda_coarse * loss_coarse

        return loss, loss_fine, loss_coarse


if __name__ == "__main__":
    # Test FoveatedVideoModel
    print("=" * 70)
    print("Testing FoveatedVideoModel")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create model
    print("\n📦 Loading model components...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_dim=384,
        llm_dim=576,
        query_dim=384,
        lambda_coarse=1.0,
    ).to(device)

    print(f"   ✓ Model loaded")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   Total parameters: {total_params:.1f}M")
    print(f"   Trainable parameters: {trainable_params:.1f}M")

    # Test forward pass
    print("\n🔄 Testing forward pass...")
    batch_size = 2
    num_frames = 4  # Small for testing

    text_embeds = model.get_empty_text_embeds(batch_size).to(device)
    raw_frames = torch.randn(batch_size, num_frames, 3, 256, 256).to(device)
    vae_latents = torch.randn(batch_size, num_frames, 4, 32, 32).to(device)

    print(f"   Input shapes:")
    print(f"     text_embeds: {text_embeds.shape}")
    print(f"     raw_frames: {raw_frames.shape}")
    print(f"     vae_latents: {vae_latents.shape}")

    # Forward pass
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, loss_fine, loss_coarse = model(text_embeds, raw_frames, vae_latents)

    print(f"\n   Output:")
    print(f"     loss: {loss.item():.4f}")
    print(f"     loss_fine: {loss_fine.item():.4f}")
    print(f"     loss_coarse: {loss_coarse.item():.4f}")
    print(f"     ratio (coarse/fine): {loss_coarse.item() / loss_fine.item():.3f}")

    # Test backward pass
    print("\n🔄 Testing backward pass...")
    loss.backward()
    print(f"   ✓ Gradients computed successfully")

    # Check gradient flow to q_static
    if model.q_static.grad is not None:
        print(f"   ✓ q_static has gradients: {model.q_static.grad.abs().mean().item():.6f}")
    else:
        print(f"   ✗ WARNING: q_static has no gradients!")

    print("\n" + "=" * 70)
    print("✓ FoveatedVideoModel test passed!")
    print("=" * 70)
