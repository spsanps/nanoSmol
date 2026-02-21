"""
Baseline VLM: DINOv2 + PixelShuffle + MLP + SmolLM2

Uses the same connector architecture as SmolVLM (PixelShuffle 4x + 2-layer MLP),
but with DINOv2 as the vision encoder instead of SigLIP.

This isolates the comparison to: 1 token/frame (foveated) vs 16 tokens/frame (baseline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Dinov2Model


class PixelShuffleConnector(nn.Module):
    """SmolVLM-style connector: PixelShuffle + 2-layer MLP with SiLU.

    PixelShuffle rearranges spatial patches into channel dimension,
    reducing token count by scale_factor^2 while increasing channels.
    """

    def __init__(self, vision_dim: int, text_dim: int, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        # After pixel shuffle, channels increase by scale_factor^2
        shuffle_dim = vision_dim * (scale_factor ** 2)

        # 2-layer MLP with SiLU (matching SmolVLM)
        hidden_dim = text_dim * 4  # Standard MLP expansion
        self.mlp = nn.Sequential(
            nn.Linear(shuffle_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] patch features from DINOv2
        Returns:
            [B, H', W', text_dim] where H'=H/scale, W'=W/scale
        """
        B, H, W, C = x.shape
        scale = self.scale_factor

        # Pixel shuffle: [B, H, W, C] -> [B, H/s, W/s, C*s*s]
        # Reshape to group patches
        assert H % scale == 0 and W % scale == 0, f"H={H}, W={W} must be divisible by scale={scale}"

        x = x.reshape(B, H // scale, scale, W // scale, scale, C)
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H/s, W/s, s, s, C]
        x = x.reshape(B, H // scale, W // scale, scale * scale * C)  # [B, H/s, W/s, C*s*s]

        # MLP projection
        x = self.mlp(x)  # [B, H/s, W/s, text_dim]

        return x


class BaselineVLM(nn.Module):
    """Baseline VLM with DINOv2 + PixelShuffle + SmolLM2.

    Uses 16 tokens per frame (vs foveated's 1 token).
    Input frames should be 224x224 for clean 16x16 patch grid.
    """

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-small",
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        pixel_shuffle_scale: int = 4,
        freeze_dino: bool = False,
        freeze_llm: bool = False,
    ):
        super().__init__()

        # Vision encoder
        self.dino = Dinov2Model.from_pretrained(dino_model)
        self.dino_dim = self.dino.config.hidden_size  # 384 for small

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm_dim = self.llm.config.hidden_size  # 576 for SmolLM2-135M

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # Connector: PixelShuffle + MLP
        self.connector = PixelShuffleConnector(
            vision_dim=self.dino_dim,
            text_dim=self.llm_dim,
            scale_factor=pixel_shuffle_scale,
        )

        # Visual scale factor (for normalization, matching foveated)
        self.visual_scale = nn.Parameter(torch.ones(1))

        # Mode token to indicate visual input
        self.visual_token = nn.Parameter(torch.randn(1, 1, self.llm_dim))

        self.pixel_shuffle_scale = pixel_shuffle_scale

    def encode_frames(self, frames):
        """Encode frames through DINOv2 + PixelShuffle connector.

        Args:
            frames: [B, T, 3, 224, 224] normalized frames
        Returns:
            [B, T, num_tokens, llm_dim] visual features
        """
        B, T, C, H, W = frames.shape

        # Flatten batch and time
        frames_flat = frames.reshape(B * T, C, H, W)

        # DINOv2 forward
        outputs = self.dino(frames_flat, return_dict=True)
        patch_features = outputs.last_hidden_state[:, 1:]  # Remove CLS, [B*T, N, D]

        # Reshape to spatial grid
        # DINOv2 with 224x224 and patch 14: 224/14 = 16 -> 16x16 = 256 patches
        num_patches = patch_features.shape[1]
        grid_size = int(num_patches ** 0.5)
        assert grid_size * grid_size == num_patches, f"Non-square patch grid: {num_patches}"

        patch_features = patch_features.reshape(B * T, grid_size, grid_size, self.dino_dim)

        # PixelShuffle + MLP
        visual_features = self.connector(patch_features)  # [B*T, H', W', llm_dim]

        # Flatten spatial dims to tokens
        H_new, W_new = visual_features.shape[1], visual_features.shape[2]
        num_tokens = H_new * W_new  # Should be 16 for scale=4 on 16x16 grid
        visual_features = visual_features.reshape(B * T, num_tokens, self.llm_dim)

        # Reshape back to [B, T, num_tokens, llm_dim]
        visual_features = visual_features.reshape(B, T, num_tokens, self.llm_dim)

        # Normalize
        visual_features = visual_features / (visual_features.std() + 1e-6) * self.visual_scale

        return visual_features

    def forward(self, frames, caption_embeds, caption_targets):
        """Forward pass for caption training.

        Args:
            frames: [B, T, 3, 224, 224] normalized frames
            caption_embeds: [B, L, llm_dim] caption token embeddings
            caption_targets: [B, L-1] caption target ids (shifted)

        Returns:
            loss: scalar caption loss
            logits: [B, L-1, vocab] caption logits
        """
        B, T, C, H, W = frames.shape

        # Encode frames
        visual_features = self.encode_frames(frames)  # [B, T, num_tokens, llm_dim]

        # Flatten visual tokens: [B, T * num_tokens, llm_dim]
        num_tokens = visual_features.shape[2]
        visual_features = visual_features.reshape(B, T * num_tokens, self.llm_dim)

        # Build sequence: [visual_token, visual_features, caption_embeds]
        visual_token = self.visual_token.expand(B, -1, -1)
        seq = torch.cat([visual_token, visual_features, caption_embeds], dim=1)

        # Forward through LLM
        outputs = self.llm.model(inputs_embeds=seq)
        logits = self.llm.lm_head(outputs.last_hidden_state)

        # Extract caption logits (after visual tokens + visual features)
        visual_len = 1 + T * num_tokens
        caption_logits = logits[:, visual_len:-1, :]  # Shifted by 1 for next-token prediction

        # Compute loss
        loss = F.cross_entropy(
            caption_logits.reshape(-1, caption_logits.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=-100,
        )

        return loss, caption_logits

    def generate_caption(self, frames, tokenizer, max_length=64):
        """Generate caption for frames.

        Args:
            frames: [1, T, 3, 224, 224] normalized frames (batch size 1)
            tokenizer: tokenizer for decoding
            max_length: max caption length

        Returns:
            caption: generated caption string
        """
        self.eval()
        device = frames.device
        B = 1
        T = frames.shape[1]

        with torch.no_grad():
            # Encode frames
            visual_features = self.encode_frames(frames)  # [1, T, num_tokens, llm_dim]
            num_tokens = visual_features.shape[2]
            visual_features = visual_features.reshape(B, T * num_tokens, self.llm_dim)

            # Start with visual token + visual features + BOS
            visual_token = self.visual_token.expand(B, -1, -1)

            bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
            current_ids = torch.tensor([[bos_id]], device=device)
            current_embeds = self.llm.model.embed_tokens(current_ids)

            seq = torch.cat([visual_token, visual_features, current_embeds], dim=1)

            generated_ids = [bos_id]

            for _ in range(max_length):
                outputs = self.llm.model(inputs_embeds=seq)
                logits = self.llm.lm_head(outputs.last_hidden_state[:, -1:, :])
                next_id = logits.argmax(dim=-1).item()

                if next_id == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_id)
                next_embed = self.llm.model.embed_tokens(torch.tensor([[next_id]], device=device))
                seq = torch.cat([seq, next_embed], dim=1)

            caption = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return caption


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test the model
    print("Testing BaselineVLM...")

    model = BaselineVLM(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        pixel_shuffle_scale=4,
    )

    total, trainable = count_parameters(model)
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # Test forward pass
    B, T = 2, 8
    frames = torch.randn(B, T, 3, 224, 224)
    caption_embeds = torch.randn(B, 32, 576)
    caption_targets = torch.randint(0, 49152, (B, 31))

    model.eval()
    with torch.no_grad():
        loss, logits = model(frames, caption_embeds, caption_targets)

    print(f"Input frames: {frames.shape}")
    print(f"Visual tokens per frame: {16 // 4 * 16 // 4}")  # After pixel shuffle 4x on 16x16
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
