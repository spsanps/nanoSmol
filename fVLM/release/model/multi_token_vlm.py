"""
Multi-Token Vision-Language Model (B1 baseline).

Standard VLM architecture: DINOv2 patch tokens → pool → project → LLM.
No foveated attention, no query mechanism, no two-pass.

Each frame produces 16 visual tokens (4x4 average pool of DINOv2 patches).
This is the standard approach used by most VLMs (LLaVA, etc.) — serves as
the baseline for our foveated 1-token-per-frame approach.

Architecture:
  DINOv2 → 14x14=196 patches → AdaptiveAvgPool2d(4,4) → 16 tokens → Linear → LLM

Loss: text cross-entropy (same as FoveatedVLM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Dinov2Model
from typing import Dict, Optional


class MultiTokenVLM(nn.Module):
    """
    Multi-token baseline VLM for comparison with foveated approach.

    Parameters
    ----------
    llm_name : str
        HuggingFace model id for SmolLM2.
    dino_name : str
        HuggingFace model id for DINOv2.
    tokens_per_frame : int
        Number of visual tokens per frame (default 16 = 4x4 pool).
    visual_scale : float
        Scale factor for visual tokens to match LLM embedding magnitude.
    """

    def __init__(
        self,
        llm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_name: str = "facebook/dinov2-small",
        tokens_per_frame: int = 16,
        visual_scale: float = 0.14,
    ):
        super().__init__()

        # ---- Vision encoder (DINOv2, no foveated attention) ----
        self.dino = Dinov2Model.from_pretrained(dino_name)
        dino_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size

        # ---- Language model ----
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, attn_implementation="sdpa", torch_dtype=torch.float32,
        )
        self.llm.config.use_cache = False
        llm_dim = self.llm.config.hidden_size

        # ---- Spatial pooling ----
        # For 224x224 images with patch_size=14: 16x16=256 patches (or 14x14 + CLS)
        # We average-pool to sqrt(tokens_per_frame) x sqrt(tokens_per_frame) grid
        pool_size = int(tokens_per_frame ** 0.5)
        assert pool_size * pool_size == tokens_per_frame, \
            f"tokens_per_frame must be a perfect square, got {tokens_per_frame}"
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # ---- Projection ----
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)

        # ---- Hyperparams ----
        self.visual_scale = visual_scale
        self.tokens_per_frame = tokens_per_frame
        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.pool_size = pool_size

    def _get_pad_token_id(self) -> int:
        pid = getattr(self.llm.config, "pad_token_id", None)
        if pid is None:
            pid = getattr(self.llm.config, "eos_token_id", 0)
        return pid

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def _encode_frames(self, frames: torch.Tensor, frame_mask=None) -> torch.Tensor:
        """
        Encode video frames to multi-token visual representations.

        frames     : [B, T, 3, 224, 224]
        frame_mask : [B, T] bool — True for real frames, False for padding.
        Returns: [B, T * tokens_per_frame, llm_dim]
        """
        B, T, C, H, W = frames.shape

        # Flatten batch and time
        flat = frames.reshape(B * T, C, H, W)

        # Skip padded frames through DINO for efficiency
        if frame_mask is not None:
            mask_flat = frame_mask.reshape(B * T)
            n_real = mask_flat.sum().item()
        else:
            mask_flat = None
            n_real = B * T

        if mask_flat is not None and n_real < B * T:
            real_frames = flat[mask_flat]
            dino_out = self.dino(real_frames)
            real_features = dino_out.last_hidden_state[:, 1:, :]  # drop CLS
            # Scatter back
            N = real_features.shape[1]
            patch_features = torch.zeros(B * T, N, self.dino_dim, dtype=real_features.dtype, device=real_features.device)
            patch_features[mask_flat] = real_features
        else:
            # DINOv2 forward → patch features [B*T, N+1, D] (CLS + patches)
            dino_out = self.dino(flat)
            patch_features = dino_out.last_hidden_state[:, 1:, :]  # drop CLS → [B*T, N, D]

        # Reshape to spatial grid
        grid_size = int(patch_features.shape[1] ** 0.5)  # 16 for 224/14
        spatial = patch_features.reshape(B * T, grid_size, grid_size, self.dino_dim)
        spatial = spatial.permute(0, 3, 1, 2)  # [B*T, D, G, G]

        # Average pool to target grid
        pooled = self.spatial_pool(spatial)  # [B*T, D, pool_size, pool_size]
        pooled = pooled.permute(0, 2, 3, 1)  # [B*T, pool_size, pool_size, D]
        pooled = pooled.reshape(B * T, self.tokens_per_frame, self.dino_dim)

        # Project to LLM space and scale
        visual = self.dino_to_llm(pooled) * self.visual_scale  # [B*T, tpf, llm_dim]

        # Reshape to [B, T * tpf, llm_dim]
        visual = visual.reshape(B, T * self.tokens_per_frame, self.llm_dim)

        return visual

    def _ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard autoregressive CE loss with shift-by-1."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        if loss_mask is not None:
            shift_mask = loss_mask[:, 1:].contiguous()
            pad_id = self._get_pad_token_id()
            shift_labels = shift_labels.clone()
            shift_labels[shift_mask == 0] = pad_id

        V = shift_logits.shape[-1]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=self._get_pad_token_id(),
            reduction="mean",
        )
        return loss

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        mode: str = "coarse_fine",  # ignored, for API compat with FoveatedVLM
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        frames         : [B, T, 3, 224, 224]
        input_ids      : [B, S]
        attention_mask : [B, S]
        loss_mask      : [B, S]
        frame_mask     : [B, T] bool — True for real frames, False for padding.
        mode           : ignored (API compat with FoveatedVLM)
        """
        B, T = frames.shape[:2]
        S = input_ids.shape[1]
        V_tokens = T * self.tokens_per_frame  # total visual tokens

        # ---- Encode frames ----
        visual_embeds = self._encode_frames(frames, frame_mask)  # [B, V_tokens, llm_dim]

        # ---- Build sequence: [visual, text] ----
        text_embeds = self._embed_text(input_ids)  # [B, S, llm_dim]
        seq = torch.cat([visual_embeds, text_embeds], dim=1)  # [B, V+S, llm_dim]

        # ---- LLM forward ----
        out = self.llm.model(inputs_embeds=seq)
        logits = self.llm.lm_head(out.last_hidden_state)  # [B, V+S, vocab]

        # ---- Loss on text portion ----
        pad_id = self._get_pad_token_id()
        visual_pad = torch.full(
            (B, V_tokens), pad_id, dtype=input_ids.dtype, device=input_ids.device,
        )
        full_labels = torch.cat([visual_pad, input_ids], dim=1)

        if loss_mask is not None:
            visual_no_loss = torch.zeros(
                B, V_tokens, dtype=loss_mask.dtype, device=loss_mask.device,
            )
            full_loss_mask = torch.cat([visual_no_loss, loss_mask], dim=1)
        else:
            visual_no_loss = torch.zeros(B, V_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
            full_loss_mask = torch.cat([visual_no_loss, attention_mask], dim=1)

        loss = self._ce_loss(logits, full_labels, full_loss_mask)

        return {
            "loss": loss,
            "fine_loss": loss,      # for logging compat
            "coarse_loss": torch.tensor(0.0, device=frames.device),
            "logits": logits,
        }

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def get_param_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_connector: float = 1e-4,
    ) -> list:
        """Differential LR param groups (same API as FoveatedVLM)."""
        connector_params = set(id(p) for p in self.dino_to_llm.parameters())

        dino_params = set()
        for p in self.dino.parameters():
            if id(p) not in connector_params:
                dino_params.add(id(p))

        groups = [
            {
                "params": [p for p in self.parameters() if id(p) in connector_params and p.requires_grad],
                "lr": lr_connector,
                "name": "connector",
            },
            {
                "params": [p for p in self.dino.parameters() if id(p) in dino_params and p.requires_grad],
                "lr": lr_backbone,
                "name": "dino",
            },
            {
                "params": [p for p in self.llm.parameters() if p.requires_grad],
                "lr": lr_backbone,
                "name": "llm",
            },
        ]
        return [g for g in groups if len(g["params"]) > 0]
