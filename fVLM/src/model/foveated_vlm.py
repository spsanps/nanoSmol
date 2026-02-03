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
        deep_query: bool = True,  # Use deep query injection for better differentiation
        freeze_dino: bool = False,  # Freeze DINO backbone (ablation winner!)
    ):
        """
        Args:
            dino_model: HuggingFace model ID for vision encoder
            llm_model: HuggingFace model ID for LLM
            dino_dim: DINO embedding dimension
            llm_dim: LLM hidden dimension
            query_dim: Query vector dimension
            lambda_coarse: Weight for auxiliary coarse loss
            deep_query: If True, query propagates through all DINO layers (more selective)
            freeze_dino: If True, freeze DINO backbone (preserves pretrained diversity)
        """
        super().__init__()

        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.query_dim = query_dim
        self.lambda_coarse = lambda_coarse
        self.freeze_dino = freeze_dino

        # Vision encoder
        # NOTE: deep_query=True is critical! Shallow mode produces nearly uniform
        # attention (output correlation ~0.98), while deep mode is selective (correlation ~0.43).
        # Without deep mode, loss_fine == loss_coarse always.
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_model,
            query_dim=query_dim,
            output_dim=dino_dim,
            deep_query=deep_query,
        )

        # Freeze DINO if requested (ablation study winner!)
        # This preserves pretrained feature diversity, forcing the query mechanism
        # to learn meaningful differentiation rather than collapsing features.
        if freeze_dino:
            for param in self.encoder.dino.parameters():
                param.requires_grad = False
            self.encoder.dino.eval()

        # Core LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model, attn_implementation='sdpa')
        self.llm.config.use_cache = False  # Disable KV cache during training

        # Projections with scaling to match LLM embedding scale (~0.14 std)
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.visual_scale = 0.14  # Scale factor to match LLM embedding std
        self.llm_to_query = nn.Linear(llm_dim, query_dim)

        # Prediction head (shared between passes)
        self.pred_head = PredictionHead(h_dim=llm_dim, latent_channels=4)

        # Learned parameters
        # NOTE: Queries should have std=1.0 (not 0.02!) so they dominate over
        # the projection bias and produce meaningful attention patterns.
        # See core_docs/foveated_vlm_proposal.md lines 529-531
        self.q_static = nn.Parameter(torch.randn(1, query_dim))  # std=1.0
        self.q_init = nn.Parameter(torch.randn(1, query_dim))    # std=1.0
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

        # Mode tokens (to signal coarse vs fine pass)
        # Scale to match LLM embedding std (~0.14) to avoid gradient explosion
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)
        self.fine_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

        # Learnable "no text" token for Phase 1 (self-supervised)
        self.no_text_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

        # Video end token for captioning (signals transition to text generation)
        self.video_end_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.14)

    def get_empty_text_embeds(self, batch_size: int) -> torch.Tensor:
        """
        Get empty text embeddings for Phase 1 (self-supervised).

        Returns learnable "no text" token for each sample.
        """
        # Return learnable no_text_token expanded to batch size
        return self.no_text_token.expand(batch_size, -1, -1)

    def get_text_embeds(self, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get text embeddings from LLM for Phase 2 (text-conditioned).

        Args:
            text_input_ids: [B, N_text] tokenized text
            text_attention_mask: [B, N_text] attention mask

        Returns:
            text_embeds: [B, N_text, llm_dim] embedded text tokens
        """
        # Get embeddings from LLM's embedding layer
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)  # [B, N_text, llm_dim]

        # Zero out padding tokens
        text_embeds = text_embeds * text_attention_mask.unsqueeze(-1)

        return text_embeds

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
        B, T, C, H, W = raw_frames.shape
        N_text = text_embeds.shape[1]

        # === Encode all frames with DINO, cache features ===
        # Batch process all frames at once for efficiency
        frames_flat = raw_frames.reshape(B * T, C, H, W)  # [B*T, 3, H, W]
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        # Reshape patch_features back to [B, T, N, D]
        patch_features_flat = cache_flat['patch_features']  # [B*T, N+1, D]
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)  # [B, T, N+1, D]

        # Create per-frame caches
        # For shallow mode: just patch_features
        # For deep mode: also includes kv_cache (handled by encoder)
        all_caches = []
        if 'kv_cache' in cache_flat:
            # Deep mode: reshape kv_cache for per-frame access
            num_layers = len(cache_flat['kv_cache'])
            for t in range(T):
                frame_kv_cache = []
                for layer_idx in range(num_layers):
                    layer_cache = cache_flat['kv_cache'][layer_idx]
                    K_all = layer_cache['K'].reshape(B, T, N, D)
                    V_all = layer_cache['V'].reshape(B, T, N, D)
                    frame_kv_cache.append({
                        'K': K_all[:, t],
                        'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({
                    'patch_features': patch_features[:, t],
                    'kv_cache': frame_kv_cache,
                })
        else:
            # Shallow mode: just patch_features per frame
            for t in range(T):
                all_caches.append({
                    'patch_features': patch_features[:, t],  # [B, N+1, D]
                })

        # === Pass 1: Query Planning with q_static ===
        q_static = self.q_static.expand(B, -1)  # [B, query_dim]

        z_coarse_list = []
        for t in range(T):
            z_t = self.encoder.query_attend(q_static, all_caches[t])
            z_coarse_list.append(z_t)
        z_coarse = torch.stack(z_coarse_list, dim=1)  # [B, T, dino_dim]
        z_coarse = self.dino_to_llm(z_coarse)  # [B, T, llm_dim]
        z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale  # Scale to match LLM

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
        z_focused = z_focused / (z_focused.std() + 1e-6) * self.visual_scale  # Scale to match LLM

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

    def forward_captioning(
        self,
        raw_frames: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        use_fine: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for captioning training.

        Args:
            raw_frames: [B, T, 3, 256, 256] video frames (ImageNet normalized)
            caption_ids: [B, L] tokenized caption (target)
            caption_mask: [B, L] attention mask for caption
            use_fine: Use fine (dynamic) queries vs coarse (static)

        Returns:
            loss: Cross-entropy loss on caption tokens
        """
        B, T, C, H, W = raw_frames.shape
        L = caption_ids.shape[1]
        device = raw_frames.device

        # Encode all frames with DINO
        frames_flat = raw_frames.reshape(B * T, C, H, W)
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Create per-frame caches (must handle both shallow and deep mode)
        all_caches = []
        if 'kv_cache' in cache_flat:
            # Deep mode: reshape kv_cache for per-frame access
            num_layers = len(cache_flat['kv_cache'])
            for t in range(T):
                frame_kv_cache = []
                for layer_idx in range(num_layers):
                    layer_cache = cache_flat['kv_cache'][layer_idx]
                    K_all = layer_cache['K'].reshape(B, T, N, D)
                    V_all = layer_cache['V'].reshape(B, T, N, D)
                    frame_kv_cache.append({
                        'K': K_all[:, t],
                        'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({
                    'patch_features': patch_features[:, t],
                    'kv_cache': frame_kv_cache,
                })
        else:
            # Shallow mode: just patch_features per frame
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        if use_fine:
            # Two-pass encoding
            q_static = self.q_static.expand(B, -1)
            z_coarse_list = [self.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
            z_coarse = torch.stack(z_coarse_list, dim=1)
            z_coarse = self.dino_to_llm(z_coarse)
            z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale

            # Get queries from LLM
            coarse_token = self.coarse_token.expand(B, -1, -1)
            no_text = self.no_text_token.expand(B, -1, -1)
            seq_pass1 = torch.cat([no_text, coarse_token, z_coarse], dim=1)
            outputs_pass1 = self.llm.model(inputs_embeds=seq_pass1)
            h_pass1 = outputs_pass1.last_hidden_state
            queries = self.llm_to_query(h_pass1[:, 2:])

            # Fine pass with shifted queries
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)
            z_visual_list = [self.encoder.query_attend(shifted_q[:, t], all_caches[t]) for t in range(T)]
            z_visual = torch.stack(z_visual_list, dim=1)
        else:
            q_static = self.q_static.expand(B, -1)
            z_visual_list = [self.encoder.query_attend(q_static, all_caches[t]) for t in range(T)]
            z_visual = torch.stack(z_visual_list, dim=1)

        # Project to LLM space
        z_visual = self.dino_to_llm(z_visual)
        z_visual = z_visual / (z_visual.std() + 1e-6) * self.visual_scale

        # Get caption embeddings (teacher forcing)
        caption_embeds = self.llm.get_input_embeddings()(caption_ids)  # [B, L, llm_dim]

        # Build sequence: [<fine/coarse>, z_1, ..., z_T, <video_end>, caption_tokens]
        mode_token = self.fine_token if use_fine else self.coarse_token
        mode_token = mode_token.expand(B, -1, -1)
        video_end = self.video_end_token.expand(B, -1, -1)

        seq = torch.cat([mode_token, z_visual, video_end, caption_embeds], dim=1)
        # seq shape: [B, 1 + T + 1 + L, llm_dim]

        # LLM forward
        outputs = self.llm.model(inputs_embeds=seq)
        hidden = outputs.last_hidden_state  # [B, 1 + T + 1 + L, llm_dim]

        # Get logits for caption positions (after video_end token)
        # Positions: mode(1) + visual(T) + video_end(1) + caption(L)
        # We predict caption[1:] from positions [1+T+1 : 1+T+1+L-1]
        caption_start = 1 + T + 1
        h_for_caption = hidden[:, caption_start-1:-1, :]  # [B, L, llm_dim]
        logits = self.llm.lm_head(h_for_caption)  # [B, L, vocab_size]

        # Cross-entropy loss
        # Sequence: [mode, z_1..z_T, video_end, cap_0..cap_{L-1}]
        # hidden[T+1] (video_end position) predicts cap_0
        # hidden[T+2] (cap_0 position) predicts cap_1
        # ...
        # So logits[:, i] directly predicts caption_ids[:, i]

        # SmolLM2 uses pad_token_id=2 (same as eos)
        pad_token_id = 2

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            caption_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction='mean'
        )

        return loss

    def forward_autoregressive_captioning(
        self,
        raw_frames: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        TRUE AUTOREGRESSIVE forward pass for captioning.

        Unlike forward_captioning (which uses coarse features to generate queries),
        this method generates each query from the PREVIOUS FINE features - exactly
        as would happen during real inference.

        This is the ground truth for measuring true inference loss.

        Args:
            raw_frames: [B, T, 3, 256, 256] video frames (ImageNet normalized)
            caption_ids: [B, L] tokenized caption (target)
            caption_mask: [B, L] attention mask for caption

        Returns:
            loss: Cross-entropy loss on caption tokens (true inference loss)
        """
        B, T, C, H, W = raw_frames.shape
        L = caption_ids.shape[1]
        device = raw_frames.device

        # === Step 1: Encode all frames with DINO (parallel - this is OK) ===
        frames_flat = raw_frames.reshape(B * T, C, H, W)
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Create per-frame caches
        all_caches = []
        if 'kv_cache' in cache_flat:
            num_layers = len(cache_flat['kv_cache'])
            for t in range(T):
                frame_kv_cache = []
                for layer_idx in range(num_layers):
                    layer_cache = cache_flat['kv_cache'][layer_idx]
                    K_all = layer_cache['K'].reshape(B, T, N, D)
                    V_all = layer_cache['V'].reshape(B, T, N, D)
                    frame_kv_cache.append({
                        'K': K_all[:, t],
                        'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({
                    'patch_features': patch_features[:, t],
                    'kv_cache': frame_kv_cache,
                })
        else:
            all_caches = [{'patch_features': patch_features[:, t]} for t in range(T)]

        # === Step 2: Autoregressive visual encoding ===
        # Key difference: queries come from PREVIOUS FINE features, not coarse
        z_fine_list = []
        query = self.q_init.expand(B, -1)  # Start with initial query

        # Enable KV cache for incremental LLM processing
        orig_use_cache = self.llm.config.use_cache
        self.llm.config.use_cache = True

        # Initial sequence: [<fine>]
        fine_token = self.fine_token.expand(B, -1, -1)
        llm_past_kv = None  # Will hold accumulated KV cache

        for t in range(T):
            # Extract features with current query (this is the "foveated" part)
            z_t = self.encoder.query_attend(query, all_caches[t])  # [B, dino_dim]
            z_fine_list.append(z_t)

            # Project to LLM space
            z_t_llm = self.dino_to_llm(z_t)  # [B, llm_dim]
            z_t_llm = z_t_llm / (z_t_llm.std() + 1e-6) * self.visual_scale
            z_t_llm = z_t_llm.unsqueeze(1)  # [B, 1, llm_dim]

            # Incremental LLM forward
            if t == 0:
                # First frame: include mode token
                seq_input = torch.cat([fine_token, z_t_llm], dim=1)  # [B, 2, llm_dim]
            else:
                # Subsequent frames: just the new visual token
                seq_input = z_t_llm  # [B, 1, llm_dim]

            outputs = self.llm.model(
                inputs_embeds=seq_input,
                past_key_values=llm_past_kv,
                use_cache=True,
            )
            llm_past_kv = outputs.past_key_values

            # Generate query for NEXT frame from current hidden state
            if t < T - 1:
                h_t = outputs.last_hidden_state[:, -1, :]  # [B, llm_dim]
                query = self.llm_to_query(h_t)  # [B, query_dim]

        # Stack fine features
        z_fine = torch.stack(z_fine_list, dim=1)  # [B, T, dino_dim]
        z_fine_llm = self.dino_to_llm(z_fine)
        z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * self.visual_scale

        # === Step 3: Compute caption loss ===
        # Now we have the autoregressive visual features, compute caption loss
        # Build sequence: [<fine>, z_1, ..., z_T, <video_end>, caption]
        # But we already processed [<fine>, z_1, ..., z_T] incrementally
        # Continue with video_end and caption

        video_end = self.video_end_token.expand(B, -1, -1)  # [B, 1, llm_dim]
        caption_embeds = self.llm.get_input_embeddings()(caption_ids)  # [B, L, llm_dim]

        # Continue LLM forward with video_end + caption (using cached KV)
        seq_caption = torch.cat([video_end, caption_embeds], dim=1)  # [B, 1+L, llm_dim]

        outputs = self.llm.model(
            inputs_embeds=seq_caption,
            past_key_values=llm_past_kv,
            use_cache=False,  # Don't need cache for this part
        )
        hidden = outputs.last_hidden_state  # [B, 1+L, llm_dim]

        # Get logits for caption positions
        # hidden[:, 0] is after video_end, predicts caption[0]
        # hidden[:, i] predicts caption[i]
        h_for_caption = hidden[:, :-1, :]  # [B, L, llm_dim] - shift for next-token prediction
        logits = self.llm.lm_head(h_for_caption)  # [B, L, vocab_size]

        # Cross-entropy loss
        pad_token_id = 2  # SmolLM2 pad token
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            caption_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction='mean'
        )

        # Restore cache setting
        self.llm.config.use_cache = orig_use_cache

        return loss

    @torch.no_grad()
    def encode_video(self, raw_frames: torch.Tensor, use_fine: bool = True) -> torch.Tensor:
        """
        Encode video frames to visual embeddings for captioning.

        Args:
            raw_frames: [B, T, 3, 256, 256] video frames (ImageNet normalized)
            use_fine: If True, use fine (dynamic) queries; else use coarse (static)

        Returns:
            z_visual: [B, T, llm_dim] visual embeddings ready for LLM
        """
        B, T, C, H, W = raw_frames.shape

        # Encode all frames with DINO
        frames_flat = raw_frames.reshape(B * T, C, H, W)
        _, cache_flat = self.encoder.encode_patches(frames_flat)

        patch_features_flat = cache_flat['patch_features']
        N, D = patch_features_flat.shape[1], patch_features_flat.shape[2]
        patch_features = patch_features_flat.reshape(B, T, N, D)

        # Create per-frame caches (must handle both shallow and deep mode)
        all_caches = []
        if 'kv_cache' in cache_flat:
            # Deep mode: reshape kv_cache for per-frame access
            num_layers = len(cache_flat['kv_cache'])
            for t in range(T):
                frame_kv_cache = []
                for layer_idx in range(num_layers):
                    layer_cache = cache_flat['kv_cache'][layer_idx]
                    K_all = layer_cache['K'].reshape(B, T, N, D)
                    V_all = layer_cache['V'].reshape(B, T, N, D)
                    frame_kv_cache.append({
                        'K': K_all[:, t],
                        'V': V_all[:, t],
                        'layer': layer_cache['layer'],
                    })
                all_caches.append({
                    'patch_features': patch_features[:, t],
                    'kv_cache': frame_kv_cache,
                })
        else:
            # Shallow mode: just patch_features per frame
            for t in range(T):
                all_caches.append({'patch_features': patch_features[:, t]})

        if use_fine:
            # Two-pass: first get queries from coarse pass, then use them
            q_static = self.q_static.expand(B, -1)
            z_coarse_list = []
            for t in range(T):
                z_t = self.encoder.query_attend(q_static, all_caches[t])
                z_coarse_list.append(z_t)
            z_coarse = torch.stack(z_coarse_list, dim=1)
            z_coarse = self.dino_to_llm(z_coarse)
            z_coarse = z_coarse / (z_coarse.std() + 1e-6) * self.visual_scale

            # Get queries from LLM
            coarse_token = self.coarse_token.expand(B, -1, -1)
            no_text = self.no_text_token.expand(B, -1, -1)
            seq_pass1 = torch.cat([no_text, coarse_token, z_coarse], dim=1)
            outputs_pass1 = self.llm.model(inputs_embeds=seq_pass1)
            h_pass1 = outputs_pass1.last_hidden_state
            h_for_queries = h_pass1[:, 2:]  # After no_text and coarse_token
            queries = self.llm_to_query(h_for_queries)

            # Use shifted queries for fine pass
            q_init = self.q_init.expand(B, -1).unsqueeze(1)
            shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)

            z_focused_list = []
            for t in range(T):
                z_t = self.encoder.query_attend(shifted_q[:, t], all_caches[t])
                z_focused_list.append(z_t)
            z_visual = torch.stack(z_focused_list, dim=1)
        else:
            # Just use coarse (static query)
            q_static = self.q_static.expand(B, -1)
            z_coarse_list = []
            for t in range(T):
                z_t = self.encoder.query_attend(q_static, all_caches[t])
                z_coarse_list.append(z_t)
            z_visual = torch.stack(z_coarse_list, dim=1)

        # Project to LLM space
        z_visual = self.dino_to_llm(z_visual)
        z_visual = z_visual / (z_visual.std() + 1e-6) * self.visual_scale

        return z_visual

    @torch.no_grad()
    def generate_caption(
        self,
        raw_frames: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_fine: bool = True,
    ) -> list:
        """
        Generate captions from video frames.

        Args:
            raw_frames: [B, T, 3, 256, 256] video frames (ImageNet normalized)
            tokenizer: HuggingFace tokenizer for decoding
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold
            use_fine: Use fine (dynamic) queries vs coarse (static)

        Returns:
            captions: List of generated caption strings
        """
        B = raw_frames.shape[0]
        device = raw_frames.device

        # Enable cache for generation (disabled during training)
        orig_use_cache = self.llm.config.use_cache
        self.llm.config.use_cache = True

        # Encode video
        z_visual = self.encode_video(raw_frames, use_fine=use_fine)

        # Build sequence: [<fine>, z_1, ..., z_T, <video_end>]
        mode_token = self.fine_token if use_fine else self.coarse_token
        mode_token = mode_token.expand(B, -1, -1)
        video_end = self.video_end_token.expand(B, -1, -1)

        # Initial sequence
        seq = torch.cat([mode_token, z_visual, video_end], dim=1)

        # Get initial hidden states AND cache
        outputs = self.llm.model(inputs_embeds=seq)
        hidden = outputs.last_hidden_state
        past_key_values = outputs.past_key_values  # Save cache from video encoding!

        # Autoregressive generation
        generated_ids = []

        for _ in range(max_new_tokens):
            # Get logits from last position
            if past_key_values is None:
                logits = self.llm.lm_head(hidden[:, -1:, :])
            else:
                logits = self.llm.lm_head(hidden)

            logits = logits[:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            for b in range(B):
                logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token)

            # Check for EOS
            if (next_token == tokenizer.eos_token_id).all():
                break

            # Get embedding for next token and continue
            next_embed = self.llm.get_input_embeddings()(next_token)
            outputs = self.llm.model(inputs_embeds=next_embed, past_key_values=past_key_values)
            hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        # Decode
        if generated_ids:
            generated_ids = torch.cat(generated_ids, dim=1)
            captions = []
            for b in range(B):
                ids = generated_ids[b].tolist()
                # Stop at EOS
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                captions.append(caption)
        else:
            captions = [""] * B

        # Restore original cache setting
        self.llm.config.use_cache = orig_use_cache

        return captions


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

    # Test captioning methods (training vs autoregressive)
    print("\n🔄 Testing captioning methods...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    # Create dummy caption
    caption_text = "A video showing some action."
    caption_ids = tokenizer(caption_text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    caption_ids_batch = caption_ids["input_ids"].expand(batch_size, -1).to(device)
    caption_mask_batch = caption_ids["attention_mask"].expand(batch_size, -1).to(device)

    model.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Training-style loss (parallel approximation)
        loss_train_fine = model.forward_captioning(raw_frames, caption_ids_batch, caption_mask_batch, use_fine=True)
        loss_train_coarse = model.forward_captioning(raw_frames, caption_ids_batch, caption_mask_batch, use_fine=False)

        # True autoregressive inference loss
        loss_autoregressive = model.forward_autoregressive_captioning(raw_frames, caption_ids_batch, caption_mask_batch)

    print(f"\n   Captioning losses:")
    print(f"     Training (fine):       {loss_train_fine.item():.4f}")
    print(f"     Training (coarse):     {loss_train_coarse.item():.4f}")
    print(f"     Autoregressive (true): {loss_autoregressive.item():.4f}")
    print(f"     Train/Inference gap:   {(loss_autoregressive.item() - loss_train_fine.item()):.4f}")

    print("\n" + "=" * 70)
    print("✓ FoveatedVideoModel test passed!")
    print("=" * 70)
