"""
Foveated Vision-Language Model.

Architecture: DINOv2 encoder + foveated cross-attention + SmolLM2 LLM.
Each video frame is compressed to ONE visual token via query-guided attention.
The LLM controls WHERE to look by generating the query for the next frame.

Three forward modes:
  1. forward_coarse_fine   -- Training (two parallel passes)
  2. forward_coarse_only   -- Fast eval (single static-query pass)
  3. forward_autoregressive -- True inference (sequential, KV-cached)

Loss: text cross-entropy only (no reconstruction, no VAE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, Optional

# Optional: Liger Kernel fused CE loss (never materializes [B, S, V] logits)
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False


class FoveatedVLM(nn.Module):
    """
    Foveated Vision-Language Model.

    Parameters
    ----------
    llm_name : str
        HuggingFace model id for SmolLM2 (e.g. "HuggingFaceTB/SmolLM2-135M-Instruct").
    dino_name : str
        HuggingFace model id for DINOv2 (e.g. "facebook/dinov2-small").
    query_dim : int
        Dimension of the foveated query vectors (matches DINO dim by default).
    visual_scale : float
        Multiplicative factor applied to projected visual tokens so their
        magnitude matches the LLM embedding std (~0.14 for SmolLM2).
    lambda_coarse : float
        Weight for the optional auxiliary coarse-pass CE loss during training.
        Set to 0 to disable.
    """

    def __init__(
        self,
        llm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dino_name: str = "facebook/dinov2-small",
        query_dim: int = 384,
        visual_scale: float = 0.14,
        lambda_coarse: float = 0.0,
        deep_query: bool = True,
        use_fused_ce: bool = False,
    ):
        super().__init__()

        # ---- delayed import so encoder.py can live next to this file ----
        from encoder import FoveatedEncoder

        # ---- Vision encoder (DINOv2 + query cross-attention) ----
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_name,
            query_dim=query_dim,
            output_dim=None,  # output_dim = dino_dim by default inside encoder
        )
        dino_dim = self.encoder.dino_dim

        # ---- Language model ----
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        )
        self.llm.config.use_cache = False  # training default; overridden per-method
        llm_dim = self.llm.config.hidden_size

        # ---- Projections ----
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.llm_to_query = nn.Linear(llm_dim, query_dim)

        # ---- Learnable queries ----
        # BUG-001 FIX: init with std=1.0 so queries dominate over projection
        # bias and produce meaningful (non-uniform) attention patterns.
        self.q_static = nn.Parameter(torch.randn(1, query_dim))   # std=1.0
        self.q_init   = nn.Parameter(torch.randn(1, query_dim))   # std=1.0

        # ---- Hyperparams stored as plain Python (not buffers) ----
        self.visual_scale = visual_scale
        self.lambda_coarse = lambda_coarse
        self.query_dim = query_dim
        self.deep_query = deep_query
        self.use_fused_ce = use_fused_ce and _HAS_LIGER

        # ---- Dimension bookkeeping (useful for external code) ----
        self.dino_dim = dino_dim
        self.llm_dim = llm_dim

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_pad_token_id(self) -> int:
        """Return pad_token_id from the LLM config (never hardcoded)."""
        pid = getattr(self.llm.config, "pad_token_id", None)
        if pid is None:
            pid = getattr(self.llm.config, "eos_token_id", 0)
        return pid

    def _llm_dtype(self) -> torch.dtype:
        """Return the dtype of the LLM parameters (e.g. bfloat16)."""
        return next(self.llm.parameters()).dtype

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, S] -> [B, S, llm_dim] via LLM embedding table."""
        return self.llm.get_input_embeddings()(input_ids)

    def _project_visual(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project DINO features to LLM space and rescale.

        z : [B, T, dino_dim]  or  [B, dino_dim]
        Returns same shape with last dim = llm_dim.
        """
        h = self.dino_to_llm(z)                       # -> llm_dim
        h = h * self.visual_scale                      # match LLM embedding magnitude
        return h

    # Maximum frames per DINO encode/query call to prevent OOM on large batches.
    _MAX_ENCODE_CHUNK = 200

    def _encode_all_frames(self, frames: torch.Tensor, frame_mask=None):
        """
        Run DINO patch encoding for every frame in the batch.

        frames     : [B, T, 3, 224, 224]
        frame_mask : [B, T] bool — True for real frames, False for padding.

        Returns (kv_cache, patch_features, mask_flat):
            kv_cache       : list of (K, V) per layer, each [n_real, N+1, D]
                             (compact — only real frames, no padding waste).
            patch_features : [n_real, N+1, D] final DINO embeddings (for shallow mode).
            mask_flat      : [B*T] bool tensor or None. Used to scatter results back.
        """
        B, T, C, H, W = frames.shape
        BT = B * T
        frames_flat = frames.reshape(BT, C, H, W)

        if frame_mask is not None:
            mask_flat = frame_mask.reshape(BT)
            n_real = mask_flat.sum().item()
        else:
            mask_flat = None
            n_real = BT

        if mask_flat is not None and n_real < BT:
            real_frames = frames_flat[mask_flat]          # [n_real, C, H, W]
        else:
            real_frames = frames_flat

        # Chunked encoding to prevent OOM on batches with many real frames
        if real_frames.shape[0] <= self._MAX_ENCODE_CHUNK:
            patch_features, kv_cache = self.encoder.encode_patches(real_frames)
        else:
            pf_chunks, kv_chunks = [], []
            for start in range(0, real_frames.shape[0], self._MAX_ENCODE_CHUNK):
                pf_chunk, kv_chunk = self.encoder.encode_patches(
                    real_frames[start:start + self._MAX_ENCODE_CHUNK]
                )
                pf_chunks.append(pf_chunk)
                kv_chunks.append(kv_chunk)
            patch_features = torch.cat(pf_chunks, dim=0)
            kv_cache = [
                (torch.cat([c[li][0] for c in kv_chunks], dim=0),
                 torch.cat([c[li][1] for c in kv_chunks], dim=0))
                for li in range(len(kv_chunks[0]))
            ]

        return kv_cache, patch_features, mask_flat

    def _batched_query_attend(self, queries: torch.Tensor, kv_cache: list,
                              patch_features: torch.Tensor = None) -> torch.Tensor:
        """Chunked query_attend (deep) or shallow_query_attend to prevent OOM."""
        n = queries.shape[0]
        if not self.deep_query:
            # Shallow mode: single cross-attention on final features
            if n <= self._MAX_ENCODE_CHUNK:
                return self.encoder.shallow_query_attend(queries, patch_features)
            chunks = []
            for start in range(0, n, self._MAX_ENCODE_CHUNK):
                end = min(start + self._MAX_ENCODE_CHUNK, n)
                chunks.append(self.encoder.shallow_query_attend(
                    queries[start:end], patch_features[start:end]))
            return torch.cat(chunks, dim=0)
        # Deep mode: propagate through all DINO layers
        if n <= self._MAX_ENCODE_CHUNK:
            return self.encoder.query_attend(queries, kv_cache)
        chunks = []
        for start in range(0, n, self._MAX_ENCODE_CHUNK):
            end = min(start + self._MAX_ENCODE_CHUNK, n)
            kv_slice = [(K[start:end], V[start:end]) for K, V in kv_cache]
            chunks.append(self.encoder.query_attend(queries[start:end], kv_slice))
        return torch.cat(chunks, dim=0)

    def _query_all_frames(
        self, query: torch.Tensor, kv_cache: list,
        B: int, T: int, mask_flat=None, patch_features=None,
    ) -> torch.Tensor:
        """
        Apply a single query to every frame in ONE batched query_attend call.

        query          : [B, query_dim]
        kv_cache       : list of (K, V) per layer, each [n_real, N+1, D]
        B, T           : batch and temporal dimensions
        mask_flat      : [B*T] bool or None
        patch_features : [n_real, N+1, D] (needed for shallow mode)
        Returns        : [B, T, dino_dim]
        """
        BT = B * T
        dd = self.encoder.dino_dim

        # Expand: same query for all T frames → [B*T, qd]
        query_exp = query.unsqueeze(1).expand(B, T, -1).reshape(BT, -1)

        if mask_flat is not None:
            n_real = mask_flat.sum().item()
            if n_real == 0:
                return torch.zeros(B, T, dd, device=query.device, dtype=query.dtype)
            query_real = query_exp[mask_flat]                     # [n_real, qd]
            z_real = self._batched_query_attend(query_real, kv_cache, patch_features)
            z_flat = torch.zeros(BT, dd, device=query.device, dtype=z_real.dtype)
            z_flat[mask_flat] = z_real
        else:
            z_flat = self._batched_query_attend(query_exp, kv_cache, patch_features)

        return z_flat.reshape(B, T, dd)

    def _query_all_frames_batched(
        self, queries: torch.Tensor, kv_cache: list,
        B: int, T: int, mask_flat=None, patch_features=None,
    ) -> torch.Tensor:
        """
        Apply per-frame queries in ONE batched query_attend call.

        queries        : [B, T, query_dim]
        kv_cache       : list of (K, V) per layer, each [n_real, N+1, D]
        B, T           : batch and temporal dimensions
        mask_flat      : [B*T] bool or None
        patch_features : [n_real, N+1, D] (needed for shallow mode)
        Returns        : [B, T, dino_dim]
        """
        BT = B * T
        dd = self.encoder.dino_dim
        queries_flat = queries.reshape(BT, -1)

        if mask_flat is not None:
            n_real = mask_flat.sum().item()
            if n_real == 0:
                return torch.zeros(B, T, dd, device=queries.device, dtype=queries.dtype)
            query_real = queries_flat[mask_flat]                   # [n_real, qd]
            z_real = self._batched_query_attend(query_real, kv_cache, patch_features)
            z_flat = torch.zeros(BT, dd, device=queries.device, dtype=z_real.dtype)
            z_flat[mask_flat] = z_real
        else:
            z_flat = self._batched_query_attend(queries_flat, kv_cache, patch_features)

        return z_flat.reshape(B, T, dd)

    def _extract_frame_kv(self, kv_cache: list, mask_flat, B: int, T: int, frame_idx: int):
        """
        Extract single-frame KV cache from flat format (for autoregressive/eval).

        Returns list of (K, V) per layer, each [B, N+1, D].
        """
        if mask_flat is not None:
            # Scatter compact caches to full [B*T] then extract frame
            N1 = kv_cache[0][0].shape[1]
            D = kv_cache[0][0].shape[2]
            frame_kv = []
            for K_real, V_real in kv_cache:
                K_full = torch.zeros(B * T, N1, D, dtype=K_real.dtype, device=K_real.device)
                V_full = torch.zeros(B * T, N1, D, dtype=V_real.dtype, device=V_real.device)
                K_full[mask_flat] = K_real
                V_full[mask_flat] = V_real
                K_t = K_full.reshape(B, T, N1, D)[:, frame_idx]  # [B, N+1, D]
                V_t = V_full.reshape(B, T, N1, D)[:, frame_idx]
                frame_kv.append((K_t, V_t))
            return frame_kv
        else:
            N1 = kv_cache[0][0].shape[1]
            D = kv_cache[0][0].shape[2]
            frame_kv = []
            for K_all, V_all in kv_cache:
                K_t = K_all.reshape(B, T, N1, D)[:, frame_idx]
                V_t = V_all.reshape(B, T, N1, D)[:, frame_idx]
                frame_kv.append((K_t, V_t))
            return frame_kv

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Standard causal attention mask [1, 1, S, S] for the LLM.
        True = masked (cannot attend), False = allowed.
        """
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    def _ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard autoregressive CE loss with shift-by-1.

        logits    : [B, S, V]   (full sequence logits)
        labels    : [B, S]      (token ids; positions without loss use pad)
        loss_mask : [B, S]      (1 = compute loss, 0 = ignore). Applied BEFORE
                    the shift so that loss_mask[i] guards label[i].

        Returns scalar loss.
        """
        # Shift: predict position i+1 from position i
        shift_logits = logits[:, :-1, :].contiguous()   # [B, S-1, V]
        shift_labels = labels[:, 1:].contiguous()        # [B, S-1]

        if loss_mask is not None:
            shift_mask = loss_mask[:, 1:].contiguous()   # [B, S-1]
            # Replace masked positions with ignore_index so CE ignores them
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

    def _fused_ce_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fused lm_head + CE loss via Liger Kernel.

        Never materializes the [B, S, V] logits tensor — computes CE in chunks
        inside the fused kernel.  Saves ~2× memory on the loss computation.

        hidden_states : [B, S, ld]  (LLM hidden states, NOT yet projected by lm_head)
        labels        : [B, S]      (token ids)
        loss_mask     : [B, S]      (1 = compute loss, 0 = ignore)

        Returns scalar loss.
        """
        # Shift: predict position i+1 from position i
        h_input = hidden_states[:, :-1, :].contiguous()  # [B, S-1, ld]
        shift_labels = labels[:, 1:].contiguous()         # [B, S-1]

        if loss_mask is not None:
            shift_mask = loss_mask[:, 1:].contiguous()
            pad_id = self._get_pad_token_id()
            shift_labels = shift_labels.clone()
            shift_labels[shift_mask == 0] = pad_id

        # Flatten for Liger: [B*(S-1), ld] and [B*(S-1)]
        BSminus1 = h_input.shape[0] * h_input.shape[1]
        return LigerFusedLinearCrossEntropyLoss(
            ignore_index=self._get_pad_token_id()
        )(
            h_input.reshape(BSminus1, -1),
            self.llm.lm_head.weight,
            shift_labels.reshape(-1),
        )

    # ------------------------------------------------------------------
    # Forward mode 1: Coarse+Fine (TRAINING)
    # ------------------------------------------------------------------

    def forward_coarse_fine(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Two-pass parallel training forward.

        Pass 1 (coarse): q_static -> all frames -> z_coarse -> LLM(visual only) -> queries
        Pass 2 (fine):   shifted queries -> all frames -> z_fine -> LLM + text -> loss

        Optimization: the coarse LLM pass processes ONLY visual tokens (not text).
        Because causal attention means visual positions never see text tokens,
        removing text produces mathematically identical hidden states at visual
        positions while reducing sequence length from T+S to T (~10-30x shorter).

        Parameters
        ----------
        frames         : [B, T, 3, 224, 224]
        input_ids      : [B, S]  tokenized text (prompt + answer)
        attention_mask : [B, S]  text attention mask
        loss_mask      : [B, S]  which tokens contribute to loss (1=yes, 0=no).
                         If None, all non-pad tokens have loss.

        Returns
        -------
        dict with keys: loss, logits, coarse_loss (optional), fine_loss
        """
        B, T = frames.shape[:2]
        S = input_ids.shape[1]

        # ---- Step 0: Encode all frames (DINO, shared across both passes) ----
        # Use prefetched DINO results if available (from CUDA stream overlap)
        prefetched = self._get_prefetched_dino()
        if prefetched is not None:
            kv_cache, patch_features, mask_flat = prefetched
        else:
            kv_cache, patch_features, mask_flat = self._encode_all_frames(frames, frame_mask)

        # ---- Pass 1: Coarse (visual tokens ONLY — text is invisible to them) ----
        q_static = self.q_static.expand(B, -1)                     # [B, qd]
        z_coarse = self._query_all_frames(q_static, kv_cache, B, T, mask_flat, patch_features)  # [B,T,dd]
        z_coarse_llm = self._project_visual(z_coarse)              # [B,T,ld]

        # Coarse LLM: process ONLY visual tokens (T tokens, not T+S).
        # Causal attention: visual pos i only sees visual pos 0..i, never text.
        # This is ~30x faster for typical T=8, S=256 batches.
        out_coarse = self.llm.model(inputs_embeds=z_coarse_llm)
        h_coarse = out_coarse.last_hidden_state                    # [B,T,ld]

        # Extract dynamic queries from visual positions
        queries = self.llm_to_query(h_coarse)                      # [B,T,qd]

        # Shift queries: frame t gets query from frame t-1; frame 0 gets q_init
        q_init = self.q_init.expand(B, 1, -1)                     # [B,1,qd]
        shifted_queries = torch.cat([q_init, queries[:, :-1]], dim=1)  # [B,T,qd]

        # ---- Pass 2: Fine ----
        z_fine = self._query_all_frames_batched(shifted_queries, kv_cache, B, T, mask_flat, patch_features)  # [B,T,dd]
        z_fine_llm = self._project_visual(z_fine)                  # [B,T,ld]

        # Build fine sequence: [visual_fine, text]
        text_embeds = self._embed_text(input_ids)                  # [B,S,ld]
        seq_fine = torch.cat([z_fine_llm, text_embeds], dim=1)     # [B,T+S,ld]

        out_fine = self.llm.model(inputs_embeds=seq_fine)
        h_fine = out_fine.last_hidden_state                        # [B,T+S,ld]

        # ---- Loss on text portion ----
        h_text = h_fine[:, T:, :]                                  # [B,S,ld]
        if loss_mask is None:
            loss_mask = attention_mask.float()

        if self.use_fused_ce:
            # Liger Kernel: fused lm_head + CE, never materializes [B,S,V] logits
            fine_loss = self._fused_ce_loss(h_text, input_ids, loss_mask)
            logits_text = None  # not available with fused loss
        else:
            logits_text = self.llm.lm_head(h_text)                # [B,S,V]
            fine_loss = self._ce_loss(logits_text, input_ids, loss_mask)

        # ---- Optional auxiliary coarse loss ----
        coarse_loss = torch.tensor(0.0, device=frames.device)
        if self.lambda_coarse > 0:
            seq_coarse_full = torch.cat([z_coarse_llm, text_embeds], dim=1)
            out_coarse_full = self.llm.model(inputs_embeds=seq_coarse_full)
            h_coarse_text = out_coarse_full.last_hidden_state[:, T:, :]
            if self.use_fused_ce:
                coarse_loss = self._fused_ce_loss(h_coarse_text, input_ids, loss_mask)
            else:
                logits_coarse = self.llm.lm_head(h_coarse_text)
                coarse_loss = self._ce_loss(logits_coarse, input_ids, loss_mask)

        # ---- Combined loss ----
        loss = fine_loss + self.lambda_coarse * coarse_loss

        return {
            "loss": loss,
            "fine_loss": fine_loss,
            "coarse_loss": coarse_loss,
            "logits": logits_text,  # [B,S,V] text positions only
        }

    # ------------------------------------------------------------------
    # Forward mode: DPO (preference training)
    # ------------------------------------------------------------------

    def forward_dpo(
        self,
        frames: torch.Tensor,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_loss_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_loss_mask: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        DPO forward pass: run coarse+fine on both chosen and rejected sequences.

        Shares DINO encoding across chosen and rejected (same visual input).
        Returns per-sample sum of log-probabilities for both chosen and rejected,
        masked by loss_mask (answer-only tokens).

        Parameters
        ----------
        frames                  : [B, T, 3, 224, 224]
        chosen_input_ids        : [B, S_c]
        chosen_attention_mask   : [B, S_c]
        chosen_loss_mask        : [B, S_c]  (1 = answer token, 0 = prompt/pad)
        rejected_input_ids      : [B, S_r]
        rejected_attention_mask : [B, S_r]
        rejected_loss_mask      : [B, S_r]
        frame_mask              : [B, T] bool (optional)

        Returns
        -------
        dict with keys:
          chosen_logps    : [B]  per-sample sum of log-probs on chosen answer tokens
          rejected_logps  : [B]  per-sample sum of log-probs on rejected answer tokens
          chosen_logits   : [B, T+S_c, V]  full logits for chosen
          rejected_logits : [B, T+S_r, V]  full logits for rejected
        """
        B, T = frames.shape[:2]

        # ---- Step 0: Encode all frames (DINO, shared across chosen & rejected) ----
        kv_cache, patch_features, mask_flat = self._encode_all_frames(frames, frame_mask)

        # ---- Coarse pass (visual tokens ONLY — text invisible in causal attn) ----
        q_static = self.q_static.expand(B, -1)                          # [B, qd]
        z_coarse = self._query_all_frames(q_static, kv_cache, B, T, mask_flat, patch_features)
        z_coarse_llm = self._project_visual(z_coarse)                    # [B, T, ld]

        # Coarse LLM: visual tokens only (T, not T+S_c). Causal attention means
        # visual positions never see text, so this is mathematically identical.
        out_coarse = self.llm.model(inputs_embeds=z_coarse_llm)
        h_coarse = out_coarse.last_hidden_state                          # [B, T, ld]

        # Extract dynamic queries from visual positions
        queries = self.llm_to_query(h_coarse)                            # [B, T, qd]

        q_init = self.q_init.expand(B, 1, -1)
        shifted_queries = torch.cat([q_init, queries[:, :-1]], dim=1)    # [B, T, qd]

        # ---- Fine pass: shared visual features ----
        z_fine = self._query_all_frames_batched(shifted_queries, kv_cache, B, T, mask_flat, patch_features)
        z_fine_llm = self._project_visual(z_fine)                        # [B, T, ld]

        # ---- Forward on CHOSEN (lm_head on text positions only) ----
        text_embeds_chosen = self._embed_text(chosen_input_ids)          # [B, S_c, ld]
        seq_chosen = torch.cat([z_fine_llm, text_embeds_chosen], dim=1)  # [B, T+S_c, ld]
        out_chosen = self.llm.model(inputs_embeds=seq_chosen)
        chosen_logits = self.llm.lm_head(out_chosen.last_hidden_state[:, T:, :])  # [B, S_c, V]

        # ---- Forward on REJECTED (lm_head on text positions only) ----
        text_embeds_rejected = self._embed_text(rejected_input_ids)      # [B, S_r, ld]
        seq_rejected = torch.cat([z_fine_llm, text_embeds_rejected], dim=1)
        out_rejected = self.llm.model(inputs_embeds=seq_rejected)
        rejected_logits = self.llm.lm_head(out_rejected.last_hidden_state[:, T:, :])  # [B, S_r, V]

        # ---- Compute per-token log-probs ----
        chosen_logps = self._sequence_logprobs(
            chosen_logits, chosen_input_ids, chosen_loss_mask,
        )
        rejected_logps = self._sequence_logprobs(
            rejected_logits, rejected_input_ids, rejected_loss_mask,
        )

        return {
            "chosen_logps": chosen_logps,       # [B]
            "rejected_logps": rejected_logps,   # [B]
            "chosen_logits": chosen_logits,     # [B, S_c, V]
            "rejected_logits": rejected_logits, # [B, S_r, V]
        }

    def _sequence_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample sum of log-probabilities on answer tokens.

        logits    : [B, S, V]  text-only logits (visual positions excluded)
        input_ids : [B, S]     text token ids
        loss_mask : [B, S]     1.0 for answer tokens, 0.0 otherwise

        Returns   : [B]        sum of log-probs per sample
        """
        B, S = input_ids.shape

        # Shift for autoregressive prediction
        shift_logits = logits[:, :-1, :]                              # [B, S-1, V]
        shift_labels = input_ids[:, 1:]                               # [B, S-1]
        shift_mask = loss_mask[:, 1:]                                 # [B, S-1]

        # Per-token log-probs: log_softmax then gather the label's prob
        log_probs = F.log_softmax(shift_logits, dim=-1)              # [B, S-1, V]
        per_token_logps = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)                                                 # [B, S-1]

        # Mask and sum per sample
        per_token_logps = per_token_logps * shift_mask                # zero out non-answer tokens
        return per_token_logps.sum(dim=-1)                            # [B]

    # ------------------------------------------------------------------
    # Forward mode 2: Coarse only (FAST EVAL)
    # ------------------------------------------------------------------

    def forward_coarse_only(
        self,
        frames: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-pass coarse forward (q_static only, no fine queries).

        Used for:
          - Training A6 ablation (coarse-only training)
          - Fast eval (wrap in torch.no_grad() externally)

        q_static -> all frames -> z_coarse -> LLM -> logits.

        Parameters
        ----------
        frames         : [B, T, 3, 224, 224]
        input_ids      : [B, S]  (optional, for loss computation)
        attention_mask : [B, S]  (optional)
        loss_mask      : [B, S]  (optional)

        Returns
        -------
        dict with keys: logits, and optionally loss
        """
        B, T = frames.shape[:2]

        kv_cache, patch_features, mask_flat = self._encode_all_frames(frames, frame_mask)

        q_static = self.q_static.expand(B, -1)
        z_coarse = self._query_all_frames(q_static, kv_cache, B, T, mask_flat, patch_features)
        z_coarse_llm = self._project_visual(z_coarse)

        if input_ids is not None:
            text_embeds = self._embed_text(input_ids)
            seq = torch.cat([z_coarse_llm, text_embeds], dim=1)
        else:
            seq = z_coarse_llm
        # dtype handled by autocast on GPU; float32 on CPU

        out = self.llm.model(inputs_embeds=seq)
        h = out.last_hidden_state                         # [B, T+S, ld]

        if input_ids is not None:
            S = input_ids.shape[1]
            pad_id = self._get_pad_token_id()
            visual_pad = torch.full(
                (B, T), pad_id, dtype=input_ids.dtype, device=input_ids.device,
            )
            full_labels = torch.cat([visual_pad, input_ids], dim=1)

            if loss_mask is not None:
                visual_no_loss = torch.zeros(
                    B, T, dtype=loss_mask.dtype, device=loss_mask.device,
                )
                full_loss_mask = torch.cat([visual_no_loss, loss_mask], dim=1)
            elif attention_mask is not None:
                visual_no_loss = torch.zeros(
                    B, T, dtype=attention_mask.dtype, device=attention_mask.device,
                )
                full_loss_mask = torch.cat([visual_no_loss, attention_mask], dim=1)
            else:
                full_loss_mask = None

            if self.use_fused_ce and self.training:
                # Fused CE: skip lm_head, never materializes [B, T+S, V]
                loss = self._fused_ce_loss(h, full_labels, full_loss_mask)
                logits = None
            else:
                logits = self.llm.lm_head(h)
                loss = self._ce_loss(logits, full_labels, full_loss_mask)

            result: Dict[str, torch.Tensor] = {"logits": logits, "loss": loss}
            result["coarse_loss"] = loss
            result["fine_loss"] = torch.tensor(0.0, device=frames.device)
        else:
            logits = self.llm.lm_head(h)
            result: Dict[str, torch.Tensor] = {"logits": logits}

        return result

    # ------------------------------------------------------------------
    # Forward mode 3: Autoregressive (TRUE INFERENCE)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_autoregressive(
        self,
        frames: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        True autoregressive inference: sequential frame-by-frame with KV cache.

        q_init -> frame_1 -> z_1 -> LLM -> q_1 -> frame_2 -> z_2 -> ...

        No coarse pass. Each query is derived from the LLM hidden state after
        processing the *previous* fine visual token -- exactly what happens at
        real inference time.

        Parameters
        ----------
        frames         : [B, T, 3, 224, 224]
        input_ids      : [B, S]  (optional, for loss computation)
        attention_mask : [B, S]  (optional)
        loss_mask      : [B, S]  (optional)

        Returns
        -------
        dict with keys: logits, and optionally loss
        """
        B, T = frames.shape[:2]
        device = frames.device

        # Encode all frames with DINO up front (this is OK -- DINO encoding
        # does not depend on the query, only query_attend does).
        kv_cache, patch_features, mask_flat = self._encode_all_frames(frames, frame_mask)

        # Enable KV cache on the LLM for incremental decoding
        orig_use_cache = self.llm.config.use_cache
        self.llm.config.use_cache = True

        query = self.q_init.expand(B, -1)    # [B, qd]
        llm_past_kv = None

        for t in range(T):
            # Foveated extraction with current query
            frame_kv = self._extract_frame_kv(kv_cache, mask_flat, B, T, t)
            z_t = self.encoder.query_attend(query, frame_kv)  # [B, dd]
            z_t_llm = self._project_visual(z_t.unsqueeze(1))            # [B,1,ld]
            # dtype handled by autocast on GPU; float32 on CPU

            # Incremental LLM forward (one visual token at a time)
            out = self.llm.model(
                inputs_embeds=z_t_llm,
                past_key_values=llm_past_kv,
                use_cache=True,
            )
            llm_past_kv = out.past_key_values

            # Derive query for the NEXT frame from the current hidden state
            if t < T - 1:
                h_t = out.last_hidden_state[:, -1, :]   # [B, ld]
                query = self.llm_to_query(h_t)                   # [B, qd]

        # ---- Now process text (if provided) using the accumulated KV cache ----
        if input_ids is not None:
            text_embeds = self._embed_text(input_ids)  # [B, S, ld]

            out_text = self.llm.model(
                inputs_embeds=text_embeds,
                past_key_values=llm_past_kv,
                use_cache=False,
            )
            # Combine visual hidden states (already in KV cache) with text states
            # for logit computation. We only need logits over the text portion
            # (plus the last visual token which predicts the first text token).
            #
            # The KV cache holds T visual positions; out_text.last_hidden_state
            # holds S text positions.  We reconstruct the full logits as
            # [visual_logits, text_logits] but only compute loss on text.
            h_text = out_text.last_hidden_state         # [B, S, ld]
            logits_text = self.llm.lm_head(h_text)      # [B, S, V]

            # For the loss we also need the logit at the last visual position
            # (it predicts the first text token).  Re-derive it:
            h_last_visual = out.last_hidden_state[:, -1:, :]   # [B,1,ld]
            logits_last_v = self.llm.lm_head(h_last_visual)    # [B,1,V]

            # Full logits over [last_visual, text] = [B, 1+S, V]
            logits = torch.cat([logits_last_v, logits_text], dim=1)

            # Labels: [pad_for_last_visual, input_ids]
            pad_id = self._get_pad_token_id()
            lv_pad = torch.full(
                (B, 1), pad_id, dtype=input_ids.dtype, device=device,
            )
            full_labels = torch.cat([lv_pad, input_ids], dim=1)

            # Loss mask
            if loss_mask is not None:
                lv_no_loss = torch.zeros(
                    B, 1, dtype=loss_mask.dtype, device=device,
                )
                full_loss_mask = torch.cat([lv_no_loss, loss_mask], dim=1)
            elif attention_mask is not None:
                lv_no_loss = torch.zeros(
                    B, 1, dtype=attention_mask.dtype, device=device,
                )
                full_loss_mask = torch.cat([lv_no_loss, attention_mask], dim=1)
            else:
                full_loss_mask = None

            loss = self._ce_loss(logits, full_labels, full_loss_mask)

            self.llm.config.use_cache = orig_use_cache
            return {"loss": loss, "logits": logits}

        else:
            # No text -- just return logits at the last visual position
            h_last = out.last_hidden_state   # [B, 1, ld]
            logits = self.llm.lm_head(h_last)
            self.llm.config.use_cache = orig_use_cache
            return {"logits": logits}

    # ------------------------------------------------------------------
    # Convenience: unified forward dispatching by name
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        mode: str = "coarse_fine",
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward entry point.

        Parameters
        ----------
        frames : Tensor [B, T, 3, 224, 224]
            Preprocessed video frames (DINOv2 normalization).
            For **video**: T = number of sampled frames (1-64).
            For **images**: replicate the single frame to T=8 to match training
            distribution (``frame.unsqueeze(0).repeat(8, 1, 1, 1)``).
            The model was trained with ``replicate_image_frames: 8`` in
            Stages 2-3, so single-frame image input will produce degraded
            results.
        input_ids : Tensor [B, S]
            Tokenized text (prompt + response).
        attention_mask : Tensor [B, S]
            1 for real tokens, 0 for padding.
        loss_mask : Tensor [B, S], optional
            1 for tokens that contribute to loss, 0 to skip.
        frame_mask : Tensor [B, T] bool, optional
            True for real frames, False for padding (for variable-length batches).
        mode : str
            "coarse_fine"    — two-pass parallel forward (recommended, uses foveation)
            "coarse_only"    — single static-query pass (fastest, no foveation)
            "autoregressive" — sequential inference with KV cache
        """
        if mode == "coarse_fine":
            return self.forward_coarse_fine(frames, input_ids, attention_mask, loss_mask, frame_mask)
        elif mode == "coarse_only":
            return self.forward_coarse_only(frames, input_ids, attention_mask, loss_mask, frame_mask)
        elif mode == "autoregressive":
            return self.forward_autoregressive(frames, input_ids, attention_mask, loss_mask, frame_mask)
        else:
            raise ValueError(
                f"Unknown forward mode '{mode}'. "
                "Expected one of: coarse_fine, coarse_only, autoregressive"
            )

    # ------------------------------------------------------------------
    # CUDA Stream Prefetch — overlap DINO encoding with LLM backward
    # ------------------------------------------------------------------

    def prefetch_dino(self, frames: torch.Tensor, frame_mask=None, stream=None):
        """
        Start DINO encoding on a separate CUDA stream.

        Call this while the previous batch's backward pass is running.
        The DINO encoder is frozen during training, so there's no gradient
        dependency between the backward pass and this prefetch.

        Args:
            frames: [B, T, 3, 224, 224] next batch's frames
            frame_mask: [B, T] bool, optional
            stream: torch.cuda.Stream to run on (caller manages lifecycle)

        Returns:
            None — results are stored internally and retrieved via
            forward_coarse_fine(..., prefetched_dino=True).
        """
        if stream is None:
            stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                self._prefetched_dino = self._encode_all_frames(frames, frame_mask)
                self._prefetch_stream = stream

    def _get_prefetched_dino(self):
        """Retrieve and clear prefetched DINO results, synchronizing the stream."""
        if hasattr(self, '_prefetched_dino') and self._prefetched_dino is not None:
            self._prefetch_stream.synchronize()
            result = self._prefetched_dino
            self._prefetched_dino = None
            self._prefetch_stream = None
            return result
        return None

    # ------------------------------------------------------------------
    # Utility methods for external callers (train.py, eval.py)
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(
        self, llm_only: bool = False, use_reentrant: bool = True,
    ) -> None:
        """Turn on activation checkpointing for LLM (and optionally DINO).

        Args:
            llm_only: If True, only enable for LLM backbone. Leave DINO
                      un-checkpointed so it can be safely torch.compiled.
                      DINO is small (22M params) so checkpointing saves
                      little memory there.
            use_reentrant: If False, use non-reentrant checkpointing which
                          is compatible with torch.compile (the reentrant
                          version causes NaN with compile). Default True
                          for backward compat; set False when using compile.
        """
        ckpt_kwargs = {"use_reentrant": use_reentrant}
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=ckpt_kwargs
        )
        if not llm_only and hasattr(self.encoder.dino, 'gradient_checkpointing_enable'):
            self.encoder.dino.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=ckpt_kwargs
            )

    def get_param_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_connector: float = 1e-4,
    ) -> list:
        """
        Return parameter groups with differential learning rates.

        Groups:
          1. Connector (dino_to_llm, llm_to_query, q_static, q_init) -- highest LR
          2. DINO encoder -- backbone LR
          3. LLM -- backbone LR

        This is a suggestion; train.py may override.
        """
        connector_params = set()
        for name, param in self.named_parameters():
            if any(k in name for k in [
                "dino_to_llm", "llm_to_query", "q_static", "q_init",
                "query_input_proj", "query_output_proj",
            ]):
                connector_params.add(id(param))

        encoder_params = set()
        for name, param in self.encoder.named_parameters():
            if id(param) not in connector_params:
                encoder_params.add(id(param))

        groups = [
            {
                "params": [p for p in self.parameters()
                           if id(p) in connector_params and p.requires_grad],
                "lr": lr_connector,
                "name": "connector",
            },
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if id(p) in encoder_params and p.requires_grad],
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
