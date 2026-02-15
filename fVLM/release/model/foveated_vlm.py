"""
Foveated Vision-Language Model (release implementation).

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
    ):
        super().__init__()

        # ---- delayed import so encoder.py can live next to this file ----
        from release.model.encoder import FoveatedEncoder

        # ---- Vision encoder (DINOv2 + query cross-attention) ----
        self.encoder = FoveatedEncoder(
            dino_model_name=dino_name,
            query_dim=query_dim,
            output_dim=None,  # output_dim = dino_dim by default inside encoder
        )
        dino_dim = self.encoder.dino_dim

        # ---- Language model ----
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, attn_implementation="sdpa", dtype=torch.float32,
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

    def _encode_all_frames(self, frames: torch.Tensor):
        """
        Run DINO patch encoding for every frame in the batch.

        frames : [B, T, 3, 224, 224]
        Returns list[T] of per-frame kv_cache lists (one (K,V) tuple per layer).
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # encode_patches returns (patch_features, kv_cache)
        # patch_features: [B*T, N+1, D]
        # kv_cache: list of (K, V) tuples, one per DINO layer
        #   each K, V: [B*T, N+1, D]
        _, kv_cache_flat = self.encoder.encode_patches(frames_flat)

        N_plus_1 = kv_cache_flat[0][0].shape[1]
        D = kv_cache_flat[0][0].shape[2]
        num_layers = len(kv_cache_flat)

        # Reshape from [B*T, ...] to [B, T, ...] and split per frame
        per_frame_caches = []
        for t in range(T):
            frame_kv = []
            for li in range(num_layers):
                K_all, V_all = kv_cache_flat[li]
                K_bt = K_all.reshape(B, T, N_plus_1, D)
                V_bt = V_all.reshape(B, T, N_plus_1, D)
                frame_kv.append((K_bt[:, t], V_bt[:, t]))
            per_frame_caches.append(frame_kv)

        return per_frame_caches

    def _query_all_frames(
        self, query: torch.Tensor, caches: list,
    ) -> torch.Tensor:
        """
        Apply a single query to every frame (parallel across T).

        query  : [B, query_dim]  (same query for every frame)
        caches : list[T] of per-frame cache dicts
        Returns: [B, T, dino_dim]
        """
        z_list = [self.encoder.query_attend(query, caches[t]) for t in range(len(caches))]
        return torch.stack(z_list, dim=1)

    def _query_all_frames_batched(
        self, queries: torch.Tensor, caches: list,
    ) -> torch.Tensor:
        """
        Apply per-frame queries to every frame (parallel across T).

        queries : [B, T, query_dim]  (different query per frame)
        caches  : list[T] of per-frame cache dicts
        Returns : [B, T, dino_dim]
        """
        T = queries.shape[1]
        z_list = [self.encoder.query_attend(queries[:, t], caches[t]) for t in range(T)]
        return torch.stack(z_list, dim=1)

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

    # ------------------------------------------------------------------
    # Forward mode 1: Coarse+Fine (TRAINING)
    # ------------------------------------------------------------------

    def forward_coarse_fine(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Two-pass parallel training forward.

        Pass 1 (coarse): q_static -> all frames -> z_coarse -> LLM -> dynamic queries
        Pass 2 (fine):   shifted queries -> all frames -> z_fine -> LLM + text -> loss

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
        per_frame_caches = self._encode_all_frames(frames)

        # ---- Pass 1: Coarse ----
        q_static = self.q_static.expand(B, -1)                     # [B, qd]
        z_coarse = self._query_all_frames(q_static, per_frame_caches)  # [B,T,dd]
        z_coarse_llm = self._project_visual(z_coarse)              # [B,T,ld]

        # Build coarse sequence: [visual_coarse, text]
        text_embeds = self._embed_text(input_ids)                  # [B,S,ld]
        seq_coarse = torch.cat([z_coarse_llm, text_embeds], dim=1) # [B,T+S,ld]
        # dtype handled by autocast on GPU; float32 on CPU

        # LLM forward (backbone only, no lm_head yet)
        out_coarse = self.llm.model(inputs_embeds=seq_coarse)
        h_coarse = out_coarse.last_hidden_state                    # [B,T+S,ld]

        # Extract dynamic queries from visual positions
        # h_coarse[:, 0..T-1] are the hidden states at visual token positions
        # Each one generates a query for the corresponding frame
        h_visual_coarse = h_coarse[:, :T, :]                      # [B,T,ld]
        queries = self.llm_to_query(h_visual_coarse)               # [B,T,qd]

        # Shift queries: frame t gets query from frame t-1; frame 0 gets q_init
        q_init = self.q_init.expand(B, 1, -1)                     # [B,1,qd]
        shifted_queries = torch.cat([q_init, queries[:, :-1]], dim=1)  # [B,T,qd]

        # ---- Pass 2: Fine ----
        z_fine = self._query_all_frames_batched(shifted_queries, per_frame_caches)  # [B,T,dd]
        z_fine_llm = self._project_visual(z_fine)                  # [B,T,ld]

        # Build fine sequence: [visual_fine, text]
        seq_fine = torch.cat([z_fine_llm, text_embeds], dim=1)     # [B,T+S,ld]
        # dtype handled by autocast on GPU; float32 on CPU

        out_fine = self.llm.model(inputs_embeds=seq_fine)
        h_fine = out_fine.last_hidden_state                        # [B,T+S,ld]

        # Get logits over the FULL sequence (visual + text positions)
        logits_full = self.llm.lm_head(h_fine)                    # [B,T+S,V]

        # ---- Loss on text portion only ----
        # The text tokens start at position T in the sequence.
        # We need labels aligned with the full sequence: visual positions get pad.
        pad_id = self._get_pad_token_id()
        visual_pad = torch.full(
            (B, T), pad_id, dtype=input_ids.dtype, device=input_ids.device,
        )
        full_labels = torch.cat([visual_pad, input_ids], dim=1)    # [B, T+S]

        # Build full loss mask: 0 for visual positions, then the provided loss_mask
        if loss_mask is not None:
            visual_no_loss = torch.zeros(
                B, T, dtype=loss_mask.dtype, device=loss_mask.device,
            )
            full_loss_mask = torch.cat([visual_no_loss, loss_mask], dim=1)  # [B,T+S]
        else:
            # Default: compute loss on all text positions that are not padding
            visual_no_loss = torch.zeros(B, T, dtype=torch.long, device=frames.device)
            text_loss_mask = attention_mask  # non-pad text positions
            full_loss_mask = torch.cat([visual_no_loss, text_loss_mask], dim=1)

        fine_loss = self._ce_loss(logits_full, full_labels, full_loss_mask)

        # ---- Optional auxiliary coarse loss ----
        coarse_loss = torch.tensor(0.0, device=frames.device)
        if self.lambda_coarse > 0:
            logits_coarse = self.llm.lm_head(h_coarse)
            coarse_loss = self._ce_loss(logits_coarse, full_labels, full_loss_mask)

        # ---- Combined loss ----
        loss = fine_loss + self.lambda_coarse * coarse_loss

        return {
            "loss": loss,
            "fine_loss": fine_loss,
            "coarse_loss": coarse_loss,
            "logits": logits_full,
        }

    # ------------------------------------------------------------------
    # Forward mode 2: Coarse only (FAST EVAL)
    # ------------------------------------------------------------------

    def forward_coarse_only(
        self,
        frames: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
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

        per_frame_caches = self._encode_all_frames(frames)

        q_static = self.q_static.expand(B, -1)
        z_coarse = self._query_all_frames(q_static, per_frame_caches)
        z_coarse_llm = self._project_visual(z_coarse)

        if input_ids is not None:
            text_embeds = self._embed_text(input_ids)
            seq = torch.cat([z_coarse_llm, text_embeds], dim=1)
        else:
            seq = z_coarse_llm
        # dtype handled by autocast on GPU; float32 on CPU

        out = self.llm.model(inputs_embeds=seq)
        logits = self.llm.lm_head(out.last_hidden_state)

        result: Dict[str, torch.Tensor] = {"logits": logits}

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

            loss = self._ce_loss(logits, full_labels, full_loss_mask)
            result["loss"] = loss
            result["coarse_loss"] = loss
            result["fine_loss"] = torch.tensor(0.0, device=frames.device)

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
        per_frame_caches = self._encode_all_frames(frames)

        # Enable KV cache on the LLM for incremental decoding
        orig_use_cache = self.llm.config.use_cache
        self.llm.config.use_cache = True

        query = self.q_init.expand(B, -1)    # [B, qd]
        llm_past_kv = None

        for t in range(T):
            # Foveated extraction with current query
            z_t = self.encoder.query_attend(query, per_frame_caches[t])  # [B, dd]
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
        mode: str = "coarse_fine",
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward entry point.

        mode : "coarse_fine" | "coarse_only" | "autoregressive"
        """
        if mode == "coarse_fine":
            return self.forward_coarse_fine(frames, input_ids, attention_mask, loss_mask)
        elif mode == "coarse_only":
            return self.forward_coarse_only(frames, input_ids, attention_mask, loss_mask)
        elif mode == "autoregressive":
            return self.forward_autoregressive(frames, input_ids, attention_mask, loss_mask)
        else:
            raise ValueError(
                f"Unknown forward mode '{mode}'. "
                "Expected one of: coarse_fine, coarse_only, autoregressive"
            )

    # ------------------------------------------------------------------
    # Utility methods for external callers (train.py, eval.py)
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Turn on activation checkpointing for the LLM backbone."""
        self.llm.gradient_checkpointing_enable()

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
