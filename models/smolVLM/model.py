"""Combined vision-language model closely tracking the SmolVLM layout."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.smolLM2 import SmolLM2

from .config import SmolVLMConfig
from .projector import SmolVLMMultiModalProjector
from .vision import SmolVLMVisionEncoder


class SmolVLM(nn.Module):
    """Minimal reproduction of the SmolVLM architecture."""

    def __init__(self, cfg: SmolVLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = SmolVLMVisionEncoder(cfg.vision)
        vision_dim = cfg.vision.hidden_size * (cfg.scale_factor ** 2)
        self.mm_projector = SmolVLMMultiModalProjector(
            vision_dim=vision_dim,
            text_dim=cfg.language.hidden_size,
            scale_factor=cfg.scale_factor,
            hidden_dim=cfg.mm_hidden_size,
        )
        self.language_model = SmolLM2(cfg.to_language_config())
        # Initialise the vision + projector layers with the same Gaussian scheme as the text model.
        self.vision_encoder.apply(self._init_weights)
        self.mm_projector.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- utils
    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------- features
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode images into the language embedding space."""

        batch, num_images, channels, height, width = pixel_values.shape
        dtype = self.language_model.tok_emb.weight.dtype
        device = pixel_values.device

        flat_pixels = pixel_values.to(dtype=dtype).view(batch * num_images, channels, height, width)
        nb_values_per_image = flat_pixels.shape[1:].numel()
        real_images_mask = (flat_pixels == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        flat_pixels = flat_pixels[real_images_mask].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(flat_pixels.size(0), height, width),
                dtype=torch.bool,
                device=device,
            )
        else:
            pixel_attention_mask = pixel_attention_mask.view(batch * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_mask].contiguous()

        if flat_pixels.numel() == 0:
            return torch.zeros((0, 0, self.cfg.language.hidden_size), device=device, dtype=dtype)

        patch_size = self.cfg.vision.patch_size
        patches = pixel_attention_mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patch_attention_mask = (patches.sum(dim=(-1, -2)) > 0)

        vision_hidden = self.vision_encoder(flat_pixels, patch_attention_mask=patch_attention_mask)
        projected = self.mm_projector(vision_hidden)
        return projected.to(dtype=dtype)

    def _merge_image_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if image_hidden_states.numel() == 0:
            return inputs_embeds
        image_hidden_states = image_hidden_states.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        special_mask = (input_ids == self.cfg.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if int(special_mask.sum().item()) != image_hidden_states.numel():
            raise ValueError("Mismatch between number of <image> tokens and image hidden states")
        merged = inputs_embeds.masked_scatter(special_mask, image_hidden_states.reshape(-1))
        return merged.view_as(inputs_embeds)

    def _run_language_model(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_len, _ = inputs_embeds.shape
        attn_mask, pos_ids = self.language_model._prepare_attention_inputs(
            attention_mask,
            position_ids,
            batch=batch,
            seq_len=seq_len,
            device=inputs_embeds.device,
        )
        hidden_states = self.language_model.dropout(inputs_embeds)
        for block in self.language_model.blocks:
            hidden_states = block(hidden_states, attn_mask=attn_mask, position_ids=pos_ids)
        hidden_states = self.language_model.norm_out(hidden_states)
        return self.language_model.lm_head(hidden_states)

    # -------------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_hidden_states is not None and pixel_values is not None:
            raise ValueError("Provide either pixel_values or precomputed image_hidden_states, not both")
        if image_hidden_states is None and pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)

        inputs_embeds = self.language_model.tok_emb(input_ids)
        if image_hidden_states is not None:
            inputs_embeds = self._merge_image_embeddings(input_ids, inputs_embeds, image_hidden_states)
        return self._run_language_model(inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)

    # ------------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        image_hidden_states = None
        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(
                generated,
                image_hidden_states=image_hidden_states,
            )[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
        return generated

    # --------------------------------------------------------------------- loading
    def load_hf_state_dict(
        self,
        hf_state: Dict[str, torch.Tensor],
        *,
        strict: bool = False,
        verbose: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        ref_param = next(self.parameters(), None)
        if dtype is None:
            dtype = ref_param.dtype if ref_param is not None else torch.float32
        if device is None:
            device = ref_param.device if ref_param is not None else torch.device("cpu")

        used_keys = set()
        missing = []
        mismatched = []

        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        def fetch(*names: str, mark_only: bool = False) -> Optional[torch.Tensor]:
            for name in names:
                if name in hf_state:
                    used_keys.add(name)
                    if mark_only:
                        return None
                    return hf_state[name].to(device=device, dtype=dtype)
            return None

        def assign(target_name: str, value: Optional[torch.Tensor]) -> None:
            if value is None:
                missing.append(target_name)
                return
            target = params.get(target_name)
            if target is None:
                target = buffers.get(target_name)
            if target is None:
                missing.append(target_name)
                return
            if tuple(target.shape) != tuple(value.shape):
                mismatched.append((target_name, tuple(target.shape), tuple(value.shape)))
                return
            target.data.copy_(value)

        # --- resolve prefixes -------------------------------------------------
        state_keys = list(hf_state.keys())

        def prefix_for(candidates):
            for prefix in candidates:
                if any(key.startswith(prefix) for key in state_keys):
                    return prefix
            return None

        vision_prefix = prefix_for([
            "model.vision_model",
            "vision_model",
        ])
        projector_prefix = prefix_for([
            "model.connector.modality_projection",
            "connector.modality_projection",
            "model.multi_modal_projector",
            "multi_modal_projector",
        ])
        text_prefix = prefix_for([
            "model.language_model.model",
            "model.text_model",
            "language_model.model",
            "text_model",
        ])

        # --- vision embeddings ------------------------------------------------
        if vision_prefix:
            assign("vision_encoder.patch_embed.weight", fetch(f"{vision_prefix}.embeddings.patch_embedding.weight"))
            assign("vision_encoder.patch_embed.bias", fetch(f"{vision_prefix}.embeddings.patch_embedding.bias"))
            assign("vision_encoder.pos_embed", fetch(f"{vision_prefix}.embeddings.position_embedding.weight"))
            cls_token = fetch(f"{vision_prefix}.embeddings.class_embedding")
            if cls_token is not None:
                assign("vision_encoder.cls_token", cls_token)
            assign("vision_encoder.ln_post.weight", fetch(f"{vision_prefix}.post_layernorm.weight"))
            assign("vision_encoder.ln_post.bias", fetch(f"{vision_prefix}.post_layernorm.bias"))

            for idx in range(len(self.vision_encoder.blocks)):
                block = f"vision_encoder.blocks.{idx}"
                assign(f"{block}.ln1.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.layer_norm1.weight"))
                assign(f"{block}.ln1.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.layer_norm1.bias"))
                assign(f"{block}.ln2.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.layer_norm2.weight"))
                assign(f"{block}.ln2.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.layer_norm2.bias"))

                assign(f"{block}.attn.q_proj.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.q_proj.weight"))
                assign(f"{block}.attn.q_proj.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.q_proj.bias"))
                assign(f"{block}.attn.k_proj.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.k_proj.weight"))
                assign(f"{block}.attn.k_proj.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.k_proj.bias"))
                assign(f"{block}.attn.v_proj.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.v_proj.weight"))
                assign(f"{block}.attn.v_proj.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.v_proj.bias"))
                assign(f"{block}.attn.o_proj.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.out_proj.weight"))
                assign(f"{block}.attn.o_proj.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.self_attn.out_proj.bias"))

                assign(f"{block}.mlp.fc1.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.mlp.fc1.weight"))
                assign(f"{block}.mlp.fc1.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.mlp.fc1.bias"))
                assign(f"{block}.mlp.fc2.weight", fetch(f"{vision_prefix}.encoder.layers.{idx}.mlp.fc2.weight"))
                assign(f"{block}.mlp.fc2.bias", fetch(f"{vision_prefix}.encoder.layers.{idx}.mlp.fc2.bias"))

        # --- projector --------------------------------------------------------
        if projector_prefix:
            if hasattr(self.mm_projector, "proj"):
                assign("mm_projector.proj.weight", fetch(f"{projector_prefix}.proj.weight", f"{projector_prefix}.linear_2.weight"))
                bias = fetch(f"{projector_prefix}.proj.bias", f"{projector_prefix}.linear_2.bias")
                if bias is not None:
                    assign("mm_projector.proj.bias", bias)
            else:
                assign("mm_projector.linear1.weight", fetch(f"{projector_prefix}.linear_1.weight"))
                assign("mm_projector.linear1.bias", fetch(f"{projector_prefix}.linear_1.bias"))
                assign("mm_projector.linear2.weight", fetch(f"{projector_prefix}.linear_2.weight"))
                assign("mm_projector.linear2.bias", fetch(f"{projector_prefix}.linear_2.bias"))

        # --- language ---------------------------------------------------------
        if text_prefix:
            assign("language_model.tok_emb.weight", fetch(f"{text_prefix}.embed_tokens.weight"))
            assign("language_model.norm_out.weight", fetch(f"{text_prefix}.norm.weight", f"{text_prefix}.model.norm.weight"))

            for idx in range(len(self.language_model.blocks)):
                block = f"language_model.blocks.{idx}"
                assign(f"{block}.ln_attn.weight", fetch(f"{text_prefix}.layers.{idx}.input_layernorm.weight"))
                assign(f"{block}.ln_mlp.weight", fetch(f"{text_prefix}.layers.{idx}.post_attention_layernorm.weight"))

                assign(f"{block}.attn.q_proj.weight", fetch(f"{text_prefix}.layers.{idx}.self_attn.q_proj.weight"))
                assign(f"{block}.attn.k_proj.weight", fetch(f"{text_prefix}.layers.{idx}.self_attn.k_proj.weight"))
                assign(f"{block}.attn.v_proj.weight", fetch(f"{text_prefix}.layers.{idx}.self_attn.v_proj.weight"))
                assign(f"{block}.attn.o_proj.weight", fetch(f"{text_prefix}.layers.{idx}.self_attn.o_proj.weight"))

                gate = fetch(f"{text_prefix}.layers.{idx}.mlp.gate_proj.weight", f"{text_prefix}.layers.{idx}.mlp.w1.weight")
                up = fetch(f"{text_prefix}.layers.{idx}.mlp.up_proj.weight", f"{text_prefix}.layers.{idx}.mlp.w3.weight")
                combined = fetch(f"{text_prefix}.layers.{idx}.mlp.fc1.weight")
                mlp_in = self._stack_gate_up(gate, up, combined)
                assign(f"{block}.mlp.in_proj.weight", mlp_in)
                assign(f"{block}.mlp.out_proj.weight", fetch(f"{text_prefix}.layers.{idx}.mlp.down_proj.weight", f"{text_prefix}.layers.{idx}.mlp.w2.weight"))

            assign("language_model.lm_head.weight", fetch("lm_head.weight", "model.lm_head.weight"))

        unused = [key for key in hf_state if key not in used_keys]
        if verbose:
            print(f"[load_hf] copied tensors: {len(used_keys)}")
            if missing:
                print(f"[load_hf] missing target params: {len(missing)} (up to 10) -> {missing[:10]}")
            if mismatched:
                print(f"[load_hf] shape mismatches: {len(mismatched)} (first) -> {mismatched[0]}")
            if unused:
                print(f"[load_hf] unused source tensors: {len(unused)} (up to 10) -> {unused[:10]}")

        if strict and (missing or mismatched):
            raise RuntimeError("[load_hf strict] missing or mismatched parameters")

    @staticmethod
    def _stack_gate_up(
        gate: Optional[torch.Tensor],
        up: Optional[torch.Tensor],
        combined: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if gate is not None and up is not None:
            return torch.cat([gate, up], dim=0)
        return combined

    # ---------------------------------------------------------------- factory
    @staticmethod
    def from_config_dict(cfg_dict: Dict) -> "SmolVLM":
        return SmolVLM(SmolVLMConfig(**cfg_dict))

