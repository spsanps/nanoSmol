"""
SmolVLM2-based Video Model with Multi-task Training

Objectives:
1. Video → Text (captioning) - uses existing lm_head
2. Text/Video → Video (generation/prediction) - new VAE decoder head

Based on SmolVLM2-256M-Video-Instruct which already has vision-language alignment.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from diffusers import AutoencoderKL


class VAEDecoderHead(nn.Module):
    """Projects LLM hidden states to VAE latent space for video generation."""

    def __init__(self, hidden_size: int = 576, latent_channels: int = 4, latent_size: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        # Project from LLM hidden to VAE latent
        # Output: (B, latent_channels, latent_size, latent_size) = (B, 4, 32, 32) for 256px
        latent_dim = latent_channels * latent_size * latent_size  # 4 * 32 * 32 = 4096

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, latent_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, hidden_size) - pooled LLM output
        Returns:
            latents: (B, 4, 64, 64) - VAE latent space
        """
        x = self.proj(hidden_states)
        return x.view(-1, self.latent_channels, self.latent_size, self.latent_size)


class SmolVLMVideo(nn.Module):
    """
    SmolVLM2 extended for video understanding and generation.

    Tasks:
    - Captioning: video frames → text (forward_caption)
    - Generation: text + context → VAE latents (forward_generate)
    - Prediction: video frames → next frame latents (forward_predict)
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        vae_name: str = "stabilityai/sd-vae-ft-mse",
        freeze_vision: bool = False,
        freeze_connector: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        # Load SmolVLM2
        print(f"Loading {model_name}...")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Get hidden size
        self.hidden_size = self.vlm.config.text_config.hidden_size

        # Load VAE for decoding (frozen)
        print(f"Loading VAE {vae_name}...")
        self.vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.bfloat16)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # VAE decoder head for video generation (match VLM dtype)
        # latent_size=32 for 256x256 images, 64 for 512x512
        self.vae_head = VAEDecoderHead(
            hidden_size=self.hidden_size,
            latent_channels=4,
            latent_size=32,  # For 256x256 images
        ).to(torch.bfloat16)

        # Optionally freeze parts
        if freeze_vision:
            self.vlm.model.vision_model.requires_grad_(False)
        if freeze_connector:
            self.vlm.model.connector.requires_grad_(False)

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.vlm.model.text_model.gradient_checkpointing_enable()

    def forward_caption(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Video → Text captioning.

        Args:
            pixel_values: (B, num_frames, C, H, W) video frames
            input_ids: (B, seq_len) text tokens
            attention_mask: (B, seq_len) attention mask
            labels: (B, seq_len) target tokens for loss

        Returns:
            dict with loss and logits
        """
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def forward_generate(
        self,
        pixel_values: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        target_latents: torch.Tensor = None,
    ) -> dict:
        """
        Text/Video → VAE latents for video generation.

        Args:
            pixel_values: (B, num_frames, C, H, W) optional context frames
            input_ids: (B, seq_len) text tokens
            attention_mask: (B, seq_len)
            target_latents: (B, 4, 64, 64) target VAE latents

        Returns:
            dict with loss and predicted latents
        """
        # Get LLM hidden states
        outputs = self.vlm.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Pool last hidden state (use last token or mean)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, -1, :]  # Last token

        # Predict latents
        pred_latents = self.vae_head(pooled)

        result = {"pred_latents": pred_latents}

        if target_latents is not None:
            loss = nn.functional.mse_loss(pred_latents, target_latents)
            result["loss"] = loss

        return result

    def forward_predict(
        self,
        pixel_values: torch.Tensor,
        target_latents: torch.Tensor = None,
    ) -> dict:
        """
        Video frames → Next frame VAE latents.

        Args:
            pixel_values: (B, num_frames, C, H, W) context frames
            target_latents: (B, 4, 64, 64) next frame latents

        Returns:
            dict with loss and predicted latents
        """
        # Create simple prompt for prediction task
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # Use a prediction prompt
        prompt = "Predict the next frame:"
        inputs = self.processor(text=[prompt] * batch_size, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        return self.forward_generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_latents=target_latents,
        )

    def forward(
        self,
        pixel_values: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        target_latents: torch.Tensor = None,
        task: str = "caption",
    ) -> dict:
        """
        Unified forward for multi-task training.

        Args:
            task: "caption", "generate", or "predict"
        """
        if task == "caption":
            return self.forward_caption(pixel_values, input_ids, attention_mask, labels)
        elif task == "generate":
            return self.forward_generate(pixel_values, input_ids, attention_mask, target_latents)
        elif task == "predict":
            return self.forward_predict(pixel_values, target_latents)
        else:
            raise ValueError(f"Unknown task: {task}")

    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate captions for video frames."""
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # Create prompt
        prompt = "Describe this video:"
        inputs = self.processor(text=[prompt] * batch_size, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Generate
        outputs = self.vlm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Decode
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return captions

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to images."""
        # VAE expects float32
        latents = latents.float()
        images = self.vae.decode(latents).sample
        # Clamp to valid range
        images = torch.clamp(images, -1, 1)
        return images


def test_model():
    """Quick test of the model."""
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SmolVLMVideo(
        freeze_vision=False,
        freeze_connector=False,
        gradient_checkpointing=True,
    )
    model = model.to(device)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.1f}M")
    print(f"Trainable: {trainable/1e6:.1f}M")

    # Test forward pass
    batch_size = 2
    num_frames = 4

    # Dummy inputs (256x256 images -> 32x32 latents)
    pixel_values = torch.randn(batch_size, num_frames, 3, 256, 256, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 1000, (batch_size, 20), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    target_latents = torch.randn(batch_size, 4, 32, 32, device=device, dtype=torch.bfloat16)

    print("\n=== Testing Caption Task ===")
    start = time.time()
    out = model(pixel_values, input_ids, attention_mask, labels=labels, task="caption")
    print(f"Caption loss: {out['loss'].item():.4f}, time: {time.time()-start:.2f}s")

    print("\n=== Testing Generate Task ===")
    start = time.time()
    out = model(pixel_values, input_ids, attention_mask, target_latents=target_latents, task="generate")
    print(f"Generate loss: {out['loss'].item():.4f}, time: {time.time()-start:.2f}s")

    print("\n=== Testing Predict Task ===")
    start = time.time()
    out = model.forward_predict(pixel_values, target_latents)
    print(f"Predict loss: {out['loss'].item():.4f}, time: {time.time()-start:.2f}s")

    # Memory usage
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {mem:.2f} GB")


if __name__ == "__main__":
    test_model()
