#!/usr/bin/env python3
"""
Gradio Web Demo for Live Attention Visualization

This creates a web interface that can access your webcam through the browser,
bypassing WSL limitations. Works on any system with a browser.

Run with: python scripts/gradio_attention_demo.py
Then open the URL in your browser (usually http://127.0.0.1:7860)
"""

import torch
import numpy as np
import cv2
import sys
from pathlib import Path
from collections import deque
import time

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class AttentionModel:
    def __init__(self, device='cuda', frame_size=256):
        self.device = device
        self.frame_size = frame_size
        self.frame_buffer = deque(maxlen=8)
        self.model = None
        self.tokenizer = None
        self.last_caption = ""
        self.last_caption_time = 0

    def load(self):
        if self.model is not None:
            return

        print("Loading model...")
        checkpoint = torch.load('outputs/multitask/checkpoints/final.pt',
                                map_location=self.device, weights_only=False)

        self.model = FoveatedVideoModel(
            dino_model='facebook/dinov2-small',
            llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
            dino_dim=384, llm_dim=576, query_dim=128, lambda_coarse=1.0,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded!")

    def preprocess(self, frame):
        """Preprocess frame for model."""
        if frame is None:
            return None, None

        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))

        # Keep BGR for display, convert to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, frame_rgb

    def normalize(self, frames_rgb):
        """Normalize frames for model input."""
        frames = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        return frames.unsqueeze(0).to(self.device)

    def get_attention(self, frames_norm):
        """Compute attention maps."""
        B, T, C, H, W = frames_norm.shape

        with torch.no_grad():
            frame = frames_norm[:, -1]
            patch_features, cache = self.model.encoder.encode_patches(frame)

            # Coarse attention
            q_coarse = self.model.q_static.expand(1, -1)
            q_embed = self.model.encoder.query_input_proj(q_coarse).unsqueeze(1)
            scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
            attn_coarse = torch.softmax(scores / (self.model.encoder.dino_dim ** 0.5), dim=-1)
            attn_coarse = attn_coarse[0, 0, 1:].cpu().numpy()

            # Fine attention with temporal context
            if T >= 2:
                coarse_features = []
                for t in range(T):
                    f = frames_norm[:, t]
                    _, c = self.model.encoder.encode_patches(f)
                    z = self.model.encoder.query_attend(q_coarse, c)
                    coarse_features.append(z)

                coarse_features = torch.stack(coarse_features, dim=1)
                coarse_proj = self.model.dino_to_llm(coarse_features) * self.model.visual_scale

                seq = torch.cat([
                    self.model.get_empty_text_embeds(1),
                    self.model.coarse_token.expand(1, 1, -1),
                    coarse_proj
                ], dim=1)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = self.model.llm(inputs_embeds=seq, output_hidden_states=True)

                h_last = out.hidden_states[-1][:, 1 + T]
                q_fine = self.model.llm_to_query(h_last.float())

                q_embed = self.model.encoder.query_input_proj(q_fine).unsqueeze(1)
                scores = torch.bmm(q_embed, patch_features.transpose(1, 2))
                attn_fine = torch.softmax(scores / (self.model.encoder.dino_dim ** 0.5), dim=-1)
                attn_fine = attn_fine[0, 0, 1:].cpu().numpy()
            else:
                attn_fine = attn_coarse.copy()

        return attn_coarse, attn_fine

    def create_heatmap(self, attn):
        """Create colored heatmap from attention."""
        grid_size = int(np.sqrt(len(attn)))
        attn_2d = attn.reshape(grid_size, grid_size)
        attn_norm = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        heatmap = cv2.resize(attn_norm, (self.frame_size, self.frame_size))
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def create_overlay(self, frame_rgb, attn, alpha=0.5):
        """Create attention overlay."""
        heatmap = self.create_heatmap(attn)
        overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def generate_caption(self, frames_norm):
        """Generate caption."""
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                caption = self.model.generate_caption(
                    frames_norm, self.tokenizer,
                    max_new_tokens=30, temperature=0.7, use_fine=True
                )[0].strip()
        return caption

    def process_frame(self, frame, generate_caption=False):
        """Process a single frame and return visualization."""
        self.load()

        if frame is None:
            return None, None, None, None, "No frame"

        # Handle both BGR and RGB input
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume RGB from gradio
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        frame_display, frame_rgb = self.preprocess(frame_bgr)
        if frame_display is None:
            return None, None, None, None, "Preprocessing failed"

        self.frame_buffer.append(frame_rgb)
        frames = list(self.frame_buffer)

        if len(frames) < 2:
            return frame_rgb, None, None, None, "Buffering frames..."

        frames_norm = self.normalize(frames)
        attn_coarse, attn_fine = self.get_attention(frames_norm)

        heatmap_coarse = self.create_heatmap(attn_coarse)
        heatmap_fine = self.create_heatmap(attn_fine)
        overlay = self.create_overlay(frame_rgb, attn_fine)

        # Stats
        ratio = attn_fine.max() / (attn_coarse.max() + 1e-8)
        stats = f"Coarse max: {attn_coarse.max():.4f}\n"
        stats += f"Fine max: {attn_fine.max():.4f}\n"
        stats += f"Focus ratio: {ratio:.1f}x"

        # Caption
        if generate_caption:
            current_time = time.time()
            if current_time - self.last_caption_time > 2.0:
                self.last_caption = self.generate_caption(frames_norm)
                self.last_caption_time = current_time
            stats += f"\n\nCaption: {self.last_caption}"

        return frame_rgb, heatmap_coarse, heatmap_fine, overlay, stats


# Global model instance
model = AttentionModel()


def process_webcam(frame, enable_caption):
    """Gradio callback for webcam processing."""
    original, coarse, fine, overlay, stats = model.process_frame(frame, enable_caption)
    return original, coarse, fine, overlay, stats


def process_image(image, enable_caption):
    """Gradio callback for image processing."""
    if image is None:
        return None, None, None, None, "No image uploaded"

    # Clear buffer for fresh start
    model.frame_buffer.clear()

    # Process same image multiple times to build context
    for _ in range(4):
        model.process_frame(image, False)

    return model.process_frame(image, enable_caption)


def create_demo():
    """Create Gradio interface."""

    with gr.Blocks(title="Foveated VLM Attention Demo") as demo:
        gr.Markdown("""
        # 🎯 Foveated VLM Live Attention Demo

        This demo shows real-time attention maps from the Foveated Vision-Language Model.

        - **Coarse Attention**: Static query - where the model looks by default
        - **Fine Attention**: Dynamic query - where the model focuses based on temporal context
        - **Overlay**: Fine attention overlaid on the original frame

        **Higher focus ratio = model is attending more selectively**
        """)

        with gr.Tabs():
            # Webcam tab
            with gr.TabItem("📷 Live Webcam"):
                with gr.Row():
                    webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")

                caption_toggle = gr.Checkbox(label="Generate Caption (slower)", value=False)

                with gr.Row():
                    out_original = gr.Image(label="Original")
                    out_coarse = gr.Image(label="Coarse Attention")
                    out_fine = gr.Image(label="Fine Attention")
                    out_overlay = gr.Image(label="Fine Overlay")

                stats_output = gr.Textbox(label="Statistics", lines=5)

                webcam.stream(
                    fn=process_webcam,
                    inputs=[webcam, caption_toggle],
                    outputs=[out_original, out_coarse, out_fine, out_overlay, stats_output]
                )

            # Image upload tab
            with gr.TabItem("🖼️ Upload Image"):
                with gr.Row():
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    process_btn = gr.Button("Process", variant="primary")

                caption_toggle_img = gr.Checkbox(label="Generate Caption", value=False)

                with gr.Row():
                    img_original = gr.Image(label="Original")
                    img_coarse = gr.Image(label="Coarse Attention")
                    img_fine = gr.Image(label="Fine Attention")
                    img_overlay = gr.Image(label="Fine Overlay")

                img_stats = gr.Textbox(label="Statistics", lines=5)

                process_btn.click(
                    fn=process_image,
                    inputs=[image_input, caption_toggle_img],
                    outputs=[img_original, img_coarse, img_fine, img_overlay, img_stats]
                )

        gr.Markdown("""
        ---
        **Model**: Foveated VLM (DINOv2-small + SmolLM2-135M)
        **Training**: Multi-task (reconstruction + captioning) on WebVid

        *Press the webcam button to start, or upload an image to analyze.*
        """)

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Foveated VLM Attention Demo")
    print("=" * 60)
    print("\nStarting Gradio server...")
    print("Open the URL in your browser to use the demo.\n")

    demo = create_demo()
    # Try to find a free port
    import socket
    def find_free_port():
        for port in range(7870, 7899):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except:
                continue
        return 7899

    port = find_free_port()
    print(f"Using port: {port}")

    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=port,
        share=True,  # Get a public URL for easy access
        show_error=True
    )
