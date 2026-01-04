#!/usr/bin/env python3
"""
Live Webcam Attention Demo

Shows real-time attention maps from the Foveated VLM model.
Displays:
- Original webcam feed
- Coarse attention heatmap
- Fine attention heatmap
- Overlay of fine attention on video

Press 'q' to quit, 's' to save screenshot, 'c' to toggle caption generation.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque
import time
from threading import Thread, Lock
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class FrameBuffer:
    """Thread-safe circular buffer for frames."""
    def __init__(self, max_frames=16):
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        self.lock = Lock()

    def add(self, frame):
        with self.lock:
            self.frames.append(frame)

    def get_frames(self, n=None):
        with self.lock:
            if n is None:
                return list(self.frames)
            return list(self.frames)[-n:] if len(self.frames) >= n else list(self.frames)

    def __len__(self):
        with self.lock:
            return len(self.frames)


def jet_colormap(value):
    """Fast jet colormap for single value."""
    v = np.clip(value, 0, 1)
    if v < 0.25:
        return (255, int(v * 4 * 255), 0)  # Blue to Cyan
    elif v < 0.5:
        return (int((0.5 - v) * 4 * 255), 255, 0)  # Cyan to Green
    elif v < 0.75:
        return (0, 255, int((v - 0.5) * 4 * 255))  # Green to Yellow
    else:
        return (0, int((1 - v) * 4 * 255), 255)  # Yellow to Red


def create_heatmap(attn, size=(256, 256)):
    """Create heatmap from attention weights."""
    grid_size = int(np.sqrt(len(attn)))
    attn_2d = attn.reshape(grid_size, grid_size)

    # Normalize
    attn_norm = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

    # Resize to target size
    heatmap = cv2.resize(attn_norm, size, interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return heatmap_color


def create_overlay(frame, attn, alpha=0.5):
    """Create attention overlay on frame."""
    heatmap = create_heatmap(attn, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlay


class LiveAttentionDemo:
    def __init__(self, device='cuda', frame_size=256, buffer_frames=8):
        self.device = device
        self.frame_size = frame_size
        self.buffer_frames = buffer_frames

        print("Loading model...")
        self.load_model()

        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try different camera indices
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Using camera index {i}")
                    break
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame_buffer = FrameBuffer(max_frames=buffer_frames)
        self.running = True
        self.show_caption = False
        self.current_caption = ""
        self.last_caption_time = 0
        self.caption_interval = 2.0  # Generate caption every 2 seconds

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

    def load_model(self):
        """Load the foveated VLM model."""
        checkpoint_path = 'outputs/multitask/checkpoints/final.pt'

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = FoveatedVideoModel(
            dino_model='facebook/dinov2-small',
            llm_model='HuggingFaceTB/SmolLM2-135M-Instruct',
            dino_dim=384, llm_dim=576, query_dim=128, lambda_coarse=1.0,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        # Tokenizer for captions
        self.tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded!")

    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Resize and crop to square
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        # Resize to model input size
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame, frame_rgb

    def normalize_for_model(self, frames_rgb):
        """Normalize frames for DINO model."""
        # frames_rgb: list of [H, W, 3] numpy arrays (0-255)
        frames = np.stack(frames_rgb, axis=0)  # [T, H, W, 3]
        frames = frames.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to tensor [T, 3, H, W]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        return frames.unsqueeze(0).to(self.device)  # [1, T, 3, H, W]

    def get_attention(self, frames_norm):
        """Get attention maps for current frames."""
        B, T, C, H, W = frames_norm.shape

        with torch.no_grad():
            # Only process last frame for real-time display
            frame = frames_norm[:, -1]  # [1, 3, H, W]

            # Get patch features
            patch_features, cache = self.model.encoder.encode_patches(frame)

            # Coarse attention
            q_coarse = self.model.q_static.expand(1, -1)
            q_embed_coarse = self.model.encoder.query_input_proj(q_coarse).unsqueeze(1)
            attn_scores_coarse = torch.bmm(q_embed_coarse, patch_features.transpose(1, 2))
            attn_coarse = torch.softmax(attn_scores_coarse / (self.model.encoder.dino_dim ** 0.5), dim=-1)
            attn_coarse = attn_coarse[0, 0, 1:].cpu().numpy()  # Skip CLS

            # For fine attention, we need context from previous frames
            if T >= 2:
                # Get coarse features for context
                coarse_features = []
                for t in range(T):
                    f = frames_norm[:, t]
                    _, c = self.model.encoder.encode_patches(f)
                    q_c = self.model.q_static.expand(1, -1)
                    z_c = self.model.encoder.query_attend(q_c, c)
                    coarse_features.append(z_c)

                coarse_features = torch.stack(coarse_features, dim=1)
                coarse_proj = self.model.dino_to_llm(coarse_features) * self.model.visual_scale

                # LLM forward for query prediction
                coarse_token = self.model.coarse_token.expand(1, 1, -1)
                text_embeds = self.model.get_empty_text_embeds(1)
                seq = torch.cat([text_embeds, coarse_token, coarse_proj], dim=1)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    llm_out = self.model.llm(inputs_embeds=seq, output_hidden_states=True)

                hidden = llm_out.hidden_states[-1]
                N_text = text_embeds.shape[1]
                h_last = hidden[:, N_text + T]  # Hidden state for last frame
                q_fine = self.model.llm_to_query(h_last.float())

                # Fine attention for last frame
                q_embed_fine = self.model.encoder.query_input_proj(q_fine).unsqueeze(1)
                attn_scores_fine = torch.bmm(q_embed_fine, patch_features.transpose(1, 2))
                attn_fine = torch.softmax(attn_scores_fine / (self.model.encoder.dino_dim ** 0.5), dim=-1)
                attn_fine = attn_fine[0, 0, 1:].cpu().numpy()
            else:
                # Not enough context, use coarse for both
                attn_fine = attn_coarse.copy()

            return attn_coarse, attn_fine

    def generate_caption(self, frames_norm):
        """Generate caption for current frames."""
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                caption = self.model.generate_caption(
                    frames_norm, self.tokenizer,
                    max_new_tokens=30, temperature=0.7, use_fine=True
                )[0].strip()
        return caption

    def draw_ui(self, display, fps, attn_coarse, attn_fine):
        """Draw UI elements on display."""
        h, w = display.shape[:2]

        # FPS counter
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Attention stats
        cv2.putText(display, f"Coarse max: {attn_coarse.max():.4f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        cv2.putText(display, f"Fine max: {attn_fine.max():.4f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        ratio = attn_fine.max() / (attn_coarse.max() + 1e-8)
        cv2.putText(display, f"Focus ratio: {ratio:.1f}x", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Caption
        if self.show_caption and self.current_caption:
            # Draw caption background
            caption_y = h - 40
            cv2.rectangle(display, (0, caption_y - 5), (w, h), (0, 0, 0), -1)
            cv2.putText(display, self.current_caption[:80], (10, caption_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Instructions
        cv2.putText(display, "Q:quit S:save C:caption", (w - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return display

    def run(self):
        """Main demo loop."""
        print("\n" + "=" * 50)
        print("Live Attention Demo")
        print("=" * 50)
        print("Controls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("  C - Toggle caption generation")
        print("=" * 50 + "\n")

        frame_count = 0
        screenshot_count = 0

        while self.running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            # Preprocess
            frame_display, frame_rgb = self.preprocess_frame(frame)
            self.frame_buffer.add(frame_rgb)

            # Get buffered frames
            frames = self.frame_buffer.get_frames()
            if len(frames) < 2:
                # Need at least 2 frames for temporal context
                cv2.imshow('Live Attention Demo', frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Normalize for model
            frames_norm = self.normalize_for_model(frames)

            # Get attention maps
            attn_coarse, attn_fine = self.get_attention(frames_norm)

            # Create heatmaps
            heatmap_coarse = create_heatmap(attn_coarse, (self.frame_size, self.frame_size))
            heatmap_fine = create_heatmap(attn_fine, (self.frame_size, self.frame_size))
            overlay = create_overlay(frame_display, attn_fine, alpha=0.5)

            # Create display layout
            # Top row: Original | Coarse | Fine
            # Bottom row: Overlay | Stats
            top_row = np.hstack([frame_display, heatmap_coarse, heatmap_fine])

            # Add labels
            cv2.putText(top_row, "Original", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(top_row, "Coarse Attn", (self.frame_size + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            cv2.putText(top_row, "Fine Attn", (2 * self.frame_size + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

            # Bottom: Overlay with stats
            bottom_left = overlay.copy()
            cv2.putText(bottom_left, "Fine Overlay", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 100), 2)

            # Stats panel
            stats_panel = np.zeros((self.frame_size, self.frame_size * 2, 3), dtype=np.uint8)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_frame_time + 1e-8)
            self.last_frame_time = current_time
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)

            # Draw stats
            y = 30
            cv2.putText(stats_panel, f"FPS: {avg_fps:.1f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(stats_panel, f"Buffer frames: {len(frames)}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 25
            cv2.putText(stats_panel, f"Coarse max: {attn_coarse.max():.4f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
            y += 25
            cv2.putText(stats_panel, f"Fine max: {attn_fine.max():.4f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            y += 25
            ratio = attn_fine.max() / (attn_coarse.max() + 1e-8)
            cv2.putText(stats_panel, f"Focus ratio: {ratio:.1f}x", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Caption generation
            if self.show_caption:
                if current_time - self.last_caption_time > self.caption_interval:
                    self.current_caption = self.generate_caption(frames_norm)
                    self.last_caption_time = current_time

                y += 40
                cv2.putText(stats_panel, "Caption:", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
                y += 20
                # Word wrap caption
                words = self.current_caption.split()
                line = ""
                for word in words[:15]:
                    if len(line + word) < 35:
                        line += word + " "
                    else:
                        cv2.putText(stats_panel, line, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        y += 18
                        line = word + " "
                if line:
                    cv2.putText(stats_panel, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Controls
            cv2.putText(stats_panel, "Controls:", (10, self.frame_size - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(stats_panel, "Q:Quit  S:Save  C:Caption", (10, self.frame_size - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            cv2.putText(stats_panel, f"Caption: {'ON' if self.show_caption else 'OFF'}", (10, self.frame_size - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100) if self.show_caption else (100, 100, 100), 1)

            bottom_row = np.hstack([bottom_left, stats_panel])

            # Combine
            display = np.vstack([top_row, bottom_row])

            # Show
            cv2.imshow('Live Attention Demo', display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                screenshot_path = f'outputs/screenshots/live_demo_{screenshot_count:03d}.png'
                Path('outputs/screenshots').mkdir(parents=True, exist_ok=True)
                cv2.imwrite(screenshot_path, display)
                print(f"Saved: {screenshot_path}")
                screenshot_count += 1
            elif key == ord('c'):
                self.show_caption = not self.show_caption
                print(f"Caption generation: {'ON' if self.show_caption else 'OFF'}")

            frame_count += 1

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Live Webcam Attention Demo')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--size', type=int, default=256, help='Frame size')
    parser.add_argument('--buffer', type=int, default=8, help='Frame buffer size')
    args = parser.parse_args()

    try:
        demo = LiveAttentionDemo(
            device=args.device,
            frame_size=args.size,
            buffer_frames=args.buffer
        )
        demo.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
