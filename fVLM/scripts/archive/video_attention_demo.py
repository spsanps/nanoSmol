#!/usr/bin/env python3
"""
Video Attention Demo

Alternative to live webcam demo - processes a video file.
Useful for testing or when webcam isn't available.

Usage:
    python scripts/video_attention_demo.py --video path/to/video.mp4
    python scripts/video_attention_demo.py --video 0  # webcam
"""

import torch
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque
import time
from transformers import AutoTokenizer
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class VideoAttentionDemo:
    def __init__(self, video_source, device='cuda', frame_size=256, buffer_frames=8):
        self.device = device
        self.frame_size = frame_size
        self.buffer_frames = buffer_frames
        self.frame_buffer = deque(maxlen=buffer_frames)

        print("Loading model...")
        self.load_model()

        print(f"Opening video source: {video_source}")
        # Handle webcam index or file path
        if video_source.isdigit():
            self.cap = cv2.VideoCapture(int(video_source))
        else:
            self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {self.total_frames} frames at {self.fps:.1f} FPS")

    def load_model(self):
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

    def preprocess_frame(self, frame):
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, frame_rgb

    def normalize_for_model(self, frames_rgb):
        frames = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        return frames.unsqueeze(0).to(self.device)

    def get_attention(self, frames_norm):
        B, T, C, H, W = frames_norm.shape

        with torch.no_grad():
            frame = frames_norm[:, -1]
            patch_features, cache = self.model.encoder.encode_patches(frame)

            # Coarse
            q_coarse = self.model.q_static.expand(1, -1)
            q_embed_coarse = self.model.encoder.query_input_proj(q_coarse).unsqueeze(1)
            attn_scores_coarse = torch.bmm(q_embed_coarse, patch_features.transpose(1, 2))
            attn_coarse = torch.softmax(attn_scores_coarse / (self.model.encoder.dino_dim ** 0.5), dim=-1)
            attn_coarse = attn_coarse[0, 0, 1:].cpu().numpy()

            # Fine with temporal context
            if T >= 2:
                coarse_features = []
                for t in range(T):
                    f = frames_norm[:, t]
                    _, c = self.model.encoder.encode_patches(f)
                    z_c = self.model.encoder.query_attend(self.model.q_static.expand(1, -1), c)
                    coarse_features.append(z_c)

                coarse_features = torch.stack(coarse_features, dim=1)
                coarse_proj = self.model.dino_to_llm(coarse_features) * self.model.visual_scale

                seq = torch.cat([
                    self.model.get_empty_text_embeds(1),
                    self.model.coarse_token.expand(1, 1, -1),
                    coarse_proj
                ], dim=1)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    llm_out = self.model.llm(inputs_embeds=seq, output_hidden_states=True)

                hidden = llm_out.hidden_states[-1]
                h_last = hidden[:, 1 + T]
                q_fine = self.model.llm_to_query(h_last.float())

                q_embed_fine = self.model.encoder.query_input_proj(q_fine).unsqueeze(1)
                attn_scores_fine = torch.bmm(q_embed_fine, patch_features.transpose(1, 2))
                attn_fine = torch.softmax(attn_scores_fine / (self.model.encoder.dino_dim ** 0.5), dim=-1)
                attn_fine = attn_fine[0, 0, 1:].cpu().numpy()
            else:
                attn_fine = attn_coarse.copy()

            return attn_coarse, attn_fine

    def create_heatmap(self, attn):
        grid_size = int(np.sqrt(len(attn)))
        attn_2d = attn.reshape(grid_size, grid_size)
        attn_norm = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        heatmap = cv2.resize(attn_norm, (self.frame_size, self.frame_size), interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    def create_overlay(self, frame, attn, alpha=0.5):
        heatmap = self.create_heatmap(attn)
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    def run(self, output_path=None):
        print("\n" + "=" * 50)
        print("Video Attention Demo")
        print("=" * 50)
        print("Controls: Q:Quit  SPACE:Pause  S:Save")
        print("=" * 50 + "\n")

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                  (self.frame_size * 3, self.frame_size * 2))
            print(f"Recording to: {output_path}")
        else:
            out = None

        paused = False
        frame_count = 0
        start_time = time.time()

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video")
                    break

                frame_display, frame_rgb = self.preprocess_frame(frame)
                self.frame_buffer.append(frame_rgb)

            frames = list(self.frame_buffer)
            if len(frames) < 2:
                cv2.imshow('Video Attention Demo', frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            frames_norm = self.normalize_for_model(frames)
            attn_coarse, attn_fine = self.get_attention(frames_norm)

            # Create display
            heatmap_coarse = self.create_heatmap(attn_coarse)
            heatmap_fine = self.create_heatmap(attn_fine)
            overlay = self.create_overlay(frame_display, attn_fine)

            # Layout
            top_row = np.hstack([frame_display, heatmap_coarse, heatmap_fine])
            cv2.putText(top_row, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(top_row, "Coarse", (self.frame_size + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            cv2.putText(top_row, "Fine", (2 * self.frame_size + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

            # Stats panel
            stats = np.zeros((self.frame_size, self.frame_size * 2, 3), dtype=np.uint8)
            elapsed = time.time() - start_time
            fps = frame_count / (elapsed + 1e-8)

            y = 30
            cv2.putText(stats, f"Frame: {frame_count}/{self.total_frames}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 30
            cv2.putText(stats, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y += 30
            cv2.putText(stats, f"Coarse max: {attn_coarse.max():.4f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            y += 25
            cv2.putText(stats, f"Fine max: {attn_fine.max():.4f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            y += 25
            ratio = attn_fine.max() / (attn_coarse.max() + 1e-8)
            cv2.putText(stats, f"Focus ratio: {ratio:.1f}x", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if paused:
                cv2.putText(stats, "PAUSED", (10, self.frame_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            bottom_row = np.hstack([overlay, stats])
            cv2.putText(bottom_row, "Fine Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 100), 2)

            display = np.vstack([top_row, bottom_row])

            if out:
                out.write(display)

            cv2.imshow('Video Attention Demo', display)

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                path = f'outputs/screenshots/video_demo_{frame_count:05d}.png'
                Path('outputs/screenshots').mkdir(parents=True, exist_ok=True)
                cv2.imwrite(path, display)
                print(f"Saved: {path}")

            if not paused:
                frame_count += 1

        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description='Video Attention Demo')
    parser.add_argument('--video', type=str, default='0', help='Video path or camera index')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--buffer', type=int, default=8)
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    args = parser.parse_args()

    demo = VideoAttentionDemo(
        video_source=args.video,
        device=args.device,
        frame_size=args.size,
        buffer_frames=args.buffer
    )
    demo.run(output_path=args.output)


if __name__ == "__main__":
    main()
