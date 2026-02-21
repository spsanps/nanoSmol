#!/usr/bin/env python3
"""
Save frame grids for the fresh generation videos so we can visually inspect.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def main():
    skip_first = 500
    num_examples = 15
    num_frames_show = 8  # Show 8 frames in grid

    frames_dir = Path('data/webvid_large/frames')
    output_dir = Path('outputs/fresh_generation')
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(list(frames_dir.glob('*.pt')))
    available = frame_files[skip_first:]

    print(f"Saving frame grids for {num_examples} videos...")

    for i, frame_file in enumerate(available[:num_examples]):
        try:
            frames = torch.load(frame_file, weights_only=True)  # (T, 3, H, W)
            video_id = frame_file.stem

            # Sample frames evenly
            T = frames.shape[0]
            indices = np.linspace(0, T - 1, num_frames_show).astype(int)

            # Create 2x4 grid
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Video {video_id}', fontsize=14)

            for j, idx in enumerate(indices):
                ax = axes[j // 4, j % 4]
                frame = frames[idx].permute(1, 2, 0).numpy()  # HWC

                # Normalize if needed
                if frame.max() > 1:
                    frame = frame / 255.0

                ax.imshow(np.clip(frame, 0, 1))
                ax.set_title(f'Frame {idx}')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_dir / f'frames_{i+1:02d}_{video_id}.png', dpi=100)
            plt.close()

            print(f"  [{i+1}/{num_examples}] Saved {video_id}")

        except Exception as e:
            print(f"  Error with {frame_file.name}: {e}")

    print(f"\nFrame grids saved to {output_dir}")


if __name__ == "__main__":
    main()
