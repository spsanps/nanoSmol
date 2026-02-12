#!/usr/bin/env python
"""
Download YouTube videos referenced by LLaVA-Video-178K and extract frames.

Reads YouTube video IDs, downloads with yt-dlp, extracts frames at 1FPS 224x224,
and updates the annotation-only Stage 3 shards with actual frame data.

Usage:
    python release/scripts/download_llava_youtube.py --ids /workspace/tmp/llava_yt_ids.txt --workers 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from release.scripts.precompute import ShardWriter


def download_and_extract(yt_id: str, output_dir: Path, tmp_dir: Path) -> dict | None:
    """Download a YouTube video and extract frames at 1FPS 224x224."""
    video_path = tmp_dir / f"yt_{yt_id}.mp4"
    frames_dir = tmp_dir / f"frames_{yt_id}"
    frames_dir.mkdir(exist_ok=True)

    try:
        # Download with yt-dlp (worst quality to save bandwidth, max 50MB)
        result = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--no-warnings", "-q",
                "-f", "worst[ext=mp4]/worst",
                "--max-filesize", "50M",
                "-o", str(video_path),
                f"https://www.youtube.com/watch?v={yt_id}",
            ],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0 or not video_path.exists() or video_path.stat().st_size < 1000:
            return None

        # Extract frames
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
                "-frames:v", "64", "-q:v", "2",
                str(frames_dir / "frame_%03d.jpg"),
            ],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            return None

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            return None

        frame_data = {}
        for i, fp in enumerate(frame_files[:64]):
            frame_data[f"{i:03d}.jpg"] = fp.read_bytes()

        return frame_data

    except Exception:
        return None
    finally:
        video_path.unlink(missing_ok=True)
        for f in frames_dir.glob("*"):
            f.unlink(missing_ok=True)
        if frames_dir.exists():
            try:
                frames_dir.rmdir()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="File with YouTube IDs, one per line")
    parser.add_argument("--output", default="/workspace/data/llava_youtube_frames")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    with open(args.ids) as f:
        yt_ids = [line.strip() for line in f if line.strip()]

    if args.max_videos > 0:
        yt_ids = yt_ids[:args.max_videos]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/workspace/tmp/llava_yt_dl")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LLaVA-YT] Downloading {len(yt_ids)} YouTube videos with {args.workers} workers")

    count = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_and_extract, yt_id, output_dir, tmp_dir): yt_id
            for yt_id in yt_ids
        }

        for fut in as_completed(futures):
            yt_id = futures[fut]
            try:
                frame_data = fut.result()
            except Exception:
                frame_data = None

            if frame_data:
                # Save frames as individual files for later matching
                vid_dir = output_dir / yt_id
                vid_dir.mkdir(exist_ok=True)
                for fname, data in frame_data.items():
                    (vid_dir / fname).write_bytes(data)

                # Also save metadata
                meta = {"yt_id": yt_id, "frame_count": len(frame_data)}
                (vid_dir / "meta.json").write_text(json.dumps(meta))
                count += 1
            else:
                errors += 1

            total = count + errors
            if total % 100 == 0:
                elapsed = time.time() - t0
                rate = count / max(elapsed, 1)
                err_pct = errors / max(total, 1) * 100
                print(
                    f"  [LLaVA-YT] {count} downloaded, {errors} failed "
                    f"({err_pct:.0f}% err), {rate:.1f} vid/s, {elapsed/60:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n[LLaVA-YT] Done: {count} videos downloaded, {errors} errors, {elapsed/60:.0f}min")


if __name__ == "__main__":
    main()
