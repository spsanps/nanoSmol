#!/usr/bin/env python
"""
Download Panda-70M video clips and extract frames.

Downloads YouTube videos from Panda-70M dataset, extracts clips at specified
timestamps, samples frames at 1FPS 224x224, and saves as webdataset shards.

Usage:
    python release/scripts/download_panda70m.py --split train_2m --target 50000 --workers 8
    python release/scripts/download_panda70m.py --split train_2m --workers 16 --target 0  # all
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def download_and_extract_clips(
    yt_id: str, timestamps: list, captions: list, tmp_dir: Path, max_frames: int = 64
) -> list[dict] | None:
    """Download a YouTube video and extract clips as frame sequences.

    Returns a list of dicts, one per clip:
        {"frames": {f"000.jpg": bytes, ...}, "caption": str, "frame_count": int}
    """
    video_path = tmp_dir / f"yt_{yt_id}.mp4"

    try:
        # Download with yt-dlp (worst quality to save bandwidth)
        result = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--no-warnings", "-q",
                "-f", "worst[ext=mp4]/worst",
                "--max-filesize", "100M",
                "-o", str(video_path),
                f"https://www.youtube.com/watch?v={yt_id}",
            ],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0 or not video_path.exists() or video_path.stat().st_size < 1000:
            return None

        clips = []
        for i, (ts, caption) in enumerate(zip(timestamps, captions)):
            if len(clips) >= 5:  # Max 5 clips per video
                break

            start_str, end_str = ts
            frames_dir = tmp_dir / f"frames_{yt_id}_{i}"
            frames_dir.mkdir(exist_ok=True)

            try:
                # Extract clip frames at 1 FPS, 224x224
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", start_str,
                        "-to", end_str,
                        "-i", str(video_path),
                        "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
                        "-frames:v", str(max_frames),
                        "-q:v", "2",
                        str(frames_dir / "frame_%03d.jpg"),
                    ],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    continue

                frame_files = sorted(frames_dir.glob("frame_*.jpg"))
                if not frame_files:
                    continue

                frame_data = {}
                for j, fp in enumerate(frame_files[:max_frames]):
                    frame_data[f"{j:03d}.jpg"] = fp.read_bytes()

                clips.append({
                    "frames": frame_data,
                    "caption": caption,
                    "frame_count": len(frame_data),
                })
            finally:
                for f in frames_dir.glob("*"):
                    f.unlink(missing_ok=True)
                if frames_dir.exists():
                    try:
                        frames_dir.rmdir()
                    except OSError:
                        pass

        return clips if clips else None

    except Exception:
        return None
    finally:
        video_path.unlink(missing_ok=True)


def process_sample(sample: dict, idx: int, tmp_dir: Path, tokenizer) -> list[dict] | None:
    """Process one Panda-70M sample (may produce multiple clips)."""
    from release.scripts.precompute import tokenize_stage1

    yt_id = sample["videoID"]
    timestamps = sample["timestamp"]
    captions = sample["caption"]

    if not timestamps or not captions:
        return None

    clips = download_and_extract_clips(yt_id, timestamps, captions, tmp_dir)
    if not clips:
        return None

    results = []
    for clip in clips:
        tok = tokenize_stage1(clip["caption"], tokenizer=tokenizer)
        meta = {
            "token_ids": tok["token_ids"],
            "loss_mask": tok["loss_mask"],
            "caption": clip["caption"],
            "frame_count": clip["frame_count"],
            "source": "panda70m",
            "yt_id": yt_id,
        }
        files = dict(clip["frames"])
        files["json"] = json.dumps(meta).encode("utf-8")
        results.append(files)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train_2m", help="Dataset split")
    parser.add_argument("--target", type=int, default=0, help="Target clip count (0=all)")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    parser.add_argument("--output", default="/workspace/data/panda70m")
    args = parser.parse_args()

    from datasets import load_dataset
    from release.scripts.precompute import get_tokenizer, ShardWriter

    print(f"[Panda-70M] Loading {args.split} split...", flush=True)
    ds = load_dataset("multimodalart/panda-70m", split=args.split, streaming=True)

    tokenizer = get_tokenizer()
    writer = ShardWriter(args.output, samples_per_shard=1000)
    tmp_dir = Path("/workspace/tmp/panda70m_dl")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    target = args.target if args.target > 0 else float("inf")
    count = 0
    errors = 0
    videos_tried = 0
    t0 = time.time()

    print(f"[Panda-70M] Target: {args.target or 'all'} clips, {args.workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        sample_iter = iter(ds)

        # Submit initial batch
        for _ in range(args.workers * 2):
            try:
                sample = next(sample_iter)
                fut = executor.submit(process_sample, sample, videos_tried, tmp_dir, tokenizer)
                futures[fut] = sample
                videos_tried += 1
            except StopIteration:
                break

        while futures and count < target:
            done = [f for f in futures if f.done()]
            if not done:
                time.sleep(0.5)
                continue

            for fut in done:
                sample = futures.pop(fut)
                try:
                    clips = fut.result()
                except Exception:
                    clips = None
                    errors += 1

                if clips:
                    for clip_files in clips:
                        if count >= target:
                            break
                        writer.write_sample(f"{count:08d}", clip_files)
                        count += 1
                else:
                    errors += 1

                # Submit more work
                if count < target:
                    try:
                        new_sample = next(sample_iter)
                        new_fut = executor.submit(
                            process_sample, new_sample, videos_tried, tmp_dir, tokenizer
                        )
                        futures[new_fut] = new_sample
                        videos_tried += 1
                    except StopIteration:
                        pass

                total = count + errors
                if total > 0 and total % 100 == 0:
                    elapsed = time.time() - t0
                    rate = count / max(elapsed, 1)
                    err_pct = errors / max(total, 1) * 100
                    print(
                        f"  [Panda-70M] {count} clips from {videos_tried} videos, "
                        f"{errors} errors ({err_pct:.0f}%), "
                        f"{rate:.1f} clips/s, {elapsed/60:.0f}min",
                        flush=True,
                    )

        # Drain remaining futures
        for fut in as_completed(futures):
            try:
                clips = fut.result()
            except Exception:
                clips = None
                errors += 1

            if clips:
                for clip_files in clips:
                    if count >= target:
                        break
                    writer.write_sample(f"{count:08d}", clip_files)
                    count += 1

    writer.close()
    elapsed = time.time() - t0
    print(
        f"\n[Panda-70M] Done: {count} clips from {videos_tried} videos, "
        f"{errors} errors, {writer._shard_idx} shards, {elapsed/60:.0f}min"
    )


if __name__ == "__main__":
    main()
