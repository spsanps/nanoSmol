#!/usr/bin/env python
"""
Pack downloaded LLaVA YouTube frames with annotation-only Stage 3 shards.

Reads annotation-only shards from /workspace/data/stage3/, matches them with
downloaded YouTube frames in /workspace/data/llava_youtube_frames/, and creates
new complete shards with both frames and text annotations.

Usage:
    python release/scripts/pack_llava_youtube.py
    python release/scripts/pack_llava_youtube.py --annotations /workspace/data/stage3 \
        --frames /workspace/data/llava_youtube_frames --output /workspace/data/stage3_packed
"""

import argparse
import io
import json
import re
import tarfile
import time
from pathlib import Path


def read_tar_samples(tar_path: Path) -> list[dict]:
    """Read all samples from a tar shard, grouped by sample key."""
    samples = {}
    with tarfile.open(str(tar_path), "r") as tar:
        for member in tar:
            if member.isfile():
                # Sample key is everything before the first dot
                parts = member.name.split(".", 1)
                key = parts[0]
                ext = parts[1] if len(parts) > 1 else ""
                data = tar.extractfile(member).read()
                if key not in samples:
                    samples[key] = {}
                samples[key][ext] = data
    return samples


def extract_youtube_id(source: str) -> str | None:
    """Extract YouTube video ID from source metadata."""
    # source format: "llava_video/0_30_s_youtube_v0_1" etc.
    # The video path in original data is like:
    # liwei_youtube_videos/videos/youtube_video_2024/ytb_XXXXX.mp4
    # We need to look at the video field stored in the annotation
    return None  # Will be extracted from the video field in JSON


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/workspace/data/stage3",
                        help="Directory with annotation-only shards")
    parser.add_argument("--frames", default="/workspace/data/llava_youtube_frames",
                        help="Directory with downloaded YouTube frame directories")
    parser.add_argument("--output", default="/workspace/data/stage3_packed",
                        help="Output directory for packed shards")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    args = parser.parse_args()

    annotations_dir = Path(args.annotations)
    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index available YouTube frames
    available_frames = set()
    if frames_dir.exists():
        for d in frames_dir.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                available_frames.add(d.name)
    print(f"[Pack] {len(available_frames)} YouTube videos with frames available")

    # Process annotation shards
    tar_files = sorted(annotations_dir.glob("*.tar"))
    print(f"[Pack] {len(tar_files)} annotation shards to process")

    shard_idx = 0
    sample_count = 0
    matched = 0
    unmatched = 0
    t0 = time.time()

    current_tar = None
    in_shard = 0

    def open_shard():
        nonlocal current_tar, shard_idx, in_shard
        if current_tar is not None:
            current_tar.close()
        path = output_dir / f"{shard_idx:05d}.tar"
        current_tar = tarfile.open(str(path), "w")
        in_shard = 0
        shard_idx += 1

    def add_file(name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        current_tar.addfile(info, io.BytesIO(data))

    for tar_path in tar_files:
        samples = read_tar_samples(tar_path)

        for sample_key, files in samples.items():
            if current_tar is None or in_shard >= args.samples_per_shard:
                open_shard()

            # Parse the JSON metadata
            json_data = files.get("json", b"{}")
            try:
                meta = json.loads(json_data)
            except json.JSONDecodeError:
                meta = {}

            source = meta.get("source", "")
            has_frames = meta.get("has_frames", False)

            # Try to match YouTube ID
            # The video path might be stored differently depending on how we saved it
            # Try to find it from the source or video_path fields
            yt_id = None

            # Check if there's a video_path with ytb_ prefix
            video_path = meta.get("video_path", "")
            if video_path:
                m = re.search(r"ytb_([A-Za-z0-9_-]+)", video_path)
                if m:
                    yt_id = m.group(1)

            if yt_id and yt_id in available_frames:
                # Load frames from downloaded directory
                vid_dir = frames_dir / yt_id
                frame_files = sorted(vid_dir.glob("*.jpg"))

                frame_count = 0
                new_key = f"{sample_count:08d}"

                for i, fp in enumerate(frame_files[:64]):
                    frame_bytes = fp.read_bytes()
                    add_file(f"{new_key}.{i:03d}.jpg", frame_bytes)
                    frame_count += 1

                # Update metadata
                meta["frame_count"] = frame_count
                meta["has_frames"] = True
                meta["yt_id"] = yt_id

                add_file(f"{new_key}.json", json.dumps(meta).encode("utf-8"))
                matched += 1
            else:
                # Keep annotation-only
                new_key = f"{sample_count:08d}"
                for ext, data in files.items():
                    add_file(f"{new_key}.{ext}", data)
                unmatched += 1

            sample_count += 1
            in_shard += 1

            if sample_count % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  [Pack] {sample_count} processed, {matched} matched, "
                      f"{unmatched} unmatched, {elapsed:.0f}s", flush=True)

    if current_tar is not None:
        current_tar.close()

    elapsed = time.time() - t0
    print(f"\n[Pack] Done: {sample_count} total samples")
    print(f"  {matched} matched with frames ({matched/max(sample_count,1)*100:.1f}%)")
    print(f"  {unmatched} annotation-only")
    print(f"  {shard_idx} shards written to {output_dir}")
    print(f"  {elapsed:.0f}s elapsed")


if __name__ == "__main__":
    main()
