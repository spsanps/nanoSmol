#!/usr/bin/env python
"""
Post-process video2dataset output shards for fVLM training.

Takes raw MP4 webdataset shards from video2dataset, extracts frames at 1FPS 224x224,
tokenizes captions with SmolLM2, and repacks as training-ready webdataset shards.

Usage:
    python release/scripts/postprocess_v2d.py \
        --input /workspace/data/webvid_v2d \
        --output /workspace/data/webvid_processed \
        --workers 6
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def extract_frames_from_mp4(mp4_bytes: bytes, tmp_dir: str, max_frames: int = 64) -> dict | None:
    """Extract frames at 1FPS 224x224 from MP4 bytes. Returns dict of frame bytes."""
    video_path = os.path.join(tmp_dir, "input.mp4")
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        with open(video_path, "wb") as f:
            f.write(mp4_bytes)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
                "-frames:v", str(max_frames), "-q:v", "2",
                os.path.join(frames_dir, "frame_%03d.jpg"),
            ],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            return None

        frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
        if not frame_files:
            return None

        frame_data = {}
        for i, fp in enumerate(frame_files[:max_frames]):
            frame_data[f"{i:03d}.jpg"] = fp.read_bytes()
        return frame_data

    except Exception:
        return None
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        for f in Path(frames_dir).glob("*"):
            f.unlink(missing_ok=True)
        if os.path.exists(frames_dir):
            try:
                os.rmdir(frames_dir)
            except OSError:
                pass


def process_shard(input_path: str, output_dir: str, shard_idx: int,
                  samples_per_shard: int = 1000) -> dict:
    """Process one v2d shard: extract frames, tokenize, repack."""
    from release.scripts.precompute import get_tokenizer, tokenize_stage1

    tokenizer = get_tokenizer()
    stats = {"input": input_path, "processed": 0, "errors": 0, "shards_written": 0}

    # Read all samples from input shard
    samples = {}
    try:
        with tarfile.open(input_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key = member.name.rsplit(".", 1)[0]
                ext = member.name.rsplit(".", 1)[1] if "." in member.name else ""
                data = tar.extractfile(member).read()
                if key not in samples:
                    samples[key] = {}
                samples[key][ext] = data
    except Exception as e:
        stats["errors"] = len(samples) or 1
        return stats

    # Process each sample
    output_path = Path(output_dir)
    current_tar = None
    current_shard_idx = shard_idx
    in_shard_count = 0
    sample_count = 0

    with tempfile.TemporaryDirectory(dir="/workspace/tmp") as tmp_dir:
        for key, files in samples.items():
            mp4_data = files.get("mp4")
            caption = files.get("txt", b"").decode("utf-8", errors="replace").strip()
            meta_raw = files.get("json", b"{}")

            if not mp4_data or not caption:
                stats["errors"] += 1
                continue

            # Extract frames
            frame_data = extract_frames_from_mp4(mp4_data, tmp_dir)
            if not frame_data:
                stats["errors"] += 1
                continue

            # Tokenize caption (Stage 1: all-text loss)
            tok = tokenize_stage1(caption, tokenizer=tokenizer)

            # Parse original metadata
            try:
                orig_meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                orig_meta = {}

            meta = {
                "token_ids": tok["token_ids"],
                "loss_mask": tok["loss_mask"],
                "caption": caption,
                "frame_count": len(frame_data),
                "source": "webvid",
                "videoid": orig_meta.get("videoid", ""),
                "duration": orig_meta.get("duration", ""),
            }

            # Write to output shard
            if current_tar is None or in_shard_count >= samples_per_shard:
                if current_tar is not None:
                    current_tar.close()
                    stats["shards_written"] += 1
                shard_path = output_path / f"{current_shard_idx:05d}.tar"
                current_tar = tarfile.open(str(shard_path), "w")
                current_shard_idx += 1
                in_shard_count = 0

            sample_key = f"{sample_count:08d}"
            # Write frames
            for ext, data in frame_data.items():
                info = tarfile.TarInfo(name=f"{sample_key}.{ext}")
                info.size = len(data)
                current_tar.addfile(info, io.BytesIO(data))

            # Write metadata
            json_bytes = json.dumps(meta).encode("utf-8")
            info = tarfile.TarInfo(name=f"{sample_key}.json")
            info.size = len(json_bytes)
            current_tar.addfile(info, io.BytesIO(json_bytes))

            sample_count += 1
            in_shard_count += 1
            stats["processed"] += 1

    if current_tar is not None:
        current_tar.close()
        stats["shards_written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/workspace/data/webvid_v2d",
                        help="Input directory with v2d raw shards")
    parser.add_argument("--output", default="/workspace/data/webvid_processed",
                        help="Output directory for processed shards")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--start-shard", type=int, default=0,
                        help="First input shard index to process")
    parser.add_argument("--end-shard", type=int, default=0,
                        help="Last input shard index (0=all)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_shards = sorted(input_dir.glob("*.tar"))
    if args.end_shard > 0:
        input_shards = input_shards[args.start_shard:args.end_shard]
    elif args.start_shard > 0:
        input_shards = input_shards[args.start_shard:]

    print(f"[PostProcess] {len(input_shards)} input shards, {args.workers} workers")
    print(f"[PostProcess] Output: {output_dir}")

    total_processed = 0
    total_errors = 0
    total_shards = 0
    t0 = time.time()

    # Each input shard gets a unique output shard index range
    # Input shard N -> output shard starting at N * (input_shard_size / samples_per_shard + 1)
    # Simpler: assign sequentially
    shard_offset = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, shard_path in enumerate(input_shards):
            # Each input shard (1000 samples) -> 1 output shard (1000 samples)
            out_idx = shard_offset + i
            fut = executor.submit(
                process_shard, str(shard_path), str(output_dir),
                out_idx, args.samples_per_shard
            )
            futures[fut] = str(shard_path)

        for fut in as_completed(futures):
            shard_name = futures[fut]
            try:
                stats = fut.result()
                total_processed += stats["processed"]
                total_errors += stats["errors"]
                total_shards += stats["shards_written"]
            except Exception as e:
                print(f"  Error processing {shard_name}: {e}", flush=True)
                total_errors += 1000  # assume full shard failed

            elapsed = time.time() - t0
            done = total_processed + total_errors
            if done > 0 and done % 5000 < 1000:
                rate = total_processed / max(elapsed, 1)
                print(f"  [PostProcess] {total_processed} processed, {total_errors} errors, "
                      f"{total_shards} shards, {rate:.1f} samp/s, {elapsed/60:.0f}min",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\n[PostProcess] Done: {total_processed} samples, {total_errors} errors, "
          f"{total_shards} shards, {elapsed/60:.0f}min")


if __name__ == "__main__":
    main()
