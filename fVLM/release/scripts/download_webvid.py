#!/usr/bin/env python
"""
Direct WebVid downloader — bypasses HF datasets library.

Downloads CSV metadata partitions directly from the TempoFunk/webvid-10M repo,
then fetches videos and extracts frames in parallel.

Usage:
    python release/scripts/download_webvid.py --target 50000 --workers 6
    python release/scripts/download_webvid.py --target 10000 --workers 4 --csv-start 0 --csv-end 10
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from release.scripts.precompute import get_tokenizer, tokenize_stage1, ShardWriter


def download_csv_partition(partition_idx: int, branch: str = "t0000") -> list[dict]:
    """Download a single CSV partition and return rows."""
    import requests

    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    url = (
        f"https://huggingface.co/datasets/TempoFunk/webvid-10M/"
        f"resolve/{branch}/data/train/partitions/{partition_idx:04d}.csv"
    )

    try:
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code != 200:
            return []
        reader = csv.DictReader(io.StringIO(r.text))
        return list(reader)
    except Exception:
        return []


def process_video(url: str, caption: str, idx: int, tmp_dir: Path, tokenizer) -> dict | None:
    """Download video, extract frames at 1 FPS 224x224, tokenize caption."""
    video_path = tmp_dir / f"vid_{idx}.mp4"
    frames_dir = tmp_dir / f"frames_{idx}"
    frames_dir.mkdir(exist_ok=True)

    try:
        # Download video (15s timeout, max 50MB)
        result = subprocess.run(
            ["wget", "-q", "-O", str(video_path), "--timeout=15", url],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0 or not video_path.exists() or video_path.stat().st_size < 1000:
            return None

        # Extract frames at 1 FPS, center-crop to 224x224
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

        tok = tokenize_stage1(caption, tokenizer=tokenizer)
        meta = {
            "token_ids": tok["token_ids"],
            "loss_mask": tok["loss_mask"],
            "caption": caption,
            "frame_count": len(frame_data),
            "source": "webvid",
        }
        frame_data["json"] = json.dumps(meta).encode("utf-8")
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
    parser = argparse.ArgumentParser(description="Direct WebVid downloader")
    parser.add_argument("--target", type=int, default=50000, help="Target sample count")
    parser.add_argument("--workers", type=int, default=6, help="Parallel download workers")
    parser.add_argument("--output", default="/workspace/data/webvid", help="Output directory")
    parser.add_argument("--csv-start", type=int, default=0, help="First CSV partition index")
    parser.add_argument("--csv-end", type=int, default=1000, help="Last CSV partition index")
    parser.add_argument("--branch", default="t0000", help="HF repo branch")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    writer = ShardWriter(args.output, samples_per_shard=1000)
    tmp_dir = Path("/workspace/tmp/webvid_dl")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0
    submitted = 0
    t0 = time.time()

    print(f"[WebVid] Target: {args.target} samples, {args.workers} workers")
    print(f"[WebVid] CSV partitions: {args.csv_start}-{args.csv_end} on branch {args.branch}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        for csv_idx in range(args.csv_start, args.csv_end):
            if count >= args.target:
                break

            # Download CSV metadata
            rows = download_csv_partition(csv_idx, branch=args.branch)
            if not rows:
                print(f"  [WebVid] CSV {csv_idx:04d}: empty or failed", flush=True)
                continue

            print(f"  [WebVid] CSV {csv_idx:04d}: {len(rows)} URLs", flush=True)

            for row in rows:
                if count >= args.target:
                    break

                url = row.get("contentUrl", "")
                caption = row.get("name", "")
                if not url or not caption:
                    continue

                # Submit to thread pool
                fut = executor.submit(process_video, url, caption, submitted, tmp_dir, tokenizer)
                futures[fut] = submitted
                submitted += 1

                # Drain completed futures when queue is full
                while len(futures) >= args.workers * 4:
                    done = [f for f in futures if f.done()]
                    if not done:
                        time.sleep(0.1)
                        continue
                    for f in done:
                        idx = futures.pop(f)
                        try:
                            result = f.result()
                        except Exception:
                            result = None
                            errors += 1

                        if result is not None:
                            writer.write_sample(f"{count:08d}", result)
                            count += 1
                        else:
                            errors += 1

                        if count % 500 == 0 and count > 0:
                            elapsed = time.time() - t0
                            rate = count / max(elapsed, 1)
                            err_rate = errors / max(count + errors, 1) * 100
                            print(
                                f"  [WebVid] {count}/{args.target} samples, "
                                f"{rate:.1f} samp/s, {err_rate:.0f}% errors, "
                                f"{elapsed/60:.0f}min elapsed",
                                flush=True,
                            )

        # Drain remaining futures
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception:
                result = None
                errors += 1

            if result is not None:
                writer.write_sample(f"{count:08d}", result)
                count += 1
            else:
                errors += 1

    writer.close()
    elapsed = time.time() - t0
    print(f"\n[WebVid] Done: {count} samples, {writer._shard_idx} shards, "
          f"{errors} errors ({errors/(count+errors)*100:.0f}% error rate), "
          f"{elapsed/60:.0f}min")


if __name__ == "__main__":
    main()
