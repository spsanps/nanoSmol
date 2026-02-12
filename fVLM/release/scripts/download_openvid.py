#!/usr/bin/env python
"""
Download and process OpenVid-1M videos into webdataset shards.

Parallel download pipeline:
  4 concurrent zip downloads → process as each completes → pack webdataset → delete

Usage:
    python release/scripts/download_openvid.py --parts 0-185 --dl-workers 4 --ffmpeg-workers 6
    python release/scripts/download_openvid.py --parts 0-69 --dl-workers 4
    python release/scripts/download_openvid.py --parts 5 --dl-workers 1  # single part
"""

import argparse
import csv
import io
import json
import os
import queue
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Parts 102-119 are split files on HuggingFace
SPLIT_PARTS = set(range(102, 120))
HF_BASE = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"
CSV_URL = f"{HF_BASE}/data/train/OpenVid-1M.csv"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with wget, return True on success."""
    try:
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", str(dest), url],
            timeout=7200,
            capture_output=True,
        )
        return result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Download failed ({desc}): {e}", flush=True)
        return False


def download_zip_part(part_num: int, download_dir: Path) -> Path | None:
    """Download a single zip part, handling split files."""
    zip_path = download_dir / f"OpenVid_part{part_num}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        return zip_path

    if part_num in SPLIT_PARTS:
        part_a = download_dir / f"OpenVid_part{part_num}_partaa"
        part_b = download_dir / f"OpenVid_part{part_num}_partab"

        ok_a = download_file(
            f"{HF_BASE}/OpenVid_part{part_num}_partaa", part_a, f"part{part_num}aa"
        )
        ok_b = download_file(
            f"{HF_BASE}/OpenVid_part{part_num}_partab", part_b, f"part{part_num}ab"
        )

        if not ok_a:
            return None

        with open(zip_path, "wb") as out:
            for p in [part_a, part_b]:
                if p.exists():
                    with open(p, "rb") as f:
                        while True:
                            chunk = f.read(64 * 1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                    p.unlink()

        return zip_path if zip_path.stat().st_size > 1_000_000 else None
    else:
        ok = download_file(
            f"{HF_BASE}/OpenVid_part{part_num}.zip", zip_path, f"part{part_num}"
        )
        return zip_path if ok else None


def extract_frames_ffmpeg(
    video_path: Path, frames_dir: Path, max_frames: int = 64
) -> list[Path]:
    """Extract 1FPS 224x224 center-crop frames from a video."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                str(frames_dir / "frame_%03d.jpg"),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return []
        return sorted(frames_dir.glob("frame_*.jpg"))
    except (subprocess.TimeoutExpired, Exception):
        return []


def process_single_video(args: tuple) -> dict | None:
    """Process one video: extract frames, return frame bytes + metadata."""
    video_path, caption, tmp_base, max_frames = args
    video_id = video_path.stem
    frames_dir = tmp_base / f"frames_{os.getpid()}_{video_id}"

    try:
        frame_files = extract_frames_ffmpeg(video_path, frames_dir, max_frames)
        if not frame_files:
            return None

        frame_data = {}
        for i, fp in enumerate(frame_files[:max_frames]):
            data = fp.read_bytes()
            if len(data) < 100 or data[:2] != b"\xff\xd8":
                continue
            frame_data[f"{i:03d}.jpg"] = data

        if not frame_data:
            return None

        meta = {
            "caption": caption,
            "frame_count": len(frame_data),
            "source": "openvid",
            "video_id": video_id,
        }
        frame_data["json"] = json.dumps(meta).encode("utf-8")
        return frame_data

    except Exception:
        return None
    finally:
        if frames_dir.exists():
            for f in frames_dir.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                frames_dir.rmdir()
            except OSError:
                pass


class ShardWriter:
    """Write samples to webdataset tar shards (NOT thread-safe)."""

    def __init__(self, output_dir: str, samples_per_shard: int = 1000, start_shard: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self._shard_idx = start_shard
        self._sample_count = 0
        self._tar = None
        self._total = 0
        self._open_new_shard()

    def _open_new_shard(self):
        if self._tar is not None:
            self._tar.close()
        path = self.output_dir / f"{self._shard_idx:06d}.tar"
        self._tar = tarfile.open(path, "w")
        self._sample_count = 0

    def write_sample(self, key: str, files: dict):
        for ext, data in files.items():
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            self._tar.addfile(info, io.BytesIO(data))
        self._sample_count += 1
        self._total += 1
        if self._sample_count >= self.samples_per_shard:
            self._shard_idx += 1
            self._open_new_shard()

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    @property
    def total_written(self):
        return self._total


def load_captions(csv_path: Path) -> dict:
    """Load video->caption mapping from OpenVid-1M.csv."""
    print("[OpenVid] Loading captions CSV...", flush=True)
    captions = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            captions[row["video"]] = row["caption"]
    print(f"[OpenVid] Loaded {len(captions)} captions", flush=True)
    return captions


def process_zip_part(
    part_num: int,
    zip_path: Path,
    captions: dict,
    writer: ShardWriter,
    tmp_dir: Path,
    workers: int = 6,
    max_frames: int = 64,
) -> tuple[int, int]:
    """Process one zip part: extract videos, ffmpeg frames, pack shards."""
    extract_dir = tmp_dir / f"extract_{part_num}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"  [Part {part_num}] Extracting zip...", flush=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            mp4_names = [n for n in zf.namelist() if n.endswith(".mp4")]
            zf.extractall(extract_dir, members=mp4_names)
    except (zipfile.BadZipFile, Exception) as e:
        print(f"  [Part {part_num}] Bad zip: {e}", flush=True)
        subprocess.run(["rm", "-rf", str(extract_dir)], capture_output=True)
        return 0, 0

    videos = list(extract_dir.rglob("*.mp4"))
    print(f"  [Part {part_num}] {len(videos)} videos extracted in {time.time()-t0:.0f}s", flush=True)

    work_items = []
    for vp in videos:
        caption = captions.get(vp.name, "")
        if caption:
            work_items.append((vp, caption, tmp_dir, max_frames))

    if not work_items:
        print(f"  [Part {part_num}] No captioned videos found!", flush=True)
        subprocess.run(["rm", "-rf", str(extract_dir)], capture_output=True)
        return 0, len(videos)

    processed = 0
    errors = 0
    t1 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_video, item): item for item in work_items}

        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                result = None
                errors += 1

            if result is not None:
                sample_key = f"{writer.total_written:08d}"
                writer.write_sample(sample_key, result)
                processed += 1
            else:
                errors += 1

            total = processed + errors
            if total > 0 and total % 1000 == 0:
                elapsed = time.time() - t1
                rate = processed / max(elapsed, 1)
                print(
                    f"    [Part {part_num}] {processed}/{total} "
                    f"({rate:.1f} vid/s, {errors} err)",
                    flush=True,
                )

    subprocess.run(["rm", "-rf", str(extract_dir)], capture_output=True)

    elapsed = time.time() - t0
    print(
        f"  [Part {part_num}] Done: {processed} clips, {errors} errors, "
        f"{elapsed:.0f}s ({processed/max(elapsed,1):.1f} vid/s)",
        flush=True,
    )
    return processed, errors


def download_worker(
    parts_q: queue.Queue,
    ready_q: queue.Queue,
    download_dir: Path,
):
    """Background worker: download zip parts and put them in ready_q."""
    while True:
        try:
            part_num = parts_q.get(timeout=10)
        except queue.Empty:
            break

        t0 = time.time()
        print(f"  [DL] Starting download of part {part_num}...", flush=True)
        zip_path = download_zip_part(part_num, download_dir)
        elapsed = time.time() - t0

        if zip_path and zip_path.exists():
            size_gb = zip_path.stat().st_size / (1024**3)
            speed = size_gb / max(elapsed, 1) * 1024
            print(
                f"  [DL] Part {part_num}: {size_gb:.1f}GB in {elapsed:.0f}s ({speed:.0f} MB/s)",
                flush=True,
            )
        else:
            print(f"  [DL] Part {part_num}: FAILED after {elapsed:.0f}s", flush=True)

        ready_q.put((part_num, zip_path))
        parts_q.task_done()


def parse_parts(parts_str: str) -> list[int]:
    """Parse part range like '0-69' or '5' or '0-9,20-29'."""
    result = []
    for segment in parts_str.split(","):
        segment = segment.strip()
        if "-" in segment:
            start, end = segment.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(segment))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default="0-185", help="Zip parts to process (e.g., 0-69, 5, 0-9,20-29)")
    parser.add_argument("--dl-workers", type=int, default=4, help="Parallel zip download threads")
    parser.add_argument("--ffmpeg-workers", type=int, default=6, help="Parallel ffmpeg workers per part")
    parser.add_argument("--output", default="/workspace/data/openvid")
    parser.add_argument("--max-frames", type=int, default=64, help="Max frames per video")
    parser.add_argument("--csv", default="", help="Path to OpenVid-1M.csv (auto-downloads if missing)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing shards")
    args = parser.parse_args()

    parts = parse_parts(args.parts)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/workspace/tmp/openvid_dl")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path("/workspace/tmp/openvid_zips")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get captions CSV
    csv_path = Path(args.csv) if args.csv else output_dir / "OpenVid-1M.csv"
    if not csv_path.exists():
        print("[OpenVid] Downloading captions CSV...", flush=True)
        ok = download_file(CSV_URL, csv_path, "CSV")
        if not ok:
            print("FATAL: Could not download OpenVid-1M.csv", flush=True)
            sys.exit(1)

    captions = load_captions(csv_path)
    if len(captions) < 100_000:
        print(
            f"FATAL: CSV only has {len(captions)} entries (expected ~1M). "
            "Deleting and re-downloading.",
            flush=True,
        )
        csv_path.unlink()
        ok = download_file(CSV_URL, csv_path, "CSV retry")
        if not ok:
            print("FATAL: Could not download OpenVid-1M.csv", flush=True)
            sys.exit(1)
        captions = load_captions(csv_path)

    # Step 2: Resume support
    progress_file = output_dir / "progress.json"
    completed_parts = set()
    start_shard = 0
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            completed_parts = set(progress.get("completed_parts", []))
            start_shard = progress.get("next_shard", 0)

    if args.resume:
        existing = sorted(output_dir.glob("*.tar"))
        if existing:
            last_shard = int(existing[-1].stem) + 1
            start_shard = max(start_shard, last_shard)

    # Filter out completed parts
    remaining_parts = [p for p in parts if p not in completed_parts]
    print(
        f"[OpenVid] {len(remaining_parts)} parts to process "
        f"({len(completed_parts)} already done), {args.dl_workers} download threads, "
        f"{args.ffmpeg_workers} ffmpeg workers",
        flush=True,
    )
    print(f"[OpenVid] Output: {output_dir}, starting shard: {start_shard}", flush=True)

    if not remaining_parts:
        print("[OpenVid] Nothing to do!", flush=True)
        return

    writer = ShardWriter(str(output_dir), samples_per_shard=1000, start_shard=start_shard)

    # Step 3: Setup parallel download pipeline
    parts_q = queue.Queue()
    ready_q = queue.Queue()

    for p in remaining_parts:
        parts_q.put(p)

    # Start download workers
    dl_threads = []
    for i in range(min(args.dl_workers, len(remaining_parts))):
        t = threading.Thread(
            target=download_worker,
            args=(parts_q, ready_q, download_dir),
            daemon=True,
        )
        t.start()
        dl_threads.append(t)

    # Step 4: Process parts as they arrive
    total_clips = 0
    total_errors = 0
    parts_done = 0
    t0 = time.time()

    for _ in range(len(remaining_parts)):
        try:
            part_num, zip_path = ready_q.get(timeout=7200)
        except queue.Empty:
            print("[OpenVid] Timed out waiting for downloads", flush=True)
            break

        elapsed = time.time() - t0
        parts_done += 1

        if total_clips > 0 and parts_done > 1:
            clips_per_part = total_clips / (parts_done - 1)
            remaining = len(remaining_parts) - parts_done
            # Time per part = elapsed / parts_done (includes download overlap)
            time_per_part = elapsed / parts_done
            eta_h = remaining * time_per_part / 3600
            eta_str = f", ETA ~{eta_h:.1f}h"
        else:
            eta_str = ""

        print(
            f"\n[OpenVid] === Part {part_num} ({parts_done}/{len(remaining_parts)}) === "
            f"Total: {total_clips} clips, {elapsed/60:.0f}min{eta_str}",
            flush=True,
        )

        if zip_path is None:
            print(f"  [Part {part_num}] Download FAILED, skipping", flush=True)
            total_errors += 1
            continue

        # Process this part
        clips, errors = process_zip_part(
            part_num, zip_path, captions, writer, tmp_dir,
            args.ffmpeg_workers, args.max_frames,
        )
        total_clips += clips
        total_errors += errors

        # Delete zip immediately
        zip_path.unlink(missing_ok=True)

        # Save progress
        completed_parts.add(part_num)
        with open(progress_file, "w") as f:
            json.dump({
                "completed_parts": sorted(completed_parts),
                "total_clips": total_clips,
                "total_errors": total_errors,
                "next_shard": writer._shard_idx,
            }, f)

        # Quick disk check
        if parts_done % 10 == 0:
            try:
                stat = os.statvfs("/workspace")
                free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
                print(f"  [Disk] {free_gb:.0f} GB free", flush=True)
            except Exception:
                pass

    # Wait for download threads to finish
    for t in dl_threads:
        t.join(timeout=10)

    writer.close()
    elapsed = time.time() - t0

    print(
        f"\n[OpenVid] DONE: {total_clips} clips from {parts_done} parts, "
        f"{total_errors} errors, {writer._shard_idx} shards, {elapsed/3600:.1f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
