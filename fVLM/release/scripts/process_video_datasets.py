#!/usr/bin/env python
"""
Process video datasets (Vript, LLaVA-Video) into webdataset shards.

Pipeline: extract MP4 from archives → ffmpeg 1FPS 224x224 center-crop → webdataset tar shards.

Usage:
    python release/scripts/process_video_datasets.py vript --output /workspace/data/vript_shards
    python release/scripts/process_video_datasets.py llava-video --output /workspace/data/llava_video_shards
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Shared utilities (from download_openvid.py) ──────────────────────────


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
            timeout=120,
        )
        if result.returncode != 0:
            return []
        return sorted(frames_dir.glob("frame_*.jpg"))
    except (subprocess.TimeoutExpired, Exception):
        return []


def process_single_video(args: tuple) -> dict | None:
    """Process one video: extract frames, return frame bytes + metadata."""
    video_path, caption, source_name, video_id, tmp_base, max_frames = args
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
            "source": source_name,
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
    """Write samples to webdataset tar shards."""

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


# ── Vript processing ──────────────────────────────────────────────────────


def load_vript_captions(captions_path: Path) -> dict:
    """Load Vript captions: video_id -> list of scene captions."""
    captions = {}  # video_id -> combined caption
    with open(captions_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            video_id = entry["meta"]["video_id"]
            cap = entry["caption"]
            scene_text = cap.get("content", "")
            scene_title = cap.get("scene_title", "")
            clip_caption = f"{scene_title}: {scene_text}" if scene_title else scene_text

            if video_id not in captions:
                captions[video_id] = []
            captions[video_id].append(clip_caption)

    # Combine all scenes into one caption per video
    combined = {}
    for vid, scenes in captions.items():
        combined[vid] = " ".join(scenes)
    return combined


def process_vript(args):
    """Process Vript short video clips into webdataset shards."""
    vript_dir = Path(args.input)
    clips_dir = vript_dir / "vript_short_videos_clips"
    captions_file = vript_dir / "vript_captions" / "vript_short_videos_captions.jsonl"

    if not clips_dir.exists():
        # Fall back to full videos
        clips_dir = vript_dir / "vript_short_videos"
        print("[Vript] Using full videos (clips not available)", flush=True)

    if not captions_file.exists():
        print(f"FATAL: Captions file not found: {captions_file}", flush=True)
        sys.exit(1)

    # Load clip-level captions (clip_id -> caption)
    clip_captions = {}
    with open(captions_file) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            clip_id = entry["clip_id"]
            cap = entry["caption"]
            scene_text = cap.get("content", "")
            scene_title = cap.get("scene_title", "")
            clip_captions[clip_id] = f"{scene_title}: {scene_text}" if scene_title else scene_text

    print(f"[Vript] Loaded {len(clip_captions)} clip captions", flush=True)

    output_dir = Path(args.output)
    tmp_dir = Path("/workspace/tmp/vript_process")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(str(output_dir), samples_per_shard=1000)

    # Process zip files
    zip_files = sorted(clips_dir.glob("*.zip"))
    print(f"[Vript] Found {len(zip_files)} zip archives", flush=True)

    total_clips = 0
    total_errors = 0

    for zip_path in zip_files:
        print(f"\n[Vript] Processing {zip_path.name}...", flush=True)

        # Extract to temp dir
        extract_dir = tmp_dir / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                mp4_names = [n for n in zf.namelist() if n.endswith(".mp4")]
                zf.extractall(extract_dir, members=mp4_names)
        except Exception as e:
            print(f"  Bad zip: {e}", flush=True)
            continue

        # Find all MP4s
        videos = list(extract_dir.rglob("*.mp4"))
        print(f"  {len(videos)} clips extracted", flush=True)

        # Match clips with captions
        work_items = []
        for vp in videos:
            # Clip filename format: {video_id}-Scene-{NNN}.mp4
            clip_id = vp.stem
            caption = clip_captions.get(clip_id, "")
            if not caption:
                # Try matching just by video ID
                video_id = clip_id.split("-Scene-")[0] if "-Scene-" in clip_id else clip_id
                # Use video-level combined caption as fallback
                caption = clip_captions.get(clip_id, f"Video clip from {video_id}")

            work_items.append((vp, caption, "vript", clip_id, tmp_dir, args.max_frames))

        # Process in parallel
        processed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, item): item for item in work_items}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception:
                    total_errors += 1
                    continue

                if result is None:
                    total_errors += 1
                    continue

                key = f"{writer._shard_idx:06d}_{writer._sample_count:04d}"
                writer.write_sample(key, result)
                processed += 1

        total_clips += processed
        print(f"  {processed} clips written, {total_errors} errors total", flush=True)

        # Cleanup
        subprocess.run(["rm", "-rf", str(extract_dir)], capture_output=True)

    writer.close()
    print(f"\n[Vript] DONE: {total_clips} clips, {writer._shard_idx + 1} shards, {total_errors} errors", flush=True)


# ── LLaVA-Video processing ───────────────────────────────────────────────


def process_llava_video(args):
    """Process LLaVA-Video-178K academic subset into webdataset shards."""
    llava_dir = Path(args.input) / "0_30_s_academic_v0_1"
    if not llava_dir.exists():
        llava_dir = Path(args.input)

    output_dir = Path(args.output)
    tmp_dir = Path("/workspace/tmp/llava_video_process")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load all annotations (captions + QA)
    annotations = {}  # video_path -> list of conversations

    for json_file in sorted(llava_dir.glob("*.json")):
        print(f"[LLaVA-Video] Loading {json_file.name}...", flush=True)
        with open(json_file) as f:
            data = json.load(f)
        for entry in data:
            video_path = entry["video"]
            convs = entry["conversations"]
            # Extract text from conversations
            texts = []
            for c in convs:
                if c["from"] == "gpt":
                    texts.append(c["value"])
            caption = " ".join(texts)

            if video_path not in annotations:
                annotations[video_path] = caption
            else:
                # Combine if multiple annotation types (cap + QA)
                annotations[video_path] = annotations[video_path] + " " + caption

    print(f"[LLaVA-Video] Loaded annotations for {len(annotations)} videos", flush=True)

    writer = ShardWriter(str(output_dir), samples_per_shard=1000)

    # Process tar.gz files
    tar_files = sorted(llava_dir.glob("*.tar.gz"))
    print(f"[LLaVA-Video] Found {len(tar_files)} tar.gz archives", flush=True)

    total_clips = 0
    total_errors = 0

    for tar_path in tar_files:
        print(f"\n[LLaVA-Video] Processing {tar_path.name}...", flush=True)

        extract_dir = tmp_dir / tar_path.stem.replace(".tar", "")
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".mp4") and m.isfile()]
                tf.extractall(extract_dir, members=members)
        except Exception as e:
            print(f"  Bad tar: {e}", flush=True)
            continue

        videos = list(extract_dir.rglob("*.mp4"))
        print(f"  {len(videos)} videos extracted", flush=True)

        # Match videos with annotations
        work_items = []
        matched = 0
        for vp in videos:
            # Reconstruct the relative path as it appears in annotations
            rel_path = str(vp.relative_to(extract_dir))
            caption = annotations.get(rel_path, "")
            if not caption:
                # Try just filename
                caption = annotations.get(vp.name, "")
            if not caption:
                # Generic fallback
                caption = f"A video clip"

            video_id = vp.stem
            if caption != "A video clip":
                matched += 1
            work_items.append((vp, caption, "llava_video", video_id, tmp_dir, args.max_frames))

        print(f"  {matched}/{len(videos)} matched with annotations", flush=True)

        # Process in parallel
        processed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, item): item for item in work_items}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception:
                    total_errors += 1
                    continue

                if result is None:
                    total_errors += 1
                    continue

                key = f"{writer._shard_idx:06d}_{writer._sample_count:04d}"
                writer.write_sample(key, result)
                processed += 1

        total_clips += processed
        print(f"  {processed} clips written ({total_clips} total), {total_errors} errors", flush=True)

        # Cleanup
        subprocess.run(["rm", "-rf", str(extract_dir)], capture_output=True)

    writer.close()
    print(
        f"\n[LLaVA-Video] DONE: {total_clips} clips, {writer._shard_idx + 1} shards, "
        f"{total_errors} errors",
        flush=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Process video datasets into webdataset shards")
    parser.add_argument("dataset", choices=["vript", "llava-video"], help="Dataset to process")
    parser.add_argument("--input", required=True, help="Input directory with raw data")
    parser.add_argument("--output", required=True, help="Output directory for shards")
    parser.add_argument("--workers", type=int, default=12, help="Parallel ffmpeg workers")
    parser.add_argument("--max-frames", type=int, default=64, help="Max frames per video")
    args = parser.parse_args()

    if args.dataset == "vript":
        process_vript(args)
    elif args.dataset == "llava-video":
        process_llava_video(args)


if __name__ == "__main__":
    main()
