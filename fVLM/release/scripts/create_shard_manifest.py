#!/usr/bin/env python
"""
Build a shard manifest (JSON) for bucketed batching.

Scans all tar shards in a directory, records per-shard metadata:
  - path, num_samples, frame_counts, total_tokens

Also computes the global frame-count distribution and creates a bucket
index for efficient batched sampling (group shards by frame-count ranges).

Usage:
    python release/scripts/create_shard_manifest.py /workspace/data/webvid/
    python release/scripts/create_shard_manifest.py /workspace/data/cauldron/ --output manifest.json
"""

import argparse
import json
import os
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path


def scan_shard(tar_path: str) -> dict:
    """Scan a single shard and return metadata."""
    result = {
        "path": os.path.basename(tar_path),
        "num_samples": 0,
        "frame_counts": [],
        "total_tokens": 0,
    }

    try:
        tf = tarfile.open(tar_path, "r")
    except Exception:
        return result

    # Count frames and tokens per sample
    json_data = {}
    frame_counts = Counter()

    for member in tf.getmembers():
        if not member.isfile():
            continue
        name = member.name
        base, ext = os.path.splitext(name)

        if ext == ".json":
            data = tf.extractfile(member)
            if data:
                try:
                    meta = json.loads(data.read().decode("utf-8"))
                    json_data[base] = meta
                except Exception:
                    pass
        elif ext in (".jpg", ".jpeg", ".png"):
            # Count frames per sample
            # Format A: "sample_001.jpg" -> group="sample"
            # Format B: "sample.001.jpg" -> name="sample.001.jpg", base="sample.001"
            m = re.match(r"^(.+?)(?:[._]\d{3})$", base)
            group = m.group(1) if m else base
            frame_counts[group] += 1

    tf.close()

    for base, meta in json_data.items():
        result["num_samples"] += 1
        token_ids = meta.get("token_ids", [])
        result["total_tokens"] += len(token_ids)

        # Frame count from image files
        fc = frame_counts.get(base, 1)
        result["frame_counts"].append(fc)

    return result


def main():
    parser = argparse.ArgumentParser(description="Create shard manifest")
    parser.add_argument("data_dir", help="Directory containing .tar shards")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: data_dir/manifest.json)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tar_files = sorted(data_dir.glob("*.tar"))

    if not tar_files:
        print(f"No .tar files found in {data_dir}")
        sys.exit(1)

    print(f"Scanning {len(tar_files)} shards in {data_dir} ...")

    shards = []
    total_samples = 0
    all_frame_counts = []

    for i, tar_path in enumerate(tar_files):
        info = scan_shard(str(tar_path))
        shards.append({
            "path": info["path"],
            "num_samples": info["num_samples"],
            "frame_counts": info["frame_counts"],
            "total_tokens": info["total_tokens"],
        })
        total_samples += info["num_samples"]
        all_frame_counts.extend(info["frame_counts"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tar_files)} shards scanned ...", flush=True)

    # Frame count distribution
    fc_dist = Counter(all_frame_counts)

    # Bucket index for batched sampling
    BUCKET_RANGES = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
    bucket_index = {f"{lo}-{hi}": [] for lo, hi in BUCKET_RANGES}

    for shard_idx, shard in enumerate(shards):
        for fc in shard["frame_counts"]:
            for lo, hi in BUCKET_RANGES:
                if lo <= fc <= hi:
                    bucket_index[f"{lo}-{hi}"].append(shard_idx)
                    break

    # Deduplicate bucket entries (shard may appear in multiple buckets)
    bucket_index = {k: sorted(set(v)) for k, v in bucket_index.items()}

    avg_tokens = sum(s["total_tokens"] for s in shards) / max(total_samples, 1)

    manifest = {
        "data_dir": str(data_dir),
        "num_shards": len(shards),
        "total_samples": total_samples,
        "avg_tokens_per_sample": round(avg_tokens, 1),
        "frame_count_distribution": {str(k): v for k, v in sorted(fc_dist.items())},
        "bucket_ranges": [f"{lo}-{hi}" for lo, hi in BUCKET_RANGES],
        "bucket_index": bucket_index,
        "shards": shards,
    }

    output_path = args.output or str(data_dir / "manifest.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {output_path}")
    print(f"  Shards:      {len(shards)}")
    print(f"  Samples:     {total_samples}")
    print(f"  Avg tokens:  {avg_tokens:.1f}")
    if all_frame_counts:
        print(f"  Frame range: {min(all_frame_counts)}-{max(all_frame_counts)}")


if __name__ == "__main__":
    main()
