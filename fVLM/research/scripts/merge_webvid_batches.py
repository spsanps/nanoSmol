#!/usr/bin/env python
"""
Merge multiple WebVid download batches into a unified shard set.

Each batch writes to webvid_batch{N}/ with its own shard numbering.
This script copies/renames all shards into a single output directory
with sequential numbering.

Usage:
    python release/scripts/merge_webvid_batches.py \
        --inputs /workspace/data/webvid /workspace/data/webvid_batch2 /workspace/data/webvid_batch3 \
        --output /workspace/data/webvid_merged
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input directories to merge")
    parser.add_argument("--output", default="/workspace/data/webvid_merged",
                        help="Output directory for merged shards")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without copying")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    total_files = 0

    for input_dir in args.inputs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  Skipping {input_dir} (doesn't exist)")
            continue

        tar_files = sorted(input_path.glob("*.tar"))
        if not tar_files:
            print(f"  Skipping {input_dir} (no .tar files)")
            continue

        print(f"  {input_dir}: {len(tar_files)} shards")

        for src in tar_files:
            dst = output_dir / f"{shard_idx:05d}.tar"
            if args.dry_run:
                print(f"    {src} -> {dst}")
            else:
                shutil.copy2(str(src), str(dst))
            shard_idx += 1
            total_files += 1

    # Also copy any manifest.json files from the first input
    for input_dir in args.inputs:
        manifest = Path(input_dir) / "manifest.json"
        if manifest.exists() and not args.dry_run:
            print(f"  Note: manifest.json found in {input_dir} (will need regeneration)")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Merged {total_files} shards "
          f"from {len(args.inputs)} batches into {output_dir}")
    if not args.dry_run:
        print(f"  Run: python release/scripts/create_shard_manifest.py {output_dir}")


if __name__ == "__main__":
    main()
