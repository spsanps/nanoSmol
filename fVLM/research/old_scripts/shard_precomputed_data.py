#!/usr/bin/env python3
"""
Consolidate individual .pt files into large shards for faster I/O.

Before: 212K separate .pt files (~5 MB each) on NTFS → slow random reads
After: ~200 shard files (~1 GB each) → fast sequential reads

Each shard is a dict: {'samples': [list of sample dicts], 'count': int}
"""

import torch
import sys
from pathlib import Path
from tqdm import tqdm
import time
import json

SAMPLES_PER_SHARD = 200  # ~1 GB per shard (fits in 16 GB RAM)
INPUT_DIR = "/mnt/d/projects/fVLM/data/frames_latents_100k/features"
OUTPUT_DIR = "/mnt/d/projects/fVLM/data/frames_latents_sharded"


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.pt"))
    print(f"Input: {len(files)} files from {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Samples per shard: {SAMPLES_PER_SHARD}")
    print(f"Expected shards: {len(files) // SAMPLES_PER_SHARD + 1}")

    # Resume: skip files already sharded
    existing_shards = sorted(output_dir.glob("shard_*.pt"))
    if existing_shards:
        # Count total samples in existing shards
        total_existing = len(existing_shards) * SAMPLES_PER_SHARD  # Approximate
        shard_idx = len(existing_shards)
        files = files[total_existing:]
        print(f"Resuming: {shard_idx} shards exist, skipping ~{total_existing} files")
    else:
        shard_idx = 0

    print(f"Remaining: {len(files)} files -> ~{len(files) // SAMPLES_PER_SHARD + 1} shards")
    print()

    samples = []
    start_time = time.time()

    for i, f in enumerate(tqdm(files, desc="Sharding")):
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
            samples.append(data)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
            continue

        if len(samples) >= SAMPLES_PER_SHARD:
            shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
            torch.save({'samples': samples, 'count': len(samples)}, shard_path)
            shard_idx += 1
            # Explicitly free memory
            del samples
            samples = []

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(files) - i - 1) / rate
            tqdm.write(f"  Shard {shard_idx}: {rate:.0f} files/s, "
                       f"ETA: {remaining/60:.1f} min")

    # Save remaining
    if samples:
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
        torch.save({'samples': samples, 'count': len(samples)}, shard_path)
        shard_idx += 1

    elapsed = time.time() - start_time
    print(f"\nDone! {shard_idx} shards in {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")

    # Save metadata
    meta = {
        'num_shards': shard_idx,
        'samples_per_shard': SAMPLES_PER_SHARD,
        'total_samples': len(files),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
