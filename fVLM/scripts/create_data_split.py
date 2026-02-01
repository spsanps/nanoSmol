#!/usr/bin/env python3
"""Create deterministic train/val split of sharded data.

Produces a JSON file listing which shards belong to train vs val.
All training scripts must load this split to prevent data leakage.
"""

import json
import random
from pathlib import Path

SHARD_DIR = Path("/mnt/d/projects/fVLM/data/frames_latents_sharded")
OUTPUT_PATH = Path(__file__).parent.parent / "configs" / "data_split.json"
SEED = 42
VAL_SHARDS = 25  # ~5,000 samples at 200/shard


def main():
    # List all shards, sorted for determinism
    shard_files = sorted(SHARD_DIR.glob("shard_*.pt"))
    shard_names = [f.name for f in shard_files]
    print(f"Found {len(shard_names)} shards in {SHARD_DIR}")

    # Deterministic shuffle
    rng = random.Random(SEED)
    indices = list(range(len(shard_names)))
    rng.shuffle(indices)

    # Split: last VAL_SHARDS for validation
    val_indices = indices[-VAL_SHARDS:]
    train_indices = indices[:-VAL_SHARDS]

    val_shards = sorted([shard_names[i] for i in val_indices])
    train_shards = sorted([shard_names[i] for i in train_indices])

    # Verify no overlap
    assert len(set(val_shards) & set(train_shards)) == 0, "Data leak: overlap between train and val!"
    assert len(val_shards) + len(train_shards) == len(shard_names), "Missing shards!"

    split = {
        "seed": SEED,
        "shard_dir": str(SHARD_DIR),
        "total_shards": len(shard_names),
        "val_count": len(val_shards),
        "train_count": len(train_shards),
        "val_shards": val_shards,
        "train_shards": train_shards,
        "samples_per_shard": 200,
        "approx_val_samples": len(val_shards) * 200,
        "approx_train_samples": len(train_shards) * 200,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit saved to {OUTPUT_PATH}")
    print(f"  Train: {len(train_shards)} shards (~{len(train_shards) * 200} samples)")
    print(f"  Val:   {len(val_shards)} shards (~{len(val_shards) * 200} samples)")
    print(f"\nFirst 5 val shards: {val_shards[:5]}")
    print(f"First 5 train shards: {train_shards[:5]}")


if __name__ == "__main__":
    main()
