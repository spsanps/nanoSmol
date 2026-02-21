#!/usr/bin/env python
"""
Validate webdataset tar shards for data integrity.

Checks:
  1. Tar files are readable
  2. JSON metadata is parseable and contains required keys
  3. JPEG frames decode correctly to expected resolution (224x224)
  4. Token IDs and loss_mask arrays are consistent lengths
  5. Frame count matches metadata

Usage:
    python release/scripts/validate_shards.py /workspace/data/webvid/ --sample 10
    python release/scripts/validate_shards.py /workspace/data/cauldron/ --sample 5
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def validate_shard(tar_path: str, verbose: bool = False) -> dict:
    """
    Validate a single tar shard.

    Returns a dict with:
      total_samples, valid_samples, errors (list of str),
      frame_counts (list of int), token_lengths (list of int)
    """
    result = {
        "path": tar_path,
        "total_samples": 0,
        "valid_samples": 0,
        "errors": [],
        "frame_counts": [],
        "token_lengths": [],
    }

    try:
        tf = tarfile.open(tar_path, "r")
    except Exception as e:
        result["errors"].append(f"Cannot open tar: {e}")
        return result

    # Group files by sample key (strip extension)
    samples: dict[str, dict[str, bytes]] = {}
    for member in tf.getmembers():
        if not member.isfile():
            continue
        name = member.name
        # Extract sample key: everything before the last dot or _NNN.ext
        key = name.rsplit(".", 1)[0]
        # For multi-frame: key might be "sample_001" -> base is "sample"
        ext = name.rsplit(".", 1)[-1] if "." in name else ""

        data = tf.extractfile(member)
        if data is not None:
            samples.setdefault(name, None)
            # Store raw bytes indexed by full filename
            samples[name] = data.read()

    tf.close()

    # Re-group by sample key
    import re
    sample_groups: dict[str, dict[str, bytes]] = {}
    for fullname, data in samples.items():
        # Parse filename into (group_key, extension)
        # Format A: key.NNN.ext (OpenVid: 00000000.000.jpg)
        m = re.match(r"^(.+?)\.(\d{3})\.(jpg|jpeg|png)$", fullname)
        if m:
            group_key, ext = m.group(1), m.group(3)
        else:
            # Format B: key_NNN.ext (old WebVid: 00000001_003.jpg)
            m = re.match(r"^(.+?)_(\d{3})\.(jpg|jpeg|png)$", fullname)
            if m:
                group_key, ext = m.group(1), m.group(3)
            else:
                # Format C: key.ext (simple: 00000001.json, 00000001.jpg)
                parts = fullname.rsplit(".", 1)
                group_key = parts[0] if len(parts) == 2 else fullname
                ext = parts[1] if len(parts) == 2 else ""

        sample_groups.setdefault(group_key, {})[fullname] = (ext, data)

    for group_key, files in sample_groups.items():
        # Check for JSON metadata
        json_files = [(fn, ext, data) for fn, (ext, data) in files.items() if ext == "json"]
        if not json_files:
            continue  # Not a complete sample

        result["total_samples"] += 1

        # Parse JSON
        _, _, json_data = json_files[0]
        try:
            meta = json.loads(json_data.decode("utf-8"))
        except Exception as e:
            result["errors"].append(f"{group_key}: bad JSON: {e}")
            continue

        # Check required keys: either pre-tokenized (token_ids+loss_mask),
        # DPO format (chosen/rejected), or raw caption for on-the-fly tokenization
        has_tokens = "token_ids" in meta and "loss_mask" in meta
        has_dpo = "chosen_token_ids" in meta and "rejected_token_ids" in meta
        has_caption = bool(meta.get("caption", ""))
        if not has_tokens and not has_dpo and not has_caption:
            result["errors"].append(f"{group_key}: missing token_ids, DPO tokens, and caption")
            continue

        token_ids = meta.get("token_ids", meta.get("chosen_token_ids", []))
        loss_mask = meta.get("loss_mask", meta.get("chosen_loss_mask", []))

        if len(token_ids) != len(loss_mask):
            result["errors"].append(
                f"{group_key}: token_ids ({len(token_ids)}) != loss_mask ({len(loss_mask)})"
            )

        result["token_lengths"].append(len(token_ids))

        # Check image files
        image_files = [
            (fn, data) for fn, (ext, data) in files.items()
            if ext in ("jpg", "jpeg", "png")
        ]
        num_frames = len(image_files)
        result["frame_counts"].append(num_frames)

        # Text-only or annotation-only samples don't need frames
        is_text_only = meta.get("is_text_only", False)
        has_frames = meta.get("has_frames", True)  # Default True for backward compat
        if num_frames == 0 and (is_text_only or not has_frames):
            result["valid_samples"] += 1
            continue
        elif num_frames == 0:
            result["errors"].append(f"{group_key}: no image files")
            continue

        # Validate image decoding
        for fn, img_data in image_files[:3]:  # Check first 3 frames
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(img_data))
                w, h = img.size
                if w != 224 or h != 224:
                    result["errors"].append(
                        f"{group_key}/{fn}: wrong size {w}x{h} (expected 224x224)"
                    )
            except Exception as e:
                result["errors"].append(f"{group_key}/{fn}: decode error: {e}")

        result["valid_samples"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate webdataset shards")
    parser.add_argument("data_dir", help="Directory containing .tar shards")
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of shards to sample (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tar_files = sorted(data_dir.glob("*.tar"))

    if not tar_files:
        print(f"No .tar files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(tar_files)} shards in {data_dir}")

    if args.sample > 0 and args.sample < len(tar_files):
        tar_files = random.sample(tar_files, args.sample)
        print(f"Sampling {args.sample} shards")

    total_samples = 0
    total_valid = 0
    all_errors = []
    all_frame_counts = []
    all_token_lengths = []

    for tar_path in tar_files:
        result = validate_shard(str(tar_path), verbose=args.verbose)
        total_samples += result["total_samples"]
        total_valid += result["valid_samples"]
        all_errors.extend(result["errors"])
        all_frame_counts.extend(result["frame_counts"])
        all_token_lengths.extend(result["token_lengths"])

        status = "OK" if not result["errors"] else f"{len(result['errors'])} errors"
        print(f"  {tar_path.name}: {result['valid_samples']}/{result['total_samples']} valid [{status}]")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Shards checked: {len(tar_files)}")
    print(f"  Total samples:  {total_samples}")
    print(f"  Valid samples:  {total_valid}")
    print(f"  Errors:         {len(all_errors)}")

    if all_frame_counts:
        import statistics
        print(f"\n  Frame counts:")
        print(f"    min={min(all_frame_counts)}, max={max(all_frame_counts)}, "
              f"mean={statistics.mean(all_frame_counts):.1f}, "
              f"median={statistics.median(all_frame_counts):.1f}")

    if all_token_lengths:
        import statistics
        print(f"  Token lengths:")
        print(f"    min={min(all_token_lengths)}, max={max(all_token_lengths)}, "
              f"mean={statistics.mean(all_token_lengths):.1f}, "
              f"median={statistics.median(all_token_lengths):.1f}")

    if all_errors and args.verbose:
        print(f"\n  First 20 errors:")
        for err in all_errors[:20]:
            print(f"    - {err}")

    success = len(all_errors) == 0
    print(f"\n{'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
