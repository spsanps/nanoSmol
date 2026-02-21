#!/usr/bin/env python
"""
Expand The Cauldron dataset from 851K to 1.5-2M samples.

Processes subsets NOT in the original precompute.py run, then re-iterates
existing subsets to capture any samples beyond the original 1M cap.

Output goes to the SAME directory (cauldron_full/) starting from shard 851,
so training configs don't need changing.

Usage:
    python release/scripts/expand_cauldron.py
    python release/scripts/expand_cauldron.py --target 2000000
    python release/scripts/expand_cauldron.py --output /workspace/data/cauldron_full
"""

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import shared utilities from precompute.py
from release.scripts.precompute import (
    get_tokenizer,
    tokenize_sft,
    ShardWriter,
)


# All 50 Cauldron subsets
ALL_SUBSETS = [
    "ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "clevr_math",
    "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa",
    "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam",
    "iconqa", "infographic_vqa", "intergps", "localized_narratives",
    "mapqa", "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa", "okvqa",
    "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql",
    "robut_wtq", "scienceqa", "screen2words", "spot_the_diff", "st_vqa",
    "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa",
    "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr",
    "websight",
]

# Subsets already processed in the original run (32)
ORIGINAL_SUBSETS = {
    "ai2d", "aokvqa", "chart2text", "clevr", "docvqa",
    "dvqa", "figureqa", "geomverse", "hateful_memes",
    "infographic_vqa", "intergps", "localized_narratives",
    "mapqa", "ocrvqa", "okvqa", "plotqa", "raven",
    "scienceqa", "screen2words", "st_vqa", "tabmwp",
    "tallyqa", "textcaps", "textvqa", "tqa",
    "visual7w", "visualmrc", "vistext", "vqarad",
    "vqav2", "vsr", "websight",
}

# New subsets not in original run (18)
NEW_SUBSETS = [s for s in ALL_SUBSETS if s not in ORIGINAL_SUBSETS]


def process_sample(sample, subset_name, tokenizer):
    """Process a single Cauldron sample. Returns (meta, img_bytes) or None."""
    from PIL import Image

    images = sample.get("images", [])
    texts = sample.get("texts", [])

    if not images or not texts:
        return None

    # Extract Q&A from texts
    user_text = ""
    assistant_text = ""
    for entry in texts:
        if entry.get("user"):
            user_text = entry["user"]
        if entry.get("assistant"):
            assistant_text = entry["assistant"]

    if not user_text or not assistant_text:
        return None

    # Process first image
    img = images[0]
    if not isinstance(img, Image.Image):
        return None

    # Resize/crop to 224x224
    img = img.convert("RGB")
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    img = img.resize((224, 224), Image.LANCZOS)

    # Save as JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    # Tokenize (answer-only loss for Stage 2)
    tok = tokenize_sft(user_text, assistant_text, stage=2, tokenizer=tokenizer)

    meta = {
        "token_ids": tok["token_ids"],
        "loss_mask": tok["loss_mask"],
        "source": f"cauldron/{subset_name}",
        "frame_count": 1,
    }

    return meta, img_bytes


def safe_iterate(dataset):
    """Iterate over a streaming dataset, catching decode errors."""
    it = iter(dataset)
    consecutive_errors = 0
    while True:
        try:
            sample = next(it)
            consecutive_errors = 0
            yield sample
        except StopIteration:
            break
        except Exception:
            consecutive_errors += 1
            if consecutive_errors > 1000:
                # Too many consecutive errors — subset is probably broken
                break
            continue


def count_existing_per_subset(data_dir: str, sample_shards: int = 20):
    """
    Sample existing shards to estimate per-subset counts.
    Returns dict of {subset_name: estimated_count}.
    """
    import random
    data_path = Path(data_dir)
    tar_files = sorted(data_path.glob("*.tar"))
    if not tar_files:
        return {}

    # Sample random shards
    sampled = random.sample(tar_files, min(sample_shards, len(tar_files)))
    subset_counts = {}
    total_sampled = 0

    for tf in sampled:
        try:
            with tarfile.open(str(tf), "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".json"):
                        f = tar.extractfile(member)
                        if f:
                            data = json.loads(f.read())
                            source = data.get("source", "")
                            if source.startswith("cauldron/"):
                                sub = source.split("/", 1)[1]
                                subset_counts[sub] = subset_counts.get(sub, 0) + 1
                            total_sampled += 1
        except Exception:
            continue

    if total_sampled == 0:
        return {}

    # Scale up to total
    total_shards = len(tar_files)
    scale = total_shards / len(sampled)
    return {k: int(v * scale) for k, v in subset_counts.items()}


def expand_cauldron(output_dir: str, target_new: int = 1_150_000, start_shard: int = 851):
    """
    Expand Cauldron dataset.

    Args:
        output_dir: Directory to write new shards (same as existing cauldron_full)
        target_new: Target number of NEW samples to add
        start_shard: Shard index to start from (851 = continue from existing)
    """
    from datasets import load_dataset

    tokenizer = get_tokenizer()

    # Create writer starting from the right shard index
    writer = ShardWriter(output_dir, samples_per_shard=1000)
    writer._shard_idx = start_shard  # Continue from existing shards

    count = 0
    errors = 0
    t0 = time.time()
    subset_stats = {}

    # Phase 1: Process NEW subsets (18 subsets not in original run)
    print(f"[Cauldron Expand] Phase 1: Processing {len(NEW_SUBSETS)} new subsets")
    print(f"  New subsets: {NEW_SUBSETS}")

    for subset in NEW_SUBSETS:
        if count >= target_new:
            break

        print(f"\n  [Cauldron] Loading NEW subset: {subset}...", flush=True)
        sub_count = 0
        sub_errors = 0

        try:
            ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True)
        except Exception as e:
            print(f"    Skipping {subset}: {e}")
            continue

        for sample in safe_iterate(ds):
            if count >= target_new:
                break

            try:
                result = process_sample(sample, subset, tokenizer)
                if result is None:
                    continue

                meta, img_bytes = result
                sample_key = f"{(851000 + count):08d}"
                writer.write_sample(sample_key, {
                    "json": json.dumps(meta).encode("utf-8"),
                    "jpg": img_bytes,
                })

                count += 1
                sub_count += 1

                if count % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = count / max(elapsed, 1)
                    print(f"    [{subset}] {count} total new samples ({rate:.0f} samp/s, "
                          f"shard {writer._shard_idx})", flush=True)

            except Exception as e:
                sub_errors += 1
                errors += 1
                continue

        subset_stats[subset] = {"count": sub_count, "errors": sub_errors, "phase": "new"}
        elapsed = time.time() - t0
        rate = count / max(elapsed, 1) if count > 0 else 0
        print(f"  [{subset}] Done: {sub_count} samples, {sub_errors} errors "
              f"({count} total, {rate:.0f} samp/s)")

    phase1_count = count
    print(f"\n[Cauldron Expand] Phase 1 complete: {phase1_count} new samples from {len(NEW_SUBSETS)} subsets")

    # Phase 2: Re-iterate EXISTING subsets to capture data beyond original 1M cap
    if count < target_new:
        remaining = target_new - count
        print(f"\n[Cauldron Expand] Phase 2: Re-iterating existing subsets for {remaining} more samples")

        # We need to know how many samples each existing subset contributed
        # The original run capped at 1M total across all 32 subsets
        # Some subsets may have much more data available
        # Strategy: iterate each existing subset, skip estimated already-processed samples

        # Estimate per-subset counts from existing shards
        print("  Estimating existing per-subset counts...")
        existing_counts = count_existing_per_subset(output_dir)
        print(f"  Estimated existing counts: {json.dumps(existing_counts, indent=2)}")

        for subset in ORIGINAL_SUBSETS:
            if count >= target_new:
                break

            existing = existing_counts.get(subset, 0)
            print(f"\n  [Cauldron] Re-iterating subset: {subset} (skip ~{existing})...", flush=True)
            sub_count = 0
            sub_errors = 0
            skipped = 0

            try:
                ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True)
            except Exception as e:
                print(f"    Skipping {subset}: {e}")
                continue

            for sample in safe_iterate(ds):
                if count >= target_new:
                    break

                # Skip samples we've already processed
                if skipped < existing:
                    skipped += 1
                    continue

                try:
                    result = process_sample(sample, subset, tokenizer)
                    if result is None:
                        continue

                    meta, img_bytes = result
                    sample_key = f"{(851000 + count):08d}"
                    writer.write_sample(sample_key, {
                        "json": json.dumps(meta).encode("utf-8"),
                        "jpg": img_bytes,
                    })

                    count += 1
                    sub_count += 1

                    if count % 5000 == 0:
                        elapsed = time.time() - t0
                        rate = count / max(elapsed, 1)
                        print(f"    [{subset}] {count} total new samples ({rate:.0f} samp/s, "
                              f"shard {writer._shard_idx})", flush=True)

                except Exception as e:
                    sub_errors += 1
                    errors += 1
                    continue

            if sub_count > 0:
                subset_stats[subset] = {"count": sub_count, "errors": sub_errors,
                                         "skipped": existing, "phase": "existing"}
                print(f"  [{subset}] Done: {sub_count} NEW samples (skipped {existing})")
            else:
                print(f"  [{subset}] Exhausted (no new samples beyond {existing})")

    writer.close()
    elapsed = time.time() - t0

    # Save progress
    progress = {
        "new_samples": count,
        "new_shards": writer._shard_idx - start_shard,
        "start_shard": start_shard,
        "end_shard": writer._shard_idx,
        "errors": errors,
        "elapsed_seconds": int(elapsed),
        "phase1_count": phase1_count,
        "phase2_count": count - phase1_count,
        "subset_stats": subset_stats,
    }
    progress_path = Path(output_dir) / "expansion_progress.json"
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[Cauldron Expand] COMPLETE")
    print(f"  New samples: {count}")
    print(f"  New shards: {writer._shard_idx - start_shard} (shards {start_shard}-{writer._shard_idx})")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed/3600:.1f}h")
    print(f"  Total Cauldron samples: {851000 + count}")
    print(f"  Progress saved to: {progress_path}")
    print(f"{'='*60}")

    return count


def main():
    parser = argparse.ArgumentParser(description="Expand Cauldron dataset")
    parser.add_argument("--output", default="/workspace/data/cauldron_full",
                        help="Output directory (default: cauldron_full)")
    parser.add_argument("--target", type=int, default=1_150_000,
                        help="Target NEW samples to add (default: 1.15M → total ~2M)")
    parser.add_argument("--start-shard", type=int, default=0,
                        help="Override start shard (0 = auto-detect from existing)")
    args = parser.parse_args()

    # Auto-detect start shard from existing data
    if args.start_shard == 0:
        existing = sorted(Path(args.output).glob("*.tar"))
        if existing:
            last = int(existing[-1].stem)
            args.start_shard = last + 1
            print(f"[Cauldron Expand] Auto-detected start shard: {args.start_shard} "
                  f"(after {len(existing)} existing shards)")
        else:
            args.start_shard = 0

    expand_cauldron(args.output, target_new=args.target, start_shard=args.start_shard)


if __name__ == "__main__":
    main()
