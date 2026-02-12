#!/usr/bin/env python
"""
Build the frozen 10K evaluation set from multiple data sources.

Target composition:
  - 3K Cauldron (VQA, images)
  - 2K OpenVid (video captioning, multi-frame)
  - 2K LLaVA YouTube / stage3_youtube (video QA, multi-frame)
  - 2K SmolTalk (text retention, text-only)
  - 1K WebVid custom (valid video captioning)

All samples are pre-tokenized so evaluate.py doesn't need a tokenizer.

Usage:
    python release/scripts/build_eval_set.py --output /workspace/data/eval/val_10k
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from release.scripts.precompute import tokenize_stage1, tokenize_sft, get_tokenizer


# ------------------------------------------------------------------ #
# ShardWriter helper (same as in precompute.py)
# ------------------------------------------------------------------ #

class ShardWriter:
    """Write webdataset-compatible tar shards."""

    def __init__(self, output_dir: str, samples_per_shard: int = 1000, prefix: str = ""):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.prefix = prefix
        os.makedirs(output_dir, exist_ok=True)
        self.shard_idx = 0
        self.sample_idx = 0
        self.current_tar = None
        self.count_in_shard = 0

    def _open_shard(self):
        if self.current_tar is not None:
            self.current_tar.close()
        name = f"{self.prefix}{self.shard_idx:05d}.tar"
        path = os.path.join(self.output_dir, name)
        self.current_tar = tarfile.open(path, "w")
        self.count_in_shard = 0

    def add_sample(self, sample_dict: dict):
        """Add a sample. Keys should be like 'jpg', '000.jpg', 'json', etc."""
        if self.current_tar is None or self.count_in_shard >= self.samples_per_shard:
            self._open_shard()
            self.shard_idx += 1

        key = f"{self.sample_idx:08d}"
        for ext, data in sample_dict.items():
            if isinstance(data, dict):
                data = json.dumps(data).encode("utf-8")
            elif isinstance(data, str):
                data = data.encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            self.current_tar.addfile(info, io.BytesIO(data))

        self.sample_idx += 1
        self.count_in_shard += 1

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()

    @property
    def total_samples(self):
        return self.sample_idx


# ------------------------------------------------------------------ #
# Source samplers
# ------------------------------------------------------------------ #

def sample_from_cauldron(n: int, data_dir: str = "/workspace/data/cauldron_full") -> list:
    """Sample n random samples from Cauldron shards."""
    print(f"  Sampling {n} from Cauldron...")
    tars = sorted([f for f in os.listdir(data_dir) if f.endswith(".tar")])
    random.shuffle(tars)

    samples = []
    for tar_name in tars:
        if len(samples) >= n:
            break
        tar_path = os.path.join(data_dir, tar_name)
        try:
            with tarfile.open(tar_path) as tf:
                # Group files by sample key
                by_key = {}
                for m in tf.getmembers():
                    if "." not in m.name:
                        continue
                    key = m.name.rsplit(".", 1)[0]
                    ext = m.name.rsplit(".", 1)[1]
                    if key not in by_key:
                        by_key[key] = {}
                    by_key[key][ext] = tf.extractfile(m).read()

                for key in sorted(by_key.keys()):
                    if len(samples) >= n:
                        break
                    s = by_key[key]
                    if "json" not in s or "jpg" not in s:
                        continue

                    meta = json.loads(s["json"])
                    # Ensure pre-tokenized
                    if "token_ids" not in meta or "loss_mask" not in meta:
                        continue

                    meta["eval_source"] = "cauldron"
                    sample_out = {"json": meta, "jpg": s["jpg"]}
                    samples.append(sample_out)
        except Exception as e:
            continue

    random.shuffle(samples)
    print(f"    Got {len(samples)} Cauldron samples")
    return samples[:n]


def sample_from_openvid(n: int, data_dir: str = "/workspace/data/openvid") -> list:
    """Sample n random samples from OpenVid shards, tokenizing on-the-fly."""
    print(f"  Sampling {n} from OpenVid...")
    tars = sorted([f for f in os.listdir(data_dir) if f.endswith(".tar")])
    if not tars:
        print("    WARNING: No OpenVid shards found")
        return []
    random.shuffle(tars)

    tokenizer = get_tokenizer()
    samples = []
    for tar_name in tars:
        if len(samples) >= n:
            break
        tar_path = os.path.join(data_dir, tar_name)
        try:
            with tarfile.open(tar_path) as tf:
                by_key = {}
                for m in tf.getmembers():
                    if "." not in m.name:
                        continue
                    # Handle numbered frames: key.000.jpg → key
                    parts = m.name.split(".")
                    if len(parts) == 3 and parts[1].isdigit():
                        key = parts[0]
                        ext = f"{parts[1]}.{parts[2]}"
                    else:
                        key = parts[0]
                        ext = parts[1]
                    if key not in by_key:
                        by_key[key] = {}
                    by_key[key][ext] = tf.extractfile(m).read()

                for key in sorted(by_key.keys()):
                    if len(samples) >= n:
                        break
                    s = by_key[key]
                    if "json" not in s:
                        continue

                    meta = json.loads(s["json"])
                    caption = meta.get("caption", "")
                    if not caption:
                        continue

                    # Collect frame data
                    frame_keys = sorted([k for k in s.keys() if k.endswith(".jpg") and k != "jpg"])
                    if not frame_keys and "jpg" in s:
                        frame_keys = ["jpg"]
                    if not frame_keys:
                        continue

                    # Tokenize as stage 1 (video captioning)
                    tok = tokenize_stage1(caption, tokenizer=tokenizer)

                    meta["token_ids"] = tok["token_ids"]
                    meta["loss_mask"] = tok["loss_mask"]
                    meta["eval_source"] = "openvid"
                    meta["frame_count"] = len(frame_keys)

                    sample_out = {"json": meta}
                    for fk in frame_keys:
                        sample_out[fk] = s[fk]
                    samples.append(sample_out)
        except Exception as e:
            continue

    random.shuffle(samples)
    print(f"    Got {len(samples)} OpenVid samples")
    return samples[:n]


def sample_from_youtube(n: int, data_dir: str = "/workspace/data/stage3_youtube") -> list:
    """Sample n random samples from LLaVA YouTube shards."""
    print(f"  Sampling {n} from LLaVA YouTube...")
    tars = sorted([f for f in os.listdir(data_dir) if f.endswith(".tar")])
    if not tars:
        print("    WARNING: No YouTube shards found")
        return []
    random.shuffle(tars)

    samples = []
    for tar_name in tars:
        if len(samples) >= n:
            break
        tar_path = os.path.join(data_dir, tar_name)
        try:
            with tarfile.open(tar_path) as tf:
                by_key = {}
                for m in tf.getmembers():
                    if "." not in m.name:
                        continue
                    parts = m.name.split(".")
                    if len(parts) == 3 and parts[1].isdigit():
                        key = parts[0]
                        ext = f"{parts[1]}.{parts[2]}"
                    else:
                        key = parts[0]
                        ext = parts[1]
                    if key not in by_key:
                        by_key[key] = {}
                    by_key[key][ext] = tf.extractfile(m).read()

                for key in sorted(by_key.keys()):
                    if len(samples) >= n:
                        break
                    s = by_key[key]
                    if "json" not in s:
                        continue

                    meta = json.loads(s["json"])
                    if "token_ids" not in meta or "loss_mask" not in meta:
                        continue

                    # Collect frames
                    frame_keys = sorted([k for k in s.keys() if k.endswith(".jpg") and k != "jpg"])
                    if not frame_keys and "jpg" in s:
                        frame_keys = ["jpg"]
                    if not frame_keys:
                        continue

                    meta["eval_source"] = "llava_youtube"
                    sample_out = {"json": meta}
                    for fk in frame_keys:
                        sample_out[fk] = s[fk]
                    samples.append(sample_out)
        except Exception as e:
            continue

    random.shuffle(samples)
    print(f"    Got {len(samples)} YouTube samples")
    return samples[:n]


def sample_from_smoltalk(n: int, data_dir: str = "/workspace/data/text_retention") -> list:
    """Sample n text-only samples from SmolTalk."""
    print(f"  Sampling {n} from SmolTalk...")
    # Pool from all 3 stages
    tars = []
    for stage_dir in ["stage1", "stage2", "stage3"]:
        d = os.path.join(data_dir, stage_dir)
        if os.path.isdir(d):
            tars.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tar")])

    if not tars:
        print("    WARNING: No SmolTalk shards found")
        return []
    random.shuffle(tars)

    samples = []
    for tar_path in tars:
        if len(samples) >= n:
            break
        try:
            with tarfile.open(tar_path) as tf:
                by_key = {}
                for m in tf.getmembers():
                    if "." not in m.name:
                        continue
                    key = m.name.rsplit(".", 1)[0]
                    ext = m.name.rsplit(".", 1)[1]
                    if key not in by_key:
                        by_key[key] = {}
                    by_key[key][ext] = tf.extractfile(m).read()

                for key in sorted(by_key.keys()):
                    if len(samples) >= n:
                        break
                    s = by_key[key]
                    if "json" not in s:
                        continue

                    meta = json.loads(s["json"])
                    if "token_ids" not in meta or "loss_mask" not in meta:
                        continue

                    meta["eval_source"] = "smoltalk"
                    # Text-only: create a 1x3x224x224 blank frame as placeholder
                    # The model handles frame_count=0 or we provide a dummy
                    meta["frame_count"] = 0
                    sample_out = {"json": meta}
                    # Add a tiny 1x1 JPEG placeholder so the dataloader doesn't skip it
                    if "jpg" in s:
                        sample_out["jpg"] = s["jpg"]
                    samples.append(sample_out)
        except Exception as e:
            continue

    random.shuffle(samples)
    print(f"    Got {len(samples)} SmolTalk samples")
    return samples[:n]


def sample_from_webvid(n: int, data_dir: str = "/workspace/data/webvid") -> list:
    """Sample n from the valid WebVid custom shards."""
    print(f"  Sampling {n} from WebVid custom...")
    tars = sorted([f for f in os.listdir(data_dir) if f.endswith(".tar")])
    if not tars:
        print("    WARNING: No WebVid shards found")
        return []
    random.shuffle(tars)

    tokenizer = get_tokenizer()
    samples = []
    for tar_name in tars:
        if len(samples) >= n:
            break
        tar_path = os.path.join(data_dir, tar_name)
        try:
            with tarfile.open(tar_path) as tf:
                by_key = {}
                for m in tf.getmembers():
                    if "." not in m.name:
                        continue
                    parts = m.name.split(".")
                    if len(parts) == 3 and parts[1].isdigit():
                        key = parts[0]
                        ext = f"{parts[1]}.{parts[2]}"
                    else:
                        key = parts[0]
                        ext = parts[1]
                    if key not in by_key:
                        by_key[key] = {}
                    by_key[key][ext] = tf.extractfile(m).read()

                for key in sorted(by_key.keys()):
                    if len(samples) >= n:
                        break
                    s = by_key[key]
                    if "json" not in s:
                        continue

                    meta = json.loads(s["json"])

                    # WebVid custom may or may not have token_ids
                    if "token_ids" not in meta:
                        caption = meta.get("caption", "")
                        if not caption:
                            continue
                        tok = tokenize_stage1(caption, tokenizer=tokenizer)
                        meta["token_ids"] = tok["token_ids"]
                        meta["loss_mask"] = tok["loss_mask"]

                    frame_keys = sorted([k for k in s.keys() if k.endswith(".jpg") and k != "jpg"])
                    if not frame_keys and "jpg" in s:
                        frame_keys = ["jpg"]
                    if not frame_keys:
                        continue

                    meta["eval_source"] = "webvid"
                    meta["frame_count"] = len(frame_keys)

                    sample_out = {"json": meta}
                    for fk in frame_keys:
                        sample_out[fk] = s[fk]
                    samples.append(sample_out)
        except Exception as e:
            continue

    random.shuffle(samples)
    print(f"    Got {len(samples)} WebVid samples")
    return samples[:n]


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Build frozen 10K eval set")
    parser.add_argument("--output", default="/workspace/data/eval/val_10k",
                        help="Output directory for eval shards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=== Building 10K Eval Set ===")
    t0 = time.time()

    # Clean output dir
    os.makedirs(args.output, exist_ok=True)
    for f in os.listdir(args.output):
        if f.endswith(".tar") or f == "manifest.json":
            os.remove(os.path.join(args.output, f))

    # Sample from each source
    all_samples = []

    cauldron = sample_from_cauldron(3000)
    all_samples.extend(cauldron)

    openvid = sample_from_openvid(2000)
    all_samples.extend(openvid)

    youtube = sample_from_youtube(2000)
    all_samples.extend(youtube)

    smoltalk = sample_from_smoltalk(2000)
    all_samples.extend(smoltalk)

    webvid = sample_from_webvid(1000)
    all_samples.extend(webvid)

    print(f"\n  Total: {len(all_samples)} samples")

    # Shuffle and write
    random.shuffle(all_samples)

    writer = ShardWriter(args.output, samples_per_shard=args.samples_per_shard)
    for s in all_samples:
        writer.add_sample(s)
    writer.close()

    elapsed = time.time() - t0

    # Write manifest
    from collections import Counter
    sources = Counter()
    for s in all_samples:
        meta = s["json"] if isinstance(s["json"], dict) else json.loads(s["json"])
        sources[meta.get("eval_source", "unknown")] += 1

    manifest = {
        "total_samples": len(all_samples),
        "num_shards": writer.shard_idx,
        "source_distribution": dict(sources),
        "seed": args.seed,
        "frozen": True,
    }

    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"  Shards: {writer.shard_idx}")
    print(f"  Samples: {len(all_samples)}")
    print(f"  Sources: {dict(sources)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
