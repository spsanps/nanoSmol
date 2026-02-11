#!/usr/bin/env python
"""
CPU data preprocessing pipeline for foveated VLM training.

Handles tokenization and webdataset packaging for all data sources:
  1. SmolTalk (text-only retention data)
  2. The Cauldron (image VQA → Stage 2)
  3. WebVid (video captioning → Stage 1)
  4. Video SFT datasets (Stage 3)

Usage:
    python release/scripts/precompute.py smoltalk --stage 1
    python release/scripts/precompute.py cauldron
    python release/scripts/precompute.py webvid --workers 6
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


# --------------------------------------------------------------------------- #
# Tokenizer setup
# --------------------------------------------------------------------------- #

_tokenizer = None

def get_tokenizer(model_path: str = "/workspace/models/SmolLM2-135M-Instruct"):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def tokenize_stage1(caption: str, tokenizer=None) -> dict:
    """
    Stage 1 tokenization: all-text loss on full chat-template caption.

    Format: <|user|>What would be the WebVid caption for this video?<|end|><|assistant|>{caption}<|end|>
    Loss mask: 1 for ALL text tokens (prompt + caption).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    prompt = "What would be the WebVid caption for this video?"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": caption},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=512)
    token_ids = encoding["input_ids"]

    # Stage 1: loss on ALL text tokens
    loss_mask = [1] * len(token_ids)

    return {"token_ids": token_ids, "loss_mask": loss_mask}


def tokenize_sft(user_text: str, assistant_text: str, stage: int = 2, tokenizer=None) -> dict:
    """
    Stage 2/3 tokenization: answer-only loss.

    Loss mask: 1 for assistant tokens only, 0 for user/system tokens.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=1024)
    token_ids = encoding["input_ids"]

    # For answer-only loss, we need to find where the assistant response starts.
    # Tokenize just the user portion to find the split point.
    user_messages = [{"role": "user", "content": user_text}]
    user_text_only = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    user_encoding = tokenizer(user_text_only, add_special_tokens=False, truncation=True, max_length=1024)
    user_len = len(user_encoding["input_ids"])

    # Loss mask: 0 for user portion, 1 for assistant portion
    loss_mask = [0] * min(user_len, len(token_ids)) + [1] * max(0, len(token_ids) - user_len)

    return {"token_ids": token_ids, "loss_mask": loss_mask}


def tokenize_text_only(user_text: str, assistant_text: str, stage: int = 1, tokenizer=None) -> dict:
    """
    SmolTalk text-only tokenization. Loss rule follows the stage:
      Stage 1: all-text loss
      Stage 2-3: answer-only loss
    """
    if stage == 1:
        # All-text loss: treat assistant_text as the full response
        if tokenizer is None:
            tokenizer = get_tokenizer()
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=512)
        token_ids = encoding["input_ids"]
        loss_mask = [1] * len(token_ids)
        return {"token_ids": token_ids, "loss_mask": loss_mask}
    else:
        return tokenize_sft(user_text, assistant_text, stage=stage, tokenizer=tokenizer)


# --------------------------------------------------------------------------- #
# WebDataset shard writer
# --------------------------------------------------------------------------- #

class ShardWriter:
    """Write samples to webdataset tar shards."""

    def __init__(self, output_dir: str, samples_per_shard: int = 1000, prefix: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.prefix = prefix

        self._shard_idx = 0
        self._sample_idx = 0
        self._in_shard_count = 0
        self._tar = None
        self._total_written = 0

    def _open_new_shard(self):
        if self._tar is not None:
            self._tar.close()
        path = self.output_dir / f"{self.prefix}{self._shard_idx:05d}.tar"
        self._tar = tarfile.open(str(path), "w")
        self._in_shard_count = 0

    def _add_file(self, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))

    def write_sample(self, sample_key: str, files: dict):
        """
        Write a sample to the current shard.

        files: dict of {extension: bytes_data}, e.g.
          {"json": json_bytes, "jpg": image_bytes, "000.jpg": frame0_bytes, ...}
        """
        if self._tar is None or self._in_shard_count >= self.samples_per_shard:
            self._open_new_shard()
            self._shard_idx += 1

        for ext, data in files.items():
            self._add_file(f"{sample_key}.{ext}", data)

        self._in_shard_count += 1
        self._total_written += 1

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    @property
    def total_written(self):
        return self._total_written


# --------------------------------------------------------------------------- #
# SmolTalk preprocessing
# --------------------------------------------------------------------------- #

def preprocess_smoltalk(stage: int, output_dir: str, max_samples: int = 0):
    """Download and tokenize SmolTalk for text retention."""
    from datasets import load_dataset

    print(f"[SmolTalk] Loading dataset for stage {stage}...")
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)

    tokenizer = get_tokenizer()
    writer = ShardWriter(output_dir, samples_per_shard=1000)

    # Target samples per stage (14% of vision data)
    targets = {1: 280_000, 2: 140_000, 3: 70_000}
    target = targets.get(stage, 100_000)
    if max_samples > 0:
        target = min(target, max_samples)

    count = 0
    t0 = time.time()

    for sample in ds:
        if count >= target:
            break

        messages = sample.get("messages", [])
        if len(messages) < 2:
            continue

        # Extract user and assistant messages
        user_text = ""
        assistant_text = ""
        for msg in messages:
            if msg["role"] == "user":
                user_text = msg["content"]
            elif msg["role"] == "assistant":
                assistant_text = msg["content"]

        if not user_text or not assistant_text:
            continue

        tok = tokenize_text_only(user_text, assistant_text, stage=stage, tokenizer=tokenizer)

        meta = {
            "token_ids": tok["token_ids"],
            "loss_mask": tok["loss_mask"],
            "source": "smoltalk",
            "is_text_only": True,
        }

        sample_key = f"{count:08d}"
        writer.write_sample(sample_key, {
            "json": json.dumps(meta).encode("utf-8"),
        })

        count += 1
        if count % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  [SmolTalk] {count}/{target} samples ({elapsed:.0f}s)", flush=True)

    writer.close()
    elapsed = time.time() - t0
    print(f"[SmolTalk] Done: {writer.total_written} samples, "
          f"{writer._shard_idx} shards, {elapsed:.0f}s")


# --------------------------------------------------------------------------- #
# The Cauldron preprocessing
# --------------------------------------------------------------------------- #

def preprocess_cauldron(output_dir: str, max_samples: int = 0):
    """Download and preprocess The Cauldron (image VQA)."""
    from datasets import load_dataset
    from PIL import Image

    print("[Cauldron] Loading dataset...")

    # The Cauldron has many subsets. Load the main ones.
    subsets = [
        "ai2d", "aokvqa", "chart2text", "clevr", "docvqa",
        "dvqa", "figureqa", "geomverse", "hateful_memes",
        "infographic_vqa", "intergps", "localized_narratives",
        "mapqa", "ocrvqa", "okvqa", "plotqa", "raven",
        "scienceqa", "screen2words", "st_vqa", "tabmwp",
        "tallyqa", "textcaps", "textvqa", "tqa",
        "visual7w", "visualmrc", "vistext", "vqarad",
        "vqav2", "vsr", "websight",
    ]

    tokenizer = get_tokenizer()
    writer = ShardWriter(output_dir, samples_per_shard=1000)
    total_target = max_samples if max_samples > 0 else 1_000_000
    count = 0
    t0 = time.time()

    for subset in subsets:
        if count >= total_target:
            break

        print(f"  [Cauldron] Loading subset: {subset}...", flush=True)
        try:
            ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True)
        except Exception as e:
            print(f"    Skipping {subset}: {e}")
            continue

        for sample in ds:
            if count >= total_target:
                break

            images = sample.get("images", [])
            texts = sample.get("texts", [])

            if not images or not texts:
                continue

            # Extract Q&A from texts
            user_text = ""
            assistant_text = ""
            for entry in texts:
                if entry.get("user"):
                    user_text = entry["user"]
                if entry.get("assistant"):
                    assistant_text = entry["assistant"]

            if not user_text or not assistant_text:
                continue

            # Process first image
            img = images[0]
            if not isinstance(img, Image.Image):
                continue

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
                "source": f"cauldron/{subset}",
                "frame_count": 1,
            }

            sample_key = f"{count:08d}"
            writer.write_sample(sample_key, {
                "json": json.dumps(meta).encode("utf-8"),
                "jpg": img_bytes,
            })

            count += 1
            if count % 5000 == 0:
                elapsed = time.time() - t0
                rate = count / max(elapsed, 1)
                print(f"  [Cauldron] {count} samples ({rate:.0f} samp/s)", flush=True)

    writer.close()
    elapsed = time.time() - t0
    print(f"[Cauldron] Done: {writer.total_written} samples, "
          f"{writer._shard_idx} shards, {elapsed:.0f}s")


# --------------------------------------------------------------------------- #
# WebVid preprocessing
# --------------------------------------------------------------------------- #

def preprocess_webvid(output_dir: str, max_samples: int = 0, workers: int = 4):
    """
    Download WebVid metadata and process video URLs.

    Phase 1: Download metadata (fast)
    Phase 2: Download videos + extract frames (slow, parallelized)
    """
    import subprocess
    import csv
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image

    print("[WebVid] Loading metadata...")
    from datasets import load_dataset

    ds = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)

    tokenizer = get_tokenizer()
    writer = ShardWriter(output_dir, samples_per_shard=1000)
    total_target = max_samples if max_samples > 0 else 2_000_000

    count = 0
    errors = 0
    t0 = time.time()
    tmp_dir = Path("/workspace/tmp/webvid_frames")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def process_video(url: str, caption: str, sample_idx: int) -> dict | None:
        """Download video, extract frames, return sample data."""
        video_path = tmp_dir / f"vid_{sample_idx}.mp4"
        frames_dir = tmp_dir / f"frames_{sample_idx}"
        frames_dir.mkdir(exist_ok=True)

        try:
            # Download video (timeout 30s, max 50MB)
            result = subprocess.run(
                ["wget", "-q", "-O", str(video_path), "--timeout=30",
                 f"--max-filesize=50M", url],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0 or not video_path.exists() or video_path.stat().st_size < 1000:
                return None

            # Extract frames at 1 FPS, resize to 224x224, center crop
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
                 "-frames:v", "64",  # cap at 64 frames
                 "-q:v", "2",  # JPEG quality
                 str(frames_dir / "frame_%03d.jpg")],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                return None

            # Collect frame bytes
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            if not frame_files:
                return None

            frame_data = {}
            for i, fp in enumerate(frame_files[:64]):
                frame_data[f"{i:03d}.jpg"] = fp.read_bytes()

            # Tokenize caption (Stage 1: all-text loss)
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
            # Cleanup
            video_path.unlink(missing_ok=True)
            for f in frames_dir.glob("*"):
                f.unlink(missing_ok=True)
            frames_dir.rmdir() if frames_dir.exists() else None

    print(f"[WebVid] Starting download+process with {workers} workers...")
    print(f"  Target: {total_target} samples")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        batch_size = workers * 4  # Keep the queue fed

        sample_iter = iter(ds)
        submitted = 0
        finished = 0

        # Submit initial batch
        for _ in range(batch_size):
            try:
                sample = next(sample_iter)
                url = sample.get("contentUrl") or sample.get("url", "")
                caption = sample.get("name") or sample.get("caption", "")
                if url and caption:
                    fut = executor.submit(process_video, url, caption, submitted)
                    futures[fut] = submitted
                    submitted += 1
            except StopIteration:
                break

        while futures and count < total_target:
            done_futures = []
            for fut in list(futures.keys()):
                if fut.done():
                    done_futures.append(fut)

            if not done_futures:
                time.sleep(0.1)
                continue

            for fut in done_futures:
                idx = futures.pop(fut)
                finished += 1

                try:
                    result = fut.result()
                except Exception:
                    result = None
                    errors += 1

                if result is not None:
                    sample_key = f"{count:08d}"
                    writer.write_sample(sample_key, result)
                    count += 1
                else:
                    errors += 1

                # Submit replacement
                if count < total_target:
                    try:
                        sample = next(sample_iter)
                        url = sample.get("contentUrl") or sample.get("url", "")
                        caption = sample.get("name") or sample.get("caption", "")
                        if url and caption:
                            new_fut = executor.submit(process_video, url, caption, submitted)
                            futures[new_fut] = submitted
                            submitted += 1
                    except StopIteration:
                        pass

                if count % 1000 == 0 and count > 0:
                    elapsed = time.time() - t0
                    rate = count / max(elapsed, 1)
                    err_rate = errors / max(finished, 1) * 100
                    print(f"  [WebVid] {count}/{total_target} done, "
                          f"{rate:.1f} samp/s, {err_rate:.0f}% errors, "
                          f"{elapsed/3600:.1f}h elapsed", flush=True)

    writer.close()
    elapsed = time.time() - t0
    print(f"[WebVid] Done: {writer.total_written} samples, "
          f"{writer._shard_idx} shards, {errors} errors, {elapsed/3600:.1f}h")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="fVLM data preprocessing")
    sub = parser.add_subparsers(dest="command")

    # SmolTalk
    p_st = sub.add_parser("smoltalk", help="Preprocess SmolTalk text retention data")
    p_st.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p_st.add_argument("--output", default=None)
    p_st.add_argument("--max-samples", type=int, default=0)

    # Cauldron
    p_ca = sub.add_parser("cauldron", help="Preprocess The Cauldron VQA data")
    p_ca.add_argument("--output", default="/workspace/data/cauldron")
    p_ca.add_argument("--max-samples", type=int, default=0)

    # WebVid
    p_wv = sub.add_parser("webvid", help="Download and preprocess WebVid-10M")
    p_wv.add_argument("--output", default="/workspace/data/webvid")
    p_wv.add_argument("--max-samples", type=int, default=0)
    p_wv.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    if args.command == "smoltalk":
        output = args.output or f"/workspace/data/text_retention/stage{args.stage}"
        preprocess_smoltalk(args.stage, output, args.max_samples)

    elif args.command == "cauldron":
        preprocess_cauldron(args.output, args.max_samples)

    elif args.command == "webvid":
        preprocess_webvid(args.output, args.max_samples, args.workers)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
