"""
WebDataset-based data loader for foveated VLM training.

Reads tar shards produced by video2dataset / the CPU precompute pipeline.
Each sample in a shard contains EITHER:
  A) Pre-extracted frames:
     - {key}.jpg or {key}_000.jpg, {key}_001.jpg, ... -- JPEG frames (224x224)
     - {key}.json -- metadata: {caption, token_ids, loss_mask, ...}
  B) Raw MP4 from video2dataset:
     - {key}.mp4 -- raw video file
     - {key}.txt -- caption text
     - {key}.json -- metadata: {videoid, duration, url, ...}

On-the-fly tokenization: if token_ids/loss_mask are missing from JSON,
the sample is tokenized at load time using the provided tokenizer.

Returns dicts with:
  frames:     [T, 3, 224, 224]  float32, ImageNet-normalized for DINO
  input_ids:  [S]               long, token IDs
  loss_mask:  [S]               float32, 1.0 for answer tokens, 0.0 otherwise
  num_frames: int               actual frame count before any padding
"""

import io
import json
import os
import re
import subprocess
import tempfile
from typing import Optional

import torch
import torchvision.transforms.functional as TF
import webdataset as wds

# ImageNet normalization for DINOv2 (same constants as src/data/llava_video_dataset.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Regex to detect multi-frame filenames like "sample_003.jpg"
_FRAME_INDEX_RE = re.compile(r"^(.+)_(\d{3})\.(jpg|jpeg|png)$")

# Regex to detect single-frame filenames like "sample.jpg"
_SINGLE_FRAME_RE = re.compile(r"^(.+)\.(jpg|jpeg|png)$")


def _load_image_tensor(data: bytes) -> torch.Tensor:
    """Decode JPEG/PNG bytes to a [3, 224, 224] float32 tensor, ImageNet-normalized."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = TF.to_tensor(img)  # [3, H, W] float32 in [0, 1]
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


def _decode_mp4_frames(mp4_bytes: bytes, max_frames: int = 64) -> list[torch.Tensor]:
    """Decode MP4 bytes to a list of [3, 224, 224] tensors at 1 FPS."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(io.BytesIO(mp4_bytes), width=224, height=224)
        fps = vr.get_avg_fps()
        total = len(vr)
        # Sample at 1 FPS
        step = max(1, int(fps))
        indices = list(range(0, total, step))[:max_frames]
        if not indices:
            return []
        batch = vr.get_batch(indices)  # [T, H, W, C] uint8
        frames = []
        for i in range(batch.shape[0]):
            t = batch[i].permute(2, 0, 1).float() / 255.0  # [3, 224, 224]
            t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            frames.append(t)
        return frames
    except ImportError:
        pass

    # Fallback: ffmpeg subprocess
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/workspace/tmp", delete=True) as f:
        f.write(mp4_bytes)
        f.flush()
        frames_dir = f.name + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", f.name,
                 "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
                 "-frames:v", str(max_frames), "-q:v", "2",
                 os.path.join(frames_dir, "frame_%03d.jpg")],
                capture_output=True, timeout=30,
            )
            from PIL import Image
            frame_files = sorted(os.listdir(frames_dir))
            frames = []
            for fname in frame_files[:max_frames]:
                fp = os.path.join(frames_dir, fname)
                img = Image.open(fp).convert("RGB")
                t = TF.to_tensor(img)
                t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                frames.append(t)
            return frames
        except Exception:
            return []
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)


def decode_sample(sample: dict, max_frames: int = 64,
                  tokenizer=None, stage: int = 1) -> Optional[dict]:
    """
    Decode a single webdataset sample dict into training tensors.

    The sample dict has keys like:
      "jpg" or "jpeg" or "png" -- single frame bytes
      "000.jpg", "001.jpg", ... -- multi-frame bytes
      "json" -- metadata JSON bytes or dict

    Returns None if the sample is malformed (caller should filter).
    """
    # ------------------------------------------------------------------
    # 1. Parse metadata JSON
    # ------------------------------------------------------------------
    meta_raw = sample.get("json")
    if meta_raw is None:
        return None

    if isinstance(meta_raw, bytes):
        try:
            meta = json.loads(meta_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    elif isinstance(meta_raw, str):
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(meta_raw, dict):
        meta = meta_raw
    else:
        return None

    token_ids = meta.get("token_ids")
    loss_mask = meta.get("loss_mask")

    # On-the-fly tokenization if pre-tokenized data is missing
    if token_ids is None or loss_mask is None:
        caption = meta.get("caption", "")
        # Also check .txt key (video2dataset stores caption as separate file)
        if not caption:
            txt_raw = sample.get("txt")
            if isinstance(txt_raw, bytes):
                caption = txt_raw.decode("utf-8", errors="replace").strip()
            elif isinstance(txt_raw, str):
                caption = txt_raw.strip()

        if not caption or tokenizer is None:
            return None

        # Tokenize on-the-fly
        from release.scripts.precompute import tokenize_stage1, tokenize_sft
        if stage == 1:
            tok = tokenize_stage1(caption, tokenizer=tokenizer)
        else:
            # For stage 2-3, caption is used as assistant response
            tok = tokenize_sft("Describe this.", caption, stage=stage, tokenizer=tokenizer)
        token_ids = tok["token_ids"]
        loss_mask = tok["loss_mask"]

    # ------------------------------------------------------------------
    # 2. Collect frames (JPEG bytes or decode from MP4)
    # ------------------------------------------------------------------
    frames: list[torch.Tensor] = []

    # Try MP4 first (video2dataset raw output)
    mp4_data = sample.get("mp4")
    if isinstance(mp4_data, bytes) and len(mp4_data) > 100:
        frames = _decode_mp4_frames(mp4_data, max_frames=max_frames)
    else:
        # Try numbered JPEG frames (000.jpg, 001.jpg, ...)
        numbered_keys: list[tuple[int, str]] = []
        for key in sample:
            m = re.match(r"^(\d{3})\.(jpg|jpeg|png)$", key)
            if m:
                numbered_keys.append((int(m.group(1)), key))

        if numbered_keys:
            numbered_keys.sort(key=lambda x: x[0])
            for _, key in numbered_keys:
                raw = sample[key]
                if isinstance(raw, bytes):
                    try:
                        frames.append(_load_image_tensor(raw))
                    except Exception:
                        continue
        else:
            # Single frame: look for jpg / jpeg / png key
            for ext in ("jpg", "jpeg", "png"):
                if ext in sample and isinstance(sample[ext], bytes):
                    try:
                        frames.append(_load_image_tensor(sample[ext]))
                    except Exception:
                        pass
                    break

    if not frames:
        return None

    # Truncate to max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]

    num_frames = len(frames)
    frames_tensor = torch.stack(frames, dim=0)  # [T, 3, 224, 224]

    # ------------------------------------------------------------------
    # 3. Build text tensors
    # ------------------------------------------------------------------
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    loss_mask_t = torch.tensor(loss_mask, dtype=torch.float32)

    # Ensure consistent lengths
    min_len = min(len(input_ids), len(loss_mask_t))
    input_ids = input_ids[:min_len]
    loss_mask_t = loss_mask_t[:min_len]

    return {
        "frames": frames_tensor,       # [T, 3, 224, 224]
        "input_ids": input_ids,         # [S]
        "loss_mask": loss_mask_t,       # [S]
        "num_frames": num_frames,       # int
    }


def _sample_decoder(max_frames: int, tokenizer=None, stage: int = 1):
    """Return a map function for use in a webdataset pipeline."""
    def _decode(sample):
        result = decode_sample(sample, max_frames=max_frames,
                               tokenizer=tokenizer, stage=stage)
        if result is None:
            return None
        return result
    return _decode


def _is_valid(sample) -> bool:
    """Filter predicate: keep only successfully decoded samples."""
    return sample is not None


def create_webdataset(
    shard_pattern: str,
    tokenizer=None,
    stage: int = 1,
    max_frames: int = 64,
    shuffle: bool = True,
    seed: int = 42,
    epoch: int = 0,
    num_workers: int = 4,
    batch_size: Optional[int] = None,
    shardshuffle: int = 1000,
) -> wds.WebDataset:
    """
    Create a webdataset pipeline that streams tar shards.

    Parameters
    ----------
    shard_pattern : str
        Brace-expansion pattern for tar shards, e.g.
        "/workspace/webvid_frames/{00000..02999}.tar"
    tokenizer : optional
        Tokenizer for on-the-fly tokenization of raw captions.
        If None, samples must have pre-tokenized token_ids in JSON.
    max_frames : int
        Maximum number of frames per sample (extras truncated). Default 64,
        matching SmolVLM2's frame cap.
    shuffle : bool
        Whether to shuffle shards and samples.  Disable for deterministic
        evaluation.
    seed : int
        Random seed for reproducible shard + sample shuffling.
    epoch : int
        Epoch counter — combined with seed for per-epoch shuffling so that
        each epoch sees a different order without losing reproducibility.
    num_workers : int
        Hint for shard splitting across DataLoader workers.  webdataset
        handles the splitting internally via its nodesplitter.
    batch_size : int, optional
        If provided, the pipeline batches internally (rare — usually the
        external DataLoader + collate_foveated handles batching).
    shardshuffle : int
        Buffer size for shard-level shuffle.  Larger = better randomisation
        at the cost of memory.  1000 shards ~= 1M samples for our shard
        size of 1000 samples/shard.

    Returns
    -------
    wds.WebDataset
        An iterable dataset that yields dicts:
          frames:     [T, 3, 224, 224]
          input_ids:  [S]
          loss_mask:  [S]
          num_frames: int
    """
    effective_seed = seed + epoch

    # Build the pipeline.
    dataset = wds.WebDataset(
        shard_pattern,
        nodesplitter=wds.split_by_worker,
        shardshuffle=shardshuffle if shuffle else False,
        seed=effective_seed if shuffle else None,
    )

    if shuffle:
        # Shuffle within a buffer of samples (after shard-level shuffle).
        dataset = dataset.shuffle(size=5000, seed=effective_seed)

    # Decode: we do NOT use wds.decode() because we need custom multi-frame
    # logic.  Instead we pass raw bytes and decode in _sample_decoder.
    dataset = dataset.map(_sample_decoder(max_frames, tokenizer=tokenizer, stage=stage))
    dataset = dataset.select(_is_valid)

    if batch_size is not None:
        dataset = dataset.batched(batch_size)

    return dataset


def make_dataloader(
    shard_pattern: str,
    batch_size: int,
    max_frames: int = 64,
    shuffle: bool = True,
    seed: int = 42,
    epoch: int = 0,
    num_workers: int = 4,
    collate_fn=None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    tokenizer=None,
    stage: int = 1,
) -> torch.utils.data.DataLoader:
    """
    Convenience wrapper: creates the webdataset pipeline and wraps it in a
    standard PyTorch DataLoader with the given collate function.

    If collate_fn is None, use release.data.collate.collate_foveated.
    """
    if collate_fn is None:
        from release.data.collate import collate_foveated
        collate_fn = collate_foveated

    dataset = create_webdataset(
        shard_pattern=shard_pattern,
        tokenizer=tokenizer,
        stage=stage,
        max_frames=max_frames,
        shuffle=shuffle,
        seed=seed,
        epoch=epoch,
        num_workers=num_workers,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return loader
