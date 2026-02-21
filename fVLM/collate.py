"""
Custom collate functions for variable-length foveated VLM batches.

Handles two axes of variation:
  1. Frame count  -- videos have 1-64 frames depending on duration.
  2. Text length  -- tokenized sequences vary per sample.

Produces padded tensors + boolean masks so the model can ignore padding.

Includes a token-budget batcher that forms variable-size batches by
capping total frames per batch (rather than a fixed batch size).
This keeps GPU work roughly constant across batches:
  - Short videos (T=1): bs up to max_batch_size
  - Long videos (T=64): bs=8
"""

import torch
from typing import Any


def collate_foveated(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate a list of sample dicts into a single padded batch.

    Each sample dict (from webdataset_loader.decode_sample) contains:
      frames:     [T_i, 3, 224, 224]   float32
      input_ids:  [S_i]                long
      loss_mask:  [S_i]                float32
      num_frames: int

    Returns a dict of padded tensors:
      frames:          [B, T_max, 3, 224, 224]  float32  -- zero-padded
      frame_mask:      [B, T_max]               bool     -- True for real frames
      input_ids:       [B, S_max]               long     -- zero-padded (pad_id=0)
      attention_mask:  [B, S_max]               bool     -- True for real tokens
      loss_mask:       [B, S_max]               float32  -- zero-padded
      num_frames:      [B]                      long     -- original frame counts

    Padding is dynamic per-batch (pad to max in batch, no bucketing).
    """
    if not batch:
        raise ValueError("collate_foveated received an empty batch")

    B = len(batch)

    # Determine max sizes from this batch
    t_max = max(sample["frames"].shape[0] for sample in batch)
    s_max = max(sample["input_ids"].shape[0] for sample in batch)

    # Frame spatial dimensions (should be uniform but read from data)
    _, C, H, W = batch[0]["frames"].shape

    # Allocate output tensors
    frames = torch.zeros(B, t_max, C, H, W, dtype=torch.float32)
    frame_mask = torch.zeros(B, t_max, dtype=torch.bool)
    input_ids = torch.zeros(B, s_max, dtype=torch.long)
    attention_mask = torch.zeros(B, s_max, dtype=torch.bool)
    loss_mask = torch.zeros(B, s_max, dtype=torch.float32)
    num_frames = torch.zeros(B, dtype=torch.long)

    # Fill in each sample
    for i, sample in enumerate(batch):
        t_i = sample["frames"].shape[0]
        s_i = sample["input_ids"].shape[0]

        frames[i, :t_i] = sample["frames"]
        frame_mask[i, :t_i] = True

        input_ids[i, :s_i] = sample["input_ids"]
        attention_mask[i, :s_i] = True
        loss_mask[i, :s_i] = sample["loss_mask"]

        num_frames[i] = sample["num_frames"]

    return {
        "frames": frames,                # [B, T_max, 3, 224, 224]
        "frame_mask": frame_mask,         # [B, T_max]
        "input_ids": input_ids,           # [B, S_max]
        "attention_mask": attention_mask,  # [B, S_max]
        "loss_mask": loss_mask,           # [B, S_max]
        "num_frames": num_frames,         # [B]
    }


def collate_dpo(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate a list of DPO sample dicts into a single padded batch.

    Each sample dict (from webdataset_loader.decode_dpo_sample) contains:
      frames:             [T_i, 3, 224, 224]   float32
      chosen_input_ids:   [S_c_i]              long
      chosen_loss_mask:   [S_c_i]              float32
      rejected_input_ids: [S_r_i]              long
      rejected_loss_mask: [S_r_i]              float32
      num_frames:         int

    Returns a dict of padded tensors:
      frames:                [B, T_max, 3, 224, 224]   float32  -- zero-padded
      frame_mask:            [B, T_max]                bool     -- True for real frames
      chosen_input_ids:      [B, S_c_max]              long     -- zero-padded
      chosen_attention_mask: [B, S_c_max]              bool     -- True for real tokens
      chosen_loss_mask:      [B, S_c_max]              float32  -- zero-padded
      rejected_input_ids:    [B, S_r_max]              long     -- zero-padded
      rejected_attention_mask: [B, S_r_max]            bool     -- True for real tokens
      rejected_loss_mask:    [B, S_r_max]              float32  -- zero-padded
      num_frames:            [B]                       long     -- original frame counts

    Chosen and rejected sequences are padded independently (different max lengths).
    """
    if not batch:
        raise ValueError("collate_dpo received an empty batch")

    B = len(batch)

    # Determine max sizes from this batch
    t_max = max(sample["frames"].shape[0] for sample in batch)
    sc_max = max(sample["chosen_input_ids"].shape[0] for sample in batch)
    sr_max = max(sample["rejected_input_ids"].shape[0] for sample in batch)

    # Frame spatial dimensions
    _, C, H, W = batch[0]["frames"].shape

    # Allocate output tensors
    frames = torch.zeros(B, t_max, C, H, W, dtype=torch.float32)
    frame_mask = torch.zeros(B, t_max, dtype=torch.bool)
    chosen_input_ids = torch.zeros(B, sc_max, dtype=torch.long)
    chosen_attention_mask = torch.zeros(B, sc_max, dtype=torch.bool)
    chosen_loss_mask = torch.zeros(B, sc_max, dtype=torch.float32)
    rejected_input_ids = torch.zeros(B, sr_max, dtype=torch.long)
    rejected_attention_mask = torch.zeros(B, sr_max, dtype=torch.bool)
    rejected_loss_mask = torch.zeros(B, sr_max, dtype=torch.float32)
    num_frames = torch.zeros(B, dtype=torch.long)

    # Fill in each sample
    for i, sample in enumerate(batch):
        t_i = sample["frames"].shape[0]
        sc_i = sample["chosen_input_ids"].shape[0]
        sr_i = sample["rejected_input_ids"].shape[0]

        frames[i, :t_i] = sample["frames"]
        frame_mask[i, :t_i] = True

        chosen_input_ids[i, :sc_i] = sample["chosen_input_ids"]
        chosen_attention_mask[i, :sc_i] = True
        chosen_loss_mask[i, :sc_i] = sample["chosen_loss_mask"]

        rejected_input_ids[i, :sr_i] = sample["rejected_input_ids"]
        rejected_attention_mask[i, :sr_i] = True
        rejected_loss_mask[i, :sr_i] = sample["rejected_loss_mask"]

        num_frames[i] = sample["num_frames"]

    return {
        "frames": frames,                              # [B, T_max, 3, 224, 224]
        "frame_mask": frame_mask,                      # [B, T_max]
        "chosen_input_ids": chosen_input_ids,          # [B, S_c_max]
        "chosen_attention_mask": chosen_attention_mask, # [B, S_c_max]
        "chosen_loss_mask": chosen_loss_mask,          # [B, S_c_max]
        "rejected_input_ids": rejected_input_ids,      # [B, S_r_max]
        "rejected_attention_mask": rejected_attention_mask,  # [B, S_r_max]
        "rejected_loss_mask": rejected_loss_mask,      # [B, S_r_max]
        "num_frames": num_frames,                      # [B]
    }


# --------------------------------------------------------------------------- #
# Token-budget batcher (dynamic batch sizing by total frame count)
# --------------------------------------------------------------------------- #

def token_budget_batcher(max_total_frames: int = 512, max_batch_size: int = 64,
                         length_bucket: bool = False, bucket_buffer: int = 256):
    """
    WebDataset compositor that forms variable-size batches by capping total
    frames per batch.  Keeps GPU work roughly constant across batches:
      - Short videos (T=1): bs up to max_batch_size
      - Long videos (T=64): bs = max_total_frames // 64 = 8

    When length_bucket=True, buffers `bucket_buffer` samples and sorts by
    total sequence length (frames + text tokens) before forming batches.
    This groups similar-length samples together, reducing padding waste.

    Usage in a WebDataset pipeline::

        dataset = dataset.compose(token_budget_batcher(512, 64))

    Each yielded item is a collated batch dict (same format as
    collate_foveated output), ready for the training loop.
    """
    def _form_batches(samples):
        """Yield batches from a list of samples respecting frame budget."""
        batch = []
        total_frames = 0
        for sample in samples:
            t = sample["frames"].shape[0]
            if total_frames + t > max_total_frames and batch:
                yield collate_foveated(batch)
                batch = []
                total_frames = 0
            batch.append(sample)
            total_frames += t
            if len(batch) >= max_batch_size:
                yield collate_foveated(batch)
                batch = []
                total_frames = 0
        if batch:
            yield collate_foveated(batch)

    def _batcher(src):
        if not length_bucket:
            yield from _form_batches(src)
            return

        # Length-bucketed batching: buffer samples, sort by total length,
        # then form batches. Similar-length samples end up together,
        # minimizing padding waste within each batch.
        buf = []
        for sample in src:
            buf.append(sample)
            if len(buf) >= bucket_buffer:
                buf.sort(key=lambda s: s["frames"].shape[0] + s["input_ids"].shape[0])
                yield from _form_batches(buf)
                buf = []
        if buf:
            buf.sort(key=lambda s: s["frames"].shape[0] + s["input_ids"].shape[0])
            yield from _form_batches(buf)

    return _batcher
