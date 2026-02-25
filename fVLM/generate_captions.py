#!/usr/bin/env python3
"""
Autoregressive caption generation on training video samples using fVLM-1.7B.

Loads 5 samples from the SFT training data (sharegpt4video_shards and vript_shards),
generates free-form captions via greedy token-by-token decoding, and compares them
to the ground truth captions from the training data.
"""

import sys
import io
import re
import json
import tarfile
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

# Ensure we can import model.py / encoder.py / tokenization.py from the project root
sys.path.insert(0, "/workspace/workdir/nanoSmol/fVLM")


# ── DINOv2 image normalization (same as data.py) ─────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
_NORM_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_NORM_STD  = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def load_image_tensor(data: bytes) -> torch.Tensor:
    """Decode JPEG bytes to [3, H, W] float32, ImageNet-normalized."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Resize to 518x518 to match DINOv2 small (patch_size=14, 518/14=37)
    # Actually check what training uses -- data.py produces 224x224
    # The model docstring says 224x224, let's use that.
    img = img.resize((224, 224))
    tensor = TF.to_tensor(img)  # [3, H, W] float32 in [0, 1]
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


def load_samples_from_tar(tar_path: str, n: int = 5) -> List[dict]:
    """
    Load n samples from a webdataset tar shard.

    Each sample has multiple numbered JPEGs (000.jpg, 001.jpg, ...) and one .json.
    Groups them by sample key prefix.

    Returns list of dicts: {frames: Tensor[T,3,224,224], gt_caption: str, source: str, user_prompt: str}
    """
    t = tarfile.open(tar_path, "r")
    members = t.getmembers()

    # Group by sample key (everything before the last dot-extension)
    # Filenames: 000000_0000.000.jpg, 000000_0000.001.jpg, 000000_0000.json
    # Key pattern: 000000_0000
    sample_groups = {}
    for m in members:
        # Parse: key is everything up to the first "." that's followed by a known extension type
        # e.g., "000000_0000.000.jpg" -> key="000000_0000", sub="000.jpg"
        # e.g., "000000_0000.json" -> key="000000_0000", sub="json"
        name = m.name
        # Split on first dot
        parts = name.split(".", 1)
        if len(parts) < 2:
            continue
        key = parts[0]  # e.g., "000000_0000"
        ext = parts[1]  # e.g., "000.jpg" or "json"

        if key not in sample_groups:
            sample_groups[key] = {}
        sample_groups[key][ext] = m

    samples = []
    for key in sorted(sample_groups.keys()):
        if len(samples) >= n:
            break
        group = sample_groups[key]

        # Load JSON metadata
        if "json" not in group:
            continue
        f = t.extractfile(group["json"])
        meta = json.load(f)

        assistant_text = meta.get("assistant", "")
        if not assistant_text:
            continue

        user_text = meta.get("user", "")
        source = meta.get("source", "")
        frame_count = meta.get("frame_count", 0)

        # Load frame JPEGs
        frame_keys = []
        for ext_key in group:
            m_frame = re.match(r"^(\d{3})\.(jpg|jpeg|png)$", ext_key)
            if m_frame:
                frame_keys.append((int(m_frame.group(1)), ext_key))
        frame_keys.sort(key=lambda x: x[0])

        if not frame_keys:
            continue

        frames = []
        for _, ext_key in frame_keys:
            raw = t.extractfile(group[ext_key]).read()
            try:
                frames.append(load_image_tensor(raw))
            except Exception as e:
                print(f"  Warning: failed to load frame {ext_key} for {key}: {e}")
                continue

        if not frames:
            continue

        frames_tensor = torch.stack(frames, dim=0)  # [T, 3, 224, 224]

        samples.append({
            "frames": frames_tensor,
            "gt_caption": assistant_text,
            "source": source,
            "user_prompt": user_text,
            "key": key,
            "num_frames": len(frames),
        })

    t.close()
    return samples


def extract_gt_answer(gt_caption: str) -> str:
    """Truncate very long ground truth for readability."""
    if len(gt_caption) > 600:
        return gt_caption[:600] + "..."
    return gt_caption


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/workspace/models/SmolLM2-1.7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load model ─────────────────────────────────────────────────────
    print("Loading model...")
    from model import FoveatedVLM

    model = FoveatedVLM(
        llm_name="/workspace/models/SmolLM2-1.7B-Instruct",
        dino_name="/workspace/models/dinov2-small",
        deep_query=True,
    )

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(
        "/workspace/checkpoints/final_1.7B/stage3/latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  Loaded checkpoint from step {ckpt.get('step', '?')}")
    del ckpt  # free memory

    model = model.to(device).eval()
    print("Model loaded and moved to GPU.")

    # ── 3. Load training samples ──────────────────────────────────────────
    print("\nLoading training samples...")

    # Load from two different sources for variety
    samples = []

    # 3 from vript_shards (detailed narrations)
    vript_samples = load_samples_from_tar(
        "/workspace/data/vript_shards/000000.tar", n=3
    )
    samples.extend(vript_samples)
    print(f"  Loaded {len(vript_samples)} samples from vript_shards")

    # 2 from sharegpt4video_shards (descriptive captions)
    sg4v_samples = load_samples_from_tar(
        "/workspace/data/sharegpt4video_shards/000000.tar", n=2
    )
    samples.extend(sg4v_samples)
    print(f"  Loaded {len(sg4v_samples)} samples from sharegpt4video_shards")

    if not samples:
        print("ERROR: No samples loaded!")
        return

    print(f"  Total: {len(samples)} samples\n")

    # ── 4. Generate captions ──────────────────────────────────────────────
    # Per-source prompts (same as tokenization.py uses during training)
    SOURCE_PROMPTS = {
        "openvid": "Write a brief caption for this video.",
        "webvid": "What would be the WebVid caption for this video?",
        "vript": "Provide a detailed narration of what happens in this video.",
        "sharegpt4video": "Describe what happens in this video in detail.",
    }
    DEFAULT_PROMPT = "Describe this."
    SYSTEM_PROMPT = "You are a helpful AI assistant."

    eos_id = tokenizer.eos_token_id
    max_new_tokens = 200

    for idx, sample in enumerate(samples):
        print(f"{'='*80}")
        print(f"=== Sample {idx+1} / {len(samples)} ===")
        print(f"  Source: {sample['source']}")
        print(f"  Frames: {sample['num_frames']}")
        print(f"  Key: {sample['key']}")
        print(f"{'='*80}")

        # Use per-source prompt (same as training)
        user_prompt = sample["user_prompt"]
        if not user_prompt:
            user_prompt = SOURCE_PROMPTS.get(sample["source"], DEFAULT_PROMPT)

        # Build chat prompt for generation
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Prepare tensors
        cur_ids = torch.tensor([input_ids], device=device, dtype=torch.long)
        cur_attn = torch.ones_like(cur_ids)
        loss_mask = torch.zeros_like(cur_ids, dtype=torch.float32)

        # Prepare frames: [1, T, 3, 224, 224]
        frames_batch = sample["frames"].unsqueeze(0).to(device)

        # Replicate single-frame images to 8 frames (matches training config)
        if frames_batch.shape[1] == 1:
            frames_batch = frames_batch.repeat(1, 8, 1, 1, 1)

        # ── Autoregressive generation via forward_autoregressive ──────────
        # The forward_autoregressive method processes visual frames first,
        # building a KV cache, then processes text. But it doesn't do
        # token-by-token generation -- it processes the full text at once.
        #
        # For actual generation, we need to:
        # 1. Run the visual encoding + sequential frame processing (builds KV cache)
        # 2. Then decode text tokens one at a time
        #
        # We'll do this by manually implementing the generation loop using
        # the model's internal methods.

        gen_ids = []

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            B, T = frames_batch.shape[:2]

            # Step 1: Encode all frames with DINO
            kv_cache, patch_features, mask_flat = model._encode_all_frames(frames_batch)

            # Step 2: Sequential frame processing (autoregressive visual)
            orig_use_cache = model.llm.config.use_cache
            model.llm.config.use_cache = True

            query = model.q_init.expand(B, -1)  # [B, qd]
            llm_past_kv = None

            for t_idx in range(T):
                # Foveated extraction with current query
                frame_kv = model._extract_frame_kv(kv_cache, mask_flat, B, T, t_idx)
                z_t = model.encoder.query_attend(query, frame_kv)  # [B, dd]
                z_t_llm = model._project_visual(z_t.unsqueeze(1))  # [B, 1, ld]

                # Incremental LLM forward (one visual token at a time)
                out = model.llm.model(
                    inputs_embeds=z_t_llm,
                    past_key_values=llm_past_kv,
                    use_cache=True,
                )
                llm_past_kv = out.past_key_values

                # Derive query for next frame
                if t_idx < T - 1:
                    h_t = out.last_hidden_state[:, -1, :]  # [B, ld]
                    query = model.llm_to_query(h_t)        # [B, qd]

            # Step 3: Feed the prompt text tokens (all at once to build context)
            text_embeds = model._embed_text(cur_ids)  # [B, S, ld]
            out_text = model.llm.model(
                inputs_embeds=text_embeds,
                past_key_values=llm_past_kv,
                use_cache=True,
            )
            llm_past_kv = out_text.past_key_values

            # Get first generated token from the last position
            h_last = out_text.last_hidden_state[:, -1:, :]  # [B, 1, ld]
            logits = model.llm.lm_head(h_last)  # [B, 1, V]
            next_token = logits[0, -1, :].argmax().item()

            if next_token != eos_id:
                gen_ids.append(next_token)

                # Step 4: Continue generating token by token
                for step in range(max_new_tokens - 1):
                    # Feed the last generated token
                    next_embed = model._embed_text(
                        torch.tensor([[next_token]], device=device)
                    )  # [B, 1, ld]

                    out_step = model.llm.model(
                        inputs_embeds=next_embed,
                        past_key_values=llm_past_kv,
                        use_cache=True,
                    )
                    llm_past_kv = out_step.past_key_values

                    logits_step = model.llm.lm_head(out_step.last_hidden_state[:, -1:, :])
                    next_token = logits_step[0, -1, :].argmax().item()

                    if next_token == eos_id:
                        break
                    gen_ids.append(next_token)

            model.llm.config.use_cache = orig_use_cache

        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gt_text = extract_gt_answer(sample["gt_caption"])

        print(f"\nPrompt: \"{user_prompt}\"")
        print(f"\n--- GROUND TRUTH ---")
        print(gt_text)
        print(f"\n--- GENERATED ({len(gen_ids)} tokens) ---")
        print(generated_text)
        print()

    print("=" * 80)
    print("Done. Generated captions for all samples.")


if __name__ == "__main__":
    main()
