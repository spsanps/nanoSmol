Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and the SmolLM/SmolVLM research threads (huggingface/smollm).

## Project philosophy

- **Simplicity and readability are non-negotiable.** Each module should fit on a single screen and read like a step-by-step walkthrough of the underlying math.
- **Education-first comments.** Inline notes should spell out tensor shapes, why operations are performed, and how they connect back to the transformer equations—aim the narration at curious beginners.
- **Descriptive names.** Prefer variable names that communicate their role in the computation over brevity, even if they are longer than typical PyTorch code.

## Goal

Experiment quickly with VLMs (Vision+Language Models) in a minimal codebase while keeping the implementation approachable. We use SmolLM, SmolVLM, and related models as reference points.

## Status

- smolLM implementation (no vision) is available, more coming soon.

## Multimodal training loop

The ``train`` package is intentionally chopped into one-screen modules so you
can read each piece without scrolling:

- ``train.data.config`` → the dataset dataclass that declares how to stream and
  filter conversations.
- ``train.data.adapters`` → a tiny registry that turns raw HF records into
  user/assistant message transcripts.
- ``train.data.tokenizer`` → converts transcripts into ``input_ids`` and
  ``labels`` while inserting image tokens the same way NanoGPT injects BOS/EOS.
- ``train.data.collator`` → pads text, normalises images, and produces dense
  tensors ready for the model.
- ``train.data.loader`` → glues everything together into a ``DataLoader``.
- ``train.engine`` → the accelerate-powered training loop that scales from a
  laptop GPU to multi-node runs.

FineVision is just the default adapter; to fine-tune a Hugging Face
``SmolVLM`` checkpoint you can run:

```bash
python scripts/train_multimodal.py \
  --experiment smolvlm-siglip \
  --model HuggingFaceM4/smolvlm-instruct \
  --streaming \
  --batch-size 8 \
  --grad-accum 4 \
  --max-steps 1000 \
  --wandb-project nanoSmol
```

The experiment wiring lives under ``experiments/``—``smolvlm-siglip`` wires up
SigLIP vision + SmolVLM, loads both modules from the Hugging Face checkpoint you
point ``--model`` at (``--model-revision`` / ``--model-token`` are available for
private repos), and hands everything to the generic training loop.  Use
``--model-local-only`` to ensure we only touch your local cache.

Adapters are pluggable (register them in ``train.data``), so future datasets can
reuse the same training loop without rewriting data plumbing.  The logger emits
loss, learning-rate, tokens/sec, samples/sec, and cumulative counters so you can
track efficiency during long runs. Checkpointing and model exports are handled
for you as well:

- ``--num-checkpoints`` / ``--checkpoint-interval`` keep at most a handful of
  restore points during the run (default ≈10 evenly spaced saves).
- ``--checkpoint-dir`` and ``--checkpoint-limit`` control where checkpoints
  land and how many we keep on disk.
- ``--final-model-dir`` saves the last-step weights (plus tokenizer) locally and
  ``--push-to-hub`` uploads that bundle to ``--hub-model-id`` using the provided
  branch/token configuration.
