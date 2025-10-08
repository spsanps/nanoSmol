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

The ``train`` package now exposes two intentionally tiny modules:

- ``train.data`` contains reusable dataset pieces – a registry of adapters,
  a conversation-aware tokenizer, and a collator that keeps tensors dense while
  masking out user tokens (NanoGPT style).
- ``train.engine`` wraps ``accelerate`` so the exact same training loop scales
  from laptops to multi-GPU rigs.  It tracks throughput, writes JSON logs, draws
  a ``matplotlib`` curve, and (optionally) streams everything to Weights &
  Biases.

FineVision is just the default adapter; to fine-tune a Hugging Face
``SmolVLM`` checkpoint you can run:

```bash
python scripts/train_multimodal.py \
  --model HuggingFaceM4/smolvlm-instruct \
  --adapter finevision \
  --streaming \
  --batch-size 8 \
  --grad-accum 4 \
  --max-steps 1000 \
  --wandb-project nanoSmol
```

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
