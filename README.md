Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and the SmolLM/SmolVLM research threads (huggingface/smollm).

## Project philosophy

- **Simplicity and readability are non-negotiable.** Each module should fit on a single screen and read like a step-by-step walkthrough of the underlying math.
- **Education-first comments.** Inline notes should spell out tensor shapes, why operations are performed, and how they connect back to the transformer equations—aim the narration at curious beginners.
- **Descriptive names.** Prefer variable names that communicate their role in the computation over brevity, even if they are longer than typical PyTorch code.

## Goal

Experiment quickly with VLMs (Vision+Language Models) in a minimal codebase while keeping the implementation approachable. We use SmolLM, SmolVLM, and related models as reference points.

## Status

- smolLM implementation (no vision) is available, more coming soon.
