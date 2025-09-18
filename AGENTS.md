# Project Philosophy (Core)

- **NanoGPT**: heavily inspired by karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT). 
- **Minimal first**: smallest working pieces
- **One-screen modules**: prefer single, readable files.
- **Config-driven**: experiments defined by configs, not code branches.
- **Deterministic runs**: seeds + pinned versions.
- **Education-first**: include comments that walk through tensor shapes, math, and reasoning so newcomers understand *why* each step exists.
- **Descriptive names**: choose variable and function names that communicate intent, even if they are longer than typical PyTorch code.

Repo Layout
- models/      — core model(s)
- scripts/     — small utilities (schema dump, conversions)
- configs/     — experiment configs
- artifacts/   — outputs (logs, checkpoints, reports) [gitignored]
- test/        — minimal smoke tests (e.g. generate from pretrained) for each model
