# Project Philosophy (Core)

- **NanoGPT**: heavily inspired by karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT). 
- **Minimal first**: smallest working pieces 
- **One-screen modules**: prefer single, readable files.
- **Config-driven**: experiments defined by configs, not code branches.
- **Deterministic runs**: seeds + pinned versions.

Repo Layout
- models/      — core model(s)
- scripts/     — small utilities (schema dump, conversions)
- configs/     — experiment configs
- artifacts/   — outputs (logs, checkpoints, reports) [gitignored]
