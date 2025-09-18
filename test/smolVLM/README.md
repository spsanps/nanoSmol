# SmolVLM Test Utilities

These scripts mirror the SmolLM helpers but target the multimodal SmolVLM
checkpoints.  They are intentionally lightweight, providing parity checks and
quick sampling for the minimal implementation in `models/smolVLM/`.

## Scripts

- `compare_hf_vs_local.py` — downloads a Hugging Face SmolVLM checkpoint,
  maps the weights into the local implementation, and runs a greedy
  teacher-forced comparison for a text+image prompt.
- `generate_smoke.py` — loads the weights and produces a short generated
  response to a tiny text+image input.

By default the scripts target `HuggingFaceTB/SmolVLM-256M-Base`, which is
public.  Pass `--hf-model` to point at private or fine-tuned variants such as
`HuggingFaceTB/SmolVLM2-256M-Video-Instruct` once you have access to them.
