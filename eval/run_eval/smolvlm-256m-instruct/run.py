#!/usr/bin/env python
"""LightEval launcher for `HuggingFaceTB/SmolVLM-256M-Instruct`.

The script composes the `python -m lighteval accelerate` call with the zero-shot
settings requested in the baseline plan:

- Deterministic decoding (no sampling, 64 new tokens, `temperature=0.0`, `top_p=1.0`).
- Chat templating forced through `override_chat_template=True` in the model args.
- Vision model loader (`--vision-model`) so image inputs are handled correctly.
- bfloat16 weights with `device_map=auto` and `trust_remote_code=True`.

Edit the constants below if you want to experiment with alternative checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "eval"
DEFAULT_TASKS = EVAL_DIR / "tasks.txt"
DEFAULT_OUTPUT = EVAL_DIR / "output" / "smolvlm-256m-instruct"

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROCESSOR_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

_BASE_MODEL_ARGS = {
    "model_name": MODEL_NAME,
    "processor": PROCESSOR_NAME,
    "dtype": "bfloat16",
    "batch_size": 1,
    "trust_remote_code": True,
    "device_map": "auto",
    "override_chat_template": True,
}
_BASE_GENERATION_ARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 64,
}


def _encode_generation_parameters(params: dict[str, Any]) -> str:
    """Convert the generation dictionary into the LightEval CLI mini-JSON format."""

    pieces: list[str] = []
    for key, value in params.items():
        if isinstance(value, bool):
            json_value = "true" if value else "false"
        else:
            json_value = json.dumps(value)
        pieces.append(f"{key}:{json_value}")
    return "{" + ",".join(pieces) + "}"


def _build_model_args(args: argparse.Namespace) -> str:
    model_args = dict(_BASE_MODEL_ARGS)
    model_args["batch_size"] = args.batch_size

    generation_args = dict(_BASE_GENERATION_ARGS)
    generation_args.update(
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    model_args["generation_parameters"] = generation_args

    parts: list[str] = []
    for key, value in model_args.items():
        if key == "generation_parameters":
            parts.append(f"{key}={_encode_generation_parameters(value)}")
            continue
        if isinstance(value, bool):
            formatted = "True" if value else "False"
        else:
            formatted = str(value)
        parts.append(f"{key}={formatted}")

    if args.extra_model_args:
        parts.append(args.extra_model_args.strip(","))

    return ",".join(parts)


def _create_output_scaffolding(output_dir: Path) -> None:
    for sub in ("results", "details", "plots"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightEval on SmolVLM-256M-Instruct")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Destination for LightEval artefacts")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS, help="Text file with LightEval task strings")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of samples per task (debug only)")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size used by LightEval")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate per sample")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (kept at 0.0 for baseline runs)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p value used during generation")
    parser.add_argument(
        "--dataset-loading-processes",
        type=int,
        default=1,
        help="Number of worker processes used to stream evaluation datasets",
    )
    parser.add_argument("--save-details", action="store_true", help="Persist per-sample parquet dumps alongside the summary")
    parser.add_argument("--extra-model-args", type=str, default=None, help="Additional raw key=value pairs appended to model_args")
    parser.add_argument("--dry-run", action="store_true", help="Print the composed LightEval command without executing it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.tasks_file.exists():
        raise FileNotFoundError(f"Task file not found: {args.tasks_file}")

    _create_output_scaffolding(args.output_dir)

    model_args = _build_model_args(args)
    tasks_argument = str(args.tasks_file)

    cmd = [
        sys.executable,
        "-m",
        "lighteval",
        "accelerate",
        model_args,
        tasks_argument,
        "--output-dir",
        str(args.output_dir),
        "--vision-model",
        "--dataset-loading-processes",
        str(args.dataset_loading_processes),
    ]

    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.save_details:
        cmd.append("--save-details")

    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[smolvlm-256m-instruct] Launching LightEval:\n{printable}")

    if args.dry_run:
        return 0

    process = subprocess.run(cmd, check=False)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
