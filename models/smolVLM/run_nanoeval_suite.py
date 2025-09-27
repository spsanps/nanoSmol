"""Kick off the NanoEval suite for the reference SmolVLM checkpoint."""

from __future__ import annotations

from pathlib import Path

from eval.nanoeval.suite import run_suite_from_yaml


def main() -> None:
    config = Path(__file__).resolve().parents[1] / "eval" / "nanoeval" / "suites" / "smolvlm-256m-instruct.yaml"
    run_suite_from_yaml(config)


if __name__ == "__main__":
    main()
