"""Smoke-test the NanoEval suite using a tiny text model."""

from __future__ import annotations

from pathlib import Path

from eval.nanoeval.suite import run_suite_from_yaml


def main() -> None:
    config = Path(__file__).resolve().parents[2] / "eval" / "nanoeval" / "suites" / "suite_smoke.yaml"
    run_suite_from_yaml(config)


if __name__ == "__main__":
    main()
