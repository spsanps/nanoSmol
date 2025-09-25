# SmolVLM Baseline Evaluations

## NanoEval (LightEval-free runners)

`eval/nanoeval` now ships a config-driven alternative to LightEval.  Every
benchmark exposes a single module that reads a YAML document, constructs the
requested model, iterates over the dataset once, and writes:

* `predictions.jsonl` — per-example rows for debugging.
* `summary.json` — aggregate metrics, scoring parameters, and model metadata.
* `metrics.csv` + `accuracy.png` — quick-look report of the primary score.

All runtime options live in the YAML.  Point `NANOEVAL_CONFIG` at the file and
launch the module:

```bash
export NANOEVAL_CONFIG=eval/nanoeval/configs/mmlu_tiny.yaml
python -m eval.nanoeval.run_mmlu

export NANOEVAL_CONFIG=eval/nanoeval/configs/hellaswag_tiny.yaml
python -m eval.nanoeval.run_hellaswag

# Requires a VLM checkpoint and the MMMU-Pro dataset access token.
export NANOEVAL_CONFIG=eval/nanoeval/configs/mmmu_pro_smolvlm.yaml
python -m eval.nanoeval.run_mmmu_pro
```

Suites under ``eval/nanoeval/suites`` bundle these tasks for a given checkpoint,
writing a ``suite_summary.json`` alongside the per-task artefacts.  The smoke
suite targets ``sshleifer/tiny-gpt2`` so it can run on CPU in under a minute.

Copy the example configs in `eval/nanoeval/configs/`, adjust the `model` and
`dataset` sections, and re-run.  Outputs default to `artifacts/nanoeval/<task>/`
so experiments stay isolated from the repo tree.

This directory wires up the LightEval baselines for the 256M SmolVLM checkpoints.  The goal is to
reproduce the paper/blog zero-shot numbers on the LightEval-supported suites that cover both text
and image reasoning tasks.

## Benchmarks

The shared [`tasks.txt`](./tasks.txt) file enumerates the LightEval task strings used in every run:

- `leaderboard|mmlu|0|0` – MMLU (all subjects), zero-shot.
- `leaderboard|hellaswag|0|0` – HellaSwag, zero-shot.
- `lighteval|mmmu_pro|0|0` – MMMU-Pro (primary multimodal benchmark).
- `# lighteval|mmmu|0|0` – Original MMMU task.  Uncomment if you need both variants.

Each task string follows the LightEval `<suite>|<task>|<fewshot>|<truncate>` convention.  Keeping the
file at the repo root makes it trivial to add/remove suites without touching the run scripts.

## Environment setup

1. Install the project dependencies (the pinned version includes LightEval):
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate with the Hugging Face Hub so LightEval can stream gated datasets:
   ```bash
   huggingface-cli login
   ```
   Alternatively export `HF_TOKEN` prior to launching a run.
3. (Optional) Point `HF_HOME`/`HF_DATASETS_CACHE` at a local scratch volume if you want to reuse the
   downloads produced by MMMU-Pro.

## Running an evaluation

Each model has a dedicated helper under [`run_eval/`](./run_eval/) that wraps the new Typer-based
`python -m lighteval accelerate` entrypoint.  The scripts set:

- `vision_model=True` so LightEval uses the VLM transformer loader.
- The SmolVLM chat template enforced through `override_chat_template=True` in the model args.
- Deterministic decoding (`temperature=0.0`, `top_p=1.0`, `max_new_tokens=64`) using LightEval's greedy default.
- `dtype=bfloat16`, `device_map=auto`, and `trust_remote_code=True` to match the official configs.

Running the baselines therefore boils down to:

```bash
# V1 – HuggingFaceTB/SmolVLM-256M-Instruct
python eval/run_eval/smolvlm-256m-instruct/run.py

# V2 – HuggingFaceTB/SmolVLM2-256M-Video-Instruct
python eval/run_eval/smolvlm2-256m-video-instruct/run.py
```

Key optional flags:

- `--output-dir`: override the destination directory (defaults live under `eval/output/<model>`).
- `--tasks-file`: point at an alternative task file.
- `--max-samples`: run on a truncated subset for smoke testing.
- `--dry-run`: print the composed LightEval command without executing it.

Both scripts stream LightEval logs to stdout/stderr so you can watch progress in real time.  Finished
runs produce timestamped JSON results in `<output-dir>/results/<model-id>/` and, if
`--save-details` is toggled, parquet dumps in `<output-dir>/details/<model-id>/`.

## Plotting & reporting

Use [`plot_results.py`](./run_eval/plot_results.py) to turn a LightEval `results_*.json` file into a
summary CSV and a horizontal bar plot.  Example:

```bash
python eval/run_eval/plot_results.py \
  --results-json eval/output/smolvlm-256m-instruct/results/HuggingFaceTB/SmolVLM-256M-Instruct/results_20250101T000000.json \
  --output eval/output/smolvlm-256m-instruct/plots/summary.png \
  --table eval/output/smolvlm-256m-instruct/summary.csv
```

The script defaults to the primary metric (`acc`, `accuracy`, `score`, …) for each task and filters
out the per-subject MMLU breakdowns so the plot stays readable.  Switch to full-detail mode with
`--mode full` if you want every sub-task on the chart.

## Output layout

The `eval/output/` tree mirrors the model names so downstream automation can glob for results and
plots:

```
output/
  smolvlm-256m-instruct/
    results/      # LightEval JSON payloads (created by the runner)
    details/      # Optional parquet dumps when --save-details is set
    plots/        # Saved visualisations from plot_results.py
  smolvlm2-256m-video-instruct/
    ...
```

Every directory ships with a lightweight README describing the expected artefacts.  The evaluation
scripts create the folders as needed, so nothing breaks if the repo is cloned onto a clean machine.
