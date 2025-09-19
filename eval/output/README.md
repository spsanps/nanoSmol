# Evaluation artefact layout

The LightEval runners deposit everything under `eval/output/<model>/`:

- `results/` – JSON summaries from LightEval (one file per run, e.g. `results_20250101T000000.json`).
- `details/` – Optional parquet dumps when `--save-details` is passed to the runner.
- `plots/` – Images produced by `eval/run_eval/plot_results.py`.

The helper scripts create these folders automatically, so you can point `--output-dir` at a fresh
location without pre-seeding it.
