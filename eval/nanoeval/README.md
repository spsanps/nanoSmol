# NanoEval (config-first baseline loops)

NanoEval keeps the original "one-screen" spirit of NanoGPT while shifting all
task parameters into small YAML files.  Each benchmark has a single Python
module that:

1. Loads a :class:`ModelConfig` and dataset settings from the YAML document.
2. Streams the corresponding Hugging Face dataset once.
3. Records predictions, writes JSON/CSV summaries, and emits a quick-look bar
   plot for the primary metric.

The scripts deliberately avoid command-line flags—the only runtime input is the
path to the YAML file, exposed via the ``NANOEVAL_CONFIG`` environment
variable.  Copy an example config, tweak it to your checkpoint, and re-run.

## Layout

```
common.py       # shared model wrapper + RNG seeding helpers
config.py       # dataclasses + YAML loader used by every task
reporting.py    # JSON/CSV/plot emitters for consistent artefacts
prompts.py      # short zero-shot prompt templates
run_mmlu.py     # MMLU macro-accuracy loop
run_hellaswag.py# HellaSwag accuracy loop
run_mmmu_pro.py # MMMU-Pro multimodal accuracy loop
configs/        # example YAML configurations (copy + customise)
```

Outputs land in ``artifacts/nanoeval/<task>/`` by default.  The directory is
git-ignored so you can iterate freely without polluting the repository.

## Running an evaluation

1. Copy one of the example YAML files (e.g. ``configs/mmlu_smolvlm.yaml``) and
   edit the ``model``/``dataset`` sections as needed.
2. Point ``NANOEVAL_CONFIG`` at that file and launch the module.

```bash
export NANOEVAL_CONFIG=eval/nanoeval/configs/mmlu_smolvlm.yaml
python -m eval.nanoeval.run_mmlu

export NANOEVAL_CONFIG=eval/nanoeval/configs/hellaswag_smolvlm.yaml
python -m eval.nanoeval.run_hellaswag

export NANOEVAL_CONFIG=eval/nanoeval/configs/mmmu_pro_smolvlm.yaml
python -m eval.nanoeval.run_mmmu_pro
```

> **Note**
> The MMMU-Pro runner currently supports only greedy letter decoding
> (`scoring.strategy: gen_letter`).  Configs requesting `rank_ll` will raise a
> `ValueError` so mismatched reports don't silently slip through.

## Model-centric suites

When you want to reproduce a paper-style table for a single checkpoint,
coordinate the three tasks with a suite YAML under ``suites/``.  The suite file
defines one model block and a list of task entries; each task inherits the
shared model unless it overrides ``model`` locally.

```bash
export NANOEVAL_SUITE=eval/nanoeval/suites/smolvlm-256m-instruct.yaml
python -m eval.nanoeval.suite
```

Running through ``suite.py`` writes a ``suite_summary.json`` next to the
per-task artefacts so downstream automation can pick up all metrics in one
place.  Copy ``suites/suite_smoke.yaml`` for a tiny, CPU-friendly example.

Each run produces three artefacts in the configured ``report`` directory:

* ``predictions.jsonl`` — one row per dataset example.
* ``summary.json`` — aggregate metrics, scoring parameters, and model metadata.
* ``metrics.csv`` & ``accuracy.png`` — tabular + visual view of the primary
  score.

All behaviour (e.g. log-likelihood scoring, maximum new tokens, per-subject
subsets) is driven by the YAML file so experimenting with new checkpoints or
decoding strategies is as simple as editing the config and re-running.
