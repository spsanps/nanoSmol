# SmolVLM-256M-Instruct LightEval runner

```bash
python eval/run_eval/smolvlm-256m-instruct/run.py
```

The helper wraps the Typer-based LightEval CLI and pins the baseline decoding setup (zero-shot,
chat templating, deterministic decoding, bfloat16 weights).  Use `--dry-run` to inspect the composed
command, `--max-samples` for smoke tests, and `--extra-model-args` to append additional
`key=value` overrides to the LightEval `model_args` string.
