# SmolVLM2-256M-Video-Instruct outputs

- `results/` – LightEval JSON summaries (one per evaluation run).
- `details/` – Optional parquet dumps when `--save-details` is enabled on the runner.
- `plots/` – Figures exported by `plot_results.py`.

Example:

```bash
python eval/run_eval/smolvlm2-256m-video-instruct/run.py
python eval/run_eval/plot_results.py \
  --results-json eval/output/smolvlm2-256m-video-instruct/results/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/results_20250101T000000.json \
  --output eval/output/smolvlm2-256m-video-instruct/plots/summary.png
```
