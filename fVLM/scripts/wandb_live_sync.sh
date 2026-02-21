#!/bin/bash
# Periodically sync CSV metrics to wandb
# Run: nohup bash scripts/wandb_live_sync.sh &

export WANDB_API_KEY=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
CSV="/workspace/checkpoints/final_1.7B/stage1/metrics_stage1-1.7B_20260221_190645.csv"

while true; do
    python scripts/sync_csv_to_wandb.py "$CSV" 2>&1 | tail -3
    echo "[$(date)] Synced. Sleeping 5 min..."
    sleep 300
done
