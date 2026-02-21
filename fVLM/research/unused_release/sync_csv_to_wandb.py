"""Sync CSV training metrics to a wandb run."""
import csv
import os
import sys

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/checkpoints/final_1.7B/stage1/metrics_stage1-1.7B_20260221_190645.csv"

    # Load WANDB key from .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v

    import wandb

    # Read CSV
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from {csv_path}")

    # Init wandb run
    run = wandb.init(
        project="foveated-vlm-final",
        name="stage1-1.7B",
        id="stage1_1_7B_synced",
        resume="allow",
    )

    train_count = 0
    eval_count = 0
    for row in rows:
        step = int(row.get("step", 0))
        event_type = row.get("event_type", "")

        if event_type == "train":
            log_dict = {}
            if row.get("train_loss"):
                log_dict["train/loss"] = float(row["train_loss"])
            if row.get("loss_fine"):
                log_dict["train/fine_loss"] = float(row["loss_fine"])
            if row.get("grad_norm"):
                log_dict["train/grad_norm"] = float(row["grad_norm"])
            if row.get("lr_connector"):
                log_dict["train/lr"] = float(row["lr_connector"])
            if row.get("lr_llm"):
                log_dict["train/lr_llm"] = float(row["lr_llm"])
            if row.get("throughput_samples_sec"):
                log_dict["train/throughput"] = float(row["throughput_samples_sec"])
            if row.get("gpu_mem_gb"):
                log_dict["train/gpu_mem_gb"] = float(row["gpu_mem_gb"])
            if row.get("samples_seen"):
                log_dict["train/samples_seen"] = int(row["samples_seen"])

            if log_dict:
                wandb.log(log_dict, step=step)
                train_count += 1

        elif event_type == "eval":
            log_dict = {}
            if row.get("val_loss"):
                log_dict["eval/val_loss"] = float(row["val_loss"])
            if row.get("val_loss_fine"):
                log_dict["eval/val_fine_loss"] = float(row["val_loss_fine"])
            if row.get("attention_entropy"):
                log_dict["eval/attention_entropy"] = float(row["attention_entropy"])

            if log_dict:
                wandb.log(log_dict, step=step)
                eval_count += 1

    print(f"Logged {train_count} train steps and {eval_count} eval steps to wandb")
    print(f"Run URL: {run.url}")
    wandb.finish()

if __name__ == "__main__":
    main()
