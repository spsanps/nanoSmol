"""CLI wrapper around the modular multimodal trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from models.smolVLM.config import SmolVLMConfig
from models.smolVLM.model import SmolVLM
from train import (
    ChatDataConfig,
    TrainingConfig,
    Trainer,
    available_adapters,
    build_chat_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal fine-tuning loop (accelerate-powered)")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model id (e.g. HuggingFaceM4/smolvlm-instruct)")
    parser.add_argument("--adapter", type=str, default="finevision", choices=available_adapters(), help="Dataset adapter key")
    parser.add_argument("--dataset", type=str, default="HuggingFaceM4/FineVision-1.0", help="Dataset repository id")
    parser.add_argument("--subset", type=str, default=None, help="Optional dataset subset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to stream")
    parser.add_argument("--streaming", action="store_true", help="Enable 🤗 streaming")
    parser.add_argument("--shuffle-buffer", type=int, default=1024, help="Shuffle buffer size when streaming")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    parser.add_argument("--min-quality", type=int, default=4, help="Drop samples below this rating (adapter-specific)")
    parser.add_argument("--max-images", type=int, default=1, help="Images per conversation")
    parser.add_argument("--max-turns", type=int, default=None, help="Optional cap on QA turns")
    parser.add_argument("--image-size", type=int, default=384, help="Image resolution for the collator")
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=1000, help="Number of optimisation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--warmup", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), metavar=("B1", "B2"), help="Adam betas")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping (disabled if negative)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--compile", action="store_true", help="Torch compile the model")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode")
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/train"), help="Directory for logs + plots")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="Weights & Biases group name")
    parser.add_argument("--resume-wandb", action="store_true", help="Resume an existing Weights & Biases run")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"), help="Folder for training checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Steps between checkpoints (overrides --num-checkpoints)")
    parser.add_argument("--num-checkpoints", type=int, default=10, help="Target number of checkpoints across the run")
    parser.add_argument("--checkpoint-limit", type=int, default=None, help="Maximum checkpoints to retain (oldest pruned)")
    parser.add_argument("--final-model-dir", type=Path, default=Path("artifacts/final-model"), help="Directory for the exported final model")
    parser.add_argument("--hub-model-id", type=str, default=None, help="Hugging Face repository to push the final model to")
    parser.add_argument("--hub-branch", type=str, default=None, help="Hugging Face branch/revision for uploads")
    parser.add_argument("--hub-private", action="store_true", help="Create the Hugging Face repo as private")
    parser.add_argument("--hub-token", type=str, default=None, help="Hugging Face user token (uses env token if omitted)")
    parser.add_argument("--hub-commit-message", type=str, default="Add trained weights", help="Commit message when pushing to Hugging Face")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the final model to Hugging Face after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_cfg = AutoConfig.from_pretrained(args.model)
    model_cfg = SmolVLMConfig.from_hf_config(hf_cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = SmolVLM(model_cfg)

    data_cfg = ChatDataConfig(
        repo_id=args.dataset,
        subset=args.subset,
        split=args.split,
        streaming=args.streaming,
        shuffle_buffer_size=args.shuffle_buffer,
        seed=args.seed,
        max_turns=args.max_turns,
        max_images=args.max_images,
        image_size=args.image_size,
        min_quality=args.min_quality,
        adapter=args.adapter,
    )

    train_cfg = TrainingConfig(
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        warmup_steps=args.warmup,
        max_grad_norm=None if args.max_grad_norm < 0 else args.max_grad_norm,
        compile_model=args.compile,
        log_every=10,
        log_dir=str(args.log_dir) if args.log_dir else None,
        mixed_precision=args.mixed_precision,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        wandb_group=args.wandb_group,
        resume_wandb=args.resume_wandb,
        checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_interval=args.checkpoint_interval,
        num_checkpoints=args.num_checkpoints,
        checkpoint_total_limit=args.checkpoint_limit,
        final_model_dir=str(args.final_model_dir) if args.final_model_dir else None,
        hub_model_id=args.hub_model_id,
        hub_revision=args.hub_branch,
        hub_token=args.hub_token,
        hub_private=args.hub_private,
        hub_commit_message=args.hub_commit_message,
        hub_push_final=args.push_to_hub,
    )

    dataloader = build_chat_dataloader(
        data_cfg,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    trainer = Trainer(model, optimizer, dataloader, train_cfg, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()

