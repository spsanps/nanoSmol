import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch

from experiments import available_experiments, get_experiment
from train import (
    ChatDataConfig,
    TrainingConfig,
    Trainer,
    available_adapters,
    build_chat_dataloader,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal fine-tuning loop (accelerate-powered)")
    parser.add_argument(
        "--experiment",
        type=str,
        default="smolvlm-siglip",
        choices=list(available_experiments()),
        help="Experiment wiring that instantiates the model/tokenizer stack",
    )
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model id for weights + tokenizer")
    parser.add_argument("--model-revision", type=str, default=None, help="Optional model revision (branch/tag/commit)")
    parser.add_argument("--model-token", type=str, default=None, help="Hugging Face token for private models")
    parser.add_argument(
        "--model-local-only",
        action="store_true",
        help="Only use locally cached model weights (no network calls)",
    )
    parser.add_argument("--adapter", type=str, default=None, help="Dataset adapter key")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset repository id")
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
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), metavar=("B1", "B2"), help="Adam betas")
    parser.add_argument("--warmup", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping (disabled if negative)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--compile", action="store_true", help="Torch compile the model")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Steps between console logs")
    parser.add_argument(
        "--save-final",
        type=Path,
        default=None,
        help="Optional directory to store the trained model weights",
    )
    return parser


def parse_args() -> Tuple[argparse.Namespace, object]:
    parser = build_parser()
    args = parser.parse_args()
    experiment = get_experiment(args.experiment)
    experiment.apply_defaults(args)

    adapter_choices = set(available_adapters())
    if args.adapter not in adapter_choices:
        parser.error(f"Unknown adapter '{args.adapter}'. Available: {sorted(adapter_choices)}")

    if args.dataset is None:
        parser.error("--dataset must be provided either explicitly or via the experiment defaults")

    return args, experiment


def maybe_save_final_model(model, tokenizer, output_dir: Optional[Path]) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(output_dir, safe_serialization=True)
    else:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)
    if callable(save_tokenizer):
        save_tokenizer(output_dir)


def main() -> None:
    args, experiment = parse_args()
    artifacts = experiment.build(args)
    model = artifacts.model
    tokenizer = artifacts.tokenizer
    model_cfg = artifacts.model_config

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
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_grad_norm=None if args.max_grad_norm < 0 else args.max_grad_norm,
        compile_model=args.compile,
        log_every=args.log_every,
        mixed_precision=args.mixed_precision,
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
        lr=train_cfg.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    callbacks = list(artifacts.callbacks)
    trainer = Trainer(
        model,
        optimizer,
        dataloader,
        train_cfg,
        artifacts.step_fn,
        callbacks=callbacks,
    )
    trainer.train()

    maybe_save_final_model(model, tokenizer, args.save_final)


if __name__ == "__main__":
    main()
