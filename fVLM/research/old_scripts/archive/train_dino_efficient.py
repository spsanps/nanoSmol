#!/usr/bin/env python3
"""
Efficient JOINT Training with Precomputed DINO Features.

Uses precomputed DINO patch features for maximum training efficiency:
- Skips DINO forward pass entirely (biggest speedup)
- Shallow query attention on cached features
- Multi-fine iterations (coarse → fine₁ → fine₂)
- Resumable checkpoints for 1-2 hour chunks
- wandb logging with plots

## CRITICAL: Joint Training with Autoregressive Captioning

This script implements JOINT training with:
1. **Caption Loss (Primary)**: Autoregressive text generation after video tokens
2. **Reconstruction Loss (Secondary)**: Next-frame VAE latent prediction

### Why Caption Loss is Essential:

The caption loss teaches the model WHERE to look:
- Reconstruction alone just learns to compress visual features
- Caption loss provides semantic signal: "what's important in this video?"
- The model learns to attend to regions relevant for describing the content
- This guides the foveated attention mechanism to focus on meaningful areas

### Sequence Format:
```
[mode_token, z_1, z_2, ..., z_T, caption_token_1, caption_token_2, ...]
         ↑                    ↑           ↑
         |                    |           |
    visual features      predict these autoregressively
```

### Training Objectives:
- loss_caption: Cross-entropy on caption tokens (positions after z_T)
- loss_reconstruction: MSE on predicted VAE latents
- loss = loss_caption + λ_recon * loss_reconstruction

Success metric: loss_fine < loss_coarse (by 5-15%+)
"""

import sys
import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
import argparse

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.prediction import PredictionHead

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Training
    "max_hours": 2.0,
    "batch_size": 32,  # Reduced for joint training (more VRAM needed)
    "grad_accum": 2,   # Effective batch = 64
    "num_frames": 16,  # Use 16 of 24 available frames
    "learning_rate": 1e-4,  # Higher LR for larger batch
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,

    # Architecture
    "dino_dim": 384,
    "llm_dim": 576,
    "query_dim": 384,
    "lambda_coarse": 1.0,  # Auxiliary loss weight for coarse pass
    "lambda_recon": 0.1,   # Reconstruction loss weight (caption loss is primary)
    "fine_iterations": 2,  # coarse → fine₁ → fine₂
    "max_caption_tokens": 64,  # Max caption length

    # Logging
    "log_interval": 25,
    "save_interval": 500,

    # Data
    "data_dir": "/mnt/d/projects/fVLM/data/precomputed_dino_100k",
    "num_workers": 0,  # 0 workers to avoid multiprocessing issues on WSL
    "pin_memory": False,

    # Checkpointing
    "output_dir": "outputs/dino_efficient_joint",  # New dir for joint training
    "resume_checkpoint": None,

    # wandb
    "wandb_project": "foveated-vlm-efficient",
    "wandb_run_name": None,
}


class EfficientFoveatedModel(nn.Module):
    """
    Efficient foveated model using precomputed DINO features.

    Skips DINO encoder entirely - uses shallow query attention
    on precomputed patch features.

    Supports JOINT training with:
    - Caption loss (autoregressive text generation)
    - Reconstruction loss (next-frame VAE latent prediction)
    """

    def __init__(
        self,
        dino_dim: int = 384,
        llm_dim: int = 576,
        query_dim: int = 384,
        llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        lambda_coarse: float = 1.0,
        lambda_recon: float = 0.1,
    ):
        super().__init__()

        self.dino_dim = dino_dim
        self.llm_dim = llm_dim
        self.query_dim = query_dim
        self.lambda_coarse = lambda_coarse
        self.lambda_recon = lambda_recon

        # Query vectors - CRITICAL: std=1.0 for differentiation!
        self.q_static = nn.Parameter(torch.randn(1, query_dim))  # Coarse query
        self.q_init = nn.Parameter(torch.randn(1, query_dim))    # Fine initial query

        # Query projection (bias=False critical!)
        self.query_proj = nn.Linear(query_dim, dino_dim, bias=False)

        # Core LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm.config.use_cache = False
        self.llm.config.output_hidden_states = True

        # Visual scale for embedding normalization
        self.visual_scale = nn.Parameter(torch.tensor(1.0))

        # Projections
        self.dino_to_llm = nn.Linear(dino_dim, llm_dim)
        self.llm_to_query = nn.Linear(llm_dim, query_dim)

        # Mode tokens (learnable)
        self.coarse_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
        self.fine_token = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)

        # Prediction head for reconstruction
        self.pred_head = PredictionHead(h_dim=llm_dim, latent_channels=4)

        # Initial VAE latent for first frame prediction
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))

    def get_vocab_size(self) -> int:
        """Get vocabulary size for caption loss computation."""
        return self.llm.config.vocab_size

    def query_attend(self, query: torch.Tensor, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Shallow query attention on precomputed DINO features.

        Args:
            query: [B, query_dim] query vector
            patch_features: [B, N, dino_dim] precomputed DINO patches

        Returns:
            z: [B, dino_dim] attended features
        """
        # Project query
        q = self.query_proj(query)  # [B, dino_dim]
        q = q.unsqueeze(1)  # [B, 1, dino_dim]

        # Cross-attention
        attn_scores = torch.bmm(q, patch_features.transpose(1, 2))  # [B, 1, N]
        attn_weights = F.softmax(attn_scores / math.sqrt(self.dino_dim), dim=-1)

        # Attended output
        z = torch.bmm(attn_weights, patch_features)  # [B, 1, dino_dim]
        return z.squeeze(1)  # [B, dino_dim]

    def forward_joint(
        self,
        dino_features: torch.Tensor,
        latents: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        tokenizer,
        fine_iterations: int = 2,
    ):
        """
        Joint forward pass with caption + reconstruction loss.

        This is the CORRECT training approach:
        - Caption loss teaches the model WHAT to look for (semantic understanding)
        - Reconstruction loss teaches the model HOW to represent visual content
        - Together they guide WHERE the model should attend

        Sequence format: [mode_token, z_1, ..., z_T, caption_tokens...]
                               ↑                        ↑
                        visual features         predict these autoregressively

        Args:
            dino_features: [B, T, N, D] precomputed DINO patch features
            latents: [B, T, 4, 32, 32] target VAE latents
            caption_ids: [B, L] tokenized caption IDs
            caption_mask: [B, L] attention mask for captions
            tokenizer: Tokenizer for embedding and loss computation
            fine_iterations: Number of fine passes (default 2)

        Returns:
            loss: Total loss (caption + λ * reconstruction)
            loss_cap_fine: Final fine caption loss
            loss_cap_coarse: Coarse caption loss
            loss_rec_fine: Final fine reconstruction loss
            loss_rec_coarse: Coarse reconstruction loss
        """
        B, T, N, D = dino_features.shape
        device = dino_features.device

        # Embed captions using LLM's embedding layer
        caption_embeds = self.llm.model.embed_tokens(caption_ids)  # [B, L, llm_dim]
        caption_targets = caption_ids[:, 1:]  # Shifted targets for autoregressive loss

        # prev_latents for reconstruction conditioning (first frame uses learned init)
        z_vae_init = self.z_vae_init.expand(B, -1, -1, -1).unsqueeze(1)  # [B, 1, 4, 32, 32]
        prev_latents = torch.cat([z_vae_init, latents[:, :-1]], dim=1)  # [B, T, 4, 32, 32]

        # ============================================
        # PASS 1: Coarse (static query)
        # ============================================

        # Expand static query for batch
        q_static = self.q_static.expand(B, -1)  # [B, query_dim]

        # Extract coarse features for each frame
        z_coarse_list = []
        for t in range(T):
            z_t = self.query_attend(q_static, dino_features[:, t])  # [B, dino_dim]
            z_coarse_list.append(z_t)
        z_coarse = torch.stack(z_coarse_list, dim=1)  # [B, T, dino_dim]

        # Project to LLM dimension with normalization
        z_coarse_llm = self.dino_to_llm(z_coarse)  # [B, T, llm_dim]
        z_coarse_llm = z_coarse_llm / (z_coarse_llm.std() + 1e-6) * self.visual_scale

        # Mode tokens
        coarse_tok = self.coarse_token.expand(B, -1, -1)  # [B, 1, llm_dim]
        fine_tok = self.fine_token.expand(B, -1, -1)

        # --- Coarse caption loss ---
        # Sequence: [coarse_token, z_coarse, caption_embeds]
        seq_cap_coarse = torch.cat([coarse_tok, z_coarse_llm, caption_embeds], dim=1)
        outputs_cap_coarse = self.llm.model(inputs_embeds=seq_cap_coarse)
        logits_cap_coarse = self.llm.lm_head(outputs_cap_coarse.last_hidden_state)
        # Caption logits are at positions after [coarse_token + z_visual], excluding last
        caption_logits_coarse = logits_cap_coarse[:, 1+T:-1, :]  # [B, L-1, vocab]
        loss_cap_coarse = F.cross_entropy(
            caption_logits_coarse.reshape(-1, caption_logits_coarse.size(-1)),
            caption_targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # --- Coarse reconstruction loss ---
        # Use hidden states at visual token positions for reconstruction
        h_coarse = outputs_cap_coarse.last_hidden_state[:, 1:1+T]  # [B, T, llm_dim]
        pred_coarse = self.pred_head(h_coarse, prev_latents)  # [B, T, 4, 32, 32]
        loss_rec_coarse = F.mse_loss(pred_coarse, latents)

        # --- Generate queries for fine pass ---
        queries = self.llm_to_query(h_coarse)  # [B, T, query_dim]

        # ============================================
        # PASS 2+: Fine iterations
        # ============================================

        loss_cap_fine = None
        loss_rec_fine = None
        q_init = self.q_init.expand(B, -1)  # [B, query_dim]

        # Build query sequence: q_init for frame 0, then shifted queries
        current_queries = torch.cat([
            q_init.unsqueeze(1),  # [B, 1, query_dim]
            queries[:, :-1]       # [B, T-1, query_dim]
        ], dim=1)  # [B, T, query_dim]

        for iteration in range(fine_iterations):
            # Extract fine features with autoregressive queries
            z_fine_list = []
            for t in range(T):
                z_t = self.query_attend(current_queries[:, t], dino_features[:, t])
                z_fine_list.append(z_t)
            z_fine = torch.stack(z_fine_list, dim=1)  # [B, T, dino_dim]

            # Project to LLM dimension with normalization
            z_fine_llm = self.dino_to_llm(z_fine)  # [B, T, llm_dim]
            z_fine_llm = z_fine_llm / (z_fine_llm.std() + 1e-6) * self.visual_scale

            # --- Fine caption loss ---
            # Sequence: [fine_token, z_fine, caption_embeds]
            seq_cap_fine = torch.cat([fine_tok, z_fine_llm, caption_embeds], dim=1)
            outputs_cap_fine = self.llm.model(inputs_embeds=seq_cap_fine)
            logits_cap_fine = self.llm.lm_head(outputs_cap_fine.last_hidden_state)
            caption_logits_fine = logits_cap_fine[:, 1+T:-1, :]
            loss_cap_fine = F.cross_entropy(
                caption_logits_fine.reshape(-1, caption_logits_fine.size(-1)),
                caption_targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )

            # --- Fine reconstruction loss ---
            h_fine = outputs_cap_fine.last_hidden_state[:, 1:1+T]
            pred_fine = self.pred_head(h_fine, prev_latents)
            loss_rec_fine = F.mse_loss(pred_fine, latents)

            # Update queries for next iteration
            if iteration < fine_iterations - 1:
                next_queries = self.llm_to_query(h_fine)
                current_queries = torch.cat([
                    q_init.unsqueeze(1),
                    next_queries[:, :-1]
                ], dim=1)

        # ============================================
        # Total loss computation
        # ============================================
        # Caption loss is PRIMARY (teaches WHERE to look)
        # Reconstruction loss is SECONDARY (teaches visual compression)
        loss_coarse = loss_cap_coarse + self.lambda_recon * loss_rec_coarse
        loss_fine = loss_cap_fine + self.lambda_recon * loss_rec_fine
        loss = loss_fine + self.lambda_coarse * loss_coarse

        return loss, loss_cap_fine, loss_cap_coarse, loss_rec_fine, loss_rec_coarse


class PrecomputedDINODataset(Dataset):
    """Dataset loading precomputed DINO features."""

    def __init__(self, data_dir: str, num_frames: int = 16):
        self.data_dir = Path(data_dir)
        self.feature_dir = self.data_dir / "features"
        self.num_frames = num_frames

        # Find all files
        self.files = sorted(self.feature_dir.glob("*.pt"), key=lambda x: int(x.stem))
        print(f"Found {len(self.files)} precomputed videos")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        dino_features = data["dino_features"].float()  # [T, N, D]
        latents = data["latents"]  # [T, 4, 32, 32]
        caption = data.get("caption", "")

        T = dino_features.shape[0]

        # Sample frames if needed
        if T > self.num_frames:
            indices = torch.linspace(0, T-1, self.num_frames).long()
            dino_features = dino_features[indices]
            latents = latents[indices]
        elif T < self.num_frames:
            # Pad
            pad_dino = dino_features[-1:].repeat(self.num_frames - T, 1, 1)
            pad_latents = latents[-1:].repeat(self.num_frames - T, 1, 1, 1)
            dino_features = torch.cat([dino_features, pad_dino], dim=0)
            latents = torch.cat([latents, pad_latents], dim=0)

        return {
            "dino_features": dino_features,
            "latents": latents,
            "caption": caption,
        }


def make_collate_fn(tokenizer, max_caption_tokens):
    """Create collate function with tokenizer for caption processing."""
    def collate_fn(batch):
        dino_features = torch.stack([b["dino_features"] for b in batch])
        latents = torch.stack([b["latents"] for b in batch])
        captions = [b["caption"] for b in batch]

        # Tokenize captions
        tokenized = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_caption_tokens,
            return_tensors="pt"
        )

        return {
            "dino_features": dino_features,
            "latents": latents,
            "caption_ids": tokenized["input_ids"],
            "caption_mask": tokenized["attention_mask"],
            "captions": captions,
        }
    return collate_fn


def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, metrics, config, output_dir):
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
    torch.save(checkpoint, output_dir / f"checkpoint_step{step}.pt")
    print(f"Saved checkpoint at step {step}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["step"], checkpoint["epoch"], checkpoint.get("metrics", {})


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if HAS_WANDB:
        run_name = config["wandb_run_name"] or f"joint_efficient_{datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project=config["wandb_project"], name=run_name, config=config, resume="allow")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset with tokenized captions
    dataset = PrecomputedDINODataset(config["data_dir"], num_frames=config["num_frames"])
    collate_fn = make_collate_fn(tokenizer, config["max_caption_tokens"])
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=config["pin_memory"],
        collate_fn=collate_fn, drop_last=True,
    )

    print(f"Dataset: {len(dataset)} videos")
    print(f"Batches/epoch: {len(dataloader)}")

    # Model
    print("Loading model...")
    model = EfficientFoveatedModel(
        dino_dim=config["dino_dim"],
        llm_dim=config["llm_dim"],
        query_dim=config["query_dim"],
        lambda_coarse=config["lambda_coarse"],
        lambda_recon=config["lambda_recon"],
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params/1e6:.1f}M trainable / {total_params/1e6:.1f}M total")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    # Scheduler (cosine with warmup)
    total_steps = int(config["max_hours"] * 3600 / 2)  # Rough estimate
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        progress = (step - config["warmup_steps"]) / max(1, total_steps - config["warmup_steps"])
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler()

    # Resume
    start_step, start_epoch = 0, 0
    metrics_history = {"loss": [], "loss_fine": [], "loss_coarse": [], "ratio": []}
    best_ratio = 0.0

    latest_ckpt = output_dir / "checkpoint_latest.pt"
    if config["resume_checkpoint"]:
        print(f"Resuming from {config['resume_checkpoint']}")
        start_step, start_epoch, loaded = load_checkpoint(
            config["resume_checkpoint"], model, optimizer, scheduler, scaler
        )
        metrics_history = loaded.get("history", metrics_history)
        best_ratio = loaded.get("best_ratio", 0.0)
    elif latest_ckpt.exists():
        print(f"Found checkpoint at {latest_ckpt}")
        start_step, start_epoch, loaded = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler
        )
        metrics_history = loaded.get("history", metrics_history)
        best_ratio = loaded.get("best_ratio", 0.0)

    # Training
    model.train()
    global_step = start_step
    epoch = start_epoch
    start_time = time.time()
    max_runtime = config["max_hours"] * 3600

    # Running metrics for averaging
    running_loss = 0.0
    running_cap_fine = 0.0
    running_cap_coarse = 0.0
    running_rec_fine = 0.0
    running_rec_coarse = 0.0
    running_count = 0
    step_times = []

    print(f"\n{'='*70}")
    print(f"JOINT TRAINING: Caption + Reconstruction, Multi-Fine ({config['fine_iterations']} iterations)")
    print(f"{'='*70}")
    print(f"Max runtime: {config['max_hours']}h")
    print(f"Batch: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"Frames: {config['num_frames']}")
    print(f"Lambda recon: {config['lambda_recon']} (caption loss is primary)")
    print(f"Lambda coarse: {config['lambda_coarse']}")
    print(f"{'='*70}")
    print()
    print("WHY JOINT TRAINING MATTERS:")
    print("  - Caption loss teaches WHERE to look (semantic attention)")
    print("  - Reconstruction alone only learns visual compression")
    print("  - Together: model learns to attend to regions that matter for understanding")
    print()
    print("SUCCESS METRIC: cap_fine < cap_coarse (fine path produces better captions)")
    print(f"{'='*70}\n")

    optimizer.zero_grad()
    accum_count = 0

    while True:
        epoch += 1

        for batch in dataloader:
            step_start = time.time()

            # Check time
            elapsed = time.time() - start_time
            if elapsed >= max_runtime:
                print(f"\nTime limit reached ({config['max_hours']}h)")
                break

            # Move to device
            dino_features = batch["dino_features"].to(device)
            latents = batch["latents"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            caption_mask = batch["caption_mask"].to(device)

            # Forward with joint loss
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, cap_fine, cap_coarse, rec_fine, rec_coarse = model.forward_joint(
                    dino_features, latents, caption_ids, caption_mask, tokenizer,
                    fine_iterations=config["fine_iterations"],
                )
                loss = loss / config["grad_accum"]

            # Backward
            scaler.scale(loss).backward()
            accum_count += 1

            # Track metrics
            running_loss += loss.item() * config["grad_accum"]
            running_cap_fine += cap_fine.item()
            running_cap_coarse += cap_coarse.item()
            running_rec_fine += rec_fine.item()
            running_rec_coarse += rec_coarse.item()
            running_count += 1

            # Optimizer step
            if accum_count >= config["grad_accum"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                step_time = time.time() - step_start
                step_times.append(step_time)
                if len(step_times) > 50:
                    step_times.pop(0)

                # Logging
                if global_step % config["log_interval"] == 0:
                    avg_loss = running_loss / running_count
                    avg_cap_fine = running_cap_fine / running_count
                    avg_cap_coarse = running_cap_coarse / running_count
                    avg_rec_fine = running_rec_fine / running_count
                    avg_rec_coarse = running_rec_coarse / running_count

                    # Caption ratio (primary metric)
                    cap_ratio = avg_cap_coarse / avg_cap_fine if avg_cap_fine > 0 else 1.0
                    cap_improvement = (cap_ratio - 1.0) * 100

                    # Reconstruction ratio (secondary)
                    rec_ratio = avg_rec_coarse / avg_rec_fine if avg_rec_fine > 0 else 1.0

                    remaining = max_runtime - elapsed
                    eta = format_time(remaining)
                    progress = elapsed / max_runtime * 100

                    # Key metric: caption_fine < caption_coarse means hypothesis validated
                    status = "✓" if cap_ratio > 1.0 else "✗"

                    print(f"Step {global_step:5d} | Loss: {avg_loss:.3f} | "
                          f"Cap[F:{avg_cap_fine:.3f} C:{avg_cap_coarse:.3f} R:{cap_ratio:.3f}] {status} | "
                          f"Rec[F:{avg_rec_fine:.3f} C:{avg_rec_coarse:.3f}] | "
                          f"LR: {scheduler.get_last_lr()[0]:.1e} | {progress:.0f}% | ETA: {eta}")

                    if HAS_WANDB:
                        wandb.log({
                            "loss": avg_loss,
                            "cap_fine": avg_cap_fine,
                            "cap_coarse": avg_cap_coarse,
                            "cap_ratio": cap_ratio,
                            "cap_improvement_pct": cap_improvement,
                            "rec_fine": avg_rec_fine,
                            "rec_coarse": avg_rec_coarse,
                            "rec_ratio": rec_ratio,
                            "lr": scheduler.get_last_lr()[0],
                            "step": global_step,
                            "epoch": epoch,
                        })

                    # Track history (primary metric is caption ratio)
                    metrics_history["loss"].append(avg_loss)
                    metrics_history["loss_fine"].append(avg_cap_fine)
                    metrics_history["loss_coarse"].append(avg_cap_coarse)
                    metrics_history["ratio"].append(cap_ratio)

                    if cap_ratio > best_ratio:
                        best_ratio = cap_ratio

                    running_loss = running_cap_fine = running_cap_coarse = 0.0
                    running_rec_fine = running_rec_coarse = 0.0
                    running_count = 0

                # Save checkpoint
                if global_step % config["save_interval"] == 0:
                    metrics = {"history": metrics_history, "best_ratio": best_ratio}
                    save_checkpoint(model, optimizer, scheduler, scaler, global_step, epoch, metrics, config, output_dir)

        # Check time after epoch
        if time.time() - start_time >= max_runtime:
            break

    # Final save
    metrics = {"history": metrics_history, "best_ratio": best_ratio}
    save_checkpoint(model, optimizer, scheduler, scaler, global_step, epoch, metrics, config, output_dir)

    # Summary
    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\n{'='*70}")
    print("JOINT TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Steps: {global_step}")
    print(f"Runtime: {elapsed_hours:.2f}h")
    print(f"Best caption ratio: {best_ratio:.3f}")

    if metrics_history["ratio"]:
        final_ratio = metrics_history["ratio"][-1]
        print(f"Final caption ratio: {final_ratio:.3f}")
        if final_ratio > 1.0:
            print(f"✓ HYPOTHESIS VALIDATED: cap_fine < cap_coarse by {(final_ratio-1)*100:.1f}%")
            print(f"  → Fine path produces better captions than coarse path")
            print(f"  → Autoregressive queries guided by previous frames help!")
        else:
            print(f"✗ Hypothesis not yet validated")
            print(f"  → May need more training or hyperparameter tuning")

    print(f"{'='*70}")

    if HAS_WANDB:
        wandb.finish()

    return global_step, best_ratio, metrics_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_hours", type=float, default=CONFIG["max_hours"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--num_frames", type=int, default=CONFIG["num_frames"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    CONFIG["max_hours"] = args.max_hours
    CONFIG["batch_size"] = args.batch_size
    CONFIG["num_frames"] = args.num_frames
    CONFIG["resume_checkpoint"] = args.resume
    CONFIG["wandb_run_name"] = args.wandb_name

    train(CONFIG)
