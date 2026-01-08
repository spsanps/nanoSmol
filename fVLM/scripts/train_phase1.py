"""
Training script for Phase 1: Self-Supervised Next-Frame Prediction

Trains the Foveated VLM on video data without text supervision.
Core hypothesis: loss_fine < loss_coarse (dynamic attention helps!)
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import sys
import time
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.foveated_vlm import FoveatedVideoModel
from src.data.dataset import create_dataloader


def setup_wandb(config):
    """Initialize Weights & Biases logging."""
    if config['logging']['wandb_project']:
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            config=config,
            name=f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    else:
        print("⚠️  W&B logging disabled (set wandb_project in config)")


def train_step(model, batch, optimizer, scaler, config, step):
    """Single training step with gradient accumulation."""

    frames = batch['frames'].cuda()           # [B, T, 3, 256, 256]
    vae_latents = batch['vae_latents'].cuda() # [B, T, 4, 32, 32]

    # Phase 1: no text, use empty embeddings
    text_embeds = model.get_empty_text_embeds(frames.shape[0]).cuda()

    # Forward pass with mixed precision
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, loss_fine, loss_coarse = model(text_embeds, frames, vae_latents)
        loss = loss / config['training']['grad_accum']  # Scale for accumulation

    # Backward pass
    scaler.scale(loss).backward()

    # Step every grad_accum steps
    if (step + 1) % config['training']['grad_accum'] == 0:
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config['training']['grad_clip']
        )

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return {
        'loss': loss.item() * config['training']['grad_accum'],
        'loss_fine': loss_fine.item(),
        'loss_coarse': loss_coarse.item(),
        'loss_ratio': loss_coarse.item() / (loss_fine.item() + 1e-8),
    }


def train(config_path):
    """Main training loop."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Foveated VLM Training - Phase 1")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"Output dir: {config['logging']['output_dir']}")

    # Create output directories
    output_dir = Path(config['logging']['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Setup W&B
    setup_wandb(config)

    # Create DataLoader
    print("\n📦 Creating DataLoader...", flush=True)
    import sys
    sys.stdout.flush()
    # Try num_workers=1 for better throughput (WSL compatibility)
    # Falls back to 0 if multiprocessing fails
    # Get frames_dir (optional, for pre-decoded frames - 20-100x faster)
    frames_dir = config['data'].get('frames_dir', None)

    try:
        dataloader = create_dataloader(
            video_dir=config['data']['video_dir'],
            latent_dir=config['data']['latent_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=2,  # More workers for I/O bound loading
            shuffle=True,
            num_frames=config['data']['num_frames'],
            frame_size=config['data']['frame_size'],
            frames_dir=frames_dir,
        )
        print(f"   Using num_workers=2 with prefetching")
    except Exception as e:
        print(f"   Multiprocessing failed ({e}), falling back to num_workers=0")
        dataloader = create_dataloader(
            video_dir=config['data']['video_dir'],
            latent_dir=config['data']['latent_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=0,
            shuffle=True,
            num_frames=config['data']['num_frames'],
            frame_size=config['data']['frame_size'],
            frames_dir=frames_dir,
        )
    print(f"   Dataset size: {len(dataloader.dataset)}")
    print(f"   Batches per epoch: {len(dataloader)}")
    print(f"   Effective batch size: {config['training']['batch_size'] * config['training']['grad_accum']}")

    # Create model
    print("\n📦 Creating model...", flush=True)
    sys.stdout.flush()
    model = FoveatedVideoModel(
        dino_model=config['model']['dino_model'],
        llm_model=config['model']['llm_model'],
        dino_dim=config['model']['dino_dim'],
        llm_dim=config['model']['llm_dim'],
        query_dim=config['model']['query_dim'],
        lambda_coarse=config['model']['lambda_coarse'],
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   Total parameters: {total_params:.1f}M")
    print(f"   Trainable parameters: {trainable_params:.1f}M")

    # Optimizer
    print("\n📦 Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Learning rate scheduler (linear warmup)
    def lr_lambda(step):
        if step < config['training']['warmup_steps']:
            return step / config['training']['warmup_steps']
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Gradient scaler for mixed precision
    scaler = GradScaler()

    # Training loop
    print("\n🚀 Starting training...")
    print(f"   Max steps: {config['training']['max_steps']}")
    max_hours = config['training'].get('max_hours', None)
    if max_hours:
        print(f"   Max hours: {max_hours}")
    print(f"   Logging every: {config['logging']['log_every']} steps")
    print(f"   Saving every: {config['logging']['save_every']} steps")
    print("=" * 70)

    model.train()
    global_step = 0
    start_time = time.time()
    running_metrics = {'loss': 0, 'loss_fine': 0, 'loss_coarse': 0, 'loss_ratio': 0}

    pbar = tqdm(total=config['training']['max_steps'], desc="Training")

    while global_step < config['training']['max_steps']:
        for batch in dataloader:
            # Training step
            metrics = train_step(model, batch, optimizer, scaler, config, global_step)

            # Update running metrics
            for k, v in metrics.items():
                running_metrics[k] += v

            # Log
            if (global_step + 1) % config['logging']['log_every'] == 0:
                avg_metrics = {k: v / config['logging']['log_every']
                               for k, v in running_metrics.items()}

                log_dict = {
                    'step': global_step,
                    'lr': scheduler.get_last_lr()[0],
                    'loss/total': avg_metrics['loss'],
                    'loss/fine': avg_metrics['loss_fine'],
                    'loss/coarse': avg_metrics['loss_coarse'],
                    'loss/ratio': avg_metrics['loss_ratio'],
                    'memory/allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                }

                if wandb.run:
                    wandb.log(log_dict, step=global_step)

                pbar.set_postfix({
                    'loss': f"{avg_metrics['loss']:.4f}",
                    'fine': f"{avg_metrics['loss_fine']:.4f}",
                    'coarse': f"{avg_metrics['loss_coarse']:.4f}",
                    'ratio': f"{avg_metrics['loss_ratio']:.3f}",
                })

                # Reset running metrics
                running_metrics = {k: 0 for k in running_metrics}

            # Save checkpoint
            if (global_step + 1) % config['logging']['save_every'] == 0:
                checkpoint_path = output_dir / 'checkpoints' / f'step_{global_step + 1}.pt'
                torch.save({
                    'step': global_step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config': config,
                }, checkpoint_path)
                print(f"\n💾 Saved checkpoint: {checkpoint_path}")

            # Update step
            global_step += 1
            pbar.update(1)
            scheduler.step()

            if global_step >= config['training']['max_steps']:
                break

            # Check time limit
            if max_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= max_hours:
                    print(f"\n⏰ Time limit reached ({max_hours}h). Stopping training.")
                    break

        # Break outer loop too if time/steps reached
        if max_hours is not None:
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= max_hours:
                break

    pbar.close()

    # Final checkpoint
    final_path = output_dir / 'checkpoints' / 'final.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"   Final checkpoint: {final_path}")
    print("=" * 70)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/phase1.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    train(args.config)
