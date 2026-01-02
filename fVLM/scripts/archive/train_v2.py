"""
Training script v2: Improved training with validation and early stopping

Key improvements over train_phase1.py:
- Train/val split (90/10)
- Validation loss tracking
- Early stopping based on val loss
- Best model checkpointing
- Resume from checkpoint
- Configurable epochs vs steps
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.foveated_vlm import FoveatedVideoModel
from src.data.dataset import FoveatedVideoDataset


def create_train_val_dataloaders(config):
    """Create train and validation dataloaders with proper split."""

    # Determine data directories
    data_config = config['data']

    # Support both old format (video_dir) and new format (data_dir)
    if 'data_dir' in data_config:
        data_dir = Path(data_config['data_dir'])
        frames_dir = data_dir / 'frames'
        latent_dir = data_dir / 'latents'
        # For WebVid, we don't need video_dir since we have frames
        video_dir = data_dir / 'videos'
        if not video_dir.exists():
            video_dir = frames_dir  # Use frames dir as dummy
    else:
        video_dir = data_config['video_dir']
        latent_dir = data_config['latent_dir']
        frames_dir = data_config.get('frames_dir')

    # Create dataset
    dataset = FoveatedVideoDataset(
        video_dir=str(video_dir),
        latent_dir=str(latent_dir),
        num_frames=data_config['num_frames'],
        frame_size=data_config['frame_size'],
        frames_dir=str(frames_dir) if frames_dir else None,
    )

    # Split into train/val
    val_ratio = config['training'].get('val_ratio', 0.1)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset split: {train_size} train, {val_size} val")

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def setup_wandb(config, run_name=None):
    """Initialize Weights & Biases logging."""
    if config['logging']['wandb_project']:
        if run_name is None:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging'].get('wandb_entity'),
            config=config,
            name=run_name,
        )
    else:
        print("W&B logging disabled")


def train_step(model, batch, optimizer, scaler, grad_accum):
    """Single training step."""
    frames = batch['frames'].cuda()
    vae_latents = batch['vae_latents'].cuda()
    text_embeds = model.get_empty_text_embeds(frames.shape[0]).cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, loss_fine, loss_coarse = model(text_embeds, frames, vae_latents)
        loss = loss / grad_accum

    scaler.scale(loss).backward()

    return {
        'loss': loss.item() * grad_accum,
        'loss_fine': loss_fine.item(),
        'loss_coarse': loss_coarse.item(),
    }


@torch.no_grad()
def validate(model, val_loader):
    """Run validation and return metrics."""
    model.eval()

    total_loss = 0
    total_fine = 0
    total_coarse = 0
    num_batches = 0

    for batch in val_loader:
        frames = batch['frames'].cuda()
        vae_latents = batch['vae_latents'].cuda()
        text_embeds = model.get_empty_text_embeds(frames.shape[0]).cuda()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, loss_fine, loss_coarse = model(text_embeds, frames, vae_latents)

        total_loss += loss.item()
        total_fine += loss_fine.item()
        total_coarse += loss_coarse.item()
        num_batches += 1

    model.train()

    return {
        'val_loss': total_loss / num_batches,
        'val_fine': total_fine / num_batches,
        'val_coarse': total_coarse / num_batches,
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, val_loss, output_dir, name='checkpoint'):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
    }
    path = output_dir / f'{name}.pt'
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('step', 0), checkpoint.get('val_loss', float('inf'))


def train(config_path, resume_from=None):
    """Main training loop."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Foveated VLM Training v2")
    print("=" * 70)
    print(f"Config: {config_path}")

    # Create output directory
    output_dir = Path(config['logging']['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # Setup W&B
    setup_wandb(config)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_train_val_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model_config = config['model']
    model = FoveatedVideoModel(
        dino_model=model_config['dino_model'],
        llm_model=model_config['llm_model'],
        dino_dim=model_config['dino_dim'],
        llm_dim=model_config['llm_dim'],
        query_dim=model_config['query_dim'],
        lambda_coarse=model_config['lambda_coarse'],
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.1f}M")

    # Optimizer
    train_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
    )

    # Scheduler
    def lr_lambda(step):
        warmup = train_config.get('warmup_steps', 500)
        if step < warmup:
            return step / warmup
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if resume_from:
        print(f"\nResuming from {resume_from}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler
        )
        print(f"Resumed at epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")

    # Training config
    num_epochs = train_config.get('num_epochs', 10)
    grad_accum = train_config.get('grad_accum', 1)
    grad_clip = train_config.get('grad_clip', 1.0)
    patience = train_config.get('patience', 5)
    log_every = config['logging'].get('log_every', 100)
    val_every = config['logging'].get('val_every', 1)  # Validate every N epochs

    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_config['batch_size']} x {grad_accum} = {train_config['batch_size'] * grad_accum}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Patience: {patience} epochs")

    # Early stopping
    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_fine = 0
        epoch_coarse = 0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Training step
            metrics = train_step(model, batch, optimizer, scaler, grad_accum)

            # Accumulate epoch metrics
            epoch_loss += metrics['loss']
            epoch_fine += metrics['loss_fine']
            epoch_coarse += metrics['loss_coarse']
            num_steps += 1

            # Optimizer step every grad_accum
            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'fine': f"{metrics['loss_fine']:.4f}",
                'coarse': f"{metrics['loss_coarse']:.4f}",
            })

            # Log to W&B
            if global_step % log_every == 0 and wandb.run:
                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/fine': metrics['loss_fine'],
                    'train/coarse': metrics['loss_coarse'],
                    'train/ratio': metrics['loss_coarse'] / (metrics['loss_fine'] + 1e-8),
                    'lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                }, step=global_step)

        # Epoch metrics
        avg_loss = epoch_loss / num_steps
        avg_fine = epoch_fine / num_steps
        avg_coarse = epoch_coarse / num_steps

        print(f"\nEpoch {epoch+1} train: loss={avg_loss:.4f}, fine={avg_fine:.4f}, coarse={avg_coarse:.4f}")

        # Validation
        if (epoch + 1) % val_every == 0:
            print("Running validation...")
            val_metrics = validate(model, val_loader)
            val_loss = val_metrics['val_loss']

            print(f"Epoch {epoch+1} val: loss={val_loss:.4f}, fine={val_metrics['val_fine']:.4f}, coarse={val_metrics['val_coarse']:.4f}")

            if wandb.run:
                wandb.log({
                    'val/loss': val_loss,
                    'val/fine': val_metrics['val_fine'],
                    'val/coarse': val_metrics['val_coarse'],
                    'val/ratio': val_metrics['val_coarse'] / (val_metrics['val_fine'] + 1e-8),
                    'epoch': epoch,
                }, step=global_step)

            # Best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, val_loss, output_dir / 'checkpoints', 'best'
                )
                print(f"New best model! val_loss={val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs")

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break

        # Regular checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, global_step, val_loss if 'val_loss' in dir() else avg_loss,
            output_dir / 'checkpoints', 'latest'
        )

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch, global_step, best_val_loss,
        output_dir / 'checkpoints', 'final'
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print("=" * 70)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_v2.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train(args.config, args.resume)
