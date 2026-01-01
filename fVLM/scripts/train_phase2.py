"""
Phase 2 Training: Text-Conditioned Foveated VLM

Key improvements over Phase 1:
- Text conditioning for guided attention
- Action-rich videos (LLaVA-Video-178K)
- More frames (32 vs 16) for longer dynamics
- Consistent temporal sampling
- Streaming mode: downloads and processes videos on-the-fly
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.foveated_vlm import FoveatedVideoModel
from src.data.llava_video_dataset import LLaVAVideoDataset, collate_fn


def train_phase2(config_path):
    """Main training loop for Phase 2."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Foveated VLM - Phase 2: Text-Conditioned Training")
    print("=" * 70)
    print(f"Frames: {config['data']['num_frames']}")
    print(f"Text conditioning: {config['data']['use_text']}")
    print(f"Dataset: {config['data']['dataset']}")

    # Output dir
    output_dir = Path(config['logging']['output_dir'])
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # W&B
    if config['logging'].get('wandb_project'):
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=config['logging'].get('run_name', f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        )

    # Create model
    print("\nCreating model...")
    model_cfg = config['model']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FoveatedVideoModel(
        dino_model=model_cfg['dino_model'],
        llm_model=model_cfg['llm_model'],
        dino_dim=model_cfg['dino_dim'],
        llm_dim=model_cfg['llm_dim'],
        query_dim=model_cfg['query_dim'],
        lambda_coarse=model_cfg['lambda_coarse'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # Load Phase 1 checkpoint if exists (warm start)
    phase1_ckpt = Path('outputs/streaming/checkpoints/final.pt')
    if phase1_ckpt.exists():
        print(f"\nLoading Phase 1 checkpoint: {phase1_ckpt}")
        ckpt = torch.load(phase1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("  ✓ Warm-started from Phase 1")
    else:
        print("\nStarting from scratch (no Phase 1 checkpoint)")

    # Optimizer
    train_cfg = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )

    # LR scheduler
    def lr_lambda(step):
        warmup = train_cfg.get('warmup_steps', 500)
        if step < warmup:
            return step / warmup
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()

    # Create dataset
    print("\nLoading dataset...")
    data_cfg = config['data']

    # Check if data is extracted
    video_dir = Path('data/videos')
    latent_dir = Path('data/latents_phase2')
    caption_json = Path('data/llava_video/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json')

    if not caption_json.exists():
        print(f"ERROR: Caption file not found: {caption_json}")
        print("Please run the download script first")
        return

    # Check for streaming mode (no precomputed latents)
    streaming_mode = data_cfg.get('streaming_vae', False) or not latent_dir.exists()

    if streaming_mode:
        print("Using streaming mode (VAE in training loop, workers for video loading)")
        # Load VAE - will be used in training loop, NOT in dataset
        print("Loading VAE encoder...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16  # FP16 for speed
        ).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        latent_dir_arg = None
        vae_for_dataset = None  # Don't pass VAE to dataset - use workers instead
    else:
        print("Using precomputed latents")
        vae = None
        vae_for_dataset = None
        latent_dir_arg = latent_dir

    # Extract videos if not present (from all available archives)
    if not video_dir.exists() or len(list(video_dir.glob("**/*.mp4"))) < 100:
        print(f"Extracting videos...")
        video_dir.mkdir(parents=True, exist_ok=True)
        import tarfile
        archive_dir = Path('data/llava_video/0_30_s_academic_v0_1')
        for i in range(1, 9):  # videos_1 through videos_8
            tar_path = archive_dir / f"0_30_s_academic_v0_1_videos_{i}.tar.gz"
            if tar_path.exists():
                print(f"  Extracting archive {i}/8...")
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=video_dir)
        print(f"  ✓ Extracted to {video_dir}")

    dataset = LLaVAVideoDataset(
        video_dir=video_dir,
        caption_json=caption_json,
        num_frames=data_cfg['num_frames'],
        frame_size=data_cfg['frame_size'],
        max_text_tokens=data_cfg['max_text_tokens'],
        llm_model=model_cfg['llm_model'],
        min_duration=data_cfg.get('min_duration', 5),
        max_duration=data_cfg.get('max_duration', 30),
        latent_dir=latent_dir_arg,
        vae_model=vae_for_dataset,  # None for streaming - VAE runs in training loop
        vae_device=device,
    )

    if len(dataset) == 0:
        print("ERROR: No videos found!")
        print("Please run: python scripts/download_llava_video.py")
        return

    # Single-process loading to avoid memory pressure from workers
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print("DataLoader: single-process (stable memory)")

    # Training
    max_steps = train_cfg['max_steps']
    grad_accum = train_cfg.get('grad_accum', 1)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    log_every = config['logging'].get('log_every', 100)
    save_every = config['logging'].get('save_every', 10000)

    print(f"\nTraining for {max_steps} steps")
    print(f"Batch size: {train_cfg['batch_size']} x {grad_accum} = {train_cfg['batch_size'] * grad_accum}")
    print(f"Dataset size: {len(dataset)} videos")
    print("=" * 70)

    model.train()
    global_step = 0
    running_loss = 0
    running_fine = 0
    running_coarse = 0

    pbar = tqdm(total=max_steps, desc="Training")

    def compute_vae_latents_batch(frames_raw, vae, device):
        """Compute VAE latents for a batch of raw frames."""
        # frames_raw: [B, T, 3, H, W] uint8
        B, T, C, H, W = frames_raw.shape
        frames_flat = frames_raw.view(B * T, C, H, W).float().to(device)
        frames_vae = frames_flat / 255.0 * 2 - 1  # Normalize to [-1, 1]

        with torch.no_grad():
            # Process in chunks to avoid OOM
            chunk_size = 16  # 16 frames at a time
            latent_chunks = []
            for i in range(0, frames_vae.shape[0], chunk_size):
                chunk = frames_vae[i:i+chunk_size].half()
                latent = vae.encode(chunk).latent_dist.sample() * 0.18215
                latent_chunks.append(latent)
            latents = torch.cat(latent_chunks, dim=0)

        # Reshape back: [B, T, 4, 32, 32]
        latents = latents.view(B, T, 4, latents.shape[-2], latents.shape[-1])
        return latents.float()

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            frames = batch['frames'].to(device)
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)

            # Compute VAE latents in training loop (streaming mode)
            if streaming_mode:
                frames_raw = batch['frames_raw']  # [B, T, 3, H, W] uint8
                latents = compute_vae_latents_batch(frames_raw, vae, device)
            else:
                latents = batch['vae_latents'].to(device)

            # Get text embeddings
            text_embeds = model.get_text_embeds(text_ids, text_mask)

            # Forward
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, loss_fine, loss_coarse = model(text_embeds, frames, latents)
                loss = loss / grad_accum

            # Backward
            scaler.scale(loss).backward()

            running_loss += loss.item() * grad_accum
            running_fine += loss_fine.item()
            running_coarse += loss_coarse.item()

            # Optimizer step
            if (global_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            pbar.update(1)

            # Log
            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                avg_fine = running_fine / log_every
                avg_coarse = running_coarse / log_every
                ratio = avg_coarse / (avg_fine + 1e-8)

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'fine': f'{avg_fine:.4f}',
                    'coarse': f'{avg_coarse:.4f}',
                    'ratio': f'{ratio:.3f}',
                })

                if wandb.run:
                    wandb.log({
                        'loss': avg_loss,
                        'loss_fine': avg_fine,
                        'loss_coarse': avg_coarse,
                        'ratio': ratio,
                        'lr': scheduler.get_last_lr()[0],
                        'step': global_step,
                    }, step=global_step)

                running_loss = 0
                running_fine = 0
                running_coarse = 0

            # Save
            if global_step % save_every == 0:
                path = output_dir / 'checkpoints' / f'step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, path)
                print(f"\nSaved: {path}")

    pbar.close()

    # Final save
    final_path = output_dir / 'checkpoints' / 'final.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)

    print("\n" + "=" * 70)
    print("Phase 2 Training complete!")
    print(f"Checkpoint: {final_path}")
    print("=" * 70)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/phase2.yaml')
    args = parser.parse_args()
    train_phase2(args.config)
