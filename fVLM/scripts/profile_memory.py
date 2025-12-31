"""
Memory profiling for Foveated VLM.

CRITICAL: Test memory usage BEFORE full training to avoid OOM.
Target: < 18GB on RTX 4090 (leave 2GB headroom)
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.foveated_vlm import FoveatedVideoModel
from src.data.dataset import create_dataloader


def profile_memory(batch_size=2, num_frames=8):
    """Profile memory usage for one training step."""

    print("=" * 70)
    print("Memory Profiling for Foveated VLM")
    print("=" * 70)

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    # Create model
    print("\n📦 Loading model...")
    model = FoveatedVideoModel(
        dino_model="facebook/dinov2-small",
        llm_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        lambda_coarse=1.0,
    ).to(device)

    model_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Model memory: {model_mem:.2f} GB")

    # Create DataLoader
    print("\n📦 Creating DataLoader...")
    try:
        dataloader = create_dataloader(
            video_dir="data/videos",
            latent_dir="data/latents",
            batch_size=batch_size,
            num_workers=0,  # Use 0 for profiling
            shuffle=False,
            num_frames=num_frames,
        )
        print(f"   ✓ DataLoader created (batch_size={batch_size}, num_frames={num_frames})")
    except ValueError as e:
        print(f"   ✗ Error: {e}")
        print("\n   Waiting for VAE preprocessing to complete...")
        return

    # Load one batch
    print("\n🔄 Loading test batch...")
    batch = next(iter(dataloader))
    frames = batch['frames'].to(device)
    vae_latents = batch['vae_latents'].to(device)

    data_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Data memory: {data_mem - model_mem:.2f} GB")

    # Forward pass
    print("\n🔄 Forward pass (bf16)...")
    text_embeds = model.get_empty_text_embeds(batch_size).to(device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, loss_fine, loss_coarse = model(text_embeds, frames, vae_latents)

    forward_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Forward memory: {forward_mem:.2f} GB")
    print(f"   Loss: {loss.item():.4f} (fine: {loss_fine.item():.4f}, coarse: {loss_coarse.item():.4f})")

    # Backward pass
    print("\n🔄 Backward pass...")
    loss.backward()

    backward_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Backward memory (peak): {backward_mem:.2f} GB")

    # Summary
    print("\n" + "=" * 70)
    print("Memory Summary")
    print("=" * 70)
    print(f"Model:            {model_mem:.2f} GB")
    print(f"Data:             {data_mem - model_mem:.2f} GB")
    print(f"Forward peak:     {forward_mem:.2f} GB")
    print(f"Backward peak:    {backward_mem:.2f} GB")
    print(f"\nTotal peak:       {backward_mem:.2f} GB")
    print(f"Available (4090): ~20 GB")
    print(f"Headroom:         {20 - backward_mem:.2f} GB")

    # Check if within limits
    if backward_mem > 18:
        print("\n⚠️  WARNING: Memory usage > 18GB!")
        print("   Recommendations:")
        print("   1. Reduce batch_size to 1")
        print("   2. Reduce num_frames to 4")
        print("   3. Enable gradient checkpointing")
    elif backward_mem > 15:
        print("\n⚠️  CAUTION: Memory usage is high (>15GB)")
        print("   Consider reducing batch_size if training is unstable")
    else:
        print("\n✓ Memory usage is safe (<15GB)")

    print("=" * 70)

    # Test gradient accumulation scenario
    print("\n" + "=" * 70)
    print("Gradient Accumulation Simulation (4 steps)")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()

    for i in range(4):
        batch = next(iter(dataloader))
        frames = batch['frames'].to(device)
        vae_latents = batch['vae_latents'].to(device)
        text_embeds = model.get_empty_text_embeds(batch_size).to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _, _ = model(text_embeds, frames, vae_latents)
            loss = loss / 4  # Scale for accumulation

        loss.backward()

        if i == 0:
            first_step_mem = torch.cuda.max_memory_allocated() / 1e9

    accum_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"First step:       {first_step_mem:.2f} GB")
    print(f"After 4 steps:    {accum_mem:.2f} GB")
    print(f"Memory growth:    {accum_mem - first_step_mem:.2f} GB")

    if accum_mem - first_step_mem > 0.5:
        print("\n⚠️  WARNING: Significant memory growth during accumulation!")
        print("   Possible memory leak - investigate")
    else:
        print("\n✓ Gradient accumulation memory stable")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    profile_memory(batch_size=args.batch_size, num_frames=args.num_frames)
