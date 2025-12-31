"""
Download pretrained models for Foveated VLM.

Downloads:
- DINOv2 ViT-S/16 (vision encoder)
- SmolLM2-135M-Instruct (core LLM)
- Stable Diffusion VAE (reconstruction target)
"""

import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load HF token
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

models = [
    ("facebook/dinov2-small", "DINOv2 ViT-S/16"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M"),
    ("stabilityai/sd-vae-ft-mse", "Stable Diffusion VAE"),
]

print("=" * 60)
print("Downloading Pretrained Models")
print("=" * 60)

for repo_id, name in models:
    print(f"\n📦 Downloading {name}...")
    print(f"   Repository: {repo_id}")

    try:
        snapshot_download(
            repo_id=repo_id,
            token=hf_token,
            resume_download=True,
        )
        print(f"   ✓ {name} downloaded successfully")
    except Exception as e:
        print(f"   ✗ Error downloading {name}: {e}")
        raise

print("\n" + "=" * 60)
print("✓ All models downloaded successfully!")
print("=" * 60)
