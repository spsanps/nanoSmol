"""
Download LLaVA-Video-178K dataset for Phase 2.

Downloads the 0-30s academic subset with captions.
Full dataset has 8 video archives (~43GB total, ~12K videos).
"""

from huggingface_hub import hf_hub_download
from pathlib import Path
import tarfile
from tqdm import tqdm
import argparse

def download_llava_video(num_archives=8, extract=True):
    """
    Download LLaVA-Video dataset.

    Args:
        num_archives: Number of video archives to download (1-8). Default all 8.
        extract: Whether to extract videos after download.
    """
    repo_id = "lmms-lab/LLaVA-Video-178K"
    repo_type = "dataset"
    local_dir = Path("data/llava_video")
    local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading LLaVA-Video-178K dataset")
    print(f"Archives to download: {num_archives}/8 (~{num_archives * 5.3:.1f}GB)")
    print("=" * 60)

    # Download caption file
    print("\n1. Downloading captions...")
    caption_file = hf_hub_download(
        repo_id=repo_id,
        filename="0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json",
        repo_type=repo_type,
        local_dir=str(local_dir),
    )
    print(f"   ✓ Captions ready")

    # Download video archives
    print(f"\n2. Downloading {num_archives} video archives...")
    downloaded_tars = []

    for i in range(1, num_archives + 1):
        filename = f"0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_{i}.tar.gz"
        size = "489MB" if i == 8 else "~5.3GB"
        print(f"   Downloading archive {i}/{num_archives} ({size})...")

        video_tar = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_dir),
        )
        downloaded_tars.append(video_tar)
        print(f"   ✓ Archive {i} complete")

    # Extract videos
    if extract:
        print(f"\n3. Extracting videos...")
        video_dir = Path("data/videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        for i, tar_path in enumerate(downloaded_tars, 1):
            print(f"   Extracting archive {i}/{len(downloaded_tars)}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc=f"Archive {i}", leave=False):
                    tar.extract(member, path=video_dir)

        # Count extracted videos
        video_count = len(list(video_dir.glob("**/*.mp4")))
        print(f"   ✓ Extracted {video_count} videos")

    print(f"\n" + "=" * 60)
    print("✓ Dataset ready!")
    print(f"  Videos: data/videos/")
    print(f"  Captions: {caption_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-archives', type=int, default=8,
                        help='Number of archives to download (1-8)')
    parser.add_argument('--no-extract', action='store_true',
                        help='Skip extraction after download')
    args = parser.parse_args()

    download_llava_video(
        num_archives=args.num_archives,
        extract=not args.no_extract
    )
