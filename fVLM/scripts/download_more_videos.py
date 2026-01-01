"""
Download additional LLaVA-Video subsets for more training data.

Downloads:
- 30_60_s_academic_v0_1 (30-60 second academic videos)
- 0_30_s_youtube_v0_1 (YouTube short videos)
"""

from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import tarfile
from tqdm import tqdm
import argparse


def download_subset(subset_name, max_archives=None):
    """Download a specific subset of LLaVA-Video."""
    repo_id = "lmms-lab/LLaVA-Video-178K"
    repo_type = "dataset"
    local_dir = Path("data/llava_video")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {subset_name}")
    print("="*60)

    # List files in this subset
    all_files = list_repo_files(repo_id, repo_type=repo_type)
    subset_files = [f for f in all_files if f.startswith(subset_name + "/")]

    # Find caption file and video archives
    caption_files = [f for f in subset_files if f.endswith('.json')]
    video_archives = sorted([f for f in subset_files if f.endswith('.tar.gz')])

    if max_archives:
        video_archives = video_archives[:max_archives]

    print(f"Caption files: {len(caption_files)}")
    print(f"Video archives: {len(video_archives)}")

    # Download captions
    for cf in caption_files:
        print(f"Downloading {cf}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=cf,
            repo_type=repo_type,
            local_dir=str(local_dir),
        )
        print(f"  ✓ Done")

    # Download video archives
    downloaded_tars = []
    for i, archive in enumerate(video_archives, 1):
        print(f"Downloading archive {i}/{len(video_archives)}: {archive}...")
        tar_path = hf_hub_download(
            repo_id=repo_id,
            filename=archive,
            repo_type=repo_type,
            local_dir=str(local_dir),
        )
        downloaded_tars.append(tar_path)
        print(f"  ✓ Done")

    # Extract videos
    video_dir = Path("data/videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting videos to {video_dir}...")
    for i, tar_path in enumerate(downloaded_tars, 1):
        print(f"Extracting archive {i}/{len(downloaded_tars)}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=video_dir)

    print(f"✓ {subset_name} complete!")
    return len(downloaded_tars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsets', nargs='+',
                        default=['30_60_s_academic_v0_1', '0_30_s_youtube_v0_1'],
                        help='Subsets to download')
    parser.add_argument('--max-archives', type=int, default=None,
                        help='Max archives per subset (for testing)')
    args = parser.parse_args()

    print("="*60)
    print("Downloading additional LLaVA-Video subsets")
    print("="*60)

    total_archives = 0
    for subset in args.subsets:
        try:
            n = download_subset(subset, args.max_archives)
            total_archives += n
        except Exception as e:
            print(f"Error downloading {subset}: {e}")
            continue

    # Count total videos
    video_dir = Path("data/videos")
    video_count = len(list(video_dir.glob("**/*.mp4")))

    print("\n" + "="*60)
    print("Download complete!")
    print(f"Total videos now available: {video_count}")
    print("="*60)


if __name__ == "__main__":
    main()
