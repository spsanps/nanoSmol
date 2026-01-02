#!/bin/bash

# Download LLaVA-Video-178K dataset (0-30s academic subset)
# Start with videos_1.tar.gz (~5-6GB)

set -e

echo "============================================================"
echo "Downloading LLaVA-Video-178K Dataset"
echo "============================================================"

# Load HF token
source .env 2>/dev/null || true

echo ""
echo "📦 Downloading metadata and first video batch..."
echo "   Dataset: lmms-lab/LLaVA-Video-178K"
echo "   Subset: 0_30_s_academic_v0_1"
echo ""

huggingface-cli download lmms-lab/LLaVA-Video-178K \
  --repo-type dataset \
  --include "0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json" \
  --include "0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_1.tar.gz" \
  --local-dir data/llava_video \
  --local-dir-use-symlinks False

echo ""
echo "📂 Extracting videos..."
tar -xzf data/llava_video/0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_1.tar.gz \
    -C data/videos/

echo ""
echo "✓ Dataset downloaded and extracted successfully!"
echo "   Videos location: data/videos/"
echo "   Metadata: data/llava_video/0_30_s_academic_v0_1/"
echo ""
echo "Note: You can download more batches later by running:"
echo "  huggingface-cli download ... --include '*_videos_2.tar.gz'"
echo "============================================================"
