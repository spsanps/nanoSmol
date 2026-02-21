# Streaming Training Design

## Problem
Current approach requires downloading all videos upfront:
- 159GB compressed archives
- 81GB extracted videos
- **Total: 240GB** (unsustainable for most systems)

## Proposed Solution: On-Demand Streaming

### Core Idea
Download videos in small batches, process them, then delete immediately. Keep only:
- Model checkpoints (~600MB per checkpoint)
- Current batch of videos (~2-5GB at a time)

### Architecture

```
Training Loop
    ↓
Streaming Dataset (manages queue)
    ↓
Download Manager (background thread)
    ↓
[Download 100 videos] → [Process in training] → [Delete]
    ↓
Repeat until all videos processed
```

### Implementation Components

#### 1. Streaming Dataset Class
```python
class StreamingLLaVAVideoDataset:
    def __init__(self, repo_id, subset_name, cache_size=100):
        self.repo_id = repo_id
        self.subset_name = subset_name
        self.cache_size = cache_size

        # Get list of all videos from metadata (JSON only, ~1MB)
        self.video_list = self._load_metadata()

        # Local cache for current batch
        self.cache_dir = Path("data/stream_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Track which videos are downloaded
        self.downloaded = set()
        self.current_batch_idx = 0

    def _download_batch(self, start_idx, end_idx):
        """Download a batch of videos on-demand."""
        # Download only the videos needed for this batch
        # Use hf_hub_download with specific file paths
        pass

    def __getitem__(self, idx):
        # If video not in cache, download its batch
        if idx not in self.downloaded:
            batch_start = (idx // self.cache_size) * self.cache_size
            batch_end = min(batch_start + self.cache_size, len(self.video_list))

            # Delete previous batch
            self._cleanup_old_batch()

            # Download new batch
            self._download_batch(batch_start, batch_end)

        # Load video from cache
        return self._load_video(idx)

    def _cleanup_old_batch(self):
        """Delete videos from previous batch to free space."""
        for f in self.cache_dir.glob("**/*.mp4"):
            f.unlink()
        self.downloaded.clear()
```

#### 2. Smart Batching Strategy
- **Cache size**: 100-200 videos (~2-5GB)
- **Pre-fetch**: Download next batch while training current batch
- **Sequential access**: Shuffle once at epoch start, then stream sequentially

#### 3. Multi-Epoch Handling
```python
# For multi-epoch training, re-download videos
# Trade-off: download time vs storage
for epoch in range(num_epochs):
    dataset.reset()  # Clear cache, start fresh
    # Re-download videos as needed during epoch
```

### Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Metadata (JSON) | ~50MB | All captions, downloaded once |
| Video cache | 2-5GB | Current batch only |
| Checkpoints | ~1GB | Per checkpoint |
| **Total** | **<10GB** | vs 240GB currently |

### Implementation Steps

1. **Modify dataset class** to support streaming mode
2. **Implement batch download** using HuggingFace Hub API
3. **Add cache management** (download/delete cycle)
4. **Test with small subset** (100 videos) first
5. **Add progress tracking** to resume interrupted downloads

### Trade-offs

**Pros:**
- 24x less disk space (10GB vs 240GB)
- Can train on full datasets without storage limits
- Easy to scale to larger datasets

**Cons:**
- Training speed depends on download speed
- Multi-epoch training requires re-downloading
- Need stable internet connection

**Mitigations:**
- Pre-fetch next batch during training (overlap download + training)
- For multi-epoch, accept re-download cost (still cheaper than 240GB storage)
- Add retry logic for network failures

### Expected Performance

Assuming:
- Download speed: 10MB/s
- Batch size: 100 videos (~2GB)
- Training speed: ~1.5s/step, 4 samples/step

```
Download time per batch: 2GB / 10MB/s = 200s = 3.3 min
Training time per batch: 100 samples / 4 per step * 1.5s = 37.5s

With pre-fetching: Download happens in background
→ No slowdown if download finishes before training completes batch
```

### Code Skeleton

```python
# configs/streaming.yaml
data:
  streaming_mode: true
  cache_size: 100  # videos per batch
  prefetch: true   # download next batch in background

# src/data/streaming_dataset.py
class StreamingLLaVAVideo(Dataset):
    """Download videos on-demand, delete after use."""

    def __init__(self, repo_id, subset, cache_size=100):
        # Load metadata only (JSON files, ~50MB)
        self.metadata = self._load_metadata()
        self.cache_size = cache_size
        self.cache = VideoCache(max_size=cache_size)

    def __getitem__(self, idx):
        # Check cache first
        if idx not in self.cache:
            # Download batch containing this idx
            self.cache.load_batch(idx)

        return self.cache.get(idx)

class VideoCache:
    """Manages video download/delete cycle."""

    def load_batch(self, idx):
        # Delete old batch
        self.clear()

        # Download new batch
        batch_indices = self._get_batch_indices(idx)
        self._download_videos(batch_indices)

    def clear(self):
        # Delete all cached videos
        for f in self.cache_dir.glob("*.mp4"):
            f.unlink()
```

### Alternative: Partial Streaming (Hybrid)

If re-downloading for multi-epoch is too slow:
1. Download 1/3 of dataset (80GB)
2. Train for 3 epochs on this subset
3. Delete and download next 1/3
4. Effectively get 3x more data diversity without 3x storage

```python
# Train on subset 1 (3 epochs)
dataset = StreamingDataset(subset="0_30s_academic_v0_1_part1")
train(dataset, epochs=3)
dataset.cleanup()

# Train on subset 2 (3 epochs)
dataset = StreamingDataset(subset="0_30s_academic_v0_1_part2")
train(dataset, epochs=3)
```

## Recommendation

For next training run:
1. **Start with small test**: Implement streaming for 1,000 videos, verify it works
2. **Scale up**: Full dataset with cache_size=100
3. **Monitor**: Track download vs training time ratio
4. **Optimize**: Adjust cache_size based on network speed

Target: <10GB total storage while training on full LLaVA-Video-178K dataset.
