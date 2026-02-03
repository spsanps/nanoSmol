#!/bin/bash
# Wait for sharding to finish, then start training
set -e

SHARD_PID=$1
SHARD_DIR="/mnt/d/projects/fVLM/data/frames_latents_sharded"

echo "Waiting for sharding process (PID $SHARD_PID) to finish..."

# Wait for the sharding process to finish
while kill -0 "$SHARD_PID" 2>/dev/null; do
    SHARD_COUNT=$(ls "$SHARD_DIR"/shard_*.pt 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] Sharding in progress... $SHARD_COUNT shards so far"
    sleep 120
done

SHARD_COUNT=$(ls "$SHARD_DIR"/shard_*.pt 2>/dev/null | wc -l)
echo ""
echo "Sharding complete! $SHARD_COUNT shards created."
echo "Starting training..."
echo ""

cd /mnt/c/Users/sanps/Desktop/Projects/dino/nanoSmolLM
python fVLM/scripts/train_joint_multifine_precomputed.py
