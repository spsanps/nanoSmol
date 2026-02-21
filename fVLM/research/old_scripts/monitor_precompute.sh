#!/bin/bash
# Monitor precompute progress and disk usage

while true; do
    clear
    echo "=========================================="
    echo "PRECOMPUTE MONITOR - $(date)"
    echo "=========================================="
    
    # Sample count
    COUNT=$(ls /mnt/d/projects/fVLM/data/precomputed/latents/ 2>/dev/null | wc -l)
    echo "Samples completed: $COUNT / 100,000"
    
    # Disk usage
    USAGE=$(du -sh /mnt/d/projects/fVLM/data/precomputed/ 2>/dev/null | cut -f1)
    echo "Current disk usage: $USAGE"
    
    # Projected
    if [ $COUNT -gt 0 ]; then
        BYTES=$(du -sb /mnt/d/projects/fVLM/data/precomputed/ 2>/dev/null | cut -f1)
        PER_SAMPLE=$((BYTES / COUNT))
        PROJECTED_GB=$(echo "scale=1; $PER_SAMPLE * 100000 / 1024 / 1024 / 1024" | bc)
        echo "Projected total: ${PROJECTED_GB} GB"
    fi
    
    # Available space
    AVAIL=$(df -h /mnt/d | tail -1 | awk '{print $4}')
    echo "Available on D: $AVAIL"
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 60
done
