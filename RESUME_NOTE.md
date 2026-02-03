# Resume Note - Joint Multi-Fine Training (2026-01-29)

## What's Running
Training IS running (PID check with `ps aux | grep train_joint`).
If not running after system restart, start with:
```bash
cd /mnt/c/Users/sanps/Desktop/Projects/dino/nanoSmolLM
nohup python fVLM/scripts/train_joint_multifine_precomputed.py > /tmp/train_multifine.log 2>&1 &
```
- Auto-resumes from checkpoint (last saved: step ~23,750)
- Metrics log: `outputs/joint_multifine_precomputed/train.log` (NOT the nohup log)
- wandb: project `foveated-vlm-joint`

## Current Training State
| Metric | Value |
|--------|-------|
| Step | ~23,750 / 100,000 |
| Caption ratio | 1.09-1.10 (fine > coarse, thesis validated) |
| Recon ratio | 1.02 |
| Speed | 1.2 s/step |
| VRAM | 17.6 GB / 24 GB |
| RAM | ~9 GB / 16 GB (4 GB free) |
| D: disk | 973 GB free |
| C: disk | ~107 GB free |

## Config
- **Architecture**: deep_query=True, freeze_dino=False, 2 fine iterations
- **Training**: BS=16, 8 frames, lr=3e-5, grad_clip=1.0, grad_accum=1
- **Data**: 513 shards (~102K videos) at `/mnt/d/projects/fVLM/data/frames_latents_sharded/`
- **Checkpoints**: `outputs/joint_multifine_precomputed/checkpoints/` (latest + milestones at 2K, 10K, 20K)
- **DataLoader**: 2 workers, prefetch_factor=2, pin_memory=True
- **Optimizations**: SDPA for LLM, TF32 matmul, cuDNN benchmark, non_blocking transfers, coarse loss every 10 steps

## Key Files
- `fVLM/scripts/train_joint_multifine_precomputed.py` - training script
- `fVLM/src/model/foveated_vlm.py` - model (SDPA on line 90)
- `outputs/joint_multifine_precomputed/train.log` - readable metrics
- `outputs/joint_multifine_precomputed/checkpoints/latest.pt` - checkpoint

## What Was Tried for Speed (and why it didn't help)
| Optimization | Result | Why |
|---|---|---|
| BS=20 | Same throughput (20/1.5s = 16/1.2s = 13.3 samples/s) | VRAM 23.5 GB, too close to limit |
| 3-4 DataLoader workers | Swap thrashing (25s/step spikes) | Each worker loads ~3 GB shard, exceeds 16 GB RAM |
| SDPA for LLM | No measurable improvement | SmolLM2-135M too small, sequences ~73 tokens |
| Sharding (1GB shards) | Same speed as individual files | Shard boundary stalls offset sequential read gains |
| Coarse loss every 10 steps | Marginal (~5%) | 2 no_grad LLM passes are cheap on 135M model |
| torch.compile | Removed, caused warnings | Dynamic shapes in multi-pass forward cause recompilation |

## What Could Actually Help (Not Yet Tried)
1. **Reduce fine iterations 2→1**: Cuts ~3 LLM passes/step (~30% speedup). Changes experiment.
2. **Move data to ext4**: NTFS through WSL2 9P is slow. Need free space on Linux partition.
3. **Increase system RAM**: 16 GB is the real bottleneck (prevents more workers/larger BS).
4. **Smaller shards**: 200→50 samples/shard = ~250 MB/shard. Workers use less RAM, enabling 4+ workers.

## System Constraints
- **16 GB RAM**: Can only run 2 DataLoader workers (each loads ~3 GB shard)
- **WSL2 on NTFS**: Slow filesystem I/O
- **RTX 4090 24 GB**: VRAM is fine at 17.6 GB with BS=16
- **1.2s/step is compute-bound**: 6-8 LLM forward passes per step in multi-fine architecture

## Known Issues
- `torch.utils.checkpoint` warning about requires_grad - harmless
- nohup log corrupted by tqdm `\r` - always read `train.log` instead
- wandb creates new run ID on each restart
- Original individual .pt files were DELETED (only shards remain on D:)
