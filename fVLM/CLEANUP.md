# Codebase Cleanup Plan

## Problem
79+ scripts in fVLM/scripts/, most are one-off experiments or duplicates.
This makes the codebase hard to navigate and maintain.

## Core Scripts (KEEP in scripts/)

### Training
- `train_joint_multifine_precomputed.py` - Main training script (precomputed data)
- `train_baseline_vlm.py` - Baseline VLM for comparison
- `train_multitask.py` - Legacy multi-task training

### Evaluation
- `evaluate_caption_loss.py` - Unified caption loss evaluation
- `test_captioning.py` - Captioning test script

### Data Preparation
- `create_data_split.py` - Train/val split creation
- `shard_precomputed_data.py` - Shard the precomputed data
- `precompute_frames_latents.py` - Main precompute script

### Visualization
- `visualize_attention.py` - Attention visualization

### Demos
- `gradio_attention_demo.py` - Interactive demo

## Archive (move to scripts/archive/)

### Legacy Training
- train_phase1.py
- train_phase2.py
- train_fast.py
- train_freeze_dino.py
- train_large_scale.py
- train_streaming_webvid.py
- train_smolvlm.py
- train_smolvlm_singleepoch.py
- train_foveated_singleepoch.py
- train_captioning_scaled.py
- train_dino_efficient.py
- train_precomputed_efficient.py
- train_joint_8h_optimized.py
- train_joint_multifine_8h.py
- train_joint_recon_caption.py
- train_foveated_optimized.py

### Experimental Scripts
- experiment_*.py (all)
- diagnostic_*.py (all)
- ablation_experiments.py
- comprehensive_ablations.py
- preliminary_experiments.py
- quick_experiments.py

### Redundant Evaluation
- evaluate_24h.py
- evaluate_captions.py
- evaluate_captions_local.py
- evaluate_fair_comparison.py
- evaluate_optimized_comparison.py
- estimate_flops.py
- find_crossover.py

### Visualization/Generation (keep only essential)
- generate_*.py (all)
- visualize_*.py (except visualize_attention.py)
- temporal_*.py
- analyze_*.py
- caption_comparison_with_gt.py
- create_caption_previews.py
- live_attention_demo.py
- video_attention_demo.py
- save_frame_grids.py

### Redundant Precompute
- precompute_all_latents.py
- precompute_webvid.py
- precompute_fast.py
- precompute_6h_max.py
- precompute_dino.py
- profile_*.py
- test_throughput_5min.py

## After Cleanup (scripts/ should have ~10 files)

```
scripts/
├── train_joint_multifine_precomputed.py
├── train_baseline_vlm.py
├── train_multitask.py
├── evaluate_caption_loss.py
├── test_captioning.py
├── create_data_split.py
├── shard_precomputed_data.py
├── precompute_frames_latents.py
├── visualize_attention.py
├── gradio_attention_demo.py
├── setup/                     # Keep as-is
└── archive/                   # All archived scripts
```
