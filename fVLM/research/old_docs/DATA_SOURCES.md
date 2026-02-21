# Data Sources and Provenance

**Purpose:** Paper-ready reference for all datasets, model weights, citations, and licenses used in the foveated VLM project. Suitable for inclusion in Methods sections and supplementary materials.

**Last updated:** 2026-02-15

---

## 1. Training Data Summary

| Dataset | Modality | Shards | Samples | Disk | Stage | License |
|---------|----------|--------|---------|------|-------|---------|
| The Cauldron | Image VQA | 2,001 | ~2.0M | 25 GB | 2 | Mixed (per-subset) |
| OpenVid-1M | Video caption | 905 | ~905K | 92 GB | 1 | CC-BY-4.0 |
| Vript (long) | Video caption | 400 | ~400K | 53 GB | 3 | Academic only |
| LLaVA-Video-178K | Video SFT | 266 | ~266K | 86 GB | 3 | Academic only |
| VISTA-400K | Temporal reasoning | 171+ | ~171K+ | 45 GB | 3 | MIT |
| VISTA (extra subsets) | Temporal reasoning | 66 | ~66K | 14 GB | 3 | MIT |
| ShareGPT4Video | Video caption | 61 | ~37K | 9.2 GB | 3 | CC-BY-NC-4.0 |
| RLAIF-V | Image preference | 84 | ~83K | 1.4 GB | DPO | CC-BY-NC-4.0 |
| SmolTalk (text retention) | Text only | 490 | ~490K | 928 MB | 1/2/3 | Apache 2.0 |
| LLaVA YouTube (pre-tok) | Video SFT | 22 | ~22K | 4.0 GB | 3 | Academic only |
| WebVid (valid subset) | Video caption | 19 | ~19K | 4.5 GB | 1 | See note |
| Vript (short) | Video caption | 11 | ~11K | 659 MB | 3 | Academic only |
| **Total** | | **~4,500** | **~4.5M** | **~335 GB** | | |

VISTA shard counts are approximate (pipeline may still be running).

### Evaluation Data

| Benchmark | Type | Size | Status |
|-----------|------|------|--------|
| Val 10K | Mixed held-out | 10K samples, ~100 MB | Ready |
| Video-MME | Video MCQ | 2,700 questions, 95 GB raw | Videos downloaded, frames not extracted |
| MVBench | Video MCQ | 4,000 questions, 17 GB raw | Videos downloaded |
| MLVU | Video MCQ | Annotations only, 2.2 MB | Videos deferred (285 GB) |

---

## 2. Training Stage Assignments

| Stage | Purpose | Datasets | Loss |
|-------|---------|----------|------|
| 1 — Video pre-training | Learn temporal alignment | OpenVid-1M, WebVid + 14% SmolTalk | All-text CE |
| 2 — Vision SFT | Learn visual QA | Cauldron (full) + 14% SmolTalk | Answer-only CE |
| 3 — Video SFT | Fine-grained video understanding | LLaVA-Video, VISTA, Vript, ShareGPT4Video + 14% SmolTalk | Answer-only CE |
| DPO (future) | Preference alignment | RLAIF-V | DPO |

---

## 3. Pre-trained Model Weights

| Model | Parameters | Source | License | Path |
|-------|-----------|--------|---------|------|
| SmolLM2-135M-Instruct | 135M | HuggingFace `HuggingFaceTB/SmolLM2-135M-Instruct` | Apache 2.0 | `/workspace/models/SmolLM2-135M-Instruct` |
| SmolLM2-360M-Instruct | 360M | HuggingFace `HuggingFaceTB/SmolLM2-360M-Instruct` | Apache 2.0 | `/workspace/models/SmolLM2-360M-Instruct` |
| SmolLM2-1.7B-Instruct | 1.7B | HuggingFace `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Apache 2.0 | `/workspace/models/SmolLM2-1.7B-Instruct` |
| DINOv2-small | 22M | HuggingFace `facebook/dinov2-small` | Apache 2.0 | `/workspace/models/dinov2-small` |

---

## 4. Full Citations

### Training Datasets

**OpenVid-1M**
```
Nan et al., "OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-Video Generation,"
ICLR 2025. arXiv:2407.02371
```

**The Cauldron** (via Idefics2)
```
Laurençon et al., "What Matters When Building Vision-Language Models?"
arXiv:2405.02246, 2024.
```

**SmolTalk / SmolLM2**
```
Ben Allal et al., "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model,"
arXiv:2502.02737, 2025.
```

**LLaVA-Video-178K**
```
Zhang et al., "Video Instruction Tuning With Synthetic Data,"
arXiv:2410.02713, 2024.
```

**VISTA-400K**
```
Ren et al., "VISTA: Enhancing Long-Duration and High-Resolution Video Understanding
by Video Spatiotemporal Augmentation," CVPR 2025. arXiv:2412.00927
```

**ShareGPT4Video**
```
Chen et al., "ShareGPT4Video: Improving Video Understanding and Generation with Better Captions,"
NeurIPS 2024 (Datasets & Benchmarks). arXiv:2406.04325
```

**Vript**
```
Yang et al., "Vript: A Video Is Worth Thousands of Words,"
NeurIPS 2024 (Datasets & Benchmarks). arXiv:2406.06040
```

**RLAIF-V**
```
Yu et al., "RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness,"
CVPR 2025 (Highlight). arXiv:2405.17220
```

### Pre-trained Models

**DINOv2**
```
Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision,"
Transactions on Machine Learning Research (TMLR), 2024. arXiv:2304.07193
```

**SmolVLM / SmolVLM2** (reference architecture)
```
Marafioti et al., "SmolVLM: Redefining Small and Efficient Multimodal Models,"
arXiv:2504.05299, 2025.
```

### Evaluation Benchmarks

**MVBench**
```
Li et al., "MVBench: A Comprehensive Multi-modal Video Understanding Benchmark,"
CVPR 2024. arXiv:2311.17005
```

**Video-MME**
```
Fu et al., "Video-MME: The First-Ever Comprehensive Evaluation Benchmark
of Multi-modal LLMs in Video Analysis," CVPR 2025. arXiv:2405.21075
```

**MLVU**
```
Zhou et al., "MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding,"
arXiv:2406.04264, 2024.
```

---

## 5. BibTeX Entries

```bibtex
@inproceedings{nan2025openvid,
  title={OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-Video Generation},
  author={Nan, Kepan and Xie, Rui and Zhou, Penghao and Fan, Tiehan and Yang, Zhenheng and Chen, Zhijie and Li, Xiang and Yang, Jian and Tai, Ying},
  booktitle={ICLR},
  year={2025}
}

@article{laurencon2024matters,
  title={What Matters When Building Vision-Language Models?},
  author={Laurençon, Hugo and Tronchon, Léo and Cord, Matthieu and Sanh, Victor},
  journal={arXiv preprint arXiv:2405.02246},
  year={2024}
}

@article{benallal2025smollm2,
  title={SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model},
  author={Ben Allal, Loubna and Lozhkov, Anton and Penedo, Guilherme and Wolf, Thomas and von Werra, Leandro},
  journal={arXiv preprint arXiv:2502.02737},
  year={2025}
}

@article{zhang2024llava_video,
  title={Video Instruction Tuning With Synthetic Data},
  author={Zhang, Yuanhan and Wu, Jinming and Li, Wei and Li, Bo and Ma, Zejun and Liu, Ziwei and Li, Chunyuan},
  journal={arXiv preprint arXiv:2410.02713},
  year={2024}
}

@inproceedings{ren2025vista,
  title={VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation},
  author={Ren, Weiming and Yang, Huan and Min, Jie and Wei, Cong and Chen, Wenhu},
  booktitle={CVPR},
  year={2025}
}

@inproceedings{chen2024sharegpt4video,
  title={ShareGPT4Video: Improving Video Understanding and Generation with Better Captions},
  author={Chen, Lin and Wei, Xilin and Li, Jinsong and Dong, Xiaoyi and Zhang, Pan and Zang, Yuhang and Chen, Zehui and Duan, Haodong and Lin, Bin and Tang, Zhenyu and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2024}
}

@inproceedings{yang2024vript,
  title={Vript: A Video Is Worth Thousands of Words},
  author={Yang, Dongjie and Huang, Suyuan and Xu, Chengqiang and Hu, Yao and Zhao, Hai},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2024}
}

@inproceedings{yu2025rlaifv,
  title={RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness},
  author={Yu, Tianyu and Zhang, Haoye and Yao, Yuan and Dang, Yunkai and Chen, Da and Lu, Xiaoman and Cui, Ganqu and He, Shanshan and Liu, Zhiyuan and Chua, Tat-Seng and Sun, Maosong},
  booktitle={CVPR},
  year={2025}
}

@article{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={Transactions on Machine Learning Research},
  year={2024}
}

@article{marafioti2025smolvlm,
  title={SmolVLM: Redefining Small and Efficient Multimodal Models},
  author={Marafioti, Andrés and Zohar, Orr and Farré, Miquel and Noyan, Merve and Bakouch, Elie and Cuenca, Pedro and Zakka, Cyril and Ben Allal, Loubna and Lozhkov, Anton and Tazi, Nouamane and others},
  journal={arXiv preprint arXiv:2504.05299},
  year={2025}
}

@inproceedings{li2024mvbench,
  title={MVBench: A Comprehensive Multi-modal Video Understanding Benchmark},
  author={Li, Kunchang and Wang, Yali and He, Yinan and Li, Yizhuo and Wang, Yi and Liu, Yi and Wang, Zun and Xu, Jilan and Chen, Guo and Luo, Ping and Wang, Limin and Qiao, Yu},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{fu2025videomme,
  title={Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis},
  author={Fu, Chaoyou and Dai, Yuhan and Luo, Yongdong and Li, Lei and Ren, Shuhuai and Zhang, Renrui and Wang, Zihan and Zhou, Chenyu and Shen, Yunhang and Zhang, Mengdan and others},
  booktitle={CVPR},
  year={2025}
}

@article{zhou2024mlvu,
  title={MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding},
  author={Zhou, Jianxin and others},
  journal={arXiv preprint arXiv:2406.04264},
  year={2024}
}
```

---

## 6. License Analysis

### Permissive (commercial use OK)

| Dataset/Model | License |
|---------------|---------|
| OpenVid-1M | CC-BY-4.0 |
| SmolTalk | Apache 2.0 |
| VISTA-400K | MIT |
| DINOv2 (standard weights) | Apache 2.0 |
| SmolLM2 (all sizes) | Apache 2.0 |
| SmolVLM | Apache 2.0 |
| MVBench | MIT |

### Non-commercial / Academic only

| Dataset | License | Restriction |
|---------|---------|-------------|
| ShareGPT4Video | CC-BY-NC-4.0 | No commercial use |
| RLAIF-V | CC-BY-NC-4.0 | No commercial use |
| LLaVA-Video-178K | Custom | Academic research and education only; must comply with OpenAI Usage Policy |
| Vript | Custom | Academic research only; no redistribution; comply with OpenAI ToS + YouTube/TikTok copyright |
| Video-MME | Custom | Academic research only; no commercial use; no redistribution (eval only) |

### Mixed

| Dataset | License | Notes |
|---------|---------|-------|
| The Cauldron | Per-subset | 50 sub-datasets each with own license. Prompt formatting is CC-BY-4.0. Individual subset licenses range from CC-BY to academic-only. |
| WebVid | Custom | Original dataset largely dead (Shutterstock killed URLs). Our subset is ~19K surviving clips. |

### Implications for Model Release

Training on all datasets constrains the resulting model to **non-commercial / academic research use only** due to CC-BY-NC-4.0 (ShareGPT4Video, RLAIF-V) and academic-only restrictions (LLaVA-Video, Vript).

To release a model for commercial use, exclude: ShareGPT4Video, RLAIF-V, LLaVA-Video-178K, and Vript from the training mix. The remaining data (OpenVid, Cauldron permissive subsets, SmolTalk, VISTA) is sufficient for Stages 1-2 and a reduced Stage 3.

---

## 7. Data Processing Notes

### Shard Format
All data is stored as WebDataset `.tar` files. Each sample consists of:
- **Images:** `{key}.jpg` (single image) or `{key}.{frame_idx:03d}.jpg` (video frames, 0-indexed)
- **Metadata:** `{key}.json` with at minimum `frame_count` and `source` fields

Frames are resized to 224x224 JPEG at 1 FPS, max 64 frames per clip.

### Two JSON Formats

**Format A (pre-tokenized):** Ready for direct consumption by the dataloader.
```json
{"token_ids": [1, 234, ...], "loss_mask": [0, 0, 1, 1, ...], "source": "cauldron/ai2d", "frame_count": 1}
```
Used by: Cauldron, WebVid, SmolTalk, RLAIF-V, stage3_youtube, stage3, val_10k

**Format B (raw caption):** Tokenized at training time by the dataloader.
```json
{"caption": "A dog runs across...", "frame_count": 8, "source": "openvid", "video_id": "abc123"}
```
Used by: OpenVid, LLaVA-Video shards, Vript shards, VISTA, ShareGPT4Video

### Processing Pipelines

- **OpenVid-1M:** Serial `hf_transfer` download of 186 zip parts, ffmpeg frame extraction (14 workers, ~20 vid/s), WebDataset packing. 160/186 parts processed (26 missing/failed from source). Total: 905 shards, 432K clips.
- **Cauldron:** HuggingFace streaming of 50 subsets via `expand_cauldron.py`. Two subsets (okvqa, clevr_math) had iteration bugs requiring `safe_iterate()` wrapper. Total: 2,001 shards, 2.0M samples.
- **LLaVA-Video:** HuggingFace download of video files, ffmpeg 1fps extraction, annotation matching, WebDataset packing. Multiple duration subsets processed in batches.
- **VISTA-400K:** Serial download of pre-extracted frame tarballs from HuggingFace, resize to 224x224, annotation matching across 12 task types, WebDataset packing with parallel workers.
- **Vript:** Short clips extracted from YouTube/TikTok videos per scene boundaries. Both short (11 shards) and long (400 shards) versions available.
- **ShareGPT4Video:** Pre-captioned video clips, processed similarly to LLaVA-Video.

### WebVid Note
WebVid-10M is effectively dead as of 2024 (Shutterstock revoked video hosting URLs). Only ~19K clips from previously-cached URLs remain usable. OpenVid-1M serves as the replacement for Stage 1 video pre-training.
