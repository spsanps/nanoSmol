# SmolVLM2 Training Reference

Distilled from SmolVLM2 blog, SmolVLM paper (arXiv:2504.05299), Idefics3 paper (arXiv:2408.12637), SmolLM2 paper (arXiv:2502.02737). This is the model we're matching/comparing against.

---

## SmolVLM2 Pipeline

### Stage 0: Context Extension (text-only, prerequisite)
- Extend SmolLM2 context: RoPE base 10K → 273K for 16K token context
- Data mix: 20% Books (Dolma) + 20% Code (The Stack 8K+) + 20% FineWeb-Edu + 20% DCLM + 20% Math (upsampled)
- Math upsampled to prevent GSM8K regression

### Stage 1: Vision SFT
- Vision encoder (SigLIP) + LLM initially frozen or DoRA
- Connector (pixel shuffle MLP) fully trainable (newly initialized)
- Data: The Cauldron + Docmatix
- Loss: Text CE (next-token prediction)

### Stage 2: Video SFT
- 3.3M samples from 10 datasets
- Mix: 33% video, 34% image, 20% text, 12% multi-image
- Frames at 1 FPS, up to 50 frames per video
- Loss: Text CE on answer tokens only (user prompts masked)

### No DPO/RLHF on the VLM
- Alignment inherited from SmolLM2-1.7B-Instruct (which had DPO on UltraFeedback)
- 14% text retention in all stages preserves instruction-following

---

## Idefics3 Staged Unfreezing (SmolVLM's parent)

| Stage | Vision (SigLIP) | Connector | LLM | Steps | Data |
|-------|-----------------|-----------|-----|-------|------|
| Pre-train 1 | Frozen | Train (new) | Frozen | 1000 | Cauldron + Docmatix + OBELICS |
| Pre-train 2 | DoRA | Train | DoRA | 3000 | Same + additional |
| Pre-train 3 | DoRA | Train | DoRA | 1500 | Synthetic focus (Docmatix, WebSight, PixelProse) |
| SFT | DoRA | Train | DoRA | 5000 | Cauldron extended + text instruction |

- Full unfreezing **diverged** at 8B. DoRA was the stable solution.
- NEFTune noise applied during SFT
- Loss only on answer tokens (not user prompts)
- Training: 5 days on 32 H100 nodes

---

## SmolVLM2 Video Datasets (10 datasets, 3.3M total)

| Dataset | Type |
|---------|------|
| LLaVA-OneVision | General VL |
| M4-Instruct | Multi-image reasoning |
| Mammoth (MAmmoTH-VL) | Math + reasoning |
| LLaVA-Video-178K | Video captioning/QA |
| FineVideo | Narrative comprehension |
| VideoStar | Temporal reasoning |
| VRipt | Video captioning |
| Vista-400K | Temporal reasoning |
| MovieChat | Long-form video QA |
| ShareGPT4Video | Video captioning |

---

## SmolVLM2 Key Findings

| Finding | Detail | Impact |
|---------|--------|--------|
| **14% text retention** | Removing text hurts 3.7% (video), 6.5% (image) | Always keep text in mix |
| **Don't reuse LLM-SFT text** | SmolTalk recycling hurts small VLMs | Curate separate text Q&A |
| **CoT ratio** | Only 0.02-0.05% CoT examples optimal | More CoT degrades small models |
| **Answer-only loss** | Masking user prompts improved accuracy | Standard for SFT |
| **3.5min video cap** | Returns diminish beyond ~3.5 min training videos | Focus on shorter videos |
| **Frame averaging hurts** | Don't average frames — sample individually | Use 1 FPS sampling |

---

## SmolVLM2 Benchmarks (our comparison targets)

| Benchmark | SmolVLM2-2.2B | Category |
|-----------|--------------|----------|
| MMMU | 42.0 | General multimodal |
| MathVista | 51.5 | Math reasoning |
| MMStar | 46.0 | General multimodal |
| OCRBench | 72.9 | OCR |
| DocVQA | 79.98 | Document QA |
| TextVQA | 73.21 | Text-based VQA |
| Video-MME | 52.1 | Video understanding |
| MLVU | 55.2 | Video language understanding |
| MVBench | 46.27 | Video benchmark |
| GPU RAM (video) | 5.2 GB | Efficiency |

---

## SmolLM2 SFT Data (SmolTalk, ~1.3M samples)

For 14% text retention, use subsets of these:

| Dataset | Samples | Purpose |
|---------|---------|---------|
| Smol-Magpie-Ultra | 400K | Core instruction-following |
| Smol-Constraint | 36K | Output formatting |
| Smol-Rewrite | 50K | Text editing |
| Smol-Summarize | 100K | Summarization |
| MetaMathQA | 50K | Math |
| NuminaMath-CoT | varies | Hard math |
| Self-OSS-Starcoder2-Instruct | 50K | Python coding |
| OpenHermes2.5 | 100K | Preserves MMLU, BBH |
| SystemChats2.0 | 30K | System prompt variety |
| APIGen-Function-Calling | 80K | Tool use |

SmolLM2-1.7B DPO: UltraFeedback dataset, 2 epochs, lr=1e-6, beta=0.5. Only 1.7B got DPO (not 135M/360M).

---

## Architecture Details

- **Vision encoder:** Shape-optimized SigLIP, 384x384 patches, 14x14 inner patches
- **Compression:** 9x pixel shuffle (vs 4x in Idefics3) → 81 tokens per 384x384 image
- **LLM:** SmolLM2-1.7B-Instruct (576-dim hidden)
- **Context:** 16K tokens
- **Checkpoint selection:** Weighted composite of MMMU + MathVista + DocVQA + TextVQA + MMStar + AI2D, saved every 25 optimization steps

---

*Distilled: 2026-02-09*
*Sources: SmolVLM2 blog, SmolVLM paper, Idefics3 paper, SmolLM2 paper, HuggingFace model cards*
