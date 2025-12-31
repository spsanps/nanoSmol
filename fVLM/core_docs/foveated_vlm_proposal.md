# Foveated Vision-Language Model: A Proposal

## Abstract

We propose a novel vision-language architecture that processes video frame-by-frame using a single attended token per frame, rather than the standard approach of feeding all patch tokens simultaneously. The core LLM receives one compressed representation per frame and outputs a query vector that determines *where to attend* in the next frame. This creates a recurrent loop between perception and attention that mirrors biological vision systems, where the brain directs eye movements based on current understanding and task demands.

---

## 1. Motivation

### 1.1 Current VLM Architectures

Modern vision-language models (LLaVA, Qwen-VL, SmolVLM, etc.) follow a uniform pattern:

```
Image → Vision Encoder → [N patch tokens] → LLM → Text Output
```

For a typical 224×224 image with 16×16 patches, this produces 196 tokens per frame. For video, this scales as `O(frames × patches)`, creating significant computational burden and context length pressure.

More fundamentally, this architecture treats vision as **input-only** — the LLM consumes visual tokens but never *controls* what visual information it receives.

### 1.2 Biological Vision

Human vision operates differently:

- **Foveated attention**: Only ~2° of visual field has high acuity; periphery is low-resolution
- **Saccades**: Eyes move 3-4 times per second to fixation points determined by the brain
- **Top-down control**: Prefrontal cortex guides where to look based on task and expectation
- **Recurrent processing**: Multiple passes through visual cortex "build up" scene understanding
- **Predictive attention**: Saccades are planned ~150-200ms ahead based on anticipated scene dynamics

The brain receives one high-resolution "sample" per fixation and must strategically choose where to look next.

### 1.3 Core Insight

The LLM controls its own visual attention:

```
Frame_t → Encoder(query_{t-1}) → z_t → LLM → query_t → [used for frame_{t+1}]
```

The model learns:
- **What to extract** given a query (the attention mechanism)
- **Where to look next** given current understanding (the query prediction)
- **How to build understanding** from sequential glimpses (the LLM's temporal modeling)

---

## 2. Model Stack

### 2.1 Component Selection

| Component | Model | Details |
|-----------|-------|---------|
| Vision Encoder | DINOv3 ViT-S/16 | Trainable, embed_dim=384, patch_size=16 |
| Core LLM | SmolLM2-135M-Instruct | hidden_size=576 |
| Reconstruction Target | Stable Diffusion VAE | Frozen, `stabilityai/sd-vae-ft-mse` |

### 2.2 Dimension Bridging

```
DINOv3 ViT-S:    embed_dim = 384
SmolLM2-135M:    hidden_size = 576
VAE latent:      4 × 32 × 32 = 4,096 values (for 256×256 input)
```

Required projections:
```python
self.dino_to_llm = nn.Linear(384, 576)   # z → LLM input
self.llm_to_query = nn.Linear(576, 384)  # query → DINO space
```

### 2.3 Patch Count

For 256×256 input images with patch_size=16:
```
Patches per side: 256 / 16 = 16
Total patches: 16 × 16 = 256 patches
```

DINOv3 produces 256 patches of dimension 384. Our query mechanism extracts a single 384-dim token from these.

---

## 3. Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Frame_t ──► Encoder ──► patches_t                                 │
│                              │                                      │
│                              ▼                                      │
│               query_{t-1} ──► Attend ──► z_t (single token)         │
│                   ▲                         │                       │
│                   │                         ▼                       │
│                   │                   ┌───────────┐                 │
│                   │                   │    LLM    │                 │
│                   │                   │           │                 │
│                   │                   │ context:  │                 │
│                   │                   │ z_1...z_t │                 │
│                   │                   └───────────┘                 │
│                   │                         │                       │
│                   │                         ▼                       │
│                   └──────────────────── query_t ──► [for frame_{t+1}]
│                                             │                       │
│                                             ▼                       │
│                                       prediction                    │
│                                             │                       │
│                                             ▼                       │
│                                   VAE latent of frame_{t+1}         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Query-Guided Attention Mechanism

The query is injected as an additional CLS-like token into the vision encoder. An asymmetric attention mask ensures the query can attend to all patch tokens, but patch tokens cannot attend to the query. The output embedding of the query position after all encoder blocks is the foveated visual token z_t.

This is cross-attention readout implemented via masked self-attention.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Encoder Forward Pass                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Patch tokens:    [p_1, p_2, ..., p_N]   ← from image patches  │
│   Query token:     [q]                     ← from LLM           │
│                                                                 │
│   Combined input:  [p_1, p_2, ..., p_N, q]                      │
│                                                                 │
│   Asymmetric Attention Mask (applied at every layer):           │
│                                                                 │
│                  p_1  p_2  ...  p_N   q                         │
│           p_1  [  1    1   ...   1    0  ]                      │
│           p_2  [  1    1   ...   1    0  ]                      │
│           ...  [  ·    ·    ·    ·    0  ]                      │
│           p_N  [  1    1   ...   1    0  ]                      │
│           q    [  1    1   ...   1    1  ]  ← only q sees all   │
│                                                                 │
│   After all encoder blocks:                                     │
│   Output: extract embedding at position q → z_t                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key properties:**

1. **Deep interaction**: The query interacts with patch features at every transformer layer, not just a single cross-attention layer. This produces richer representations.

2. **Patch features unchanged**: Because patches cannot attend to the query, their representations are identical regardless of which query is used.

3. **KV caching efficiency**: Since patch representations are query-independent, we can cache the patch key-value pairs after a single forward pass. Subsequent queries only require propagating the query token through the cached KVs — significantly cheaper than re-encoding.

```python
class FoveatedEncoder(nn.Module):
    def __init__(self, base_encoder, query_dim=384, output_dim=384):
        self.encoder = base_encoder  # DINOv3
        self.query_input_proj = nn.Linear(query_dim, 384)
        self.query_output_proj = nn.Linear(384, output_dim)
        
    def encode_patches(self, images):
        """
        Run once per frame. Produces patch embeddings and caches KV.
        
        Args:
            images: [B, C, H, W]
        Returns:
            patches: [B, N, D_enc] final patch embeddings
            kv_cache: list of (K, V) tuples for each encoder layer
        """
        patches = self.encoder.patch_embed(images)  # [B, N, D_enc]
        patches = patches + self.encoder.pos_embed
        
        kv_cache = []
        for block in self.encoder.blocks:
            # Standard self-attention among patches
            patches, kv = block(patches, return_kv=True)
            kv_cache.append(kv)
        
        return patches, kv_cache
    
    def query(self, q, kv_cache):
        """
        Run per query using cached KVs. Cheap!
        
        Args:
            q: [B, D_q] query vector from LLM
            kv_cache: cached (K, V) from encode_patches
        Returns:
            z: [B, D_out] foveated visual token
        """
        # Project query to encoder dimension
        q_embed = self.query_input_proj(q).unsqueeze(1)  # [B, 1, D_enc]
        
        # Run query through each encoder block
        # Query attends to cached patch KVs; patches don't update
        for block, (K, V) in zip(self.encoder.blocks, kv_cache):
            # Compute query's attention over cached keys
            Q = block.attn.to_q(q_embed)  # [B, 1, D]
            
            # Attention: query attends to all patches
            attn_scores = (Q @ K.transpose(-2, -1)) / sqrt(D)
            attn_weights = softmax(attn_scores, dim=-1)  # [B, 1, N]
            
            # Weighted sum of cached values
            attn_out = attn_weights @ V  # [B, 1, D]
            
            # FFN and residual connections
            q_embed = q_embed + attn_out
            q_embed = q_embed + block.ffn(q_embed)
        
        return self.query_output_proj(q_embed.squeeze(1))  # [B, D_out]
```

---

## 4. Sequence Structure

### 4.1 Multimodal Sequence

The model processes a unified sequence of text and visual tokens:

```
[text_1, ..., text_N, <video_start>, z_1, z_2, ..., z_T, <video_end>, text...]
```

- Text tokens are standard LLM tokens (build context)
- `<video_start>` signals transition to visual mode
- z_t are foveated visual tokens (one per frame)
- `<video_end>` signals return to text mode

### 4.2 Prediction Structure

```
Position:     0    ...   N-1      N              N+1         N+2    ...   N+T
Token:      text_1 ... text_N  <video_start>    z_1         z_2    ...   z_T
                                    │            │           │             │
h output:                         h_vs         h_1         h_2    ...   h_T
                                    │            │           │             │
Predicts:                         z_1          z_2         z_3    ...   z_{T+1}
                                    │            │           │             │
Cond. on:                         z_0          z_1         z_2    ...   z_{T-1}
                                (learned)
```

The h at `<video_start>` position has seen all text context and predicts the first frame — "the first image that pops into your head" given the text description.

---

## 5. Training: Two-Pass Parallel Architecture

### 5.1 The Sequential Dependency Problem

Naive training is sequential:
```
q_0 → z_1 → LLM → q_1 → z_2 → LLM → q_2 → ...
```

We can't compute `z_2` without `q_1`, which requires processing `z_1`. This breaks parallelization.

### 5.2 Two-Pass Solution

**Key insight**: Use a learned static query (which doesn't depend on previous frames) to extract coarse features in parallel, then use the LLM's predicted queries for focused extraction in parallel.

**Pass 1: Query Planning (Static Query)**

A learned static query `q_static` attends to each frame independently. This acts as a learned saliency filter, extracting a "default summary" of each frame while preserving spatial information that simple average pooling would destroy.

```
All frames → Encoder(q_static) → z°_1, z°_2, ..., z°_T
                                        │
        [text..., <video_start_coarse>, z°_1, ..., z°_T]
                                        │
                            ┌───────────┴───────────┐
                            │    LLM (causal mask)  │
                            └───────────────────────┘
                                        │
                                        ▼
                            [q_1, q_2, ..., q_T]  ← all computed in parallel
```

**Pass 2: Focused Extraction (Dynamic Queries)**

The predicted queries from Pass 1 are used for focused extraction.

```
Queries:      [q_init, q_1, q_2, ..., q_{T-1}]   ← shifted by 1
                  │      │     │           │
                  ▼      ▼     ▼           ▼
Frames:      [f_1,    f_2,   f_3,  ...,  f_T]
                  │      │     │           │
                  ▼      ▼     ▼           ▼
Encoder:     [z_1,    z_2,   z_3,  ...,  z_T]    ← all computed in parallel
                            │
        [text..., <video_start_fine>, z_1, ..., z_T]
                            │
                ┌───────────┴───────────┐
                │    LLM (causal mask)  │
                └───────────────────────┘
                            │
                            ▼
                      predictions
                            │
                            ▼
                Loss: pred_t vs vae_latent_{t+1}
```

**Both passes are fully parallel** via causal masking.

**Why q_static instead of average pooling?**

Average pooling destroys spatial information due to permutation invariance. For Pass 1 to predict accurate "where to look" queries, the LLM needs to track object positions and motion across frames. A learned static query preserves this by:

1. Acting as a learned saliency filter (attending to informative regions)
2. Exploiting residual positional information in DINOv3 features
3. Providing consistent extraction across frames (enabling motion detection)

Both streams use the same attention mechanism — only the query source differs (static parameter vs. LLM prediction).

### 5.3 Text Conditions Both Passes

Text provides task/content guidance for attention planning:

```
Pass 1: [text..., <video_start_coarse>, z°_1, z°_2, ..., z°_T]
Pass 2: [text..., <video_start_fine>, z_1, z_2, ..., z_T, <video_end>, ...]
```

If text says "watch the red ball," that influences where to look.

---

## 6. Prediction Head

### 6.1 VAE Latent Prediction

The reconstruction target is the Stable Diffusion VAE latent of the next frame:
- Input image: 256×256×3
- VAE latent: 4×32×32 (downsample factor 8)
- Use deterministic targets: `latent_dist.mean` (not sampling)
- Apply scaling factor consistently: `vae.config.scaling_factor`

VAE latents are precomputed and cached for training efficiency.

### 6.2 FiLM-Style Conditioning

The prediction head receives:
- `h_t`: LLM hidden state (576 dims) — semantic understanding
- `z_vae_t`: Previous frame's VAE latent (4×32×32) — spatial structure

The LLM output modulates how to transform the previous latent into the next latent:

```python
class PredictionHead(nn.Module):
    """FiLM-style conditioning: h_t modulates transformation of z_vae_t"""
    
    def __init__(self, h_dim=576, latent_channels=4):
        super().__init__()
        
        # FiLM parameters from h
        self.h_to_film = nn.Sequential(
            nn.Linear(h_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 2)  # gamma and beta
        )
        
        # Encoder for VAE latent
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 1, 1),
        )
        
        # Decoder back to latent
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, 3, 1, 1),
        )
    
    def forward(self, h, z_vae_prev):
        """
        h: [B, T, 576] or [B, 576]
        z_vae_prev: [B, T, 4, 32, 32] or [B, 4, 32, 32]
        """
        # Handle batched time dimension
        if h.dim() == 3:
            B, T, D = h.shape
            h = h.reshape(B * T, D)
            z_vae_prev = z_vae_prev.reshape(B * T, *z_vae_prev.shape[2:])
            batched = True
        else:
            batched = False
            B = h.shape[0]
        
        # Get FiLM parameters
        film = self.h_to_film(h)  # [B, 512]
        gamma, beta = film.chunk(2, dim=-1)  # [B, 256] each
        
        # Encode previous latent
        feat = self.encoder(z_vae_prev)  # [B, 256, 32, 32]
        
        # Apply FiLM modulation
        gamma = gamma.view(-1, 256, 1, 1)
        beta = beta.view(-1, 256, 1, 1)
        feat = gamma * feat + beta
        
        # Decode to prediction
        pred = self.decoder(feat)  # [B, 4, 32, 32]
        
        if batched:
            pred = pred.reshape(B, T, *pred.shape[1:])
        
        return pred
```

### 6.3 Learned Initial Latent (z_0)

For the first frame prediction, we need a conditioning latent. This is a learned parameter:

```python
self.z_init = nn.Parameter(torch.zeros(1, 4, 32, 32))
```

The model learns what a "blank canvas" looks like. When h has seen text but no visual input, it transforms this blank into the imagined first frame.

### 6.4 Why Non-Residual?

We predict the full latent `ẑ_{t+1} = Head(h_t, z_vae_t)` rather than a delta.

**Residual prediction risks collapse**: The model can learn `Δ ≈ 0` (predict nothing changes), which gives low loss on static scenes but learns nothing useful.

**Non-residual has no collapse mode**: The model must always output a full 4×32×32 latent. For static scenes, it learns an identity-like transformation (which is correct). For dynamic scenes, it must model actual changes.

---

## 7. Auxiliary Coarse Loss

### 7.1 Motivation: Gradient Path Length

Without auxiliary loss, the gradient path to `q_static` is very long:

```
loss_fine → pred_head → h_pass2 → LLM → z_focused → encoder.query 
          → shifted_q → queries → query_head → h_pass1 → LLM 
          → z_coarse → encoder.query → q_static
```

That's 2× full LLM forwards plus multiple projections. Gradients could weaken or become noisy.

### 7.2 Solution: Auxiliary Reconstruction Loss on Pass 1

Add a reconstruction loss directly on Pass 1:

```
Pass 1:
    z_coarse → LLM → h_pass1 ─┬─→ pred_head → loss_coarse (auxiliary)
                              │
                              └─→ query_head → queries (for Pass 2)

Pass 2:
    z_focused → LLM → h_pass2 → pred_head → loss_fine (main)

total_loss = loss_fine + λ_coarse * loss_coarse
```

Now `q_static` gets gradient through a much shorter path:

```
loss_coarse → pred_head → h_pass1 → LLM → z_coarse → encoder.query → q_static
```

### 7.3 Benefits

**Direct supervision for Pass 1**: The Pass 1 LLM learns to actually understand the video, not just "output something that helps Pass 2."

**Stronger signal to q_static**: Currently q_static only learns "be useful for query planning." With auxiliary loss: "extract information that enables reconstruction." These are aligned but the second is more direct.

**Built-in ablation metric**:
- If `loss_fine < loss_coarse`: Dynamic queries help! Core hypothesis validated.
- If `loss_fine ≈ loss_coarse`: Focused attention isn't adding value. Investigate.
- If `loss_fine > loss_coarse`: Something wrong with query learning. Debug.

**Training stability**: Auxiliary losses are well-established for deep networks (GoogLeNet/Inception, etc.). Prevents early layers from getting lost.

### 7.4 Shared Prediction Head

Both passes use the same prediction head:

```python
pred_coarse = self.pred_head(h_coarse, prev_latents)
pred_fine = self.pred_head(h_fine, prev_latents)
```

This is simpler (fewer parameters) and forces the LLM to produce similar "understanding" representations in both passes. The only difference should be the *quality* of information extracted, not the format.

---

## 8. Training Code

```python
class FoveatedVideoModel(nn.Module):
    def __init__(self, lambda_coarse=1.0):
        # Vision encoder
        self.encoder = FoveatedEncoder(
            base_encoder=DINOv3_ViT_S_16(),  # trainable
            query_dim=384,
            output_dim=384
        )
        
        # Core LLM
        self.llm = SmolLM2_135M()  # hidden_size=576
        
        # Projections
        self.dino_to_llm = nn.Linear(384, 576)
        self.llm_to_query = nn.Linear(576, 384)
        
        # Output heads
        self.query_head = nn.Linear(576, 384)
        self.pred_head = PredictionHead(h_dim=576, latent_channels=4)  # shared
        
        # Learned queries
        self.q_static = nn.Parameter(torch.randn(1, 384))   # for Pass 1
        self.q_init = nn.Parameter(torch.randn(1, 384))     # for first frame in Pass 2
        
        # Learned initial VAE latent
        self.z_vae_init = nn.Parameter(torch.zeros(1, 4, 32, 32))
        
        # Mode tokens (in LLM embedding space)
        self.coarse_token = nn.Parameter(torch.randn(1, 1, 576))
        self.fine_token = nn.Parameter(torch.randn(1, 1, 576))
        
        # Loss weighting
        self.lambda_coarse = lambda_coarse
    
    def forward(self, text_embeds, raw_frames, vae_latents):
        """
        text_embeds: [B, N_text, 576] - pre-embedded text tokens
        raw_frames: [B, T, 3, 256, 256] - video frames
        vae_latents: [B, T, 4, 32, 32] - precomputed VAE latents (scaled)
        
        Returns:
            loss: combined loss
            loss_fine: Pass 2 reconstruction loss (main)
            loss_coarse: Pass 1 reconstruction loss (auxiliary)
        """
        B, T = raw_frames.shape[:2]
        N_text = text_embeds.shape[1]
        
        # === Encode all frames with DINO, cache KVs ===
        all_kv_caches = []
        for t in range(T):
            _, kv_cache = self.encoder.encode_patches(raw_frames[:, t])
            all_kv_caches.append(kv_cache)
        
        # === Pass 1: Query Planning with q_static ===
        q_static = self.q_static.expand(B, -1)  # [B, 384]
        
        z_coarse_list = []
        for t in range(T):
            z_t = self.encoder.query(q_static, all_kv_caches[t])  # [B, 384]
            z_coarse_list.append(z_t)
        z_coarse = torch.stack(z_coarse_list, dim=1)  # [B, T, 384]
        z_coarse = self.dino_to_llm(z_coarse)  # [B, T, 576]
        
        # Build Pass 1 sequence: [text, <coarse>, z°_1, ..., z°_T]
        coarse_token = self.coarse_token.expand(B, 1, -1)
        seq_pass1 = torch.cat([text_embeds, coarse_token, z_coarse], dim=1)
        
        h_pass1 = self.llm(seq_pass1)  # [B, N_text + 1 + T, 576]
        
        # Extract queries from positions after <coarse> token
        h_for_queries = h_pass1[:, N_text + 1:]  # [B, T, 576]
        queries = self.query_head(h_for_queries)  # [B, T, 384]
        
        # === Auxiliary loss on Pass 1 ===
        h_coarse_for_pred = h_pass1[:, N_text : N_text + T]  # [B, T, 576]
        
        # Conditioning latents: [z_vae_init, z_vae_1, ..., z_vae_{T-1}]
        z_vae_init = self.z_vae_init.expand(B, 1, -1, -1, -1)  # [B, 1, 4, 32, 32]
        prev_latents = torch.cat([z_vae_init, vae_latents[:, :-1]], dim=1)  # [B, T, 4, 32, 32]
        
        # Targets: [z_vae_1, z_vae_2, ..., z_vae_T]
        target_latents = vae_latents  # [B, T, 4, 32, 32]
        
        # Coarse prediction (shared head)
        pred_coarse = self.pred_head(h_coarse_for_pred, prev_latents)  # [B, T, 4, 32, 32]
        loss_coarse = F.mse_loss(pred_coarse, target_latents)
        
        # === Shift queries: q_t used for frame_{t+1} ===
        q_init = self.q_init.expand(B, 1, -1)  # [B, 1, 384]
        shifted_q = torch.cat([q_init, queries[:, :-1]], dim=1)  # [B, T, 384]
        
        # === Pass 2: Focused Extraction with dynamic queries ===
        z_focused_list = []
        for t in range(T):
            z_t = self.encoder.query(shifted_q[:, t], all_kv_caches[t])  # [B, 384]
            z_focused_list.append(z_t)
        z_focused = torch.stack(z_focused_list, dim=1)  # [B, T, 384]
        z_focused = self.dino_to_llm(z_focused)  # [B, T, 576]
        
        # Build Pass 2 sequence: [text, <fine>, z_1, ..., z_T]
        fine_token = self.fine_token.expand(B, 1, -1)
        seq_pass2 = torch.cat([text_embeds, fine_token, z_focused], dim=1)
        
        h_pass2 = self.llm(seq_pass2)  # [B, N_text + 1 + T, 576]
        
        # === Main loss on Pass 2 ===
        h_fine_for_pred = h_pass2[:, N_text : N_text + T]  # [B, T, 576]
        
        # Fine prediction (same shared head)
        pred_fine = self.pred_head(h_fine_for_pred, prev_latents)  # [B, T, 4, 32, 32]
        loss_fine = F.mse_loss(pred_fine, target_latents)
        
        # === Combined loss ===
        loss = loss_fine + self.lambda_coarse * loss_coarse
        
        return loss, loss_fine, loss_coarse
```

---

## 9. Inference

### 8.1 Frame-by-Frame Processing

```
════════════════════════════════════════════════════════════════════
Initialization (process text)
════════════════════════════════════════════════════════════════════

text_embeds → LLM → build text context in KV cache

════════════════════════════════════════════════════════════════════
Frame t arrives
════════════════════════════════════════════════════════════════════

# Encode patches once, cache KV
patches_t, kv_cache_t = Encoder.encode_patches(frame_t)

COARSE STREAM (planning):
    z°_t = Encoder.query(q_static, kv_cache_t)
    h_plan_t = LLM_step(z°_t, kv_cache=kv_coarse)
    q_t = query_head(h_plan_t)      // query for frame_{t+1}
    update kv_coarse

FOCUSED STREAM (understanding):
    z_t = Encoder.query(q_{t-1}, kv_cache_t)  // use PREVIOUS query
    h_exec_t = LLM_step(z_t, kv_cache=kv_focused)
    pred_t = pred_head(h_exec_t, z_vae_{t-1})  // predict VAE latent
    update kv_focused

Output: pred_t, q_t
```

### 8.2 Efficiency Analysis

Per frame:
- 1× Encoder patch forward (produces KV cache)
- 2× Encoder query forward (cheap — just propagate query through cached KVs)
- 2× LLM single-token step with KV cache
- 1× Prediction head forward

The KV caching in the encoder is critical: encoding patches is expensive (full transformer forward), but querying with a cached KV is cheap (just the query token propagates).

### 8.3 Train/Inference Consistency

Training with causal mask is mathematically equivalent to inference with KV cache:
- `q_t` only depends on `z°_1, ..., z°_t` in both cases
- `pred_t` only depends on `z_1, ..., z_t` in both cases

**No train/inference mismatch.**

### 8.4 Inference Modes

**Video Prediction** (given real frames, predict next):
```
Input: frames 1, 2, 3 (real)
Output: VAE latent for frame 4
Decode via frozen VAE for visualization
```

**Video Captioning** (given frames, generate text):
```
Input: frames 1, 2, ..., T
After <video_end>: generate text autoregressively
Output: "The video shows a ball bouncing..."
```

---

## 10. Gradient Flow

### 10.1 Main Path (Pass 2 → Pass 1)

```
loss_fine
    │
    ▼
pred_latents_fine = pred_head(h_fine_for_pred, prev_latents)
    │
    ├──────────► prev_latents (no gradient — precomputed)
    │
    ▼
h_fine_for_pred = h_pass2[:, N_text : N_text + T]
    │
    ▼
h_pass2 = LLM(seq_pass2)
    │
    ▼
z_focused = Encoder.query(shifted_q, kv_caches)
    │
    ├──────────► kv_caches (DINO gradients)
    │
    ▼
shifted_q = [q_init, queries[:, :-1]]
    │
    ▼
queries = query_head(h_pass1[:, N_text + 1:])
    │
    ▼
h_pass1 = LLM(seq_pass1)
    │
    ▼
z_coarse = Encoder.query(q_static, kv_caches)
    │
    ├──────────► kv_caches (DINO gradients)
    │
    ▼
q_static (learned parameter)
```

### 10.2 Auxiliary Path (Direct to Pass 1)

```
loss_coarse
    │
    ▼
pred_latents_coarse = pred_head(h_coarse_for_pred, prev_latents)
    │
    ▼
h_coarse_for_pred = h_pass1[:, N_text : N_text + T]
    │
    ▼
h_pass1 = LLM(seq_pass1)
    │
    ▼
z_coarse = Encoder.query(q_static, kv_caches)
    │
    ▼
q_static (learned parameter)  ← MUCH SHORTER PATH
```

**End-to-end differentiable.** The auxiliary loss provides a direct gradient path to Pass 1 components, while the main loss ensures Pass 2 learns to improve upon Pass 1.

---

## 11. Why Selective Attention Emerges

The training objective is next-frame VAE latent prediction. Consider a video with a ball moving across a static background:

**If model attends uniformly:**
- z_focused captures: background texture, diffuse ball signal
- Prediction of ball position in frame_{t+1}: poor
- Loss: high

**If model attends to the ball:**
- z_focused captures: ball position, motion direction
- Prediction of ball position in frame_{t+1}: accurate
- Loss: low

The gradient tells the query: "attending to the ball region reduced loss; do more of that."

Static background contributes nothing to next-frame prediction (it's the same in VAE space). Dynamic regions contain all the predictive information. The model learns to attend to what changes.

---

## 12. Relation to Prior Work

### 11.1 JEPA (Joint Embedding Predictive Architecture)

LeCun's proposal for self-supervised learning via prediction in representation space.

**Similarity**: Prediction in embedding space rather than pixel space

**Difference**: We have explicit attention control loop; JEPA uses masking

### 11.2 V-JEPA

Meta's video extension of JEPA.

**Similarity**: Video, predicting in representation space

**Difference**: V-JEPA predicts masked regions; we predict next frame from selective attention

### 11.3 Perceiver / Perceiver IO

Uses learned queries to compress arbitrary inputs.

**Similarity**: Query-based attention over visual tokens

**Difference**: Our queries are *dynamic* (output by LLM based on context), theirs are *learned* (fixed parameters)

### 11.4 Recurrent Visual Attention (RAM, DRAW)

Classic work on learned attention policies.

**Similarity**: Learned where-to-look policies

**Difference**: They used RL (REINFORCE) due to hard attention; we use end-to-end differentiable soft attention

### 11.5 MAE (Masked Autoencoder)

Predicts masked patches from visible patches.

**Similarity**: Reconstruction objective forces understanding

**Difference**: We *select* which patches to see; MAE randomly masks

---

## 13. Evaluation

### 13.1 Intrinsic Metrics

**Reconstruction quality**:
- MSE in VAE latent space
- Decode and measure PSNR/SSIM in pixel space

**Attention selectivity**:
- Attention entropy (lower = more selective)
- Attention coverage (fraction of patches with significant attention weight)

**Temporal coherence**:
- Do attention patterns track moving objects?
- Visualize attention heatmaps over time

**Core hypothesis test (built-in)**:
- Compare `loss_fine` vs `loss_coarse` during training
- If `loss_fine < loss_coarse`: Dynamic foveated attention helps
- Ratio `loss_coarse / loss_fine` quantifies improvement from selective attention

### 13.2 Ablations

- Fixed queries vs learned dynamic queries (baseline comparison)
- q_static vs average pooling for Pass 1 (spatial awareness)
- Coarse-only vs two-pass (does selective attention help?)
- Query dimension ablation
- With vs without text conditioning
- λ_coarse sweep (auxiliary loss weighting)

---

## Appendix A: Full Training Diagram

```
═══════════════════════════════════════════════════════════════════════════════
                                PASS 1: QUERY PLANNING
                       (Static query q_static, parallel via causal mask)
                              + AUXILIARY RECONSTRUCTION LOSS
═══════════════════════════════════════════════════════════════════════════════

Frame 1              Frame 2              Frame 3              Frame 4
   │                    │                    │                    │
   ▼                    ▼                    ▼                    ▼
┌──────┐            ┌──────┐            ┌──────┐            ┌──────┐
│DINOv3│            │DINOv3│            │DINOv3│            │DINOv3│
│      │            │      │            │      │            │      │
│[p,q] │            │[p,q] │            │[p,q] │            │[p,q] │
│ mask │            │ mask │            │ mask │            │ mask │
└──────┘            └──────┘            └──────┘            └──────┘
   │                    │                    │                    │
   ▼                    ▼                    ▼                    ▼
  z°_1                z°_2                z°_3                z°_4
(q_static)          (q_static)          (q_static)          (q_static)
   │                    │                    │                    │
   └────────────────────┴────────────────────┴────────────────────┘
                                    │
                [text, <video_start_coarse>, z°_1, z°_2, z°_3, z°_4]
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │      LLM (causal)     │
                        └───────────────────────┘
                                    │
                                    ▼
                      [h_vs, h°_1, h°_2, h°_3, h°_4]
                         │    │    │    │    │
                         │    │    │    │    │
           ┌─────────────┴────┴────┴────┴────┴─────────────┐
           │                                               │
           ▼                                               ▼
    query_head                                    pred_head (shared)
           │                                               │
           ▼                                               │
   [q_1, q_2, q_3, q_4]                          cond on: z_0  z_1  z_2  z_3
           │                                               │
    shift: [q_init, q_1, q_2, q_3]                        ▼
           │                                    loss_coarse (auxiliary)
           │                                               │
           ▼                                               ▼
      (to Pass 2)                               targets: z_1  z_2  z_3  z_4


═══════════════════════════════════════════════════════════════════════════════
                              PASS 2: FOCUSED EXTRACTION
                      (Dynamic queries from Pass 1, parallel)
                              + MAIN RECONSTRUCTION LOSS
═══════════════════════════════════════════════════════════════════════════════

Queries:            q_init             q_1              q_2              q_3
                       │                │                │                │
                       ▼                ▼                ▼                ▼
                 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
                 │  DINOv3  │    │  DINOv3  │    │  DINOv3  │    │  DINOv3  │
Frames:          │  [p, q]  │    │  [p, q]  │    │  [p, q]  │    │  [p, q]  │
                 │   mask   │    │   mask   │    │   mask   │    │   mask   │
                 └──────────┘    └──────────┘    └──────────┘    └──────────┘
                       │                │                │                │
                       ▼                ▼                ▼                ▼
                     z_1              z_2              z_3              z_4
                       │                │                │                │
                       └────────────────┴────────────────┴────────────────┘
                                               │
                        [text, <video_start_fine>, z_1, z_2, z_3, z_4]
                                               │
                                               ▼
                                   ┌───────────────────────┐
                                   │      LLM (causal)     │
                                   └───────────────────────┘
                                               │
                                               ▼
                                         [h_vs, h_1, h_2, h_3, h_4]
                                           │    │    │    │
                                           ▼    ▼    ▼    ▼
                              pred_head (shared) with FiLM conditioning
                                           │    │    │    │
                                cond on:  z_0  z_1  z_2  z_3
                               (learned)   │    │    │    │
                                           ▼    ▼    ▼    ▼
                              loss_fine (main) vs targets: z_1  z_2  z_3  z_4


═══════════════════════════════════════════════════════════════════════════════
                                  COMBINED LOSS
═══════════════════════════════════════════════════════════════════════════════

                    total_loss = loss_fine + λ_coarse * loss_coarse

        If loss_fine < loss_coarse → dynamic foveated attention helps!
```

---

## Appendix B: Inference Diagram

```
═══════════════════════════════════════════════════════════════════════════════
                              STREAMING INFERENCE
                          (Frame-by-frame with KV cache)
═══════════════════════════════════════════════════════════════════════════════

Text: "A ball bounces across the floor" → process → KV cache initialized

Frame 1          Frame 2          Frame 3          Frame 4
   │                │                │                │
   ▼                ▼                ▼                ▼
┌──────┐         ┌──────┐         ┌──────┐         ┌──────┐
│DINOv3│         │DINOv3│         │DINOv3│         │DINOv3│
│encode│         │encode│         │encode│         │encode│
│ + KV │         │ + KV │         │ + KV │         │ + KV │
└──────┘         └──────┘         └──────┘         └──────┘
   │                │                │                │
   ▼                ▼                ▼                ▼

COARSE STREAM (plans where to look, uses q_static + cached KV)
───────────────────────────────────────────────────────────────────────────────
   │                │                │                │
   ▼                ▼                ▼                ▼
query(q_static)  query(q_static)  query(q_static)  query(q_static)
   │                │                │                │
   ▼                ▼                ▼                ▼
  z°_1             z°_2             z°_3             z°_4
   │                │                │                │
   ▼                ▼                ▼                ▼
┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐
│ LLM │───kv────►│ LLM │───kv────►│ LLM │───kv────►│ LLM │
└─────┘          └─────┘          └─────┘          └─────┘
   │                │                │                │
   ▼                ▼                ▼                ▼
  q_1              q_2              q_3              q_4
   │                │                │                │
   │    ┌───────────┘    ┌──────────┘     ┌──────────┘
   │    │                │                │
   ▼    ▼                ▼                ▼

FOCUSED STREAM (understands and predicts, uses dynamic query + cached KV)
───────────────────────────────────────────────────────────────────────────────
   │    │                │                │
   ▼    ▼                ▼                ▼
query(q_init)     query(q_1)       query(q_2)       query(q_3)
   │                │                │                │
   ▼                ▼                ▼                ▼
  z_1              z_2              z_3              z_4
   │                │                │                │
   ▼                ▼                ▼                ▼
┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐
│ LLM │───kv────►│ LLM │───kv────►│ LLM │───kv────►│ LLM │
└─────┘          └─────┘          └─────┘          └─────┘
   │                │                │                │
   ▼                ▼                ▼                ▼
pred(h,z_0)     pred(h,z_1)      pred(h,z_2)      pred(h,z_3)
   │                │                │                │
   ▼                ▼                ▼                ▼
ẑ_vae_1          ẑ_vae_2          ẑ_vae_3          ẑ_vae_4
```

---

## Appendix C: Attention Mask Detail

The asymmetric attention mask is the key mechanism enabling efficient query-based extraction:

```
                  Patch tokens              Query
                  ────────────              ─────
            p_1   p_2   p_3  ...  p_N        q
           ┌───┬─────┬─────┬───┬─────┬─────────┐
     p_1   │ ✓ │  ✓  │  ✓  │...│  ✓  │    ✗    │
           ├───┼─────┼─────┼───┼─────┼─────────┤
     p_2   │ ✓ │  ✓  │  ✓  │...│  ✓  │    ✗    │
           ├───┼─────┼─────┼───┼─────┼─────────┤
     p_3   │ ✓ │  ✓  │  ✓  │...│  ✓  │    ✗    │
           ├───┼─────┼─────┼───┼─────┼─────────┤
     ...   │...│ ... │ ... │...│ ... │   ...   │
           ├───┼─────┼─────┼───┼─────┼─────────┤
     p_N   │ ✓ │  ✓  │  ✓  │...│  ✓  │    ✗    │
           ├───┼─────┼─────┼───┼─────┼─────────┤
      q    │ ✓ │  ✓  │  ✓  │...│  ✓  │    ✓    │
           └───┴─────┴─────┴───┴─────┴─────────┘

✓ = can attend
✗ = cannot attend

Result:
- Patches interact with each other normally (standard ViT behavior)
- Patches CANNOT see the query (their representations are query-independent)
- Query CAN see all patches (extracts information based on query content)
- Query CAN see itself (allows self-gating if needed)
```

This asymmetry enables KV caching: compute patch KVs once, reuse for any query.
