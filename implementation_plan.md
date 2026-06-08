# Zero-Shot Anomaly Detection via Patch Prediction (PPAD)

## Overview

Anomaly detection by **predictive patch coding**. The core hypothesis: in a normal image, any patch can be accurately predicted from its surrounding context. Anomalous patches break this predictability — their actual appearance diverges from what context implies.

### Method Summary

For each patch `i` in an image:
1. **hp** = Encode the isolated patch → "ground truth" patch representation
2. **hi** = Encode the image with patch `i` masked out → context representation  
3. **ho** = Predictor(hi, pos_i) → predicted patch representation
4. **score_i** = 1 - cos_sim(hp, ho) → patch-level anomaly score
5. **image_score** = aggregate(score_i over all patches) → image-level score

---

## Architecture

### Encoders

We will use a **pretrained ViT (DINOv2 ViT-B/14 or ViT-L/14)** as the backbone for both encoders. DINOv2 features are well-suited for this:
- Rich local/global semantic representations
- Handles variable patch masking naturally via CLS + patch tokens
- State-of-the-art for few-shot/zero-shot anomaly detection (used in WinCLIP, MuSc, etc.)

| Encoder | Input | Output |
|---|---|---|
| Patch Encoder (fp) | Single patch (cropped image region) | hp ∈ R^d |
| Context Encoder (fc) | Full image with patch masked | hi ∈ R^d |

**Key design choice**: Both encoders share the same frozen DINOv2 backbone weights. The two "encoders" are actually the **same model** used in two different ways:
- **hp**: Extract the patch token at position `i` from a **full image** forward pass (no masking)
- **hi**: Extract the CLS token or mean of remaining patch tokens from a **masked** forward pass

This avoids training and keeps the method truly zero-shot.

### Predictor

A **lightweight MLP or Transformer** that takes:
- Input: hi (context embedding, dim d) + positional encoding of the target patch
- Output: ho (predicted patch embedding, dim d)

**Two design options:**

**Option A (Simple — no training)**: Nearest-neighbor prediction from a memory bank of normal patch embeddings built during a "calibration" pass on training data. → Truly zero-shot but requires a small normal set.

**Option B (Lightweight trained predictor)**: Train a small MLP predictor on normal images from the training split. The predictor learns to map (context, position) → patch embedding. Only the predictor is trained; encoders are frozen. → Requires training data but is still zero-shot w.r.t. test categories.

**Option C (Truly zero-shot, no training data)**: Use the ViT's own masked attention mechanism — feed the masked image and use the unmasked patch's predicted representation via cross-attention. This requires a MAE-style or I-JEPA-style pretrained model.

**Proposed approach**: Implement **Option B first** (simple MLP predictor trained on normal MVTec training images with frozen DINOv2), with an ablation of Option A.

---

## Implementation Plan

### Directory Structure

```
/home/rgb/Desktop/research/ad/latent/
├── data/
│   └── mvtec/          # symlink or download script
├── models/
│   ├── encoders.py     # DINOv2 patch + context encoders
│   ├── predictor.py    # MLP/Transformer predictor
│   └── ppad.py         # Full PPAD model wrapper
├── datasets/
│   └── mvtec.py        # MVTec dataset loader
├── train.py            # Train predictor on normal images
├── evaluate.py         # Run evaluation on MVTec test set
├── anomaly_map.py      # Produce per-pixel anomaly maps
├── utils/
│   ├── metrics.py      # AUROC, PRO, etc.
│   └── visualization.py
├── configs/
│   └── default.yaml
└── README.md
```

---

## File-by-File Plan

### [NEW] `models/encoders.py`
- `PatchEncoder`: wraps DINOv2, extracts CLS or patch token for a given patch position from a full image pass
- `ContextEncoder`: wraps same DINOv2, masks a patch (zeros/noise/mean fill) and extracts global context embedding (CLS token or mean of remaining tokens)
- Positional encoding utilities (2D sinusoidal or learned)

### [NEW] `models/predictor.py`
- `PatchPredictor`: small MLP (3-4 layers, hidden_dim=512) or Transformer
  - Input: context embedding + positional embedding (concatenated or added)
  - Output: predicted patch embedding

### [NEW] `models/ppad.py`
- Full model: `PPAD(patch_encoder, context_encoder, predictor)`
- `forward(image) → anomaly_scores` per patch
- `get_anomaly_map(image) → (H, W) heatmap` (upsampled via bilinear)

### [NEW] `datasets/mvtec.py`
- Standard MVTec loader: train (only normal), test (normal + anomalous)
- Returns images at 224×224 or 256×256
- Provides masks for pixel-level evaluation

### [NEW] `train.py`
- Train only the predictor (encoders frozen)
- Loss: cosine similarity loss between predicted ho and actual hp
- Epochs: ~50, Adam optimizer, LR ~1e-4
- Save checkpoint per category or one global checkpoint

### [NEW] `evaluate.py`
- Compute image-level AUROC
- Compute pixel-level AUROC (using upsampled patch scores)
- Report per-category and mean results in a table

---

## Open Questions

> [!IMPORTANT]
> **Q1: Truly zero-shot or predictor-trained?**
> Option B requires seeing normal training images to train the predictor. If you want 100% zero-shot (no data at all), we need Option C (MAE/I-JEPA backbone). Which do you prefer?

> [!IMPORTANT]
> **Q2: Patch extraction strategy**
> Two options for how to get hp:
> - (a) Run DINOv2 on the full image, take the ViT patch token at position i → avoids extra forward passes but hp includes global context
> - (b) Crop the patch and run it through the encoder independently → cleaner separation but slower
> Which is more faithful to your described idea?

> [!NOTE]
> **Q3: Predictor architecture**
> MLP (fast, simple) vs small Transformer (more expressive, attends over all patch positions)?

> [!NOTE]
> **Q4: MVTec location**
> Do you have MVTec AD already downloaded, and if so where? Or should I include a download script?

---

## Verification Plan

### Metrics
- Image-level AUROC per category + mean
- Pixel-level AUROC per category + mean  
- Compare against published baselines (WinCLIP, PatchCore, etc.)

### Visual Verification
- Plot anomaly heatmaps overlaid on test images
- Verify normal images produce low scores and anomalous ones produce high scores

### Baseline Comparison
Sanity check: with a random predictor (untrained), scores should be ~random. After training predictor, normal images should score low, anomalies should score high.
