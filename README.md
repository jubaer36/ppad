# PPAD — Patch Predictive Anomaly Detection

Zero-shot anomaly detection via **patch prediction in embedding space**.
Supports **MVTec AD**, **MVTec AD 2**, and **VisA** — train on any, evaluate on any.

## Idea

For each patch `i` in a test image:

```
hp = DINOv2( crop(image, i) )            # actual patch embedding
hi = DINOv2( image with patch i zeroed ) # context embedding
ho = Transformer_Predictor( hi, pos_i )  # predicted patch embedding
score_i = 1 - cosine_sim(hp, ho)         # anomaly score
```

Normal patches → context predicts them accurately → low score.  
Anomalous patches → context cannot predict them → high score.

## Files

| File | Purpose |
|---|---|
| `model.py` | Frozen DINOv2 encoders + Transformer predictor |
| `dataset.py` | Loaders for MVTec AD, MVTec AD 2, and VisA |
| `train.py` | Train predictor on normal images |
| `evaluate.py` | Compute image-level & pixel-level AUROC + visualizations |
| `visualize.py` | Overlay heatmaps on test images |

## Supported Datasets

| `--dataset` | Categories | Train split | Test split |
|---|---|---|---|
| `mvtec` | 15 | `train/good/` | `test/<defect>/` |
| `mvtec2` | 8 | `train/` (flat) | `test_public/good/` + `test_public/bad/` |
| `visa` | 12 | first 80% of `Data/Images/Normal/` | last 20% normal + all `Data/Images/Anomaly/` |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train predictor

**Single category:**
```bash
python train.py \
  --dataset mvtec \
  --data_path /path/to/mvtec \
  --category bottle \
  --epochs 50 \
  --batch_size 8
```

**All categories — one shared model:**
```bash
python train.py \
  --dataset visa \
  --data_path /path/to/visa \
  --category all \
  --epochs 50
```

**Custom subset — one shared model:**
```bash
python train.py \
  --dataset mvtec2 \
  --data_path /path/to/mvtec2 \
  --category sheet_metal,vial,fabric \
  --epochs 50
```

Checkpoints are saved to `checkpoints/<dataset>/<tag>/best.pt`  
where `<tag>` is the category name (single) or `all` / `cat1+cat2` (multi).

### 2. Evaluate

**Same-dataset evaluation:**
```bash
python evaluate.py \
  --dataset mvtec \
  --data_path /path/to/mvtec \
  --category bottle \
  --ckpt_dir checkpoints
```

**Evaluate all categories (auto-loads per-category or shared checkpoint):**
```bash
python evaluate.py \
  --dataset visa \
  --data_path /path/to/visa \
  --category all \
  --ckpt_dir checkpoints
```

**Cross-dataset zero-shot evaluation (train on X, test on Y):**
```bash
# Train on all MVTec, evaluate on all VisA
python train.py \
  --dataset mvtec --data_path /path/to/mvtec --category all

python evaluate.py \
  --dataset visa  --data_path /path/to/visa  --category all \
  --ckpt_dir checkpoints --ckpt_tag mvtec/all

# Train on VisA, evaluate on MVTec AD 2
python train.py \
  --dataset visa --data_path /path/to/visa --category all

python evaluate.py \
  --dataset mvtec2 --data_path /path/to/mvtec2 --category all \
  --ckpt_dir checkpoints --ckpt_tag visa/all
```

Results (AUROC + seg-F1) saved to `checkpoints/results.json`.  
Per-image visualizations (image | GT mask | anomaly map) saved to `visualizations/<dataset>/<category>/`.

### 3. Visualize heatmaps
```bash
python visualize.py \
  --data_path /path/to/mvtec \
  --category bottle \
  --num_images 20
```

## Key Hyperparameters

| Arg | Default | Notes |
|---|---|---|
| `--dataset` | `mvtec` | Dataset to train/evaluate on |
| `--patch_grid` | `4` | 4×4 = 16 patches per image |
| `--encoder` | `dinov2_vits14` | Swap to `dinov2_vitb14` for stronger features |
| `--epochs` | `50` | ~10 min per category on a single GPU |
| `--lr` | `1e-4` | AdamW with cosine decay |
| `--ckpt_tag` | _(auto)_ | Override checkpoint path for cross-dataset eval |

## Architecture

- **Encoder**: Frozen DINOv2 ViT-S/14 (embed_dim=384). CLS token used as embedding.
- **Predictor**: 2-layer Transformer encoder on a 2-token sequence `[context | query]`.  
  The query token output is the predicted patch embedding.
- **Loss**: Cosine similarity loss on normal training images only.
