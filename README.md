# PPAD — Patch Predictive Anomaly Detection

Zero-shot anomaly detection on MVTec AD via **patch prediction in embedding space**.

## Idea

For each patch `i` in a test image:

```
hp = DINOv2( crop(image, i) )           # actual patch embedding
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
| `dataset.py` | MVTec AD loader (train/test splits) |
| `train.py` | Train predictor on normal images |
| `evaluate.py` | Compute image-level & pixel-level AUROC |
| `visualize.py` | Overlay heatmaps on test images |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train predictor
```bash
python train.py \
  --data_path /path/to/mvtec \
  --category bottle \
  --epochs 50 \
  --batch_size 8
# Use --category all to train all 15 categories sequentially
```

Checkpoints saved to `checkpoints/<category>/best.pt`.

### 2. Evaluate
```bash
python evaluate.py \
  --data_path /path/to/mvtec \
  --category bottle \
  --ckpt_dir checkpoints
# Results saved to checkpoints/results.json
```

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
| `--patch_grid` | `4` | 4×4 = 16 patches per image |
| `--encoder` | `dinov2_vits14` | Swap to `dinov2_vitb14` for stronger features |
| `--epochs` | `50` | ~10 min per category on a single GPU |
| `--lr` | `1e-4` | AdamW with cosine decay |

## Architecture

- **Encoder**: Frozen DINOv2 ViT-S/14 (embed_dim=384). CLS token used as embedding.
- **Predictor**: 2-layer Transformer encoder on a 2-token sequence `[context | query]`.  
  The query token output is the predicted patch embedding.
- **Loss**: Cosine similarity loss on normal training images only.
