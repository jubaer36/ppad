"""
evaluate.py - Evaluate PPAD on MVTec test set.
Computes image-level and pixel-level AUROC and saves per-image
side-by-side visualizations: [Image | GT Mask | Anomaly Map].

Example:
  python evaluate.py --data_path /datasets/mvtec --category bottle \
                     --ckpt_dir checkpoints --vis_dir visualizations
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset, MVTEC_CATEGORIES
from model import PPAD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate PPAD on MVTec')
    p.add_argument('--data_path', required=True)
    p.add_argument('--category',  required=True, help='Category name or "all"')
    p.add_argument('--ckpt_dir',  default='checkpoints',
                   help='Root checkpoint directory (expects <ckpt_dir>/<category>/best.pt)')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--vis_dir',    default='visualizations',
                   help='Root directory for per-image artifacts')
    p.add_argument('--no_vis',     action='store_true',
                   help='Skip saving visualization images')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

_IMG_MEAN = np.array([0.485, 0.456, 0.406])
_IMG_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(tensor):
    """[3, H, W] normalized tensor → [H, W, 3] float32 in [0, 1]."""
    img = tensor.permute(1, 2, 0).cpu().float().numpy()
    return np.clip(img * _IMG_STD + _IMG_MEAN, 0.0, 1.0)


def save_visualization(img_tensor, mask_tensor, heatmap_tensor,
                       label: int, img_score: float, save_path: Path):
    """
    Save a 3-panel figure:
      Left   : original image (denormalized)
      Middle : ground-truth mask (gray)
      Right  : anomaly heatmap (hot colormap) blended over the image

    Parameters
    ----------
    img_tensor   : [3, H, W]  normalized image
    mask_tensor  : [H, W]     GT binary mask (0/1)
    heatmap_tensor: [H, W]    raw anomaly scores
    label        : 0=normal, 1=anomaly
    img_score    : scalar image-level anomaly score
    save_path    : output .png path
    """
    img_np  = _denorm(img_tensor)                          # [H, W, 3]
    mask_np = mask_tensor.cpu().numpy()                    # [H, W]
    heat_np = heatmap_tensor.cpu().numpy()                 # [H, W]

    # Normalize heatmap to [0, 1] for display
    h_min, h_max = heat_np.min(), heat_np.max()
    heat_norm = (heat_np - h_min) / (h_max - h_min + 1e-8)

    # Blend heatmap over image (alpha composite)
    cmap   = cm.get_cmap('hot')
    heat_rgb = cmap(heat_norm)[..., :3]                    # [H, W, 3]
    blend    = 0.55 * img_np + 0.45 * heat_rgb             # [H, W, 3]
    blend    = np.clip(blend, 0.0, 1.0)

    status = 'ANOMALY' if label == 1 else 'NORMAL'
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={'wspace': 0.05})
    fig.patch.set_facecolor('#1a1a2e')

    # Panel 1 — image
    axes[0].imshow(img_np)
    axes[0].set_title('Image', color='white', fontsize=11, pad=6)
    axes[0].axis('off')

    # Panel 2 — GT mask
    axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Mask', color='white', fontsize=11, pad=6)
    axes[1].axis('off')

    # Panel 3 — anomaly map blended
    axes[2].imshow(blend)
    axes[2].set_title(
        f'Anomaly Map  [{status}]  score={img_score:.3f}',
        color='tomato' if label == 1 else 'lightgreen',
        fontsize=11, pad=6,
    )
    axes[2].axis('off')

    # Colorbar for anomaly map
    sm = plt.cm.ScalarMappable(cmap='hot', norm=Normalize(vmin=h_min, vmax=h_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluate one category
# ---------------------------------------------------------------------------

def evaluate_category(args, category: str) -> dict:
    device = torch.device(args.device)

    # Load checkpoint
    ckpt_path = Path(args.ckpt_dir) / category / 'best.pt'
    if not ckpt_path.exists():
        print(f'  [SKIP] No checkpoint found: {ckpt_path}')
        return {}

    ckpt = torch.load(ckpt_path, map_location=device)

    model = PPAD(
        patch_grid   = ckpt['patch_grid'],
        img_size     = ckpt['img_size'],
        encoder_name = ckpt['encoder'],
    ).to(device)
    model.predictor.load_state_dict(ckpt['predictor'])
    model.eval()

    # Dataset — batch_size=1 so each image gets its own artifact file
    dataset = MVTecDataset(args.data_path, category, split='test',
                           img_size=ckpt['img_size'])
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=4, pin_memory=True)

    vis_dir = Path(args.vis_dir) / category
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f'  Saving visualizations → {vis_dir}')

    img_scores, img_labels = [], []
    px_scores,  px_labels  = [], []

    with torch.no_grad():
        for sample_idx, (images, masks, labels) in enumerate(
                tqdm(loader, desc=f'  {category}')):
            images = images.to(device)          # [1, 3, H, W]

            # Anomaly map [1, 1, H, W]
            heatmap = model.get_anomaly_map(images)

            # Image-level score = max over spatial positions
            img_score = heatmap.flatten(1).max(dim=1).values  # [1]

            img_scores.append(img_score.cpu().numpy())
            img_labels.append(labels.numpy())

            # Pixel-level
            px_scores.append(heatmap.squeeze(1).cpu().numpy())   # [1, H, W]
            px_labels.append(masks.squeeze(1).cpu().numpy())     # [1, H, W]

            # ---- Per-image visualization --------------------------------
            if not args.no_vis:
                label_int  = int(labels[0])
                score_val  = float(img_score[0])
                status_str = 'anomaly' if label_int == 1 else 'normal'
                fname      = f'{sample_idx:04d}_{status_str}.png'

                save_visualization(
                    img_tensor     = images[0],
                    mask_tensor    = masks[0, 0],
                    heatmap_tensor = heatmap[0, 0],
                    label          = label_int,
                    img_score      = score_val,
                    save_path      = vis_dir / fname,
                )

    img_scores = np.concatenate(img_scores)
    img_labels = np.concatenate(img_labels)
    px_scores  = np.concatenate(px_scores).ravel()
    px_labels  = np.concatenate(px_labels).ravel().astype(int)

    img_auroc = roc_auc_score(img_labels, img_scores) * 100
    px_auroc  = roc_auc_score(px_labels,  px_scores)  * 100 if px_labels.sum() > 0 else float('nan')

    # Seg-F1: find optimal threshold per category from GT pixel labels
    if px_labels.sum() > 0:
        precision, recall, thresholds = precision_recall_curve(px_labels, px_scores)
        f1_per_thresh = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx  = int(np.argmax(f1_per_thresh))
        seg_f1    = float(f1_per_thresh[best_idx]) * 100
        threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    else:
        seg_f1    = float('nan')
        threshold = float('nan')

    print(f'  {category:<15} img-AUROC: {img_auroc:.1f}%   px-AUROC: {px_auroc:.1f}%   seg-F1: {seg_f1:.1f}%  (thr={threshold:.4f})')
    return {'img_auroc': img_auroc, 'px_auroc': px_auroc, 'seg_f1': seg_f1, 'threshold': threshold}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    categories = MVTEC_CATEGORIES if args.category == 'all' else [args.category]

    print(f'\n{"="*60}')
    print(f'  PPAD Evaluation  |  ckpt_dir: {args.ckpt_dir}')
    print(f'{"="*60}')

    results = {}
    for cat in categories:
        results[cat] = evaluate_category(args, cat)

    # Summary
    valid = {k: v for k, v in results.items() if v}
    if len(valid) > 1:
        mean_img = np.mean([v['img_auroc'] for v in valid.values()])
        mean_px  = np.mean([v['px_auroc'] for v in valid.values()
                            if not np.isnan(v['px_auroc'])])
        mean_f1  = np.mean([v['seg_f1']   for v in valid.values()
                            if not np.isnan(v['seg_f1'])])
        print(f'\n  Mean img-AUROC: {mean_img:.1f}%   Mean px-AUROC: {mean_px:.1f}%   Mean seg-F1: {mean_f1:.1f}%')

    # Save results JSON
    out = Path(args.ckpt_dir) / 'results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved → {out}')


if __name__ == '__main__':
    main()
