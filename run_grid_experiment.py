"""
run_grid_experiment.py
======================
Comparative experiment: single-scale 8×8 grid  vs  multi-scale 4×4+8×8+16×16 fusion.

For each configuration the script:
  1. Trains the PPAD predictor on normal (good) images.
  2. Evaluates on the test split (image-AUROC, pixel-AUROC, seg-F1).
  3. Saves two kinds of visualisations per test image:
       a) Side-by-side: [Image | Ground-Truth Mask | Anomaly-Map Overlay]
       b) Raw un-upsampled patch-score grid next to the GT mask
          downsampled to the same resolution.
  4. Writes a combined JSON report of all metrics.

Usage (single category):
  python run_grid_experiment.py \
      --dataset mvtec --data_path /data/mvtec --category bottle

Usage (all categories):
  python run_grid_experiment.py \
      --dataset mvtec --data_path /data/mvtec --category all

Results go to:
  <output_dir>/
    single_8x8/
      <dataset>/<category>/
        best.pt                        (checkpoint)
        results.json                   (per-category metrics)
        vis/
          0000_anomaly.png             (side-by-side)
          0000_anomaly_patchgrid_8.png (raw grid + downsampled mask)
    multi_4_8_16/
      <dataset>/<category>/
        best.pt
        results.json
        vis/
          0000_anomaly.png
          0000_anomaly_patchgrid_4.png
          0000_anomaly_patchgrid_8.png
          0000_anomaly_patchgrid_16.png
    comparison_results.json            (combined report)
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_dataset, ALL_CATEGORIES
from model import PPAD


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Grid-scale experiment: single 8×8 vs multi-scale 4+8+16 fusion')
    p.add_argument('--dataset',    default='mvtec',
                   help='Dataset name: mvtec | mvtec2 | visa')
    p.add_argument('--data_path',  required=True, help='Path to dataset root')
    p.add_argument('--category',   required=True,
                   help='Category name, "all", or comma-separated list')
    p.add_argument('--output_dir', default='experiments/grid_comparison',
                   help='Root output directory for both configurations')
    p.add_argument('--img_size',   type=int, default=224)
    p.add_argument('--encoder',    default='dinov2_vits14')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max_vis',    type=int, default=50,
                   help='Max visualisations per category (0 = unlimited)')
    p.add_argument('--no_vis',     action='store_true',
                   help='Skip saving visualisation images')
    return p.parse_args()


# ===================================================================
# Visualisation helpers
# ===================================================================

_IMG_MEAN = np.array([0.485, 0.456, 0.406])
_IMG_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(tensor):
    """[3, H, W] normalised tensor → [H, W, 3] float32 in [0, 1]."""
    img = tensor.permute(1, 2, 0).cpu().float().numpy()
    return np.clip(img * _IMG_STD + _IMG_MEAN, 0.0, 1.0)


def save_side_by_side(img_tensor, mask_tensor, heatmap_tensor,
                      label: int, img_score: float, save_path: Path,
                      config_name: str = ''):
    """
    3-panel figure: [Image | GT Mask | Anomaly Heatmap blended over image].
    """
    img_np  = _denorm(img_tensor)
    mask_np = mask_tensor.cpu().numpy()
    heat_np = heatmap_tensor.cpu().numpy()

    h_min, h_max = heat_np.min(), heat_np.max()
    heat_norm = (heat_np - h_min) / (h_max - h_min + 1e-8)

    cmap     = plt.get_cmap('hot')
    heat_rgb = cmap(heat_norm)[..., :3]
    blend    = np.clip(0.55 * img_np + 0.45 * heat_rgb, 0.0, 1.0)

    status = 'ANOMALY' if label == 1 else 'NORMAL'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={'wspace': 0.05})
    fig.patch.set_facecolor('#1a1a2e')

    if config_name:
        fig.suptitle(config_name, color='#7ec8e3', fontsize=12,
                     fontweight='bold', y=0.98)

    # Panel 1 — image
    axes[0].imshow(img_np)
    axes[0].set_title('Image', color='white', fontsize=11, pad=6)
    axes[0].axis('off')

    # Panel 2 — GT mask
    axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Mask', color='white', fontsize=11, pad=6)
    axes[1].axis('off')

    # Panel 3 — anomaly overlay
    axes[2].imshow(blend)
    axes[2].set_title(
        f'Anomaly Map  [{status}]  score={img_score:.3f}',
        color='tomato' if label == 1 else 'lightgreen',
        fontsize=11, pad=6,
    )
    axes[2].axis('off')

    sm = plt.cm.ScalarMappable(cmap='hot', norm=Normalize(vmin=h_min, vmax=h_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_patch_grid_vis(patch_scores, mask_tensor, g: int,
                        label: int, img_score: float, save_path: Path,
                        config_name: str = ''):
    """
    2-panel: [Raw patch-score grid (g×g) | GT mask downsampled to g×g].
    """
    score_grid = patch_scores.cpu().float().numpy().reshape(g, g)

    mask_hw = mask_tensor.cpu().float()
    mask_4d = mask_hw.unsqueeze(0).unsqueeze(0)
    mask_ds = F.adaptive_avg_pool2d(mask_4d, (g, g)).squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2),
                             gridspec_kw={'wspace': 0.30})
    fig.patch.set_facecolor('#1a1a2e')

    status       = 'ANOMALY' if label == 1 else 'NORMAL'
    status_color = 'tomato' if label == 1 else 'lightgreen'

    if config_name:
        fig.suptitle(config_name, color='#7ec8e3', fontsize=12,
                     fontweight='bold', y=0.98)

    # Panel 1 — raw patch scores
    im0 = axes[0].imshow(score_grid, cmap='hot', interpolation='nearest')
    axes[0].set_title(
        f'Patch Scores ({g}×{g})  [{status}]\nscore={img_score:.3f}',
        color=status_color, fontsize=10, pad=6,
    )
    axes[0].set_xticks(range(g))
    axes[0].set_yticks(range(g))
    axes[0].tick_params(colors='white', labelsize=7)
    # Annotate cells (skip text if grid is very large)
    if g <= 16:
        for r in range(g):
            for c in range(g):
                val = score_grid[r, c]
                tc = 'black' if val > (score_grid.max() + score_grid.min()) / 2 else 'white'
                axes[0].text(c, r, f'{val:.2f}', ha='center', va='center',
                             fontsize=max(4, 9 - g // 3), color=tc)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    # Panel 2 — downsampled GT mask
    im1 = axes[1].imshow(mask_ds, cmap='gray', vmin=0.0, vmax=1.0,
                         interpolation='nearest')
    axes[1].set_title(f'GT Mask ↓{g}×{g}  (avg-pool)',
                      color='white', fontsize=10, pad=6)
    axes[1].set_xticks(range(g))
    axes[1].set_yticks(range(g))
    axes[1].tick_params(colors='white', labelsize=7)
    if g <= 16:
        for r in range(g):
            for c in range(g):
                val = mask_ds[r, c]
                tc = 'black' if val > 0.5 else 'white'
                axes[1].text(c, r, f'{val:.2f}', ha='center', va='center',
                             fontsize=max(4, 9 - g // 3), color=tc)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ===================================================================
# Training
# ===================================================================

def train_model(args, grids: list, dataset_name: str, category: str,
                out_dir: Path):
    """Train a PPAD model with the given grid configuration. Returns model."""
    print(f'\n{"="*60}')
    print(f'  Training  grids={grids}  [{dataset_name}] {category}')
    print(f'{"="*60}')

    device = torch.device(args.device)

    dataset = make_dataset(dataset_name, args.data_path, category,
                           split='train', img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    print(f'  Training images: {len(dataset)}')

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model = PPAD(patch_grids=grids, img_size=args.img_size,
                 encoder_name=args.encoder).to(device)

    optimizer = torch.optim.AdamW(model.predictors.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs)

    best_loss = float('inf')
    ckpt_dir  = out_dir / dataset_name / category
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, _, _ in tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}',
                                 leave=False):
            images = images.to(device)
            outputs = model(images)

            loss = 0.0
            for g, (hp, ho) in outputs.items():
                loss += (1.0 - F.cosine_similarity(ho, hp, dim=-1)).mean()
            loss /= len(outputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictors.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f'  Epoch {epoch:3d} | loss: {avg_loss:.4f} | '
              f'lr: {scheduler.get_last_lr()[0]:.2e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                'epoch':       epoch,
                'loss':        best_loss,
                'predictors':  model.predictors.state_dict(),
                'patch_grids': model.patch_grids,
                'img_size':    args.img_size,
                'encoder':     args.encoder,
                'dataset':     dataset_name,
                'categories':  [category],
            }
            torch.save(ckpt, ckpt_dir / 'best.pt')

    print(f'  Best checkpoint → {ckpt_dir / "best.pt"}  (loss={best_loss:.4f})')

    # Reload best weights
    best_state = torch.load(ckpt_dir / 'best.pt', map_location=device)
    model.predictors.load_state_dict(best_state['predictors'])
    return model


# ===================================================================
# Evaluation + Visualisation
# ===================================================================

def evaluate_model(args, model, dataset_name: str, category: str,
                   out_dir: Path, config_name: str) -> dict:
    """
    Evaluate a trained model. Saves side-by-side and patch-grid
    visualisations and returns a dict of metrics.
    """
    print(f'\n{"-"*60}')
    print(f'  Evaluating  [{config_name}]  {dataset_name}/{category}')
    print(f'{"-"*60}')

    device = next(model.parameters()).device
    model.eval()

    dataset = make_dataset(dataset_name, args.data_path, category,
                           split='test', img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=4, pin_memory=True)

    vis_dir = out_dir / dataset_name / category / 'vis'
    do_vis  = not args.no_vis
    if do_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f'  Saving visualisations → {vis_dir}')

    img_scores, img_labels = [], []
    px_scores,  px_labels  = [], []

    vis_count = 0

    with torch.no_grad():
        for sample_idx, (images, masks, labels) in enumerate(
                tqdm(loader, desc=f'  [{dataset_name}] {category}')):
            images = images.to(device)

            outputs = model(images)
            heatmap = outputs['fused']                           # [1,1,H,W]
            img_score = heatmap.flatten(1).max(dim=1).values     # [1]

            img_scores.append(img_score.cpu().numpy())
            img_labels.append(labels.numpy())
            px_scores.append(heatmap.squeeze(1).cpu().numpy())
            px_labels.append(masks.squeeze(1).cpu().numpy())

            # ---------- visualisations ----------
            if do_vis and (args.max_vis == 0 or vis_count < args.max_vis):
                label_int  = int(labels[0])
                score_val  = float(img_score[0])
                status_str = 'anomaly' if label_int == 1 else 'normal'
                fname      = f'{sample_idx:04d}_{status_str}'

                # (a) Side-by-side: image | GT mask | anomaly overlay
                save_side_by_side(
                    img_tensor     = images[0],
                    mask_tensor    = masks[0, 0],
                    heatmap_tensor = heatmap[0, 0],
                    label          = label_int,
                    img_score      = score_val,
                    save_path      = vis_dir / f'{fname}.png',
                    config_name    = config_name,
                )

                # (b) Raw patch-score grid(s) vs downsampled GT mask
                for g in model.patch_grids:
                    save_patch_grid_vis(
                        patch_scores = outputs[g][0],   # [N]
                        mask_tensor  = masks[0, 0],
                        g            = g,
                        label        = label_int,
                        img_score    = score_val,
                        save_path    = vis_dir / f'{fname}_patchgrid_{g}.png',
                        config_name  = config_name,
                    )

                vis_count += 1

    # ---------- metrics ----------
    img_scores = np.concatenate(img_scores)
    img_labels = np.concatenate(img_labels)
    px_scores  = np.concatenate(px_scores).ravel()
    px_labels  = np.concatenate(px_labels).ravel().astype(int)

    img_auroc = roc_auc_score(img_labels, img_scores) * 100
    px_auroc  = (roc_auc_score(px_labels, px_scores) * 100
                 if px_labels.sum() > 0 else float('nan'))

    if px_labels.sum() > 0:
        precision, recall, thresholds = precision_recall_curve(px_labels, px_scores)
        f1_arr    = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx  = int(np.argmax(f1_arr))
        seg_f1    = float(f1_arr[best_idx]) * 100
        threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    else:
        seg_f1    = float('nan')
        threshold = float('nan')

    print(f'  {category:<15} img-AUROC: {img_auroc:.1f}%   '
          f'px-AUROC: {px_auroc:.1f}%   seg-F1: {seg_f1:.1f}%  '
          f'(thr={threshold:.4f})')

    metrics = {
        'img_auroc': img_auroc,
        'px_auroc':  px_auroc,
        'seg_f1':    seg_f1,
        'threshold': threshold,
    }

    # Save per-config, per-category results
    results_path = out_dir / dataset_name / category / 'results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ===================================================================
# Run one full configuration (train + eval)
# ===================================================================

def run_config(args, config_name: str, grids: list, dataset_name: str,
               categories: list, output_root: Path) -> dict:
    """Train and evaluate for all requested categories. Returns results dict."""
    out_dir = output_root / config_name
    results = {}

    for cat in categories:
        # Train
        model = train_model(args, grids, dataset_name, cat, out_dir)
        # Evaluate
        metrics = evaluate_model(args, model, dataset_name, cat,
                                 out_dir, config_name)
        results[cat] = metrics

        # Free GPU memory before next category
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ===================================================================
# Summary report
# ===================================================================

def compute_means(results: dict) -> dict:
    """Compute mean metrics across categories (ignoring NaN)."""
    if not results:
        return {}
    vals = {}
    for metric in ('img_auroc', 'px_auroc', 'seg_f1'):
        arr = [v[metric] for v in results.values()
               if not np.isnan(v.get(metric, float('nan')))]
        vals[f'mean_{metric}'] = float(np.mean(arr)) if arr else float('nan')
    return vals


def print_summary_table(all_results: dict, categories: list):
    """Print a formatted comparison table to stdout."""
    configs = list(all_results.keys())
    header = f'{"Category":<18}'
    for cfg in configs:
        header += f'  {cfg:>20} (img / px / F1)'
    print(f'\n{"="*80}')
    print('  COMPARISON SUMMARY')
    print(f'{"="*80}')
    print(f'  {header}')
    print(f'  {"-"*len(header)}')

    for cat in categories:
        row = f'  {cat:<18}'
        for cfg in configs:
            m = all_results[cfg].get(cat, {})
            img = m.get('img_auroc', float('nan'))
            px  = m.get('px_auroc', float('nan'))
            f1  = m.get('seg_f1', float('nan'))
            row += f'  {img:6.1f} / {px:6.1f} / {f1:6.1f}      '
        print(row)

    # Means
    print(f'  {"-"*len(header)}')
    row = f'  {"MEAN":<18}'
    for cfg in configs:
        means = compute_means(all_results[cfg])
        row += (f'  {means.get("mean_img_auroc",0):6.1f} / '
                f'{means.get("mean_px_auroc",0):6.1f} / '
                f'{means.get("mean_seg_f1",0):6.1f}      ')
    print(row)
    print(f'{"="*80}\n')


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()
    dataset_name = args.dataset
    output_root  = Path(args.output_dir)

    # Resolve categories
    all_cats = ALL_CATEGORIES[dataset_name]
    if args.category == 'all':
        categories = all_cats
    else:
        categories = [c.strip() for c in args.category.split(',')]

    # Two experiment configurations
    CONFIGS = {
        'single_8x8':    [8],
        'multi_4_8_16':  [4, 8, 16],
    }

    all_results = {}
    for config_name, grids in CONFIGS.items():
        print(f'\n{"#"*70}')
        print(f'#  CONFIG: {config_name}   grids={grids}')
        print(f'{"#"*70}')
        results = run_config(args, config_name, grids, dataset_name,
                             categories, output_root)
        all_results[config_name] = results

    # Summary
    print_summary_table(all_results, categories)

    # Save combined report
    report = {}
    for cfg_name, results in all_results.items():
        report[cfg_name] = {
            'grids':      CONFIGS[cfg_name],
            'categories': results,
            'means':      compute_means(results),
        }
    report_path = output_root / 'comparison_results.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Combined report → {report_path}')


if __name__ == '__main__':
    main()
