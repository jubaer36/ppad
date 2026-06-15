"""
visualize.py - Overlay anomaly heatmaps on test images for qualitative inspection.

Example:
  python visualize.py --data_path /datasets/mvtec --category bottle \
                      --ckpt_dir checkpoints --num_images 10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MVTecDataset
from model import PPAD

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """[3, H, W] normalized tensor → [H, W, 3] uint8 numpy."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMG_STD + IMG_MEAN
    return np.clip(img, 0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',  required=True)
    p.add_argument('--category',   required=True)
    p.add_argument('--ckpt_dir',   default='checkpoints')
    p.add_argument('--output_dir', default='visualizations')
    p.add_argument('--num_images', type=int, default=8)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt_dir) / args.category / 'best.pt'
    ckpt      = torch.load(ckpt_path, map_location=device)

    model = PPAD(patch_grids=ckpt.get('patch_grids', [4, 8, 16]),
                 img_size=ckpt['img_size'],
                 encoder_name=ckpt['encoder']).to(device)
    model.predictors.load_state_dict(ckpt['predictors'])
    model.eval()

    dataset = MVTecDataset(args.data_path, args.category, split='test',
                           img_size=ckpt['img_size'])
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    out_dir = Path(args.output_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    shown = 0
    with torch.no_grad():
        for i, (images, masks, labels) in enumerate(loader):
            if shown >= args.num_images:
                break
            images = images.to(device)
            heatmap = model.get_anomaly_map(images)  # [1, 1, H, W]

            img_np  = denormalize(images[0])                   # [H, W, 3]
            heat_np = heatmap[0, 0].cpu().numpy()              # [H, W]
            mask_np = masks[0, 0].cpu().numpy()                # [H, W]
            label   = int(labels[0])

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np);    axes[0].set_title('Image');         axes[0].axis('off')
            axes[1].imshow(heat_np, cmap='hot'); axes[1].set_title(
                f'Anomaly Map  ({"ANOMALY" if label else "NORMAL"})');     axes[1].axis('off')
            axes[2].imshow(mask_np, cmap='gray'); axes[2].set_title('GT Mask'); axes[2].axis('off')

            plt.tight_layout()
            fig.savefig(out_dir / f'{i:04d}_label{label}.png', dpi=120)
            plt.close(fig)
            shown += 1

    print(f'Saved {shown} visualization(s) → {out_dir}')


if __name__ == '__main__':
    main()
