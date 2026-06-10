"""
train.py - Train the Transformer predictor on normal (good) images.
The DINOv2 encoder is always frozen; only the predictor is updated.

Single category:
  python train.py --dataset mvtec --data_path /data/mvtec --category bottle

All categories in a dataset (one shared model):
  python train.py --dataset visa  --data_path /data/visa  --category all

Custom subset:
  python train.py --dataset mvtec2 --data_path /data/mvtec2 --category sheet_metal,vial

Checkpoints are saved to:  <output_dir>/<dataset>/<tag>/best.pt
  where <tag> is the category name (single) or 'all' / 'cat1+cat2' (multi).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_dataset, ALL_CATEGORIES
from model import PPAD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train PPAD predictor')
    p.add_argument('--dataset',      default='mvtec',
                   help='Dataset name: mvtec | mvtec2 | visa')
    p.add_argument('--data_path',    required=True,  help='Path to dataset root')
    p.add_argument('--category',     required=True,
                   help='Category name, "all", or comma-separated list (e.g. bottle,cable)')
    p.add_argument('--output_dir',   default='checkpoints', help='Where to save checkpoints')
    p.add_argument('--patch_grid',   type=int,   default=4,     help='Patches per dim (4→16 patches)')
    p.add_argument('--img_size',     type=int,   default=224)
    p.add_argument('--encoder',      default='dinov2_vits14')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training for one category
# ---------------------------------------------------------------------------

def train_category(args, dataset_name: str, category: str):
    print(f'\n{"="*60}')
    print(f'  Training [{dataset_name}]: {category}')
    print(f'{"="*60}')

    device = torch.device(args.device)

    # Dataset
    dataset = make_dataset(dataset_name, args.data_path, category,
                           split='train', img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    print(f'  Training images: {len(dataset)}')

    # Model — encoder is frozen inside PPAD
    model = PPAD(patch_grid=args.patch_grid, img_size=args.img_size,
                 encoder_name=args.encoder).to(device)

    # Only optimise predictor parameters
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    out_dir   = Path(args.output_dir) / dataset_name / category
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, _, _ in tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            images = images.to(device)
            B, N   = images.shape[0], model.num_patches

            # ---- encode (no grad, encoder is frozen) ----
            hp = model._encode_all_patches(images)    # [B, N, D]
            hi = model._encode_all_contexts(images)   # [B, N, D]

            # ---- predict (differentiable) ----
            hi_flat  = hi.reshape(B * N, -1)
            idx_flat = torch.arange(N, device=device).repeat(B)
            ho_flat  = model.predictor(hi_flat, idx_flat)            # [B*N, D]

            ho = ho_flat.view(B, N, -1)                              # [B, N, D]

            # Cosine-similarity loss on normalized vectors
            loss = (1.0 - F.cosine_similarity(ho, hp, dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f'  Epoch {epoch:3d} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                'epoch':       epoch,
                'loss':        best_loss,
                'predictor':   model.predictor.state_dict(),
                'patch_grid':  args.patch_grid,
                'img_size':    args.img_size,
                'encoder':     args.encoder,
                'dataset':     dataset_name,
                'categories':  [category],
            }
            torch.save(ckpt, out_dir / 'best.pt')

    print(f'  Saved best checkpoint → {out_dir / "best.pt"}  (loss={best_loss:.4f})')


# ---------------------------------------------------------------------------
# Training on a combined multi-category dataset (one shared model)
# ---------------------------------------------------------------------------

def train_all(args, dataset_name: str, categories: list):
    all_cats = ALL_CATEGORIES[dataset_name]
    tag = 'all' if set(categories) == set(all_cats) else '+'.join(categories)
    print(f'\n{"="*60}')
    print(f'  Training ONE shared [{dataset_name}] model on: {categories}')
    print(f'{"="*60}')

    device = torch.device(args.device)

    # Combined dataset via factory
    dataset = make_dataset(dataset_name, args.data_path, categories,
                           split='train', img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    print(f'  Total training images: {len(dataset)}')

    # One model for all categories
    model = PPAD(patch_grid=args.patch_grid, img_size=args.img_size,
                 encoder_name=args.encoder).to(device)

    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    out_dir   = Path(args.output_dir) / dataset_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, _, _ in tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            images = images.to(device)
            B, N   = images.shape[0], model.num_patches

            hp = model._encode_all_patches(images)    # [B, N, D]
            hi = model._encode_all_contexts(images)   # [B, N, D]

            hi_flat  = hi.reshape(B * N, -1)
            idx_flat = torch.arange(N, device=device).repeat(B)
            ho_flat  = model.predictor(hi_flat, idx_flat)

            ho   = ho_flat.view(B, N, -1)
            loss = (1.0 - F.cosine_similarity(ho, hp, dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f'  Epoch {epoch:3d} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                'epoch':      epoch,
                'loss':       best_loss,
                'predictor':  model.predictor.state_dict(),
                'patch_grid': args.patch_grid,
                'img_size':   args.img_size,
                'encoder':    args.encoder,
                'dataset':    dataset_name,
                'categories': categories,
            }
            torch.save(ckpt, out_dir / 'best.pt')

    print(f'  Saved best checkpoint → {out_dir / "best.pt"}  (loss={best_loss:.4f})')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dataset_name = args.dataset

    # Resolve categories
    all_cats = ALL_CATEGORIES[dataset_name]
    if args.category == 'all':
        categories = all_cats
    else:
        categories = [c.strip() for c in args.category.split(',')]

    if len(categories) == 1:
        # Single category → per-category checkpoint
        train_category(args, dataset_name, categories[0])
    else:
        # Multiple categories → one shared model
        train_all(args, dataset_name, categories)


if __name__ == '__main__':
    main()
