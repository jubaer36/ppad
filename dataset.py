"""
dataset.py - MVTec AD dataset loader
"""
from typing import List
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def get_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


class MVTecDataset(Dataset):
    """
    split='train' : only normal (good) images — used to train the predictor.
    split='test'  : all test images with labels and optional GT masks.
    """

    def __init__(self, root: str, category: str, split: str = 'train', img_size: int = 224):
        self.root = Path(root)
        self.transform = get_transform(img_size)
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.samples = []   # (img_path, mask_path_or_None, label)
        self._load(split, category)

    def _load(self, split, category):
        split_dir = self.root / category / split
        if split == 'train':
            good_dir = split_dir / 'good'
            for p in sorted(good_dir.glob('*')):
                if p.suffix.lower() in ('.png', '.jpg', '.bmp'):
                    self.samples.append((p, None, 0))
        else:
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == 'good' else 1
                for p in sorted(defect_dir.glob('*')):
                    if p.suffix.lower() not in ('.png', '.jpg', '.bmp'):
                        continue
                    mask = None
                    if label == 1:
                        mask = (self.root / category / 'ground_truth'
                                / defect_dir.name / (p.stem + '_mask.png'))
                        if not mask.exists():
                            mask = None
                    self.samples.append((p, mask, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        H, W = image.shape[1], image.shape[2]
        if mask_path is not None:
            mask = self.mask_transform(Image.open(mask_path).convert('L'))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, H, W)
        return image, mask, label


class MultiCategoryDataset(Dataset):
    """
    Combines normal (good) training images from multiple MVTec categories
    into a single dataset so that one shared model can be trained on all.

    Only supports split='train' (normal images only).
    """

    def __init__(self, root: str, categories: List[str], img_size: int = 224):
        self.samples: list = []
        for cat in categories:
            cat_ds = MVTecDataset(root, cat, split='train', img_size=img_size)
            self.samples.extend(cat_ds.samples)
        # Reuse the transform from the last per-category dataset
        self.transform       = cat_ds.transform
        self.mask_transform  = cat_ds.mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        H, W = image.shape[1], image.shape[2]
        mask = torch.zeros(1, H, W)   # train split → always normal, no mask
        return image, mask, label
