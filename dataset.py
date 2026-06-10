"""
dataset.py - Dataset loaders for MVTec AD, MVTec AD 2, and VisA.

All datasets expose the same __getitem__ contract:
    image : [3, H, W]  normalized float tensor
    mask  : [1, H, W]  binary float tensor  (0 = normal, 1 = anomaly)
    label : int        0 = normal, 1 = anomaly

Use make_dataset() to get the right loader by name.
"""
from typing import List, Union
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as T

MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]

MVTEC2_CATEGORIES = [
    'sheet_metal', 'vial', 'fabric', 'fruit_jelly',
    'rice_bag', 'wallplugs', 'walnuts', 'capacitor',
]

VISA_CATEGORIES = [
    'capsules', 'candle', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
    'macaroni1', 'macaroni2', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
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


# ---------------------------------------------------------------------------
# MVTec AD 2
# ---------------------------------------------------------------------------

class MVTec2Dataset(Dataset):
    """
    MVTec AD 2 dataset loader.

    Directory layout expected:
      <root>/<category>/train/             normal images (flat)
      <root>/<category>/test_public/good/  normal test images
      <root>/<category>/test_public/bad/   anomalous test images
      <root>/<category>/ground_truth/bad/  pixel masks (*_mask.png or *.png)

    split='train' → normal training images only.
    split='test'  → test_public/good (label 0) + test_public/bad (label 1).
    """

    _EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    def __init__(self, root: str, category: str, split: str = 'train', img_size: int = 224):
        self.root = Path(root)
        self.transform = get_transform(img_size)
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.samples: list = []
        self._load(split, category)

    def _load(self, split, category):
        cat_dir = self.root / category
        if split == 'train':
            for p in sorted((cat_dir / 'train').glob('*')):
                if p.suffix.lower() in self._EXTS:
                    self.samples.append((p, None, 0))
        else:
            good_dir = cat_dir / 'test_public' / 'good'
            if good_dir.exists():
                for p in sorted(good_dir.glob('*')):
                    if p.suffix.lower() in self._EXTS:
                        self.samples.append((p, None, 0))
            bad_dir = cat_dir / 'test_public' / 'bad'
            gt_dir  = cat_dir / 'ground_truth' / 'bad'
            if bad_dir.exists():
                for p in sorted(bad_dir.glob('*')):
                    if p.suffix.lower() not in self._EXTS:
                        continue
                    mask = None
                    for candidate_name in (p.stem + '_mask.png', p.stem + '.png', p.name):
                        cand = gt_dir / candidate_name
                        if cand.exists():
                            mask = cand
                            break
                    self.samples.append((p, mask, 1))

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


# ---------------------------------------------------------------------------
# VisA
# ---------------------------------------------------------------------------

class VisADataset(Dataset):
    """
    VisA (Visual Anomaly) dataset loader.

    Directory layout expected:
      <root>/<category>/Data/Images/Normal/   normal images
      <root>/<category>/Data/Images/Anomaly/  anomalous images
      <root>/<category>/Data/Masks/Anomaly/   pixel masks (same stem as anomaly img, .png)

    VisA has no official train/test split for normal images, so we apply a
    deterministic 80/20 split on the sorted list of normal images:
      split='train' → first 80% of normal images
      split='test'  → last 20% of normal images + all anomaly images

    All anomaly images (with their masks) always go to the test split.
    """

    TRAIN_RATIO = 0.8
    _EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

    def __init__(self, root: str, category: str, split: str = 'train', img_size: int = 224):
        self.root = Path(root)
        self.transform = get_transform(img_size)
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.samples: list = []
        self._load(split, category)

    def _load(self, split, category):
        data_dir   = self.root / category / 'Data'
        normal_dir = data_dir / 'Images' / 'Normal'
        anomaly_dir = data_dir / 'Images' / 'Anomaly'
        mask_dir    = data_dir / 'Masks'  / 'Anomaly'

        # Sorted for determinism; case-insensitive extension check
        all_normal = sorted(
            p for p in normal_dir.glob('*') if p.suffix.lower() in self._EXTS
        )
        n_train = int(len(all_normal) * self.TRAIN_RATIO)

        if split == 'train':
            for p in all_normal[:n_train]:
                self.samples.append((p, None, 0))
        else:
            # Last 20% normal (test normal)
            for p in all_normal[n_train:]:
                self.samples.append((p, None, 0))
            # All anomaly images
            for p in sorted(
                q for q in anomaly_dir.glob('*') if q.suffix.lower() in self._EXTS
            ):
                mask = mask_dir / (p.stem + '.png')
                self.samples.append((p, mask if mask.exists() else None, 1))

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


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    'mvtec':  (MVTecDataset,   MVTEC_CATEGORIES),
    'mvtec2': (MVTec2Dataset,  MVTEC2_CATEGORIES),
    'visa':   (VisADataset,    VISA_CATEGORIES),
}

ALL_CATEGORIES = {name: cats for name, (_, cats) in _REGISTRY.items()}


def make_dataset(
    dataset_name: str,
    root: str,
    categories: Union[str, List[str]],
    split: str,
    img_size: int = 224,
) -> Dataset:
    """
    Factory that returns a Dataset (or ConcatDataset for multiple categories).

    Parameters
    ----------
    dataset_name : 'mvtec' | 'mvtec2' | 'visa'
    root         : path to the dataset root directory
    categories   : single category name, 'all', or list of category names
    split        : 'train' | 'test'
    img_size     : images are resized to (img_size x img_size)

    Returns
    -------
    A Dataset whose __getitem__ yields (image [3,H,W], mask [1,H,W], label).
    """
    if dataset_name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid options: {list(_REGISTRY)}"
        )
    cls, all_cats = _REGISTRY[dataset_name]

    if isinstance(categories, str):
        categories = all_cats if categories == 'all' else [c.strip() for c in categories.split(',')]

    datasets = [cls(root, cat, split=split, img_size=img_size) for cat in categories]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
