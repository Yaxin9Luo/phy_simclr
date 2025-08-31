# dataset.py
# ImageNet + SimCLR objective (for ViT or any image encoder)

import os
import random
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

# --- SimCLR augmentation (ImageNet) -----------------------------------------
# RandomResizedCrop: scale=(0.08,1.0), ratio=(3/4,4/3)
# ColorJitter: (0.8, 0.8, 0.8, 0.2) applied with p=0.8
# RandomGrayscale p=0.2
# GaussianBlur p=0.5 with kernel size ≈ 10% of image size, sigma∈[0.1, 2.0]
# Normalization: ImageNet mean/std
# Refs: Chen et al., 2020 (SimCLR), Appendix A. :contentReference[oaicite:1]{index=1}

class RandomGaussianBlur:
    """Gaussian blur with probability p. Kernel size = ~10% of shorter side (odd, >=3)."""
    def __init__(self, p: float = 0.5, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        k = max(3, int(0.1 * min(w, h)))
        if k % 2 == 0:
            k += 1
        # Build a GaussianBlur transform on-the-fly to use this kernel size
        gb = transforms.GaussianBlur(kernel_size=k, sigma=self.sigma_range)
        return gb(img)

def make_simclr_transform(img_size: int = 224, color_s: float = 1.0) -> transforms.Compose:
    """SimCLR ImageNet augmentation pipeline."""
    color_jitter = transforms.ColorJitter(
        0.8 * color_s,  # brightness
        0.8 * color_s,  # contrast
        0.8 * color_s,  # saturation
        0.2 * color_s,  # hue
    )
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        RandomGaussianBlur(p=0.5, sigma_range=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class TwoCropTransform:
    """Create two differently augmented views of the same image (SimCLR positives)."""
    def __init__(self, base_transform: transforms.Compose):
        self.base_transform = base_transform

    def __call__(self, x: Image.Image):
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        return xi, xj

# --- Dataset ----------------------------------------------------------------
class ImageNetSimCLR(Dataset):
    """
    Wraps torchvision.datasets.ImageFolder to output two augmented views per image.
    Labels are returned for convenience but are unused by SimCLR loss.
    """
    def __init__(self, root: str, split: str = "train", img_size: int = 224):
        split_dir = os.path.join(root, split)
        base_t = make_simclr_transform(img_size=img_size, color_s=1.0)
        self.dataset = datasets.ImageFolder(split_dir, transform=TwoCropTransform(base_t))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        (xi, xj), y = self.dataset[idx]  # xi, xj: (C,H,W) tensors
        return xi, xj, y  # y is optional; SimCLR ignores it