"""
Video SimCLR dataset utilities and a simple synthetic dataset for quick testing.

Provides:
- make_video_simclr_transform: per-frame SimCLR-style augs for videos
- VideoTwoClipTransform: returns two augmented views of the same clip
- SyntheticBouncingBalls: on-the-fly synthetic moving balls sequences
"""

from typing import Tuple, List, Optional
import math
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ApplyToFrames:
    """Apply a torchvision transform to each frame in a clip tensor or PIL Image list.

    Input clip shape: (T, C, H, W) tensor or list of PIL Images (length T).
    Output clip shape: (T, C, H, W) tensor.
    """
    def __init__(self, frame_transform: transforms.Compose):
        self.frame_transform = frame_transform

    def __call__(self, clip):
        frames: List[torch.Tensor] = []
        if isinstance(clip, torch.Tensor):
            # clip: (T, C, H, W) in [0,1] or [0,255]
            t = clip.size(0)
            for i in range(t):
                img = transforms.ToPILImage()(clip[i])
                frames.append(self.frame_transform(img))
        else:
            # list of PIL images
            for img in clip:
                frames.append(self.frame_transform(img))
        return torch.stack(frames, dim=0)  # (T, C, H, W)


def make_video_simclr_transform(img_size: int = 224, color_s: float = 1.0) -> transforms.Compose:
    """Per-frame SimCLR augmentations adapted for video clips.

    Note: To preserve motion semantics, we avoid strong geometric transforms that
    distort temporal consistency (e.g., heavy rotations). RandomResizedCrop is kept moderate.
    """
    color_jitter = transforms.ColorJitter(
        0.8 * color_s,  # brightness
        0.8 * color_s,  # contrast
        0.8 * color_s,  # saturation
        0.2 * color_s,  # hue
    )

    frame_t = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # Replicate grayscale to 3 channels if needed happens before ToTensor in synthetic dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return ApplyToFrames(frame_t)


class VideoTwoClipTransform:
    """Create two differently augmented views of the same clip (SimCLR positives)."""
    def __init__(self, base_transform: transforms.Compose):
        self.base_transform = base_transform

    def __call__(self, clip):
        xi = self.base_transform(clip)
        xj = self.base_transform(clip)
        return xi, xj


class SyntheticBouncingBalls(Dataset):
    """
    Simple on-the-fly synthetic video dataset of bouncing balls.

    Each sample is a sequence of T grayscale frames with 1-3 balls moving with
    elastic reflections in a square box. Returns two augmented clips for SimCLR
    along with a lightweight label (number of balls - 1) for optional diagnostics.
    """

    def __init__(
        self,
        length: int = 1000,
        num_frames: int = 16,
        box_size: int = 96,
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.length = length
        self.num_frames = num_frames
        self.box_size = box_size
        self.img_size = img_size
        self.transform = transform if transform is not None else VideoTwoClipTransform(
            make_video_simclr_transform(img_size=img_size)
        )
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _draw_ball(canvas: np.ndarray, cx: float, cy: float, r: float, intensity: int = 255):
        h, w = canvas.shape
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        canvas[mask] = np.maximum(canvas[mask], intensity)

    def _generate_clip(self) -> Tuple[List[Image.Image], int]:
        # Random number of balls (1-3)
        n_balls = np.random.randint(1, 4)
        # Ball states: [x, y, vx, vy, r]
        balls = []
        for _ in range(n_balls):
            r = np.random.uniform(3.0, 6.0)
            x = np.random.uniform(r, self.box_size - r)
            y = np.random.uniform(r, self.box_size - r)
            vx = np.random.uniform(-2.0, 2.0)
            vy = np.random.uniform(-2.0, 2.0)
            balls.append([x, y, vx, vy, r])

        frames: List[Image.Image] = []
        for _ in range(self.num_frames):
            canvas = np.zeros((self.box_size, self.box_size), dtype=np.uint8)
            for b in balls:
                self._draw_ball(canvas, b[0], b[1], b[4])
                # Update with elastic reflection
                b[0] += b[2]
                b[1] += b[3]
                if b[0] - b[4] < 0 or b[0] + b[4] >= self.box_size:
                    b[2] = -b[2]
                    b[0] = np.clip(b[0], b[4], self.box_size - b[4] - 1)
                if b[1] - b[4] < 0 or b[1] + b[4] >= self.box_size:
                    b[3] = -b[3]
                    b[1] = np.clip(b[1], b[4], self.box_size - b[4] - 1)

            img = Image.fromarray(canvas, mode="L")  # grayscale
            # convert to 3-channel PIL for downstream per-frame transforms
            img = img.convert("RGB")
            # Resize here to avoid heavy ops per-frame later if desired
            img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            frames.append(img)

        return frames, n_balls - 1  # zero-indexed label

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        frames, label = self._generate_clip()
        # Transform returns (xi, xj) each as (T, C, H, W)
        xi, xj = self.transform(frames)
        return xi, xj, label

