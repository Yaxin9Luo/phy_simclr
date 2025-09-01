"""
Video SimCLR dataset utilities.

Provides:
- make_video_simclr_transform: per-frame SimCLR-style augs for videos
- VideoTwoClipTransform: returns two augmented views of the same clip
- SyntheticBouncingBalls: on-the-fly synthetic moving balls (for generation)
- DiskVideoClipsDataset: disk-backed reader for saved frame folders
"""

from typing import Tuple, List, Optional
import math
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os



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
        # Saving options
        save_dir: Optional[str] = None,
        save_every: int = 0,
        save_limit: Optional[int] = None,
        save_raw: bool = True,
        save_aug_i: bool = False,
        save_aug_j: bool = False,
        deterministic: bool = True,
    ):
        super().__init__()
        self.length = length
        self.num_frames = num_frames
        self.box_size = box_size
        self.img_size = img_size
        self.transform = transform if transform is not None else VideoTwoClipTransform(
            make_video_simclr_transform(img_size=img_size)
        )
        self.base_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        # Save controls
        self.save_dir = save_dir
        self.save_every = int(save_every) if save_every is not None else 0
        self.save_limit = save_limit
        self.save_raw = save_raw
        self.save_aug_i = save_aug_i
        self.save_aug_j = save_aug_j
        self.deterministic = deterministic
        # for de-normalization when saving augmented tensors
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    @staticmethod
    def _draw_ball(canvas: np.ndarray, cx: float, cy: float, r: float, intensity: int = 255):
        h, w = canvas.shape
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        canvas[mask] = np.maximum(canvas[mask], intensity)

    def _generate_clip(self, rng: np.random.Generator) -> Tuple[List[Image.Image], int]:
        # Random number of balls (1-3)
        n_balls = rng.integers(1, 4)
        # Ball states: [x, y, vx, vy, r]
        balls = []
        for _ in range(n_balls):
            r = rng.uniform(3.0, 6.0)
            x = rng.uniform(r, self.box_size - r)
            y = rng.uniform(r, self.box_size - r)
            vx = rng.uniform(-2.0, 2.0)
            vy = rng.uniform(-2.0, 2.0)
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

    def _maybe_save_pil_frames(self, frames: List[Image.Image], out_dir: str, prefix: str):
        os.makedirs(out_dir, exist_ok=True)
        for t, img in enumerate(frames):
            img.save(os.path.join(out_dir, f"{prefix}_frame_{t:02d}.png"))

    def _maybe_save_tensor_frames(self, frames: torch.Tensor, out_dir: str, prefix: str):
        # frames: (T, C, H, W), normalized with ImageNet stats
        os.makedirs(out_dir, exist_ok=True)
        T = frames.size(0)
        for t in range(T):
            x = frames[t].detach().cpu()
            x = (x * self.norm_std + self.norm_mean).clamp(0, 1)
            img = transforms.ToPILImage()(x)
            img.save(os.path.join(out_dir, f"{prefix}_frame_{t:02d}.png"))

    def __getitem__(self, idx: int):
        # Deterministic per-index RNG for reproducible visualization if desired
        if self.deterministic:
            rng = np.random.default_rng(self.base_seed + idx)
        else:
            rng = np.random.default_rng()

        frames, label = self._generate_clip(rng)
        # Transform returns (xi, xj) each as (T, C, H, W)
        xi, xj = self.transform(frames)

        # Optional saving
        if self.save_dir and (self.save_every > 0) and (idx % self.save_every == 0):
            if (self.save_limit is None) or (idx // self.save_every < self.save_limit):
                sample_dir = os.path.join(self.save_dir, f"sample_{idx:06d}")
                if self.save_raw:
                    self._maybe_save_pil_frames(frames, sample_dir, prefix="raw")
                if self.save_aug_i:
                    self._maybe_save_tensor_frames(xi, sample_dir, prefix="aug_i")
                if self.save_aug_j:
                    self._maybe_save_tensor_frames(xj, sample_dir, prefix="aug_j")

        return xi, xj, label


class DiskVideoClipsDataset(Dataset):
    """
    Disk-backed dataset that reads video clips saved as directories of frames.

    Expected directory structure:
      root/
        sample_000000/
          raw_frame_00.png, raw_frame_01.png, ...
          metadata.json (optional, may contain label info)
        sample_000001/
          ...

    Returns two augmented views of each clip suitable for SimCLR.
    """

    def __init__(self, root_dir: str, img_size: int = 224, frame_glob_prefix: str = "raw_frame_",
                 transform: Optional[transforms.Compose] = None, use_metadata_label: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.frame_glob_prefix = frame_glob_prefix
        self.transform = transform if transform is not None else VideoTwoClipTransform(
            make_video_simclr_transform(img_size=img_size)
        )
        self.use_metadata_label = use_metadata_label

        # collect sample directories
        subs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)] if os.path.isdir(root_dir) else []
        self.sample_dirs = sorted([p for p in subs if os.path.isdir(p)])
        if len(self.sample_dirs) == 0:
            raise FileNotFoundError(f"No sample directories found in {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def _load_clip_frames(self, sample_dir: str) -> List[Image.Image]:
        # List files matching prefix and png
        names = [n for n in os.listdir(sample_dir) if n.startswith(self.frame_glob_prefix) and n.endswith('.png')]
        if len(names) == 0:
            # fallback: any png frames
            names = [n for n in os.listdir(sample_dir) if n.endswith('.png')]
        names = sorted(names)
        frames: List[Image.Image] = []
        for n in names:
            img = Image.open(os.path.join(sample_dir, n)).convert('RGB')
            # Resize to img_size if needed
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            frames.append(img)
        if len(frames) == 0:
            raise FileNotFoundError(f"No frames in {sample_dir}")
        return frames

    def _maybe_load_label(self, sample_dir: str) -> int:
        if not self.use_metadata_label:
            return -1
        meta_path = os.path.join(sample_dir, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                return int(meta.get('label_num_balls_minus_1', -1))
            except Exception:
                return -1
        return -1

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        frames = self._load_clip_frames(sample_dir)
        label = self._maybe_load_label(sample_dir)
        xi, xj = self.transform(frames)  # (T,C,H,W)
        return xi, xj, label
