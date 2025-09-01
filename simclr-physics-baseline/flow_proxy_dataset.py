"""
Flow-proxy dataset for physics-aware pairing.

Loads flow_stats.json (computed by compute_flow_stats.py) and builds positives
by KNN in the flow-feature space. Each sample returns two different clips:
  xi = aug(anchor subclip), xj = aug(positive neighbor subclip)
so that SimCLR treats them as positives. Negatives remain in-batch others.
"""

import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from video_dataset import make_video_simclr_transform, ApplyToFrames


class FlowProxyDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        stats_path: str,
        img_size: int = 224,
        k_pos: int = 5,
        allow_same_clip_pos: bool = True,
        min_time_separation: int = 0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.k_pos = max(1, k_pos)
        self.allow_same_clip_pos = allow_same_clip_pos
        self.min_time_separation = int(min_time_separation)

        with open(stats_path, 'r') as f:
            entries: List[Dict] = json.load(f)
        if len(entries) == 0:
            raise ValueError(f"No entries in stats file: {stats_path}")

        self.entries = entries
        self.features = np.stack([np.array(e['feat'], dtype=np.float32) for e in entries], axis=0)  # (N,D)
        # L2 normalize for cosine similarity
        self.features = self.features / (np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-8)

        # Precompute neighbor lists by cosine similarity
        sim = self.features @ self.features.T  # (N,N)
        # For each row, argsort descending similarity
        order = np.argsort(-sim, axis=1)
        self.nn_indices: List[List[int]] = []
        for i in range(order.shape[0]):
            candi = order[i].tolist()
            # drop self
            candi = [j for j in candi if j != i]
            # optional: enforce different clip or time separation
            if not self.allow_same_clip_pos or self.min_time_separation > 0:
                si = self.entries[i]['sample_dir']
                st_i, ed_i = int(self.entries[i]['start']), int(self.entries[i]['end'])
                tmp = []
                for j in candi:
                    sj = self.entries[j]['sample_dir']
                    st_j, ed_j = int(self.entries[j]['start']), int(self.entries[j]['end'])
                    if (not self.allow_same_clip_pos) and (sj == si):
                        continue
                    if self.min_time_separation > 0 and (sj == si):
                        # enforce windows are separated in time
                        if abs(st_j - st_i) < self.min_time_separation:
                            continue
                    tmp.append(j)
                candi = tmp
            self.nn_indices.append(candi)

        self.frame_transform = ApplyToFrames(make_video_simclr_transform(img_size=img_size))

    def __len__(self):
        return len(self.entries)

    def _load_subclip(self, entry: Dict) -> List[Image.Image]:
        sdir = os.path.join(self.root_dir, entry['sample_dir'])
        names = [n for n in os.listdir(sdir) if n.endswith('.png') and n.startswith('raw_frame_')]
        if len(names) == 0:
            names = [n for n in os.listdir(sdir) if n.endswith('.png')]
        names = sorted(names)
        st, ed = int(entry['start']), int(entry['end'])
        names = names[st:ed]
        frames: List[Image.Image] = []
        for n in names:
            img = Image.open(os.path.join(sdir, n)).convert('RGB')
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            frames.append(img)
        return frames

    def __getitem__(self, idx: int):
        anchor = self.entries[idx]
        nns = self.nn_indices[idx]
        if len(nns) == 0:
            # fallback: pick any other index
            j = (idx + 1) % len(self.entries)
        else:
            # pick one of top-k positives
            j = nns[0] if self.k_pos <= 1 else np.random.choice(nns[:self.k_pos])

        frames_a = self._load_subclip(anchor)
        frames_p = self._load_subclip(self.entries[j])
        xa = self.frame_transform(frames_a)  # (T,C,H,W)
        xp = self.frame_transform(frames_p)
        return xa, xp, -1

