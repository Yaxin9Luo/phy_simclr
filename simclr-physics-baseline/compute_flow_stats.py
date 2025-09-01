"""
Compute optical-flow-based motion descriptors for saved video clips and (optionally)
for sliding subclips. Produces a JSON file with entries used by a proxy sampler.

Usage:
  python compute_flow_stats.py \
    --root_dir simclr-physics-baseline/synthetic_videos \
    --output simclr-physics-baseline/flow_stats.json \
    --window 16 --stride 16
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_frames(sample_dir: str) -> List[np.ndarray]:
    names = [n for n in os.listdir(sample_dir) if n.endswith('.png') and n.startswith('raw_frame_')]
    if len(names) == 0:
        # fallback any png
        names = [n for n in os.listdir(sample_dir) if n.endswith('.png')]
    names = sorted(names)
    frames = []
    for n in names:
        img = Image.open(os.path.join(sample_dir, n)).convert('L')  # grayscale
        frames.append(np.array(img))
    return frames


def compute_flow_pair(f0: np.ndarray, f1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optical flow and return (mag, ang). Falls back to frame diff if cv2 missing."""
    try:
        import cv2
        flow = cv2.calcOpticalFlowFarneback(f0, f1, None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2, flags=0)
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx*fx + fy*fy)
        ang = (np.arctan2(fy, fx) + 2*np.pi) % (2*np.pi)
        return mag, ang
    except Exception:
        # Simple frame-diff proxy
        df = np.abs(f1.astype(np.float32) - f0.astype(np.float32))
        mag = df
        ang = np.zeros_like(df)
        return mag, ang


def descriptor_for_window(frames: List[np.ndarray], start: int, end: int, n_orient_bins: int = 8) -> np.ndarray:
    mags = []
    angs = []
    for t in range(start, end - 1):
        mag, ang = compute_flow_pair(frames[t], frames[t+1])
        mags.append(mag)
        angs.append(ang)
    if len(mags) == 0:
        return np.zeros(n_orient_bins + 6, dtype=np.float32)

    mags = np.stack(mags, axis=0)  # (T-1,H,W)
    angs = np.stack(angs, axis=0)

    # spatial mean per frame
    mt = mags.reshape(mags.shape[0], -1).mean(axis=1)  # (T-1,)
    mean_mag = float(mt.mean())
    std_mag = float(mt.std())
    p90_mag = float(np.percentile(mt, 90))
    max_mag = float(mt.max())
    var_mag = float(mt.var())
    dmt = np.diff(mt)
    mean_abs_dmag = float(np.mean(np.abs(dmt))) if dmt.size > 0 else 0.0
    var_dmag = float(np.var(dmt)) if dmt.size > 0 else 0.0

    # orientation histogram weighted by magnitude
    ang_flat = angs.reshape(-1)
    mag_flat = mags.reshape(-1)
    bins = np.linspace(0, 2*np.pi, num=n_orient_bins+1, endpoint=True)
    hist, _ = np.histogram(ang_flat, bins=bins, weights=mag_flat)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-8)
    desc = np.concatenate([
        np.array([mean_mag, std_mag, p90_mag, max_mag, var_mag, mean_abs_dmag, var_dmag], dtype=np.float32),
        hist.astype(np.float32)
    ], axis=0)
    return desc


def main():
    parser = argparse.ArgumentParser(description='Compute flow-based stats for video clips')
    parser.add_argument('--root_dir', type=str, required=True, help='Directory with sample_* folders')
    parser.add_argument('--output', type=str, default='flow_stats.json')
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--orient_bins', type=int, default=8)
    args = parser.parse_args()

    samples = sorted([d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))])
    entries = []
    for d in tqdm(samples, desc="Scanning clips"):
        sdir = os.path.join(args.root_dir, d)
        frames = load_frames(sdir)
        T = len(frames)
        if T < 2:
            continue
        w = min(args.window, T)
        stride = max(1, args.stride)
        starts = list(range(0, T - w + 1, stride)) if T >= w else [0]
        for st in starts:
            ed = st + w
            desc = descriptor_for_window(frames, st, ed, n_orient_bins=args.orient_bins)
            entries.append({
                'sample_dir': d,
                'start': int(st),
                'end': int(ed),
                'feat': desc.tolist(),
            })

    out_path = args.output if os.path.isabs(args.output) else os.path.join(args.root_dir, args.output)
    with open(out_path, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"Saved {len(entries)} entries to {out_path}")


if __name__ == '__main__':
    main()
