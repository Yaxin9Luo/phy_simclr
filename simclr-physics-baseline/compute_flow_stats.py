"""
Compute optical-flow-based motion descriptors for saved video clips and (optionally)
for sliding subclips. Produces a JSON file with entries used by a proxy sampler.
Supports multi-process sharding via torchrun (LOCAL_RANK/WORLD_SIZE) and optional
GPU optical flow via OpenCV CUDA TV-L1 if available.

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


def compute_flow_pair(f0: np.ndarray, f1: np.ndarray, use_cuda: bool = False, cuda_device: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optical flow and return (mag, ang).
    Tries GPU TV-L1 (OpenCV CUDA) if requested and available; else CPU Farneback; else frame diff.
    """
    try:
        import cv2
        if use_cuda and hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'OpticalFlowDual_TVL1_create'):
            try:
                cv2.cuda.setDevice(int(cuda_device))
            except Exception:
                pass
            tvl1 = cv2.cuda.OpticalFlowDual_TVL1_create()
            g0 = cv2.cuda_GpuMat()
            g1 = cv2.cuda_GpuMat()
            g0.upload(f0)
            g1.upload(f1)
            flow_gpu = tvl1.calc(g0, g1, None)
            flow = flow_gpu.download()
        else:
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


def descriptor_for_window(frames: List[np.ndarray], start: int, end: int, n_orient_bins: int = 8,
                          use_cuda: bool = False, cuda_device: int = 0) -> np.ndarray:
    mags = []
    angs = []
    for t in range(start, end - 1):
        mag, ang = compute_flow_pair(frames[t], frames[t+1], use_cuda=use_cuda, cuda_device=cuda_device)
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
    parser.add_argument('--use_cuda_flow', action='store_true', help='Use OpenCV CUDA TV-L1 if available')
    parser.add_argument('--distributed', action='store_true', help='Shard work across processes using LOCAL_RANK/WORLD_SIZE envs')
    args = parser.parse_args()

    samples = sorted([d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))])
    rank = int(os.environ.get('LOCAL_RANK', '0')) if args.distributed else 0
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if args.distributed else 1
    if world_size < 1:
        world_size = 1
    shard = samples[rank::world_size]
    entries = []
    for d in tqdm(shard, desc=f"Rank {rank} scanning"):
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
            desc = descriptor_for_window(frames, st, ed, n_orient_bins=args.orient_bins,
                                         use_cuda=args.use_cuda_flow, cuda_device=rank)
            entries.append({
                'sample_dir': d,
                'start': int(st),
                'end': int(ed),
                'feat': desc.tolist(),
            })

    # Write shard-specific output
    base_out = args.output if os.path.isabs(args.output) else os.path.join(args.root_dir, args.output)
    out_dir = os.path.dirname(base_out)
    os.makedirs(out_dir, exist_ok=True)
    if world_size > 1:
        shard_path = f"{base_out}.rank{rank}.json"
        with open(shard_path, 'w') as f:
            json.dump(entries, f, indent=2)
        print(f"Rank {rank}: saved {len(entries)} entries to {shard_path}")
        # Attempt merge on rank 0 if all shard files exist
        if rank == 0:
            merged = []
            for r in range(world_size):
                part = f"{base_out}.rank{r}.json"
                if os.path.isfile(part):
                    with open(part, 'r') as f:
                        merged.extend(json.load(f))
            if len(merged) > 0:
                with open(base_out, 'w') as f:
                    json.dump(merged, f, indent=2)
                print(f"Merged {len(merged)} entries to {base_out}")
    else:
        with open(base_out, 'w') as f:
            json.dump(entries, f, indent=2)
        print(f"Saved {len(entries)} entries to {base_out}")


if __name__ == '__main__':
    main()
