"""
Validate flow-proxy pairing by printing anchor/positive distances and (optionally)
saving side-by-side GIFs for visual confirmation.

Usage:
  python validate_flow_pairs.py \
    --root_dir simclr-physics-baseline/synthetic_videos \
    --stats simclr-physics-baseline/synthetic_videos/flow_stats.json \
    --num_pairs 10 --k_pos 5 --out_dir simclr-physics-baseline/flow_pair_viz --fps 8
"""

import os
import json
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_entries(stats_path: str) -> List[Dict]:
    with open(stats_path, 'r') as f:
        return json.load(f)


def build_features(entries: List[Dict]) -> np.ndarray:
    feats = np.stack([np.array(e['feat'], dtype=np.float32) for e in entries], axis=0)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats


def nearest_neighbors(feats: np.ndarray) -> np.ndarray:
    sim = feats @ feats.T
    order = np.argsort(-sim, axis=1)  # descending similarity
    return sim, order


def load_window_frames(root_dir: str, entry: Dict, resize_to: int = None) -> List[Image.Image]:
    sdir = os.path.join(root_dir, entry['sample_dir'])
    names = [n for n in os.listdir(sdir) if n.endswith('.png') and n.startswith('raw_frame_')]
    if len(names) == 0:
        names = [n for n in os.listdir(sdir) if n.endswith('.png')]
    names = sorted(names)
    st, ed = int(entry['start']), int(entry['end'])
    names = names[st:ed]
    frames = []
    for n in names:
        img = Image.open(os.path.join(sdir, n)).convert('RGB')
        if resize_to is not None and img.size != (resize_to, resize_to):
            img = img.resize((resize_to, resize_to), resample=Image.BILINEAR)
        frames.append(img)
    return frames


def save_side_by_side_gif(frames_a: List[Image.Image], frames_b: List[Image.Image], out_path: str, fps: int = 8):
    T = min(len(frames_a), len(frames_b))
    if T == 0:
        return
    w, h = frames_a[0].size
    merged = []
    for t in range(T):
        canvas = Image.new('RGB', (w * 2, h))
        canvas.paste(frames_a[t], (0, 0))
        canvas.paste(frames_b[t], (w, 0))
        merged.append(canvas)
    duration_ms = int(1000 / max(1, fps))
    merged[0].save(out_path, save_all=True, append_images=merged[1:], duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser(description='Validate flow-proxy positive pairing')
    parser.add_argument('--root_dir', type=str, required=True, help='Root dir containing sample_* folders')
    parser.add_argument('--stats', type=str, required=True, help='Path to flow_stats.json')
    parser.add_argument('--num_pairs', type=int, default=10)
    parser.add_argument('--k_pos', type=int, default=5)
    parser.add_argument('--no_same_clip_pos', action='store_true')
    parser.add_argument('--min_time_sep', type=int, default=0)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--out_dir', type=str, default=None, help='If set, saves side-by-side GIFs here')
    parser.add_argument('--fps', type=int, default=8)

    args = parser.parse_args()

    entries = load_entries(args.stats)
    feats = build_features(entries)
    sim, order = nearest_neighbors(feats)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    n = len(entries)
    picked = np.linspace(0, n - 1, num=min(args.num_pairs, n), dtype=int)

    print("\nTop-1 positives for sampled anchors (cosine similarity):")
    for idx in tqdm(picked, desc="Validating"):
        # Build candidate list excluding self and obeying constraints
        candidates = [j for j in order[idx] if j != idx]
        if args.no_same_clip_pos or args.min_time_sep > 0:
            si = entries[idx]['sample_dir']
            st_i = int(entries[idx]['start'])
            tmp = []
            for j in candidates:
                sj = entries[j]['sample_dir']
                st_j = int(entries[j]['start'])
                if args.no_same_clip_pos and (sj == si):
                    continue
                if args.min_time_sep > 0 and (sj == si) and (abs(st_j - st_i) < args.min_time_sep):
                    continue
                tmp.append(j)
            candidates = tmp
        if len(candidates) == 0:
            continue
        j = candidates[0] if args.k_pos <= 1 else np.random.choice(candidates[:args.k_pos])

        s = float(sim[idx, j])
        same_clip = entries[idx]['sample_dir'] == entries[j]['sample_dir']
        print(f"Anchor #{idx} ({entries[idx]['sample_dir']} [{entries[idx]['start']},{entries[idx]['end']}]) -> "
              f"Pos #{j} ({entries[j]['sample_dir']} [{entries[j]['start']},{entries[j]['end']}]) | sim={s:.4f} | same_clip={same_clip}")

        if args.out_dir:
            frames_a = load_window_frames(args.root_dir, entries[idx], resize_to=args.resize)
            frames_b = load_window_frames(args.root_dir, entries[j], resize_to=args.resize)
            out_path = os.path.join(args.out_dir, f"pair_anchor{idx:06d}_pos{j:06d}.gif")
            save_side_by_side_gif(frames_a, frames_b, out_path, fps=args.fps)


if __name__ == '__main__':
    main()

