"""
Generate and save synthetic bouncing-balls videos to disk for visualization
or offline training/testing.

Saves each sample as a directory of PNG frames; optionally writes an MP4.

Usage examples:
  # Save 50 samples (16 frames @224), frames only
  python generate_synthetic_videos.py --output_dir synthetic_videos --num_samples 50 --num_frames 16 --img_size 224

  # Save MP4s alongside frames
  python generate_synthetic_videos.py --output_dir synthetic_videos --num_samples 20 --write_video --fps 8
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

from video_dataset import SyntheticBouncingBalls


def save_frames(frames, out_dir, prefix="raw"):
    os.makedirs(out_dir, exist_ok=True)
    for t, img in enumerate(frames):
        assert isinstance(img, Image.Image)
        img.save(os.path.join(out_dir, f"{prefix}_frame_{t:02d}.png"))


def save_as_mp4(frames, out_path, fps=8, codec="mp4v"):
    try:
        import cv2
    except Exception as e:
        print(f"OpenCV not available ({e}); attempting GIF fallback...")
        # Fallback to GIF using PIL if possible
        out_gif = os.path.splitext(out_path)[0] + ".gif"
        try:
            save_as_gif(frames, out_gif, fps=fps)
            print(f"Saved GIF instead: {out_gif}")
        except Exception as e2:
            print(f"GIF fallback failed ({e2}); skipping video writing: {out_path}")
        return
    if len(frames) == 0:
        return
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for img in frames:
        arr = np.array(img)  # RGB
        bgr = arr[:, :, ::-1]
        vw.write(bgr)
    vw.release()


def save_as_gif(frames, out_path, fps=8):
    if len(frames) == 0:
        return
    duration_ms = int(1000 / max(1, fps))
    first, rest = frames[0], frames[1:]
    first.save(out_path, save_all=True, append_images=rest, duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bouncing-balls videos")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save samples')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--box_size', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--write_video', action='store_true', help='Also save an MP4 per sample')
    parser.add_argument('--fps', type=int, default=8, help='FPS for MP4 writing')
    parser.add_argument('--codec', type=str, default='mp4v', help='FourCC codec for OpenCV (e.g., mp4v, avc1)')
    parser.add_argument('--prefix', type=str, default='sample')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a generator object to reuse drawing/utilities
    gen = SyntheticBouncingBalls(
        length=1,
        num_frames=args.num_frames,
        box_size=args.box_size,
        img_size=args.img_size,
        save_dir=None,
        deterministic=True,
    )

    print(f"Saving {args.num_samples} synthetic videos to: {args.output_dir}")
    for i in tqdm(range(args.num_samples), desc="Generating"):
        rng = np.random.default_rng(args.seed + i)
        frames, label = gen._generate_clip(rng)  # list of RGB PIL images

        sample_dir = os.path.join(args.output_dir, f"{args.prefix}_{i:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save frames
        save_frames(frames, sample_dir, prefix="raw")

        # Save MP4 if requested
        if args.write_video:
            mp4_path = os.path.join(sample_dir, f"{args.prefix}_{i:06d}.mp4")
            save_as_mp4(frames, mp4_path, fps=args.fps, codec=args.codec)

        # Save metadata
        meta = {
            'num_frames': args.num_frames,
            'img_size': args.img_size,
            'box_size': args.box_size,
            'seed': int(args.seed + i),
            'label_num_balls_minus_1': int(label),
        }
        with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
