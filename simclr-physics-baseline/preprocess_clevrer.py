#!/usr/bin/env python
"""
Standalone script to preprocess CLEVRER dataset for SimCLR training.
"""

import argparse
from dataset import preprocess_clevrer_dataset


def main():
    parser = argparse.ArgumentParser(description='Preprocess CLEVRER dataset')
    parser.add_argument('--clevrer_root', type=str, required=True, 
                       help='Path to CLEVRER dataset root (containing video subdirectories or videos/ and annotations/)')
    parser.add_argument('--output_dir', type=str, default='processed_clevrer',
                       help='Output directory for processed frames')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process (for testing)')
    parser.add_argument('--num_frames', type=int, default=20,
                       help='Number of frames to extract per video')
    parser.add_argument('--frame_size', type=int, default=256,
                       help='Size to resize frames to')
    
    args = parser.parse_args()
    
    print(f"Preprocessing CLEVRER {args.split} split...")
    print(f"Input: {args.clevrer_root}")
    print(f"Output: {args.output_dir}")
    print(f"Note: Will look for videos in subdirectories (e.g., video_00000-01000/) or standard structure")
    
    # Preprocess dataset
    metadata = preprocess_clevrer_dataset(
        args.clevrer_root, 
        args.output_dir,
        split=args.split,
        max_videos=args.max_videos
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Processed {len(metadata)} videos")
    print(f"Metadata saved to: {args.output_dir}/{args.split}_metadata.json")


if __name__ == "__main__":
    main()
