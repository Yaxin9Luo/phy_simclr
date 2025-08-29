"""
Dataset preprocessing and PyTorch Dataset class for CLEVRER dataset.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Optional
import random
import cv2
from pathlib import Path


def extract_frames_from_video(video_path: str, num_frames: int = 20, 
                            start_frame: int = 0) -> List[np.ndarray]:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get total number of frames in video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    if total_frames <= num_frames:
        # If video has fewer frames than requested, use all frames
        frame_indices = list(range(total_frames))
    else:
        # Sample frames evenly from the video
        frame_indices = np.linspace(start_frame, 
                                  min(start_frame + num_frames, total_frames - 1), 
                                  num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def preprocess_clevrer_video(video_path: str, output_dir: str, video_id: str,
                           num_frames: int = 20, resize: Tuple[int, int] = (256, 256)):
    """Preprocess a CLEVRER video and save frames."""
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, f"video_{video_id}")
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Extract frames
    frames = extract_frames_from_video(video_path, num_frames)
    
    # Save frames as images
    for i, frame in enumerate(frames):
        # Resize frame
        frame_pil = Image.fromarray(frame)
        frame_pil = frame_pil.resize(resize, Image.BILINEAR)
        
        # Save frame
        frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.png")
        frame_pil.save(frame_path)
    
    return len(frames)


def preprocess_clevrer_dataset(clevrer_root: str, output_dir: str, 
                             split: str = 'train', max_videos: Optional[int] = None):
    """Preprocess CLEVRER dataset videos."""
    # Create output directory
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Find all video files across subdirectories
    video_files = []
    video_paths = []
    
    # Check if videos are in subdirectories (e.g., video_00000-01000/)
    subdirs = sorted([d for d in os.listdir(clevrer_root) if d.startswith('video_') and '-' in d])
    
    if subdirs:
        # Videos are organized in range directories
        for subdir in subdirs:
            subdir_path = os.path.join(clevrer_root, subdir)
            if os.path.isdir(subdir_path):
                for video_file in sorted(os.listdir(subdir_path)):
                    if video_file.endswith('.mp4'):
                        video_files.append(video_file)
                        video_paths.append(os.path.join(subdir_path, video_file))
                        if max_videos and len(video_files) >= max_videos:
                            break
            if max_videos and len(video_files) >= max_videos:
                break
    else:
        # Try standard CLEVRER structure (videos/train/)
        video_dir = os.path.join(clevrer_root, 'videos', split)
        if os.path.exists(video_dir):
            for video_file in sorted(os.listdir(video_dir)):
                if video_file.endswith('.mp4'):
                    video_files.append(video_file)
                    video_paths.append(os.path.join(video_dir, video_file))
                    if max_videos and len(video_files) >= max_videos:
                        break
    
    # Load annotations if available
    annotations = {}
    annotation_path = os.path.join(clevrer_root, 'annotations', split, f'annotation_{split}.json')
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
            for item in annotation_data:
                annotations[item['video_filename']] = item
    
    print(f"Processing {len(video_files)} videos from {split} split...")
    
    metadata = []
    for i, (video_file, video_path) in enumerate(zip(video_files, video_paths)):
        video_id = video_file.replace('.mp4', '')
        
        # Preprocess video
        num_frames = preprocess_clevrer_video(
            video_path, output_split_dir, video_id
        )
        
        # Store metadata
        meta_item = {
            'video_id': video_id,
            'video_file': video_file,
            'num_frames': num_frames,
            'output_dir': os.path.join(split, f"video_{video_id}")
        }
        
        # Add annotation info if available
        if video_file in annotations:
            meta_item['annotation'] = annotations[video_file]
        
        metadata.append(meta_item)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(video_files)} videos")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f'{split}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Preprocessing complete. Metadata saved to {metadata_path}")
    return metadata


class CLEVRERDataset(Dataset):
    """PyTorch Dataset for loading CLEVRER video clips for contrastive learning."""
    
    def __init__(self, data_dir: str, metadata_file: str, transform=None, 
                 num_frames: int = 20, frame_size: int = 128):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Found {len(self.metadata)} videos in dataset")
        
        # Default augmentation pipeline if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=frame_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def _load_frames(self, video_dir: str) -> List[Image.Image]:
        """Load all frames from a video directory."""
        frames = []
        frame_paths = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
        
        # Sample frames if we have more than needed
        if len(frame_paths) > self.num_frames:
            indices = np.linspace(0, len(frame_paths)-1, self.num_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        for frame_path in frame_paths[:self.num_frames]:
            full_path = os.path.join(video_dir, frame_path)
            if os.path.exists(full_path):
                frame = Image.open(full_path).convert('RGB')
                frames.append(frame)
        
        # Pad with last frame if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (self.frame_size, self.frame_size)))
        
        return frames
    
    def _apply_consistent_augmentation(self, frames: List[Image.Image]) -> torch.Tensor:
        """Apply the same spatial augmentation to all frames in a clip."""
        if not frames:
            return torch.zeros((self.num_frames, 3, self.frame_size, self.frame_size))
        
        augmented_frames = []
        
        # Get parameters for consistent spatial augmentations
        # RandomResizedCrop parameters
        if any(isinstance(t, transforms.RandomResizedCrop) for t in self.transform.transforms):
            crop_transform = next(t for t in self.transform.transforms if isinstance(t, transforms.RandomResizedCrop))
            crop_params = crop_transform.get_params(frames[0], crop_transform.scale, crop_transform.ratio)
            i, j, h, w = crop_params
        else:
            i, j, h, w = 0, 0, frames[0].height, frames[0].width
        
        # Random flip decisions
        h_flip = random.random() < 0.5
        
        # ColorJitter parameters
        color_jitter_transform = None
        for t in self.transform.transforms:
            if isinstance(t, transforms.ColorJitter):
                # Get the parameters once for consistent application
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = t.get_params(
                    t.brightness, t.contrast, t.saturation, t.hue
                )
                color_jitter_transform = (fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                break
        
        for frame in frames:
            # Apply consistent crop
            frame = transforms.functional.resized_crop(frame, i, j, h, w, (self.frame_size, self.frame_size))
            
            # Apply consistent flip
            if h_flip:
                frame = transforms.functional.hflip(frame)
            
            # Apply color jitter if available
            if color_jitter_transform is not None:
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = color_jitter_transform
                # Apply transformations in the order determined by fn_idx
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        frame = transforms.functional.adjust_brightness(frame, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        frame = transforms.functional.adjust_contrast(frame, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        frame = transforms.functional.adjust_saturation(frame, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        frame = transforms.functional.adjust_hue(frame, hue_factor)
            
            # Convert to tensor and normalize
            frame_tensor = transforms.functional.to_tensor(frame)
            frame_tensor = transforms.functional.normalize(
                frame_tensor, 
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            augmented_frames.append(frame_tensor)
        
        # Stack frames along time dimension
        clip = torch.stack(augmented_frames, dim=0)  # Shape: (T, C, H, W)
        return clip
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a video and return two augmented views of the same clip.
        Returns: (clip_i, clip_j) where each clip has shape (T, C, H, W)
        """
        video_meta = self.metadata[index]
        video_dir = os.path.join(self.data_dir, video_meta['output_dir'])
        
        # Load frames
        frames = self._load_frames(video_dir)
        
        # Create two different augmented views
        clip_i = self._apply_consistent_augmentation(frames)
        clip_j = self._apply_consistent_augmentation(frames)
        
        return clip_i, clip_j


# Backward compatibility alias
BouncingBallDataset = CLEVRERDataset


if __name__ == "__main__":
    # Example preprocessing script
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CLEVRER dataset')
    parser.add_argument('--clevrer_root', type=str, required=True, 
                       help='Path to CLEVRER dataset root (containing videos/ and annotations/)')
    parser.add_argument('--output_dir', type=str, default='processed_clevrer',
                       help='Output directory for processed frames')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process (for testing)')
    
    args = parser.parse_args()
    
    # Preprocess dataset
    preprocess_clevrer_dataset(
        args.clevrer_root, 
        args.output_dir,
        split=args.split,
        max_videos=args.max_videos
    )
