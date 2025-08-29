# SimCLR Physics-Aware Representation Learning

This project implements a SimCLR-based contrastive learning framework for learning physics-aware representations from videos of 3D moving objects. It supports both synthetic bouncing ball data and the CLEVRER dataset.

## Project Structure

```
simclr-physics-baseline/
├── data/                     # Directory for the generated dataset
├── runs/                     # Directory for saving models and logs
├── main.py                   # Main script to run the training
├── model.py                  # Defines the encoder and projection head
├── dataset.py                # Dataset generation and PyTorch Dataset class
├── utils.py                  # Utility functions, including the loss function
├── config.py                 # Configuration management
├── evaluate.py               # Linear probing evaluation script
├── preprocess_clevrer.py     # CLEVRER preprocessing script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

###  Using CLEVRER Dataset

1. Download CLEVRER dataset from [http://clevrer.csail.mit.edu/](http://clevrer.csail.mit.edu/)
2. Preprocess the videos:

```bash
# For dataset with subdirectory structure (video_00000-01000/, etc.)
python preprocess_clevrer.py --clevrer_root ./clevrer_data --output_dir processed_clevrer --max_videos 1000

# Or for standard CLEVRER structure
python preprocess_clevrer.py --clevrer_root /path/to/clevrer --output_dir processed_clevrer --max_videos 1000
```

3. Train the model:

```bash
# Single GPU training
python main.py --processed_dir processed_clevrer --epochs 100 --batch_size 32

# Multi-GPU training with DDP (8 GPUs)
python main.py --ddp --processed_dir processed_clevrer --epochs 100 --batch_size 256

# Or use the convenience script
chmod +x train_ddp.sh
./train_ddp.sh

# Train with ViViT transformer model
python main.py --ddp --base_model vivit_small --batch_size 128 --lr 0.0001 --processed_dir processed_clevrer

# Or use the ViViT training script
chmod +x train_vivit_ddp.sh
./train_vivit_ddp.sh
```

### Resume Training

```bash
python main.py --resume runs/checkpoint_epoch_50.pth --processed_dir processed_clevrer
```

### Evaluate with Linear Probing

After training, evaluate the learned representations:

```bash
python evaluate.py --checkpoint runs/final_model.pth
```

## Key Components

### Dataset (`dataset.py`)

- **CLEVRERDataset**: PyTorch Dataset for loading CLEVRER video clips
- **preprocess_clevrer_dataset()**: Extracts and saves frames from CLEVRER videos
- **extract_frames_from_video()**: Efficiently samples frames from video files
- Supports consistent augmentations across all frames in a video clip

### Model Architecture (`model.py`)

- **Encoder**: Supports both CNN (ResNet) and Transformer (ViViT) architectures
  - ResNet: Processes frames independently then aggregates with mean pooling
  - ViViT: Processes entire video sequences with spatiotemporal attention
- **ProjectionHead**: MLP that maps features to a lower-dimensional space for contrastive learning
- **SimCLRModel**: Combined encoder and projection head

### ViViT Support (`vivit_model.py`)

- **Vision Transformer for Videos**: State-of-the-art video understanding architecture
- **Spatiotemporal Attention**: Captures both spatial and temporal relationships
- **Multiple Model Sizes**: 
  - `vivit_tiny`: 192 dim, 3 heads (fastest, least memory)
  - `vivit_small`: 384 dim, 6 heads (balanced)
  - `vivit_base`: 768 dim, 12 heads (best performance)

### Training (`main.py`)

- Implements the SimCLR training loop with NT-Xent loss
- Supports checkpointing and resuming
- Configurable through command-line arguments or `config.py`

### Evaluation (`evaluate.py`)

- Linear probing: Trains a linear classifier on frozen encoder features
- Task: Classify the number of balls (1, 2, or 3) in a video
- Compares against random feature baseline

## Distributed Training (DDP)

This implementation supports efficient multi-GPU training using PyTorch's DistributedDataParallel:

- **Automatic GPU detection**: Uses all available GPUs by default
- **Linear scaling**: Learning rate is automatically scaled with the number of GPUs
- **Efficient data loading**: Each GPU processes a subset of the data
- **Synchronized training**: Gradients are averaged across all GPUs

### DDP Usage:
```bash
# Use all available GPUs
python main.py --ddp --batch_size 256 --processed_dir processed_clevrer

# Use specific number of GPUs
python main.py --ddp --num_gpus 4 --batch_size 128 --processed_dir processed_clevrer
```

### DDP Performance Tips:
- Use a batch size of 32 per GPU (e.g., 256 for 8 GPUs)
- The learning rate is automatically scaled by the number of GPUs
- Ensure your dataset is large enough to benefit from multi-GPU training

## Configuration

All hyperparameters can be configured in `config.py` or overridden via command-line arguments:

- `batch_size`: Total training batch size across all GPUs (default: 256)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 3e-4)
- `temperature`: Temperature for NT-Xent loss (default: 0.5)
- `num_frames`: Number of frames per video (default: 20)
- `box_size`: Frame size in pixels (default: 256 for CLEVRER)

## Data Augmentations

The framework applies consistent spatial augmentations across all frames in a video clip:
- Random resized crop
- Random horizontal flip
- Color jittering (brightness, contrast, saturation, hue)
- Normalization with ImageNet statistics

This ensures that the same physical event is captured from different "viewpoints".

## Results

After training, the model learns representations that capture physical properties of the scenes. The linear probing evaluation measures how well these representations transfer to downstream tasks.

Expected results:
- Random baseline: ~33% accuracy (random guessing for 3 classes)
- Trained SimCLR: 80-95% accuracy (depending on training duration and hyperparameters)

## CLEVRER Dataset

This implementation now supports the [CLEVRER dataset](http://clevrer.csail.mit.edu/), which provides:
- High-quality 3D rendered videos of collision events
- Complex physics interactions between objects
- Annotations for various reasoning tasks

To use CLEVRER:
1. Download the dataset from the official website
2. Use the preprocessing script to extract frames
3. Train SimCLR on the extracted frames

## Extending the Framework

This baseline can be extended to:
1. Use more complex physics simulations (e.g., from Blender)
2. Add more sophisticated augmentations
3. Implement different downstream tasks (e.g., collision prediction, object counting)
4. Try different encoder architectures (3D CNNs, Vision Transformers)
5. Experiment with different aggregation strategies for temporal features

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or number of frames
2. **Dataset not found**: For CLEVRER, run preprocessing first with `python preprocess_clevrer.py`
3. **Poor performance**: Train for more epochs or adjust learning rate
4. **OpenCV import error**: Install with `pip install opencv-python`

## Citation

This implementation is based on the SimCLR paper:
```
@article{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```
