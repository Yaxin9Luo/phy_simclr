# ViViT Implementation Comparison

This document compares our PyTorch implementation with the [official Google Scenic ViViT implementation](https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/model.py).

## Key Similarities ✓

1. **Model Architecture**: We implement the "Factorized encoder" (Model 2) from the ViViT paper, which is the most efficient variant
2. **Patch Embedding**: 3D patch embedding with configurable patch size (default: 4×16×16)
3. **Positional Embeddings**: Separate spatial and temporal embeddings
4. **Temporal Embeddings**: Support for both learnable and sinusoidal temporal embeddings
5. **Model Configurations**: Tiny, Small, Base, and Large variants with matching dimensions

## Key Differences

### 1. Framework
- **Official**: JAX/Flax implementation
- **Ours**: PyTorch implementation

### 2. Positional Embedding Initialization
- **Official**: Uses specific initialization schemes from Scenic
- **Ours**: Standard PyTorch initialization (Xavier uniform for patches, normal for embeddings)

### 3. Attention Implementation
- **Official**: Uses Flax's MultiHeadDotProductAttention with additional features
- **Ours**: Standard PyTorch multi-head attention

### 4. Training Features
- **Official**: Includes stochastic depth, gradient clipping, and other training tricks
- **Ours**: Simplified for SimCLR training

### 5. Pretrained Weights
- **Official**: Supports loading ImageNet-21K pretrained weights
- **Ours**: Training from scratch (no pretrained weights available)

## Model Specifications

| Model | Embed Dim | Depth | Heads | Params |
|-------|-----------|-------|-------|---------|
| Tiny  | 192       | 12    | 3     | ~6M     |
| Small | 384       | 12    | 6     | ~22M    |
| Base  | 768       | 12    | 12    | ~87M    |
| Large | 1024      | 24    | 16    | ~305M   |

## Implementation Details

### Factorized Spatial-Temporal Encoding
Our implementation follows Model 2 from the paper:
1. Extract 3D patches from video
2. Add spatial positional embeddings to each frame
3. Add temporal embeddings across frames
4. Process with transformer blocks
5. Aggregate temporal information via mean pooling

### Temporal Aggregation
- **Official**: Supports multiple aggregation methods (mean, max, learned)
- **Ours**: Mean pooling of CLS tokens across temporal dimension

### Input Requirements
- Video shape: (B, T, C, H, W)
- Default: T=20 frames, H=W=256 pixels
- Patches: 4×16×16 (temporal×height×width)

## Usage Recommendations

1. **For CLEVRER dataset**: Use `vivit_small` or `vivit_base`
2. **Memory constraints**: Use `vivit_tiny` for limited GPU memory
3. **Best performance**: Use `vivit_base` with sufficient compute
4. **Batch size**: Reduce compared to ResNet due to higher memory usage

## Future Improvements

1. Add stochastic depth for better regularization
2. Implement MAE-style pretraining for better initialization
3. Add support for variable-length videos
4. Implement additional aggregation methods
5. Add gradient checkpointing for larger models
