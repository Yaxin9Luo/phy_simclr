#!/bin/bash

# DDP Training Script for SimCLR with ViViT
# Usage: ./train_vivit_ddp.sh

# Set CUDA visible devices (use all 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: Set OMP_NUM_THREADS to prevent CPU oversubscription
export OMP_NUM_THREADS=4

echo "Starting DDP training with ViViT on 8 GPUs..."
echo "Model: ViViT-Small"
echo "Total batch size: 128 (16 per GPU)"

# Run training with DDP and ViViT
python main.py \
    --ddp \
    --processed_dir processed_clevrer \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0001 \
    --temperature 0.5 \
    --checkpoint_dir runs_vivit_ddp \
    --base_model vivit_large \
    "$@"  # Allow passing additional arguments
