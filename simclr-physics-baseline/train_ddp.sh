#!/bin/bash

# DDP Training Script for SimCLR
# Usage: ./train_ddp.sh

# Set CUDA visible devices (use all 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: Set OMP_NUM_THREADS to prevent CPU oversubscription
export OMP_NUM_THREADS=4

echo "Starting DDP training on 8 GPUs..."
echo "Total batch size: 256 (32 per GPU)"

# Run training with DDP
python main.py \
    --ddp \
    --processed_dir processed_clevrer \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.0003 \
    --temperature 0.5 \
    --checkpoint_dir runs_ddp \
    "$@"  # Allow passing additional arguments
