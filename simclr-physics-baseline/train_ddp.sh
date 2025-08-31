#!/bin/bash

# DDP Training Script for SimCLR with ViT
# Usage: ./train_ddp.sh

# Set CUDA visible devices (use all 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: Set OMP_NUM_THREADS to prevent CPU oversubscription
export OMP_NUM_THREADS=4


# Run training with DDP and ViT (with pretrained ImageNet weights)
# Using 224x224 resolution to match pretrained ViT models
# Models: google/vit-base-patch16-224, google/vit-large-patch16-224
python main.py \
    --ddp \
    --data_dir /data/yaxin/data/imagenet \
    --batch_size 1024 \
    --epochs 100 \
    --lr 1e-3 \
    --temperature 0.3 \
    --checkpoint_dir runs_vit_imagenet_ddp \
    --base_model vit_base \
    "$@"  # Allow passing additional arguments
