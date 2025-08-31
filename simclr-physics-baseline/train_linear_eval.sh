#!/bin/bash

# Linear Evaluation Training Script for SimCLR
# Usage: ./train_linear_eval.sh

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=7

# Run linear evaluation
python linear_eval.py \
    --checkpoint_path /data/yaxin/phy_simclr/simclr-physics-baseline/runs_deepspeed_vit_imagenet/epoch_20 \
    --data_dir /home/yaxin/imagenet \
    --batch_size 256 \
    --epochs 90 \
    --lr 0.1 \
    --weight_decay 1e-6 \
    --num_workers 4 \
    --save_dir linear_eval_results \
    --device cuda \
    "$@"  # Allow passing additional arguments
