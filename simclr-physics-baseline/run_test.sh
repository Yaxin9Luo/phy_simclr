#!/bin/bash

# Test Linear Classifier Script
# Usage: ./run_test.sh

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=7

# Run testing
python test_linear_eval.py \
    --model_path /data/yaxin/phy_simclr/simclr-physics-baseline/linear_eval_results/best_model.pth \
    --simclr_checkpoint /data/yaxin/phy_simclr/simclr-physics-baseline/runs_deepspeed_vit_imagenet/epoch_20 \
    --data_dir /home/yaxin/imagenet \
    --batch_size 256 \
    --num_workers 8 \
    --device cuda \
    --save_results test_results.json \
    "$@"  # Allow passing additional arguments
