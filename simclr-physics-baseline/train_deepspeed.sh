#!/bin/bash

# DeepSpeed Training Script for SimCLR with ViT and ZeRO-1
# Usage: ./train_deepspeed.sh

# Set CUDA visible devices (use all 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: Set OMP_NUM_THREADS to prevent CPU oversubscription
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 
# DeepSpeed distributed training
deepspeed --num_gpus=8 main.py \
    --deepspeed \
    --deepspeed_config deepspeed_config.json \
    --data_dir /home/yaxin/imagenet \
    --base_model vit_base \
    --pretrained \
    --batch_size 4096 \
    --epochs 100 \
    --lr 1e-3 \
    --temperature 0.1 \
    --checkpoint_dir runs_deepspeed_vit_imagenet \
    --resume /data/yaxin/phy_simclr/simclr-physics-baseline/runs_deepspeed_vit_imagenet/epoch_20 \
    "$@"  # Allow passing additional arguments
