#!/bin/bash

# Alternative DeepSpeed Training Script using torchrun
# Usage: ./train_torchrun_deepspeed.sh

# Set CUDA visible devices (use 4 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# Optional: Set OMP_NUM_THREADS to prevent CPU oversubscription
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 
# Use torchrun for more reliable distributed initialization
# This approach often avoids the TCP store connection issues
torchrun --nproc_per_node=7 \
    --master_addr=localhost \
    --master_port=29500 \
    main.py \
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
    --resume /data/yaxin/phy_simclr/simclr-physics-baseline/runs_deepspeed_vit_imagenet/epoch_20
    "$@"  # Allow passing additional arguments
