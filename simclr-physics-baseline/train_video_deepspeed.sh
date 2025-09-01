#!/bin/bash

# DeepSpeed Training Script for VideoSimCLR on saved synthetic videos
# Usage: ./train_video_deepspeed.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1

# Path to your saved synthetic videos directory
DATA_DIR=/data/yaxin/phy_simclr/simclr-physics-baseline/synthetic_videos

# Launch DeepSpeed across 8 GPUs
deepspeed --num_gpus=8 train_video.py \
  --deepspeed \
  --deepspeed_config deepspeed_config.json \
  --data_dir ${DATA_DIR} \
  --base_model vit_base \
  --img_size 224 \
  --projection_dim 128 \
  --temperature 0.2 \
  --epochs 10 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 0.05 \
  --temporal transformer \
  --use_proxy_sampler \
  --k_pos 5 \
  --checkpoint_dir video_runs_deepspeed \
  "$@"
