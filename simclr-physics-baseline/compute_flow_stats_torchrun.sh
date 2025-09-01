#!/bin/bash

# Distributed flow stats computation using torchrun across 8 GPUs.
# Requires OpenCV with CUDA for GPU optical flow (optional).

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_DIR=/data/yaxin/phy_simclr/simclr-physics-baseline/synthetic_videos
OUT=flow_stats.json

torchrun --nproc_per_node=8 \
  compute_flow_stats.py \
  --root_dir ${ROOT_DIR} \
  --output ${OUT} \
  --window 16 \
  --stride 16 \
  --use_cuda_flow \
  --distributed \
  "$@"

