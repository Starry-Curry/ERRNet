#!/usr/bin/env bash
set -euo pipefail

# Linux server example. Prepare datasets and checkpoints manually before running.
# This uses the architecture-matched ERRNet --hyper pretrained initialization.
python train_rap_errnet.py \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --name rap_errnet_hyper_pretrained \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 8 \
  --num_workers 6 \
  --progress_bar

python eval_all.py \
  --model rap_errnet \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --ckpt checkpoints/rap_errnet_hyper_pretrained/best.pt \
  --data_root ./data \
  --save_dir results/rap_errnet_hyper_pretrained \
  --save_images
