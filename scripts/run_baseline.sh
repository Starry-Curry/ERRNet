#!/usr/bin/env bash
set -euo pipefail

# Linux server example. Do not run this on Windows PowerShell.
python train_errnet.py --name errnet_baseline --hyper
python eval_all.py \
  --model errnet \
  --ckpt checkpoints/errnet_baseline/latest.pt \
  --data_root ./data \
  --save_dir results/errnet_baseline \
  --save_images

