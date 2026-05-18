#!/usr/bin/env bash
set -euo pipefail

# Linux server example ablations. Prepare datasets manually before running.
python train_rap_errnet.py --name rap_no_prior --data_root ./data --use_physics_synthesis --no_prior_head --use_gated_blocks --use_refinement
python train_rap_errnet.py --name rap_no_gating --data_root ./data --use_physics_synthesis --use_prior_head --no_gated_blocks --use_refinement
python train_rap_errnet.py --name rap_no_refine --data_root ./data --use_physics_synthesis --use_prior_head --use_gated_blocks --no_refinement

