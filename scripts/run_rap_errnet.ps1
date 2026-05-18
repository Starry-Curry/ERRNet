param(
    [string]$DataRoot = ".\data",
    [string]$Name = "rap_errnet_main"
)

# Windows example. This starts training only when you explicitly run it.
python train_rap_errnet.py `
  --config configs/rap_errnet.yaml `
  --name $Name `
  --data_root $DataRoot `
  --use_physics_synthesis `
  --use_prior_head `
  --use_gated_blocks `
  --use_refinement `
  --num_workers 0

