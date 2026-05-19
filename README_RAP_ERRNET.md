# RAP-ERRNet

RAP-ERRNet means **Reflection-Aware Physics-guided ERRNet**. It keeps the course ERRNet baseline usable, then adds physics-guided synthesis, a reflection prior head, optional prior-gated residual adaptation, and lightweight residual refinement.

## 1. Project Summary

The task is single image reflection removal: given a blended image `I`, recover the clean transmission image `T`. This implementation is for the course project code stage on Windows: code, documentation, and syntax checks only. Large datasets are not downloaded by any script in this repository.

## 2. Relation To ERRNet Baseline

The original baseline entry points remain:

```bash
python train_errnet.py --name errnet --hyper
python test_errnet.py --name errnet --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```

RAP-ERRNet is added through new files. `models/rap_errnet.py` reuses `models.arch.errnet` as a coarse backbone. The code does not rewrite the original DRNet residual blocks. `use_gated_blocks` enables a lightweight prior-gated adapter after the coarse output, so the baseline code path stays intact.

Two backbone modes are supported:

- `configs/rap_errnet.yaml`: 3-channel RGB backbone, useful for quick local/debug runs and from-scratch ablations.
- `configs/rap_errnet_hyper_pretrained.yaml`: ERRNet `--hyper` backbone, RGB plus VGG19 hypercolumn features, 1475 channels total. This is the recommended main experiment because it can fully load the course pretrained `--hyper` ERRNet checkpoint.

This fork was found under `D:\Classes\DIP\DIP-Lab\ERRNet`, while the design document is in the parent directory.

## 3. RAP-ERRNet Improvements

1. Physics-guided reflection synthesis in `datasets/reflection_synthesis.py`.
2. Reflection prior prediction head in `models/modules/reflection_prior.py`.
3. Prior-gated residual adapter in `models/modules/gated_blocks.py`.
4. Lightweight residual refinement in `models/modules/refinement.py`.

All main additions are controlled by flags: `--use_prior_head`, `--use_gated_blocks`, `--use_refinement`, `--use_hypercolumn_backbone`, and their `--no_*` ablation variants.

## 4. Dataset Links And Layout

Recommended root:

```text
data/
  VOC2012/
    JPEGImages/
    VOC2012_224_train_png.txt
    cropped_224/train/
  zhang2018/
    train/blended/
    train/transmission_layer/
    test/blended/
    test/transmission_layer/
  ceilnet/testdata_reflection_synthetic_table2/
    blended/
    transmission_layer/
  sir2/
    Objects/
    Postcard/
    Wild/
  openrr5k/
    train/blended/
    train/transmission/
    val/blended/
    val/transmission/
  self_collected/test/scene_001/
    blended.png
    transmission.png
```

Reference links: ERRNet fork `https://github.com/innerway-xq/ERRNet`, PASCAL VOC 2012 official page, Zhang/Berkeley reflection dataset `https://github.com/ceciliavision/perceptual-reflection-removal`, CEILNet `https://github.com/fqnchina/CEILNet`, SIR2 `https://sir2data.github.io/`, OpenRR-5k `https://github.com/caijie0620/OpenRR-5k`.

## 5. Windows Local Development

Windows local stage should only compile and import-check with random tensors. Do not download datasets, run training, or execute `.sh` files locally.

```powershell
python -m py_compile train_rap_errnet.py eval_all.py visualize_results.py
python -m py_compile models/rap_errnet.py models/modules/reflection_prior.py models/modules/refinement.py models/modules/gated_blocks.py
python -m py_compile datasets/reflection_synthesis.py datasets/unified_reflection_dataset.py losses/reflection_losses.py metrics/reflection_metrics.py
```

## 6. Linux Server Training

Prepare datasets manually on the server first. The scripts only read existing files.

```bash
python scripts/prepare_voc_crops.py \
  --input_dir data/VOC2012/JPEGImages \
  --output_dir data/VOC2012/cropped_224/train \
  --list_file data/VOC2012/VOC2012_224_train_png.txt
```

## 7. Baseline Commands

```bash
python train_errnet.py --name errnet_baseline --hyper
python test_errnet.py --name errnet_baseline --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet_baseline/latest.pt --hyper
python eval_all.py --model errnet --ckpt checkpoints/errnet_baseline/latest.pt --data_root ./data --save_dir results/errnet_baseline --save_images
```

Note: original `--hyper` baseline checkpoints are still best evaluated with `test_errnet.py --hyper` so the baseline path stays exactly comparable to the course code.

## 8. RAP-ERRNet Commands

```bash
python train_rap_errnet.py \
  --config configs/rap_errnet.yaml \
  --name rap_errnet_main \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement

python eval_all.py \
  --model rap_errnet \
  --ckpt checkpoints/rap_errnet_main/best.pt \
  --data_root ./data \
  --save_dir results/rap_errnet_main \
  --save_images
```

Recommended A6000/server main run with full ERRNet hypercolumn pretrained initialization:

```bash
python train_rap_errnet.py \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --name rap_errnet_hyper_pretrained \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 8 \
  --epochs 100 \
  --num_workers 6 \
  --device auto \
  --progress_bar

python eval_all.py \
  --model rap_errnet \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --ckpt checkpoints/rap_errnet_hyper_pretrained/best.pt \
  --data_root ./data \
  --save_dir results/rap_errnet_hyper_pretrained \
  --save_images
```

The `--config` argument is required during evaluation for hypercolumn RAP checkpoints, because the model must be reconstructed with the same 1475-channel backbone used during training.

## 9. Ablation Commands

```bash
python train_rap_errnet.py --name rap_no_prior --data_root ./data --use_physics_synthesis --no_prior_head --use_gated_blocks --use_refinement
python train_rap_errnet.py --name rap_no_gating --data_root ./data --use_physics_synthesis --use_prior_head --no_gated_blocks --use_refinement
python train_rap_errnet.py --name rap_no_refine --data_root ./data --use_physics_synthesis --use_prior_head --use_gated_blocks --no_refinement
```

## 10. Self-Collected Data

Use one folder per scene:

```text
data/self_collected/test/scene_001/
  blended.png
  transmission.png
  mask.png          # optional
  meta.json         # optional
```

If `mask.png` is missing, the dataset adapter creates a pseudo mask from `abs(input - target)`.

## 11. FAQ

`lambda_perc` defaults to `0.0`. The loss code will not download VGG weights automatically. If you want perceptual loss on the server, provide local weights and enable it in the config.

If a dataset is missing, `eval_all.py` warns and skips it. Training raises a clear error when no usable training dataset is found.

The current `use_gated_blocks` implementation is a compatibility adapter, not an invasive replacement of ERRNet internal residual blocks.

For hypercolumn pretrained fine-tuning, the added gated/refinement residual
branches are initialized to produce zero correction. This keeps the initial
RAP-ERRNet output aligned with the pretrained ERRNet baseline and avoids
randomly perturbing strong synthetic-benchmark performance before training.

If a RAP checkpoint was trained with `use_hypercolumn_backbone: true`, evaluate it with the matching config. Otherwise `eval_all.py` will intentionally stop with a clear backbone shape mismatch instead of silently skipping the first convolution.

## 12. Tables And Visualizations

Evaluation writes:

```text
results/<name>/metrics_all.csv
results/<name>/metrics_<dataset>.csv
results/<name>/outputs/
results/<name>/visualizations/
```

Build contact sheets:

```bash
python visualize_results.py --input_dir results/rap_errnet_main --output_dir results/rap_errnet_main/contact_sheets
```
