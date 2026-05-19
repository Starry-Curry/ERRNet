# RAP-ERRNet Experiment Log

Last updated: 2026-05-19

This document records the implementation and experiment progress for the course
project method **RAP-ERRNet: Reflection-Aware Physics-guided ERRNet**.

## 1. Current Status

The project has moved from implementation to experiment production.

| Area | Status | Notes |
| --- | --- | --- |
| RAP code implementation | Done | Model, synthesis, dataset adapter, losses, metrics, train/eval scripts are implemented. |
| Local syntax checks | Done | `py_compile` passed for new modules. |
| A6000 environment | Done | ERRNet baseline and RAP debug training both run successfully. |
| Baseline pretrained evaluation | Mostly done | CEILNet, real20, SIR2 Objects/Postcard/Wild have been evaluated. |
| RAP main training | Running | `rap_errnet_main`, from scratch, 100 epochs planned. |
| RAP hyper-pretrained path | Ready | Config uses the same ERRNet `--hyper` 1475-channel backbone as the course baseline. |
| RAP staged zero-res path | Ready | `configs/rap_errnet_hyper_zerores_staged.yaml` freezes the backbone first, then uses differential LR fine-tuning. |
| RAP extra-data fine-tune path | Ready | `configs/rap_errnet_hyper_zerores_extra_finetune.yaml` resumes model weights with reset epoch/optimizer for short OpenRR or `extra_train` adaptation. |
| RAP full evaluation | Pending | Run after mid/final checkpoint is available. |
| Ablation experiments | Pending | No-prior, no-gating, no-refinement variants. |
| Self-collected data | Pending | Need at least 5 paired scenes. |

## 2. Implementation Iterations

| Commit | Summary |
| --- | --- |
| `612b075` | Added RAP-ERRNet implementation: prior head, gated block, refinement, synthesis, dataset adapter, loss, metrics, train/eval/visualization scripts and README. |
| `6803b64` | Fixed unified dataset batch keys by adding `reflection` for paired real samples. |
| `9eb02da` | Reduced repeated missing-mask warnings and added per-epoch batch progress logging. |
| `de3f3dd` | Added dynamic progress bar for RAP training. |
| `dde1a8a` | Cleaned progress-bar output so dynamic progress and line logs do not interleave. |
| `c4d9e5d` | Recorded mid-training RAP evaluation and baseline comparison. |
| current update | Added staged zero-residual RAP config, differential LR parameter groups, optional extra training data, anchor/delta losses, and pseudo-mask downweighting. |

## 3. Environment And Data

Main training machine:

```text
NVIDIA RTX A6000, using CUDA_VISIBLE_DEVICES=2
Conda env: rap-errnet
```

Prepared data:

```text
datasets/processed_data/
  VOCdevkit/VOC2012/PNGImages/      15287 images
  real_train/blended/               89 images
  real_train/transmission_layer/    89 images
  testdata_CEILNET_table2/          100 pairs
  real20/                           20 pairs
  objects/                          200 pairs
  postcard/                         179 pairs
  wild/                             101 pairs
```

RAP data links are available under `data/`:

```text
data/VOC2012/cropped_224/train
data/zhang2018/train
data/zhang2018/test
data/ceilnet/testdata_reflection_synthetic_table2
data/sir2/Objects
data/sir2/Postcard
data/sir2/Wild
```

Dataset availability check:

```text
voc=True
zhang_train=True
zhang20=True
ceilnet=True
sir2_objects=True
sir2_postcard=True
sir2_wild=True
openrr_train=False
openrr_val=False
self=False
```

## 4. Baseline Evaluation

Baseline uses the assistant-provided/course ERRNet pretrained checkpoint:

```text
checkpoints/errnet/errnet_060_00463920.pt
```

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=2 python test_errnet.py \
  --name errnet_a6000 \
  --dataset <dataset> \
  -r \
  --gpu_ids 0 \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --hyper
```

Recorded baseline metrics:

| Dataset | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| CEILNet Table2 | 27.8765 | 0.9407 | 0.9808 | 0.0048 |
| Zhang real20 | 23.5531 | 0.8285 | 0.8877 | 0.0201 |
| SIR2 Objects | 24.8533 | 0.8980 | 0.9817 | 0.0029 |
| SIR2 Postcard | 22.0702 | 0.8773 | 0.9463 | 0.0044 |
| SIR2 Wild | 25.1778 | 0.8861 | 0.9359 | 0.0083 |

These values match the course README baseline closely, so the baseline
reproduction is considered valid. Baseline retraining is not required unless the
report explicitly needs a from-scratch baseline.

## 5. RAP Main Training

Current experiment:

```text
Experiment name: rap_errnet_main
Initialization: from scratch
Config: configs/rap_errnet.yaml
Backbone pretrained path: null
Training data: VOC physics synthesis + Zhang real_train
Samples: 15376
Batches per epoch: 961
Batch size: 16
Epochs: 100
```

Command:

```bash
CUDA_VISIBLE_DEVICES=2 python train_rap_errnet.py \
  --config configs/rap_errnet.yaml \
  --name rap_errnet_main \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 16 \
  --epochs 100 \
  --num_workers 8 \
  --device auto \
  --progress_bar
```

Training loss snapshots:

| Epoch | Total | Pix | Grad | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.491144 | 0.348003 | 0.068579 | Initial epoch after stable run. |
| 35 | 0.124302 | 0.064475 | 0.015117 | Training already much lower than early stage. |
| 40 | 0.120442 | 0.061090 | 0.014793 | Continued stable decrease. |
| 45 | 0.116881 | 0.057954 | 0.014496 | No divergence. |
| 49 | 0.114482 | 0.055872 | 0.014293 | Slow but steady decrease. |
| 50 | 0.113675 | 0.055135 | 0.014238 | Good mid-training checkpoint. |
| 51 partial | 0.1134 | 0.0549 | 0.0142 | Ongoing, about 58% through epoch when observed. |
| 100 | 0.094574 | 0.038360 | 0.013172 | Final from-scratch run converged, but test metrics remain below baseline on CEILNet/Zhang20. |

Interpretation:

- Training is healthy: total loss, pixel loss, and gradient loss all decrease.
- The loss curve is flattening after epoch 35, which is expected.
- Train loss alone does not prove metric improvement; PSNR/SSIM/NCC/LMSE must be
  checked on the held-out test sets.

Estimated speed:

```text
About 15-16 minutes per epoch.
100 epochs: about 25-27 hours.
```

## 6. Mid-Training Evaluation Plan

It is valid to start mid-training evaluation after an epoch checkpoint is saved.
To avoid reading a checkpoint while the training process is writing it, copy a
snapshot first:

```bash
cd /home/wangyihan/zc/DIP/ERRNet
cp checkpoints/rap_errnet_main/best.pt checkpoints/rap_errnet_main/best_mid_snapshot.pt
```

Then evaluate, preferably on a GPU not used by training if available:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_all.py \
  --model rap_errnet \
  --ckpt checkpoints/rap_errnet_main/best_mid_snapshot.pt \
  --data_root ./data \
  --save_dir results/rap_errnet_main_mid \
  --save_images \
  --device auto
```

If no spare GPU is available, wait until the training job finishes or pause the
training job before evaluation.

Expected output:

```text
results/rap_errnet_main_mid/metrics_all.csv
results/rap_errnet_main_mid/metrics_ceilnet.csv
results/rap_errnet_main_mid/metrics_zhang20.csv
results/rap_errnet_main_mid/metrics_sir2_objects.csv
results/rap_errnet_main_mid/metrics_sir2_postcard.csv
results/rap_errnet_main_mid/metrics_sir2_wild.csv
results/rap_errnet_main_mid/visualizations/
```

## 7. Mid-Training Evaluation Result

Snapshot:

```text
Checkpoint: checkpoints/rap_errnet_main/best_mid_snapshot.pt
Training stage: about epoch 50, from-scratch RAP
Save dir: results/rap_errnet_main_mid
```

Metrics:

| Dataset | RAP PSNR | RAP SSIM | RAP NCC | RAP LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Mid-Eval Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEILNet Table2 | 19.0065 | 0.8315 | 0.8734 | 0.0136 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | Much worse; synthetic CEILNet is not recovered well by from-scratch RAP at this stage. |
| Zhang real20 | 19.4201 | 0.7303 | 0.8101 | 0.0214 | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Worse; real20 generalization needs improvement. |
| SIR2 Objects | 25.7091 | 0.9082 | 0.9853 | 0.0027 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Improved on all metrics. |
| SIR2 Postcard | 20.9513 | 0.8783 | 0.9498 | 0.0041 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Mixed: PSNR lower, SSIM/NCC/LMSE slightly better. |
| SIR2 Wild | 25.0728 | 0.9063 | 0.9490 | 0.0048 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Strong structural/local improvement; PSNR nearly tied. |

Interpretation:

- The method shows promising gains on SIR2 Objects and Wild, especially SSIM,
  NCC, and LMSE.
- The current from-scratch run is not competitive on CEILNet Table2 and Zhang
  real20.
- This suggests that the added physics/prior/refinement components can help
  real-world structural quality, but the 3-channel from-scratch backbone is too
  weak compared with the pretrained hypercolumn ERRNet on some benchmarks.
- Recommended next step: keep the current run as `RAP from scratch`, but start a
  `RAP pretrained-init` run as the likely main result candidate.

## 7.1 Final From-Scratch Evaluation Result

Snapshot:

```text
Checkpoint: checkpoints/rap_errnet_main/best.pt
Training stage: epoch 100, from-scratch RAP
Save dir: results/rap_errnet_main_final_best
```

Metrics:

| Dataset | RAP PSNR | RAP SSIM | RAP NCC | RAP LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Final Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Average | 22.9575 | 0.8844 | 0.9464 | 0.0056 | - | - | - | - | Average is not directly comparable to baseline because baseline was recorded per dataset. |
| CEILNet Table2 | 19.1752 | 0.8395 | 0.8802 | 0.0127 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | Still far below baseline; from-scratch 3-channel RAP is not suitable as the main result. |
| Zhang real20 | 19.5196 | 0.7320 | 0.8173 | 0.0209 | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Still worse than baseline. |
| SIR2 Objects | 25.7196 | 0.9109 | 0.9855 | 0.0028 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Improves structural metrics and PSNR over baseline. |
| SIR2 Postcard | 20.8931 | 0.8816 | 0.9508 | 0.0038 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Mixed: PSNR lower, SSIM/NCC/LMSE better. |
| SIR2 Wild | 25.5726 | 0.9119 | 0.9524 | 0.0045 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Clearly improves SSIM/NCC/LMSE and slightly improves PSNR. |

Interpretation:

- The from-scratch model learned useful reflection-aware behavior on SIR2,
  especially real-world structural quality.
- It does not preserve the strong synthetic benchmark behavior of pretrained
  ERRNet, with CEILNet remaining about 8.7 dB below the baseline.
- This confirms the project should not use from-scratch RAP as the main method.
  It should be reported as a diagnostic/reference experiment showing that the
  added prior/refinement idea helps some real subsets but needs pretrained
  ERRNet anchoring for stable benchmark performance.

## 8. Next Experiments

### 8.1 Finish Main Run

Let `rap_errnet_main` complete to 100 epochs unless validation metrics clearly
show severe degradation.

Final evaluation:

```bash
CUDA_VISIBLE_DEVICES=2 python eval_all.py \
  --model rap_errnet \
  --ckpt checkpoints/rap_errnet_main/best.pt \
  --data_root ./data \
  --save_dir results/rap_errnet_main \
  --save_images \
  --device auto
```

### 8.2 Pretrained-Backbone Run

Because the method is positioned as an ERRNet improvement, the strongest next
run should use the same `--hyper` input path as the pretrained baseline. The new
config is:

```text
configs/rap_errnet_hyper_pretrained.yaml
```

This config sets:

```text
use_hypercolumn_backbone: true
pretrained_errnet_path: checkpoints/errnet/errnet_060_00463920.pt
```

That means the first ERRNet convolution is `1475 -> 256`, so the course
pretrained `--hyper` checkpoint can be loaded completely instead of skipping the
first layer.

```bash
CUDA_VISIBLE_DEVICES=0 python train_rap_errnet.py \
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
```

Evaluate with the same config:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_all.py \
  --model rap_errnet \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --ckpt checkpoints/rap_errnet_hyper_pretrained/best.pt \
  --data_root ./data \
  --save_dir results/rap_errnet_hyper_pretrained \
  --save_images \
  --device auto
```

This is now the recommended main result candidate, because the from-scratch
mid-eval is weak on CEILNet and Zhang real20, and a 3-channel RAP backbone is
not architecture-matched to the pretrained `--hyper` baseline.

### 8.3 Ablations

Minimum recommended ablations:

```bash
# no prior head
CUDA_VISIBLE_DEVICES=2 python train_rap_errnet.py --config configs/rap_errnet.yaml --name rap_no_prior --data_root ./data --use_physics_synthesis --no_prior_head --use_gated_blocks --use_refinement --batch_size 16 --epochs 30 --num_workers 8 --device auto --progress_bar

# no gated adapter
CUDA_VISIBLE_DEVICES=2 python train_rap_errnet.py --config configs/rap_errnet.yaml --name rap_no_gating --data_root ./data --use_physics_synthesis --use_prior_head --no_gated_blocks --use_refinement --batch_size 16 --epochs 30 --num_workers 8 --device auto --progress_bar

# no refinement
CUDA_VISIBLE_DEVICES=2 python train_rap_errnet.py --config configs/rap_errnet.yaml --name rap_no_refine --data_root ./data --use_physics_synthesis --use_prior_head --use_gated_blocks --no_refinement --batch_size 16 --epochs 30 --num_workers 8 --device auto --progress_bar
```

For the report, a 30-epoch ablation is acceptable as a first-pass trend if time
is limited. If a variant is close to the main model, extend it to 100 epochs.

## 9. Remaining Deliverables

1. Finish RAP main training and evaluate all test sets.
2. Compare RAP metrics against the baseline table in Section 4.
3. Generate qualitative visualizations from `results/*/visualizations`.
4. Collect at least 5 self-collected paired scenes:

```text
data/self_collected/test/scene_001/blended.png
data/self_collected/test/scene_001/transmission.png
...
```

5. Run self-collected evaluation after adding those scenes.
6. Run ablations and prepare the ablation table.
7. Write final paper and PPT using:

```text
Method: physics synthesis + prior head + gated adapter + refinement
Baseline: pretrained ERRNet
Metrics: PSNR, SSIM, NCC, LMSE
Visuals: Input | Output | GT | Error Map | Prior Map
```

## 10. Current Risk Notes

- The current main RAP run is from scratch. This is valid, but it may need more
  epochs to beat a pretrained ERRNet baseline.
- A hypercolumn pretrained RAP run is recommended as the main next experiment.
- `eval_all.py --model errnet` is not the preferred baseline path for the
  hypercolumn pretrained ERRNet; use `test_errnet.py --hyper` for baseline
  metrics.
- Self-collected data is still missing and is required by the course project.

## 11. Dataset Name Mapping

The baseline table in `README_DIP26.md` and `test_errnet.py` uses the original
course dataset names. `eval_all.py` uses more explicit unified-dataset aliases.
The underlying processed images are the same when the `data/` symlinks point to
`datasets/processed_data`.

| Course/Baseline Name | Unified Eval Name | Processed Directory | Meaning |
| --- | --- | --- | --- |
| `ceilnet_table2` / CEILNet Table 2 | `ceilnet` | `datasets/processed_data/testdata_CEILNET_table2` | CEILNet synthetic Table 2 test set, 100 pairs. |
| `real20` | `zhang20` | `datasets/processed_data/real20` | Zhang/Berkeley real reflection test set, 20 pairs. |
| `objects` | `sir2_objects` | `datasets/processed_data/objects` | SIR2 Objects subset, 200 pairs. |
| `postcard` | `sir2_postcard` | `datasets/processed_data/postcard` | SIR2 Postcard subset, 179 pairs in the provided package. |
| `wild` | `sir2_wild` | `datasets/processed_data/wild` | SIR2 Wild subset, 101 pairs in the provided package. |

Baseline reference from the course checkpoint:

```text
Checkpoint: checkpoints/errnet/errnet_060_00463920.pt
Evaluator: test_errnet.py --hyper
```

| Dataset | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| CEILNet Table 2 | 27.88 | 0.9407 | 0.9808 | 0.0048 |
| real20 | 23.55 | 0.8285 | 0.8877 | 0.0201 |
| objects | 24.85 | 0.8980 | 0.9817 | 0.0029 |
| postcard | 22.07 | 0.8773 | 0.9463 | 0.0044 |
| wild | 25.18 | 0.8860 | 0.9359 | 0.0083 |

For reporting, use one naming style consistently. Recommended table labels:
`CEILNet Table 2`, `Zhang real20`, `SIR2 Objects`, `SIR2 Postcard`, and
`SIR2 Wild`.

## 12. From-Scratch RAP Epoch63 Evaluation

Snapshot:

```text
Checkpoint: checkpoints/rap_errnet_main/best_epoch63_snapshot.pt
Training stage: about epoch 63, from-scratch 3-channel RAP
Save dir: results/rap_errnet_main_epoch63
```

Metrics:

| Dataset | RAP PSNR | RAP SSIM | RAP NCC | RAP LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Average | 22.9937 | 0.8799 | 0.9463 | 0.0058 | - | - | - | - | Slightly improved over the epoch50 snapshot. |
| CEILNet Table 2 | 19.0772 | 0.8324 | 0.8796 | 0.0131 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | Still much weaker than baseline. |
| Zhang real20 | 19.6146 | 0.7342 | 0.8193 | 0.0210 | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Improved over epoch50, but still weak. |
| SIR2 Objects | 25.9151 | 0.9091 | 0.9858 | 0.0027 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Better than baseline on all metrics. |
| SIR2 Postcard | 20.9276 | 0.8735 | 0.9493 | 0.0044 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Mixed; NCC is better, PSNR/SSIM lower. |
| SIR2 Wild | 25.4174 | 0.9093 | 0.9540 | 0.0045 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Better than baseline on PSNR/SSIM/NCC/LMSE. |

Epoch63 compared with epoch50:

| Dataset | Epoch50 PSNR | Epoch63 PSNR | Delta |
| --- | ---: | ---: | ---: |
| Average | 22.8558 | 22.9937 | +0.1379 |
| CEILNet Table 2 | 19.0065 | 19.0772 | +0.0707 |
| Zhang real20 | 19.4201 | 19.6146 | +0.1945 |
| SIR2 Objects | 25.7091 | 25.9151 | +0.2060 |
| SIR2 Postcard | 20.9513 | 20.9276 | -0.0237 |
| SIR2 Wild | 25.0728 | 25.4174 | +0.3446 |

Interpretation:

- From-scratch RAP is still learning after epoch50, with small but consistent
  gains on average metrics.
- The improvements concentrate on SIR2 Objects and SIR2 Wild, supporting the
  claim that the reflection prior and refinement branch help real-scene
  structure recovery.
- CEILNet Table 2 and Zhang real20 remain much weaker than the pretrained
  ERRNet baseline, which confirms that the 3-channel from-scratch backbone is
  not the best main result candidate.
- The main report should therefore emphasize the hypercolumn pretrained RAP run
  as the primary method and keep this run as a from-scratch reference.

## 13. Hyper-Pretrained RAP Mid Evaluation

Snapshot:

```text
Checkpoint: checkpoints/rap_errnet_hyper_pretrained_ppu_bs32_mid.pt
Training source: Alibaba Cloud PPU run, rap_errnet_hyper_pretrained_ppu_bs32
Training stage: about epoch 37
Model config: configs/rap_errnet_hyper_pretrained.yaml
Evaluator: eval_all.py on A6000 GPU1
```

Evaluation note:

- `ceilnet`, `sir2_objects`, `sir2_postcard`, and `sir2_wild` were evaluated
  without `--save_images`.
- `zhang20` OOMed on the 48GB A6000 during full-resolution hypercolumn
  evaluation. This is an evaluation-memory issue caused by upsampling VGG19
  hypercolumn features to the original image size, not a checkpoint-loading
  issue. For final reporting, evaluate `zhang20` on the 98GB PPU or use a
  carefully documented resized/tiled diagnostic only as an auxiliary result.

Metrics:

| Dataset | RAP-Hyper PSNR | RAP-Hyper SSIM | RAP-Hyper NCC | RAP-Hyper LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEILNet Table 2 | 24.0434 | 0.9081 | 0.9592 | 0.0073 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | Much better than from-scratch RAP, still below pretrained baseline. |
| Zhang real20 | OOM on A6000 | OOM | OOM | OOM | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Needs PPU/full-memory evaluation. |
| SIR2 Objects | 25.9135 | 0.9103 | 0.9863 | 0.0025 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Better than baseline on all metrics. |
| SIR2 Postcard | 22.3617 | 0.8907 | 0.9534 | 0.0038 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Better than baseline on all metrics. |
| SIR2 Wild | 26.1276 | 0.9174 | 0.9586 | 0.0041 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Strong improvement over baseline. |

Four-dataset average excluding Zhang real20:

| Method | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| Baseline ERRNet `--hyper` | 24.9945 | 0.9005 | 0.9612 | 0.0051 |
| RAP-Hyper mid | 24.6116 | 0.9066 | 0.9644 | 0.0044 |

Interpretation:

- Hyper-pretrained RAP is already a stronger main-result candidate than the
  3-channel from-scratch RAP.
- Compared with from-scratch RAP epoch63, CEILNet improves sharply
  (`19.0772 -> 24.0434` PSNR), confirming that architecture-matched pretrained
  initialization fixes a major weakness of the first run.
- On SIR2 Objects/Postcard/Wild, the model beats the pretrained ERRNet baseline
  on SSIM, NCC, and LMSE, and also improves PSNR on all three subsets.
- The only unresolved benchmark is Zhang real20, which needs full-memory PPU
  evaluation before drawing final conclusions.

## 14. CEILNet Gap Analysis And Code Mitigation

Observed issue:

```text
CEILNet Table 2 baseline:       27.8765 PSNR
From-scratch RAP epoch63:       19.0772 PSNR
Hyper-pretrained RAP epoch37:   24.0434 PSNR
```

Analysis:

- CEILNet Table 2 is a synthetic benchmark where the original ERRNet `--hyper`
  baseline is already very strong.
- The first hyper-pretrained RAP run loads the ERRNet backbone, but the newly
  added gated adapter and refinement head were randomly initialized.
- Because these branches add residual corrections after the coarse ERRNet
  output, random initialization can perturb a strong pretrained output before
  the new branches learn meaningful corrections.
- This hurts CEILNet more than SIR2 because CEILNet rewards close pixel-level
  agreement with synthetic ground truth, while SIR2 gains more from structural
  and local reflection-aware corrections.

Code mitigation:

- `_GatedAdapter.out_proj` is now zero-initialized.
- `LightweightRefinement` final RGB prediction layer is now zero-initialized.

Expected behavior:

- A newly initialized `rap_errnet_hyper_pretrained` model should start from the
  pretrained ERRNet output, with zero residual correction.
- Training can still learn non-zero prior-aware corrections, but it no longer
  starts by randomly damaging CEILNet performance.

Recommended follow-up run:

```bash
python train_rap_errnet.py \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --name rap_errnet_hyper_pretrained_zerores_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 32 \
  --epochs 100 \
  --num_workers 8 \
  --device auto \
  --progress_bar
```

This should be treated as the improved hyper-pretrained candidate. The existing
`rap_errnet_hyper_pretrained_ppu_bs32` run remains useful as an ablation of
non-zero random residual initialization.

## 15. Qualitative Comparison Visualization

`eval_all.py` can save RAP and baseline outputs into the same visualization
image. This is useful for checking whether PSNR drops are caused by residual
reflection, color shift, oversmoothing, or localized artifacts.

Command pattern:

```bash
python eval_all.py \
  --model rap_errnet \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --ckpt checkpoints/rap_errnet_hyper_pretrained_ppu_bs32_mid.pt \
  --baseline_ckpt checkpoints/errnet/errnet_060_00463920.pt \
  --baseline_hyper \
  --data_root ./data \
  --save_dir results/rap_errnet_hyper_pretrained_ppu_bs32_mid_ceilnet_compare \
  --datasets ceilnet \
  --save_images \
  --device auto
```

The saved visualization order is:

```text
Input | Baseline | Output | GT | Error Map | Prior Map
```

For large full-resolution datasets, run this per dataset or on CEILNet first,
because loading both the baseline hypercolumn model and RAP-Hyper increases
evaluation memory usage.
