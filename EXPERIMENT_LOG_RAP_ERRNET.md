# RAP-ERRNet Experiment Log

Last updated: 2026-05-22

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
| BP-RAP v1.3 RIC path | Evaluated | RIC-only short fine-tune from RAP-Hyper is complete and recorded in Section 7.4. |
| BP-RAP v1.3 RIC+FSS path | Evaluated | Short fine-tune from RIC-only is complete and recorded in Section 7.5. |
| RAP extra-data fine-tune path | Ready | `configs/rap_errnet_hyper_zerores_extra_finetune.yaml` resumes model weights with reset epoch/optimizer for short OpenRR or `extra_train` adaptation. |
| Unified final evaluation | Done | Baseline and RAP variants were re-evaluated with `eval_all.py`; Zhang20 uses `--max_long_edge 512` to match the course real20 setting. |
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
| `5adc327` | Added BP-RAP v1.3 optional RIC and frequency-selective losses, configs, and design plan. |
| `096b7d7` | Fixed `scripts/sanity_zerores.py` import path for direct script execution. |

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

## 7.2 Final Hyper-Pretrained Evaluation Result

Snapshot:

```text
Checkpoint: checkpoints/rap_errnet_hyper_pretrained_ppu_bs32/best.pt
Training stage: epoch 100, hypercolumn pretrained RAP
Config: configs/rap_errnet_hyper_pretrained.yaml
Save dirs: results/rap_errnet_hyper_pretrained_ppu_bs32_final_best_*
Evaluator note: Zhang real20 was evaluated on CPU because full-resolution
hypercolumn inference OOMs on GPU.
```

Metrics:

| Dataset | RAP-Hyper PSNR | RAP-Hyper SSIM | RAP-Hyper NCC | RAP-Hyper LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Delta / Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEILNet Table2 | 23.8254 | 0.9070 | 0.9534 | 0.0072 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | +4.65 dB over from-scratch, but still -4.05 dB below baseline. |
| Zhang real20 | 20.0619 | 0.7412 | 0.8341 | 0.0209 | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Improved over from-scratch, but still below baseline. |
| SIR2 Objects | 25.8933 | 0.9115 | 0.9858 | 0.0026 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Improves all metrics over baseline. |
| SIR2 Postcard | 22.5328 | 0.8954 | 0.9517 | 0.0036 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Improves all metrics over baseline. |
| SIR2 Wild | 25.8443 | 0.9185 | 0.9578 | 0.0040 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Strong improvement on all metrics, especially SSIM/LMSE. |

Interpretation:

- Hypercolumn pretrained initialization clearly helps compared with from-scratch
  RAP, especially on CEILNet and SIR2.
- The model consistently improves the real-world SIR2 subsets, supporting the
  reflection-prior/refinement idea for real scenes.
- It still degrades CEILNet and Zhang real20 relative to the strong ERRNet
  baseline. The most likely cause is unrestricted fine-tuning drift: the new
  residual modules and backbone can change low-reflection regions even when the
  pretrained baseline is already correct.
- This result should be reported as `RAP-Hyper`. It is currently the strongest
  RAP variant on SIR2, but a staged zero-residual run is needed to test whether
  CEILNet/Zhang20 degradation can be reduced.

## 7.3 Final Hyper-ZeroRes-Staged Evaluation Result

Snapshot:

```text
Checkpoint: checkpoints/rap_errnet_hyper_zerores_staged_ppu_bs32/best.pt
Training stage: epoch 100, staged hypercolumn pretrained RAP
Config: configs/rap_errnet_hyper_zerores_staged.yaml
Save dirs: results/rap_errnet_hyper_zerores_staged_ppu_bs32_final_best_*
Evaluator note: Zhang real20 was evaluated on CPU because full-resolution
hypercolumn inference OOMs on GPU.
```

Metrics:

| Dataset | Staged PSNR | Staged SSIM | Staged NCC | Staged LMSE | RAP-Hyper PSNR | RAP-Hyper SSIM | RAP-Hyper NCC | RAP-Hyper LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEILNet Table2 | 23.8376 | 0.9116 | 0.9648 | 0.0070 | 23.8254 | 0.9070 | 0.9534 | 0.0072 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | Slightly more stable than RAP-Hyper, but still far below baseline. |
| Zhang real20 | 20.1955 | 0.7440 | 0.8275 | 0.0196 | 20.0619 | 0.7412 | 0.8341 | 0.0209 | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Small PSNR/SSIM/LMSE gain over RAP-Hyper; NCC drops. |
| SIR2 Objects | 25.4906 | 0.9076 | 0.9862 | 0.0025 | 25.8933 | 0.9115 | 0.9858 | 0.0026 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Still better than baseline, but weaker than RAP-Hyper on PSNR/SSIM. |
| SIR2 Postcard | 21.6596 | 0.8836 | 0.9482 | 0.0041 | 22.5328 | 0.8954 | 0.9517 | 0.0036 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Better than baseline on SSIM/NCC/LMSE, but lower PSNR and weaker than RAP-Hyper. |
| SIR2 Wild | 24.9851 | 0.9055 | 0.9424 | 0.0051 | 25.8443 | 0.9185 | 0.9578 | 0.0040 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Better structural metrics than baseline, but loses PSNR and is weaker than RAP-Hyper. |

Interpretation:

- Staged zero-residual training does reduce some drift: CEILNet SSIM/NCC/LMSE
  and Zhang PSNR/SSIM/LMSE improve slightly over RAP-Hyper.
- The improvement is too small to solve the CEILNet/Zhang20 gap.
- The stronger anchor and staged schedule also restrict useful corrections,
  reducing the large SIR2 gains seen in RAP-Hyper.
- For the final report, use RAP-Hyper as the main performance-oriented RAP
  result and report Hyper-ZeroRes-Staged as a conservative ablation. The staged
  run is useful evidence that the project explored baseline-preserving training,
  but it is not the best final checkpoint.

## 7.4 BP-RAP v1.3 RIC-Only Short Fine-Tune Result

Snapshot:

```text
Checkpoint: checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt
Training source: resumed model-only from checkpoints/rap_errnet_hyper_pretrained_ppu_bs32/best.pt
Training stage: 20 epochs, warmup_new_modules only, backbone frozen
Config: configs/bp_rap_hyper_zerores_staged_ric.yaml
Flags: --use_physics_synthesis --use_ric --batch_size 32 --epochs 20
Save dirs: results/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24_final_best_*
Evaluator note: Zhang real20 was evaluated on CPU because full-resolution
hypercolumn inference can OOM on GPU.
```

Training log reading:

```text
Final epoch total=0.045397 pix=0.026953 grad=0.012405
anchor=0.006250 delta=0.014587 freq=0.000000 ric=0.023927
```

RIC was active and stable. `delta` stayed small, so the short fine-tune did not
show evidence of uncontrolled residual drift.

Metrics:

| Dataset | BP-RAP RIC PSNR | BP-RAP RIC SSIM | BP-RAP RIC NCC | BP-RAP RIC LMSE | RAP-Hyper PSNR | RAP-Hyper SSIM | RAP-Hyper NCC | RAP-Hyper LMSE | Baseline PSNR | Baseline SSIM | Baseline NCC | Baseline LMSE | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEILNet Table2 | 23.9710 | 0.9076 | 0.9533 | 0.0072 | 23.8254 | 0.9070 | 0.9534 | 0.0072 | 27.8765 | 0.9407 | 0.9808 | 0.0048 | Small PSNR/SSIM gain over RAP-Hyper, but still far below baseline. |
| Zhang real20 | 20.1036 | 0.7433 | 0.8339 | 0.0208 | 20.0619 | 0.7412 | 0.8341 | 0.0209 | 23.5531 | 0.8285 | 0.8877 | 0.0201 | Tiny PSNR/SSIM/LMSE gain over RAP-Hyper; still below baseline. |
| SIR2 Objects | 25.8905 | 0.9121 | 0.9858 | 0.0026 | 25.8933 | 0.9115 | 0.9858 | 0.0026 | 24.8533 | 0.8980 | 0.9817 | 0.0029 | Essentially preserves RAP-Hyper while slightly improving SSIM/LMSE. |
| SIR2 Postcard | 22.4479 | 0.8950 | 0.9525 | 0.0036 | 22.5328 | 0.8954 | 0.9517 | 0.0036 | 22.0702 | 0.8773 | 0.9463 | 0.0044 | Slight PSNR/SSIM drop from RAP-Hyper, but NCC/LMSE improve. |
| SIR2 Wild | 26.1264 | 0.9192 | 0.9576 | 0.0041 | 25.8443 | 0.9185 | 0.9578 | 0.0040 | 25.1778 | 0.8861 | 0.9359 | 0.0083 | Best PSNR/SSIM among RAP variants so far; strong over baseline. |

Aggregate comparison:

| Method | Mean PSNR over 5 sets | SIR2-only mean PSNR | SIR2-only mean SSIM | SIR2-only mean NCC | SIR2-only mean LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline ERRNet `--hyper` | 24.7062 | 24.0338 | 0.8871 | 0.9546 | 0.0052 |
| RAP-Hyper | 23.6315 | 24.7568 | 0.9085 | 0.9651 | 0.0034 |
| Hyper-ZeroRes-Staged | 23.2337 | 24.0451 | 0.8989 | 0.9589 | 0.0039 |
| BP-RAP RIC-only | 23.7079 | 24.8216 | 0.9088 | 0.9653 | 0.0034 |

Interpretation:

- RIC-only is a small but useful improvement over RAP-Hyper on the overall
  five-set mean and the SIR2-only mean. It particularly helps SIR2 Wild and
  slightly improves CEILNet/Zhang PSNR.
- It does not solve the CEILNet and Zhang gap against the pretrained ERRNet
  baseline. The result should not be claimed as universally better than
  baseline.
- Compared with Hyper-ZeroRes-Staged, RIC-only is clearly stronger on SIR2 and
  overall mean PSNR, but staged still has better CEILNet NCC/LMSE and Zhang
  LMSE. This supports treating RIC as a performance-oriented short fine-tune
  rather than a strict do-no-harm solution.
- Recommended reporting position: use `BP-RAP RIC-only` as the current best
  RAP-family result for real-scene/SIR2 performance, and keep the original
  ERRNet baseline as the synthetic CEILNet reference.

## 7.5 BP-RAP v1.3 RIC+FSS Short Fine-Tune Result

Snapshot:

```text
Checkpoint: checkpoints/bp_rap_hyper_ric_freq_ft_from_ric_ppu_bs24/best.pt
Training source: resumed model-only from checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt
Training stage: 10 epochs, warmup_new_modules only, backbone frozen
Config: configs/bp_rap_hyper_zerores_staged_ric_freq.yaml
Flags: --use_physics_synthesis --use_ric --use_freq_loss --batch_size 24 --epochs 10
Save dirs: results/bp_rap_hyper_ric_freq_ft_from_ric_ppu_bs24_final_best_*
Evaluator note: Zhang real20 was evaluated on CPU because full-resolution
hypercolumn inference can OOM on GPU.
```

Training log reading:

```text
Final epoch total=0.045842 pix=0.026775 grad=0.012386
anchor=0.006385 delta=0.014661 freq=0.020965 ric=0.022247
```

The added frequency loss was active and stable. Training loss changed only
slightly, which is expected for a short regularization fine-tune from an already
stable RIC checkpoint.

Metrics:

| Dataset | RIC+FSS PSNR | RIC+FSS SSIM | RIC+FSS NCC | RIC+FSS LMSE | RIC-only PSNR | RIC-only SSIM | RIC-only NCC | RIC-only LMSE | Delta / Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEILNet Table2 | 23.9003 | 0.9073 | 0.9532 | 0.0072 | 23.9710 | 0.9076 | 0.9533 | 0.0072 | PSNR/SSIM/NCC slightly drop; LMSE is almost unchanged. |
| Zhang real20 | 20.0988 | 0.7428 | 0.8340 | 0.0207 | 20.1036 | 0.7433 | 0.8339 | 0.0208 | Essentially tied; tiny LMSE/NCC gain but no meaningful PSNR gain. |
| SIR2 Objects | 25.9041 | 0.9119 | 0.9858 | 0.0026 | 25.8905 | 0.9121 | 0.9858 | 0.0026 | Tiny PSNR/LMSE gain, SSIM nearly tied. |
| SIR2 Postcard | 22.4792 | 0.8955 | 0.9527 | 0.0035 | 22.4479 | 0.8950 | 0.9525 | 0.0036 | Small improvement across all metrics. |
| SIR2 Wild | 26.0929 | 0.9187 | 0.9576 | 0.0041 | 26.1264 | 0.9192 | 0.9576 | 0.0041 | Slightly worse than RIC-only. |

Aggregate comparison:

| Method | Mean PSNR over 5 sets | SIR2-only mean PSNR | SIR2-only mean SSIM | SIR2-only mean NCC | SIR2-only mean LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| BP-RAP RIC-only | 23.7079 | 24.8216 | 0.9088 | 0.9653 | 0.0034 |
| BP-RAP RIC+FSS | 23.6951 | 24.8254 | 0.9087 | 0.9654 | 0.0034 |

Interpretation:

- RIC+FSS does not improve the five-set mean over RIC-only. It slightly helps
  SIR2 Objects/Postcard, but loses CEILNet and SIR2 Wild.
- The frequency loss behaves as a mild regularizer rather than a clear
  performance booster at the current weight (`lambda_freq=0.03`).
- For the final method, keep `BP-RAP RIC-only` as the preferred v1.3 result.
  Report `RIC+FSS` as an ablation showing that frequency-selective supervision
  is stable but not consistently beneficial under the current schedule.

## 7.6 Unified Final Evaluation With Matched eval_all Protocol

Motivation:

- Earlier baseline numbers were produced by `test_errnet.py --hyper`, while RAP
  variants were mostly evaluated by `eval_all.py`.
- To avoid mixing metric/data-loader implementations in the final paper table,
  all methods were re-evaluated with the repaired `eval_all.py`.
- The primary ERRNet baseline is explicitly evaluated with `--errnet_hyper`,
  so it uses the same 1475-channel RGB+VGG hypercolumn input as the course
  checkpoint.

Protocol:

```text
Evaluator: eval_all.py
Metrics: metrics/reflection_metrics.py, data_range=1.0
CEILNet/SIR2 resize: none
Zhang20/real20 resize: --max_long_edge 512
Device: --device auto
Baseline model: --model errnet --errnet_hyper
RAP models: --model rap_errnet with matching config
```

This is the recommended protocol for the final report's main comparison table.
The original `test_errnet.py --hyper` baseline remains a course-baseline
reproduction check, not the mixed-protocol main table.

Metrics:

| Method | Dataset | PSNR | SSIM | NCC | LMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| ERRNet baseline | CEILNet Table2 | 27.6414 | 0.9407 | 0.9808 | 0.0047 |
| ERRNet baseline | Zhang real20, 512 | 23.4367 | 0.8285 | 0.8877 | 0.0203 |
| ERRNet baseline | SIR2 Objects | 24.6983 | 0.8980 | 0.9817 | 0.0029 |
| ERRNet baseline | SIR2 Postcard | 21.8856 | 0.8773 | 0.9463 | 0.0044 |
| ERRNet baseline | SIR2 Wild | 24.8763 | 0.8861 | 0.9359 | 0.0083 |
| RAP-Hyper | CEILNet Table2 | 23.9566 | 0.9076 | 0.9536 | 0.0071 |
| RAP-Hyper | Zhang real20, 512 | 21.0440 | 0.7840 | 0.8604 | 0.0258 |
| RAP-Hyper | SIR2 Objects | 25.8587 | 0.9118 | 0.9857 | 0.0026 |
| RAP-Hyper | SIR2 Postcard | 22.4229 | 0.8950 | 0.9514 | 0.0036 |
| RAP-Hyper | SIR2 Wild | 25.9825 | 0.9192 | 0.9579 | 0.0040 |
| RAP-Staged | CEILNet Table2 | 23.8376 | 0.9116 | 0.9648 | 0.0070 |
| RAP-Staged | Zhang real20, 512 | 21.9808 | 0.7971 | 0.8743 | 0.0229 |
| RAP-Staged | SIR2 Objects | 25.4906 | 0.9076 | 0.9862 | 0.0025 |
| RAP-Staged | SIR2 Postcard | 21.6596 | 0.8836 | 0.9482 | 0.0041 |
| RAP-Staged | SIR2 Wild | 24.9851 | 0.9055 | 0.9424 | 0.0051 |
| BP-RAP RIC | CEILNet Table2 | 23.9710 | 0.9076 | 0.9533 | 0.0072 |
| BP-RAP RIC | Zhang real20, 512 | 21.0237 | 0.7854 | 0.8603 | 0.0256 |
| BP-RAP RIC | SIR2 Objects | 25.8905 | 0.9121 | 0.9858 | 0.0026 |
| BP-RAP RIC | SIR2 Postcard | 22.4479 | 0.8950 | 0.9525 | 0.0036 |
| BP-RAP RIC | SIR2 Wild | 26.1264 | 0.9192 | 0.9576 | 0.0041 |
| BP-RAP RIC+FSS | CEILNet Table2 | 23.9003 | 0.9073 | 0.9532 | 0.0072 |
| BP-RAP RIC+FSS | Zhang real20, 512 | 21.0333 | 0.7857 | 0.8606 | 0.0255 |
| BP-RAP RIC+FSS | SIR2 Objects | 25.9041 | 0.9119 | 0.9858 | 0.0026 |
| BP-RAP RIC+FSS | SIR2 Postcard | 22.4792 | 0.8955 | 0.9527 | 0.0035 |
| BP-RAP RIC+FSS | SIR2 Wild | 26.0929 | 0.9187 | 0.9576 | 0.0041 |

Aggregate:

| Method | 5-set mean PSNR | 5-set mean SSIM | 5-set mean NCC | 5-set mean LMSE | SIR2 mean PSNR | SIR2 mean SSIM | SIR2 mean NCC | SIR2 mean LMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERRNet baseline | 24.5077 | 0.8861 | 0.9465 | 0.0081 | 23.8201 | 0.8871 | 0.9546 | 0.0052 |
| RAP-Hyper | 23.8530 | 0.8835 | 0.9418 | 0.0086 | 24.7547 | 0.9087 | 0.9650 | 0.0034 |
| RAP-Staged | 23.5908 | 0.8811 | 0.9432 | 0.0083 | 24.0451 | 0.8989 | 0.9589 | 0.0039 |
| BP-RAP RIC | 23.8919 | 0.8839 | 0.9419 | 0.0086 | 24.8216 | 0.9088 | 0.9653 | 0.0034 |
| BP-RAP RIC+FSS | 23.8820 | 0.8838 | 0.9420 | 0.0086 | 24.8254 | 0.9087 | 0.9654 | 0.0034 |

Interpretation:

- The unified evaluation confirms the central trade-off. ERRNet remains much
  stronger on CEILNet and Zhang real20, so the final paper should not claim
  overall dominance over the course baseline.
- RAP/BP-RAP variants consistently improve all SIR2 subsets over the ERRNet
  baseline. For BP-RAP RIC, the SIR2 PSNR gains are +1.1922 dB on Objects,
  +0.5623 dB on Postcard, and +1.2501 dB on Wild.
- `BP-RAP RIC` is the preferred main method: it has the best 5-set mean PSNR
  among RAP-family models and nearly the best SIR2 aggregate metrics.
- `RIC+FSS` very slightly improves SIR2 mean PSNR/NCC and Postcard, but lowers
  CEILNet and Wild relative to RIC-only. Keep it as an ablation rather than the
  main method.
- `RAP-Staged` is useful as a conservative baseline-preserving ablation. It
  improves Zhang relative to RAP-Hyper/RIC, but sacrifices the SIR2 gains.

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
