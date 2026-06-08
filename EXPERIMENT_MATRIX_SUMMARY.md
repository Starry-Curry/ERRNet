# ERRNet / BP-RAP / RAFA Experiment Matrix Summary

This document is a compact experiment index for deciding what should enter the
final course paper. The full chronological record is in
`EXPERIMENT_LOG_RAP_ERRNET.md`; this file reorganizes the same work by method,
training data, generation or adaptation strategy, evaluation protocol, and
result status.

## 0. Existing Records

| File | Role | How to use |
| --- | --- | --- |
| `EXPERIMENT_LOG_RAP_ERRNET.md` | Full chronological log | Source of truth for commands, checkpoints, metric tables, and decisions. |
| `RAP_ERRNET_EXPERIMENT_REPORT.md` | Stage report | Shorter narrative summary before the latest RAFA/balanced/visualization updates. |
| `BP-RAP-ERRNet_v1.3_design_and_execution_plan.md` | v1.3 design | BP-RAP, RIC, FSS, theory-style justification, planned ablations. |
| `design_v1.4.md` | v1.4 planning draft | RAFA, reflection-strength balanced sampling, module soup, next-stage strategy. |
| `BP_RAP_V1_4_RAFA_ITERATION_PLAN.md` | v1.4 execution plan | Server-side command plan for OpenRR/RAFA/soup/visualization experiments. |
| `paper/neurips_2026.tex` | Current paper draft | Paper-style narrative, not a raw experiment index. |
| `results/NEXT_STAGE_SWEEP_SUMMARY.md` | Sweep summary | Post-hoc residual/gate/low-frequency sweep and early OpenRR/soup 512 diagnostics. |

Conclusion: a complete log already exists, but there was no single matrix that
lists all comparable experiments. This file fills that role.

## 1. Dataset Inventory

### 1.1 Training Data

| Training source | Count / scope | Used by | Notes |
| --- | ---: | --- | --- |
| Pascal VOC cropped clean images | 7,643 crops, 224 x 224 | ERRNet course training reference, RAP/BP-RAP course-fair training, RAFA course replay | Used with physics-based synthetic reflection generation. |
| Zhang real train | 89 paired real images | RAP/BP-RAP course-fair training, OpenRR-FT, RAFA replay | No provided masks in our pipeline; pseudo mask uses `abs(input-target)`. |
| OpenRR train 1k | 1,000 paired real images | OpenRR-FT, Soup alpha, RAFA1k | Extra-data setting only; not course-fair. |
| OpenRR train 3k | 3,000 paired real images | RAFA3k, Balanced RAFA3k, Old04 | Extra-data setting only; selected for final external-data adaptation. |
| Self-collected paired images | Pending, required 5+ | Final course requirement | Not completed in current metrics; must be added before submission. |

### 1.2 Test Data

| Test set | Count | Protocol name in code/log | Formal evaluation resolution |
| --- | ---: | --- | --- |
| CEILNet Table2 synthetic | 100 | `ceilnet` | Native resolution in formal tables; sometimes 512 for diagnostic sweeps. |
| Zhang real20 | 20 | `zhang20` | `--max_long_edge 512` in final formal protocol. |
| SIR2 Objects | 200 | `sir2_objects` | Native resolution in formal tables. |
| SIR2 Postcard | 179 | `sir2_postcard` | Native resolution in formal tables. |
| SIR2 Wild | 101 | `sir2_wild` | Native resolution in formal tables. |
| OpenRR val | 300 | `openrr_val` | Native resolution in formal extra-data tables; 512 in quick diagnostics. |
| Self-collected test set | Pending | `self` / custom layout | Must be evaluated after collection. |

### 1.3 Evaluation Protocols

| Protocol | Datasets | Resize rule | Use in paper |
| --- | --- | --- | --- |
| Course 5-set formal | CEILNet, Zhang20, SIR2 Objects/Postcard/Wild | Zhang20 at 512; others native | Main baseline vs method table. |
| External OpenRR zero-shot | OpenRR val | Native | Important external generalization result. |
| Six-set formal extra-data | Course 5 sets + OpenRR val | Zhang20 at 512; CEILNet/SIR2/OpenRR native | Main RAFA/OpenRR adaptation table. |
| Fast 512 diagnostic | Course 5 sets + OpenRR val | All datasets `--max_long_edge 512` | Screening only; do not mix with formal numbers in main table. |

Metrics are PSNR, SSIM, NCC, and LMSE. Higher is better for PSNR/SSIM/NCC;
lower is better for LMSE.

## 2. Reflection Generation / Adaptation Strategies

### 2.1 Physics Synthesis Used During Training

All RAP/BP-RAP/RAFA runs using course synthesis share the same synthetic
reflection parameter family:

| Parameter | Value range |
| --- | --- |
| Reflection intensity `beta` | 0.15 to 0.75 |
| Blur/nonlinearity `kappa` | 0.0 to 0.25 |
| First blur sigma | 1.0 to 5.0 |
| Second blur sigma | 5.0 to 15.0 |
| Second component alpha | 0.0 to 0.35 |
| Spatial shift | -8 to 8 pixels |
| Noise std | 0.005 |

This is the core generated training data strategy for VOC clean images. Zhang
real and OpenRR real pairs do not use synthetic reflection generation.

### 2.2 Inference / Checkpoint Combination Strategies

| Strategy | Training required | Purpose | Result status |
| --- | --- | --- | --- |
| Raw model forward | No extra inference trick | Main inference for ERRNet, BP-RAP, RAFA | Used for all formal results. |
| Post-hoc residual scaling | No | Scale BP-RAP residual by 0.25/0.5/0.75/1.0 | Diagnostic; full residual was best. |
| Prior-gated residual | No | Gate residual with prior, gamma 1.0/1.5/2.0 | Diagnostic; worse than full residual. |
| Low-frequency anchoring | No | Anchor low-frequency component after inference | Diagnostic; worse than full residual. |
| Checkpoint soup | No new training | Average base BP-RAP RIC and OpenRR-FT weights | Useful fallback; alpha=0.25 formally evaluated. |
| Module-only soup | No new full training | Soup only selected modules around RAFA | Diagnostic; did not beat raw RAFA. |
| RAFA replay adaptation | Yes | Real-data fine-tuning with course replay and teacher distillation | Best extra-data algorithm. |
| Reflection-strength balanced replay | Yes | Balance OpenRR replay across weak/strong and veil/ghost bins | Metric-best final checkpoint by six-set mean, but gain is marginal. |

### 2.3 Qualitative Visualization Generation

| Visualization strategy | Output | Purpose |
| --- | --- | --- |
| Even/random dataset contact sheets | `results/final_visuals_balanced` | Early broad visual scan. |
| Per-dataset better/similar/worse groups | `results/final_visuals_by_dataset` | Diagnose where RAFA beats/ties/loses to ERRNet per dataset. |
| Global top-better examples | `results/final_visuals_top_better/contact_sheets/top_better.png` | Paper main qualitative figure: 5-6 examples where final model improves over ERRNet. |
| Result summary plot | `paper/figures/result_summary.png` | Paper quantitative trend figure. |

## 3. Method Matrix

| ID | Method / run | Course-fair? | Training data | Config / key flags | Checkpoint / result location | Main result status |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | ERRNet-Hyper baseline | Yes | Course checkpoint, no new training in this project | `eval_all.py --model errnet --errnet_hyper` | `checkpoints/errnet/errnet_060_00463920.pt` | Strong baseline; best on CEILNet/Zhang. |
| M1 | RAP from scratch | Yes | VOC physics synthesis + Zhang real89 | `configs/rap_errnet.yaml`; 3-channel backbone; batch 8; 100 epochs | `checkpoints/rap_errnet_main/best.pt` | Diagnostic only; SIR2 improves, CEILNet/Zhang much worse. |
| M2 | RAP-Hyper | Yes | VOC physics synthesis + Zhang real89 | `configs/rap_errnet_hyper_pretrained.yaml`; ERRNet hypercolumn init; residual/prior/gate/refine | `checkpoints/rap_errnet_hyper_pretrained_ppu_bs32/best.pt` | Strong SIR2 result; still below baseline on CEILNet/Zhang. |
| M3 | RAP-Staged / ZeroRes | Yes | VOC physics synthesis + Zhang real89 | `configs/rap_errnet_hyper_zerores_staged.yaml`; 20-epoch frozen warmup + 80-epoch conservative finetune; anchor/delta regularization | `checkpoints/rap_errnet_hyper_zerores_staged_ppu_bs32/best.pt` | Conservative ablation; improves stability but reduces SIR2 gains. |
| M4 | BP-RAP RIC | Yes | VOC physics synthesis + Zhang real89 | `configs/bp_rap_hyper_zerores_staged_ric.yaml`; `lambda_ric=0.03`; RIC prob 0.5; residual scale 0.1 | `checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt` | Course-fair main method. |
| M5 | BP-RAP RIC + FSS | Yes | VOC physics synthesis + Zhang real89 | `configs/bp_rap_hyper_zerores_staged_ric_freq.yaml`; `lambda_freq=0.03` | `checkpoints/bp_rap_hyper_ric_freq_ft_from_ric_ppu_bs24/best.pt` | Stable but not consistently better than RIC; ablation only. |
| M6 | BP-RAP post-hoc calibration sweep | Yes | No training; uses M4 checkpoint | Residual scale 0.25/0.5/0.75/1.0; prior gate gamma 1.0/1.5/2.0; low-frequency anchors 0.25/0.5/0.75 | `results/NEXT_STAGE_SWEEP_SUMMARY.md` | Diagnostic only; raw residual/full scale wins. |
| M7 | OpenRR zero-shot comparison | Yes for model training; external test only | No OpenRR train data | Compare M0 and M4 directly on OpenRR val | `results/openrr_zero_errnet`, `results/openrr_zero_bp_rap_ric` | Strong external validation: M4 beats ERRNet by +1.37 dB. |
| M8 | BP-RAP RIC + OpenRR-FT 1k | No | Zhang real89 + OpenRR train 1k | M4 init; `--use_openrr --max_openrr_pairs 1000`; 10 epochs; frozen backbone; no physics synthesis, so RIC skipped | `checkpoints/bp_rap_ric_openrr1k_realonly_freeze_e10/best.pt` | 512 diagnostic only; high OpenRR gain but source-domain drift. |
| M9 | OpenRR checkpoint soup | No | No new training; mixes M4 and M8 | `tools/soup_checkpoints.py`; alpha=0.25/0.50/0.75; selected alpha=0.25 formal | `checkpoints/bp_rap_ric_openrr1k_soup_a0p25/best.pt` | Good fallback; superseded by RAFA. |
| M10 | RAFA-OpenRR1k | No | OpenRR train 1k + VOC physics synthesis + Zhang real89 | `configs/bp_rap_rafa_openrr1k.yaml`; OpenRR/course 0.5/0.5; samples/epoch 1200; teacher=M4; `lambda_old=0.2`; frozen 10 epochs | `checkpoints/bp_rap_rafa_openrr1k_e10/best.pt` | First principled extra-data adaptation; better than soup. |
| M11 | Module-only soup after RAFA1k | No | No new training | Soup selected modules around RAFA; alpha=0.25/0.50/0.75 | Diagnostic table in log | Not selected; raw RAFA remains better. |
| M12 | RAFA-OpenRR3k | No | OpenRR train 3k + VOC physics synthesis + Zhang real89 | RAFA config with CLI overrides: max OpenRR pairs 3000, samples/epoch 1600; teacher=M4; `lambda_old=0.2`; frozen 10 epochs | `checkpoints/bp_rap_rafa_openrr3k_e10/best.pt` | Main extra-data adaptation algorithm. |
| M13 | Strength-balanced RAFA-OpenRR3k | No | OpenRR train 3k + VOC physics synthesis + Zhang real89 | `configs/bp_rap_rafa_openrr3k_balanced.yaml`; OpenRR/course 0.5/0.5; bin-balanced OpenRR replay; `lambda_old=0.2` | `checkpoints/bp_rap_rafa_openrr3k_balanced_e10/best.pt` | Final metric-best checkpoint by six-set mean. |
| M14 | RAFA-OpenRR3k Old04 | No | OpenRR train 3k + VOC physics synthesis + Zhang real89 | `configs/bp_rap_rafa_openrr3k_old04.yaml`; same as RAFA3k but `lambda_old=0.4` | `checkpoints/bp_rap_rafa_openrr3k_old04_e10/best.pt` | Negative diagnostic; stronger old distillation hurts. |

## 4. Formal Course 5-Set Results

This is the cleanest course-fair table. It excludes OpenRR train and should be
the main baseline comparison in the paper.

| Method | 5-set PSNR | 5-set SSIM | 5-set NCC | 5-set LMSE | SIR2 PSNR | SIR2 SSIM | SIR2 NCC | SIR2 LMSE | Paper role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ERRNet baseline | 24.5077 | 0.8861 | 0.9465 | 0.0081 | 23.8201 | 0.8871 | 0.9546 | 0.0052 | Required baseline. |
| RAP-Hyper | 23.8530 | 0.8835 | 0.9418 | 0.0086 | 24.7547 | 0.9087 | 0.9650 | 0.0034 | Method development result. |
| RAP-Staged | 23.5908 | 0.8811 | 0.9432 | 0.0083 | 24.0451 | 0.8989 | 0.9589 | 0.0039 | Conservative ablation. |
| BP-RAP RIC | 23.8919 | 0.8839 | 0.9419 | 0.0086 | 24.8216 | 0.9088 | 0.9653 | 0.0034 | Course-fair main method. |
| BP-RAP RIC+FSS | 23.8820 | 0.8838 | 0.9420 | 0.0086 | 24.8254 | 0.9087 | 0.9654 | 0.0034 | Stable ablation, not main. |

Key reading:

- ERRNet remains stronger on CEILNet and Zhang real20.
- BP-RAP RIC improves SIR2 PSNR over ERRNet by:
  - Objects: +1.1922 dB
  - Postcard: +0.5623 dB
  - Wild: +1.2501 dB
- The honest claim is not "universally better than ERRNet"; it is "better on
  real-scene SIR2/OpenRR style cases, weaker on some aligned benchmarks."

## 5. Per-Dataset Formal Course Results

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

## 6. Early Diagnostic Results

### 6.1 From-Scratch RAP

| Run | Config | Training | Result | Decision |
| --- | --- | --- | --- | --- |
| RAP mid epoch ~50 | `configs/rap_errnet.yaml` | VOC physics + Zhang real89 | CEILNet/Zhang much worse than ERRNet; SIR2 Objects and Wild improved. | Diagnostic only. |
| RAP final epoch 100 | `configs/rap_errnet.yaml` | VOC physics + Zhang real89 | Average PSNR 22.9575; SIR2 Objects/Wild improve, CEILNet remains about 8.7 dB below baseline. | Do not use as main method. |

This experiment is useful in the paper only as a short ablation showing why
pretrained ERRNet anchoring is necessary.

### 6.2 Hyper-Pretrained Mid Evaluation

There is also a RAP-Hyper mid checkpoint:

| Method | PSNR | SSIM | NCC | LMSE | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| RAP-Hyper mid | 24.6116 | 0.9066 | 0.9644 | 0.0044 | Auxiliary/diagnostic; not part of final formal table. |

Use only if the paper needs a training-progress figure; otherwise omit.

## 7. OpenRR External / Extra-Data Results

### 7.1 OpenRR Zero-Shot

No OpenRR train data is used here. This is an external generalization test.

| Method | OpenRR val PSNR | SSIM | NCC | LMSE | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| ERRNet baseline | 25.4874 | 0.9480 | 0.9641 | 0.0030 | Baseline external reference. |
| BP-RAP RIC | 26.8597 | 0.9600 | 0.9693 | 0.0018 | +1.3723 dB over ERRNet; important paper result. |

### 7.2 Fast 512 Post-Hoc Sweep

All six datasets are resized to max long edge 512. This is a screening
protocol, not the final formal table.

| Method | Count | PSNR | SSIM | NCC | LMSE | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| residual_0.25 | 6 | 23.6692 | 0.8675 | 0.9416 | 0.0094 | Too weak. |
| residual_0.5 | 6 | 23.8718 | 0.8812 | 0.9431 | 0.0088 | Better but below full residual. |
| residual_0.75 | 6 | 24.0310 | 0.8903 | 0.9441 | 0.0083 | Close. |
| residual_1.0 | 6 | 24.1389 | 0.8944 | 0.9447 | 0.0080 | Best; keep raw BP-RAP. |
| gate_1.0 | 6 | 23.9072 | 0.8834 | 0.9433 | 0.0087 | Worse. |
| gate_1.5 | 6 | 23.8031 | 0.8768 | 0.9426 | 0.0090 | Worse. |
| gate_2.0 | 6 | 23.7179 | 0.8710 | 0.9420 | 0.0093 | Worse. |
| lowfreq_0.25 | 6 | 23.7485 | 0.8821 | 0.9427 | 0.0087 | Worse. |
| lowfreq_0.5 | 6 | 23.8054 | 0.8826 | 0.9429 | 0.0087 | Worse. |
| lowfreq_0.75 | 6 | 23.8579 | 0.8830 | 0.9431 | 0.0087 | Worse. |

Paper use: one sentence in ablation/negative results is enough. Do not make this
a main table.

### 7.3 OpenRR-FT and Soup

| Method | Training / construction | Evaluation type | Six-set PSNR | SSIM | NCC | LMSE | OpenRR val PSNR | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenRR-FT 1k | M4 init; Zhang real89 + OpenRR train 1k; frozen 10 epochs; RIC skipped | 512 diagnostic | 24.4858 | 0.8865 | 0.9444 | 0.0082 | 29.4965 | Strong target gain but drift; not final. |
| Soup alpha=0.25 | `0.75 * BP-RAP RIC + 0.25 * OpenRR-FT` | 512 diagnostic | 24.3010 | 0.8950 | 0.9449 | 0.0079 | 27.8549 | Stable fallback. |
| Soup alpha=0.50 | Average weighted toward OpenRR-FT | 512 diagnostic | 24.4533 | 0.8946 | 0.9450 | 0.0079 | 28.5462 | Higher mean but more OpenRR-biased. |
| Soup alpha=0.75 | More OpenRR-FT weight | 512 diagnostic | 24.5303 | 0.8919 | 0.9449 | 0.0080 | 29.1822 | Close to OpenRR-FT behavior. |

Soup alpha=0.25 was also formally evaluated:

| Dataset | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| CEILNet Table2 | 23.9450 | 0.9064 | 0.9531 | 0.0071 |
| Zhang real20, 512 | 21.0559 | 0.7857 | 0.8601 | 0.0256 |
| SIR2 Objects | 26.0059 | 0.9129 | 0.9861 | 0.0026 |
| SIR2 Postcard | 22.6883 | 0.8961 | 0.9531 | 0.0036 |
| SIR2 Wild | 26.0795 | 0.9199 | 0.9578 | 0.0040 |
| OpenRR val | 27.4103 | 0.9614 | 0.9704 | 0.0017 |
| Six-set mean | 24.5308 | 0.8971 | 0.9468 | 0.0074 |

Paper use: optional ablation/fallback. RAFA is a cleaner algorithmic story.

## 8. RAFA and Balanced RAFA Results

### 8.1 RAFA Formal Results

| Method | Training data | Config / key difference | Six-set PSNR | SSIM | NCC | LMSE | OpenRR val PSNR | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RAFA1k | OpenRR train 1k + VOC physics + Zhang real89 | `configs/bp_rap_rafa_openrr1k.yaml`; OpenRR/course 0.5/0.5; `lambda_old=0.2` | 24.5779 | 0.8967 | 0.9468 | 0.0074 | 27.7031 | Better than soup; useful progression. |
| RAFA3k | OpenRR train 3k + VOC physics + Zhang real89 | RAFA1k config with 3k/samples=1600 overrides | 24.6336 | 0.8968 | 0.9468 | 0.0074 | 27.8483 | Main extra-data algorithm. |
| Balanced RAFA3k | Same as RAFA3k | `configs/bp_rap_rafa_openrr3k_balanced.yaml`; OpenRR bin-balanced replay | 24.6390 | 0.8968 | 0.9467 | 0.0074 | 27.8349 | Metric-best final checkpoint by mean. |

### 8.2 Per-Dataset RAFA3k vs Balanced RAFA3k

| Dataset | RAFA3k PSNR | RAFA3k SSIM | RAFA3k NCC | RAFA3k LMSE | Balanced PSNR | Balanced SSIM | Balanced NCC | Balanced LMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CEILNet Table2 | 24.0504 | 0.9076 | 0.9534 | 0.0072 | 24.0417 | 0.9073 | 0.9533 | 0.0072 |
| Zhang real20, 512 | 21.0620 | 0.7821 | 0.8594 | 0.0256 | 21.0535 | 0.7827 | 0.8591 | 0.0256 |
| SIR2 Objects | 26.0450 | 0.9141 | 0.9863 | 0.0025 | 26.0617 | 0.9141 | 0.9863 | 0.0025 |
| SIR2 Postcard | 22.6764 | 0.8948 | 0.9532 | 0.0036 | 22.6947 | 0.8945 | 0.9531 | 0.0036 |
| SIR2 Wild | 26.1194 | 0.9203 | 0.9578 | 0.0040 | 26.1477 | 0.9205 | 0.9578 | 0.0040 |
| OpenRR val | 27.8483 | 0.9618 | 0.9708 | 0.0016 | 27.8349 | 0.9618 | 0.9707 | 0.0016 |
| Six-set mean | 24.6336 | 0.8968 | 0.9468 | 0.0074 | 24.6390 | 0.8968 | 0.9467 | 0.0074 |

Balanced RAFA improves the formal mean by only +0.0054 dB over uniform RAFA3k.
It is fair to select it as metric-best, but the paper should frame it as a
sampling refinement, not the main conceptual contribution.

### 8.3 Reflection-Strength Bins

The OpenRR3k strength analysis produced:

| Bin | Count |
| --- | ---: |
| strong_ghost | 377 |
| strong_veil | 1123 |
| weak_ghost | 1123 |
| weak_veil | 377 |

Earlier OpenRR1k analysis produced the same qualitative imbalance:

| Dataset | strong_ghost | strong_veil | weak_ghost | weak_veil |
| --- | ---: | ---: | ---: | ---: |
| OpenRR train 1k | 148 | 352 | 352 | 148 |
| Zhang train89 | 13 | 32 | 32 | 12 |

Paper use: include only if explaining balanced sampling; otherwise too detailed
for the main text.

### 8.4 Negative RAFA Old04

| Method | Difference | Evaluation type | Six-set PSNR | OpenRR val PSNR | Decision |
| --- | --- | --- | ---: | ---: | --- |
| RAFA3k Old04 | `lambda_old=0.4` instead of 0.2 | 512 diagnostic only | 24.3367 | 28.0531 | Do not formally evaluate; stronger old distillation hurts target adaptation. |

Paper use: one-line negative ablation if needed.

## 9. What Has Result Comparisons But Should Not Be Main

| Item | Why it is not main |
| --- | --- |
| From-scratch RAP | Too weak on CEILNet/Zhang; useful only to justify pretrained baseline anchoring. |
| RAP-Staged | Conservative but reduces SIR2 gains; not the best trade-off. |
| BP-RAP RIC+FSS | Almost tied with RIC, not enough gain to justify as final method. |
| Post-hoc residual/gate/lowfreq sweep | No training innovation; full residual wins. |
| OpenRR-FT 1k real-only | Strong OpenRR 512 diagnostic but source-domain drift and mixed protocol. |
| Soup alpha 0.25 | Useful fallback but less algorithmically clean than RAFA. |
| Module-only soup | Did not beat raw RAFA. |
| Old04 | Negative diagnostic. |

## 10. Recommended Paper Selection

### 10.1 Must Include

| Paper section | Recommended content |
| --- | --- |
| Baseline | ERRNet-Hyper, course 5-set formal metrics. |
| Main method | BP-RAP RIC, course 5-set formal metrics, emphasizing SIR2 gains and CEILNet/Zhang weakness. |
| Extra-data method | RAFA3k and/or Balanced RAFA3k, six-set formal metrics with OpenRR val. |
| External validation | OpenRR zero-shot: BP-RAP RIC vs ERRNet, +1.3723 dB. |
| Qualitative | Global top-better figure with 5-6 examples; optionally mention failure cases in appendix. |
| Self-collected data | Pending; required by course. |

### 10.2 Good Ablations To Include If Space Allows

| Ablation | Why include |
| --- | --- |
| RAP-Hyper vs BP-RAP RIC | Shows RIC improves the selected course-fair RAP family. |
| RIC vs RIC+FSS | Shows FSS was explored but not selected. |
| RAFA1k vs RAFA3k vs Balanced RAFA3k | Shows extra-data scaling and sampling refinement. |
| Soup vs RAFA | Shows why RAFA is a better final extra-data story than checkpoint averaging. |

### 10.3 Appendix / Not Necessary In Main Paper

| Item | Placement |
| --- | --- |
| Full post-hoc calibration sweep | Appendix or omit. |
| Old04 negative diagnostic | Appendix or one sentence. |
| Early from-scratch epoch50/epoch63 | Appendix or omit unless discussing iteration history. |
| Per-dataset better/similar/worse visual sheets | Appendix; main text should use top-better figure. |

## 11. Current Gaps

| Gap | Impact | Next action |
| --- | --- | --- |
| Self-collected 5 images not evaluated | Course requirement; must be fixed before final submission. | Collect paired images, run ERRNet/BP-RAP/RAFA metrics and visualization. |
| Structural ablations w/o prior/gate/refinement are not completed | Would strengthen algorithm-completion score, but not strictly necessary if time is limited. | Run only if self-collected data and report figures are already done. |
| Formal SOTA table is not fully reproduced | Could improve literature comparison, but course baseline comparison is more important. | Use cited paper numbers carefully, or avoid overclaiming SOTA. |

## 12. Bottom-Line Decision

For the paper, the cleanest story is:

1. Reproduce ERRNet-Hyper as the strong baseline.
2. Present BP-RAP RIC as the course-fair improved method.
3. State honestly that BP-RAP RIC loses on CEILNet/Zhang but improves all SIR2
   subsets and OpenRR zero-shot.
4. Present RAFA3k as the extra-data adaptation algorithm.
5. Use Balanced RAFA3k as the final metric-best checkpoint, but describe its
   gain as marginal.
6. Put self-collected data and qualitative examples in the final report before
   adding any more training sweeps.
