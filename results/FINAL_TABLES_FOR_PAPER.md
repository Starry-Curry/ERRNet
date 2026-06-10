# Final Tables For Paper

All values use the formal protocol: CEILNet/SIR2/OpenRR at native resolution and Zhang20 with `--max_long_edge 512`. OpenRR-train methods are reported as extra-data results and should not be mixed into the course-fair table.

## Table 1. Dataset And Protocol

| Split | Dataset | Count | Protocol | Role |
| --- | --- | ---: | --- | --- |
| Train | Pascal VOC cropped | 7,643 | 224 x 224 synthesis | course-fair training |
| Train | Zhang real89 | 89 | paired real | course-fair training/replay |
| Train | OpenRR train | 1k / 3k | paired real | extra-data adaptation |
| Test | CEILNet Table2 | 100 | native | course test, synthetic |
| Test | Zhang real20 | 20 | max long edge 512 | course test, real |
| Test | SIR2 Objects | 200 | native | course test, real |
| Test | SIR2 Postcard | 179 | native | course test, real |
| Test | SIR2 Wild | 101 | native | course test, real |
| External | OpenRR val | 300 | native | external real benchmark |
| Self | self-collected | >=5 | native unless documented resize | course required, pending |

## Table 2. Course-Fair Results

| Method | Uses OpenRR train? | Course PSNR | SSIM | NCC | LMSE | SIR2 PSNR | SIR2 SSIM | SIR2 NCC | SIR2 LMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERRNet baseline | N | 24.5077 | 0.8861 | 0.9465 | 0.0081 | 23.8201 | 0.8871 | 0.9546 | 0.0052 |
| BP-RAP RIC | N | 23.8919 | 0.8839 | 0.9419 | 0.0086 | 24.8216 | 0.9088 | 0.9653 | 0.0034 |
| RIC+FSS | N | 23.8820 | 0.8838 | 0.9420 | 0.0086 | 24.8254 | 0.9087 | 0.9654 | 0.0034 |

Reading: BP-RAP RIC is not a universal replacement for ERRNet, but it improves SIR2 real-scene performance by about 1 dB PSNR.

## Table 3. Structure Ablation

| Method | Prior | Gate | Refine | RIC | FSS | Course PSNR | SIR2 PSNR | OpenRR PSNR | Main reading |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BP-RAP RIC | Y | Y | Y | Y | N | 23.8919 | 24.8216 | 26.8597 | full course-fair model |
| w/o prior | N | Y | Y | Y | N | 23.9071 | 24.8273 | 26.8180 | constant prior P=1 |
| w/o gate | Y | N | Y | Y | N | 23.8790 | 24.8126 | 26.7596 | remove prior-gated adapter |
| w/o refinement | Y | Y | N | Y | N | 23.5954 | 24.5825 | 26.3458 | residual refinement is most important |
| w/o RIC | Y | Y | Y | N | N | 23.8964 | 24.8301 | 26.8846 | RIC has limited metric impact |
| RIC+FSS | Y | Y | Y | Y | Y | 23.8820 | 24.8254 | 26.8909 | frequency supervision is near-neutral |

## Table 4. RAFA / OpenRR Adaptation

| Method | Uses OpenRR train? | OpenRR pairs | Course replay? | lambda_old | Balanced? | Course PSNR | SIR2 PSNR | OpenRR PSNR | Six-set PSNR | Role |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| ERRNet baseline | N | 0 | N | 0 | N | 24.5077 | 23.8201 | 25.4874 | 24.6709 | baseline |
| BP-RAP RIC | N | 0 | N | 0 | N | 23.8919 | 24.8216 | 26.8597 | 24.3865 | course-fair method |
| OpenRR-FT 1k | Y | 1000 | N | 0 | N | 23.7930 | 24.8573 | 28.9560 | 24.6535 | target-only adaptation |
| Soup a0.25 | indirect | 1000 | N/A | N/A | N | 23.9549 | 24.9246 | 27.4103 | 24.5308 | checkpoint soup |
| RAFA1k | Y | 1000 | Y | 0.2 | N | 23.9529 | 24.9043 | 27.7031 | 24.5779 | replay adaptation |
| RAFA3k | Y | 3000 | Y | 0.2 | N | 23.9906 | 24.9469 | 27.8483 | 24.6336 | replay adaptation |
| Balanced RAFA3k | Y | 3000 | Y | 0.2 | Y | 23.9999 | 24.9680 | 27.8349 | 24.6390 | balanced sampling |
| Old04 | Y | 3000 | Y | 0.4 | N | 23.9663 | 24.9159 | 27.6352 | 24.5778 | stronger distillation diagnostic |
| RAFA3k w/o old | Y | 3000 | Y | 0.0 | N | 24.0085 | 24.9772 | 28.0698 | 24.6853 | best single RAFA model |
| OpenRR-only 3k | Y | 3000 | N | 0.0 | N | 23.5304 | 24.7896 | 29.2549 | 24.4845 | highest OpenRR, worse course |

## Table 5. Final Inference Fusion

| Method | Course PSNR | d vs ERRNet | SIR2 PSNR | d vs ERRNet | OpenRR PSNR | d vs ERRNet | Six-set PSNR | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ERRNet | 24.5077 | 0.0000 | 23.8201 | 0.0000 | 25.4874 | 0.0000 | 24.6709 | baseline |
| Fusion a=0.25 | 24.8235 | +0.3159 | 24.4009 | +0.5808 | 26.3980 | +0.9106 | 25.0859 | conservative fusion |
| Fusion a=0.50 | 24.7864 | +0.2787 | 24.8360 | +1.0159 | 27.2208 | +1.7334 | 25.1921 | recommended final method |
| Fusion a=0.75 | 24.5022 | -0.0054 | 25.0495 | +1.2294 | 27.8463 | +2.3589 | 25.0596 | real-data biased fusion |
| RAFA3k no-old | 24.0085 | -0.4992 | 24.9772 | +1.1571 | 28.0698 | +2.5824 | 24.6853 | best single RAFA |
| Adaptive fusion | 24.6497 | +0.1420 | 25.0170 | +1.1970 | 27.9558 | +2.4684 | 25.2007 | promising but selector not final |

## Table 6. Fusion Per-Dataset Delta vs ERRNet

| Method | CEILNet | Zhang20 | Objects | Postcard | Wild | OpenRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fusion a=0.25 | -0.0805 | -0.0826 | +0.6394 | +0.5012 | +0.6018 | +0.9106 |
| Fusion a=0.50 | -0.9618 | -0.6923 | +1.1269 | +0.8489 | +1.0720 | +1.7334 |
| RAFA3k no-old | -3.5994 | -2.3679 | +1.3820 | +0.8870 | +1.2023 | +2.5824 |

## Table 7. Fusion Win Counts

| Dataset | Method | Count | Beat ERRNet | Beat RAFA | Mean d vs ERRNet | Mean d vs RAFA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CEILNet | Fusion a=0.50 | 100 | 40 (40.0%) | 98 (98.0%) | -0.9618 | +2.6375 |
| Zhang20 | Fusion a=0.50 | 20 | 8 (40.0%) | 19 (95.0%) | -0.6923 | +1.6757 |
| SIR2 Objects | Fusion a=0.50 | 200 | 200 (100.0%) | 56 (28.0%) | +1.1269 | -0.2551 |
| SIR2 Postcard | Fusion a=0.50 | 179 | 160 (89.4%) | 95 (53.1%) | +0.8489 | -0.0382 |
| SIR2 Wild | Fusion a=0.50 | 101 | 77 (76.2%) | 57 (56.4%) | +1.0720 | -0.1302 |
| OpenRR val | Fusion a=0.50 | 300 | 271 (90.3%) | 102 (34.0%) | +1.7334 | -0.8490 |

## Table 8. Self-Collected Results

Pending. Required methods:

| Method | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| ERRNet baseline | TODO | TODO | TODO | TODO |
| RAFA3k no-old | TODO | TODO | TODO | TODO |
| Fusion a=0.50 | TODO | TODO | TODO | TODO |

## Figure Caption Drafts

**Figure 1. Method overview.** The system starts from ERRNet-Hyper, adds BP-RAP residual refinement for course-fair real-scene improvement, adapts the residual branch with RAFA using OpenRR and course replay, and finally fuses ERRNet and RAFA predictions at inference time.

**Figure 2. Course PSNR delta.** BP-RAP improves SIR2 but loses CEILNet/Zhang, motivating extra-data adaptation and fusion.

**Figure 3. RAFA/OpenRR adaptation.** OpenRR-train methods improve OpenRR and SIR2; course replay prevents severe course degradation, while old distillation is not always beneficial.

**Figure 4. Fusion summary.** Fusion a=0.50 improves Course, SIR2, OpenRR, and Six-set means over ERRNet while reducing RAFA's CEILNet/Zhang degradation.

**Figure 5. Main qualitative comparison.** Input, ERRNet, RAFA3k-no-old, Fusion a=0.50, and GT. Use SIR2/OpenRR/self examples as the main positive cases.

**Figure 6. Failure/protection cases.** CEILNet/Zhang examples show that ERRNet remains stronger for some strong low-frequency and global-tone reflection cases; fusion protects against RAFA but does not fully match ERRNet.
