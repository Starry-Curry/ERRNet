# RAP-ERRNet Final Iteration Design v1.2

Last updated: 2026-05-19

This document consolidates the initial RAP-ERRNet proposal, current experiment
evidence, and recent SIRR research directions into the final recommended project
design.

## 1. Current Evidence

### 1.1 Baseline

The course baseline is the pretrained ERRNet `--hyper` checkpoint:

```text
checkpoints/errnet/errnet_060_00463920.pt
```

It is strong on CEILNet Table 2 and good on real test sets.

### 1.2 From-Scratch RAP

The 3-channel from-scratch RAP run shows that prior/refinement modules can learn,
especially on SIR2 Objects and SIR2 Wild. However, it remains weak on CEILNet
and Zhang real20. This run should be reported as a from-scratch reference, not
as the main method.

### 1.3 Hyper-Pretrained RAP

The hypercolumn-pretrained RAP run is the main result candidate. It inherits the
course baseline architecture and weights, then learns reflection-aware residual
corrections. Mid-evaluation already improves SIR2 Objects/Postcard/Wild over
baseline and sharply improves CEILNet relative to from-scratch RAP, although
CEILNet still lags behind the baseline.

### 1.4 Visual Diagnosis

Qualitative comparison shows that many RAP outputs are close to baseline, but
CEILNet losses are often caused by small color/brightness drift or local
over-correction rather than complete failure to remove reflections. The prior
maps are still too smooth and sometimes behave like saliency/edge maps instead
of accurate reflection-location maps.

## 2. Research Takeaways

Recent work suggests four useful directions:

1. Real paired data and explicit reflection-location guidance are valuable. RRW
   and MaxRF show that high-quality paired data can be used to derive explicit
   reflection locations for a two-stage detection/removal framework.
2. Dual-stream interaction is now common. DSIT separates transmission and
   reflection streams and uses attention to exchange information.
3. Information-preserving decoupling is important. RDNet uses reversible
   decoupling to avoid losing transmission details while separating reflection
   features.
4. Generative/diffusion methods such as L-DiffER emphasize multi-condition
   constraints to preserve color and structure. For this project, this mainly
   supports adding fidelity/anchor constraints rather than adopting a heavy
   diffusion model.

## 3. Final Method

The final project method should be named:

```text
RAP-ERRNet-Hyper-ZeroRes
```

Architecture:

```text
I
 -> VGG19 hypercolumn + ERRNet backbone
 -> coarse transmission T0
 -> ReflectionPriorHead(I) = P
 -> zero-initialized prior-gated adapter
 -> zero-initialized residual refinement
 -> final transmission T
```

Forward:

```text
P = PriorHead(I)
T0 = ERRNet_hyper(I)
Tg = GatedAdapter(T0, P)
R0 = I - Tg
delta = residual_scale * tanh(Refine([I, Tg, R0, P])) * P
T = clamp(Tg + delta, 0, 1)
```

The gated adapter and refinement final layer are zero-initialized, so the model
starts from the pretrained ERRNet output and learns conservative corrections.

## 4. Data Strategy

### 4.1 Main Training Data

Use the course-standard training set for the main comparison:

```text
VOC physics synthesis + Zhang real89 train
```

Do not use CEILNet Table 2, Zhang real20, SIR2 Objects/Postcard/Wild, or
self-collected test images for training.

### 4.2 Synthesis Improvements

Keep the physics-guided synthesis, but tune it to better cover CEILNet-like and
real-world reflections:

- mix blurred and sharper ghost reflections;
- sample both low-frequency veil reflections and local object-like reflections;
- add highlight/saturation clipping before blending;
- add mild gamma/color temperature perturbation;
- keep mask `M` smoother for veil reflection and sharper for local reflection;
- preserve a deterministic seed path for reproducibility.

Recommended synthesis formula:

```text
Rb1 = Blur(R, sigma1)
Rb2 = warp(Blur(R, sigma2), dx, dy)
Rg = alpha1 * Rb1 + alpha2 * Rb2
M  = normalize(Blur(noise, sigma_m))
I  = clip((1 - kappa * M) * T + beta * M * Rg + eps)
```

Add optional sharper local component:

```text
I = clip(I + beta_local * M_local * sharpen(R))
```

This should be implemented as an optional synthesis mode, not as a default that
breaks comparability.

### 4.3 Extra Data

Use OpenRR-5k or RRW-style real paired data only as an extra-data fine-tuning
experiment:

```text
RAP-ERRNet-Hyper-ZeroRes
RAP-ERRNet-Hyper-ZeroRes + Extra Real Fine-tuning
```

Report them separately. Mixing extra datasets into the main run would make
baseline comparison less clean.

## 5. Loss Design

Current loss remains a good base:

```text
L = L_pix + lambda_grad L_grad + lambda_ssim L_ssim
  + lambda_mask L_mask + lambda_clean L_clean
```

Final recommended additions:

### 5.1 Baseline Anchor Loss

Protect areas where the prior says reflection is weak:

```text
L_anchor = mean((1 - stopgrad(P)) * |T - T0|)
```

This directly addresses CEILNet degradation caused by unnecessary residual
changes.

Recommended weight:

```text
lambda_anchor = 0.05 to 0.10
```

### 5.2 Residual Magnitude Regularization

Encourage the refinement to remain local:

```text
L_delta = mean(|delta|)
```

Recommended weight:

```text
lambda_delta = 0.01
```

### 5.3 Pseudo-Mask Downweighting

For synthetic VOC pairs, the reflection mask is known and reliable. For real
paired images without masks, `abs(I - T)` pseudo masks are noisy and should not
be treated as ground truth.

Recommended:

```text
lambda_mask_synth = 0.05
lambda_mask_pseudo = 0.005 to 0.01
```

If implementation time is limited, set global `lambda_mask` lower for
hyper-pretrained training:

```text
lambda_mask = 0.01
```

### 5.4 Loss Weights For Final Main Run

Recommended starting point:

```yaml
loss:
  lambda_pix: 1.0
  lambda_perc: 0.0
  lambda_grad: 0.1
  lambda_ssim: 0.2
  lambda_mask: 0.01
  lambda_clean: 0.05
  lambda_excl: 0.0
  lambda_reflection_weight: 2.0
  lambda_anchor: 0.05
  lambda_delta: 0.01
```

## 6. Training Plan

### Stage 0: Baseline Reproduction

Use the course checkpoint and `test_errnet.py --hyper`. No baseline retraining
is required unless the report specifically asks for it.

### Stage 1: Conservative RAP Warm-Up

Train only the new RAP modules:

```text
freeze ERRNet backbone
train PriorHead + GatedAdapter + Refinement
epochs: 10-20
lr: 1e-4
batch_size: 32 on PPU, 8-16 on A6000
```

Purpose: learn reflection priors and residual corrections without damaging the
pretrained backbone.

### Stage 2: Full Fine-Tuning

Unfreeze the backbone with a lower learning rate:

```text
backbone lr: 1e-5
new modules lr: 5e-5
epochs: continue to 100 total
```

If separate parameter-group LR is not implemented, use a conservative global LR:

```text
lr = 5e-5
```

### Stage 3: Optional Extra Real Fine-Tuning

Use OpenRR-5k or other extra real paired data only after the main result is
stable:

```text
lr = 1e-5 to 2e-5
epochs = 10-30
```

Report separately as extra-data fine-tuning.

## 7. Evaluation Plan

Primary metrics:

```text
PSNR, SSIM, NCC, LMSE
```

Primary datasets:

```text
CEILNet Table 2
Zhang real20
SIR2 Objects
SIR2 Postcard
SIR2 Wild
self-collected 5 scenes
```

For hypercolumn RAP, full-resolution Zhang20 can OOM on 48GB A6000. Evaluate it
on the 98GB PPU or use a documented diagnostic setting separately.

Qualitative visualization:

```text
Input | Baseline | Output | GT | Error Map | Prior Map
```

Use CEILNet first for direct baseline comparison, then SIR2 Wild/Objects for
real-world structural improvements.

## 8. Experiment Table

Minimum report table:

| Method | Training Data | Init | Notes |
| --- | --- | --- | --- |
| ERRNet baseline | course standard | pretrained | `test_errnet.py --hyper` |
| RAP from scratch | course standard | random | reference only |
| RAP-Hyper | course standard | ERRNet pretrained | current main candidate |
| RAP-Hyper-ZeroRes | course standard | ERRNet pretrained | final main method |
| RAP-Hyper-ZeroRes + Extra | course + extra real | fine-tuned | optional |

Minimum ablations:

```text
no prior
no gated adapter
no refinement
no anchor loss
```

If time is limited, run ablations for 30 epochs and clearly mark them as
short-run ablations.

## 9. Recommended Next Actions

1. Pull the zero-residual code update on the PPU server.
2. Start `rap_errnet_hyper_pretrained_zerores_ppu_bs32` with `lr=5e-5`.
3. Evaluate CEILNet and SIR2 Wild at epoch 20-30.
4. If CEILNet improves while SIR2 remains strong, keep it as the final main
   method.
5. If CEILNet is still weak, implement anchor/delta losses and reduce
   `lambda_mask` to `0.01`.
6. Generate qualitative comparisons for CEILNet and SIR2 Wild.
7. Only after main results are stable, run extra-data fine-tuning.

## 10. Report Claim

Suggested final claim:

> RAP-ERRNet improves the course ERRNet baseline by adding reflection-location
> awareness and conservative prior-guided residual refinement. To preserve the
> strong pretrained baseline behavior, RAP residual branches are zero-initialized
> and trained with physics-guided synthesis plus real paired supervision. The
> method is especially effective on real-world SIR2 subsets, improving structural
> and local metrics, while conservative anchoring is used to avoid degrading
> synthetic paired benchmarks such as CEILNet.

