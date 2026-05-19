# RAP-ERRNet Design Update v1.1

Last updated: 2026-05-19

## 1. Why This Update

The original v1.0 design correctly defined the three RAP components:

1. Physics-guided reflection synthesis.
2. Reflection prior prediction.
3. Reflection-aware refinement.

The first mid-training experiment showed a practical issue: the initial RAP run
used a 3-channel ERRNet backbone trained from scratch, while the course baseline
uses ERRNet with `--hyper`, where RGB is concatenated with VGG19 hypercolumn
features and the first convolution is `1475 -> 256`.

This makes the comparison asymmetric. The baseline starts from a stronger
architecture and a pretrained checkpoint, while the initial RAP run has to learn
both the backbone and the new prior/refinement modules from scratch.

## 2. Updated Main Method

The recommended main method is now:

```text
RAP-ERRNet-Hyper:
  ERRNet --hyper coarse backbone
  + full pretrained ERRNet initialization
  + reflection prior head
  + prior-gated residual adapter
  + lightweight residual refinement
  + physics-guided synthesis
```

This is implemented by:

```text
configs/rap_errnet_hyper_pretrained.yaml
```

Key settings:

```yaml
model:
  use_hypercolumn_backbone: true
  pretrained_errnet_path: checkpoints/errnet/errnet_060_00463920.pt
```

## 3. Expected Benefit

The update should mainly help datasets where the from-scratch 3-channel RAP run
underperformed:

- CEILNet Table2
- Zhang real20

It preserves the promising behavior already observed on SIR2 Objects and Wild,
where the prior/refinement path improved structural metrics such as SSIM, NCC,
and LMSE.

## 4. Experiment Positioning

Use the experiments as follows:

| Experiment | Purpose |
| --- | --- |
| `rap_errnet_main` | From-scratch RAP reference; useful for showing that the proposed modules can learn. |
| `rap_errnet_hyper_pretrained` | Main result candidate; architecture-matched and initialized from the course ERRNet baseline. |
| `rap_no_prior` | Ablation for reflection prior head. |
| `rap_no_gating` | Ablation for prior-gated adapter. |
| `rap_no_refine` | Ablation for residual refinement. |

The baseline does not need to be retrained for the main report if the provided
course checkpoint is used consistently and its metrics are reproduced with
`test_errnet.py --hyper`.

## 5. Recommended Commands

Debug first:

```bash
CUDA_VISIBLE_DEVICES=0 python train_rap_errnet.py \
  --config configs/rap_errnet_hyper_pretrained.yaml \
  --name rap_errnet_hyper_pretrained_debug \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 4 \
  --epochs 1 \
  --num_workers 4 \
  --device auto \
  --debug \
  --progress_bar
```

Main run:

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

Evaluation:

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

## 6. Report Wording

Suggested wording:

> After the initial from-scratch RAP-ERRNet experiment, we observed that the
> original ERRNet baseline used VGG19 hypercolumn features and a pretrained
> checkpoint, whereas the first RAP implementation used a 3-channel backbone.
> To make the comparison architecture-consistent and to better isolate the
> effect of reflection-aware priors and refinement, we introduced a hypercolumn
> RAP-ERRNet variant that reuses the original ERRNet `--hyper` input pathway and
> loads the provided pretrained ERRNet weights before training the RAP modules.

