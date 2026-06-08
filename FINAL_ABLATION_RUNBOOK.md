# Final Ablation And Paper Figure Runbook

This runbook is for the final BP-RAP / RAFA course-paper experiments. It keeps
course-fair evaluation, OpenRR extra-data evaluation, self-collected evaluation,
and fast diagnostics separate.

Do not put all-512 diagnostic numbers into the main paper tables.

## 0. Pull Latest Code

```bash
cd /mnt/workspace/ERRNet
git pull origin dip26

python - <<'PY'
from datasets.unified_reflection_dataset import list_available_datasets
print(list_available_datasets("./data"))
PY
```

Expected before final runs:

```text
ceilnet=True
zhang20=True
sir2_objects=True
sir2_postcard=True
sir2_wild=True
openrr_train=True
openrr_val=True
self=True   # after self-collected data is prepared
```

Self-collected layout:

```text
data/self_collected/test/scene_001/blended.png
data/self_collected/test/scene_001/transmission.png
...
```

## 1. Shared Variables

```bash
cd /mnt/workspace/ERRNet

ERRNET=checkpoints/errnet/errnet_060_00463920.pt
RAP_HYPER=checkpoints/rap_errnet_hyper_pretrained_ppu_bs32/best.pt
BPRAP=checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt
BPRAP_FSS=checkpoints/bp_rap_hyper_ric_freq_ft_from_ric_ppu_bs24/best.pt
OPENRR_FT=checkpoints/bp_rap_ric_openrr1k_realonly_freeze_e10/best.pt
SOUP=checkpoints/bp_rap_ric_openrr1k_soup_a0p25/best.pt
RAFA1K=checkpoints/bp_rap_rafa_openrr1k_e10/best.pt
RAFA3K=checkpoints/bp_rap_rafa_openrr3k_e10/best.pt
RAFA_BAL=checkpoints/bp_rap_rafa_openrr3k_balanced_e10/best.pt
RAFA_OLD04=checkpoints/bp_rap_rafa_openrr3k_old04_e10/best.pt

COURSE_NATIVE=ceilnet,sir2_objects,sir2_postcard,sir2_wild
ZHANG=zhang20
OPENRR=openrr_val
SELF=self
```

## 2. Train Required Structure Ablations

Use BP-RAP RIC as the full model reference. All ablations below start from
`RAP_HYPER`, use the same 20-epoch frozen-backbone schedule, and use the formal
course data only.

### 2.1 w/o prior

The model disables the learned prior head. Internally this uses a constant
all-ones prior `P=1`, while gate/refinement/RIC remain active.

```bash
RUN=bp_rap_ablate_no_prior_ric_e20
python train_rap_errnet.py \
  --config configs/bp_rap_ablate_no_prior_ric.yaml \
  --name $RUN \
  --resume $RAP_HYPER \
  --resume_model_only \
  --data_root ./data \
  --use_physics_synthesis \
  --use_ric \
  --no_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 24 \
  --epochs 20 \
  --lr 5.0e-5 \
  --new_lr 5.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

### 2.2 w/o gate

```bash
RUN=bp_rap_ablate_no_gate_ric_e20
python train_rap_errnet.py \
  --config configs/bp_rap_ablate_no_gate_ric.yaml \
  --name $RUN \
  --resume $RAP_HYPER \
  --resume_model_only \
  --data_root ./data \
  --use_physics_synthesis \
  --use_ric \
  --use_prior_head \
  --no_gated_blocks \
  --use_refinement \
  --batch_size 24 \
  --epochs 20 \
  --lr 5.0e-5 \
  --new_lr 5.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

### 2.3 w/o refinement

This disables the residual refinement head, so `Delta=0` and output is `Tg`.

```bash
RUN=bp_rap_ablate_no_refine_ric_e20
python train_rap_errnet.py \
  --config configs/bp_rap_ablate_no_refine_ric.yaml \
  --name $RUN \
  --resume $RAP_HYPER \
  --resume_model_only \
  --data_root ./data \
  --use_physics_synthesis \
  --use_ric \
  --use_prior_head \
  --use_gated_blocks \
  --no_refinement \
  --batch_size 24 \
  --epochs 20 \
  --lr 5.0e-5 \
  --new_lr 5.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

### 2.4 w/o RIC

Strict version: same schedule as BP-RAP RIC but `lambda_ric=0`.

```bash
RUN=bp_rap_ablate_no_ric_e20
python train_rap_errnet.py \
  --config configs/bp_rap_ablate_no_ric.yaml \
  --name $RUN \
  --resume $RAP_HYPER \
  --resume_model_only \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 24 \
  --epochs 20 \
  --lr 5.0e-5 \
  --new_lr 5.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

If time runs out, use the existing `RAP_HYPER` checkpoint as the w/o RIC
reference, but label it clearly as "RAP-Hyper reference" rather than strict
same-schedule w/o RIC.

## 3. Train RAFA / OpenRR New Ablations

### 3.1 RAFA3k w/o old distillation

Same as RAFA3k, but `lambda_old=0`.

```bash
RUN=bp_rap_rafa_openrr3k_no_old_e10
python train_rap_errnet.py \
  --config configs/bp_rap_rafa_openrr3k_no_old.yaml \
  --name $RUN \
  --resume $BPRAP \
  --resume_model_only \
  --data_root ./data \
  --use_openrr \
  --max_openrr_pairs 3000 \
  --use_course_replay \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --lambda_old 0 \
  --old_loss_prob 0 \
  --batch_size 24 \
  --epochs 10 \
  --lr 2.0e-5 \
  --new_lr 2.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

### 3.2 Optional: OpenRR-only 3k / w/o course replay

This is not course-fair. It is essentially OpenRR-FT 3k and must be labeled as
OpenRR-only adaptation.

```bash
RUN=bp_rap_rafa_openrr3k_noreplay_e10
python train_rap_errnet.py \
  --config configs/bp_rap_rafa_openrr3k_noreplay.yaml \
  --name $RUN \
  --resume $BPRAP \
  --resume_model_only \
  --data_root ./data \
  --use_openrr \
  --max_openrr_pairs 3000 \
  --no_zhang_train \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 24 \
  --epochs 10 \
  --lr 2.0e-5 \
  --new_lr 2.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

## 4. Formal Evaluation Functions

Use these functions to avoid accidentally mixing the all-512 diagnostic protocol
with formal evaluation.

```bash
eval_errnet_formal () {
  TAG=$1
  python eval_all.py --model errnet --ckpt $ERRNET --errnet_hyper \
    --data_root ./data --save_dir results/${TAG}_standard \
    --datasets $COURSE_NATIVE --device auto
  python eval_all.py --model errnet --ckpt $ERRNET --errnet_hyper \
    --data_root ./data --save_dir results/${TAG}_zhang20_512 \
    --datasets $ZHANG --max_long_edge 512 --device auto
  python eval_all.py --model errnet --ckpt $ERRNET --errnet_hyper \
    --data_root ./data --save_dir results/${TAG}_openrr \
    --datasets $OPENRR --device auto
  python eval_all.py --model errnet --ckpt $ERRNET --errnet_hyper \
    --data_root ./data --save_dir results/${TAG}_self \
    --datasets $SELF --device auto
}

eval_rap_formal () {
  TAG=$1
  CKPT=$2
  CONFIG=$3
  python eval_all.py --model rap_errnet --config $CONFIG --ckpt $CKPT \
    --data_root ./data --save_dir results/${TAG}_standard \
    --datasets $COURSE_NATIVE --device auto
  python eval_all.py --model rap_errnet --config $CONFIG --ckpt $CKPT \
    --data_root ./data --save_dir results/${TAG}_zhang20_512 \
    --datasets $ZHANG --max_long_edge 512 --device auto
  python eval_all.py --model rap_errnet --config $CONFIG --ckpt $CKPT \
    --data_root ./data --save_dir results/${TAG}_openrr \
    --datasets $OPENRR --device auto
  python eval_all.py --model rap_errnet --config $CONFIG --ckpt $CKPT \
    --data_root ./data --save_dir results/${TAG}_self \
    --datasets $SELF --device auto
}
```

If self-collected native resolution causes OOM, rerun only self with a documented
resize and note it in the report. Do not silently mix resized self metrics with
native self metrics.

## 5. Run Formal Evaluations

### 5.1 Baselines and course-fair methods

```bash
eval_errnet_formal final_eval_errnet

eval_rap_formal final_eval_bprap_ric \
  $BPRAP \
  configs/bp_rap_hyper_zerores_staged_ric.yaml

eval_rap_formal final_eval_bprap_ric_fss \
  $BPRAP_FSS \
  configs/bp_rap_hyper_zerores_staged_ric_freq.yaml
```

### 5.2 Structure ablations

```bash
eval_rap_formal final_eval_ablate_no_prior \
  checkpoints/bp_rap_ablate_no_prior_ric_e20/best.pt \
  configs/bp_rap_ablate_no_prior_ric.yaml

eval_rap_formal final_eval_ablate_no_gate \
  checkpoints/bp_rap_ablate_no_gate_ric_e20/best.pt \
  configs/bp_rap_ablate_no_gate_ric.yaml

eval_rap_formal final_eval_ablate_no_refine \
  checkpoints/bp_rap_ablate_no_refine_ric_e20/best.pt \
  configs/bp_rap_ablate_no_refine_ric.yaml

eval_rap_formal final_eval_ablate_no_ric \
  checkpoints/bp_rap_ablate_no_ric_e20/best.pt \
  configs/bp_rap_ablate_no_ric.yaml
```

### 5.3 RAFA / OpenRR adaptation methods

```bash
eval_rap_formal final_eval_openrr_ft_1k \
  $OPENRR_FT \
  configs/bp_rap_hyper_zerores_staged_ric.yaml

eval_rap_formal final_eval_soup_a0p25 \
  $SOUP \
  configs/bp_rap_hyper_zerores_staged_ric.yaml

eval_rap_formal final_eval_rafa1k \
  $RAFA1K \
  configs/bp_rap_rafa_openrr1k.yaml

eval_rap_formal final_eval_rafa3k \
  $RAFA3K \
  configs/bp_rap_rafa_openrr1k.yaml

eval_rap_formal final_eval_rafa3k_balanced \
  $RAFA_BAL \
  configs/bp_rap_rafa_openrr3k_balanced.yaml

eval_rap_formal final_eval_rafa3k_old04 \
  $RAFA_OLD04 \
  configs/bp_rap_rafa_openrr3k_old04.yaml

eval_rap_formal final_eval_rafa3k_no_old \
  checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt \
  configs/bp_rap_rafa_openrr3k_no_old.yaml

# Optional
eval_rap_formal final_eval_rafa3k_noreplay \
  checkpoints/bp_rap_rafa_openrr3k_noreplay_e10/best.pt \
  configs/bp_rap_rafa_openrr3k_noreplay.yaml
```

## 6. Generate Tables And Numeric Figures

This creates:

- `results/ABLATION_STRUCTURE_SUMMARY.md`
- `results/ABLATION_RAFA_SUMMARY.md`
- `results/SELF_COLLECTED_SUMMARY.md`
- `results/FINAL_TABLES_FOR_PAPER.md`
- `paper/figures/method_overview.png/.pdf`
- `paper/figures/course_delta_psnr.png/.pdf`
- `paper/figures/openrr_adaptation_summary.png/.pdf`

```bash
python tools/build_final_tables_and_plots.py \
  --results_root results \
  --figures_dir paper/figures \
  --dpi 220
```

## 7. Generate Qualitative Figures

This creates:

- `paper/figures/qualitative_main.png/.pdf`
- `paper/figures/prior_error_analysis.png/.pdf`
- `paper/figures/failure_cases.png/.pdf`
- `results/paper_qualitative/selection_summary.csv`

```bash
python tools/make_paper_qualitative_figures.py \
  --data_root ./data \
  --figures_dir paper/figures \
  --out_dir results/paper_qualitative \
  --max_long_edge 512 \
  --errnet_ckpt $ERRNET \
  --errnet_hyper \
  --bprap_ckpt $BPRAP \
  --bprap_config configs/bp_rap_hyper_zerores_staged_ric.yaml \
  --rafa_ckpt $RAFA_BAL \
  --rafa_config configs/bp_rap_rafa_openrr3k_balanced.yaml \
  --rafa_label Balanced-RAFA3k \
  --device auto
```

## 8. What To Download Back

Minimum package for local paper writing:

```text
results/ABLATION_STRUCTURE_SUMMARY.md
results/ABLATION_RAFA_SUMMARY.md
results/SELF_COLLECTED_SUMMARY.md
results/FINAL_TABLES_FOR_PAPER.md
results/paper_qualitative/selection_summary.csv
paper/figures/*.png
paper/figures/*.pdf
```

If you want local re-plotting without rerunning models, also download:

```text
results/final_eval_*/*.csv
results/final_eval_*/metrics_*.csv
```

If you want local qualitative regeneration, download the checkpoints and data
too, but this is usually unnecessary. It is better to generate qualitative
figures on the server and download the final PNG/PDF files.

For backup to OSS:

```bash
BACKUP=/mnt/data/ERRNet_final_ablation_$(date +%Y%m%d)
mkdir -p $BACKUP
cp -av results/ABLATION_STRUCTURE_SUMMARY.md \
  results/ABLATION_RAFA_SUMMARY.md \
  results/SELF_COLLECTED_SUMMARY.md \
  results/FINAL_TABLES_FOR_PAPER.md \
  $BACKUP/
cp -av results/final_eval_* $BACKUP/ 2>/dev/null || true
cp -av results/paper_qualitative $BACKUP/ 2>/dev/null || true
mkdir -p $BACKUP/paper
cp -av paper/figures $BACKUP/paper/
```

