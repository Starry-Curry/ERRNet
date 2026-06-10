# 强反射保护训练 Runbook

目标：在尽量保留 RAFA / OpenRR / SIR2 优势的同时，减少 CEILNet 和 Zhang20 上相对 ERRNet 的大幅退化。

核心思路：

- VOC physics synthesis 中加入 `strong_prob` 强反射模式，生成更接近 CEILNet/Zhang20 失败样例的大面积低频反射、较强 ghost、全局 veil。
- 训练时只对 `hard_synth=1` 的强合成样本启用 `lambda_hard_anchor`，把最终输出轻微拉回 frozen ERRNet backbone 输出。
- 这不是全局 old distillation，不会强迫所有 OpenRR/真实样例靠近旧模型。

## 1. 拉取代码

```bash
cd /mnt/workspace/ERRNet
git pull origin dip26

python - <<'PY'
from datasets.unified_reflection_dataset import list_available_datasets
print(list_available_datasets("./data"))
PY
```

需要至少可用：

```text
voc=True
zhang_train=True
zhang20=True
ceilnet=True
sir2_objects=True
sir2_postcard=True
sir2_wild=True
openrr_train=True
openrr_val=True
```

## 2. 主实验：RAFA3k w/o old + hard synthesis 继续微调

这是优先跑的实验。它从当前 extra-data 最优 `RAFA3k w/o old` 继续微调 6 epoch。

```bash
cd /mnt/workspace/ERRNet

RUN=bp_rap_rafa_openrr3k_no_old_hard_synth_e6

python train_rap_errnet.py \
  --config configs/bp_rap_rafa_openrr3k_no_old_hard_synth.yaml \
  --name $RUN \
  --resume checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt \
  --resume_model_only \
  --data_root ./data \
  --use_openrr \
  --max_openrr_pairs 3000 \
  --use_course_replay \
  --balance_openrr_bins \
  --reflection_bins_csv results/reflection_strength_openrr3k_zhang.csv \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --lambda_old 0 \
  --old_loss_prob 0 \
  --batch_size 24 \
  --epochs 6 \
  --lr 1.0e-5 \
  --new_lr 1.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

训练日志里应出现 `hard_anchor=...`。如果一直是 0，说明没有启用 `--use_physics_synthesis` 或强合成样本没有进入 batch。

## 3. 备选实验：课程公平 hard-synth BP-RAP

这个不使用 OpenRR train，可以作为课程公平设置下的保护实验。优先级低于第 2 节。

```bash
cd /mnt/workspace/ERRNet

RUN=bp_rap_hyper_ric_hard_synth_e12

python train_rap_errnet.py \
  --config configs/bp_rap_hyper_ric_hard_synth.yaml \
  --name $RUN \
  --resume checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt \
  --resume_model_only \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --batch_size 24 \
  --epochs 12 \
  --lr 2.0e-5 \
  --new_lr 2.0e-5 \
  --backbone_lr 0 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

## 4. 正式评估

不要把 all-512 diagnostic 放进主论文表格。正式协议如下：

- CEILNet：native
- Zhang20：`--max_long_edge 512`
- SIR2 Objects/Postcard/Wild：native
- OpenRR val：native

对第 2 节主实验跑：

```bash
cd /mnt/workspace/ERRNet

RUN=bp_rap_rafa_openrr3k_no_old_hard_synth_e6

python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr3k_no_old_hard_synth.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --data_root ./data \
  --save_dir results/${RUN}_standard \
  --datasets ceilnet,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --device auto

python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr3k_no_old_hard_synth.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --data_root ./data \
  --save_dir results/${RUN}_zhang20_512 \
  --datasets zhang20 \
  --max_long_edge 512 \
  --device auto
```

如果也跑了第 3 节课程公平实验，把 `RUN` 和 `--config` 换成：

```bash
RUN=bp_rap_hyper_ric_hard_synth_e12
CONFIG=configs/bp_rap_hyper_ric_hard_synth.yaml
```

## 5. 结果怎么看

用新增汇总脚本对比 ERRNet、RAFA3k w/o old 和 hard-synth 新结果：

```bash
cd /mnt/workspace/ERRNet

python tools/summarize_formal_runs.py \
  --run ERRNet=results/final_eval_errnet_standard,results/final_eval_errnet_zhang20_512 \
  --run RAFA3k_no_old=results/final_eval_rafa3k_no_old_standard,results/final_eval_rafa3k_no_old_zhang20_512 \
  --run HardSynth=results/bp_rap_rafa_openrr3k_no_old_hard_synth_e6_standard,results/bp_rap_rafa_openrr3k_no_old_hard_synth_e6_zhang20_512 \
  --baseline ERRNet \
  --out results/HARD_SYNTHESIS_SUMMARY.md
```

重点看四个读数：

1. `CEILNet d`：相对 ERRNet 的负数是否缩小。原来 RAFA/BP-RAP 大约明显为负，目标是至少少掉一部分。
2. `Zhang20 d`：相对 ERRNet 的负数是否缩小。
3. `SIR2 PSNR`：不能大幅掉。允许小幅回退，但如果掉超过 0.3 dB，要谨慎。
4. `OpenRR PSNR`：不能把 OpenRR 适应完全抹掉。如果 OpenRR 下降很多，说明 hard anchor 太强或 course ratio 太高。

理想结果：

```text
CEILNet/Zhang20 比 RAFA3k_no_old 更接近 ERRNet；
SIR2/OpenRR 比 RAFA3k_no_old 只小幅下降；
Six-set mean 不明显变差，或者 trade-off 更适合论文叙事。
```

如果 CEILNet/Zhang20 没改善：

- 下一轮把 `strong_prob` 提到 `0.75`
- 把 `course_replay_ratio` 提到 `0.7`
- 把 `lambda_hard_anchor` 从 `0.12` 提到 `0.20`

如果 SIR2/OpenRR 掉太多：

- 把 `lambda_hard_anchor` 降到 `0.06`
- 把 `course_replay_ratio` 降回 `0.5`
- 保留 `strong_prob=0.60`

## 6. 生成诊断图

先用现有 CEILNet/Zhang diagnostic 脚本比较 hard-synth 新模型：

```bash
cd /mnt/workspace/ERRNet

python tools/make_ceilnet_zhang_diagnostics.py \
  --data_root ./data \
  --datasets ceilnet,zhang20 \
  --out_dir results/ceilnet_zhang_diagnostics_hardsynth \
  --figures_dir paper/figures \
  --max_long_edge 512 \
  --worst_k 10 \
  --similar_k 6 \
  --better_k 4 \
  --cell_width 190 \
  --errnet_ckpt checkpoints/errnet/errnet_060_00463920.pt \
  --errnet_hyper \
  --bprap_ckpt checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt \
  --bprap_config configs/bp_rap_hyper_zerores_staged_ric.yaml \
  --rafa_ckpt checkpoints/bp_rap_rafa_openrr3k_no_old_hard_synth_e6/best.pt \
  --rafa_config configs/bp_rap_rafa_openrr3k_no_old_hard_synth.yaml \
  --rafa_label HardSynth \
  --device auto
```

下载或备份这些文件：

```bash
BACKUP=/mnt/data/ERRNet_hard_synth_$(date +%Y%m%d)
mkdir -p $BACKUP/results $BACKUP/checkpoints $BACKUP/paper/figures

cp -av checkpoints/bp_rap_rafa_openrr3k_no_old_hard_synth_e6 $BACKUP/checkpoints/
cp -av results/bp_rap_rafa_openrr3k_no_old_hard_synth_e6_standard $BACKUP/results/
cp -av results/bp_rap_rafa_openrr3k_no_old_hard_synth_e6_zhang20_512 $BACKUP/results/
cp -av results/HARD_SYNTHESIS_SUMMARY.md $BACKUP/results/
cp -av results/ceilnet_zhang_diagnostics_hardsynth $BACKUP/results/
cp -av paper/figures/ceilnet_*_diagnostics.* $BACKUP/paper/figures/ 2>/dev/null || true
cp -av paper/figures/zhang20_*_diagnostics.* $BACKUP/paper/figures/ 2>/dev/null || true

du -sh $BACKUP
find $BACKUP -maxdepth 4 -type f | sort
```
