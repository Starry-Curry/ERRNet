# ERRNet-RAFA 推理融合 Runbook

目的：HardSynth 训练没有明显拉回 CEILNet/Zhang20，因此下一步优先测试推理阶段融合：

```text
T_fused = alpha * T_rafa + (1 - alpha) * T_errnet
```

其中：

- `alpha=0` 等价于 ERRNet；
- `alpha=1` 等价于 RAFA；
- 中间值测试是否能在 CEILNet/Zhang20 上接近 ERRNet，同时保留 SIR2/OpenRR 的一部分优势。

这个实验不重新训练，成本低，适合快速判断是否存在可用 trade-off。

## 1. 拉取代码

```bash
cd /mnt/workspace/ERRNet
git pull origin dip26
```

## 2. 跑 RAFA3k w/o old 的融合 sweep

正式协议仍然拆成两段：

- CEILNet/SIR2/OpenRR：native
- Zhang20：`--max_long_edge 512`

```bash
cd /mnt/workspace/ERRNet

ERRNET=checkpoints/errnet/errnet_060_00463920.pt
RAFA=checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt
RAFA_CFG=configs/bp_rap_rafa_openrr3k_no_old.yaml

python tools/eval_fusion_sweep.py \
  --data_root ./data \
  --datasets ceilnet,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --save_dir results/fusion_rafa3k_no_old_standard \
  --errnet_ckpt $ERRNET \
  --errnet_hyper \
  --rafa_ckpt $RAFA \
  --rafa_config $RAFA_CFG \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --include_adaptive \
  --device auto

python tools/eval_fusion_sweep.py \
  --data_root ./data \
  --datasets zhang20 \
  --max_long_edge 512 \
  --save_dir results/fusion_rafa3k_no_old_zhang20_512 \
  --errnet_ckpt $ERRNET \
  --errnet_hyper \
  --rafa_ckpt $RAFA \
  --rafa_config $RAFA_CFG \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --include_adaptive \
  --device auto
```

## 3. 汇总正式结果

```bash
cd /mnt/workspace/ERRNet

python tools/summarize_formal_runs.py \
  --run ERRNet=results/fusion_rafa3k_no_old_standard/alpha_0p00,results/fusion_rafa3k_no_old_zhang20_512/alpha_0p00 \
  --run Fusion_a0p25=results/fusion_rafa3k_no_old_standard/alpha_0p25,results/fusion_rafa3k_no_old_zhang20_512/alpha_0p25 \
  --run Fusion_a0p50=results/fusion_rafa3k_no_old_standard/alpha_0p50,results/fusion_rafa3k_no_old_zhang20_512/alpha_0p50 \
  --run Fusion_a0p75=results/fusion_rafa3k_no_old_standard/alpha_0p75,results/fusion_rafa3k_no_old_zhang20_512/alpha_0p75 \
  --run RAFA3k_no_old=results/fusion_rafa3k_no_old_standard/alpha_1p00,results/fusion_rafa3k_no_old_zhang20_512/alpha_1p00 \
  --run Fusion_adaptive=results/fusion_rafa3k_no_old_standard/adaptive,results/fusion_rafa3k_no_old_zhang20_512/adaptive \
  --baseline ERRNet \
  --out results/FUSION_SWEEP_SUMMARY.md
```

## 4. 怎么读结果

优先看 `results/FUSION_SWEEP_SUMMARY.md`。

判断标准：

1. `Fusion_a0p25` 或 `Fusion_a0p50` 的 CEILNet/Zhang20 dPSNR 是否明显比 RAFA3k_no_old 小。
2. SIR2 dPSNR 是否仍为正。只要 SIR2 mean 仍明显高于 ERRNet，这个融合就有论文叙事价值。
3. OpenRR dPSNR 是否仍为正。OpenRR 不能退回到 ERRNet 附近。
4. `Fusion_adaptive` 是否优于最好的 constant alpha。如果没有，不要在论文里主推 adaptive，直接用 constant fusion 当推理策略。

推荐选择：

- 如果 `alpha=0.5`：CEILNet/Zhang 明显改善，SIR2/OpenRR 仍有优势，选 `Fusion_a0p50`。
- 如果 `alpha=0.25`：CEILNet/Zhang 更接近 ERRNet，但 SIR2/OpenRR 优势还保留一部分，选 `Fusion_a0p25`。
- 如果所有中间 alpha 都只是退化折中，没有新优势，就不要把 fusion 当最终主方法，只作为失败分析。

## 5. 需要下载/备份的文件

```bash
BACKUP=/mnt/data/ERRNet_fusion_sweep_$(date +%Y%m%d)
mkdir -p $BACKUP/results

cp -av results/fusion_rafa3k_no_old_standard $BACKUP/results/
cp -av results/fusion_rafa3k_no_old_zhang20_512 $BACKUP/results/
cp -av results/FUSION_SWEEP_SUMMARY.md $BACKUP/results/

du -sh $BACKUP
find $BACKUP -maxdepth 4 -type f | sort
```

你发给我时，至少需要：

- `results/FUSION_SWEEP_SUMMARY.md`
- `results/fusion_rafa3k_no_old_standard/summary.csv`
- `results/fusion_rafa3k_no_old_zhang20_512/summary.csv`
- `results/fusion_rafa3k_no_old_standard/features/`
- `results/fusion_rafa3k_no_old_zhang20_512/features/`

`features/*.csv` 用来分析：哪些样例适合 RAFA，哪些样例应该回退 ERRNet。后续如果要做真正的输入驱动 adaptive selector，就靠这些特征来定规则。
