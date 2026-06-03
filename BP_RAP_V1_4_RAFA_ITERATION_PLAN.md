# BP-RAP-ERRNet v1.4 RAFA 训练迭代方案

Last updated: 2026-06-04

## 1. 目标

v1.4 的目标不是推翻当前 BP-RAP RIC 主线，而是解决 OpenRR-FT 暴露出的核心问题：

```text
OpenRR-only fine-tuning improves OpenRR val strongly, but causes CEILNet/SIR2 Wild drift.
```

因此新增：

```text
RAFA = Replay-Anchored Fine-tuning for Real Reflection Adaptation
```

中文表述：

```text
基于课程分布回放与旧模型蒸馏约束的真实反射数据适配
```

RAFA 的定位：

- `BP-RAP RIC` 仍是 course-fair main method。
- `OpenRR-Soup alpha=0.25` 仍是当前 extra-data fallback。
- `RAFA` 是下一阶段冲 A 的中等风险增强：尝试把 OpenRR 真实数据适配从“事后 soup”
  推进到“训练时防漂移”。

## 2. 已新增的代码能力

### 2.1 Replay sampler

`train_rap_errnet.py` 新增可选参数：

```text
--use_course_replay
--openrr_ratio
--course_replay_ratio
--samples_per_epoch
```

当训练集是 `VOC + Zhang + OpenRR` 的 `ConcatDataset` 时，脚本会用
`WeightedRandomSampler` 近似控制采样比例。例如：

```text
openrr_ratio=0.5
course_replay_ratio=0.5
samples_per_epoch=1200
```

表示每个 epoch 抽样 1200 个样本，其中约 50% 来自 OpenRR，50% 来自课程 replay
数据（VOC synthesis + Zhang real89）。

### 2.2 Old-model distillation

新增参数：

```text
--teacher_ckpt
--lambda_old
--old_loss_prob
```

当 `lambda_old > 0` 时，训练脚本加载 teacher model。默认 teacher 是已有
BP-RAP RIC checkpoint：

```text
checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt
```

对 batch 中的课程 replay 样本（`voc`, `zhang_train`）计算：

```text
L_old = mean(abs(student_output - teacher_output))
```

OpenRR 样本不计算 `L_old`，这样可以让 OpenRR 真实 paired supervision 推动目标域
适配，同时用课程 replay 样本防止旧分布遗忘。

### 2.3 RAFA config

新增配置：

```text
configs/bp_rap_rafa_openrr1k.yaml
```

默认设置：

```text
OpenRR train pairs: 1000
OpenRR/course sampling ratio: 0.5 / 0.5
samples_per_epoch: 1200
epochs: 10
batch_size: 24
freeze backbone: true
new-module lr: 2e-5
lambda_old: 0.2
lambda_anchor: 0.05
lambda_delta: 0.01
lambda_ric: 0.02
```

### 2.4 Reflection-strength analysis

新增脚本：

```text
tools/analyze_reflection_strength.py
```

它统计 paired samples 的：

```text
strength = mean(abs(input - target))
hf_ratio = mean(abs(highpass(input - target))) / (strength + eps)
```

并按 dataset median 分成：

```text
weak_veil
strong_veil
weak_ghost
strong_ghost
```

这一步当前用于数据分析和报告，不直接改变训练采样。

### 2.5 Module-only soup

已有 `tools/soup_checkpoints.py` 支持：

```text
--module_filter prior_head,gated_adapter,refinement
```

这会只平均新增反射感知模块，保留 `ckpt_a` 的 backbone。它比 full soup 更符合
“不动 ERRNet 强基线主干，只融合反射残差分支”的叙事。

## 3. 阿里云服务器执行流程

以下命令均假设工作目录是：

```bash
cd /mnt/workspace/ERRNet
```

先更新代码：

```bash
git pull origin dip26
```

检查 OpenRR 数据：

```bash
python - <<'PY'
from datasets.unified_reflection_dataset import list_available_datasets
print(list_available_datasets("./data"))
PY

find -L data/openrr5k/train/blended -type f | wc -l
find -L data/openrr5k/train/transmission -type f | wc -l
```

预期：

```text
openrr_train=True
5000
5000
```

## 4. 实验 A：Reflection-strength 数据分析

先做 OpenRR/Zhang 的反射强度统计：

```bash
python tools/analyze_reflection_strength.py \
  --data_root ./data \
  --datasets openrr_train,zhang_train \
  --max_pairs 1000 \
  --max_long_edge 512 \
  --out results/reflection_strength_openrr1k_zhang.csv
```

查看每类数量：

```bash
python - <<'PY'
import csv
from collections import Counter

path = "results/reflection_strength_openrr1k_zhang.csv"
cnt = Counter()
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cnt[(row["dataset"], row["bin"])] += 1
for key, value in sorted(cnt.items()):
    print(key, value)
PY
```

报告用途：

- 说明 OpenRR 真实反射强度/频率类型更丰富。
- 解释为什么直接 OpenRR-FT 有收益但会漂移。
- 为 future work 的 balanced reflection sampling 做铺垫。

## 5. 实验 B：RAFA-OpenRR1k 训练

主 RAFA 实验：

```bash
RUN=bp_rap_rafa_openrr1k_e10

python train_rap_errnet.py \
  --config configs/bp_rap_rafa_openrr1k.yaml \
  --name $RUN \
  --resume checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt \
  --resume_model_only \
  --teacher_ckpt checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt \
  --data_root ./data \
  --use_physics_synthesis \
  --use_openrr \
  --max_openrr_pairs 1000 \
  --use_course_replay \
  --openrr_ratio 0.5 \
  --course_replay_ratio 0.5 \
  --samples_per_epoch 1200 \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --batch_size 24 \
  --epochs 10 \
  --lr 2.0e-5 \
  --new_lr 2.0e-5 \
  --backbone_lr 0 \
  --lambda_old 0.2 \
  --num_workers 8 \
  --device auto \
  --log_interval 20
```

日志中应看到：

```text
[i] replay sampler enabled ...
[i] loaded replay teacher ...
old=...
ric=...
```

如果显存不足，优先改：

```bash
--batch_size 16
```

如果仍然不足，降低 teacher distillation 频率：

```bash
--old_loss_prob 0.5
```

## 6. 实验 C：RAFA 快速诊断评估

先跑全 512 快速诊断，便于和 OpenRR-FT/Soup 诊断表比较：

```bash
RUN=bp_rap_rafa_openrr1k_e10

python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr1k.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --data_root ./data \
  --save_dir results/${RUN}_best_512 \
  --datasets ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --max_long_edge 512 \
  --device auto
```

汇总：

```bash
python tools/summarize_results.py \
  --method RAFA=results/${RUN}_best_512 \
  --datasets ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --out results/${RUN}_512_summary.md

cat results/${RUN}_512_summary.md
```

判断标准：

- OpenRR val 应高于 BP-RAP RIC zero-shot。
- CEILNet 和 SIR2 Wild 不应像 OpenRR-FT 那样明显下降。
- 如果 RAFA 接近或超过 OpenRR-Soup alpha=0.25，同时课程集不漂移，则 RAFA 可作为新的
  extra-data 主结果。

## 7. 实验 D：RAFA 正式评估

如果 512 诊断可接受，再跑正式协议：

CEILNet/SIR2/OpenRR 原分辨率：

```bash
RUN=bp_rap_rafa_openrr1k_e10

python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr1k.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --data_root ./data \
  --save_dir results/${RUN}_standard \
  --datasets ceilnet,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --device auto
```

Zhang20 单独 512：

```bash
python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr1k.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --data_root ./data \
  --save_dir results/${RUN}_zhang20_512 \
  --datasets zhang20 \
  --max_long_edge 512 \
  --device auto
```

生成正式汇总：

```bash
python tools/summarize_results.py \
  --method RAFA=results/${RUN}_standard,results/${RUN}_zhang20_512 \
  --datasets ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --out results/${RUN}_standard_summary.md

cat results/${RUN}_standard_summary.md
```

## 8. 实验 E：Module-only Soup

如果 RAFA 表现不错，做 RAFA 与 BP-RAP RIC 的 module-only soup：

```bash
BASE=bp_rap_hyper_ric_ft_from_hyper_ppu_bs24
RAFA=bp_rap_rafa_openrr1k_e10

for A in 0.25 0.50 0.75; do
  TAG=${A/./p}
  python tools/soup_checkpoints.py \
    --ckpt_a checkpoints/$BASE/best.pt \
    --ckpt_b checkpoints/$RAFA/best.pt \
    --alpha $A \
    --module_filter prior_head,gated_adapter,refinement \
    --out checkpoints/bp_rap_rafa_module_soup_a${TAG}/best.pt
done
```

快速诊断：

```bash
for A in 0.25 0.50 0.75; do
  TAG=${A/./p}
  python eval_all.py \
    --model rap_errnet \
    --config configs/bp_rap_rafa_openrr1k.yaml \
    --ckpt checkpoints/bp_rap_rafa_module_soup_a${TAG}/best.pt \
    --data_root ./data \
    --save_dir results/bp_rap_rafa_module_soup_a${TAG}_512 \
    --datasets ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
    --max_long_edge 512 \
    --device auto
done
```

汇总：

```bash
python tools/summarize_results.py \
  --method RAFA=results/bp_rap_rafa_openrr1k_e10_best_512 \
  --method ModSoup_a0p25=results/bp_rap_rafa_module_soup_a0p25_512 \
  --method ModSoup_a0p50=results/bp_rap_rafa_module_soup_a0p50_512 \
  --method ModSoup_a0p75=results/bp_rap_rafa_module_soup_a0p75_512 \
  --datasets ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --out results/bp_rap_rafa_module_soup_512_summary.md

cat results/bp_rap_rafa_module_soup_512_summary.md
```

## 9. 实验 F：可视化

对最终候选模型保存可视化。以 RAFA 为例：

```bash
RUN=bp_rap_rafa_openrr1k_e10

python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr1k.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --baseline_ckpt checkpoints/errnet/errnet_060_00463920.pt \
  --baseline_hyper \
  --data_root ./data \
  --save_dir results/${RUN}_standard_vis \
  --datasets ceilnet,sir2_objects,sir2_postcard,sir2_wild,openrr_val \
  --save_images \
  --device auto

python eval_all.py \
  --model rap_errnet \
  --config configs/bp_rap_rafa_openrr1k.yaml \
  --ckpt checkpoints/$RUN/best.pt \
  --baseline_ckpt checkpoints/errnet/errnet_060_00463920.pt \
  --baseline_hyper \
  --data_root ./data \
  --save_dir results/${RUN}_zhang20_512_vis \
  --datasets zhang20 \
  --max_long_edge 512 \
  --save_images \
  --device auto
```

输出列：

```text
Input | Baseline | Output | GT | Error Map | Prior Map
```

## 10. 何时采用 RAFA

采用 RAFA 的条件：

```text
OpenRR val >= OpenRR-Soup alpha=0.25 或接近
SIR2 mean 不低于 BP-RAP RIC 太多
CEILNet/SIR2 Wild 不出现 OpenRR-FT 那种明显漂移
self-collected visually better or comparable
```

如果 RAFA 不如 OpenRR-Soup：

```text
仍把 RAFA 作为 v1.4 negative/diagnostic result 报告。
最终 extra-data candidate 保持 OpenRR-Soup alpha=0.25。
```

## 11. 推荐时间安排

| 时间 | 工作 |
| --- | --- |
| Day 1 | 拉代码，跑 reflection-strength 分析和 RAFA-1k。 |
| Day 2 | 跑 RAFA 512 诊断；若可接受，跑正式评估。 |
| Day 3 | 做 module-only soup；与 full soup/RAFA 对比。 |
| Day 4-5 | 采集 self-collected paired data。 |
| Day 6 | 评估 self-collected，生成可视化。 |
| Day 7+ | 写论文和 PPT，整理成功/失败案例。 |

## 12. 最终报告推荐表述

如果 RAFA 成功：

```text
We further introduce replay-anchored real-data adaptation (RAFA), which combines
OpenRR supervision with course-distribution replay and old-model distillation.
Compared with direct OpenRR fine-tuning, RAFA reduces source-domain drift while
retaining real-scene adaptation gains.
```

如果 RAFA 不成功：

```text
RAFA is reported as a diagnostic continual-adaptation attempt. The result shows
that preventing drift during real-data adaptation remains non-trivial; therefore,
we select OpenRR-Soup alpha=0.25 as a more stable extra-data solution.
```
