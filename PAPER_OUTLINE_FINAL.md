# 课程论文最终大纲

目标：把现有 LaTeX 从“项目代码文档式叙事”改成一篇常见论文节奏的实验报告。核心主线是：

> ERRNet 是强 baseline；BP-RAP 证明反射感知残差细化对真实场景有效；RAFA 利用 OpenRR 真实数据进一步提升真实域；最终通过 ERRNet-RAFA inference fusion 解决 RAFA 在 CEILNet/Zhang 上的退化，并取得更好的综合结果。

当前 `paper/neurips_2026.tex` 已更新为 fusion 主线。若后续重写摘要或标题，应保持最终方法宏为：

```latex
\newcommand{\finalmodel}{ERRNet--RAFA Fusion}
```

推荐最终标题：

```text
面向真实场景的单图像反射去除：
基于 ERRNet 的反射感知残差细化、真实数据适配与推理融合
```

## 摘要

叙事方向：

1. 单图像反射去除是不适定问题，真实反射包含 veil、ghost、错位、色调变化。
2. 课程 baseline ERRNet-Hyper 在 CEILNet/Zhang 上很强，但真实场景 SIR2/OpenRR 仍存在过暗、雾化、细节损失。
3. 我们提出三阶段改进：
   - BP-RAP：反射先验引导的 baseline-preserving residual refinement。
   - RAFA：OpenRR 真实数据 + course replay 的真实域适配。
   - Fusion：推理阶段融合 ERRNet 与 RAFA，利用二者互补性。
4. 给最终数字：
   - Fusion a=0.50 相对 ERRNet：Course +0.28 dB，SIR2 +1.02 dB，OpenRR +1.73 dB，Six-set +0.52 dB。
5. 明确限制：
   - CEILNet/Zhang 个别强像素对齐场景仍由 ERRNet 占优。

不要在摘要里写 “全面超过 ERRNet”。

## 1. Introduction

应包含课程要求中的“背景”和“任务概述以及难点”。

### 1.1 背景

内容：

- 玻璃、橱窗、车窗、展柜中的反射会影响图像恢复和视觉理解。
- 单图像反射去除目标是从混合图像恢复透射层。
- 可用简单模型引入：

```latex
I = T + R
```

然后说明真实情况更复杂：模糊、重影、曝光差、色偏、空间错位。

### 1.2 难点

重点写：

- 单图像输入缺少多视角/偏振线索，是病态分解。
- 真实反射与背景纹理相互混叠。
- 不同 benchmark 偏好不同：
  - CEILNet/Zhang 更关注像素对齐和强反射去除。
  - SIR2/OpenRR 更关注真实场景结构、亮度和局部自然性。

这为后文“为什么需要 fusion”埋伏笔。

### 1.3 本文贡献

建议列 4 点：

1. 复现并系统评估 ERRNet-Hyper 强 baseline。
2. 提出 BP-RAP，使用 reflection prior、prior-gated adapter、residual refinement 和 baseline-preserving loss。
3. 提出 RAFA，使用 OpenRR 真实配对数据和 course replay 做 extra-data adaptation。
4. 提出 ERRNet-RAFA inference fusion，在综合指标上超过 ERRNet，并分析成功/失败场景。

论证口径：

- 写“课程公平”和“额外数据”两个 setting。
- 强调所有 OpenRR-train 结果都单独报告，避免公平性争议。

## 2. Related Work

对应课程要求中的“总结过去、现有方法”。

### 2.1 Traditional reflection removal

简述：

- 基于梯度稀疏、边缘统计、层分解、多图像、偏振等方法。
- 优点：可解释。
- 缺点：复杂真实反射和单图像场景鲁棒性有限。

### 2.2 Deep single-image reflection removal

至少提：

- CEILNet：深度网络用于单图像反射去除/平滑。
- Zhang et al.：perceptual losses 和真实反射数据，本课程 Zhang train/test 来自这里。
- ERRNet：misaligned training data + network enhancements，是课程 baseline。

### 2.3 Real-world datasets and domain adaptation

提：

- SIR2：Objects/Postcard/Wild 真实 benchmark。
- OpenRR：额外真实 paired 数据。
- 说明本文把 OpenRR 作为 extra-data adaptation，不混入课程公平主结果。

## 3. Baseline And Proposed Method

这一节对应“方法介绍、Baseline 算法介绍、改进算法介绍”。建议不要拆成过多代码细节，重点讲算法逻辑。

### 3.1 ERRNet-Hyper baseline

讲：

- ERRNet 是课程 baseline。
- 使用课程提供的 hypercolumn checkpoint。
- 它在 CEILNet/Zhang 上非常强，是强 baseline。

数据可引用：

- ERRNet Course PSNR 24.5077。
- CEILNet 27.6414，Zhang20 23.4367。

论证口径：

- 我们不是拿弱 baseline 做对比。
- 后续方法都以这个强模型为基础。

### 3.2 BP-RAP: baseline-preserving reflection-aware residual refinement

讲方法流程：

```text
I -> ERRNet-Hyper -> T0
I -> Prior Head -> P
T0, P -> Prior-gated Adapter -> Tg
[I, Tg, I-Tg, P] -> Residual Refinement -> Delta
Output = Tg + Delta
```

写公式：

```latex
T_0 = G_{\mathrm{ERRNet}}(I), \quad
P = H_p(I)
```

```latex
\hat{T} = \mathrm{clip}(T_g + s \cdot P \odot \tanh(H_r([I,T_g,I-T_g,P])), 0, 1)
```

要讲清：

- 不是重写整图，而是受控残差修正。
- `residual_scale=0.1`。
- anchor/delta/clean consistency 避免破坏低反射区域。

### 3.3 RIC and FSS

讲：

- RIC：同一 clean image 不同反射合成，输出应一致。
- FSS：低频 veil 和高频 ghost 的频率监督探索。

口径：

- RIC/FSS 是合理训练约束，但消融显示不是主要数值来源。
- 不要过度吹。

### 3.4 RAFA: replay-anchored real-data adaptation

讲：

- 引入 OpenRR train 3k 真实配对数据。
- 只用 OpenRR 会导致课程分布退化。
- RAFA 使用 course replay，把 VOC/Zhang 与 OpenRR 混合采样。
- old distillation 做过，但最新 no-old 最好。

应写的关键实验发现：

- OpenRR-only 3k：OpenRR 29.2549 最高，但 Course 23.5304 低。
- RAFA3k w/o old：Course 24.0085，SIR2 24.9772，OpenRR 28.0698，是单模型最好折中。

论证口径：

- Course replay 必要。
- Old distillation 不是最终关键，甚至会限制适配。

### 3.5 Final inference fusion

这是最终方法重点。

公式：

```latex
\hat{T}_{fusion} = \alpha \hat{T}_{RAFA} + (1-\alpha)\hat{T}_{ERRNet}
```

主设置：

```text
alpha = 0.50
```

讲清楚：

- 不是训练新模型，不额外消耗显卡。
- 每张图都融合，不是硬路由。
- 选择 alpha=0.50 是因为它在 Course/SIR2/OpenRR/Six-set 上均优于 ERRNet，同时保护 CEILNet/Zhang。

不要把 adaptive fusion 写成最终主方法。可以作为分析项：

- Adaptive Six-set 25.2007 略高。
- 但 CEILNet/Zhang 保护差，Zhang 平均 alpha 0.7653，说明 selector 不够准。

## 4. Experiments

对应“实验介绍、实验设置、数据集介绍、实验设置介绍、指标介绍”。

### 4.1 Datasets

表格内容：

- Train:
  - Pascal VOC 7,643 crops。
  - Zhang train 89。
  - OpenRR train 1k/3k extra-data。
- Test:
  - CEILNet 100。
  - Zhang20 20，max long edge 512。
  - SIR2 Objects 200。
  - SIR2 Postcard 179。
  - SIR2 Wild 101。
  - OpenRR val 300。
  - Self-collected >=5，待补。

公平性口径：

- Course-fair tables 不使用 OpenRR train。
- Extra-data tables 明确标注 Uses OpenRR train。

### 4.2 Metrics

写：

- PSNR：像素保真。
- SSIM：结构相似。
- NCC：相关性。
- LMSE：局部缩放误差。

强调：

- PSNR 在 CEILNet/Zhang 上尤其敏感。
- 真实场景还需要看定性结果。

### 4.3 Implementation details

写：

- Backbone：ERRNet-Hyper。
- Optimizer：Adam。
- Fine-tune：frozen backbone，新模块学习率。
- BP-RAP RIC：VOC synthesis + Zhang real89。
- RAFA：OpenRR train 3k + course replay。
- Fusion：只在推理阶段做线性融合。

不要堆太多命令行，把命令放附录。

## 5. Quantitative Results

对应“实现方法在数据集上的各种指标结果&与baseline的定量对比”。

### 5.1 Course-fair comparison

使用 Table 2：

| Method | Course | SIR2 |
| --- | ---: | ---: |
| ERRNet | 24.5077 | 23.8201 |
| BP-RAP RIC | 23.8919 | 24.8216 |

论证：

- BP-RAP RIC 牺牲 CEILNet/Zhang。
- 但 SIR2 提升 +1.0015 dB 左右。
- 说明反射感知残差细化对真实场景有效。

### 5.2 Structure ablation

使用 Table 3。

重点：

- w/o refinement 下降明显：Course 23.5954，SIR2 24.5825。
- prior/gate/RIC/FSS 差异小。

结论：

- 主要数值收益来自 residual refinement 与 baseline-preserving 框架。
- prior/RIC/FSS 更像解释性和训练约束。

### 5.3 RAFA / OpenRR adaptation

使用 Table 4。

重点对比：

- RAFA3k w/o old 是单模型最好。
- OpenRR-only 3k 说明只追目标域会损害课程数据。
- HardSynth 是负结果，可放到分析或附录。

### 5.4 Final fusion results

核心表，建议放主文：

| Method | Course | SIR2 | OpenRR | Six-set |
| --- | ---: | ---: | ---: | ---: |
| ERRNet | 24.5077 | 23.8201 | 25.4874 | 24.6709 |
| RAFA3k no-old | 24.0085 | 24.9772 | 28.0698 | 24.6853 |
| Fusion a=0.50 | 24.7864 | 24.8360 | 27.2208 | 25.1921 |

论证：

- Fusion a=0.50 是最稳主方法。
- 相对 ERRNet：
  - Course +0.2787。
  - SIR2 +1.0159。
  - OpenRR +1.7334。
  - Six-set +0.5212。
- 相对 RAFA：
  - CEILNet/Zhang 大幅保护。
  - OpenRR/SIR2 有一定损失，但总体更稳。

### 5.5 Win-count analysis

用 Table 7 支撑“不是平均数偶然”。

重点写：

- Fusion a=0.50 在 SIR2 Objects 上 200/200 优于 ERRNet。
- Postcard 160/179，Wild 77/101，OpenRR 271/300。
- CEILNet/Zhang 虽未超过 ERRNet，但 98/100 和 19/20 优于 RAFA。

这说明：

- 在真实场景，Fusion 大多数样本确实好于 ERRNet。
- 在强 benchmark，Fusion 保护了 RAFA 的退化。

## 6. Qualitative Results And Analysis

对应“样例介绍”和定性分析。

### 6.1 Main qualitative figure

图：`paper/figures/fusion_qualitative_main.png`

建议正文只展示前三行加后续 self 行：

- SIR2 Wild：Fusion 同时超过 ERRNet 和 RAFA，强正例。
- SIR2 Objects/Postcard：中等稳定提升。
- OpenRR：Fusion 明显优于 ERRNet，但弱于 RAFA，体现折中。
- Self-collected：待补。

列：

```text
Input | ERRNet | RAFA | Fusion a=0.50 | GT
```

### 6.2 Failure / protection cases

图：`paper/figures/fusion_failure_cases.png`

写：

- CEILNet airplane：ERRNet 去除强低频反射更彻底。
- Zhang bicycle：ERRNet 更接近全局暗色 GT。
- Fusion 比 RAFA 好，但仍不能完全达到 ERRNet。

结论：

- CEILNet/Zhang 的失败来自强低频反射和全局色调校正。
- 这解释为什么 RAFA 单模型会在这两个数据集大幅下降。

### 6.3 Adaptive fusion analysis

可以放分析或附录。

关键数据：

- Zhang20 中 adaptive 平均 alpha=0.7653，80% RAFA-leaning。
- 但 RAFA 只在 5/20 Zhang20 样本上赢 ERRNet。
- 说明当前 selector 错把不少 Zhang 样本交给 RAFA。

结论：

- Adaptive 有潜力，但当前规则不够可靠。
- 最终选 constant alpha=0.50 更稳。

## 7. Self-Collected Evaluation

课程硬性要求，必须补。

数据格式：

```text
data/self_collected/test/scene_001/blended.png
data/self_collected/test/scene_001/transmission.png
...
```

至少跑：

- ERRNet。
- RAFA3k no-old。
- Fusion a=0.50。

表格：

| Method | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| ERRNet | TODO | TODO | TODO | TODO |
| RAFA3k no-old | TODO | TODO | TODO | TODO |
| Fusion a=0.50 | TODO | TODO | TODO | TODO |

还要放 1 行定性图：

```text
Input | ERRNet | RAFA | Fusion | GT
```

如果自采标注不是严格配准，需要在文本中说明 PSNR/SSIM 可能受拍摄误差影响，定性结果同样重要。

## 8. Discussion

建议讨论三点：

### 8.1 Why BP-RAP helps SIR2

- ERRNet 有时过暗、过平滑、反射区域雾化。
- BP-RAP/RAFA 的残差细化能保留更多真实结构。

### 8.2 Why CEILNet/Zhang remain difficult

- CEILNet/Zhang 需要强力去除低频反射和全局色调校正。
- BP-RAP/RAFA 为了保护真实场景细节，修改更保守。
- PSNR 强惩罚全局色调和大面积残留。

### 8.3 Why fusion works

- ERRNet 提供像素保真和强 benchmark 稳定性。
- RAFA 提供真实场景反射适配。
- 线性 fusion 用低成本得到更稳综合表现。

## 9. Conclusion And Reflection

对应课程要求“结论与感悟”。

内容：

- 总结最终方法和结果。
- 说明从“直接训练新模型”到“尊重强 baseline、做受控残差、做真实域适配、做推理融合”的迭代过程。
- 反思：
  - 图像复原中平均指标不能完全代表真实效果。
  - 不同 benchmark 偏好不同，算法需要明确适用域。
  - 强 baseline 上改进要避免破坏已有能力。

可以写得稍微课程化：

> 本项目的主要收获不是简单堆网络，而是通过实验认识到 benchmark trade-off，并设计低成本方法利用不同模型的互补性。

## Appendix

建议包含：

### A. Training commands

放关键命令：

- BP-RAP RIC。
- RAFA3k no-old。
- Fusion sweep。
- Self evaluation。

### B. Full ablation tables

放结构消融、RAFA 全表、HardSynth 负结果。

### C. More qualitative results

放 CEILNet/Zhang diagnostics：

- worst。
- similar。
- better。

### D. Contributions

课程要求说明每个人贡献。若单人：

```text
本项目由本人独立完成，包括文献阅读、baseline 复现、方法设计、代码实现、实验评估、可视化分析和论文撰写。
```

多人则按成员写分工。

## 推荐正文表图顺序

1. Figure 1: Method overview。
2. Table 1: Dataset/protocol。
3. Table 2: Course-fair results。
4. Table 3: Final fusion results。
5. Figure 2: Fusion per-dataset delta / OpenRR adaptation plot。
6. Table 4: Structure ablation。
7. Figure 3: Main qualitative examples。
8. Figure 4: Failure/protection cases。
9. Table 5: Self-collected results。

## 当前 LaTeX 需要改的重点

`paper/neurips_2026.tex` 当前主要问题：

- 摘要还在讲 `Strength-balanced RAFA-OpenRR3k`，已经过时。
- 方法部分没有把 Fusion 作为最终方法。
- 表格还没有最终 Fusion a=0.50。
- 定性图还不是最终 `fusion_qualitative_main`。
- 自采数据仍是 TODO。

下一步应按本大纲重写 LaTeX，而不是在旧文上小修小补。
