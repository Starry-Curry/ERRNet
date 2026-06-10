# ERRNet / BP-RAP / RAFA / Fusion 项目实验总览

这个文件用于快速回答：本课程项目做了什么、方法如何迭代、当前最终方案是什么、实验结果能支持怎样的论文叙事。

如果只想写论文，优先读第 1、5、6、7、8、10 节。  
如果要追溯完整命令和历史实验，看 `EXPERIMENT_LOG_RAP_ERRNET.md`。  
如果要看最终论文结构，看 `PAPER_OUTLINE_FINAL.md`。

## 1. 当前最终结论

最终建议主线不是“单模型全面超过 ERRNet”，而是：

> ERRNet 在 CEILNet/Zhang20 这类强像素对齐 benchmark 上很强；RAFA 在 SIR2/OpenRR 真实场景上更强；最终使用 ERRNet-RAFA inference fusion 利用二者互补性，在保持课程数据集稳定性的同时显著提升真实场景表现和总体均值。

当前建议论文主方法：

| 层级 | 方法 | 是否使用 OpenRR train | 论文定位 |
| --- | --- | --- | --- |
| Baseline | ERRNet-Hyper | 否 | 课程指定强 baseline |
| Course-fair 改进 | BP-RAP RIC | 否 | 证明反射先验/残差细化对 SIR2 真实场景有效 |
| Extra-data 单模型 | RAFA3k w/o old | 是，OpenRR train 3k | 真实数据适配主模型 |
| 最终推理策略 | ERRNet-RAFA Fusion `alpha=0.50` | 使用 RAFA checkpoint，推理时融合 ERRNet 和 RAFA | 最终推荐方法 |
| 保守推理策略 | Fusion `alpha=0.25` | 同上 | 更重视 CEILNet/Zhang 稳定性时使用 |
| 实验性策略 | Adaptive fusion | 同上 | Six-set 最高，但 selector 对 Zhang/CEILNet 不够准，不建议作为主方法 |

最终推荐写法：

- **强结果**：Fusion `alpha=0.50` 的 Course 5-set、SIR2、OpenRR 和 Six-set mean 都优于 ERRNet。
- **诚实限制**：Fusion `alpha=0.50` 在 CEILNet/Zhang20 单项仍略低于 ERRNet，但大幅缓解 RAFA 单模型的退化。
- **方法贡献**：不是单纯调参，而是形成了“baseline-preserving residual refinement + real-data replay adaptation + inference fusion”的完整迭代路线。

## 2. 应读文件

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `EXPERIMENT_MATRIX_SUMMARY.md` | 当前中文总览 | 快速理解项目全貌 |
| `EXPERIMENT_LOG_RAP_ERRNET.md` | 实验流水账 | 查历史命令、checkpoint、阶段记录 |
| `PAPER_OUTLINE_FINAL.md` | 最终论文大纲 | 改写 LaTeX 的蓝图 |
| `results/FINAL_TABLES_FOR_PAPER.md` | 最终表格草稿 | 论文表格、caption 直接参考 |
| `results/FUSION_ROUTING_ANALYSIS.md` | Fusion 统计 | 写最终方法、win count、routing 分析 |
| `HARD_REFLECTION_TRAINING_RUNBOOK.md` | HardSynth 训练命令 | 写负结果/训练优化尝试 |
| `ADAPTIVE_FUSION_RUNBOOK.md` | Fusion sweep 命令 | 复现实验 |
| `BP-RAP-ERRNet_v1.3_design_and_execution_plan.md` | BP-RAP/RIC/FSS 设计 | 写方法细节 |
| `BP_RAP_V1_4_RAFA_ITERATION_PLAN.md` | RAFA 设计 | 写 extra-data 适配 |

## 3. 数据集与正式协议

### 3.1 训练数据

| 数据 | 数量 | 用途 | 是否课程公平 |
| --- | ---: | --- | --- |
| Pascal VOC cropped images | 7,643 | 物理合成反射训练 | 是 |
| Zhang real train | 89 | 真实训练 / replay | 是 |
| OpenRR train 1k | 1,000 | OpenRR-FT / Soup / RAFA1k | 否 |
| OpenRR train 3k | 3,000 | RAFA3k / Balanced / no-old / OpenRR-only | 否 |
| Self-collected | 至少 5 对，待补 | 课程要求自采测试 | 测试，不用于训练 |

### 3.2 正式测试协议

| 测试集 | 数量 | 正式协议 |
| --- | ---: | --- |
| CEILNet Table2 | 100 | native resolution |
| Zhang real20 | 20 | `--max_long_edge 512` |
| SIR2 Objects | 200 | native resolution |
| SIR2 Postcard | 179 | native resolution |
| SIR2 Wild | 101 | native resolution |
| OpenRR val | 300 | native resolution |
| Self-collected | >=5 | native，若 OOM 需说明 resize |

指标：

| 指标 | 越大/越小 | 说明 |
| --- | --- | --- |
| PSNR | 越大越好 | 像素误差 |
| SSIM | 越大越好 | 结构相似 |
| NCC | 越大越好 | 归一化相关性 |
| LMSE | 越小越好 | 局部 MSE |

不要把 all-512 diagnostic 数字放进主表。主表只用上述正式协议。

## 4. 方法迭代路线

### 4.1 ERRNet-Hyper baseline

课程指定 baseline 是 ERRNet。实际 checkpoint `checkpoints/errnet/errnet_060_00463920.pt` 是 hypercolumn 版本：

- 输入是 RGB + VGG hypercolumn 特征。
- 在 CEILNet 和 Zhang20 上非常强。
- 后续所有改进都必须尊重这个强 baseline。

论文口径：

> We use the course-provided ERRNet-Hyper checkpoint as a strong baseline rather than a weak reimplementation.

### 4.2 From-scratch RAP：失败但有启发

早期从零训练 RAP-ERRNet：

- VOC physics synthesis + Zhang real89。
- 模型含 prior、gate、refinement。
- 结果在 SIR2 Objects/Wild 有潜力，但 CEILNet/Zhang 远低于 ERRNet。

代表结果：

| 方法 | CEILNet | Zhang20 | Objects | Postcard | Wild |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAP from scratch | 19.1752 | 19.5196 | 25.7196 | 20.8931 | 25.5726 |

结论：不能从零训练替代 ERRNet-Hyper，必须 baseline-preserving。

### 4.3 RAP-Hyper / BP-RAP

核心思想：

> 以 ERRNet 输出为强初始解，只学习反射先验引导的小残差，不重写整图。

流程：

```text
Input I
  -> ERRNet-Hyper backbone 得到 T0
  -> Prior Head 得到 P
  -> Prior-gated Adapter 得到 Tg
  -> Residual Refinement 得到 Delta
  -> 输出 T = Tg + Delta
```

关键约束：

| 设计 | 作用 |
| --- | --- |
| zero-residual initialization | 初始不破坏 ERRNet 输出 |
| residual scale 0.1 | 限制修改幅度 |
| prior-gated residual | 只在反射区域强修正 |
| anchor loss | 低反射区域贴近 ERRNet |
| delta regularization | 防止全局漂移 |
| pseudo mask downweight | 对无 mask 样本降低伪 mask 权重 |

### 4.4 RIC / FSS

RIC：对同一 clean image 合成不同反射，约束输出一致，用于反射不变性训练。

FSS：低/高频监督，尝试分别处理 veil 和 ghost 反射。

最新消融显示：

- residual refinement 是最关键结构。
- prior/gate/RIC/FSS 数值提升不强，更多体现为可解释性和稳定性。
- 论文不能过度声称 RIC/FSS 是主要数值来源。

### 4.5 RAFA：真实数据回放适配

RAFA 用 OpenRR train 作为额外真实配对数据。核心不是只用 OpenRR fine-tune，而是：

| 设计 | 作用 |
| --- | --- |
| OpenRR supervision | 提升真实反射场景 |
| Course replay | 防止忘记 VOC/Zhang 课程分布 |
| Optional old distillation | 约束 replay 上接近旧模型，但最新结果显示不是必需 |
| Frozen backbone | 只调新模块，降低破坏 ERRNet 的风险 |

重要结论：

- `RAFA3k w/o old` 是当前最好的 extra-data 单模型。
- `OpenRR-only 3k` OpenRR 最高，但 Course 明显下降，说明 course replay 必要。
- `lambda_old=0.4` 和 `0.2` 并非最优，old distillation 会限制适配。

### 4.6 HardSynth：训练优化负结果

为了拉回 CEILNet/Zhang，我们尝试 strong reflection synthesis + hard anchor：

| 方法 | CEILNet | Zhang20 | SIR2 mean | OpenRR |
| --- | ---: | ---: | ---: | ---: |
| RAFA3k w/o old | 24.0420 | 21.0688 | 24.9772 | 28.0698 |
| HardSynth | 24.0700 | 21.0556 | 24.8881 | 28.1179 |

结论：

- CEILNet 只提升 `+0.028 dB`，Zhang 略降。
- SIR2 下降。
- 说明继续训练并不能自然解决 CEILNet/Zhang 与真实场景之间的 trade-off。

论文中可作为负结果：强反射合成不能充分弥合 benchmark gap，因此需要推理融合。

### 4.7 最终推理融合

最终推理：

```text
T_err  = ERRNet(I)
T_rafa = RAFA3k_no_old(I)
T_final = alpha * T_rafa + (1 - alpha) * T_err
```

主方法使用 `alpha=0.50`。

解释口径：

- ERRNet 擅长 CEILNet/Zhang 强像素对齐场景。
- RAFA 擅长 SIR2/OpenRR 真实场景。
- Fusion 利用互补性，避免 RAFA 单模型在 CEILNet/Zhang 上大幅崩，同时保留真实场景收益。

注意：

- `Fusion a=0.50` 不是硬路由，每张图都使用 50% ERRNet + 50% RAFA。
- Adaptive fusion 是连续权重，不是“多少张走 ERRNet/多少张走 RAFA”的硬分类。
- 当前 adaptive selector 对 Zhang/CEILNet 不够准，不建议作为主方法。

## 5. 课程公平结果

不使用 OpenRR train 的正式结果：

| Method | Course PSNR | SSIM | NCC | LMSE | SIR2 PSNR | SIR2 SSIM | SIR2 NCC | SIR2 LMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERRNet baseline | 24.5077 | 0.8861 | 0.9465 | 0.0081 | 23.8201 | 0.8871 | 0.9546 | 0.0052 |
| BP-RAP RIC | 23.8919 | 0.8839 | 0.9419 | 0.0086 | 24.8216 | 0.9088 | 0.9653 | 0.0034 |
| RIC+FSS | 23.8820 | 0.8838 | 0.9420 | 0.0086 | 24.8254 | 0.9087 | 0.9654 | 0.0034 |

解读：

- BP-RAP RIC 不全面超过 ERRNet。
- 主要价值在 SIR2 真实场景，SIR2 PSNR 提升约 `+1.00 dB`。
- CEILNet/Zhang 的下降说明 baseline-preserving 还不足以处理强 synthetic / pixel-aligned benchmark。

## 6. 结构消融

| Method | Prior | Gate | Refine | RIC | FSS | Course PSNR | SIR2 PSNR | OpenRR PSNR | Main reading |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BP-RAP RIC | Y | Y | Y | Y | N | 23.8919 | 24.8216 | 26.8597 | full course-fair model |
| w/o prior | N | Y | Y | Y | N | 23.9071 | 24.8273 | 26.8180 | prior 数值收益有限 |
| w/o gate | Y | N | Y | Y | N | 23.8790 | 24.8126 | 26.7596 | gate 收益有限 |
| w/o refinement | Y | Y | N | Y | N | 23.5954 | 24.5825 | 26.3458 | refinement 最关键 |
| w/o RIC | Y | Y | Y | N | N | 23.8964 | 24.8301 | 26.8846 | RIC 不是主要数值来源 |
| RIC+FSS | Y | Y | Y | Y | Y | 23.8820 | 24.8254 | 26.8909 | FSS 与 full 基本持平 |

论文写法：

- 强调 residual refinement 是有效结构。
- prior/gate/RIC 提供反射感知和可解释性，但不能写成大幅提升来源。

## 7. RAFA / OpenRR 适配结果

| Method | OpenRR train | Pairs | Replay | lambda_old | Balanced | Course PSNR | SIR2 PSNR | OpenRR PSNR | Six-set PSNR | Role |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| ERRNet | N | 0 | N | 0 | N | 24.5077 | 23.8201 | 25.4874 | 24.6709 | baseline |
| BP-RAP RIC | N | 0 | N | 0 | N | 23.8919 | 24.8216 | 26.8597 | 24.3865 | course-fair method |
| OpenRR-FT 1k | Y | 1000 | N | 0 | N | 23.7930 | 24.8573 | 28.9560 | 24.6535 | real-only adaptation |
| Soup a0.25 | indirect | 1000 | N/A | N/A | N | 23.9549 | 24.9246 | 27.4103 | 24.5308 | soup fallback |
| RAFA1k | Y | 1000 | Y | 0.2 | N | 23.9529 | 24.9043 | 27.7031 | 24.5779 | replay adaptation |
| RAFA3k | Y | 3000 | Y | 0.2 | N | 23.9906 | 24.9469 | 27.8483 | 24.6336 | extra-data baseline |
| Balanced RAFA3k | Y | 3000 | Y | 0.2 | Y | 23.9999 | 24.9680 | 27.8349 | 24.6390 | balanced sampling |
| Old04 | Y | 3000 | Y | 0.4 | N | 23.9663 | 24.9159 | 27.6352 | 24.5778 | stronger distillation diagnostic |
| RAFA3k w/o old | Y | 3000 | Y | 0.0 | N | 24.0085 | 24.9772 | 28.0698 | 24.6853 | best single RAFA model |
| OpenRR-only 3k | Y | 3000 | N | 0 | N | 23.5304 | 24.7896 | 29.2549 | 24.4845 | highest OpenRR, worse course |

主要结论：

- OpenRR-only 3k 说明只追 OpenRR 会损害课程集合。
- RAFA3k w/o old 是单模型最佳折中。
- 但 RAFA3k w/o old 在 CEILNet/Zhang20 上仍明显低于 ERRNet，因此需要 fusion。

## 8. 最终 Fusion 结果

### 8.1 均值

| Method | Course PSNR | d vs ERRNet | SIR2 PSNR | d vs ERRNet | OpenRR PSNR | d vs ERRNet | Six-set PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERRNet | 24.5077 | 0.0000 | 23.8201 | 0.0000 | 25.4874 | 0.0000 | 24.6709 |
| Fusion a=0.25 | 24.8235 | +0.3159 | 24.4009 | +0.5808 | 26.3980 | +0.9106 | 25.0859 |
| Fusion a=0.50 | 24.7864 | +0.2787 | 24.8360 | +1.0159 | 27.2208 | +1.7334 | 25.1921 |
| Fusion a=0.75 | 24.5022 | -0.0054 | 25.0495 | +1.2294 | 27.8463 | +2.3589 | 25.0596 |
| RAFA3k no-old | 24.0085 | -0.4992 | 24.9772 | +1.1571 | 28.0698 | +2.5824 | 24.6853 |
| Adaptive fusion | 24.6497 | +0.1420 | 25.0170 | +1.1970 | 27.9558 | +2.4684 | 25.2007 |

建议主推 `Fusion a=0.50`：

- Course mean 比 ERRNet `+0.2787 dB`。
- SIR2 比 ERRNet `+1.0159 dB`。
- OpenRR 比 ERRNet `+1.7334 dB`。
- Six-set 比 ERRNet `+0.5212 dB`。
- 比 RAFA 单模型更稳，避免 CEILNet/Zhang 大崩。

`Fusion a=0.25` 是保守方案：Course 最高，CEILNet/Zhang 几乎贴近 ERRNet，但 SIR2/OpenRR 收益较小。  
`Adaptive fusion` Six-set 略高，但 selector 对 CEILNet/Zhang 保护不如 constant `a=0.50`，不建议主推。

### 8.2 逐数据集相对 ERRNet

| Method | CEILNet | Zhang20 | Objects | Postcard | Wild | OpenRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fusion a=0.25 | -0.0805 | -0.0826 | +0.6394 | +0.5012 | +0.6018 | +0.9106 |
| Fusion a=0.50 | -0.9618 | -0.6923 | +1.1269 | +0.8489 | +1.0720 | +1.7334 |
| RAFA3k no-old | -3.5994 | -2.3679 | +1.3820 | +0.8870 | +1.2023 | +2.5824 |

关键论证：

- RAFA 在真实数据上强，但 CEILNet/Zhang 损失过大。
- Fusion a=0.50 把 CEILNet/Zhang 损失显著拉回，同时保留大部分 SIR2/OpenRR 提升。

### 8.3 胜出样本数

Fusion a=0.50 相对 ERRNet 的胜出比例：

| Dataset | Beat ERRNet | Beat RAFA | Mean d vs ERRNet | Mean d vs RAFA |
| --- | ---: | ---: | ---: | ---: |
| CEILNet | 40/100 (40.0%) | 98/100 (98.0%) | -0.9618 | +2.6375 |
| Zhang20 | 8/20 (40.0%) | 19/20 (95.0%) | -0.6923 | +1.6757 |
| SIR2 Objects | 200/200 (100.0%) | 56/200 (28.0%) | +1.1269 | -0.2551 |
| SIR2 Postcard | 160/179 (89.4%) | 95/179 (53.1%) | +0.8489 | -0.0382 |
| SIR2 Wild | 77/101 (76.2%) | 57/101 (56.4%) | +1.0720 | -0.1302 |
| OpenRR val | 271/300 (90.3%) | 102/300 (34.0%) | +1.7334 | -0.8490 |

这组数字是论文强结果的主要支撑：

- 在真实场景 SIR2/OpenRR 上，大多数样本比 ERRNet 好。
- 在 CEILNet/Zhang 上，大多数样本比 RAFA 好，说明 fusion 的保护作用成立。

## 9. 定性图和失败分析

已生成：

| 图 | 路径 | 用途 |
| --- | --- | --- |
| Fusion main qualitative | `paper/figures/fusion_qualitative_main.png/.pdf` | 正文主图，建议使用前三行 |
| Fusion failure cases | `paper/figures/fusion_failure_cases.png/.pdf` | 失败分析 |
| CEILNet/Zhang diagnostics | `paper/figures/ceilnet_*_diagnostics.png`, `zhang20_*_diagnostics.png` | 附录或分析 |
| Prior/error analysis | `paper/figures/prior_error_analysis.png` | 解释 prior 与误差 |

主图建议：

- SIR2 Wild success：Fusion 同时优于 ERRNet 和 RAFA，强正例。
- SIR2 Objects/Postcard median success：展示稳定提升。
- OpenRR success：展示 Fusion 大幅优于 ERRNet，但也说明 RAFA 单模型 OpenRR 更强。
- 自采 success：待补。

不要把 CEILNet protection 当主成功案例，因为该例 Fusion 虽然比 RAFA 好，但仍明显低于 ERRNet。应放在 protection/failure analysis。

## 10. 论文叙事口径

### 可以强写

- 复现了强 ERRNet-Hyper baseline。
- BP-RAP RIC 在 SIR2 真实场景上明显提升。
- RAFA3k w/o old 利用 OpenRR 真实配对数据显著提升真实域。
- Fusion a=0.50 在 Course mean、SIR2、OpenRR、Six-set mean 上均超过 ERRNet。
- Fusion a=0.50 在 SIR2/OpenRR 大多数样本上超过 ERRNet。
- Fusion 显著缓解 RAFA 在 CEILNet/Zhang 上的大幅退化。

### 不能过度写

- 不能说 BP-RAP RIC 全面超过 ERRNet。
- 不能说 RAFA 单模型是最终最鲁棒模型。
- 不能说 Fusion a=0.50 每个数据集都超过 ERRNet。
- 不能把 Adaptive fusion 写成最终主方法，除非进一步调 selector。
- 不能把 OpenRR 训练结果混入课程公平主表。

### 推荐最终摘要句

> Compared with ERRNet-Hyper, the final ERRNet-RAFA fusion improves Course 5-set PSNR by 0.28 dB, SIR2 mean PSNR by 1.02 dB, OpenRR val PSNR by 1.73 dB, and six-set mean PSNR by 0.52 dB. The gain is mainly from real-scene SIR2/OpenRR images, while the fusion also prevents the large CEILNet/Zhang degradation observed in the RAFA-only model.

## 11. 还缺什么

| 项 | 状态 | 下一步 |
| --- | --- | --- |
| Self-collected 5 paired scenes | 未完成 | 明天采集，按 `data/self_collected/test/scene_xxx` 放置 |
| Self metrics | 未完成 | 跑 ERRNet / RAFA / Fusion a=0.50 |
| Self qualitative | 未完成 | 生成单独 self 图 |
| LaTeX 正文 | 旧版 | 按 `PAPER_OUTLINE_FINAL.md` 重写 |
| PPT | 未开始 | 可复用论文图表 |

自采数据完成前，论文中对应表格必须标 `TODO` 或先不写最终数值。
