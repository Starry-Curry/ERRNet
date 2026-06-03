# BP-RAP-ERRNet 阶段性实验报告

Last updated: 2026-06-04

## 摘要

本项目面向单图反射去除任务，在课程提供的 ERRNet-Hyper 强基线之上，提出
BP-RAP-ERRNet（Baseline-Preserving Reflection-Aware Proximal Refinement）。
方法不重建一个新的大规模 SOTA 网络，而是将 ERRNet 输出视为初始解，通过反射
概率先验、先验门控残差适配、轻量 refinement、zero-residual 初始化和
baseline-preserving 约束，在高反射区域学习受控残差修正，并在低反射区域尽量
保持原 ERRNet 输出。实验表明，本方法不能全面超过 ERRNet，尤其在 CEILNet 和
Zhang real20 等强像素对齐基准上仍弱于 baseline；但在 SIR2 三个真实场景子集
和 OpenRR 外部真实数据集上取得稳定提升。进一步的 OpenRR 真实数据微调和
checkpoint soup 显示，真实数据适配可以提升外部泛化，但需要通过 soup 控制对
课程数据集的漂移。当前最合理的最终表述是：BP-RAP-ERRNet 是一种针对强基线的
轻量、可解释、真实场景友好的增强方法，而不是全面 SOTA 替代方案。

## 1. 任务与目标

单图反射去除（Single Image Reflection Removal, SIRR）的目标是从含反射图像
`I` 中恢复透射层 `T`。该任务本质上是 ill-posed，真实 paired label 难以获得，
且不同测试集的反射形态和评价偏好差异明显。

本课程项目的实际目标包括：

- 复现课程 ERRNet-Hyper baseline。
- 在不破坏原始 baseline 代码路径的前提下提出增强方法。
- 在课程测试集上进行公平比较。
- 通过外部真实数据集 OpenRR 检验真实场景泛化。
- 给出失败分析、消融和可视化，而不是只报告单一指标。

## 2. 方法概述

### 2.1 总体思想

ERRNet-Hyper 作为强 baseline，先给出 coarse transmission：

```text
T0 = ERRNet_Hyper(I)
```

BP-RAP-ERRNet 不直接重新预测整张图，而是在 `T0` 上做受控残差修正：

```text
P = PriorHead(I)
Tg = GatedAdapter(T0, P)
R0 = I - Tg
Delta = residual_scale * tanh(Refine([I, Tg, R0, P])) * P
Tout = clamp(Tg + Delta, 0, 1)
```

其中：

- `P` 是一通道 reflection prior map。
- `GatedAdapter` 用 prior 调节 ERRNet 输出。
- `Refine` 是轻量残差 refinement head。
- `P * Delta` 让强修正集中在高 prior 区域。
- `residual_scale=0.1` 显式限制最大修正幅度。

### 2.2 模块设计

| 模块 | 作用 |
| --- | --- |
| ERRNet-Hyper backbone | 继承课程 pretrained baseline 的强恢复能力。 |
| ReflectionPriorHead | 预测反射概率图，提供位置先验。 |
| Prior-gated adapter | 在 coarse output 上进行轻量先验门控适配。 |
| LightweightRefinement | 从 `I, Tg, I-Tg, P` 预测局部 RGB 残差。 |
| Zero-residual initialization | 新增残差分支初始输出为 0，避免训练初期破坏 baseline。 |

### 2.3 损失函数

主损失由像素、梯度、SSIM、mask、clean consistency、anchor 和 residual 正则组成：

```text
L = L_pix + lambda_grad L_grad + lambda_ssim L_ssim
  + lambda_mask L_mask + lambda_clean L_clean
  + lambda_anchor L_anchor + lambda_delta L_delta
```

关键约束包括：

- Reflection-weighted L1：在高反射 mask 区域加权重建。
- Pseudo-mask downweight：真实 paired data 没有可靠 mask，`abs(I-T)` 伪 mask
  只低权重参与监督。
- Baseline anchor：在低 prior 区域约束输出接近 ERRNet 输出。
- Delta regularization：抑制不必要的全局改动。

### 2.4 可选 v1.3 约束

Reflection-Invariant Consistency（RIC）：

```text
同一 clean image T 合成两个不同 reflection observation Ia, Ib
要求 G(Ia) 和 G(Ib) 尽量一致
```

Frequency-Selective Supervision（FSS）：

```text
Gaussian low-pass 约束 veil reflection
Laplacian high-pass 约束 ghost/edge reflection
```

实验表明，RIC-only 是当前最优课程公平版本；RIC+FSS 稳定但没有明显超过 RIC。

## 3. 数据与评价协议

### 3.1 训练数据

课程公平主训练：

```text
VOC physics synthesis + Zhang/Berkeley real89 train
```

额外数据实验：

```text
OpenRR train 1k + Zhang real89
```

OpenRR 只作为 extra-data fine-tuning 单独报告，不并入课程公平主结果。

### 3.2 测试数据

| 数据集 | 用途 |
| --- | --- |
| CEILNet Table2 | synthetic paired benchmark，强像素对齐。 |
| Zhang real20 | 真实 paired benchmark，按课程协议 longest edge 512。 |
| SIR2 Objects/Postcard/Wild | 真实场景 benchmark。 |
| OpenRR val | 外部真实反射 benchmark。 |
| Self-collected paired data | 待补充，课程项目要求。 |

### 3.3 指标

使用 PSNR、SSIM、NCC、LMSE。正式协议中 CEILNet/SIR2/OpenRR 使用原分辨率，
Zhang real20 使用 `--max_long_edge 512`。

## 4. 实验迭代

| 阶段 | 结论 |
| --- | --- |
| RAP from scratch | 能学习真实反射修复，但 CEILNet/Zhang 很弱，不能作为主结果。 |
| RAP-Hyper | 引入 ERRNet-Hyper pretrained backbone，SIR2 提升明显。 |
| Hyper-ZeroRes-Staged | 更保守，减小漂移，但 SIR2 收益减弱。 |
| BP-RAP RIC | 当前课程公平主方法，SIR2 和 OpenRR zero-shot 最有价值。 |
| BP-RAP RIC+FSS | 稳定但未明显超过 RIC-only，作为 ablation。 |
| Post-hoc calibration | residual scale / prior gate / lowfreq anchor 均未优于原输出。 |
| OpenRR-FT | OpenRR val 大幅提升，但 CEILNet/SIR2 Wild 出现漂移。 |
| OpenRR-Soup alpha=0.25 | 当前最佳 extra-data 候选，平衡 OpenRR 增益和课程集稳定性。 |

## 5. 课程公平结果

### 5.1 与 ERRNet baseline 对比

| Dataset | ERRNet PSNR | BP-RAP RIC PSNR | Delta |
| --- | ---: | ---: | ---: |
| CEILNet Table2 | 27.6414 | 23.9710 | -3.6704 |
| Zhang real20, 512 | 23.4367 | 21.0237 | -2.4130 |
| SIR2 Objects | 24.6983 | 25.8905 | +1.1922 |
| SIR2 Postcard | 21.8856 | 22.4479 | +0.5623 |
| SIR2 Wild | 24.8763 | 26.1264 | +1.2501 |

SIR2 aggregate：

| Method | SIR2 mean PSNR | SIR2 mean SSIM | SIR2 mean NCC | SIR2 mean LMSE |
| --- | ---: | ---: | ---: | ---: |
| ERRNet baseline | 23.8201 | 0.8871 | 0.9546 | 0.0052 |
| BP-RAP RIC | 24.8216 | 0.9088 | 0.9653 | 0.0034 |

### 5.2 解释

ERRNet 在 CEILNet 和 Zhang real20 上仍然更强，说明原 pretrained baseline 对强
像素对齐 benchmark 的 fidelity 很好。BP-RAP RIC 在这两项上下降，主要来自残差
分支对颜色、亮度或局部纹理的轻微修正。但在 SIR2 三个真实场景子集上，BP-RAP
RIC 全面提升 PSNR、SSIM、NCC 和 LMSE，说明 prior-guided residual 更适合真实
复杂反射的局部修复。

因此，课程公平主结果应表述为：

```text
BP-RAP RIC sacrifices CEILNet/Zhang pixel fidelity but improves real-scene SIR2
quality. It is not a universal replacement for ERRNet, but a real-scene-oriented
baseline-preserving enhancement.
```

## 6. OpenRR 外部验证与额外数据结果

### 6.1 OpenRR zero-shot

| Method | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| ERRNet baseline | 25.4874 | 0.9480 | 0.9641 | 0.0030 |
| BP-RAP RIC | 26.8597 | 0.9600 | 0.9693 | 0.0018 |

BP-RAP RIC 在未使用 OpenRR train 的情况下提升 `+1.3723 dB`，说明 SIR2 上的提升
并非只针对课程数据集。

### 6.2 OpenRR-Soup alpha=0.25

OpenRR-FT 直接微调虽然让 OpenRR val 大涨，但带来 CEILNet/SIR2 Wild 漂移。
因此使用 checkpoint soup：

```text
theta = 0.75 * BP-RAP_RIC + 0.25 * OpenRR_FT
```

正式结果：

| Dataset | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| CEILNet Table2 | 23.9450 | 0.9064 | 0.9531 | 0.0071 |
| Zhang real20, 512 | 21.0559 | 0.7857 | 0.8601 | 0.0256 |
| SIR2 Objects | 26.0059 | 0.9129 | 0.9861 | 0.0026 |
| SIR2 Postcard | 22.6883 | 0.8961 | 0.9531 | 0.0036 |
| SIR2 Wild | 26.0795 | 0.9199 | 0.9578 | 0.0040 |
| OpenRR val | 27.4103 | 0.9614 | 0.9704 | 0.0017 |
| Six-set mean | 24.5308 | 0.8971 | 0.9468 | 0.0074 |

OpenRR-Soup alpha=0.25 是当前最佳 extra-data candidate。它不替代课程公平主
方法，但适合作为额外真实数据适配结果。

## 7. 与 SOTA 的关系

本项目不应声称全面 SOTA。近期 SIRR 方法包括 Location-aware reflection removal、
RDNet、PromptRR、Dereflection Any Image、OpenRR/RRW-style 数据驱动方法等，它们
往往使用更复杂网络、更大真实数据或 diffusion/generative priors。相比之下，本项目
的贡献在于：

- 在课程 ERRNet-Hyper 强 baseline 上做轻量增强。
- 明确提出 baseline-preserving residual refinement。
- 在真实场景 SIR2 和 OpenRR 上取得稳定提升。
- 提供 prior map、error map 和 residual 的可解释可视化。
- 给出正结果、负结果和数据适配分析。

合理定位是：

```text
Not a new global SOTA, but a reproducible and interpretable real-scene enhancement
of a strong ERRNet baseline.
```

## 8. 局限性

1. CEILNet 和 Zhang real20 明显低于 ERRNet baseline。
2. Prior map 可能学习到边缘/显著性，而不一定是真 reflection location。
3. OpenRR-FT 存在目标域过拟合，需要 soup 才能平衡。
4. FSS 当前没有明显收益。
5. 自采 paired data 还未完成。
6. 还没有完整跑近期 SOTA 方法，因此不能做严格 SOTA 排名。

## 9. 现阶段结论

当前最合理的最终方法选择：

```text
Course-fair main method: BP-RAP RIC
Extra-data method: BP-RAP RIC + OpenRR-Soup alpha=0.25
```

报告主 claim：

```text
BP-RAP-ERRNet formulates ERRNet enhancement as baseline-preserving
reflection-aware residual refinement. It does not universally outperform ERRNet,
but it improves real-scene reflection removal on SIR2 and OpenRR while keeping
the method lightweight and interpretable.
```

## 10. 后续两周计划

### 第 1 阶段：定性分析与自采数据

必须完成：

1. 采集至少 5 组 paired self-collected scenes。
2. 评估 ERRNet、BP-RAP RIC、OpenRR-Soup。
3. 对每个数据集生成 best / median / worst 可视化。
4. 检查 prior map 是否真的落在反射区域。

推荐可视化组合：

```text
Input | ERRNet Baseline | BP-RAP/Soup Output | GT | Error Map | Prior Map
```

### 第 2 阶段：补强消融

优先级从高到低：

1. 整理已有 RIC / RIC+FSS / staged / OpenRR-FT / Soup 对比。
2. 若时间允许，补 `w/o prior`、`w/o refinement`、`w/o anchor` 短跑。
3. 不建议再做新的大规模主训练。

### 第 3 阶段：低风险优化

可以尝试但不阻塞最终报告：

| 方向 | 建议 |
| --- | --- |
| 推理优化 | TTA / multi-scale inference，小成本试验。 |
| 微调优化 | OpenRR + VOC synthesis 混合短训，让 RIC 真正参与，但要防止漂移。 |
| Soup 优化 | module-only soup，只平均 prior/refinement。 |
| 选择策略 | 用 CEILNet/SIR2 Wild 不下降作为约束，OpenRR 作为优化目标。 |

### 第 4 阶段：论文与答辩材料

最终报告建议三张主表：

1. Course-fair comparison：ERRNet vs BP-RAP RIC。
2. OpenRR external validation：ERRNet vs BP-RAP RIC vs Soup。
3. Ablation / extra-data table：RIC、RIC+FSS、OpenRR-FT、Soup。

答辩重点：

- 为什么 CEILNet/Zhang 会掉。
- 为什么 SIR2/OpenRR 会涨。
- 为什么 OpenRR-FT 需要 soup。
- Prior map 是否可解释。
- 失败案例是什么。

## 11. 下一步操作清单

立即执行：

1. 生成 OpenRR-Soup 可视化并挑 best / median / worst。
2. 采集 self-collected paired data。
3. 跑 self-collected evaluation。
4. 将可视化结果整理为论文图。

暂不建议：

1. 继续长时间 OpenRR-only 训练。
2. 重构成 RDNet/diffusion 类大模型。
3. 追求全面 SOTA claim。

可以作为 future work：

1. 更可靠的 reflection location supervision。
2. 更真实的 synthesis distribution。
3. Adapter-only 或 module-only fine-tuning。
4. Generative prior / frequency prompt。
