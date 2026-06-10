# ERRNet / BP-RAP / RAFA 项目实验总览

这个文件用于回答一个核心问题：**我们这个反射去除课程项目到底做了什么、为什么这样做、跑了哪些训练和测试、最新结果说明了什么、论文里应该写哪些。**

如果只想看最终论文写法，优先读第 1、4、5、6、9 节。  
如果想追溯完整实验流水账，看 `EXPERIMENT_LOG_RAP_ERRNET.md`。

## 1. 当前项目一句话结论

我们从课程指定的 **ERRNet-Hyper baseline** 出发，没有重新训练一个大而不稳定的新网络，而是做了两层改进：

1. **课程公平改进：BP-RAP RIC**
   - 不使用 OpenRR 训练数据。
   - 在 ERRNet 输出基础上加入反射先验、门控适配、残差细化和 RIC 一致性训练。
   - 结果不是全数据集碾压 ERRNet：CEILNet 和 Zhang20 仍弱于 ERRNet，但在 SIR2 三个真实场景子集上明显优于 ERRNet。

2. **额外真实数据适配：RAFA**
   - 使用 OpenRR train 作为额外真实配对数据。
   - 通过课程数据 replay 防止只适配 OpenRR 后损害课程测试集。
   - 最新结果显示，**RAFA3k w/o old distillation** 是当前 extra-data 最优方法：Course 5-set、SIR2、OpenRR val、Six-set mean 都优于 Balanced RAFA3k。

最终论文建议这样定位：

| 场景 | 推荐方法 | 是否使用 OpenRR train | 论文定位 |
| --- | --- | --- | --- |
| 课程公平主结果 | BP-RAP RIC | 否 | 和 ERRNet baseline 公平比较 |
| 额外真实数据主结果 | RAFA3k w/o old | 是，OpenRR train 3k | 作为 extra-data adaptation 方法 |
| 旧 final checkpoint | Balanced RAFA3k | 是，OpenRR train 3k | 之前的 metric-best，现在被 w/o old 超过 |
| 自采数据 | 待补 | 待评估 | 课程硬性要求，明天补 |

## 2. 读哪些文件

| 文件 | 内容 | 什么时候看 |
| --- | --- | --- |
| `EXPERIMENT_LOG_RAP_ERRNET.md` | 完整实验流水账，包含命令、checkpoint、metric 表和决策 | 想查某次实验细节 |
| `EXPERIMENT_MATRIX_SUMMARY.md` | 当前这个中文总览文件 | 想快速理解项目做了什么 |
| `FINAL_ABLATION_RUNBOOK.md` | 最终消融和绘图的服务器命令 | 要在阿里云复现实验或生成图 |
| `RAP_ERRNET_EXPERIMENT_REPORT.md` | 阶段性实验报告 | 早期总结参考 |
| `BP-RAP-ERRNet_v1.3_design_and_execution_plan.md` | BP-RAP / RIC / FSS 的详细设计 | 写方法部分时参考 |
| `design_v1.4.md` | RAFA、balanced sampling、soup 的规划 | 写 extra-data 适配时参考 |
| `BP_RAP_V1_4_RAFA_ITERATION_PLAN.md` | RAFA 执行方案 | 查 OpenRR 训练流程 |
| `results/FINAL_TABLES_FOR_PAPER.md` | 最新生成的论文表格草稿 | 写论文表格时直接用 |

## 3. 数据集和正式评估协议

### 3.1 训练数据

| 训练数据 | 数量 | 用途 | 是否课程公平 |
| --- | ---: | --- | --- |
| Pascal VOC cropped clean images | 7,643 张 224 x 224 crop | 用物理合成反射图训练 | 是 |
| Zhang real train | 89 对真实训练图像 | 课程真实训练数据 / replay | 是 |
| OpenRR train 1k | 1,000 对真实配对图像 | OpenRR-FT、Soup、RAFA1k | 否 |
| OpenRR train 3k | 3,000 对真实配对图像 | RAFA3k、Balanced、w/o old、OpenRR-only | 否 |
| 自采 paired data | 至少 5 对，未完成 | 课程要求的自采测试 | 待补 |

注意：OpenRR train 是额外数据，不能和课程公平主结果混在一起。论文中要明确分成：

- course-fair setting
- extra-data / OpenRR adaptation setting

### 3.2 测试数据和正式协议

| 测试集 | 数量 | 正式协议 | 备注 |
| --- | ---: | --- | --- |
| CEILNet Table2 | 100 | native resolution | 合成测试集，ERRNet 很强 |
| Zhang real20 | 20 | `--max_long_edge 512` | 课程真实测试集 |
| SIR2 Objects | 200 | native resolution | 真实场景 |
| SIR2 Postcard | 179 | native resolution | 真实场景 |
| SIR2 Wild | 101 | native resolution | 真实场景 |
| OpenRR val | 300 | native resolution | 外部真实 benchmark |
| Self-collected | 待补 | 建议 native，若 OOM 需说明 resize | 课程硬性要求 |

正式指标：

| 指标 | 越大/越小越好 | 含义 |
| --- | --- | --- |
| PSNR | 越大越好 | 像素误差，越高越接近 GT |
| SSIM | 越大越好 | 结构相似度 |
| NCC | 越大越好 | 归一化相关性 |
| LMSE | 越小越好 | 局部 MSE，更关注局部保真 |

重要提醒：之前有一些 all-512 diagnostic 结果，那些只用于快速筛选，不放进主论文表格。

## 4. 我们的方法到底是什么

### 4.1 Baseline：ERRNet-Hyper

课程指定 baseline 是 ERRNet。我们实际评估的强 baseline 是 **ERRNet-Hyper**：

- 输入不是普通 RGB 3 通道，而是 RGB + VGG hypercolumn 特征。
- 这也是课程 checkpoint `checkpoints/errnet/errnet_060_00463920.pt` 的结构。
- 在 CEILNet 和 Zhang20 上非常强，是我们很难超过的主要原因。

论文里 baseline 可以写成：

> We reproduce the course ERRNet-Hyper baseline and use it as the strong reference model.

### 4.2 早期尝试：RAP from scratch

我们最开始尝试从零训练一个 RAP-ERRNet：

- 使用 VOC 物理合成 + Zhang real89。
- 模型包含 prior、gate、refinement。
- 但 backbone 是 3-channel from-scratch。

结果：

| 方法 | CEILNet | Zhang20 | SIR2 Objects | SIR2 Postcard | SIR2 Wild |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAP from scratch | 19.1752 | 19.5196 | 25.7196 | 20.8931 | 25.5726 |

结论：

- SIR2 Objects/Wild 有一些提升潜力。
- CEILNet 和 Zhang20 远低于 ERRNet。
- 说明从零训练不适合作为主方法，必须以 ERRNet-Hyper 为强初始解。

### 4.3 RAP-Hyper：引入 ERRNet-Hyper 预训练

RAP-Hyper 是中间版本：

- 使用 ERRNet-Hyper checkpoint 初始化 backbone。
- 保留 prior、gate、refinement。
- 训练数据仍是 VOC 合成 + Zhang real89。

它不是最终方法，但很关键，因为后面的 BP-RAP / RAFA 都沿用了这个强初始化路线。

RAP-Hyper 的作用：

- 证明“反射先验 + 残差细化”在真实 SIR2 场景有效。
- 同时暴露问题：直接 fine-tune 会损害 CEILNet/Zhang 的像素保真。

### 4.4 BP-RAP：Baseline-preserving Reflection-Aware Prior refinement

BP-RAP 的核心思想是：

> 不重新生成整张图，而是在 ERRNet 输出的基础上做受控的小残差修正。

流程：

```text
Input I
  -> ERRNet-Hyper 得到初始透射层 T0
  -> Prior Head 预测反射先验 P
  -> Prior-gated Adapter 得到 Tg
  -> Residual Refinement 预测 Delta
  -> 输出 T = Tg + Delta
```

关键设计：

| 模块 | 作用 |
| --- | --- |
| Prior Head | 预测哪里可能有反射 |
| Prior-gated Adapter | 用 prior 控制特征修正位置 |
| Residual Refinement | 只学小残差，不重写整图 |
| residual scale 0.1 | 限制残差幅度 |
| anchor loss | 低反射区域尽量贴近 ERRNet |
| delta regularization | 防止过度修改 |
| pseudo mask | 无 mask 数据用 `abs(input-target)` 近似反射区域 |

### 4.5 RIC：Reflection-Invariant Consistency

RIC 是课程公平方法 BP-RAP RIC 的训练约束：

- 对同一 clean image 合成两种不同反射观测。
- 要求模型输出的透射层保持一致。

动机：

> 如果两个输入只是反射不同，透射层应该相同。模型应学习对反射扰动不敏感。

实际结果显示，RIC 的数值提升不算强，但作为方法叙事是合理的。最新结构消融里 `w/o RIC` 和 full 非常接近，因此论文里不要过度吹 RIC，只说它是稳定训练和反射不变性的约束。

### 4.6 FSS：Frequency-Selective Supervision

FSS 是频率监督探索：

- 目标是区分低频 veil reflection 和高频 ghost reflection。
- 配置为 `lambda_freq=0.03`。

结果：

- 和 BP-RAP RIC 基本持平。
- 没有成为主方法。
- 可以作为“探索过但收益有限”的 ablation。

### 4.7 RAFA：Replay-Anchored Fine-tuning for Real Reflection Adaptation

RAFA 是我们后期最重要的 extra-data 方法。

问题背景：

- OpenRR train 是额外真实配对数据。
- 直接用 OpenRR fine-tune 会显著提升 OpenRR val，但可能让课程数据集变差。

RAFA 的做法：

| 设计 | 作用 |
| --- | --- |
| OpenRR supervision | 利用真实配对数据提升真实反射场景 |
| Course replay | 每个 epoch 混入 VOC synthesis + Zhang real89，防止忘记课程分布 |
| Teacher distillation | 让 student 在 replay 样本上接近原 BP-RAP RIC teacher |
| Frozen backbone | 只微调新模块，降低破坏 ERRNet backbone 的风险 |

最新结果很重要：`lambda_old=0.2` 的旧蒸馏并不是最优，**RAFA3k w/o old** 反而最好。这说明：

- course replay 是必要的；
- old distillation 可能过度限制模型适配 OpenRR；
- 论文里可以把 RAFA 的关键说成 “OpenRR supervision + course replay”，而不是把 old distillation 说成必不可少。

### 4.8 Reflection-strength balanced sampling

Balanced RAFA3k 做了一个采样优化：

- 先分析 OpenRR train 的反射强度和高频比例。
- 把样本分成 weak/strong 与 veil/ghost 四类。
- 在 OpenRR 采样内部做均衡。

它曾经是 metric-best checkpoint，但最新被 RAFA3k w/o old 超过。

现在论文建议：

- 可以把 balanced sampling 写成一个尝试过的 refinement。
- 不要把它作为最终最强算法。

## 5. 最新课程公平结果

课程公平结果只比较没有使用 OpenRR train 的方法。

| Method | Uses OpenRR train? | Course 5-set PSNR | SSIM | NCC | LMSE | SIR2 PSNR | SIR2 SSIM | SIR2 NCC | SIR2 LMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERRNet baseline | N | 24.5077 | 0.8861 | 0.9465 | 0.0081 | 23.8201 | 0.8871 | 0.9546 | 0.0052 |
| BP-RAP RIC | N | 23.8919 | 0.8839 | 0.9419 | 0.0086 | 24.8216 | 0.9088 | 0.9653 | 0.0034 |
| RIC+FSS | N | 23.8820 | 0.8838 | 0.9420 | 0.0086 | 24.8254 | 0.9087 | 0.9654 | 0.0034 |

解读：

1. ERRNet 的 Course 5-set mean 仍最高。
2. BP-RAP RIC 的优势集中在 SIR2 三个真实场景子集。
3. BP-RAP RIC 相对 ERRNet 的 SIR2 PSNR 提升：

| SIR2 子集 | BP-RAP RIC 相对 ERRNet PSNR 提升 |
| --- | ---: |
| Objects | +1.1922 dB |
| Postcard | +0.5623 dB |
| Wild | +1.2501 dB |

论文应该诚实写：

> BP-RAP RIC does not dominate ERRNet on all benchmarks. It trades some pixel-aligned fidelity on CEILNet/Zhang for better real-scene SIR2 performance.

中文意思：

> 我们的方法不是所有数据集都比 ERRNet 强，而是在真实场景数据集上更有优势。

## 6. 最新结构消融结果

结构消融以 BP-RAP RIC 为 full model，所有结果都是 formal evaluation，不是 all-512 diagnostic。

| Method | Prior | Gate | Refine | RIC | FSS | Course 5-set PSNR | SIR2 PSNR | OpenRR val PSNR | Main reading |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BP-RAP RIC | Y | Y | Y | Y | N | 23.8919 | 24.8216 | 26.8597 | full course-fair model |
| w/o prior | N | Y | Y | Y | N | 23.9071 | 24.8273 | 26.8180 | prior 数值收益不明显 |
| w/o gate | Y | N | Y | Y | N | 23.8790 | 24.8126 | 26.7596 | gate 对 OpenRR 有一定帮助 |
| w/o refinement | Y | Y | N | Y | N | 23.5954 | 24.5825 | 26.3458 | refinement 是最关键模块 |
| w/o RIC | Y | Y | Y | N | N | 23.8964 | 24.8301 | 26.8846 | RIC 数值不强，基本持平 |
| RIC+FSS | Y | Y | Y | Y | Y | 23.8820 | 24.8254 | 26.8909 | FSS 基本持平 |

这个表的核心结论：

1. **Residual refinement 最重要**  
   去掉 refinement 后，Course 5-set、SIR2、OpenRR 都明显下降。

2. **prior/gate/RIC/FSS 的单独数值收益不强**  
   这说明论文里不能把每个模块都吹成大幅提升。更合理的写法是：
   - prior/gate 提供空间反射适配机制；
   - RIC/FSS 是训练约束探索；
   - 真正稳定提升主要来自 baseline-preserving residual refinement 和后续 RAFA replay。

3. **w/o prior 甚至略高于 full**  
   这不是说 prior 完全没用，而是说明当前 prior head 的监督较弱，pseudo mask 噪声可能限制了它。论文里可以把 prior 作为可解释模块，但不要把它当作主要数值贡献。

## 7. OpenRR / RAFA 最新结果

这是 extra-data setting，使用 OpenRR train 的方法不能和课程公平结果混在一起。

| Method | Uses OpenRR train? | OpenRR pairs | Course replay? | lambda_old | Balanced? | Course 5-set PSNR | SIR2 PSNR | OpenRR val PSNR | Six-set PSNR | Role |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| ERRNet baseline | N | 0 | N | 0 | N | 24.5077 | 23.8201 | 25.4874 | 24.6709 | baseline |
| BP-RAP RIC | N | 0 | N | 0 | N | 23.8919 | 24.8216 | 26.8597 | 24.3865 | course-fair main |
| OpenRR-FT 1k | Y | 1000 | N | 0 | N | 23.7930 | 24.8573 | 28.9560 | 24.6535 | real-only target adaptation |
| Soup a0.25 | indirect | 1000 | N/A | N/A | N | 23.9549 | 24.9246 | 27.4103 | 24.5308 | checkpoint soup fallback |
| RAFA1k | Y | 1000 | Y | 0.2 | N | 23.9529 | 24.9043 | 27.7031 | 24.5779 | replay anchored adaptation |
| RAFA3k | Y | 3000 | Y | 0.2 | N | 23.9906 | 24.9469 | 27.8483 | 24.6336 | main RAFA algorithm |
| Balanced RAFA3k | Y | 3000 | Y | 0.2 | Y | 23.9999 | 24.9680 | 27.8349 | 24.6390 | previous metric-best |
| Old04 | Y | 3000 | Y | 0.4 | N | 23.9663 | 24.9159 | 27.6352 | 24.5778 | stronger old distillation |
| RAFA3k w/o old | Y | 3000 | Y | 0.0 | N | 24.0085 | 24.9772 | 28.0698 | 24.6853 | current best extra-data |
| OpenRR-only 3k | Y | 3000 | N | 0.0 | N | 23.5304 | 24.7896 | 29.2549 | 24.4845 | no course replay |

最新结论：

1. **RAFA3k w/o old 是当前 extra-data 最优方法**
   - Course 5-set: 24.0085
   - SIR2 mean: 24.9772
   - OpenRR val: 28.0698
   - Six-set mean: 24.6853

2. **OpenRR-only 3k 的 OpenRR val 最高，但课程集掉得明显**
   - OpenRR val: 29.2549
   - Course 5-set: 23.5304
   - 说明只用 OpenRR 会过度偏向 OpenRR，course replay 是必要的。

3. **old distillation 不是必要项**
   - RAFA3k w/o old 比 RAFA3k 和 Balanced RAFA3k 都好。
   - 说明 teacher old loss 可能限制了真实数据适配。

4. **Balanced sampling 不是最终主贡献**
   - Balanced RAFA3k 比 RAFA3k 稍好，但被 w/o old 超过。
   - 可以作为探索实验，不作为最终选择。

论文建议最终 extra-data 方法改成：

> RAFA3k w/o old distillation

更准确的命名可以是：

> RAFA-OpenRR3k without old-model distillation

或者：

> RAFA-OpenRR3k replay-only

## 8. OpenRR zero-shot 结果为什么重要

在没有使用 OpenRR train 的情况下，BP-RAP RIC 已经在 OpenRR val 上超过 ERRNet：

| Method | OpenRR val PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| ERRNet baseline | 25.4874 | 0.9480 | 0.9641 | 0.0030 |
| BP-RAP RIC | 26.8597 | 0.9600 | 0.9693 | 0.0018 |

提升：

| 指标 | BP-RAP RIC vs ERRNet |
| --- | ---: |
| PSNR | +1.3723 dB |
| SSIM | +0.0120 |
| NCC | +0.0052 |
| LMSE | -0.0012 |

这很适合写进论文，因为它说明：

> BP-RAP RIC 的真实场景提升不是只在 SIR2 上出现，它对 OpenRR 这种外部真实 benchmark 也有 zero-shot 泛化能力。

## 9. 我们到底做了哪些工作

按项目阶段整理：

### 阶段 1：复现和评估 ERRNet baseline

完成内容：

- 配置课程数据集。
- 跑 CEILNet、Zhang20、SIR2 Objects/Postcard/Wild。
- 确认 ERRNet-Hyper 是强 baseline。

结果：

- ERRNet 在 CEILNet/Zhang20 很强。
- 但在 SIR2 真实场景上仍有提升空间。

### 阶段 2：实现 RAP-ERRNet

完成内容：

- 新增统一数据集适配器。
- 新增反射物理合成。
- 新增指标 PSNR/SSIM/NCC/LMSE。
- 新增 RAP 模型：
  - prior head
  - gated adapter
  - residual refinement
  - residual scale
  - anchor/delta losses

结果：

- from-scratch RAP 不稳定。
- RAP-Hyper 证明真实场景 SIR2 有提升。

### 阶段 3：BP-RAP 和 RIC

完成内容：

- 从 RAP-Hyper checkpoint 出发。
- 冻结 backbone，短程 fine-tune 新模块。
- 加入 RIC 反射不变一致性训练。
- 加入 FSS 探索。

结果：

- BP-RAP RIC 成为课程公平主方法。
- SIR2 三子集均超过 ERRNet。
- CEILNet/Zhang20 仍低于 ERRNet。

### 阶段 4：OpenRR 外部验证

完成内容：

- 下载并整理 OpenRR train/val。
- 跑 OpenRR val zero-shot。
- 跑 OpenRR-FT 1k。
- 跑 checkpoint soup。

结果：

- BP-RAP RIC zero-shot 在 OpenRR val 比 ERRNet 高 +1.37 dB。
- OpenRR-FT 能大幅提升 OpenRR，但有 source-domain drift。
- Soup 能缓和 drift，但算法叙事不如 RAFA。

### 阶段 5：RAFA 真实数据适配

完成内容：

- 实现 replay sampler。
- 实现 course replay。
- 实现 old-model distillation。
- 跑 RAFA1k、RAFA3k、Balanced RAFA3k。
- 跑 Old04、w/o old、OpenRR-only 3k。

结果：

- RAFA3k 明显优于 RAFA1k。
- Balanced RAFA3k 曾是旧最优。
- 最新 RAFA3k w/o old 成为当前 extra-data 最优。
- OpenRR-only 证明 course replay 重要。

### 阶段 6：最终消融和绘图

完成内容：

- 跑结构消融：
  - w/o prior
  - w/o gate
  - w/o refinement
  - w/o RIC
  - RIC+FSS
- 生成论文数值图：
  - method_overview
  - course_delta_psnr
  - openrr_adaptation_summary
- 增加定性图脚本：
  - qualitative_existing
  - qualitative_self
  - prior_error_analysis
  - failure_cases

结果：

- refinement 是最关键结构模块。
- RAFA3k w/o old 是当前 extra-data 最佳。
- 自采数据还没补，明天需要补。

## 10. 定性图怎么选图和展示

最新定性脚本：

```text
tools/make_paper_qualitative_figures.py
```

它会生成：

| 文件 | 内容 |
| --- | --- |
| `paper/figures/qualitative_main.png/.pdf` | 合并主图，包含现有数据集 + self 占位 |
| `paper/figures/qualitative_existing.png/.pdf` | 只包含现有公开数据集 |
| `paper/figures/qualitative_self.png/.pdf` | 只包含自采数据，目前 self 缺失时是占位 |
| `paper/figures/prior_error_analysis.png/.pdf` | prior 和 error map 分析 |
| `paper/figures/failure_cases.png/.pdf` | CEILNet/Zhang20 失败案例 |

选图规则：

```text
delta = PSNR(final_model) - PSNR(ERRNet)
```

具体规则：

| 图 | 选图方式 |
| --- | --- |
| SIR2 Wild success | 选 delta 最大的成功样例 |
| SIR2 Objects/Postcard success | 两个数据集合并后选 median improved，避免只挑极端 |
| OpenRR success | 选 delta 最大的成功样例 |
| Self success | 自采数据可用后选 delta 最大的样例 |
| CEILNet failure | 选 delta 最小，即 ERRNet 赢最多 |
| Zhang20 failure | 选 delta 最小，即 ERRNet 赢最多 |

展示列：

```text
qualitative_existing / qualitative_self:
Input | ERRNet | BP-RAP RIC | RAFA final | GT

prior_error_analysis:
Input | GT | ERRNet Error | BP-RAP Error | RAFA Error | Prior Map

failure_cases:
Input | ERRNet | BP-RAP RIC | RAFA final | GT
```

当前建议定性图使用的 final model：

```text
RAFA3k w/o old
checkpoint: checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt
config: configs/bp_rap_rafa_openrr3k_no_old.yaml
```

## 11. 论文里建议写哪些结果

### 正文必须写

| 内容 | 原因 |
| --- | --- |
| ERRNet baseline | 课程要求 |
| BP-RAP RIC | 课程公平主方法 |
| Course 5-set formal 表 | 课程要求 |
| SIR2 mean / per-subset 提升 | 我们相对 baseline 的主要优势 |
| OpenRR zero-shot | 证明真实场景外部泛化 |
| RAFA3k w/o old | 当前 extra-data 最优 |
| RAFA/OpenRR adaptation 表 | 展示额外数据适配效果 |
| 自采数据 | 课程硬性要求，待补 |
| qualitative_existing 和 failure_cases | 展示成功和失败案例 |

### 可以写成消融

| 消融 | 结论 |
| --- | --- |
| w/o refinement | 明显下降，证明 refinement 关键 |
| w/o prior / w/o gate | 单独贡献有限，说明 prior/gate 主要是可解释空间适配 |
| w/o RIC / RIC+FSS | 基本持平，说明这两个约束稳定但不是主提升来源 |
| RAFA3k vs RAFA3k w/o old | old distillation 不必要，去掉后更好 |
| RAFA3k w/o old vs OpenRR-only 3k | course replay 防止只偏向 OpenRR |

### 不建议正文重点写

| 内容 | 原因 |
| --- | --- |
| RAP from scratch | 太弱，只作为早期探索 |
| all-512 diagnostic sweep | 不是正式协议 |
| Module-only soup | 没超过 RAFA |
| Balanced sampling 作为主贡献 | 最新已不是最优 |
| Old04 | 负结果，最多一行 |

## 12. 当前最可信的论文叙事

建议论文逻辑：

1. 任务很难，ERRNet 是强 baseline。
2. 我们先复现 ERRNet-Hyper。
3. 直接从零训练 RAP 不稳定，所以采用 baseline-preserving 思路。
4. BP-RAP 在 ERRNet 输出上做反射感知残差细化。
5. BP-RAP RIC 在 SIR2 和 OpenRR zero-shot 上表现更好，但在 CEILNet/Zhang20 上不如 ERRNet。
6. 为进一步提升真实场景，使用 OpenRR 做 extra-data adaptation。
7. 直接 OpenRR-only 会损害课程分布，因此提出 RAFA：OpenRR supervision + course replay。
8. 最新消融显示，去掉 old distillation 后效果更好，最终 extra-data 方法采用 RAFA3k w/o old。
9. 最后展示定量、定性、失败案例和自采数据。

## 13. 当前最终选择

### 课程公平最终方法

```text
BP-RAP RIC
checkpoint: checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt
config: configs/bp_rap_hyper_zerores_staged_ric.yaml
```

理由：

- 不使用 OpenRR train。
- SIR2 三个真实子集均优于 ERRNet。
- OpenRR val zero-shot 也优于 ERRNet。
- 虽然 Course 5-set mean 低于 ERRNet，但这是合理 trade-off。

### 额外数据最终方法

```text
RAFA3k w/o old
checkpoint: checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt
config: configs/bp_rap_rafa_openrr3k_no_old.yaml
```

理由：

- 当前 Six-set mean 最高：24.6853。
- Course 5-set、SIR2、OpenRR val 都优于 Balanced RAFA3k。
- 比 OpenRR-only 更稳，说明 course replay 有价值。

## 14. 还缺什么

| 缺口 | 重要性 | 下一步 |
| --- | --- | --- |
| 自采数据 5 组 | 非常高，课程硬性要求 | 明天采集并放到 `data/self_collected/test/scene_xxx/` |
| 自采数据 formal eval | 非常高 | 跑 ERRNet、BP-RAP RIC、RAFA3k w/o old |
| 自采 qualitative_self | 高 | 自采 eval 后重跑定性图脚本 |
| 论文更新最终方法 | 高 | 把 Balanced RAFA3k 改成 RAFA3k w/o old |
| 检查定性图是否符合肉眼观感 | 高 | 下载 `paper/figures/*.png` 看图 |

## 15. 当前应下载/保留的结果文件

服务器上应保留：

```text
results/ABLATION_STRUCTURE_SUMMARY.md
results/ABLATION_RAFA_SUMMARY.md
results/FINAL_TABLES_FOR_PAPER.md
results/SELF_COLLECTED_SUMMARY.md
results/paper_qualitative/selection_summary.csv
paper/figures/*.png
paper/figures/*.pdf
results/final_eval_*/
```

写论文最需要：

```text
paper/figures/method_overview.png/.pdf
paper/figures/course_delta_psnr.png/.pdf
paper/figures/openrr_adaptation_summary.png/.pdf
paper/figures/qualitative_existing.png/.pdf
paper/figures/prior_error_analysis.png/.pdf
paper/figures/failure_cases.png/.pdf
results/FINAL_TABLES_FOR_PAPER.md
```

明天补自采后还要更新：

```text
results/SELF_COLLECTED_SUMMARY.md
paper/figures/qualitative_self.png/.pdf
paper/figures/qualitative_main.png/.pdf
results/final_eval_*_self/
```

## 16. 最终一句话

这个项目目前已经不是简单“跑了一个 baseline”。完整工作包括：

- 复现 ERRNet-Hyper baseline；
- 实现统一数据和指标评估；
- 设计并实现 BP-RAP 反射感知残差细化；
- 探索 RIC、FSS、prior/gate/refinement 等结构消融；
- 引入 OpenRR 做真实数据适配；
- 设计 RAFA replay 机制避免 OpenRR-only 的分布漂移；
- 通过最新消融发现 RAFA3k w/o old 是当前 extra-data 最优；
- 生成论文用表格和图；
- 还剩自采数据需要补齐。

论文最稳的核心主张是：

> 我们的方法不是在所有 benchmark 上全面超过 ERRNet，而是在真实反射场景上表现更好；通过 RAFA 引入真实 OpenRR 数据和课程 replay 后，进一步提升真实数据表现，同时避免纯 OpenRR fine-tuning 的分布漂移。
