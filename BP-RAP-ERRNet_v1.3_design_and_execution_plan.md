# BP-RAP-ERRNet v1.3 项目设计与执行计划

> 适用场景：复旦大学数字图像处理课程大 PJ：Single Image Reflection Removal / Reflection Removal  
> 当前版本：v1.3  
> 建议方法名：**BP-RAP-ERRNet: Baseline-Preserving Reflection-Aware Proximal Refinement**  
> 中文名：**基于强基线保持与反射感知近端残差优化的单图反射去除方法**

---

## 0. 版本结论

本版本不建议再把项目表述为“把若干 SOTA 模块缝合到 ERRNet 上”。更推荐把项目统一表述为：

> 以课程 ERRNet-Hyper 为强基线和初始解，将反射去除的改进过程建模为一个 **baseline-preserving proximal residual refinement** 问题。模型只在高反射、高不确定区域中学习受控残差修正，在低反射区域通过 anchor loss 和 zero-residual initialization 保持 ERRNet 原输出，从而实现“真实复杂反射区域增强修复 + 非反射区域 do-no-harm”。

因此，v1.3 不是推翻 v1.2，而是对 v1.2 的**理论重构和轻量增强**：

- v1.2 的 `RAP-ERRNet-Hyper-ZeroRes-Staged` 是主干和保底最终模型；
- v1.3 将 v1.2 解释为一个近端残差优化框架；
- v1.3 可选加入两个轻量训练约束：
  - **Reflection-Invariant Consistency, RIC**；
  - **Frequency-Selective Supervision, FSS**。

最终实验策略：

1. 立刻训练 v1.2 staged zero-res 主线，作为最终项目的保底主结果。
2. 同步让 agent 实现 v1.3 的 RIC / FSS，可开关，默认关闭。
3. v1.2 主线训练完成后，再训练或短训 v1.3-RIC / v1.3-RIC+FSS，作为增强版或探索性消融。
4. 如果 v1.3 明显提升且不破坏 CEILNet，则把 v1.3 作为最终主方法；否则将 v1.2 作为最终主方法，v1.3 作为探索性扩展。

---

## 1. 论文调研后的设计依据

### 1.1 ERRNet 的角色：课程强基线与初始解

ERRNet 指出 SIRR 的两个核心难点：

1. 单图反射去除本质上是 ill-posed；
2. 真实世界 dense labels 难以获得。

其主要策略是：

- 在网络中加入 context encoding modules，以利用高层上下文缓解强反射区域的不确定性；
- 设计 alignment-invariant loss，使 misaligned real-world training data 可用于训练。

在本项目中，ERRNet 不应被替代，而应作为：

$$
T_0 = F_b(I)
$$

其中 $F_b$ 是课程提供的 pretrained ERRNet-Hyper baseline。

后续所有改进都应围绕 $T_0$ 做 conservative residual correction，而不是重新从零预测整张图。

---

### 1.2 Location-aware SIRR 的启发：反射位置是重要先验，但不应照搬整套网络

Location-aware SIRR 提出 reflection detection module，用 multi-scale Laplacian features 回归 probabilistic reflection confidence map，即 RCMap。该 map 判断一个区域是 reflection-dominated 还是 transmission-dominated，并控制后续 feature flow。

这说明：

> 反射去除不应该对整张图施加同等强度的修复，而应显式区分“反射主导区域”和“透射主导区域”。

但直接照搬其 recurrent detection-and-removal framework 会偏离课程 ERRNet baseline。因此本项目只保留核心思想：

$$
P = H_\psi(I), \quad P\in[0,1]^{H\times W}
$$

其中 $P$ 是轻量 ReflectionPriorHead 输出的 reflection probability map，用于控制残差修正强度。

---

### 1.3 DSRNet 的启发：真实反射不是简单线性叠加，应使用残差建模非线性

DSRNet 指出，传统模型如：

$$
I = T + R
$$

或者：

$$
I = \alpha T + \beta R
$$

无法充分刻画真实玻璃反射中的衰减、过曝、吸收、非线性叠加等现象。因此 DSRNet 引入 learnable residue term：

$$
I = T + R + \Phi(T,R)
$$

其中 $\Phi$ 用来承接不符合线性叠加假设的残差信息。

本项目不直接预测完整 reflection layer，也不构建重型 dual-stream decomposition network，而是把该思想转化为 ERRNet 输出上的受控修正：

$$
\hat{T}=T_0+\Delta
$$

其中 $\Delta$ 捕捉 ERRNet baseline 未能处理的局部反射残差和非线性误差。

---

### 1.4 RDNet 的启发：分离过程可能损失透射层细节，应采用 do-no-harm 机制

RDNet 从 Information Bottleneck 角度指出，高层语义特征在层层传播中可能压缩或丢弃有价值信息；双流网络的固定交互模式也可能限制性能。因此 RDNet 使用 reversible encoder 和 dynamic prompt 来保留信息并动态校准。

课程项目不适合复现完整 RDNet，但其核心启发是：

> 反射去除模型不仅要“去掉反射”，还要避免损伤已经正确的 transmission details。

因此 BP-RAP-ERRNet 引入：

- zero-residual initialization；
- baseline anchor loss；
- residual magnitude regularization；
- staged fine-tuning。

这些设计共同构成 do-no-harm 机制。

---

### 1.5 RRW 的启发：真实配对数据和反射位置显式建模很重要

RRW 从真实世界采集流程和 reflection location perception 两个角度重新审视 SIRR，提出大规模真实 reflection pairs，并用 MaxRF 显式刻画反射位置。

对本项目的启发是：

1. 主实验仍应严格使用课程指定训练集，保证和 baseline 公平；
2. 真实数据增强只能作为 extra fine-tuning 单独报告；
3. 自采数据必须尽可能构造 paired scene，即同机位有反射图和无反射参考图；
4. 反射位置 map 不应只是可视化，而应参与训练和约束。

---

### 1.6 PromptRR 的启发：低频 veil reflection 和高频 ghost reflection 应区别处理

PromptRR 指出，现有深度学习 SIRR 方法常常难以捕捉 low-frequency 和 high-frequency differences。其使用 diffusion model 生成 LF/HF frequency prompts，然后引导 restoration network。

本项目不引入 diffusion，因为这会增加复杂度且偏离课程 baseline。我们只吸收频域思想，用 Gaussian / Laplacian decomposition 构造轻量 Frequency-Selective Supervision：

$$
L_\sigma(X)=G_\sigma(X)
$$

$$
H_\sigma(X)=X-G_\sigma(X)
$$

通过低频和高频 loss 分别约束 veil-like reflection 和 edge/ghost reflection。

---

### 1.7 Dereflection Any Image 的启发：多样化反射和反射不变性有助于泛化

Dereflection Any Image 认为，现有方法泛化不足的重要原因是：

- 高质量、多样化真实数据不足；
- restoration priors 不充分；
- 模型对不同反射角度、强度和模式不够稳定。

它引入 diversified data 和 reflection-invariant finetuning。

本项目不引入 diffusion priors，但可以用现有 physics synthesis 构造轻量 RIC：对同一张 clean transmission $T$ 合成两个不同 reflection observation，要求模型输出一致。

---

## 2. 最终方法：BP-RAP-ERRNet

### 2.1 总体流程

输入含反射图像：

$$
I\in[0,1]^{H\times W\times 3}
$$

ERRNet-Hyper baseline 输出：

$$
T_0=F_b(I)
$$

反射概率图：

$$
P=H_\psi(I),\quad P\in[0,1]^{H\times W}
$$

可选 prior-gated adapter 得到：

$$
T_g=A_\omega(T_0,P)
$$

残差输入：

$$
R_0=I-T_g
$$

受控残差：

$$
\Delta = P\odot \rho\cdot \tanh U_\phi([I,T_g,R_0,P])
$$

最终输出：

$$
\hat{T}=\mathrm{clip}(T_g+\Delta,0,1)
$$

其中：

- $F_b$：pretrained ERRNet-Hyper backbone；
- $H_\psi$：ReflectionPriorHead；
- $A_\omega$：prior-gated adapter；
- $U_\phi$：lightweight residual refinement；
- $P$：反射区域概率图；
- $\rho$：残差幅度上界，建议 $0.1$。

---

### 2.2 为什么不是 SOTA 缝合？

本方法并不复制 Location-aware / DSRNet / RDNet / PromptRR 的完整网络，而是将近期工作指出的核心问题统一到一个优化目标里：

| 近期研究指出的问题 | BP-RAP-ERRNet 的对应设计 |
|---|---|
| 强反射区域需要显式位置感知 | Reflection prior map $P$ |
| 真实反射不满足简单线性叠加 | 受控残差 $\Delta$ |
| 模型容易破坏透射细节 | baseline anchor + zero-residual |
| 反射低频/高频形态差异明显 | Frequency-Selective Supervision |
| 模型容易记住合成反射模式 | Reflection-Invariant Consistency |
| 真实数据与测试公平性冲突 | extra fine-tuning 单独报告 |

所以本文贡献不是“拼接模块”，而是：

> 将强基线增强建模为一个由反射概率控制的近端残差优化问题。

---

## 3. 核心优化目标

BP-RAP 将训练目标写为：

$$
\min_{\Delta}\
\underbrace{\left\|(1+\alpha P)\odot(T_g+\Delta-T)\right\|_1}_{\mathcal{L}_{rw}:\text{ reflection-weighted reconstruction}}
+\lambda_a\underbrace{\left\|(1-\mathrm{sg}(P))\odot\Delta\right\|_1}_{\mathcal{L}_{anchor}:\text{ baseline-preserving anchor}}
+\lambda_\delta\underbrace{\|\Delta\|_1}_{\mathcal{L}_{delta}:\text{ residual sparsity}}
$$

其中 $\mathrm{sg}(P)$ 表示 stop-gradient，避免 anchor loss 把 $P$ 学成全 1。

完整损失为：

$$
\mathcal{L}
=
\lambda_{pix}\mathcal{L}_{rw}
+\lambda_{grad}\mathcal{L}_{grad}
+\lambda_{ssim}\mathcal{L}_{ssim}
+\lambda_{mask}\mathcal{L}_{mask}
+\lambda_{clean}\mathcal{L}_{clean}
+\lambda_a\mathcal{L}_{anchor}
+\lambda_\delta\mathcal{L}_{delta}
+\lambda_{ric}\mathcal{L}_{RIC}
+\lambda_{freq}\mathcal{L}_{freq}
$$

推荐默认权重：

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
  lambda_ric: 0.0      # 默认关闭，单独开实验
  lambda_freq: 0.0     # 默认关闭，单独开实验
```

---

## 4. 理论性质与证明

### 4.1 性质一：低反射区域的退化有上界

为了分析 anchor 的作用，先忽略 SSIM、gradient、mask 等项，只看单像素二次目标：

$$
\min_{\Delta_i}
(1+\alpha P_i)^2(T_{g,i}+\Delta_i-T_i)^2
+
\lambda_a(1-P_i)^2\Delta_i^2
$$

对 $\Delta_i$ 求导：

$$
2(1+\alpha P_i)^2(T_{g,i}+\Delta_i-T_i)
+
2\lambda_a(1-P_i)^2\Delta_i=0
$$

整理得到：

$$
\left[(1+\alpha P_i)^2+\lambda_a(1-P_i)^2\right]\Delta_i
=
(1+\alpha P_i)^2(T_i-T_{g,i})
$$

所以闭式最优解为：

$$
\Delta_i^*
=
\frac{(1+\alpha P_i)^2}{(1+\alpha P_i)^2+\lambda_a(1-P_i)^2}
(T_i-T_{g,i})
$$

定义：

$$
c_i=
\frac{(1+\alpha P_i)^2}{(1+\alpha P_i)^2+\lambda_a(1-P_i)^2}
$$

则：

$$
\Delta_i^*=c_i(T_i-T_{g,i})
$$

当 $P_i\to0$，即低反射区域：

$$
c_i\approx\frac{1}{1+\lambda_a}
$$

若 $\lambda_a$ 较大，则：

$$
\Delta_i^*\approx0
$$

因此：

$$
\hat{T}_i\approx T_{g,i}\approx T_{0,i}
$$

即模型会保持 baseline 输出。

当 $P_i\to1$，即高反射区域：

$$
c_i\approx1
$$

因此：

$$
\Delta_i^*\approx T_i-T_{g,i}
$$

模型允许充分修正 baseline error。

**结论：** 在二次局部近似下，BP-RAP 会根据 $P_i$ 在“保持 baseline”和“充分残差修正”之间自动插值。

---

### 4.2 性质二：Zero-residual 初始化保证初始等价于 baseline

若 gated adapter 的输出层和 refinement head 的最后一层均初始化为 0，则：

$$
U_{\phi_0}(\cdot)=0
$$

因此：

$$
\Delta_{\phi_0}=P\odot\rho\cdot\tanh(0)=0
$$

所以：

$$
\hat{T}_{\phi_0}=T_g+0
$$

若 adapter 也为 zero-residual identity，则：

$$
T_g=T_0
$$

于是：

$$
\hat{T}_{\phi_0}=T_0=F_b(I)
$$

**结论：** BP-RAP 在训练开始时与 pretrained ERRNet-Hyper 功能等价，不会因为新增随机模块而破坏 baseline。

---

### 4.3 性质三：残差幅度有显式上界

因为：

$$
\Delta=P\odot\rho\cdot\tanh U_\phi(\cdot)
$$

且：

$$
P\in[0,1],\quad |\tanh(x)|\le1
$$

所以：

$$
|\Delta_i|\le\rho P_i
$$

当 $P_i\to0$：

$$
|\Delta_i|\to0
$$

当 $P_i\to1$：

$$
|\Delta_i|\le\rho
$$

**结论：** 模型对每个像素的最大修改幅度有上界，因此比 unconstrained refinement 更不容易产生全局颜色漂移。

---

## 5. 新增创新一：Reflection-Invariant Consistency, RIC

### 5.1 动机

单图反射去除真正需要学习的不是某一种合成反射模式，而是：

> 同一个透射场景 $T$ 叠加不同反射扰动后，模型恢复出的 transmission 应保持一致。

因此，对同一张 clean image $T$，用 physics synthesis 生成两个不同反射版本：

$$
I_a=\mathcal{S}(T,R_a,\eta_a)
$$

$$
I_b=\mathcal{S}(T,R_b,\eta_b)
$$

模型输出：

$$
\hat{T}_a=G_\theta(I_a)
$$

$$
\hat{T}_b=G_\theta(I_b)
$$

加入一致性损失：

$$
\mathcal{L}_{RIC}=\|\hat{T}_a-\hat{T}_b\|_1
$$

更推荐 prior-weighted 版本：

$$
\mathcal{L}_{RIC}=\|(1+\beta_{ric}\bar{P})\odot(\hat{T}_a-\hat{T}_b)\|_1
$$

其中：

$$
\bar{P}=\frac{P_a+P_b}{2}
$$

### 5.2 有效性解释

假设模型输出可以分解为：

$$
G(I_k)=T+h(R_k)
$$

其中 $h(R_k)$ 是输出中残留的 reflection-dependent component。

则：

$$
G(I_a)-G(I_b)=h(R_a)-h(R_b)
$$

如果最小化：

$$
\mathbb{E}_{a,b}\|G(I_a)-G(I_b)\|_1
$$

模型会被迫压制依赖 $R$ 的输出成分。理想情况下：

$$
h(R_a)=h(R_b)=0
$$

于是：

$$
G(I_a)=G(I_b)=T
$$

**结论：** RIC 不是为了贴合某一个测试集，而是从分布层面提升 reflection invariance，因此可以减少“刷分式优化”的嫌疑。

### 5.3 实现建议

新增配置：

```yaml
loss:
  lambda_ric: 0.03
  ric_prob: 0.5
  ric_prior_weight: 1.0
```

训练时只在 synthetic VOC physics synthesis batch 上启用，不在 real paired data 上启用。

---

## 6. 新增创新二：Frequency-Selective Supervision, FSS

### 6.1 动机

反射通常包括两类频率形态：

1. **低频 veil reflection**：雾状、亮度漂移、泛白；
2. **高频 ghost reflection**：边缘、文字、灯光轮廓。

PromptRR 用 diffusion 生成 frequency prompts。本项目采用更轻量、更符合 DIP 课程的方式：Gaussian / Laplacian 分解。

### 6.2 公式

低通：

$$
L_\sigma(X)=G_\sigma(X)
$$

高通：

$$
H_\sigma(X)=X-G_\sigma(X)
$$

频域监督：

$$
\mathcal{L}_{freq}
=
\lambda_L\|(1+\alpha_LP)\odot(L_\sigma(\hat{T})-L_\sigma(T))\|_1
+
\lambda_H\|(1+\alpha_HP)\odot(H_\sigma(\hat{T})-H_\sigma(T))\|_1
$$

建议参数：

```yaml
loss:
  lambda_freq: 0.03
  lambda_freq_low: 1.0
  lambda_freq_high: 0.5
  freq_sigma: 3.0
```

### 6.3 投影视角解释

设 baseline error 为：

$$
e=T_g-T
$$

若反射误差主要落在某个频率子空间 $\mathcal{S}$ 中，最优修正可写为：

$$
\Delta^*=-\Pi_{\mathcal{S}}e
$$

修正后误差：

$$
e+\Delta^*=e-\Pi_{\mathcal{S}}e
$$

于是：

$$
\|e+\Delta^*\|_2^2
=
\|e\|_2^2-\|\Pi_{\mathcal{S}}e\|_2^2
$$

只要反射误差在该频率子空间中有非零能量，投影修正就能降低误差；同时因为修正被限制在特定频率结构中，它更不容易破坏无关的 transmission details。

---

## 7. Reliability-Weighted Mask Supervision

真实 paired data 没有可靠 reflection mask。若用：

$$
\tilde{M}=\mathrm{Norm}(|I-T|)
$$

作为 pseudo mask，它会混入 misalignment、曝光差、阴影、边缘差异等噪声。

设真实 mask 为 $M$，伪标签为：

$$
\tilde{M}=M+\epsilon
$$

则直接监督：

$$
\mathcal{L}_{mask}=\mathrm{BCE}(P,\tilde{M})
$$

可能让 prior head 学到噪声。因此引入可靠性权重：

$$
\mathcal{L}_{mask}=r\cdot\mathrm{BCE}(P,\tilde{M})+r\cdot\lambda_{dice}(1-\mathrm{Dice}(P,\tilde{M}))
$$

其中：

$$
r=\begin{cases}
1,&\text{synthetic mask}\\
\eta,&\text{pseudo mask from real pairs}
\end{cases}
$$

推荐：

$$
\eta=0.2
$$

这对应代码中的：

```yaml
lambda_mask: 0.01
lambda_mask_pseudo_scale: 0.2
```

---

## 8. 数据策略

### 8.1 主训练数据

主实验只使用课程标准训练集：

```text
VOC physics synthesis + Zhang real89 train
```

不要把以下数据用于主训练：

```text
CEILNet Table2
Zhang real20
SIR2 Objects/Postcard/Wild
self-collected final test scenes
```

### 8.2 Extra data

OpenRR / RRW-style / extra_train 只能作为单独 fine-tuning 实验：

```text
BP-RAP-ERRNet
BP-RAP-ERRNet + Extra Real FT
```

这样可以保证主实验与 ERRNet baseline 公平。

### 8.3 自采数据

自采必须是 paired scenes：

```text
data/self_collected/test/scene_001/blended.png
data/self_collected/test/scene_001/transmission.png
data/self_collected/test/scene_001/meta.json
```

建议采集 8–12 组，最后选 5 组报告。

---

## 9. 后续训练计划

### 9.1 不建议直接跳过 v1.2

当前最稳妥路线是：

> 先训练 v1.2 staged zero-res 主线，再训练 v1.3 的 RIC/FSS 增强版。

理由：

1. v1.2 已经是稳定、可解释、代码准备好的最终主线；
2. v1.3 的核心理论本质上包含 v1.2；
3. RIC / FSS 虽然创新更强，但未验证，不能让它阻塞主结果；
4. 课程项目要有一个可靠 final checkpoint，而不是全押在新 loss 上。

因此执行顺序为：

```text
A. 立刻启动 v1.2: BP-RAP-Hyper-ZeroRes-Staged
B. 同步让 agent 实现 v1.3 RIC/FSS
C. v1.2 训练结束后完整评估
D. 再启动 v1.3-RIC 或从 v1.2 checkpoint 短训
E. 如果 v1.3 更好，用 v1.3 做最终主方法；否则 v1.2 做主方法，v1.3 做探索性实验
```

---

### 9.2 Stage 0：zero-res sanity check

目标：确认未经训练的 BP-RAP 初始化输出与 ERRNet-Hyper baseline 近似一致。

```bash
python eval_all.py \
  --model rap_errnet \
  --config configs/rap_errnet_hyper_zerores_staged.yaml \
  --ckpt checkpoints/errnet/errnet_060_00463920.pt \
  --data_root ./data \
  --save_dir results/zerores_init_sanity \
  --datasets ceilnet \
  --save_images \
  --device auto
```

如果当前脚本不支持直接这样 eval init model，可让 agent 添加 `--init_from_errnet_only` 或单独写 `sanity_zerores.py`。

---

### 9.3 Stage 1：v1.2 staged zero-res 主实验

命令：

```bash
python train_rap_errnet.py \
  --config configs/rap_errnet_hyper_zerores_staged.yaml \
  --name bp_rap_hyper_zerores_staged_ppu_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 32 \
  --num_workers 8 \
  --device auto \
  --progress_bar
```

训练设置：

```text
Warm-up: freeze ERRNet backbone, train new modules 20 epochs
Fine-tune: unfreeze backbone, differential LR, train 80 epochs
```

中途评估：

```text
After epoch 20: CEILNet + SIR2 Objects + SIR2 Wild
After epoch 100: all datasets
```

---

### 9.4 Stage 2：v1.3-RIC 实验

若 agent 快速实现 RIC，则可以训练：

```bash
python train_rap_errnet.py \
  --config configs/bp_rap_hyper_zerores_staged_ric.yaml \
  --name bp_rap_hyper_zerores_staged_ric_ppu_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --batch_size 32 \
  --num_workers 8 \
  --device auto \
  --progress_bar
```

若时间不足，则从 v1.2 checkpoint 做短训：

```bash
python train_rap_errnet.py \
  --config configs/bp_rap_hyper_zerores_staged_ric.yaml \
  --name bp_rap_hyper_zerores_ric_ft \
  --resume checkpoints/bp_rap_hyper_zerores_staged_ppu_bs32/best.pt \
  --resume_model_only \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-5 \
  --num_workers 8 \
  --device auto \
  --progress_bar
```

该短训结果在报告中应标记为：

```text
BP-RAP + RIC fine-tuning
```

而不是和完整训练结果混为一谈。

---

### 9.5 Stage 3：v1.3-RIC+FSS 探索实验

如果 RIC 稳定，再加入 FSS：

```bash
python train_rap_errnet.py \
  --config configs/bp_rap_hyper_zerores_staged_ric_freq.yaml \
  --name bp_rap_hyper_zerores_staged_ric_freq_ppu_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --use_freq_loss \
  --batch_size 32 \
  --num_workers 8 \
  --device auto \
  --progress_bar
```

FSS 建议先短训，不要一开始作为最终主方法。

---

## 10. 消融实验

最低必须完成：

| Ablation | 目的 |
|---|---|
| w/o prior | 证明 reflection prior 是否有贡献 |
| w/o gated adapter | 证明 prior-guided feature modulation 是否有贡献 |
| w/o refinement | 证明 residual correction 是否有贡献 |
| w/o anchor | 证明 baseline-preserving 机制是否减少 drift |

建议新增：

| Ablation | 目的 |
|---|---|
| w/o RIC | 证明 reflection-invariant consistency 是否有助于泛化 |
| w/o FSS | 证明 frequency-selective supervision 是否有效 |
| w/o pseudo-mask downweight | 证明伪 mask 降权是否必要 |

如果时间不足，ablation 可以 30 epoch short-run，但报告中必须说明。

---

## 11. 最终论文叙事

推荐最终 claim：

> We formulate ERRNet enhancement as a baseline-preserving reflection-aware proximal residual refinement problem. Instead of replacing ERRNet with a heavier SOTA architecture, BP-RAP starts from the pretrained ERRNet-Hyper solution and learns a zero-initialized, prior-gated residual correction. The proximal anchor constrains modifications in low-reflection regions, while reflection-invariant consistency and frequency-selective supervision improve robustness to diverse reflection patterns. Experiments show that BP-RAP improves real-world structural and local metrics on SIR2 subsets while controlling degradation on strong synthetic benchmarks.

中文表述：

> 本项目不是简单拼接近期 SOTA 模块，而是将 ERRNet 的增强过程建模为“强基线保持的反射感知近端残差优化”问题。模型以 ERRNet-Hyper 输出为初始解，通过反射概率图控制残差修正位置，并利用 zero-residual 初始化、anchor loss 和 residual regularization 防止低反射区域被过度修改。在此基础上，引入反射不变一致性和频率选择性监督，以提升模型对不同反射模式的泛化能力。

---

## 12. 给 Agent 的执行 Prompt

```text
你是一名熟悉 PyTorch、图像复原、single image reflection removal、ERRNet 和当前 RAP-ERRNet 工程的高级工程师。请在当前项目仓库中读取并遵循以下设计文档：

BP-RAP-ERRNet_v1.3_design_and_execution_plan.md

当前目标不是重构整个项目，而是在现有 RAP-ERRNet v1.2 代码基础上，新增 v1.3 的两个可选训练约束，并保证 v1.2 主线不被破坏。

一、总原则
1. 不要删除或破坏原始 ERRNet baseline。
2. 不要破坏已有 RAP-ERRNet v1.2 staged zero-res 训练配置。
3. 所有 v1.3 新功能必须默认关闭，通过 config 和命令行开关启用。
4. 不要自动下载任何数据集。
5. 不要大规模改 backbone。
6. 所有路径用 pathlib，兼容 Linux 和 Windows。
7. 新增代码要能通过 py_compile。
8. 完成后给出新增文件、修改文件、运行命令和注意事项。

二、需要实现的新功能

功能 1：Reflection-Invariant Consistency Loss, RIC
- 新增命令行开关：--use_ric
- 新增 config 字段：
  loss.lambda_ric
  loss.ric_prob
  loss.ric_prior_weight
- 只在 synthetic VOC physics synthesis batch 上启用。
- 对同一个 transmission T 采样两种不同 reflection synthesis：Ia, Ib。
- forward 两次得到 output_a, output_b 和 prior_a, prior_b。
- 计算：
  L_RIC = mean(abs(output_a - output_b))
- 如果 ric_prior_weight > 0，则使用：
  P_bar = 0.5 * (prior_a + prior_b)
  L_RIC = mean((1 + ric_prior_weight * P_bar) * abs(output_a - output_b))
- 该 loss 乘以 lambda_ric 后加入 total loss。
- 若 batch 不是 synthetic 或无法构造 paired double synthesis，则跳过 RIC 并 warning 一次。

功能 2：Frequency-Selective Supervision, FSS
- 新增命令行开关：--use_freq_loss
- 新增 config 字段：
  loss.lambda_freq
  loss.lambda_freq_low
  loss.lambda_freq_high
  loss.freq_sigma
  loss.freq_prior_weight
- 用 Gaussian blur 得到低频：
  L(x)=GaussianBlur(x, sigma)
- 高频：
  H(x)=x-L(x)
- 计算：
  L_low = L1(L(pred), L(target))
  L_high = L1(H(pred), H(target))
- 若有 prior 且 freq_prior_weight > 0，则使用 prior 加权：
  weight = 1 + freq_prior_weight * prior
- 总频域 loss：
  L_freq = lambda_freq_low * L_low + lambda_freq_high * L_high
- 该 loss 乘以 lambda_freq 后加入 total loss。
- 必须 CPU/GPU 都能运行，不依赖 OpenCV。

三、需要新增/修改的文件

优先修改：
- losses/reflection_losses.py
- train_rap_errnet.py
- configs/rap_errnet_hyper_zerores_staged.yaml

新增配置：
- configs/bp_rap_hyper_zerores_staged.yaml
- configs/bp_rap_hyper_zerores_staged_ric.yaml
- configs/bp_rap_hyper_zerores_staged_ric_freq.yaml

可选新增：
- losses/frequency_losses.py
- scripts/sanity_zerores.py

四、配置建议

bp_rap_hyper_zerores_staged.yaml：
- 与现有 rap_errnet_hyper_zerores_staged.yaml 保持一致
- lambda_ric: 0.0
- lambda_freq: 0.0

bp_rap_hyper_zerores_staged_ric.yaml：
- 基于 staged zero-res
- lambda_ric: 0.03
- ric_prob: 0.5
- ric_prior_weight: 1.0
- lambda_freq: 0.0

bp_rap_hyper_zerores_staged_ric_freq.yaml：
- 基于 RIC config
- lambda_freq: 0.03
- lambda_freq_low: 1.0
- lambda_freq_high: 0.5
- freq_sigma: 3.0
- freq_prior_weight: 1.0

五、sanity check

如方便，请新增 scripts/sanity_zerores.py：
- 加载 ERRNet-Hyper baseline checkpoint
- 初始化 BP-RAP-Hyper-ZeroRes
- 不训练，输入同一张图，比较 baseline output 和 BP-RAP init output
- 输出 mean absolute difference
- 若 zero-res 生效，该差异应接近 0

六、运行命令请写入 README 或输出说明

主线训练：
python train_rap_errnet.py \
  --config configs/bp_rap_hyper_zerores_staged.yaml \
  --name bp_rap_hyper_zerores_staged_ppu_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --batch_size 32 \
  --num_workers 8 \
  --device auto \
  --progress_bar

RIC 训练：
python train_rap_errnet.py \
  --config configs/bp_rap_hyper_zerores_staged_ric.yaml \
  --name bp_rap_hyper_zerores_staged_ric_ppu_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --batch_size 32 \
  --num_workers 8 \
  --device auto \
  --progress_bar

RIC + FSS 训练：
python train_rap_errnet.py \
  --config configs/bp_rap_hyper_zerores_staged_ric_freq.yaml \
  --name bp_rap_hyper_zerores_staged_ric_freq_ppu_bs32 \
  --data_root ./data \
  --use_physics_synthesis \
  --use_prior_head \
  --use_gated_blocks \
  --use_refinement \
  --use_ric \
  --use_freq_loss \
  --batch_size 32 \
  --num_workers 8 \
  --device auto \
  --progress_bar

七、验收
1. python -m py_compile train_rap_errnet.py
2. python -m py_compile losses/reflection_losses.py
3. 如果新增 losses/frequency_losses.py，也做 py_compile。
4. 不要求本地完整训练，但要求 --help 正常。
5. 不允许破坏现有 v1.2 config 和训练入口。
```

---

## 13. 最终建议

现在不要直接全押 v1.3。最稳路线是：

1. **马上启动 v1.2 staged zero-res 主实验**，它就是 BP-RAP 的核心版本；
2. **同步实现 v1.3 的 RIC / FSS**，不要等待；
3. **v1.2 完成后完整评估并作为保底最终结果**；
4. **RIC 若稳定提升，再升级为最终主方法**；
5. **FSS 先作为探索性实验**，如果没有提升也可以在论文中作为 negative/diagnostic result 简短讨论；
6. **必须补自采 5 组 paired data**，这是课程项目硬要求。
