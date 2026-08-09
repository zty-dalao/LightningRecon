# Phase A 码本坍塌：尚未实现的改进项

本文只记录当前代码中**尚未实现**的码本防坍塌方案，作为后续实验与开发清单。
已经完成的K-means初始化、EMA更新、dead-code重置和TensorBoard诊断不在本文中
重复实现。

## 当前基线

当前Phase A已经具备：

- Anatomy码本`512×64`、Boundary码本`256×32`；
- 从多个训练batch均匀收集特征并进行10轮K-means初始化；
- K-means空簇使用高误差真实特征重新初始化；
- `decay=0.99`的EMA命中次数、特征和与码字更新；
- 预热100次EMA更新后，每100次检查并重置死亡码字；
- Anatomy与Boundary各自的perplexity、active fraction和dead-code诊断。

这些措施能够降低坍塌概率，但不能保证编码器不会长期集中到少数码字。

## 1. Boundary结构感知混合采样

### 状态

尚未实现。当前K-means初始化和dead-code重置都从空间位置中进行普通均匀采样。

### 问题

CT中空气、均匀软组织等低结构体素远多于边缘体素。Boundary latent使用普通均匀
采样时，大量候选特征可能来自低梯度区域，使多个Boundary码字表达相近背景，
而骨骼和组织边缘覆盖不足。

### 建议设计

第一版只作用于Boundary码本，不改变Anatomy码本：

```text
Boundary候选特征 = 70% 全空间均匀采样
                 + 30% CT高梯度位置采样
```

- 结构位置直接从归一化CT的一阶三维梯度获得，不依赖body mask；
- 保留多数均匀样本，避免码本只记住强边缘而丢失普通组织；
- 混合比例应做成参数，不能硬编码后失去消融能力；
- K-means初始化与dead-code重置应复用同一采样接口；
- Anatomy继续使用普通均匀采样，以免全局语义被边缘分布主导。

### 预期涉及文件

- `src/ema_codebook.py`：量化器接受可选采样权重或候选索引；
- `src/dual_domain/anatomy_prior.py`：仅向Boundary量化器传递结构信息；
- `src/train_phase_a_prior.py`：新增开关、混合比例和TensorBoard元数据；
- `tests/test_dual_domain.py`：验证采样比例、无边缘体积和小体积边界情况。

## 2. 可微的码本使用多样性损失

### 状态

尚未实现。当前perplexity和active fraction均已`detach`，只用于监控，不参与反向传播。

### 问题

当前commitment loss只要求编码器特征靠近已经选中的码字，并不会直接阻止编码器
长期选择少数码字。dead-code重置只能事后替换死亡码字，不能保证新码字随后会被使用。

### 建议设计

基于特征到码字的距离构造温度控制的soft assignment，再对batch平均使用分布施加
轻量熵约束或与平滑目标分布的KL约束：

```text
p(k|z) = softmax(-distance(z, code_k) / temperature)
p_batch(k) = mean_z p(k|z)
L_diversity = sum_k p_batch(k) × log(p_batch(k) + eps)
```

实施要求：

- 只推动Encoder产生更有区分度的特征，不通过梯度更新EMA码字；
- 从小系数开始，例如`1e-3`，再根据日志做消融；
- 不要求所有码字严格等频使用，避免破坏医学组织天然长尾分布；
- Anatomy和Boundary分别记录该损失，允许使用不同系数；
- 距离计算必须采样或分块，不能在128³全部体素上构造完整soft assignment矩阵。

### 风险

系数过大会为了提高perplexity而强行拆分相似组织，可能提高码本使用率却降低
重建PSNR/SSIM。因此该损失不能只以active fraction为验收依据。

## 3. Encoder预热后再进行K-means初始化

### 状态

尚未实现。当前流程是在优化器正式更新前，使用随机初始化的CT Encoder收集latent
并完成K-means。

### 问题

随机Encoder的特征虽然比随机码字初始化更可控，但训练初期Encoder分布变化很快，
K-means中心可能迅速过时，随后完全依赖EMA追赶新的特征分布。

### 建议设计

```text
第1阶段：旁路量化器，预热Encoder和Decoder 5～10个epoch
第2阶段：固定当前Encoder，跨多个病例收集latent并执行K-means
第3阶段：恢复正常VQ训练和EMA更新
```

需要解决：

- 旁路量化期间Decoder接收连续latent，切换到量化latent时可能产生损失跳变；
- resume checkpoint必须保存当前处于预热、初始化还是VQ训练状态；
- TensorBoard横轴和checkpoint选择不能把预热期与正式VQ期混为一谈；
- 必须与当前“训练前直接K-means”做独立消融，不能与结构采样同时启用后再判断效果。

## 4. 基于诊断指标的自动告警或干预

### 状态

尚未实现。当前代码会记录使用率，但不会根据指标自动改变采样方式、损失系数、
dead-code阈值或训练状态。

### 建议设计

第一版只做告警，不自动修改训练超参数：

```text
如果normalized_perplexity和ema_active_fraction连续多个epoch低于配置阈值：
    写入Codebook/*_collapse_warning = 1
    在终端打印Anatomy或Boundary的具体诊断
否则：
    collapse_warning = 0
```

自动改变损失或采样会破坏实验可比性，因此只有在告警规则稳定后，才考虑加入
显式命令行开启的自动干预模式。

## 推荐实施顺序

不要一次性启用全部方案。建议按以下顺序实验：

1. 使用当前基线训练至少50个epoch，记录两层码本全部诊断指标；
2. 如果主要是Boundary利用率低，先实现Boundary结构感知混合采样；
3. 如果重置后的码字仍快速死亡，再小系数加入可微多样性损失；
4. Encoder预热属于训练流程变化较大的独立消融，最后单独验证；
5. 指标阈值稳定后，再加入只告警、不自动调参的坍塌监控。

## 建议观察指标

主要指标：

- `Codebook/anatomy_normalized_perplexity`；
- `Codebook/anatomy_batch_active_fraction`；
- `Codebook/anatomy_ema_active_fraction`；
- `Codebook/boundary_normalized_perplexity`；
- `Codebook/boundary_batch_active_fraction`；
- `Codebook/boundary_ema_active_fraction`；
- `Codebook/*_dead_codes_reinitialized`；
- `Codebook/*_dead_codes_reinitialized_total`。

辅助质量指标：

- Phase A验证PSNR和SSIM；
- `Val/boundary_edge`；
- image、structural、laplacian和VQ loss；
- 码字重置后验证损失是否出现周期性尖峰。

不能以“所有码字完全均匀使用”为目标。更合理的验收标准是：

- normalized perplexity和EMA active fraction不再持续下降；
- dead-code累计重置曲线逐渐变缓；
- 提高使用率的同时，验证PSNR/SSIM和边缘质量不下降；
- 相同数据划分、随机种子和训练轮数下优于当前基线。

## 当前决策

在获得新版Phase A前50个epoch的实际日志前，以上项目保持未启用状态。优先根据
Anatomy和Boundary各自的坍塌位置选择最小改动，避免为追求码本使用率而损害
最终CT重建质量。
