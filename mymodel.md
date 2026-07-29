# LightningRecon 双域解剖先验重建模型

## 1. 模型目标

当前模型面向 Varian Halcyon Thorax Fast 的 6/8/10-view 稀疏投影重建。
核心思路不是让网络直接从少量投影“凭空生成”完整 CT，而是：

```text
真实 CT 学习解剖与组织边缘 codebook
                     ↓
稀疏投影预测当前病例的 CT latent/token 排列
                     ↓
codebook 解码为 128³ 粗糙基础体积
                     ↓
稀疏投影连续特征在基础体积上进行 residual/gate 雕刻
                     ↓
生成最终 256³ CT
                     ↓
冻结的体素→投影模型重新生成实测角度投影，形成双域闭环
```

旧的 `SparseViewReconstruction`、旧 Dataset、旧训练/推理入口已经删除。
旧 checkpoint 与新模型结构不兼容，不能 resume 或直接推理。

## 2. Thorax Fast 数据格式

数据根目录为 `data/thorax_fast`：

```text
data/thorax_fast/
├── README.md
├── config.yaml
├── splits.json
├── meta_info.json
└── processed/
    ├── images/
    │   ├── ct/{case}.nii.gz
    │   └── cbct/{case}.nii.gz
    ├── projections/{case}.pickle
    └── overlap/
```

### 2.1 CT/CBCT

| 项目 | 数值 |
|---|---|
| 空间矩阵 | 256×256×256 |
| spacing | 2.0×2.0×2.0 mm |
| 存储格式 | uint8 NIfTI |
| 存储范围 | [0,255] |
| 对应 HU | [-1000,1000] |
| 模型范围 | [0,1] |

`processed/images/ct` 是配准后的干净 pCT，作为主重建和解剖先验目标；
`processed/images/cbct` 与真实 XIM 扫描对应，更适合训练体素到投影模型。

### 2.2 投影

pickle 内容：

```python
{
    "projs": uint8[K, 320, 1280],
    "projs_max": float,
    "angles": float32[K],  # 弧度
}
```

正确恢复对数衰减值：

```python
attenuation = projs.astype(float32) / 255.0 * projs_max
```

模型固定使用 `[0,10]` 投影窗并映射到 `[-1,1]`：

```python
normalized = clip(attenuation / 10.0, 0.0, 1.0) * 2.0 - 1.0
```

不能再额外除以 `0.2`。该旧处理会使真实样本约 67.7% 像素饱和。

多数491帧样本的角度同时包含 `-π` 与 `+π`。两者是同一物理方向，
新加载器默认去掉最后一个周期端点：

```text
491存储帧 → 490有效方向 → 60/64-view均匀基准集
```

### 2.3 当前完整配对病例

| split | CT+CBCT+投影 |
|---|---:|
| train | 116 |
| val/eval | 26 |
| test | 26 |

原始 CT/CBCT 共184例，投影文件168例。加载器会明确报告并跳过缺少所需
文件的病例，不会填充投影或重新随机划分。

## 3. 数据加载

新加载器：

[src/thorax_fast_dataset.py](src/thorax_fast_dataset.py)

### 3.1 主重建训练

```python
from src.thorax_fast_dataset import ThoraxFastDataset

dataset = ThoraxFastDataset(
    data_root="data/thorax_fast",
    split="train",
    volume_keys=("ct",),
    projection_views=None,
    final_view=6,
    projection_size=(128, 128),
)
```

`projection_views=None` 时自动使用内置基准集：

| final_view | 基准集 |
|---:|---:|
| 6 | 60 |
| 8 | 64 |
| 10 | 60 |

### 3.2 CT codebook 预训练

```python
dataset = ThoraxFastDataset(
    data_root="data/thorax_fast",
    split="train",
    volume_keys=("ct",),
    require_projections=False,
)
```

### 3.3 体素到投影模型训练

```python
dataset = ThoraxFastDataset(
    data_root="data/thorax_fast",
    split="train",
    volume_keys=("cbct",),
    projection_views=6,
)
```

## 4. 模型模块

代码位于 `src/dual_domain/`：

```text
src/
├── ema_codebook.py
├── losses.py
├── view_protocol.py
├── thorax_fast_dataset.py
└── dual_domain/
    ├── __init__.py
    ├── blocks.py
    ├── anatomy_prior.py
    ├── projection_encoder.py
    ├── refiner.py
    ├── forward_projector.py
    ├── losses.py
    ├── model.py
    └── README.md
```

### 4.1 CT 解剖先验

文件：`anatomy_prior.py`

```text
CT 256³
  ↓ stride=2
Boundary latent 128³，32通道
  ↓ stride=2
Anatomy latent 64³，64通道
  ↓
Anatomy codebook：512×64
Boundary codebook：256×32
  ↓
Prior Decoder
  ↓
x_base 128³
```

codebook 直接在真实 CT 域训练，因此比从投影特征学习更接近：

- 空气、肺、软组织和骨的局部模式；
- 不同 HU 组织之间的边缘；
- 胸廓、肺部和脊柱的多尺度解剖组合。

codebook 是局部“词汇表”；三维 latent/token map 决定这些词在当前病例中的
排列方式。

### 4.2 稀疏投影编码

文件：`projection_encoder.py`

```text
[B,V,1,128,128] 投影 + 弧度角
          ↓
投影、sinθ、cosθ 三通道
          ↓
共享二维 CNN
          ↓
4层跨视角 Transformer
          ↓
Learned Volume Lift：16³
          ↓
32³ → 64³ → 128³
          ↓
预测 Anatomy latent 和 Boundary latent
```

当前提升模块明确称为 learned lift，不声称是精确 cone-beam 反投影。

### 4.3 基础体积雕刻

文件：`refiner.py`

输入：

- `base_volume`：codebook 解码的 128³ 基础体积；
- `prior_features`：CT 解剖先验特征；
- `projection_features`：当前病例未量化的连续投影特征。

输出：

- `residual_logits`：投影要求的病例特异修改；
- `gate`：每个区域相信 residual 的程度；
- `final_volume`：最终 256³ CT。

计算：

```python
base_logits = logit(clamp(x_base))
refined_logits = base_logits + gate * residual_logits
final_volume = sigmoid(upsample(refined_logits, 256))
```

最终输出严格位于 `[0,1]`。大量三维卷积停留在128³，256³只处理单通道
logits，以控制4090D显存。

### 4.4 体素到投影模型

文件：`forward_projector.py`

```text
CBCT体素 + 真实角度
       ↓
可微旋转与射线方向积分
       ↓
近似投影
       ↓
角度和Halcyon几何条件的2D修正网络
       ↓
真实预处理投影域[-1,1]
```

当前使用内存受控的平行束近似，不是精确 Halcyon cone-beam projector。
已保存的固定几何条件：

| 参数 | 数值 |
|---|---:|
| DSD | 1540 mm |
| DSO | 1000 mm |
| detector | 1280×320 |
| detector spacing | 0.336×1.344 mm |
| voxel spacing | 2×2×2 mm |

获得逐帧完整投影矩阵后，只需替换 `_analytic_projection()`。

## 5. 训练顺序

### Phase A：CT-VQ 解剖先验

训练：

- `HierarchicalAnatomyPrior`；
- `AnatomyPriorLoss`；
- Anatomy/Boundary EMA codebook。

损失：

```text
1.00 × 基础体积 Charbonnier
0.05 × Laplacian
0.10 × Structural
0.05 × VQ
```

训练入口：

```bash
python -m src.train_phase_a_prior \
  --data_root data/thorax_fast \
  --run_version v3 \
  --epochs 200 \
  --batch_size 1 \
  --grad_accum 4 \
  --amp
```

日志目录为 `logs/thorax_phaseA_prior_v3/tensorboard`，checkpoint 为：

```text
logs/thorax_phaseA_prior_v3/phase_A_best_v3.pth
logs/thorax_phaseA_prior_v3/phase_A_last_v3.pth
logs/thorax_phaseA_prior_v3/phase_A_epoch=XXXX_v3.pth
```

### Phase B：体素到投影模型

使用真实 CBCT、真实角度和真实投影单独训练：

- `LearnedForwardProjector`；
- `ForwardProjectorLoss`。

第一版损失：

```text
1.00 × 投影 Charbonnier
0.10 × 二维梯度损失
```

投影器必须在独立验证集上合格后冻结。

训练入口：

```bash
python -m src.train_phase_b_projector \
  --data_root data/thorax_fast \
  --run_version v3 \
  --epochs 150 \
  --views_per_case 6 \
  --projection_size 128 128 \
  --integration_size 96 \
  --batch_size 1 \
  --grad_accum 4 \
  --amp
```

训练集每次读取随机选择真实角度，验证集固定为均匀角度。日志目录为
`logs/thorax_phaseB_projector_v3/tensorboard`，最佳模型为
`logs/thorax_phaseB_projector_v3/phase_B_best_v3.pth`。

### Phase C：主重建模型

创建 `DualDomainReconstructionModel`，加载 Phase A 的先验并调用：

```python
model.freeze_pretrained_prior()
```

主训练阶段：

```text
Stage 1：60/64 views，高视角 latent 对齐和基础体积学习
Stage 2：从统一基准集逐级降低至 final_view
Stage 3：固定厂家 final_view 协议微调
```

主损失包含：

- 最终 CT Charbonnier；
- Laplacian；
- Structural；
- 基础体积监督；
- CT latent 蒸馏；
- residual 正则；
- 输入视角投影闭环；
- 随机隐藏视角投影闭环。

投影器参数应冻结，但其前向不能放在 `torch.no_grad()` 中，否则投影损失
无法穿过投影器更新重建体积。

训练入口（以最终 6-view、v3 为例）：

```bash
python -m src.train_phase_c_reconstruction \
  --data_root data/thorax_fast \
  --phase_a_checkpoint logs/thorax_phaseA_prior_v3/phase_A_best_v3.pth \
  --phase_b_checkpoint logs/thorax_phaseB_projector_v3/phase_B_best_v3.pth \
  --run_version v3 \
  --final_view 6 \
  --stage1_epochs 150 \
  --stage2_epochs_per_view 30 \
  --stage3_epochs 100 \
  --projection_size 128 128 \
  --batch_size 1 \
  --grad_accum 4 \
  --amp
```

默认内置确定性课程：

```text
final_view=6 ：60→54→48→36→24→12→6
final_view=8 ：64→56→48→32→24→16→8
final_view=10：60→50→40→30→20→10
```

可通过 `--view_schedule 60,48,24,12,6` 覆盖，但必须严格递减并以
`final_view` 结束。Phase C 的验证和最终测试始终使用固定的
`final_view` 均匀协议；test 不参与最佳 checkpoint 选择，只在训练结束后
加载 best checkpoint 评估一次。

Phase C 日志目录及 checkpoint：

```text
logs/thorax_phaseC_finalview=6_v3/tensorboard
logs/thorax_phaseC_finalview=6_v3/phase_C_best_v3.pth
logs/thorax_phaseC_finalview=6_v3/phase_C_last_v3.pth
logs/thorax_phaseC_finalview=6_v3/phase_C_epoch=XXXX_v3.pth
```

完整断点续训使用相应入口的 `--resume`，并保持其它结构和课程参数不变：

```bash
python -m src.train_phase_c_reconstruction \
  --data_root data/thorax_fast \
  --phase_a_checkpoint logs/thorax_phaseA_prior_v3/phase_A_best_v3.pth \
  --phase_b_checkpoint logs/thorax_phaseB_projector_v3/phase_B_best_v3.pth \
  --run_version v3 \
  --final_view 6 \
  --resume logs/thorax_phaseC_finalview=6_v3/phase_C_last_v3.pth
```

同时查看同一版本的三个 TensorBoard：

```bash
tensorboard --logdir_spec \
phaseA_v3:logs/thorax_phaseA_prior_v3/tensorboard,phaseB_v3:logs/thorax_phaseB_projector_v3/tensorboard,phaseC_6view_v3:logs/thorax_phaseC_finalview=6_v3/tensorboard \
--port 6006
```

## 6. 推理

常规推理只需要主重建模型：

```python
from src.dual_domain import DualDomainReconstructionModel

model = DualDomainReconstructionModel()
model.load_state_dict(checkpoint["model_state"])
model.eval()

with torch.no_grad():
    outputs = model(projections, angles)

base_128 = outputs["base_volume"]
residual = outputs["residual_logits"]
gate = outputs["gate"]
volume_256 = outputs["final_volume"]
```

常规推理不需要 `LearnedForwardProjector`。只有在线计算投影一致性质量指标时，
才额外运行：

```python
reprojected = frozen_projector(volume_256, angles)
```

## 7. 输出与诊断

三个入口记录：

```text
Train/Loss/*
Train/LearningRate/*
Codebook/anatomy_normalized_perplexity
Codebook/anatomy_batch_active_fraction
Codebook/anatomy_ema_active_fraction
Codebook/boundary_normalized_perplexity
Codebook/boundary_batch_active_fraction
Codebook/boundary_ema_active_fraction
Val/base_psnr
Val/final_psnr
Val/final_ssim
Sculptor/residual_abs_mean
Sculptor/gate_mean
Val/input_cycle
Val/heldout_cycle
```

图像诊断应同时显示相同病例、相同切片的：

```text
真实CT
基础体积
最终体积
最终误差
residual
gate
```

## 8. 当前状态

已完成：

- 真实 Thorax Fast 格式核验；
- 新数据加载器；
- CT 分层 codebook；
- 稀疏投影编码器；
- 基础体积 Decoder；
- residual/gate 雕刻器；
- 可单独训练的体素到投影模型；
- 三类训练损失；
- Phase A/B/C 三个独立训练入口；
- val 选择 best、test 最终单次评估、best/last/周期 checkpoint；
- 完整 optimizer/scheduler/scaler/RNG resume；
- 轻量端到端前向、反向和真实样本测试。

验证结果：

```text
主重建模型参数：1,136,235
前向投影器参数：48,833
单元测试：3项新架构测试全部通过
```

旧式单文件 `train.py`/`inference.py` 已删除。新架构只使用
`train_phase_a_prior.py`、`train_phase_b_projector.py` 和
`train_phase_c_reconstruction.py`；旧 checkpoint 不兼容。
