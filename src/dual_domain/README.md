# 双域解剖先验重建模型

该目录实现以下训练与推理链路：

```text
真实 CT → CT Encoder → 分层 codebook → Prior Decoder → 粗糙基础体积

稀疏投影 + 角度
    → Projection Encoder
    → 预测 CT latent
    → 冻结 codebook
    → 基础体积
    → Residual Sculptor
    → 最终 256³ CT

最终 CT + 真实角度
    → 单独预训练并冻结的 Forward Projector
    → 重建投影
    → 投影域循环一致性
```

## 模型整体框架

模型不是把所有任务放进一个网络同时从头训练，而是把“CT解剖先验”、
“体素到实测投影”和“稀疏投影重建”拆成三个可独立验证的阶段：

```text
                           Phase A（先训练）
CT 256³ ──→ CT Anatomy Encoder ──→ Anatomy latent 64³×64
                         │          Boundary latent 128³×32
                         ↓
                  EMA Anatomy/Boundary codebook
                         ↓
                   Prior Decoder ──→ x_base 128³
                         │
                         └──────────────┐
                                        │ 载入到Phase C
                           Phase B      │
CBCT或配准CT 256³ + angle ──→ Frozen-ready Forward Projector
                                      ↓ │
                              实测投影 128²
                                        │ 载入并冻结到Phase C
                                        ↓
                           Phase C（主重建）
稀疏实测投影 + angle ──→ Projection Encoder（4层跨视角Transformer）
                                  ↓
                       Anatomy/Boundary latent
                                  ↓
                    冻结的双层EMA codebook量化
                                  ↓
                    Prior Decoder ──→ x_base 128³
                                  ↓
             Residual Sculptor（128³语义融合→256³×8轻量细化）
                                  ↓
                         final CT 256³，[0,1]
                                  ↓
                冻结Forward Projector──→投影闭环弱约束
```

Phase C 推理只需要主重建模型。Phase A 的真实CT教师分支和 Phase B 投影器
只在训练时提供先验、latent监督和投影一致性，不增加部署时的必需输入。

## 三个阶段的训练流程与模块状态

### Phase A：CT解剖先验预训练

```text
输入：归一化CT 256³
  → CT Encoder提取64³ Anatomy latent和128³ Boundary latent
  → 正式训练前从多个batch收集特征并完成K-means码本初始化
  → 双层EMA codebook量化
  → Prior Decoder重建128³基础体积并预测Boundary edge
输出：x_base 128³、两层CT域码本和可复用Prior Decoder
验证：x_base与下采样到128³的CT标签计算PSNR/SSIM和损失
```

训练状态：

| 模块 | Phase A状态 | 更新方式 |
|---|---|---|
| CT Anatomy Encoder | 训练 | 反向传播 |
| Anatomy Transformer | 训练 | 反向传播 |
| Anatomy/Boundary量化适配卷积 | 训练 | 反向传播 |
| 两层codebook码字与命中统计 | 训练 | K-means初始化后使用EMA更新，不使用梯度更新码字 |
| Prior Decoder | 训练 | 反向传播 |
| Boundary edge head | 训练 | 边缘平衡辅助损失 |
| Projection Encoder/Sculptor/Forward Projector | 不创建 | 不参与Phase A |

### Phase B：体素到实测投影模型预训练

```text
输入：CBCT 256³（默认）或配准CT 256³ + 真实角度
  → 可微旋转/射线积分近似
  → 角度与Halcyon几何条件的二维修正网络
输出：与实测XIM相同预处理域的投影[-1,1]
监督：真实投影的Charbonnier强度损失 + 二维梯度损失
```

Phase B 的 B1/B2 训练相同的 `LearnedForwardProjector` 参数，不冻结其中某个
子网络；区别只在损失权重与学习率：

| 子阶段 | 默认epoch | 训练模块 | 目标 |
|---|---:|---|---|
| B1 | 1～150 | 完整Forward Projector可训练部分 | 先拟合投影整体强度，梯度系数0.10 |
| B2 | 151～250 | 完整Forward Projector可训练部分 | 低学习率边缘微调，梯度系数平滑升至0.25 |

几何常量和解析积分没有需要优化的参数；真正被优化的是可学习投影修正部分。
`--volume_source ct`是独立消融实验，不应覆盖CBCT基线。

### Phase C：双域主重建训练

```text
输入：稀疏实测投影、真实角度；训练时额外读取配对CT标签
  → Projection Encoder预测连续Anatomy/Boundary latent
  → 查询冻结的CT域codebook，用最近码字替换连续latent
  → Prior Decoder生成病例对应的x_base，不是解码全部码字
  → Sculptor结合x_base、prior feature和投影feature进行病例特异雕刻
  → 输出final CT 256³
  → 冻结投影器将final CT重投影，形成约5%的弱投影闭环约束
```

Phase C 的冻结/训练策略：

| 模块 | Stage 1：高视角 | Stage 2：逐级稀疏 | Stage 3：固定最终协议 |
|---|---|---|---|
| Phase A CT教师Encoder | 冻结，仅生成蒸馏目标 | 冻结 | 冻结 |
| Anatomy/Boundary EMA码本及适配层 | 冻结，仅查询 | 冻结，仅查询 | 冻结，仅查询 |
| Prior Decoder | 训练，LR=5e-5 | 训练，LR=2e-5 | 冻结 |
| Boundary edge head | 冻结且不执行 | 冻结且不执行 | 冻结且不执行 |
| Projection Encoder | 训练，LR=1e-4 | 训练，LR=5e-5 | 微调，LR=1e-5 |
| Residual Sculptor | 训练，LR=1e-4 | 训练，LR=5e-5 | 微调，LR=2e-5 |
| Phase B Forward Projector参数 | 冻结 | 冻结 | 冻结 |

“Forward Projector参数冻结”不等于对其前向使用`no_grad`。投影器权重不更新，
但投影损失的梯度仍穿过投影器回到`final_volume`，从而约束主重建网络。
Stage 1使用60或64个基准视角；Stage 2从固定基准网格中随机选择指定数量的
子集逐级降到`final_view`；Stage 3固定最终厂家协议。验证与测试始终使用
确定性的`final_view`均匀子集。

## 文件职责

| 文件 | 作用 |
|---|---|
| `blocks.py` | GroupNorm基础块、角度embedding和Depthwise 3D高分辨率块 |
| `anatomy_prior.py` | 全局Anatomy Transformer、Boundary CNN、双层EMA codebook和先验解码器 |
| `projection_encoder.py` | 稀疏投影编码、4层跨视角 Transformer、三维 latent 预测 |
| `refiner.py` | 128³融合后以8通道Depthwise 3D块细化256³ residual/gate |
| `forward_projector.py` | 可单独训练的体素→投影近似模型 |
| `losses.py` | CT 先验、主重建、前向投影器三类损失 |
| `model.py` | 常规推理使用的主重建模型 |

数据加载位于 `src/thorax_fast_dataset.py`。
当 `projection_views=None` 时，加载器会根据 `final_view` 自动选择 60-view
（最终6/10 views）或64-view（最终8 views）基准集。

训练入口省略 `--data_root` 时会依次检查项目内 `data/thorax_fast` 和
`~/autodl-tmp/thorax`；显式传入 `--data_root` 时始终使用指定目录。

## 推荐训练顺序

```bash
# Phase A：CT 解剖先验
python -m src.train_phase_a_prior \
  --run_version v4 \
  --boundary_residual_blocks 3 \
  --anatomy_transformer_layers 2 \
  --anatomy_transformer_heads 4 \
  --anatomy_context_size 8 \
  --boundary_context_channels 8 \
  --boundary_edge_weight 0.05 \
  --amp

# Phase B：CBCT 体素到真实投影（基线）
python -m src.train_phase_b_projector \
  --run_version v3 \
  --volume_source cbct \
  --epochs 250 --base_epochs 150 \
  --gradient_weight 0.10 --edge_gradient_weight 0.25 \
  --edge_gradient_ramp_epochs 25 \
  --edge_lr 2e-5 --min_lr 1e-6 \
  --amp

# Phase B：配准CT到真实投影（独立消融，不覆盖CBCT基线）
python -m src.train_phase_b_projector \
  --run_version v4 \
  --volume_source ct \
  --epochs 250 --base_epochs 150 \
  --amp

# Phase C：双域主重建
python -m src.train_phase_c_reconstruction \
  --phase_a_checkpoint logs/thorax_phaseA_prior_v4/phase_A_best_v4.pth \
  --phase_b_checkpoint logs/thorax_phaseB_projector_v3/phase_B_best_composite_v3.pth \
  --run_version v4 --final_view 6 --highres_channels 8 --amp
```

## 训练命令参数说明

以下表格以代码中的当前默认值为准。布尔参数`--amp`、`--compute_ssim`和
`--checkpoint_highres`当前默认开启；需要关闭时分别使用对应的`--no_*`参数。

### 通用运行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--data_root` | 自动查找 | Thorax Fast根目录；依次查找项目内`data/thorax_fast`和`~/autodl-tmp/thorax` |
| `--run_version` | 必填 | 运行版本，如`v4`；进入日志目录及checkpoint文件名，不能随意与旧实验混用 |
| `--log_dir` | `./logs` | TensorBoard、配置和checkpoint的根目录 |
| `--batch_size` | `1` | 单次前向的病例数；256³训练通常保持1 |
| `--grad_accum` | `4` | 梯度累计步数；有效batch约为`batch_size × grad_accum` |
| `--num_workers` | `2` | DataLoader并行读取进程数 |
| `--weight_decay` | `1e-5` | AdamW权重衰减 |
| `--eval_every` | `5` | 每多少个epoch执行一次验证 |
| `--save_every` | `25` | 每多少个epoch额外保存一次周期checkpoint；last仍每轮保存 |
| `--seed` | `42` | 数据采样与模型初始化随机种子 |
| `--deterministic` | 关闭 | 启用PyTorch确定性算法；可复现性更强，但可能降低速度 |
| `--amp` / `--no_amp` | 开启 | 开启/关闭CUDA自动混合精度 |
| `--compute_ssim` / `--no_ssim` | 开启 | Phase A/C验证时开启/关闭SSIM；关闭可缩短验证时间 |
| `--resume PATH` | 无 | 从相应Phase的last或epoch checkpoint完整恢复模型、优化器、AMP、RNG等状态 |

### Phase A 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--epochs` | `200` | Phase A总训练轮数 |
| `--lr` | `1e-4` | CT先验模型AdamW初始学习率 |
| `--anatomy_codebook_size` | `512` | Anatomy码本的码字数量 |
| `--boundary_codebook_size` | `256` | Boundary码本的码字数量 |
| `--anatomy_dim` | `64` | 每个Anatomy码字及latent向量维度 |
| `--boundary_dim` | `32` | 每个Boundary码字及latent向量维度 |
| `--base_channels` | `16` | CT Encoder第一层及Boundary局部主干通道数 |
| `--prior_feature_channels` | `32` | Prior Decoder在128³输出的特征通道数 |
| `--boundary_residual_blocks` | `3` | 128³ Boundary局部分支残差块数量 |
| `--anatomy_transformer_layers` | `2` | 8³全局Anatomy Transformer Encoder层数 |
| `--anatomy_transformer_heads` | `4` | 每个Anatomy Transformer层的注意力head数；必须整除`anatomy_dim` |
| `--anatomy_context_size` | `8` | 全局Anatomy注意力网格边长；默认最多产生`8³=512`个token |
| `--boundary_context_channels` | `8` | 从全局Anatomy回注入Boundary分支的上下文通道数 |
| `--boundary_edge_weight` | `0.05` | Boundary edge辅助损失在Phase A总损失中的系数 |
| `--kmeans_init_batches` | `8` | 两层码本初始化前累计特征的训练batch数，避免只用单病例初始化 |

### Phase B 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--epochs` | `250` | B1+B2总轮数 |
| `--base_epochs` | `150` | B1最后一个epoch；下一轮进入B2 |
| `--views_per_case` | `6` | 每病例每次训练/验证送入投影器的真实角度数量；训练随机选，验证均匀选 |
| `--volume_source` | `cbct` | 输入体积域；`cbct`为同次采集基线，`ct`为配准CT消融实验 |
| `--projection_size H W` | `128 128` | 将监督投影统一调整到的二维尺寸，也是投影器输出尺寸 |
| `--integration_size` | `96` | 可微解析投影内部旋转积分网格尺寸；越大越耗显存和计算 |
| `--correction_channels` | `32` | 二维可学习投影修正网络的基础通道数 |
| `--dsd` | `1540.0` | source-to-detector distance，单位mm，作为几何条件保存 |
| `--dso` | `1000.0` | source-to-origin distance，单位mm，作为几何条件保存 |
| `--gradient_weight` | `0.10` | B1二维梯度损失系数，也是B2平滑增强的起点 |
| `--edge_gradient_weight` | `0.25` | B2最终二维梯度损失系数 |
| `--edge_gradient_ramp_epochs` | `25` | B2开始后用half-cosine从起始梯度系数升至目标值的轮数 |
| `--selection_gradient_weight` | `0.20` | 固定checkpoint排序口径`Val/image + 该系数 × Val/gradient` |
| `--lr` | `1e-4` | B1起始学习率 |
| `--edge_lr` | `2e-5` | 进入B2时重启的学习率 |
| `--min_lr` | `1e-6` | B1和B2余弦衰减的最低学习率 |

Phase B没有SSIM参数，因为其输出是二维投影，主要使用投影PSNR、强度误差和
二维梯度误差评价。推荐交给Phase C的是`phase_B_best_composite_*.pth`。

### Phase C 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--phase_a_checkpoint` | 必填 | 已完成K-means初始化的Phase A先验checkpoint |
| `--phase_b_checkpoint` | 必填 | Phase B投影器checkpoint，推荐best composite |
| `--final_view` | `6` | 部署、验证和测试的最终固定视角数；可选`6/8/10` |
| `--view_schedule` | 内置映射 | 用逗号覆盖高到低课程，如`60,48,24,12,6`；必须严格递减并以`final_view`结束 |
| `--stage1_epochs` | `150` | 最高视角Stage 1训练轮数 |
| `--stage2_epochs_per_view` | `30` | Stage 2课程中每个视角数量训练的轮数，包括降到`final_view`的这一档 |
| `--stage3_epochs` | `100` | 固定最终厂家协议微调轮数；设为0可关闭Stage 3 |
| `--transformer_layers` | `4` | Projection Encoder跨视角Transformer层数；head固定为4 |
| `--refinement_channels` | `16` | Projection Encoder输出的128³病例细化特征通道数 |
| `--highres_channels` | `8` | Sculptor在最终256³分辨率保留的轻量特征通道数 |
| `--checkpoint_highres` / `--no_checkpoint_highres` | 开启 | 是否对256³细化块使用梯度检查点；开启可省显存但增加重计算 |
| `--projection_size H W` | `128 128` | Phase C输入投影尺寸，必须与Phase B投影器checkpoint兼容 |
| `--cycle_input_views` | `6` | 每病例最多选择多少个已输入视角计算重投影闭环 |
| `--heldout_cycle_views` | `6` | 从未输入但属于基准集的角度中最多抽多少个计算held-out闭环；无可用视角时为0项 |

内置视角课程为：

```text
final_view=6 ：60→54→48→36→24→12→6
final_view=8 ：64→56→48→32→24→16→8
final_view=10：60→50→40→30→20→10
```

总训练epoch由命令自动推导：

```text
total_epochs = stage1_epochs
             + (课程长度 - 1) × stage2_epochs_per_view
             + stage3_epochs
```

Phase A 的 Anatomy 分支在最多`8³=512`个token上使用2层、4-head全局
Transformer；Boundary保持128³，用3个同分辨率残差块提取局部边缘，再融合
8通道全局上下文，并通过权重0.05的边缘平衡辅助损失监督，防止稀疏边缘目标
退化为全零预测。Phase C 不再把
128³单通道logits直接插值为最终结果，而是在256³保留8通道并使用Depthwise
3D残差块细化；默认启用梯度检查点控制24GB显存。

该结构的Phase A checkpoint格式为v2，必须重新训练；旧Phase A/Phase C权重
不会被静默加载。Phase B结构未变化，已有的v3 composite checkpoint可以复用。

Phase B 的前150轮拟合整体投影强度；后100轮重启学习率并平滑增强二维梯度损失。
推荐将固定口径 `Val/composite = Val/image + 0.20 × Val/gradient` 最低的
`phase_B_best_composite_*.pth` 交给 Phase C。完整的恢复训练、日志和checkpoint
说明见项目根目录 `mymodel.md`。

Phase C 加载并冻结 Phase A 的教师 Encoder/codebook，加载并冻结 Phase B
投影器；投影器前向仍保留关于重建体积的梯度。训练按高视角、逐级稀疏、
固定最终协议三阶段进行。验证集选择 best checkpoint，test 只在结束时
评估一次。三个入口均保存 best、last、周期 checkpoint 和完整 resume 状态。

Phase C 的输入/隐藏投影闭环权重分别为 Stage 1 `0.004/0.002`、Stage 2
`0.0025/0.00125`、Stage 3 `0.002/0.001`，按既有raw loss量级使投影项合计
约占total的5%。实际占比由 `Train/Loss/diagnostic/projection_fraction` 和
`Val/projection_fraction`直接监控，主体仍是CT图像与结构监督。

`--volume_source ct`训练的Phase B会把配准残差、采集时相差异和实测噪声也
吸收到映射中；它与Phase C的CT风格输出更匹配，但不是理想无噪声物理投影器，
应与默认CBCT版本使用相同split和seed进行消融比较。

完整参数、目录命名、断点续训和 TensorBoard 命令见项目根目录
`mymodel.md`。

## 常规推理

```python
model.eval()
with torch.no_grad():
    outputs = model(projections, angles)
volume_256 = outputs["final_volume"]
```

常规推理不需要加载 `LearnedForwardProjector`。只有需要在线计算投影一致性
质量指标时才额外运行它。

## 当前前向投影器边界

`LearnedForwardProjector` 使用可微平行束旋转积分作为第一版几何主体，
DSO/DSD 和探测器参数作为修正网络条件。它不是精确的 Halcyon cone-beam
投影器。获得每帧完整投影矩阵后，应替换内部 `_analytic_projection()`，
模型其余接口和训练损失可以保持不变。
