# LightningRecon：固定稀疏视角快速 sCT 雕刻

本版本不再要求稀疏投影先预测 codebook。模型面向如下放疗流程：治疗前先完成一次
完整 CBCT 扫描和配准，得到当前患者的粗体积；之后用固定 6/8/10 个真实投影视角，
快速更新患者特异性的 HU 风格体积。

## 整体框架

```text
Stage 1（单独训练、结果可控）
完整 FDK-CBCT A ──轻量3D残差U-Net──> HU风格sCT基底 B
                         监督：配对pCT

Stage 2（冻结Stage 1）
真实稀疏投影 + 每张投影角度 ──2D CNN──> 多视角特征
                                      │
                              32³快速特征反投影
                                      │
B ──轻量3D编码────────────────────────┤
                                      ↓
                            投影证据门控残差
                                      ↓
                         final = B + gate × residual
```

Stage 1负责HU风格和低频主体，Stage 2只在投影证据支持的位置修改B。这样能够分别
检查“CBCT到sCT转换是否可靠”和“6-view究竟带来了多少额外可利用证据”，不会把两个
问题藏进一个不可控的端到端网络。

## 数据格式

默认依次查找：

1. 项目内 `data/thorax_fast`；
2. 当前目录下 `data/thorax_fast`；
3. `~/autodl-tmp/thorax`。

也可以用 `--data_root` 显式指定。需要以下文件：

```text
thorax_fast/
├── splits.json（也兼容meta_info.json）
└── processed/
    ├── images/
    │   ├── cbct/{case}.nii.gz   # 归一化uint8-like，256³
    │   └── ct/{case}.nii.gz     # 配对pCT，归一化uint8-like，256³
    └── projections/{case}.pickle
        ├── projs                # [K,H,W] uint8真实投影
        ├── angles               # [K] 弧度，与projs逐张对应
        └── projs_max            # 恢复对数衰减值的尺度
```

每个病例的K可以不同，不要求491。加载器会去除重复的周期端点，并从该病例全部有效
视角中建立固定均匀基准集。

## 视角课程

内置映射如下：

```text
final_view=6 : 60 → 54 → 48 → 36 → 24 → 12 → 6
final_view=8 : 64 → 56 → 48 → 32 → 24 → 16 → 8
final_view=10: 60 → 50 → 40 → 30 → 20 → 10
```

以6-view为例：先从病例全部K个有效方向中均匀选60个；固定的等间隔6-view是这60个
视角中的锚点。54/48/...训练阶段保留全部6个锚点，再从剩余位置随机补足。因此最终
部署方向从课程开始就一直参与训练。验证和推理只使用确定性的等间隔final_view。

## 安装

```bash
conda activate deeplearning
pip install -r requirements.txt
```

## Stage 1：训练CBCT到sCT基底

```bash
python -m src.train_stage1_sct \
  --data_root ~/autodl-tmp/thorax \
  --run_version v5 \
  --epochs 150 \
  --lr 2e-4 \
  --base_channels 4 \
  --volume_size 256 256 256
```

主要参数：

- `--data_root`：数据集根目录；省略时自动查找。
- `--run_version`：实验版本，日志目录为`logs/stage1_sct_v5`。
- `--epochs`：Stage 1训练轮数。
- `--base_channels`：高分辨率层通道数；默认4是24GB显存的保守设置。
- `--laplacian_weight`：多尺度高频损失系数，默认0.10。
- `--structural_weight`：三方向梯度结构损失系数，默认0.10。
- `--resume`：载入`stage1_last.pth`并恢复优化器、调度器和AMP状态。

根据验证集PSNR保存`stage1_best.pth`，每轮覆盖保存完整`stage1_last.pth`。测试集不
参与checkpoint选择。

Stage 1日志目录为：

```text
logs/stage1_sct_{run_version}
```

例如`--run_version v5`时启动TensorBoard：

```bash
tensorboard --logdir logs/stage1_sct_v5 --port 6006
```

Stage 1记录`Train`和`Val`两套指标：总损失、image/laplacian/structural/residual
四项raw loss、四项weighted loss、完整体积PSNR和学习率。`raw`表示损失函数原值，
`weighted`表示乘以系数后对total loss的实际贡献。

## Stage 2：训练投影证据雕刻器

```bash
python -m src.train_stage2_sculptor \
  --data_root ~/autodl-tmp/thorax \
  --stage1_checkpoint logs/stage1_sct_v5/stage1_best.pth \
  --run_version v5 \
  --final_view 6 \
  --epochs_per_view 30 \
  --final_epochs 60 \
  --projection_size 128 128 \
  --volume_size 256 256 256
```

主要参数：

- `--stage1_checkpoint`：已收敛的Stage 1模型，Stage 2中完全冻结。
- `--final_view`：部署和验证使用的固定视角数，可选6、8、10。
- `--view_schedule`：可覆盖内置课程，例如`60,48,24,12,6`。
- `--epochs_per_view`：每个课程视角数训练多少轮，默认30。
- `--final_epochs`：到达最终协议后额外巩固轮数，默认60。
- `--projection_size`：2D CNN输入尺寸。原始投影只在选中后才缩放。
- `--projection_channels`：二维投影特征通道数，默认24。
- `--evidence_size`：快速特征反投影网格，默认32³。
- `--structural_weight`：最终三维梯度损失，默认0.15。

日志目录示例为`logs/stage2_sculpt_finalview=6_v5`。验证始终使用6-view，不随训练
阶段变化。TensorBoard启动命令：

```bash
tensorboard --logdir logs/stage2_sculpt_finalview=6_v5 --port 6006
```

Stage 2记录：

- `Loss/total`：总损失；
- `Loss/raw/*`：image、laplacian、structural、preserve、gate、residual原值；
- `Loss/weighted/*`：上述六项乘系数后的贡献；
- `Metrics/PSNR_base`：冻结Stage 1基底B的PSNR；
- `Metrics/PSNR_final`：投影雕刻后最终sCT的PSNR；
- `Diagnostics/gate_mean`：体积内平均门控强度；
- `Protocol/views`：当前训练课程使用的视角数；
- `Optimizer/LR`：当前学习率。

同时查看Stage 1和Stage 2可以直接指向总日志目录：

```bash
tensorboard --logdir logs --port 6006
```

## 推理

```bash
python -m src.infer_fast_sculpt \
  --data_root ~/autodl-tmp/thorax \
  --stage1_checkpoint logs/stage1_sct_v5/stage1_best.pth \
  --stage2_checkpoint logs/stage2_sculpt_finalview=6_v5/stage2_best.pth \
  --final_view 6 \
  --split test \
  --output_dir outputs/finalview6
```

输出NIfTI保持输入CBCT的affine，目前体素值为网络使用的`[0,1]`归一化值。若用于
剂量计算，必须按照制作`processed/images`时使用的同一HU裁剪范围反归一化，不能
凭空假设`[0,1]`与HU的映射。

## 当前限制

1. 当前回顾性数据中的6-view来自生成完整CBCT的同一次采集。它适合验证原始投影
   是否补偿FDK/Stage 1丢失的细节，但不能代替真正的纵向在线6-view临床验证。
2. pickle目前只有角度，没有SAD、SDD、探测器像素间距和主点。代码因此在32³特征
   空间采用平行束近似反投影。获得厂家完整几何后，应只替换
   `FastFeatureBackprojector`，其余训练流程无需改变。
3. `evidence_gate`目前由配对CT差异提供软监督。它表示“哪里值得修改”，不能直接
   当成严格的统计不确定度。
4. 训练标签必须是同一患者且已经配准的CBCT/pCT。剩余配准误差会被网络错误学习
   为需要雕刻的结构变化。

## 建议消融

至少比较：A（CBCT）、B（Stage 1）、B+6-view但无角度、B+6-view+角度、完整门控
模型；再加入角度打乱和跨患者投影替换实验。核心结果是完整模型相对B的提升，这才
能直接证明原始6-view带来的可利用信息增益。
