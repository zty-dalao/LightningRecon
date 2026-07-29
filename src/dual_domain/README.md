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

## 文件职责

| 文件 | 作用 |
|---|---|
| `blocks.py` | GroupNorm 二维/三维基础块与角度 embedding |
| `anatomy_prior.py` | CT 教师编码器、双层 EMA codebook、基础体积解码器 |
| `projection_encoder.py` | 稀疏投影编码、4层跨视角 Transformer、三维 latent 预测 |
| `refiner.py` | 在基础体积上预测 residual 和 gate，轻量输出 256³ |
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
  --run_version v3 --amp

# Phase B：CBCT 体素到真实投影
python -m src.train_phase_b_projector \
  --run_version v3 --amp

# Phase C：双域主重建
python -m src.train_phase_c_reconstruction \
  --phase_a_checkpoint logs/thorax_phaseA_prior_v3/phase_A_best_v3.pth \
  --phase_b_checkpoint logs/thorax_phaseB_projector_v3/phase_B_best_v3.pth \
  --run_version v3 --final_view 6 --amp
```

Phase C 加载并冻结 Phase A 的教师 Encoder/codebook，加载并冻结 Phase B
投影器；投影器前向仍保留关于重建体积的梯度。训练按高视角、逐级稀疏、
固定最终协议三阶段进行。验证集选择 best checkpoint，test 只在结束时
评估一次。三个入口均保存 best、last、周期 checkpoint 和完整 resume 状态。

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
