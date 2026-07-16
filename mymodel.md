基于你的设计（可训练低频基座 + 高频Codebook补全 + 3D U-Net体素输出 + 双监督），我为你梳理了完整的实施步骤。

----

整体架构概览

```text
训练阶段 (利用491个完整投影)：
  491张投影 → 2D CNN → 构建3D代价体 → 低频基座(FiLM调制) + 高频Codebook(残差) → 融合 → 3D U-Net → 256³体素
                                                                                                    ↓
                                                                                            双监督: L_CBCT + L_pCT

推理阶段 (仅6个稀疏投影)：
  6张投影 → 2D CNN → 构建稀疏3D代价体 → 低频基座(FiLM调制) + 高频Codebook(残差) → 融合 → 3D U-Net → 256³体素
`````

----

📦 第一步：数据准备与预处理

1.1 数据集结构

```text
dataset/
├── train/
│   ├── patient_001/
│   │   ├── projections/          # 491张投影图 (角度 0°~360°)
│   │   ├── ct_volume.nii.gz      # 规划CT (pCT) 256³
│   │   └── cbct_volume.nii.gz    # 全视角CBCT重建 256³
│   └── patient_002/
│       └── ...
├── test/
│   └── ...
└── meta_info.json
````

----

1.2 数据预处理

```python
# 1. 所有体素统一重采样到 256³，间距归一化 (如 1.5mm)
# 2. CT值裁剪到 [-1000, 1000] HU，归一化到 [0, 1]
# 3. 投影图归一化到 [0, 1]
# 4. 生成相机参数 (角度、旋转轴、探测器几何)
```

关键文件： ```bash meta_info.json```

```json
{
    "train": ["patient_001", "patient_002", ...],
    "test": ["patient_101", ...],
    "spacing": [1.5, 1.5, 1.5],
    "num_views": 491,
    "angles": [0, 0.733, 1.466, ...]  // 491个角度
}
```

----

🏗️ 第二步：模型架构设计

2.1 2D特征提取器 (Encoder_2D)

```python
class Encoder2D(nn.Module):
    """
    输入: [B, V, 1, 256, 256]  # V=491(训练) 或 6(推理)
    输出: [B, V, 64, 64, 64]   # 特征图
    """
    def __init__(self):
        super().__init__()
        # 共享权重的2D CNN (处理每个视角)
        self.cnn = nn.Sequential(
            Conv2D(1, 32, 3, stride=1),   # 256→256
            Conv2D(32, 64, 3, stride=2),  # 256→128
            Conv2D(64, 64, 3, stride=2),  # 128→64
            # ... 更多层
        )
    
    def forward(self, x):
        B, V, C, H, W = x.shape
        x = x.view(B*V, C, H, W)
        features = self.cnn(x)  # [B*V, 64, 64, 64]
        return features.view(B, V, 64, 64, 64)
```

2.2 3D代价体构建 (Cost Volume)

```python
def build_cost_volume(features_2d, camera_params, volume_shape=(128, 128, 128)):
    """
    features_2d: [B, V, 64, H_feat, W_feat]
    camera_params: 每个视角的内外参 [B, V, 3x4]
    
    输出: [B, 64, 128, 128, 128] 3D特征体
    """
    # 1. 定义3D网格坐标
    grid_3d = create_3d_grid(volume_shape)  # [128³, 3]
    
    # 2. 投影到每个2D特征图
    for v in range(V):
        # 3D点投影到2D坐标
        coords_2d = project_3d_to_2d(grid_3d, camera_params[v])  # [128³, 2]
        
        # 双线性插值采样
        sampled = grid_sample(features_2d[:, v, :, :, :], coords_2d)  # [B, 64, 128³]
        
        # 累加/方差聚合
        volume += sampled
    
    return variance(volume)  # 或 mean
```

2.3 低频基座 (Learnable Prior Base)

```python
class LowFreqBase(nn.Module):
    """
    可训练的通用解剖结构基座
    """
    def __init__(self):
        super().__init__()
        # 低频基座: 存储通用形状 (例如用数据集的平均CT初始化)
        self.base_feat = nn.Parameter(torch.randn(1, 32, 128, 128, 128))
        
        # 调制器: 从6张图的全局特征生成 scale 和 shift
        self.modulator = nn.Sequential(
            nn.Linear(6 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 2)  # 每个通道一个 scale 和 shift
        )
    
    def forward(self, global_feat):
        # global_feat: [B, 6*64] 从6个视角提取的全局特征
        mod_params = self.modulator(global_feat)  # [B, 64]
        scale, shift = mod_params.chunk(2, dim=1)  # [B, 32], [B, 32]
        
        # FiLM调制: 将通用基座雕琢为特定患者
        base = self.base_feat.expand(B, -1, -1, -1, -1)  # [B, 32, 128, 128, 128]
        base_modulated = base * scale.view(B, 32, 1, 1, 1) + shift.view(B, 32, 1, 1, 1)
        return base_modulated
```

2.4 高频Codebook (残差补全)

```python
class HighFreqCodebook(nn.Module):
    """
    补全稀疏视角无法覆盖的高频细节 (骨骼边缘、血管纹理)
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = nn.Conv3d(64, num_embeddings, kernel_size=1)
        self.decoder = nn.Conv3d(embedding_dim, 32, kernel_size=1)  # 输出32通道残差
    
    def forward(self, volume_raw):
        # volume_raw: [B, 64, 128, 128, 128] 从代价体直接提取的稀疏3D特征
        
        # 1. 编码为码本索引
        logits = self.encoder(volume_raw)  # [B, 1024, 128, 128, 128]
        indices = logits.argmax(dim=1)      # [B, 128, 128, 128]
        
        # 2. 查表
        z_q = self.codebook(indices)        # [B, 128, 128, 128, 64]
        z_q = z_q.permute(0, 4, 1, 2, 3)    # [B, 64, 128, 128, 128]
        
        # 3. STE直通 (反向传播时跳过量化)
        z_q = volume_raw + (z_q - volume_raw).detach()
        
        # 4. 解码为残差 (高频细节)
        residual = self.decoder(z_q)         # [B, 32, 128, 128, 128]
        return residual
```

2.5 3D U-Net (体素解码器)

```python
class UNet3D(nn.Module):
    """
    输入: [B, 64, 128, 128, 128] (低频32 + 高频32)
    输出: [B, 1, 256, 256, 256] (HU值体素)
    """
    def __init__(self):
        super().__init__()
        # 编码器 (下采样)
        self.enc1 = ConvBlock(64, 128)    # 128→64
        self.enc2 = ConvBlock(128, 256)   # 64→32
        self.enc3 = ConvBlock(256, 512)   # 32→16
        
        # 解码器 (上采样 + 跳跃连接)
        self.dec3 = ConvBlock(512+256, 256)  # 16→32
        self.dec2 = ConvBlock(256+128, 128)  # 32→64
        self.dec1 = ConvBlock(128+64, 64)    # 64→128
        
        # 输出头
        self.head = nn.Conv3d(64, 1, kernel_size=1)
        
        # 上采样层
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    
    def forward(self, x):
        # x: [B, 64, 128, 128, 128]
        e1 = self.enc1(x)   # [B, 128, 64, 64, 64]
        e2 = self.enc2(e1)  # [B, 256, 32, 32, 32]
        e3 = self.enc3(e2)  # [B, 512, 16, 16, 16]
        
        d3 = self.dec3(torch.cat([self.up(e3), e2], dim=1))  # [B, 256, 32, 32, 32]
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))  # [B, 128, 64, 64, 64]
        d1 = self.dec1(torch.cat([self.up(d2), x], dim=1))   # [B, 64, 128, 128, 128]
        
        # 上采样到 256³
        out = self.up(d1)    # [B, 64, 256, 256, 256]
        out = self.head(out) # [B, 1, 256, 256, 256]
        return out
```

2.6 完整模型 (组装所有组件)

```python
class SparseViewReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_2d = Encoder2D()
        self.low_freq_base = LowFreqBase()
        self.high_freq_codebook = HighFreqCodebook()
        self.unet_3d = UNet3D()
        
    def forward(self, projs, camera_params, is_train=True):
        # projs: [B, V, 1, 256, 256]  V=491(训练) 或 6(推理)
        
        # 1. 2D特征提取
        features_2d = self.encoder_2d(projs)  # [B, V, 64, 64, 64]
        
        # 2. 构建3D代价体 (只取前6个视图作为条件输入)
        V_cond = 6  # 始终用6个视图作为条件
        volume_raw = build_cost_volume(
            features_2d[:, :V_cond, :, :, :], 
            camera_params[:, :V_cond, :, :]
        )  # [B, 64, 128, 128, 128]
        
        # 3. 低频基座 (通过6个视图的全局特征调制)
        global_feat = features_2d[:, :V_cond, :, :, :].mean(dim=[2,3,4]).flatten(1)  # [B, 6*64]
        low_freq = self.low_freq_base(global_feat)  # [B, 32, 128, 128, 128]
        
        # 4. 高频Codebook残差
        high_freq = self.high_freq_codebook(volume_raw)  # [B, 32, 128, 128, 128]
        
        # 5. 融合特征
        fused_features = torch.cat([low_freq, high_freq], dim=1)  # [B, 64, 128, 128, 128]
        
        # 6. 3D U-Net解码
        volume_pred = self.unet_3d(fused_features)  # [B, 1, 256, 256, 256]
        
        return volume_pred, low_freq, high_freq, volume_raw
```

----

🎯 第三步：损失函数设计 (双监督)

3.1 解剖结构一致性损失 (CBCT监督)

```python
def cbct_consistency_loss(pred, cbct_gt):
    """
    确保生成的体素在解剖结构上与全视角CBCT一致
    """
    # 1. SSIM损失 (关注结构)
    ssim_loss = 1 - ssim(pred, cbct_gt, data_range=1.0)
    
    # 2. 感知损失 (使用预训练的3D CNN提取特征)
    percep_loss = perceptual_loss(pred, cbct_gt)
    
    # 3. 梯度损失 (边缘一致性)
    grad_loss = torch.mean(torch.abs(gradient(pred) - gradient(cbct_gt)))
    
    return ssim_loss + 0.1 * percep_loss + 0.05 * grad_loss
```

3.2 CT值准确性损失 (pCT监督)

```python
def pct_accuracy_loss(pred, pct_gt):
    """
    确保生成的体素在HU值分布上与规划CT接近
    """
    # 1. MSE损失 (数值精度)
    mse_loss = torch.mean((pred - pct_gt) ** 2)
    
    # 2. 直方图匹配损失 (全局分布一致性)
    hist_loss = histogram_matching_loss(pred, pct_gt)
    
    # 3. 感知损失 (在HU值域上的感知)
    percep_loss = perceptual_loss(pred, pct_gt)
    
    return mse_loss + 0.1 * hist_loss + 0.05 * percep_loss
```

3.3 总损失

```python
def total_loss(volume_pred, cbct_gt, pct_gt, low_freq, high_freq, volume_raw):
    # 1. 双监督损失
    L_cbct = cbct_consistency_loss(volume_pred, cbct_gt)
    L_pct = pct_accuracy_loss(volume_pred, pct_gt)
    
    # 2. 辅助正则化损失
    # 2a. 低频基座平滑性 (防止过拟合)
    L_smooth = total_variation(low_freq)
    
    # 2b. 高频Codebook使用均匀性 (perplexity)
    L_perplexity = codebook_perplexity(high_freq)
    
    # 2c. 稀疏代价体的稀疏性约束
    L_sparse = torch.norm(volume_raw, p=1)
    
    # 3. 总损失
    L_total = L_cbct + L_pct + 0.01 * L_smooth + 0.1 * L_perplexity + 0.001 * L_sparse
    return L_total
```

----

🚀 第四步：训练策略

4.1 训练循环 (伪代码)

```python
model = SparseViewReconstruction().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(num_epochs):
    for batch in dataloader_train:
        # batch包含: projs [B, 491, 1, 256, 256], camera_params [B, 491, 3, 4], 
        #           cbct_gt [B, 1, 256, 256, 256], pct_gt [B, 1, 256, 256, 256]
        
        # 前向传播 (只用前6个视角作为条件)
        volume_pred, low_freq, high_freq, volume_raw = model(
            projs[:, :6, :, :, :],  # 只取6个view
            camera_params[:, :6, :, :]
        )
        
        # 计算损失 (使用全视角的CBCT和pCT作为监督)
        loss = total_loss(volume_pred, cbct_gt, pct_gt, low_freq, high_freq, volume_raw)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```
4.2 渐进式训练策略

```python
# 阶段1 (0-50 epochs): 只使用CBCT监督 (让网络先学解剖结构)
L_total = L_cbct + 0.01 * L_smooth

# 阶段2 (50-100 epochs): 引入pCT监督 (开始校正CT值)
L_total = L_cbct + 0.5 * L_pct + 0.01 * L_smooth

# 阶段3 (100-300 epochs): 完整损失 (微调)
L_total = L_cbct + L_pct + 0.01 * L_smooth + 0.1 * L_perplexity

# 阶段4 (300+ epochs): 冻结低频基座，微调其余部分
low_freq_base.base_feat.requires_grad = False
```

----

🧪 第五步：推理流程

```python
def inference(model, projs_6view, camera_params_6view):
    """
    输入: 6个稀疏视图 (新病人)
    输出: 256³ 体素 (快速推理 < 1秒)
    """
    model.eval()
    with torch.no_grad():
        volume_pred, _, _, _ = model(projs_6view, camera_params_6view)
    return volume_pred  # [1, 1, 256, 256, 256]
```

----

📊 第六步：评估指标

```python
def evaluate(volume_pred, volume_gt):
    # 1. PSNR (峰值信噪比)
    psnr = compute_psnr(volume_pred, volume_gt)
    
    # 2. SSIM (结构相似性)
    ssim = compute_ssim(volume_pred, volume_gt)
    
    # 3. MAE (平均绝对误差) - 关注CT值准确性
    mae = torch.mean(torch.abs(volume_pred - volume_gt))
    
    # 4. HU值直方图相似度
    hist_sim = histogram_similarity(volume_pred, volume_gt)
    
    return {'PSNR': psnr, 'SSIM': ssim, 'MAE': mae, 'HistSim': hist_sim}
```

----

💡 第七步：关键超参数参考

|参数|推荐值|说明|
|----|----|----|
|Learning Rate|1e-4|AdamW优化器|
|Batch Size|2|32G显存|
|Codebook Size|1024|高频细节容量|
|Codebook Dim|64|特征维度|
|低频基座通道|32|64³和128³分别设|
|权重衰减|1e-5|防止过拟合|
|梯度裁剪|1.0|稳定训练|
|总训练轮数|300-500|视收敛情况|

----

📝 实施建议

1. 基座初始化: 用训练集中所有病人的平均CT体素初始化 base_feat，可以极大加速收敛。
2. 数据增强: 训练时随机裁剪、旋转、添加高斯噪声，提升泛化能力。
3. 分阶段训练: 严格按照上述渐进式策略，先学结构再学细节。
4. 监控指标: 每5个epoch在验证集上计算PSNR/SSIM，选最佳模型。
5. 显存优化: 如果32G不够，可以：
    - 使用梯度累积 (accumulation)
    - 减小batch size到1
    - 使用混合精度训练 (AMP)

这个方案从数据准备到推理部署形成了一个完整的闭环，且每个组件都有明确的输入输出定义。如果需要我展开某个模块的具体代码实现，随时告诉我。