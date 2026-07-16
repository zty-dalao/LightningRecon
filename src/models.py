"""
SparseViewReconstruction: 从稀疏 X 射线投影重建 CT 体素。

架构 (from mymodel.md):
  投影 → 2D CNN → 3D 特征体 → 低频基座 (FiLM) + 高频 Codebook (VQ) → 融合 → 3D U-Net → 体素

数据维度:
  输入投影: (B, V, 1, 256, 256)   — V=491 (训练) 或 V=6 (推理)
  输出体素: (B, 1, D, H, W)       — 由 vol_size 决定 (默认 128³)
"""

import torch                                                      # PyTorch 核心
import torch.nn as nn                                             # 神经网络模块
import torch.nn.functional as F                                   # 函数式 API (激活函数等)


# =========================================================================
# 2D 基础模块
# =========================================================================

class ConvBlock2D(nn.Module):
    """2D 卷积块: Conv2d → BatchNorm2d → ReLU。"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(                                    # 2D 卷积层
            in_ch, out_ch, kernel, stride, padding, bias=False     # bias=False 因为后面有 BN
        )
        self.bn = nn.BatchNorm2d(out_ch)                          # 2D 批归一化
        self.act = nn.ReLU(inplace=True)                          # ReLU 激活 (inplace 省内存)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))                    # Conv → BN → ReLU


# =========================================================================
# 3D 基础模块
# =========================================================================

class ConvBlock3D(nn.Module):
    """3D 卷积块: Conv3d → BatchNorm3d → ReLU。"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(                                    # 3D 卷积层
            in_ch, out_ch, kernel, stride, padding, bias=False     # bias=False 因为后面有 BN
        )
        self.bn = nn.BatchNorm3d(out_ch)                          # 3D 批归一化
        self.act = nn.ReLU(inplace=True)                          # ReLU 激活

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))                    # Conv → BN → ReLU


class ResBlock3D(nn.Module):
    """3D 残差块: 两个 3×3×3 卷积 + 跳跃连接。"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)  # 第一个 3×3×3 卷积
        self.bn1 = nn.BatchNorm3d(ch)                              # BN1
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)  # 第二个 3×3×3 卷积
        self.bn2 = nn.BatchNorm3d(ch)                              # BN2

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)), inplace=True)         # Conv1 → BN1 → ReLU
        r = self.bn2(self.conv2(r))                                # Conv2 → BN2 (不加激活)
        return F.relu(x + r, inplace=True)                        # 残差连接 + ReLU


# =========================================================================
# 2D 特征提取器 (Encoder_2D): 共享权重处理每个视角
# =========================================================================

class Encoder2D(nn.Module):
    """
    共享权重的 2D CNN，独立处理每个视角的投影图。

    输入:  (B, V, 1, H, W)   — V 个视角的投影图
    输出:  (B, V, C_out, H', W')  — 每个视角的 2D 特征图
    """
    def __init__(self, in_ch=1, base_ch=32, out_ch=64, n_down=1):
        """
        Args:
            in_ch:    输入通道数 (灰度投影=1)
            base_ch:  基础通道数
            out_ch:   输出通道数
            n_down:   下采样次数: 1→输出 128×128 (256输入), 2→输出 64×64
        """
        super().__init__()
        # ---- 构建卷积层序列 ----
        # 第一层: 大核提取纹理，不下采样
        layers = [
            ConvBlock2D(in_ch, base_ch, kernel=7, stride=1, padding=3),  # 保持分辨率，大核感受野
        ]
        # 后续层: 逐级下采样+通道翻倍
        ch = base_ch                                                     # 当前通道数
        for i in range(n_down):                                          # 每级: 下采样 + 通道翻倍
            layers += [
                ConvBlock2D(ch, ch * 2, kernel=3, stride=2),             # 分辨率减半
                ConvBlock2D(ch * 2, ch * 2, kernel=3, stride=1),         # 保持分辨率
            ]
            ch *= 2                                                      # 通道翻倍
        self.cnn = nn.Sequential(*layers)                                 # 打包为 Sequential
        self.final = nn.Conv2d(ch, out_ch, kernel_size=1)                 # 1×1 卷积投影到输出通道

    def forward(self, x):
        # x: (B, V, C, H, W) — 批量 × 视角 × 通道 × 高 × 宽
        B, V, C, H, W = x.shape                                          # 拆解各维度
        x = x.reshape(B * V, C, H, W)                                    # 合并 B 和 V 维度 → (B*V, C, H, W)
        feat = self.cnn(x)                                                # 2D CNN 提取特征 → (B*V, base_ch*2, H', W')
        feat = self.final(feat)                                           # 1×1 卷积降维 → (B*V, out_ch, H', W')
        _, C_out, H_out, W_out = feat.shape                               # 获取输出形状
        return feat.reshape(B, V, C_out, H_out, W_out)                    # 恢复 V 维度


# =========================================================================
# 可学习的 3D 特征体构建器 (替代几何代价体)
# =========================================================================

class FeatureVolumeBuilder(nn.Module):
    """
    将多视角 2D 特征聚合为 3D 特征体（无需相机参数）。

    原理:
      1. 跨视角均值池化 → 视角无关的 2D 特征
      2. 学习深度概率分布 (每个空间位置在深度维度的权重)
      3. 特征 × 深度概率 → 3D 特征体

    输入:  (B, V, C, H, W)   — 多视角 2D 特征
    输出:  (B, C_vol, D, H, W)  — 3D 特征体
    """
    def __init__(self, in_ch=64, vol_ch=64, vol_depth=128):
        super().__init__()
        self.vol_depth = vol_depth                                      # 体素深度维度大小

        # ---- 跨视角融合: 压缩 V 个视角的特征 ----
        self.view_fusion = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1),                # 1×1 卷积升维
            nn.BatchNorm2d(in_ch * 2),                                  # BN 归一化
            nn.ReLU(inplace=True),                                      # 非线性激活
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),                # 1×1 卷积降维回原通道
        )

        # ---- 深度提升: 从 2D 特征预测深度分布 ----
        self.depth_lift = nn.Sequential(
            nn.Conv2d(in_ch, vol_depth, kernel_size=3, padding=1),     # 3×3 卷积 → vol_depth 通道
            nn.BatchNorm2d(vol_depth),                                  # BN
            nn.ReLU(inplace=True),                                      # ReLU
            nn.Conv2d(vol_depth, vol_depth, kernel_size=3, padding=1), # 再一个 3×3 卷积
            nn.Softmax(dim=1),                                          # softmax 得到深度概率分布
        )

        # ---- 3D 精炼: 对构建的特征体做卷积 ----
        self.refine = nn.Sequential(
            ConvBlock3D(vol_ch, vol_ch),                                # 3D 卷积块
            ResBlock3D(vol_ch),                                         # 3D 残差块
        )

    def forward(self, features_2d):
        # features_2d: (B, V, C, H, W)
        B, V, C, H, W = features_2d.shape                               # 拆解形状

        # 1. 跨视角均值池化 → (B, C, H, W)
        fused = features_2d.mean(dim=1)                                 # 沿视角维度取均值

        # 2. 视角融合 → (B, C, H, W)
        fused = self.view_fusion(fused)                                 # 2D 卷积精炼

        # 3. 深度提升: 预测每个 (h,w) 位置的深度分布 → (B, vol_depth, H, W)
        depth_prob = self.depth_lift(fused)                             # softmax 后的深度概率

        # 4. 构造 3D 特征体: 特征 × 深度概率
        # fused: (B, C, H, W) → unsqueeze(2) → (B, C, 1, H, W)
        # depth_prob: (B, D, H, W) → unsqueeze(1) → (B, 1, D, H, W)
        # 广播相乘 → (B, C, D, H, W)
        volume = fused.unsqueeze(2) * depth_prob.unsqueeze(1)           # 外积构建 3D 体

        # 5. 3D 精炼 → (B, C, D, H, W)
        volume = self.refine(volume)                                    # 3D 卷积 + 残差

        return volume


# =========================================================================
# 低频基座 (Learnable Prior Base): 可训练的通用解剖结构模板
# =========================================================================

class LowFreqBase(nn.Module):
    """
    可训练的通用解剖结构基座，使用 FiLM 调制适配到具体患者。

    FiLM (Feature-wise Linear Modulation):
      对基座特征的每个通道做 scale + shift 变换，
      scale 和 shift 由多视角全局特征的统计量 (均值+标准差) 预测。

    视角聚合: 对 V 个视角的特征做均值+标准差池化 → 固定维度向量，
    因此支持任意数量的输入视角，训练时可随机变化视角数。

    输入:  多视角特征 (B, V, C)    — 每个视角一个 C 维特征向量
    输出:  调制后基座 (B, C_base, D, H, W)
    """
    def __init__(self, base_ch=32, vol_shape=(128, 128, 128), feat_ch=64):
        super().__init__()
        D, H, W = vol_shape                                             # 体素空间尺寸

        # ---- 可学习的通用基座 ----
        # 用高斯噪声初始化，训练中逐渐形成通用解剖结构先验
        self.base_feat = nn.Parameter(                                  # 可训练参数
            torch.randn(1, base_ch, D, H, W) * 0.02                     # 小方差初始化
        )

        # ---- FiLM 调制器 ----
        # 输入: 视角池化后的统计量 (均值 + 标准差) → 2*feat_ch 维
        # 输出: (scale, shift) 各 base_ch 维
        self.modulator = nn.Sequential(
            nn.Linear(feat_ch * 2, 256),                                # 全连接 1: 2C → 256
            nn.ReLU(inplace=True),                                      # 非线性
            nn.Linear(256, base_ch * 2),                                # 全连接 2: 256 → 2*base_ch
        )

    def forward(self, view_features):
        """
        Args:
            view_features: (B, V, C) — V 个视角的特征向量 (V 可变)

        Returns:
            (B, base_ch, D, H, W) — 调制后的基座
        """
        B, V, C = view_features.shape                                   # 批量、视角数、特征维度

        # ---- 视角池化: 均值 + 标准差 → 固定维度 (B, 2C) ----
        mean_feat = view_features.mean(dim=1)                           # 视角均值 → (B, C)
        std_feat = view_features.std(dim=1, unbiased=False)             # 视角标准差 → (B, C)
        global_feat = torch.cat([mean_feat, std_feat], dim=1)           # 拼接 → (B, 2C)

        # ---- FiLM 调制 ----
        mod = self.modulator(global_feat)                               # → (B, base_ch*2)
        scale, shift = mod.chunk(2, dim=1)                              # 拆分为 (B, base_ch) 各一份
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)         # → (B, base_ch, 1, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)         # → (B, base_ch, 1, 1, 1)

        # ---- 扩展基座 + FiLM ----
        base = self.base_feat.expand(B, -1, -1, -1, -1)                # (B, base_ch, D, H, W)
        return base * (1.0 + scale) + shift                             # FiLM 调制


# =========================================================================
# 高频 Codebook (VQ 残差补全): 补全稀疏视角丢失的细节
# =========================================================================

class HighFreqCodebook(nn.Module):
    """
    向量量化码本，补全稀疏视角无法覆盖的高频细节。

    工作流程:
      1. 编码器将 3D 特征映射到码本空间
      2. 最近邻查找量化编码
      3. 解码器从量化编码恢复高频残差
      4. STE (直通估计器) 保证反向传播

    输入:  3D 特征体 (B, C_in, D, H, W)
    输出:  高频残差  (B, C_out, D, H, W) + VQ 损失 + 困惑度
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64,
                 in_ch=64, out_ch=32):
        super().__init__()
        # ---- 码本: N 个 embedding_dim 维的可学习向量 ----
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)     # (N_embed, emb_dim)

        # ---- 编码器: 3D 特征 → embedding_dim 维连续编码 ----
        self.encoder = nn.Sequential(
            ConvBlock3D(in_ch, embedding_dim, kernel=1, padding=0),    # 1×1×1 卷积 → emb_dim
        )

        # ---- 解码器: 量化特征 → 残差 ----
        self.decoder = nn.Sequential(
            nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1),    # 1×1×1 卷积
            nn.BatchNorm3d(embedding_dim),                              # BN
            nn.ReLU(inplace=True),                                      # ReLU
            nn.Conv3d(embedding_dim, out_ch, kernel_size=1),           # 1×1×1 投影到输出通道
        )

        # ---- VQ 承诺损失权重 ----
        self.commitment_cost = 0.25                                     # 平衡编码器与码本的更新速度

    def forward(self, z):
        # z: (B, C_in, D, H, W) — 输入 3D 特征体
        B, _, D, H, W = z.shape                                        # 拆解形状
        N_embed = self.codebook.weight.shape[0]                         # 码本大小 (1024)
        emb_dim = self.codebook.weight.shape[1]                         # 码本向量维度 (64)

        # 1. 编码 → (B, emb_dim, D, H, W)
        z_e = self.encoder(z)                                           # 连续编码到 emb_dim 维

        # 2. 展平: (B, emb_dim, D, H, W) → (B*D*H*W, emb_dim)
        z_e_flat = z_e.permute(0, 2, 3, 4, 1).contiguous()             # (B, D, H, W, emb_dim)
        z_e_flat = z_e_flat.reshape(-1, emb_dim)                        # (N_vox, emb_dim)

        # 3. 内存高效的最近邻量化: 分批计算距离，避免 (N_vox × N_embed) 全矩阵
        N_vox = z_e_flat.shape[0]                                       # 总体素数
        codebook_w = self.codebook.weight                               # (N_embed, emb_dim)
        cb_norm2 = torch.sum(codebook_w ** 2, dim=1)                    # (N_embed,) 码本向量模平方

        # 分批处理以节省显存 (每批最多 65536 个体素)
        CHUNK = 65536                                                   # 每批体素数
        indices = torch.empty(N_vox, dtype=torch.long, device=z.device) # 预分配索引张量
        for start in range(0, N_vox, CHUNK):                            # 逐批处理
            end = min(start + CHUNK, N_vox)                             # 批结束位置
            chunk = z_e_flat[start:end]                                 # (chunk, emb_dim)
            # ||chunk - codebook||² = ||chunk||² + ||cb||² - 2·chunk·cb^T
            c_norm2 = torch.sum(chunk ** 2, dim=1, keepdim=True)        # (chunk, 1)
            dot = torch.matmul(chunk, codebook_w.t())                    # (chunk, N_embed)
            dist = c_norm2 + cb_norm2.unsqueeze(0) - 2 * dot            # (chunk, N_embed)
            indices[start:end] = dist.argmin(dim=1)                     # 最近邻索引

        # 4. 查表获取量化向量 → (N_vox, emb_dim)
        z_q_flat = self.codebook(indices)                               # 用索引从码本取值

        # 5. 重塑回 3D → (B, emb_dim, D, H, W)
        z_q = z_q_flat.view(B, D, H, W, emb_dim)                        # (B, D, H, W, emb_dim)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()                   # (B, emb_dim, D, H, W)

        # 6. STE (Straight-Through Estimator): 前向用量化值，反向传梯度到 z_e
        z_q_st = z_e + (z_q - z_e).detach()                             # 直通估计器

        # 7. 解码为高频残差 → (B, out_ch, D, H, W)
        residual = self.decoder(z_q_st)                                 # 从量化编码解码残差

        # 8. VQ 损失: 码本损失 + 承诺损失
        vq_loss = (                                                     # VQ 总损失
            F.mse_loss(z_q.detach(), z_e)                               # 码本损失
            + self.commitment_cost * F.mse_loss(z_e, z_q.detach())      # 承诺损失
        )

        # 9. 困惑度: 衡量码本利用率 (越高表示越均匀)
        perplexity = self._calc_perplexity(indices, N_embed)            # 计算困惑度

        return residual, vq_loss, perplexity

    def _calc_perplexity(self, indices, n_embed):
        """计算码本困惑度: exp(-Σ p_i log p_i)。值越高表示码本利用率越均匀。"""
        encodings = F.one_hot(indices, n_embed).float()                 # (B*D*H*W, N_embed) one-hot
        avg_probs = encodings.mean(dim=0)                                # 每个码本向量的平均使用概率
        perplexity = torch.exp(                                         # exp(熵) = 困惑度
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))        # 加 epsilon 防 log(0)
        )
        return perplexity


# =========================================================================
# 3D U-Net 解码器: 融合低高频特征 → 输出体素
# =========================================================================

class UNet3D(nn.Module):
    """
    3D U-Net: 融合低频基座 + 高频残差 → 最终 CT 体素。

    编码-解码结构:
      enc1(128) → down(64) → enc2(64) → down(32) → enc3(32) → down(16)
      → bottleneck(16) → up(32) → dec3(32) → up(64) → dec2(64) → up(128) → dec1(128)

    输入:  (B, in_ch, D, H, W)    — 默认 (B, 64, 128, 128, 128)
    输出:  (B, out_ch, D, H, W)   — 默认 (B, 1, 128, 128, 128)
    """
    def __init__(self, in_ch=64, base_ch=64, out_ch=1, final_upsample=False):
        super().__init__()
        self.final_upsample = final_upsample                            # 是否最后上采样 2×

        # ---- 编码器 (下采样路径) ----
        self.enc1 = nn.Sequential(                                      # 第 1 级: 128³
            ConvBlock3D(in_ch, base_ch),                                 # 3D 卷积块
            ResBlock3D(base_ch),                                         # 残差块
        )
        self.down1 = nn.Conv3d(base_ch, base_ch,                        # 128³ → 64³
                               kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(                                      # 第 2 级: 64³
            ConvBlock3D(base_ch, base_ch * 2),                           # 通道翻倍
            ResBlock3D(base_ch * 2),
        )
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 2,               # 64³ → 32³
                               kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(                                      # 第 3 级: 32³
            ConvBlock3D(base_ch * 2, base_ch * 4),                       # 通道再翻倍
            ResBlock3D(base_ch * 4),
        )
        self.down3 = nn.Conv3d(base_ch * 4, base_ch * 4,               # 32³ → 16³
                               kernel_size=2, stride=2)

        # ---- 瓶颈层 (最深层) ----
        self.bottleneck = nn.Sequential(
            ConvBlock3D(base_ch * 4, base_ch * 8),                       # 通道翻倍
            ResBlock3D(base_ch * 8),                                     # 残差
            ConvBlock3D(base_ch * 8, base_ch * 4),                       # 压缩回 base_ch*4
        )

        # ---- 解码器 (上采样路径 + 跳跃连接) ----
        self.up3 = nn.ConvTranspose3d(base_ch * 4, base_ch * 4,        # 16³ → 32³
                                      kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(                                      # 解码级 3
            ConvBlock3D(base_ch * 8, base_ch * 4),                      # 跳跃连接: base_ch*4 + base_ch*4 = base_ch*8
            ResBlock3D(base_ch * 4),
        )

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2,        # 32³ → 64³
                                      kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(                                      # 解码级 2
            ConvBlock3D(base_ch * 4, base_ch * 2),                      # base_ch*2 + base_ch*2
            ResBlock3D(base_ch * 2),
        )

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch,            # 64³ → 128³
                                      kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(                                      # 解码级 1
            ConvBlock3D(base_ch * 2, base_ch),                          # base_ch + base_ch
            ResBlock3D(base_ch),
        )

        # ---- 输出头 ----
        if final_upsample:                                              # 可选: 128³ → 256³
            self.upsample_final = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=False
            )
        self.head = nn.Conv3d(base_ch, out_ch, kernel_size=1)           # 1×1×1 卷积 → 输出通道

    def forward(self, x):
        # x: (B, in_ch, D, H, W)

        # ---- 编码 ----
        e1 = self.enc1(x)                                               # (B, base_ch, 128, 128, 128)
        e2 = self.enc2(self.down1(e1))                                  # (B, base_ch*2, 64, 64, 64)
        e3 = self.enc3(self.down2(e2))                                  # (B, base_ch*4, 32, 32, 32)

        # ---- 瓶颈 ----
        b = self.bottleneck(self.down3(e3))                             # (B, base_ch*4, 16, 16, 16)

        # ---- 解码 (上采样 + 跳跃连接) ----
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))            # (B, base_ch*4, 32, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))           # (B, base_ch*2, 64, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))           # (B, base_ch, 128, 128, 128)

        # ---- 输出 ----
        if self.final_upsample:                                         # 若需更大输出
            d1 = self.upsample_final(d1)                                # (B, base_ch, 256, 256, 256)
        out = self.head(d1)                                             # (B, out_ch, D_out, H_out, W_out)

        return out


# =========================================================================
# 完整模型: SparseViewReconstruction
# =========================================================================

class SparseViewReconstruction(nn.Module):
    """
    稀疏视角 CT 重建模型。

    固定内部分辨率: 128³ (编码器 256→128 下采样一次)。
    数据集应预先将体素 resize 到 128³，投影 resize 到 256×256。

    训练: projs (B, V, 1, 256, 256) → volume (B, 1, 128, 128, 128)
          V 在训练时随机采样 (如 6~491)，增强对稀疏视角的泛化能力。
    推理: projs (B, V, 1, 256, 256) → volume (B, 1, 128, 128, 128)
          V 任意 (如 6, 8, 10, 491)。

    处理流程:
      1. Encoder2D:    投影图 → 多视角 2D 特征 (256→128)
      2. VolumeBuilder: 2D 特征 → 3D 特征体 (无需相机参数)
      3. LowFreqBase:   多视角特征 → 视角池化 → FiLM 调制 → 低频解剖基座
      4. HighFreqCodebook: 3D 特征体 → VQ 量化 → 高频残差
      5. UNet3D:        低+高频融合 → 最终体素
    """

    # 编码器固定参数: 256 输入 → 128 特征图 (n_down=1)
    ENC_INPUT_SIZE = 256                                             # 投影图输入分辨率
    ENC_FEAT_SIZE = 128                                              # 2D 特征图分辨率
    VOL_SIZE = (128, 128, 128)                                       # 内部 3D 体素分辨率

    def __init__(
        self,
        max_views=491,                                                # 最大视角数 (用于构造时的参数维度)
        vol_ch=64,                                                    # 3D 特征体通道数
        base_ch=32,                                                   # 低频基座通道数
        codebook_size=1024,                                           # 码本大小
        codebook_dim=64,                                              # 码本向量维度
        final_upsample=False,                                         # UNet3D 是否最后上采样 2×
    ):
        super().__init__()
        self.max_views = max_views                                    # 最大视角数 (仅记录)
        D, H, W = self.VOL_SIZE                                       # 内部体素尺寸 (128, 128, 128)

        # ---- 2D 特征提取: 256×256 → 128×128 (n_down=1) ----
        self.encoder_2d = Encoder2D(                                   # 共享权重的 2D CNN
            in_ch=3, base_ch=32, out_ch=vol_ch, n_down=1               # 3 通道输入: [投影图, sin(θ), cos(θ)]
        )

        # ---- 3D 特征体构建 (可学习，无需相机参数) ----
        self.volume_builder = FeatureVolumeBuilder(
            in_ch=vol_ch, vol_ch=vol_ch, vol_depth=D                   # 深度 = 128
        )

        # ---- 低频基座 (解剖结构先验) ----
        # feat_ch=vol_ch: 视角池化后每个视角贡献 vol_ch 维特征
        # 通过均值+标准差池化，支持任意数量的输入视角
        self.low_freq_base = LowFreqBase(
            base_ch=base_ch,                                           # 基座特征通道数
            vol_shape=(D, H, W),                                       # 体素空间尺寸 128³
            feat_ch=vol_ch,                                            # 每个视角的特征维度 (池化后固定)
        )

        # ---- 高频 Codebook (细节补全) ----
        self.high_freq_codebook = HighFreqCodebook(
            num_embeddings=codebook_size,                              # 码本容量
            embedding_dim=codebook_dim,                                # 码本向量维度
            in_ch=vol_ch,                                              # 输入通道 = 3D 特征体通道
            out_ch=base_ch,                                            # 输出通道 = base_ch (与 low_freq 对齐)
        )

        # ---- 3D U-Net 解码器 ----
        self.unet_3d = UNet3D(
            in_ch=base_ch + base_ch,                                   # 低频 + 高频 = base_ch*2
            base_ch=64,                                                # U-Net 基础通道数
            out_ch=1,                                                  # 单通道 CT 体素输出
            final_upsample=final_upsample,                             # 是否最终上采样
        )

    def forward(self, projs, n_sample_views=None):
        """
        Args:
            projs:          (B, V_total, 1, 256, 256)  — 全部投影图
            n_sample_views: int 或 None  — 本次使用的视角数。
                            None: 使用全部视角 (推理默认)
                            整数: 分层均匀随机采样 n_sample_views 个视角
                                  (将 491 等分 n 区，每区随机抽 1 个，
                                   确保覆盖 360° 全角度，避免视角扎堆)

        Returns:
            volume_pred: (B, 1, 128, 128, 128)  — 预测 CT 体素
            low_freq:    (B, base_ch, 128, 128, 128)  — 低频基座特征
            vq_loss:     标量  — VQ 承诺损失
            perplexity:  标量  — 码本困惑度
        """
        B, V_total = projs.shape[:2]                                  # 批量大小、总视角数

        # ---- 1. 选择视角 ----
        if n_sample_views is None or n_sample_views >= V_total:       # 推理模式: 使用全部视角
            projs_cond = projs                                        # 不做采样
            cond_idx = torch.arange(V_total, device=projs.device)     # 全部索引 [0, 1, ..., V_total-1]
        else:                                                         # 训练模式: 分层均匀随机采样
            # 将 [0, V_total) 等分成 n_sample_views 个区间，
            # 每个区间内随机选 1 个视角，确保视角在 360° 上均匀分散。
            # 例: 491 视角选 6 → 分成 6 个 ~82 大小的区间，各抽 1 个。
            # 这样不会出现"6 个视角全挤在相邻角度"的退化情况。
            cond_idx = torch.zeros(n_sample_views,                    # 预分配索引张量
                                   dtype=torch.long, device=projs.device)
            stride = V_total // n_sample_views                        # 每区基础大小
            for i in range(n_sample_views):                           # 逐区抽样
                start = i * stride                                    # 区间起点
                end = (i + 1) * stride if i < n_sample_views - 1 else V_total  # 区间终点 (末区包含余数)
                cond_idx[i] = torch.randint(start, end, (1,),         # 区内随机取 1 个
                                            device=projs.device)
            cond_idx = cond_idx.sort().values                          # 按视角编号排序 (保持几何连续)
            projs_cond = projs[:, cond_idx, :, :, :]                  # (B, n_sample_views, 1, H, W)

        # ---- 1.5 角度编码: 为每个视角生成 sin/cos 特征图 ----
        # 计算选中视角的弧度: θ_i = idx_i × 2π / V_total
        n_use = projs_cond.shape[1]                                   # 实际使用的视角数
        H_proj, W_proj = projs_cond.shape[3], projs_cond.shape[4]     # 投影图空间尺寸
        theta = cond_idx.float() * (2 * torch.pi / V_total)           # (n_use,) 弧度
        sin_map = torch.sin(theta).view(1, -1, 1, 1).expand(          # (1, n_use, 1, 1) → (B, n_use, H, W)
            B, -1, H_proj, W_proj
        )
        cos_map = torch.cos(theta).view(1, -1, 1, 1).expand(          # 同上
            B, -1, H_proj, W_proj
        )
        # 拼接为 3 通道输入: [投影图, sin(θ), cos(θ)] → (B, n_use, 3, H, W)
        projs_with_angle = torch.cat([                                # 沿通道维拼接
            projs_cond[:, :, 0:1, :, :],                              # 投影图 (保持 1 通道)
            sin_map.unsqueeze(2),                                     # sin 角度编码
            cos_map.unsqueeze(2),                                     # cos 角度编码
        ], dim=2)                                                     # → (B, n_use, 3, H, W)

        # ---- 2. 2D 特征提取: 每个视角独立处理 (3 通道输入) ----
        features_2d = self.encoder_2d(projs_with_angle)             # (B, n_use, vol_ch, 128, 128)

        # ---- 3. 构建 3D 特征体: 多视角 → 3D ----
        volume_raw = self.volume_builder(features_2d)                 # (B, vol_ch, 128, 128, 128)

        # ---- 4. 低频基座: 视角池化 + FiLM 调制 ----
        # 对每个视角做空间全局平均池化 → (B, n_use, vol_ch)
        view_feat = features_2d.mean(dim=[3, 4])                      # 空间池化 → (B, n_use, vol_ch)
        low_freq = self.low_freq_base(view_feat)                      # 视角池化+调制 → (B, base_ch, 128³)

        # ---- 5. 高频 Codebook 残差: VQ 量化补全细节 ----
        high_freq, vq_loss, perplexity = self.high_freq_codebook(
            volume_raw                                                  # → (B, base_ch, 128³)
        )

        # ---- 6. 融合低高频特征 ----
        fused = torch.cat([low_freq, high_freq], dim=1)               # (B, base_ch*2, 128, 128, 128)

        # ---- 7. 3D U-Net 解码 → 最终体素 ----
        volume_pred = self.unet_3d(fused)                             # (B, 1, 128, 128, 128)

        return volume_pred, low_freq, vq_loss, perplexity
