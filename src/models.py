"""
SparseViewReconstruction: 单反投影 + HF(64³)/MF(128³)双码本 + Add融合。

三阶段架构:
  阶段1 (预训练码本): 从原始网格均匀采样60/64 views → 训练全部权重和EMA码字
  阶段2 (稀疏适应):    从60/64-view基准集随机抽取逐级降低的视角数
  阶段3 (目标微调):    从同一基准集随机抽取final_view，冻结码本适配层

数据维度:
  输入: (B, V, 3, 256, 256)
  输出: (B, 1, 512, 512, 512)
"""

# PyTorch 张量运算与神经网络组件。
import torch
import torch.nn as nn
import torch.nn.functional as F

# HF/MF 分支共用的三维 EMA 向量量化器。
from .ema_codebook import EMAVectorQuantizer3D


# =========================================================================
# 基础模块
# =========================================================================

class ConvBlock2D(nn.Module):
    """二维卷积、批归一化和 ReLU 组成的基础特征块。"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ResBlock2D(nn.Module):
    """保持通道数和空间尺寸不变的二维残差块。"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        # 残差支路经过两次卷积，最后与输入逐元素相加。
        r = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(r)), inplace=True)

class ConvBlock3D(nn.Module):
    """三维卷积、批归一化和 ReLU 组成的体素特征块。"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ResBlock3D(nn.Module):
    """保持三维网格和通道数不变的体素残差块。"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch)
    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(r)), inplace=True)

class UpBlock3D(nn.Module):
    """三线性插值 2× + Conv3D"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        # 先以三线性插值放大空间网格，再用卷积学习局部细化。
        return self.conv(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))


# =========================================================================
# 2D CNN 编码器
# =========================================================================

class MultiScaleCNN2D(nn.Module):
    """自适应下采样 (2× stride=2), 输出 feat(H/4×W/4, 256ch)"""
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock2D(in_ch, base_ch, 7, 1, 3), ResBlock2D(base_ch))
        self.enc1 = nn.Sequential(ConvBlock2D(base_ch, base_ch*2, 3, 2, 1), ResBlock2D(base_ch*2))
        self.enc2 = nn.Sequential(ConvBlock2D(base_ch*2, base_ch*4, 3, 2, 1), ResBlock2D(base_ch*4))
        self.enc3 = nn.Sequential(ConvBlock2D(base_ch*4, base_ch*8, 3, 1, 1), ResBlock2D(base_ch*8))

    def forward(self, x):
        # 将视角维合并进 batch，使每张投影共享同一个二维编码器。
        B, V = x.shape[:2]
        x = x.reshape(B * V, *x.shape[2:])
        x = self.enc3(self.enc2(self.enc1(self.stem(x))))           # (B*V, 256, H/4, W/4)
        # 恢复独立的 batch 与 view 维，供跨视角 Transformer 使用。
        _, C, H, W = x.shape
        return x.reshape(B, V, C, H, W)


# =========================================================================
# 跨视角 Transformer (空间压缩版)
# =========================================================================

class ViewTransformer(nn.Module):
    """空间压缩到 8² 后做跨视角自注意力"""
    def __init__(self, dim=256, num_heads=4, num_layers=4, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        # 输入形状为 (病例, 视角, 通道, 高, 宽)。
        B, V, C, H, W = x.shape
        # 保留未经 Transformer 修改的局部二维特征作为残差。
        identity = x
        # 空间压缩
        ps = self.pool_size
        x_pool = F.adaptive_avg_pool2d(
            x.reshape(B*V, C, H, W), (ps, ps)).reshape(B, V, C, ps, ps)
        # 每个压缩后的空间位置形成一条长度为 V 的视角 token 序列。
        x_pool = x_pool.permute(0, 3, 4, 1, 2).reshape(B*ps*ps, V, C)
        # Transformer
        x_out = self.transformer(x_pool)
        # 恢复空间
        x_out = x_out.reshape(B, ps, ps, V, C).permute(0, 3, 4, 1, 2)
        x_out = F.interpolate(x_out.reshape(B*V, C, ps, ps), size=(H, W),
                              mode='bilinear', align_corners=False).reshape(B, V, C, H, W)
        # 残差相加，保留原始局部纹理并注入跨视角全局信息。
        return identity + x_out


# =========================================================================
# 可微分反投影: 2D(V,C,H,W) → 3D(D³)
# =========================================================================

class BackProjection3D(nn.Module):
    """V个视角→深度维度 + 3D卷积 → 立方体"""
    def __init__(self, in_ch, mid_ch, out_ch, vol_depth=64):
        super().__init__()
        self.vol_depth = vol_depth
        self.conv = nn.Sequential(
            ConvBlock3D(in_ch, mid_ch, 3, padding=1),
            ResBlock3D(mid_ch),
            ConvBlock3D(mid_ch, out_ch, 3, padding=1),
        )

    def forward(self, feat_2d):
        # 将视角维移动到三维插值的“深度”位置。
        B, V, C, H, W = feat_2d.shape
        D = self.vol_depth
        x = feat_2d.permute(0, 2, 1, 3, 4).contiguous()            # (B, C, V, H, W)
        x = F.interpolate(x, size=(D, D, D), mode='trilinear', align_corners=False)
        return self.conv(x)                                         # (B, out_ch, D, D, D)



# =========================================================================
# FiLM 调制块 (条件来自 3D 全局池化)
# =========================================================================

class FiLMBlock3D(nn.Module):
    """global_avg_pool(cond_3d) → MLP → γ,β → FiLM + Conv3D"""
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_ch, cond_ch//2), nn.ReLU(inplace=True), nn.Linear(cond_ch//2, ch*2))
        self.conv = ConvBlock3D(ch, ch, 3, padding=1)

    def forward(self, x, cond):
        # 条件体积先做全局池化，再预测每个输出通道的缩放和偏置。
        gb = self.mlp(F.adaptive_avg_pool3d(cond, (1,1,1)).flatten(1))
        gamma, beta = gb.chunk(2, dim=1)
        return self.conv(x * (1.0 + gamma.view(-1,gamma.shape[1],1,1,1)) + beta.view(-1,beta.shape[1],1,1,1))


# =========================================================================
# 渐进式解码器: HF↑→FiLM→Add(MF)→可配置目标分辨率
# =========================================================================

class ProgressiveDecoder(nn.Module):
    """
    A_hf(64³,128ch) → up(128³,64ch) → FiLM → +B_mf(128³,64ch)
    → up(256³) → up(target³)
    """
    def __init__(self, hf_ch=128, mf_ch=64, n_ups=2):
        super().__init__()
        self.hf_up = UpBlock3D(hf_ch, mf_ch)                        # 64³→128³
        self.film = FiLMBlock3D(mf_ch, mf_ch)
        # 可配置的上采样链 (ch//4 通道压缩，适配 512³ 显存)
        # 每次额外上采样都将边长扩大两倍，并逐步压缩通道以控制显存。
        ups = []; ch = mf_ch
        for i in range(n_ups):
            next_ch = max(4, ch // 4)
            ups.append(UpBlock3D(ch, next_ch))
            ch = next_ch
        self.ups = nn.ModuleList(ups)
        # 线性 1×1×1 卷积输出单通道归一化 CT；这里没有 Sigmoid。
        self.head = nn.Conv3d(ch, 1, 1)

    def forward(self, A_hf, B_mf, vol_3d):
        # HF 从 64³ 上采样到 128³后，先由共享体积进行 FiLM 调制。
        a = self.hf_up(A_hf)
        a = self.film(a, vol_3d)
        # HF 与 MF 必须具有相同形状，采用逐元素加法融合。
        x = a + B_mf
        for up in self.ups: x = up(x)
        return self.head(x)


# =========================================================================
# 完整模型
# =========================================================================

class SparseViewReconstruction(nn.Module):
    """
    单BP→64³ → HF(64³码本)+MF(↑128³码本) → Add融合 → 可配置目标分辨率

    阶段控制:
      stage1: 训练全部 (码本可学习)
      stage2: 冻结 EMA 码字更新, 微调编码器/解码器和量化适配层
      stage3: 冻结 EMA 码字和量化适配层, 微调编码器/解码器
    """

    def __init__(
        self,
        n_decoder_ups=2,
        transformer_layers=4,
        hf_codebook_size=512,
        mf_codebook_size=256,
        kmeans_iters=10,
        kmeans_samples_per_code=4,
        kmeans_init_batches=8,
        dead_code_threshold=0.1,
        dead_code_check_interval=100,
        dead_code_warmup_steps=100,
    ):
        """
        n_decoder_ups: 解码器上采样次数
          1: 128³→256³ (推荐8GB显卡测试)
          2: 128³→256³→512³ (需要≥16GB)
        transformer_layers: 跨视角 TransformerEncoderLayer 堆叠数
        """
        super().__init__()
        # ---- 1. 2D CNN + Transformer ----
        self.cnn = MultiScaleCNN2D()
        self.transformer = ViewTransformer(num_layers=transformer_layers)

        # ---- 2. 单反投影 → 64³ ----
        self.backproj = BackProjection3D(in_ch=256, mid_ch=128, out_ch=64, vol_depth=64)

        # ---- 3. 双码本 (EMA 更新, 防止坍塌) ----
        codebook_kwargs = {
            'beta': 0.25,
            'kmeans_iters': kmeans_iters,
            'kmeans_samples_per_code': kmeans_samples_per_code,
            'kmeans_init_batches': kmeans_init_batches,
            'dead_code_threshold': dead_code_threshold,
            'dead_code_check_interval': dead_code_check_interval,
            'dead_code_warmup_steps': dead_code_warmup_steps,
        }
        self.codebook_hf = EMAVectorQuantizer3D(
            n_embed=hf_codebook_size, embedding_dim=128, **codebook_kwargs
        )
        self.codebook_mf = EMAVectorQuantizer3D(
            n_embed=mf_codebook_size, embedding_dim=64, **codebook_kwargs
        )

        # 通道投影 (vol 64ch → codebook dim)
        self.proj_hf = nn.Conv3d(64, 128, 1)                        # 64→128 for HF query
        self.proj_mf = nn.Conv3d(64, 64, 1)                         # 64→64  for MF query

        # ---- 4. MF 上采样 (64³→128³) + 通道调整 ----
        self.mf_up = nn.Sequential(
            UpBlock3D(64, 64),                                      # 64³→128³, 64ch保持
            ConvBlock3D(64, 64, 3, padding=1),                     # 精炼
        )

        # ---- 5. 渐进式解码器 ----
        self.decoder = ProgressiveDecoder(hf_ch=128, mf_ch=64, n_ups=n_decoder_ups)

    def _encode_volume(self, projs):
        """Encode projections into the shared 64³ latent volume."""
        # 每个视角先独立提取二维特征。
        feat = self.cnn(projs)                                      # (B, V, 256, 64, 64)
        # 再在相同空间位置上交换跨视角信息。
        feat = self.transformer(feat)
        return self.backproj(feat)                                  # (B, 64, 64, 64, 64)

    def _quantize_volume(self, vol):
        """Build HF/MF queries and apply both quantizers."""
        # HF 查询保持 64³ 网格，只调整到 128 个码本通道。
        q_hf = self.proj_hf(vol)                                    # (B, 128, 64³)
        A_hf, vq_hf, perp_hf = self.codebook_hf(q_hf)
        # MF 查询在量化前提高到 128³，以表达更密集的中尺度结构。
        vol_mf = self.mf_up(vol)                                    # (B, 64, 128³)
        q_mf = self.proj_mf(vol_mf)                                 # (B, 64, 128³)
        B_mf, vq_mf, perp_mf = self.codebook_mf(q_mf)
        # VQ 损失相加；perplexity 仅作为两个分支的汇总监控值。
        return A_hf, B_mf, vq_hf + vq_mf, (perp_hf + perp_mf) / 2.0

    def collect_codebook_initialization(self, projs):
        """Collect initialization features without running the decoder."""
        # 只运行编码和量化路径，不运行高显存的最终解码器。
        vol = self._encode_volume(projs)
        self._quantize_volume(vol)
        return (
            bool(self.codebook_hf.codebook.kmeans_initialized.item())
            and bool(self.codebook_mf.codebook.kmeans_initialized.item())
        )

    def forward(self, projs):
        # 完整前向：二维编码 → 跨视角融合 → 三维潜变量 → 双码本 → 解码。
        vol = self._encode_volume(projs)
        A_hf, B_mf, vq_loss, perplexity = self._quantize_volume(vol)
        out = self.decoder(A_hf, B_mf, vol)                         # (B, 1, target³)
        return out, vq_loss, perplexity

    def freeze_codebooks(self):
        """阶段2: 冻结双码本 (停止 EMA 更新)"""
        self.codebook_hf.freeze()
        self.codebook_mf.freeze()
        print('[Codebooks] EMA updates disabled.')

    def unfreeze_codebooks(self):
        """Enable EMA codeword updates."""
        self.codebook_hf.unfreeze()
        self.codebook_mf.unfreeze()

    def set_codebook_adapters_trainable(self, trainable):
        """Toggle pre/post-quantization convolutions; EMA tensors stay non-gradient."""
        # 这里只控制量化器内部的 pre_quant/post_quant 卷积。
        self.codebook_hf.set_adapter_trainable(trainable)
        self.codebook_mf.set_adapter_trainable(trainable)

    def codebook_diagnostics(self):
        """Diagnostics from the latest HF/MF quantizer forward pass."""
        # 为避免 HF/MF 同名指标互相覆盖，统一增加分支前缀。
        diagnostics = {}
        for prefix, codebook in (
            ('hf', self.codebook_hf),
            ('mf', self.codebook_mf),
        ):
            for name, value in codebook.diagnostics().items():
                diagnostics[f'{prefix}_{name}'] = value
        return diagnostics

    def freeze_encoder(self):
        """Optional decoder-only mode; not used by the default three-stage training."""
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.transformer.parameters(): p.requires_grad = False
        for p in self.backproj.parameters(): p.requires_grad = False
        self.freeze_codebooks()
        print('[Optional] Encoder + EMA codewords frozen; decoder remains trainable.')
