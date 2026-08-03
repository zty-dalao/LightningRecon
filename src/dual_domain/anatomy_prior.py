"""从真实 CT 学习分层解剖 codebook 与粗糙基础体积。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ema_codebook import EMAVectorQuantizer3D

from .blocks import (
    ConvNormAct3D,
    DownBlock3D,
    ResidualBlock3D,
    UpBlock3D,
)


class CTAnatomyEncoder(nn.Module):
    """把 256³ CT 编码为 128³ 边缘特征和 64³ 解剖特征。

    ``input_size`` 可在测试中缩小，但必须能被 4 整除。默认尺寸下：

    * boundary_latent: ``[B,boundary_dim,128,128,128]``；
    * anatomy_latent: ``[B,anatomy_dim,64,64,64]``。
    """

    def __init__(
        self,
        base_channels: int = 16,
        anatomy_dim: int = 64,
        boundary_dim: int = 32,
        boundary_residual_blocks: int = 3,
        anatomy_transformer_layers: int = 2,
        anatomy_transformer_heads: int = 4,
        anatomy_context_size: int = 8,
        boundary_context_channels: int = 8,
    ):
        super().__init__()
        if boundary_residual_blocks < 1:
            raise ValueError("boundary_residual_blocks must be positive")
        if anatomy_transformer_layers < 1:
            raise ValueError("anatomy_transformer_layers must be positive")
        if anatomy_dim % anatomy_transformer_heads:
            raise ValueError(
                "anatomy_dim must be divisible by anatomy_transformer_heads"
            )
        if anatomy_context_size < 1 or boundary_context_channels < 1:
            raise ValueError("context sizes/channels must be positive")
        # 第一层立即降到一半分辨率，避免在 256³ 上长期保留多通道特征。
        self.to_boundary = nn.Sequential(
            ConvNormAct3D(1, base_channels, stride=2),
            *(
                ResidualBlock3D(base_channels)
                for _ in range(boundary_residual_blocks)
            ),
        )
        # 分辨率不变，增加一倍维度
        # 降低一半分辨率，同时维度翻倍。
        self.to_anatomy = DownBlock3D(base_channels, base_channels * 2)
        # 分辨率不变，维度再翻倍
        self.to_global_context = DownBlock3D(
            base_channels * 2, anatomy_dim
        )
        self.anatomy_context_size = int(anatomy_context_size)
        self.context_position = nn.Parameter(
            torch.zeros(
                1,
                anatomy_dim,
                self.anatomy_context_size,
                self.anatomy_context_size,
                self.anatomy_context_size,
            )
        )
        nn.init.normal_(self.context_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=anatomy_dim,
            nhead=anatomy_transformer_heads,
            dim_feedforward=anatomy_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.anatomy_transformer = nn.TransformerEncoder(
            layer, num_layers=anatomy_transformer_layers
        )
        self.local_anatomy_projection = ConvNormAct3D(
            base_channels * 2, anatomy_dim, kernel_size=1
        )
        self.anatomy_fusion = nn.Sequential(
            ConvNormAct3D(anatomy_dim * 2, anatomy_dim, kernel_size=1),
            ResidualBlock3D(anatomy_dim),
        )
        self.boundary_context_projection = ConvNormAct3D(
            anatomy_dim, boundary_context_channels, kernel_size=1
        )
        self.boundary_fusion = nn.Sequential(
            ConvNormAct3D(
                base_channels + boundary_context_channels,
                base_channels,
            ),
            ResidualBlock3D(base_channels),
        )
        self.boundary_head = nn.Conv3d(
            base_channels, boundary_dim, kernel_size=1
        )

    def _global_anatomy_context(
        self, anatomy_features: torch.Tensor
    ) -> torch.Tensor:
        """在最多8³个token上计算覆盖完整体积的全局注意力。"""
        context = self.to_global_context(anatomy_features)
        pool_shape = tuple(
            min(self.anatomy_context_size, int(size))
            for size in context.shape[2:]
        )
        context = F.adaptive_avg_pool3d(context, pool_shape)
        position = F.interpolate(
            self.context_position,
            size=pool_shape,
            mode="trilinear",
            align_corners=False,
        )
        context = context + position.to(context)
        batch, channels, depth, height, width = context.shape
        tokens = context.flatten(2).transpose(1, 2)
        tokens = self.anatomy_transformer(tokens)
        context = tokens.transpose(1, 2).reshape(
            batch, channels, depth, height, width
        )
        return F.interpolate(
            context,
            size=anatomy_features.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, ct: torch.Tensor) -> dict[str, torch.Tensor]:
        boundary_local = self.to_boundary(ct)
        anatomy_features = self.to_anatomy(boundary_local)
        global_context = self._global_anatomy_context(anatomy_features)
        local_anatomy = self.local_anatomy_projection(anatomy_features)
        anatomy_latent = self.anatomy_fusion(
            torch.cat((local_anatomy, global_context), dim=1)
        )
        boundary_context = self.boundary_context_projection(global_context)
        boundary_context = F.interpolate(
            boundary_context,
            size=boundary_local.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        boundary_features = self.boundary_fusion(
            torch.cat((boundary_local, boundary_context), dim=1)
        )
        boundary_latent = self.boundary_head(boundary_features)
        return {
            "anatomy_latent": anatomy_latent,                   # [B,64,64，64，64]
            "boundary_latent": boundary_latent,                 # [B,32,128,128,128]
        }


class AnatomyPriorDecoder(nn.Module):
    """把分层量化特征解码为 1/2 分辨率的基础体积。"""

    def __init__(
        self,
        anatomy_dim: int = 64,
        boundary_dim: int = 32,
        feature_channels: int = 32,
    ):
        super().__init__()
        self.anatomy_up = UpBlock3D(anatomy_dim, feature_channels)
        self.boundary_projection = ConvNormAct3D(
            boundary_dim, feature_channels, kernel_size=1
        )
        self.boundary_edge_head = nn.Sequential(
            ResidualBlock3D(feature_channels),
            nn.Conv3d(feature_channels, 1, kernel_size=1),
        )
        self.fusion = nn.Sequential(
            ConvNormAct3D(feature_channels * 2, feature_channels),
            ResidualBlock3D(feature_channels),
            ResidualBlock3D(feature_channels),
        )
        self.base_head = nn.Conv3d(feature_channels, 1, kernel_size=1)

    def forward(
        self,
        anatomy_quantized: torch.Tensor,
        boundary_quantized: torch.Tensor,
        *,
        compute_boundary_edge: bool = True,
    ) -> dict[str, torch.Tensor]:
        anatomy = self.anatomy_up(anatomy_quantized)
        boundary = self.boundary_projection(boundary_quantized)
        if anatomy.shape[2:] != boundary.shape[2:]:
            raise ValueError(
                "解剖和边缘量化特征尺寸不一致："
                f"{anatomy.shape[2:]} vs {boundary.shape[2:]}"
            )
        prior_features = self.fusion(torch.cat((anatomy, boundary), dim=1))
        base_logits = self.base_head(prior_features)
        outputs = {
            "prior_features": prior_features,
            "base_logits": base_logits,
            "base_volume": torch.sigmoid(base_logits),
        }
        if compute_boundary_edge:
            outputs["boundary_edge"] = torch.sigmoid(
                self.boundary_edge_head(boundary)
            )
        return outputs


class HierarchicalAnatomyPrior(nn.Module):
    """CT 教师编码器、双层 EMA codebook 和基础体积解码器。

    该模块先用真实 CT 独立预训练。主重建模型随后复用并冻结 codebook，
    让稀疏投影预测 CT 域中的同一组离散特征。
    """

    def __init__(
        self,
        anatomy_codebook_size: int = 512,
        boundary_codebook_size: int = 256,
        anatomy_dim: int = 64,
        boundary_dim: int = 32,
        base_channels: int = 16,
        prior_feature_channels: int = 32,
        kmeans_init_batches: int = 8,
        boundary_residual_blocks: int = 3,
        anatomy_transformer_layers: int = 2,
        anatomy_transformer_heads: int = 4,
        anatomy_context_size: int = 8,
        boundary_context_channels: int = 8,
    ):
        super().__init__()
        self.anatomy_dim = int(anatomy_dim)
        self.boundary_dim = int(boundary_dim)
        self.prior_feature_channels = int(prior_feature_channels)   # 

        self.encoder = CTAnatomyEncoder(
            base_channels=base_channels,
            anatomy_dim=anatomy_dim,
            boundary_dim=boundary_dim,
            boundary_residual_blocks=boundary_residual_blocks,
            anatomy_transformer_layers=anatomy_transformer_layers,
            anatomy_transformer_heads=anatomy_transformer_heads,
            anatomy_context_size=anatomy_context_size,
            boundary_context_channels=boundary_context_channels,
        )
        quantizer_kwargs = {
            "beta": 0.25,                               # codebook 中码字的loss 系数，用于更新码本
            "kmeans_iters": 10,                         # 收集 CT latent→ 随机选择初始聚类中心→ 将特征分配给最近中心→ 更新聚类中心→ 重复10次→ 将中心写入codebook
            "kmeans_samples_per_code": 4,               # K-means 为每个码字准备的目标采样数量
            "kmeans_init_batches": kmeans_init_batches, # 表示从多少个训练 batch 中收集 K-means 初始化样本。防止只从1个病人进行聚类，导致一开始特征就有所偏移
            "dead_code_threshold": 0.1,                 # 用 EMA 命中统计判断码字是否死亡。条件：dead = cluster_size < 0.1。这里的 cluster_size 不是当前 batch 的原始命中次数，而是经过 EMA 平滑后的使用统计。低于0.1会被当前 batch 中的真实特征重新初始化。
            "dead_code_check_interval": 100,            # 每经过100次有效 EMA 更新，检查一次死亡码字。在116个训练集中，可以认为是1次epoch检查一次死亡码字
            "dead_code_warmup_steps": 100,              # 训练前100次 EMA 更新不执行死亡码字重置。原因是训练刚开始时：Encoder 特征仍不稳定；一些码字暂时没有命中很正常；K-means统计刚开始接受EMA更新；过早判断死亡可能造成码字频繁重置。
        }
        self.anatomy_codebook = EMAVectorQuantizer3D(
            n_embed=anatomy_codebook_size,
            embedding_dim=anatomy_dim,
            **quantizer_kwargs,
        )
        self.boundary_codebook = EMAVectorQuantizer3D(
            n_embed=boundary_codebook_size,
            embedding_dim=boundary_dim,
            **quantizer_kwargs,
        )
        self.decoder = AnatomyPriorDecoder(
            anatomy_dim=anatomy_dim,
            boundary_dim=boundary_dim,
            feature_channels=prior_feature_channels,
        )

    def encode_ct(self, ct: torch.Tensor) -> dict[str, torch.Tensor]:
        """生成可作为投影分支蒸馏教师的连续 CT latent。"""
        return self.encoder(ct)

    def quantize(
        self,
        anatomy_latent: torch.Tensor,
        boundary_latent: torch.Tensor,
        *,
        update_codebook: bool,
    ) -> dict[str, torch.Tensor]:
        """查询双层码本；主重建阶段通过 ``update_codebook=False`` 冻结 EMA。"""
        anatomy_q, anatomy_vq, anatomy_perplexity = self.anatomy_codebook(
            anatomy_latent, no_update=not update_codebook
        )
        boundary_q, boundary_vq, boundary_perplexity = self.boundary_codebook(
            boundary_latent, no_update=not update_codebook
        )
        return {
            "anatomy_quantized": anatomy_q,
            "boundary_quantized": boundary_q,
            "vq_loss": anatomy_vq + boundary_vq,
            "anatomy_perplexity": anatomy_perplexity,
            "boundary_perplexity": boundary_perplexity,
        }

    def decode(
        self,
        anatomy_quantized: torch.Tensor,
        boundary_quantized: torch.Tensor,
        *,
        compute_boundary_edge: bool = True,
    ) -> dict[str, torch.Tensor]:
        return self.decoder(
            anatomy_quantized,
            boundary_quantized,
            compute_boundary_edge=compute_boundary_edge,
        )

    def forward(
        self,
        ct: torch.Tensor,
        *,
        update_codebook: bool = True,
    ) -> dict[str, torch.Tensor]:
        """CT-VQ 自编码器前向，用于单独预训练解剖先验。"""
        latents = self.encode_ct(ct)
        quantized = self.quantize(
            latents["anatomy_latent"],
            latents["boundary_latent"],
            update_codebook=update_codebook,
        )
        decoded = self.decode(
            quantized["anatomy_quantized"],
            quantized["boundary_quantized"],
        )
        return {**latents, **quantized, **decoded}

    def freeze_codebooks(self) -> None:
        """冻结 EMA 数值和量化器内部适配卷积。"""
        self.anatomy_codebook.freeze()
        self.boundary_codebook.freeze()
        self.anatomy_codebook.set_adapter_trainable(False)
        self.boundary_codebook.set_adapter_trainable(False)

    def freeze_teacher_encoder(self) -> None:
        """CT 教师只产生监督 latent，不在主重建阶段更新。"""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
