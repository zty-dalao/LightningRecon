"""稀疏投影编码、跨视角融合与病例特异三维 latent 预测。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    AngleEmbedding,
    ConvNormAct2D,
    ConvNormAct3D,
    ResidualBlock2D,
    ResidualBlock3D,
    UpBlock3D,
)


class ProjectionImageEncoder(nn.Module):
    """共享二维 CNN，把每张投影压缩到 1/4 分辨率。"""

    def __init__(self, base_channels: int = 16, feature_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            ConvNormAct2D(3, base_channels, kernel_size=5, stride=2),
            ResidualBlock2D(base_channels),
            ConvNormAct2D(
                base_channels, feature_dim, kernel_size=3, stride=2
            ),
            ResidualBlock2D(feature_dim),
        )

    def forward(
        self, projections: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        """输入 ``[B,V,1,H,W]``，输出 ``[B,V,C,H/4,W/4]``。"""
        if projections.ndim != 5 or projections.shape[2] != 1:
            raise ValueError(
                "projections 必须为 [B,V,1,H,W]，"
                f"实际 {tuple(projections.shape)}"
            )
        batch, views, _, height, width = projections.shape
        if angles.shape != (batch, views):
            raise ValueError(
                f"angles 应为 {(batch, views)}，实际 {tuple(angles.shape)}"
            )

        theta = angles.to(projections)
        sin_map = torch.sin(theta).view(batch, views, 1, 1, 1)
        cos_map = torch.cos(theta).view(batch, views, 1, 1, 1)
        sin_map = sin_map.expand(-1, -1, -1, height, width)
        cos_map = cos_map.expand(-1, -1, -1, height, width)
        encoded_input = torch.cat((projections, sin_map, cos_map), dim=2)

        features = self.network(
            encoded_input.reshape(batch * views, 3, height, width)
        )
        return features.reshape(
            batch, views, *features.shape[1:]
        )


class CrossViewTransformer(nn.Module):
    """在压缩后的每个二维位置上对所有视角进行自注意力。"""

    def __init__(
        self,
        feature_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        pool_size: int = 8,
    ):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim 必须能被 num_heads 整除")
        self.pool_size = int(pool_size)
        self.angle_embedding = AngleEmbedding(feature_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers
        )

    def forward(
        self, features: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        batch, views, channels, height, width = features.shape
        pooled = F.adaptive_avg_pool2d(
            features.reshape(batch * views, channels, height, width),
            (self.pool_size, self.pool_size),
        ).reshape(
            batch,
            views,
            channels,
            self.pool_size,
            self.pool_size,
        )

        # 角度 embedding 直接加到每个视角的所有空间 token。
        angle_tokens = self.angle_embedding(angles.to(features))
        pooled = pooled + angle_tokens[:, :, :, None, None]

        tokens = pooled.permute(0, 3, 4, 1, 2).reshape(
            batch * self.pool_size * self.pool_size,
            views,
            channels,
        )
        tokens = self.transformer(tokens)
        transformed = tokens.reshape(
            batch,
            self.pool_size,
            self.pool_size,
            views,
            channels,
        ).permute(0, 3, 4, 1, 2)

        transformed = F.interpolate(
            transformed.reshape(
                batch * views,
                channels,
                self.pool_size,
                self.pool_size,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, channels, height, width)
        return features + transformed


class LearnedVolumeLift(nn.Module):
    """把角度感知的多视角二维特征提升为低分辨率三维体素特征。

    这里明确称为 learned lift，而不冒充精确 cone-beam 反投影。真实几何
    一致性由单独训练并冻结的前向投影模型在损失中提供。
    """

    def __init__(
        self,
        feature_dim: int = 64,
        volume_channels: int = 32,
        seed_size: int = 16,
    ):
        super().__init__()
        self.seed_size = int(seed_size)
        self.to_seed = nn.Sequential(
            ConvNormAct2D(feature_dim, volume_channels),
            nn.AdaptiveAvgPool2d((seed_size, seed_size)),
        )
        self.depth_embedding = nn.Parameter(
            torch.zeros(1, volume_channels, seed_size, 1, 1)
        )
        nn.init.normal_(self.depth_embedding, std=0.02)
        self.volume_refinement = nn.Sequential(
            ConvNormAct3D(volume_channels, volume_channels),
            ResidualBlock3D(volume_channels),
            ResidualBlock3D(volume_channels),
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        # Transformer 已在视角间交换信息，此处求均值得到病例级二维表征。
        fused_2d = view_features.mean(dim=1)
        seed_2d = self.to_seed(fused_2d)
        seed_3d = seed_2d.unsqueeze(2).expand(
            -1, -1, self.seed_size, -1, -1
        )
        seed_3d = seed_3d + self.depth_embedding
        return self.volume_refinement(seed_3d)


class SparseProjectionEncoder(nn.Module):
    """从 6/8/10-view 投影预测 CT codebook 所需的分层 latent。

    默认投影 128² 时，内部依次产生 16³、32³、64³、128³ 特征；为了与
    CT 先验的默认 latent 对齐，``anatomy_latent`` 为 64³，
    ``boundary_latent`` 与 ``refinement_features`` 为 128³。
    """

    def __init__(
        self,
        feature_dim: int = 64,
        volume_channels: int = 32,
        anatomy_dim: int = 64,
        boundary_dim: int = 32,
        refinement_channels: int = 16,
        transformer_layers: int = 4,
        seed_size: int = 16,
    ):
        super().__init__()
        self.image_encoder = ProjectionImageEncoder(
            base_channels=16, feature_dim=feature_dim
        )
        self.view_transformer = CrossViewTransformer(
            feature_dim=feature_dim,
            num_heads=4,
            num_layers=transformer_layers,
            pool_size=min(8, seed_size),
        )
        self.volume_lift = LearnedVolumeLift(
            feature_dim=feature_dim,
            volume_channels=volume_channels,
            seed_size=seed_size,
        )

        half_channels = max(16, volume_channels // 2)
        self.to_32 = UpBlock3D(volume_channels, volume_channels)
        self.to_64 = UpBlock3D(volume_channels, half_channels)
        self.anatomy_head = nn.Conv3d(
            half_channels, anatomy_dim, kernel_size=1
        )
        self.to_128 = UpBlock3D(half_channels, refinement_channels)
        self.boundary_head = nn.Conv3d(
            refinement_channels, boundary_dim, kernel_size=1
        )

    def forward(
        self, projections: torch.Tensor, angles: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(projections, angles)
        view_features = self.view_transformer(image_features, angles)
        seed_volume = self.volume_lift(view_features)
        volume_32 = self.to_32(seed_volume)
        volume_64 = self.to_64(volume_32)
        anatomy_latent = self.anatomy_head(volume_64)
        volume_128 = self.to_128(volume_64)
        boundary_latent = self.boundary_head(volume_128)
        return {
            "image_features": image_features,
            "view_features": view_features,
            "seed_volume": seed_volume,
            "anatomy_latent": anatomy_latent,
            "boundary_latent": boundary_latent,
            "refinement_features": volume_128,
        }

