"""在 codebook 基础体积上执行病例特异的残差雕刻。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct3D, ResidualBlock3D


class ResidualVolumeSculptor(nn.Module):
    """预测 1/2 分辨率 residual/gate，再轻量生成最终 256³ 体积。

    大部分卷积停留在 128³；最终 256³ 只保存单通道 logits/volume，
    避免在高分辨率上维持大量三维特征。
    """

    def __init__(
        self,
        prior_feature_channels: int = 32,
        projection_feature_channels: int = 16,
        hidden_channels: int = 24,
        output_size: tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        in_channels = (
            1 + prior_feature_channels + projection_feature_channels
        )
        self.fusion = nn.Sequential(
            ConvNormAct3D(in_channels, hidden_channels),
            ResidualBlock3D(hidden_channels),
            ResidualBlock3D(hidden_channels),
        )
        self.residual_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        self.gate_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)

    def forward(
        self,
        base_volume: torch.Tensor,
        prior_features: torch.Tensor,
        projection_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected_shape = base_volume.shape[2:]
        if prior_features.shape[2:] != expected_shape:
            raise ValueError("prior_features 与 base_volume 空间尺寸不一致")
        if projection_features.shape[2:] != expected_shape:
            raise ValueError("projection_features 与 base_volume 空间尺寸不一致")

        fused = self.fusion(
            torch.cat(
                (base_volume, prior_features, projection_features), dim=1
            )
        )
        residual_logits = self.residual_head(fused)
        gate = torch.sigmoid(self.gate_head(fused))

        # 在 logit 空间雕刻可保证最终 sigmoid 输出严格位于 [0,1]。
        base_logits = torch.logit(base_volume.clamp(1e-4, 1.0 - 1e-4))
        refined_logits = base_logits + gate * residual_logits
        final_logits = F.interpolate(
            refined_logits,
            size=self.output_size,
            mode="trilinear",
            align_corners=False,
        )
        final_volume = torch.sigmoid(final_logits)
        base_volume_full = F.interpolate(
            base_volume,
            size=self.output_size,
            mode="trilinear",
            align_corners=False,
        )
        return {
            "residual_logits": residual_logits,
            "gate": gate,
            "refined_logits": refined_logits,
            "base_volume_full": base_volume_full,
            "final_volume": final_volume,
        }

