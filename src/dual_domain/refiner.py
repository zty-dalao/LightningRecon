"""在 codebook 基础体积上执行病例特异的残差雕刻。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import (
    ConvNormAct3D,
    DepthwiseSeparableResidualBlock3D,
    ResidualBlock3D,
)


class ResidualVolumeSculptor(nn.Module):
    """在128³融合语义，并以8通道Depthwise 3D块细化256³输出。

    昂贵的语义主干仍停留在128³；最终分辨率只运行紧凑的深度可分离卷积，
    并可通过梯度检查点控制24GB显存下的训练峰值。
    """

    def __init__(
        self,
        prior_feature_channels: int = 32,
        projection_feature_channels: int = 16,
        hidden_channels: int = 24,
        highres_channels: int = 8,
        checkpoint_highres: bool = True,
        output_size: tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        if highres_channels < 1:
            raise ValueError("highres_channels must be positive")
        self.checkpoint_highres = bool(checkpoint_highres)
        in_channels = (
            1 + prior_feature_channels + projection_feature_channels
        )
        self.fusion = nn.Sequential(
            ConvNormAct3D(in_channels, hidden_channels),
            ResidualBlock3D(hidden_channels),
            ResidualBlock3D(hidden_channels),
        )
        self.highres_seed = ConvNormAct3D(
            hidden_channels, highres_channels, kernel_size=1
        )
        self.highres_refinement = nn.Sequential(
            DepthwiseSeparableResidualBlock3D(highres_channels),
        )
        self.residual_head = nn.Conv3d(
            highres_channels, 1, kernel_size=1
        )
        self.gate_head = nn.Conv3d(highres_channels, 1, kernel_size=1)

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
        highres = self.highres_seed(fused)
        highres = F.interpolate(
            highres,
            size=self.output_size,
            mode="trilinear",
            align_corners=False,
        )
        if self.checkpoint_highres and self.training:
            highres = checkpoint(
                self.highres_refinement,
                highres,
                use_reentrant=False,
            )
        else:
            highres = self.highres_refinement(highres)
        residual_logits = self.residual_head(highres)
        gate = torch.sigmoid(self.gate_head(highres))

        # Sculpt directly in the final-resolution logit space.  The old path
        # predicted one 128^3 channel and could only trilinearly interpolate it.
        base_logits = torch.logit(base_volume.clamp(1e-4, 1.0 - 1e-4))
        base_logits_full = F.interpolate(
            base_logits,
            size=self.output_size,
            mode="trilinear",
            align_corners=False,
        )
        refined_logits = base_logits_full + gate * residual_logits
        final_volume = torch.sigmoid(refined_logits)
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
            "highres_features": highres,
            "base_volume_full": base_volume_full,
            "final_volume": final_volume,
        }
