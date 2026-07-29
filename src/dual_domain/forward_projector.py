"""可单独训练的体素到真实投影模型。

第一版采用“可微近似射线积分 + 角度条件二维修正网络”。近似积分提供明确
几何方向，修正网络学习 Halcyon 投影中的散射、噪声和厂家处理风格。

注意：这里的旋转积分是内存受控的平行束近似，不等价于精确 cone-beam
projector。若以后获得每帧完整投影矩阵，应替换 ``_analytic_projection``，
其余训练和循环一致性接口可以保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct2D, ResidualBlock2D


@dataclass(frozen=True)
class ApproximateProjectorGeometry:
    """README/config.yaml 中的 Varian Halcyon 固定几何元数据。"""

    dsd_mm: float = 1540.0
    dso_mm: float = 1000.0
    detector_pixels: tuple[int, int] = (1280, 320)
    detector_spacing_mm: tuple[float, float] = (0.336, 1.344)
    voxel_spacing_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)

    def normalized_vector(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """返回量纲稳定的几何条件向量。"""
        detector_width = (
            self.detector_pixels[0] * self.detector_spacing_mm[0]
        )
        detector_height = (
            self.detector_pixels[1] * self.detector_spacing_mm[1]
        )
        values = (
            self.dso_mm / self.dsd_mm,
            detector_width / self.dsd_mm,
            detector_height / self.dsd_mm,
            self.voxel_spacing_mm[0] / 10.0,
            self.voxel_spacing_mm[1] / 10.0,
            self.voxel_spacing_mm[2] / 10.0,
        )
        return torch.tensor(values, device=device, dtype=dtype)


class ProjectionCorrectionNetwork(nn.Module):
    """把近似线积分修正到真实预处理投影域 [-1,1]。"""

    def __init__(self, hidden_channels: int = 32, geometry_dim: int = 6):
        super().__init__()
        # 输入为近似投影、sinθ、cosθ 和广播后的几何 embedding。
        self.geometry_embedding = nn.Sequential(
            nn.Linear(geometry_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.encoder = nn.Sequential(
            ConvNormAct2D(3 + hidden_channels, hidden_channels),
            ResidualBlock2D(hidden_channels),
            ResidualBlock2D(hidden_channels),
        )
        self.residual_head = nn.Conv2d(
            hidden_channels, 1, kernel_size=3, padding=1
        )

    def forward(
        self,
        analytic_projection: torch.Tensor,
        angles: torch.Tensor,
        geometry_vector: torch.Tensor,
    ) -> torch.Tensor:
        batch_views, _, height, width = analytic_projection.shape
        sin_map = torch.sin(angles).view(batch_views, 1, 1, 1).expand(
            -1, -1, height, width
        )
        cos_map = torch.cos(angles).view(batch_views, 1, 1, 1).expand(
            -1, -1, height, width
        )
        geometry = self.geometry_embedding(geometry_vector)
        geometry = geometry.view(1, -1, 1, 1).expand(
            batch_views, -1, height, width
        )
        features = self.encoder(
            torch.cat(
                (analytic_projection, sin_map, cos_map, geometry), dim=1
            )
        )
        return self.residual_head(features)


class LearnedForwardProjector(nn.Module):
    """从归一化 CT 和任意弧度角生成同尺寸真实投影近似。

    输入 CT 为 ``[B,1,D,H,W]``、数值范围 [0,1]；角度为 ``[B,V]``；
    输出为 ``[B,V,1,H_proj,W_proj]``、范围 [-1,1]。
    """

    def __init__(
        self,
        projection_size: tuple[int, int] = (128, 128),
        integration_size: int = 96,
        correction_channels: int = 32,
        geometry: ApproximateProjectorGeometry | None = None,
    ):
        super().__init__()
        if integration_size <= 0:
            raise ValueError("integration_size 必须为正数")
        self.projection_size = tuple(int(v) for v in projection_size)
        self.integration_size = int(integration_size)
        self.geometry = geometry or ApproximateProjectorGeometry()
        self.correction = ProjectionCorrectionNetwork(
            hidden_channels=correction_channels
        )

    def _analytic_projection(
        self, volume: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        """对一个角度执行可微旋转和射线方向平均，限制峰值显存。"""
        batch = volume.shape[0]
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        transform = torch.zeros(
            batch, 3, 4, device=volume.device, dtype=volume.dtype
        )
        # 在 H/W 横断面旋转，D（头脚方向）保持不变。
        transform[:, 0, 0] = cos_theta
        transform[:, 0, 1] = -sin_theta
        transform[:, 1, 0] = sin_theta
        transform[:, 1, 1] = cos_theta
        transform[:, 2, 2] = 1.0

        grid = F.affine_grid(
            transform, volume.shape, align_corners=False
        )
        rotated = F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # 使用 mean 而不是 sum，使数值尺度不依赖 integration_size。
        projection = rotated.mean(dim=-1)
        return F.interpolate(
            projection,
            size=self.projection_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self, volume: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError(
                f"volume 必须为 [B,1,D,H,W]，实际 {tuple(volume.shape)}"
            )
        batch, views = angles.shape
        if batch != volume.shape[0]:
            raise ValueError("angles batch 与 volume batch 不一致")

        # 投影前降到固定积分网格；逐角度处理避免一次创建 B×V 个 3D grid。
        integration_volume = F.interpolate(
            volume,
            size=(self.integration_size,) * 3,
            mode="trilinear",
            align_corners=False,
        )
        analytic_views = []
        flat_angles = []
        for view_index in range(views):
            angle = angles[:, view_index].to(volume)
            analytic_views.append(
                self._analytic_projection(integration_volume, angle)
            )
            flat_angles.append(angle)

        analytic = torch.stack(analytic_views, dim=1)
        analytic_flat = analytic.reshape(
            batch * views, 1, *self.projection_size
        )
        angle_flat = torch.stack(flat_angles, dim=1).reshape(-1)
        geometry_vector = self.geometry.normalized_vector(
            device=volume.device, dtype=volume.dtype
        )
        residual = self.correction(
            analytic_flat, angle_flat, geometry_vector
        )

        # logit 残差既保持近似投影主体，又确保最终输出位于 [-1,1]。
        analytic_logits = torch.logit(
            analytic_flat.clamp(1e-4, 1.0 - 1e-4)
        )
        normalized_01 = torch.sigmoid(analytic_logits + residual)
        normalized = normalized_01 * 2.0 - 1.0
        return normalized.reshape(
            batch, views, 1, *self.projection_size
        )

