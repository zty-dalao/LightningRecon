"""Stage 2：用稀疏投影证据在sCT基底B上执行快速局部雕刻。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock2D, ConvBlock3D, UpBlock3D


class ProjectionEncoder(nn.Module):
    """共享2D CNN编码每张投影，并显式输入其真实角度。"""

    def __init__(self, channels: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock2D(3, 8, stride=2),
            ConvBlock2D(8, 16, stride=2),
            ConvBlock2D(16, channels, stride=2),
        )

    def forward(self, projections, angles):
        if projections.ndim != 5 or projections.shape[2] != 1:
            raise ValueError("projections必须是[B,V,1,H,W]")
        batch, views, _, height, width = projections.shape
        if angles.shape != (batch, views):
            raise ValueError("angles必须是[B,V]并与投影视角一一对应")
        sin_map = torch.sin(angles).view(batch, views, 1, 1, 1)
        cos_map = torch.cos(angles).view(batch, views, 1, 1, 1)
        sin_map = sin_map.expand(-1, -1, -1, height, width)
        cos_map = cos_map.expand_as(sin_map)
        x = torch.cat((projections, sin_map, cos_map), dim=2)
        x = self.network(x.flatten(0, 1))
        return x.reshape(batch, views, *x.shape[1:])


class FastFeatureBackprojector(nn.Module):
    """在低分辨率网格上进行快速角度感知特征反投影。

    当前数据只提供角度、未提供完整锥束几何，因此这里采用可训练特征上的
    平行束近似。它不是FDK，也不用于生成最终灰度；作用只是把二维证据放到
    一个粗三维坐标系中。将来获得SAD/SDD/探测器间距后可替换此模块。
    """

    def __init__(self, volume_size: int = 32, chunk_views: int = 8):
        super().__init__()
        self.volume_size = int(volume_size)
        self.chunk_views = int(chunk_views)

    def _grid(self, angles, dtype, device):
        size = self.volume_size
        axis = torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)
        z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
        theta = angles[:, :, None, None, None]
        detector_u = x * torch.cos(theta) + y * torch.sin(theta)
        detector_v = z.expand_as(detector_u)
        return torch.stack((detector_u, detector_v), dim=-1)

    def forward(self, features, angles):
        batch, views, channels, _, _ = features.shape
        total = features.new_zeros(
            batch, channels, self.volume_size, self.volume_size,
            self.volume_size
        )
        total_sq = torch.zeros_like(total)
        for start in range(0, views, self.chunk_views):
            stop = min(start + self.chunk_views, views)
            current = stop - start
            grid = self._grid(
                angles[:, start:stop], features.dtype, features.device
            )
            # 把(D,H)折叠成grid_sample的输出高度，避免逐切片Python循环。
            grid = grid.reshape(
                batch * current,
                self.volume_size * self.volume_size,
                self.volume_size,
                2,
            )
            sampled = F.grid_sample(
                features[:, start:stop].reshape(
                    batch * current, channels,
                    features.shape[-2], features.shape[-1]
                ),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(
                batch, current, channels,
                self.volume_size, self.volume_size, self.volume_size
            )
            total = total + sampled.sum(dim=1)
            total_sq = total_sq + sampled.square().sum(dim=1)
        mean = total / float(views)
        variance = (total_sq / float(views) - mean.square()).clamp_min(0.0)
        return mean, variance


class ProjectionGuidedSculptor(nn.Module):
    """融合B与投影均值/分歧，输出有置信度门控的256³残差。"""

    def __init__(
        self,
        base_channels: int = 4,
        projection_channels: int = 24,
        evidence_size: int = 32,
    ):
        super().__init__()
        c = int(base_channels)
        self.projection_encoder = ProjectionEncoder(projection_channels)
        self.backprojector = FastFeatureBackprojector(evidence_size)

        self.base0 = ConvBlock3D(1, c)
        self.base1 = ConvBlock3D(c, c * 2, stride=2)
        self.base2 = ConvBlock3D(c * 2, c * 4, stride=2)
        self.base3 = ConvBlock3D(c * 4, c * 6, stride=2)
        fusion_in = c * 6 + projection_channels * 2 + 1
        self.fusion = ConvBlock3D(fusion_in, c * 8)
        self.up2 = UpBlock3D(c * 8, c * 4, c * 4)
        self.up1 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up0 = UpBlock3D(c * 2, c, c)
        self.residual = nn.Conv3d(c, 1, 1)
        self.gate = nn.Conv3d(c, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, base_sct, projections, angles):
        p2d = self.projection_encoder(projections, angles)
        evidence_mean, evidence_variance = self.backprojector(p2d, angles)

        b0 = self.base0(base_sct)
        b1 = self.base1(b0)
        b2 = self.base2(b1)
        b3 = self.base3(b2)
        target_size = b3.shape[-3:]
        evidence_mean = F.interpolate(
            evidence_mean, target_size, mode="trilinear", align_corners=False
        )
        evidence_variance = F.interpolate(
            evidence_variance, target_size,
            mode="trilinear", align_corners=False
        )
        count = math.log2(projections.shape[1] + 1.0) / math.log2(65.0)
        count_map = b3.new_full((b3.shape[0], 1, *target_size), count)
        fused = self.fusion(torch.cat(
            (b3, evidence_mean, evidence_variance, count_map), dim=1
        ))
        x = self.up2(fused, b2)
        x = self.up1(x, b1)
        features = self.up0(x, b0)
        residual = self.residual(features)
        gate = torch.sigmoid(self.gate(features))
        base_logits = torch.logit(base_sct.clamp(1e-4, 1.0 - 1e-4))
        final = torch.sigmoid(base_logits + gate * residual)
        return {
            "final_sct": final,
            "sculpt_residual": residual,
            "evidence_gate": gate,
            "evidence_mean": evidence_mean,
            "evidence_variance": evidence_variance,
        }
