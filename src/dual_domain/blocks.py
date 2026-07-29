"""双域模型共用的轻量二维/三维网络块。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """选择能整除通道数的 GroupNorm 组数。"""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct2D(nn.Module):
    """二维卷积 + GroupNorm + SiLU，适合 batch_size=1。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock2D(nn.Module):
    """保持二维形状不变的残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct2D(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.conv2(self.conv1(x)), inplace=True)


class ConvNormAct3D(nn.Module):
    """三维卷积 + GroupNorm + SiLU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock3D(nn.Module):
    """保持体素网格不变的三维残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct3D(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.conv2(self.conv1(x)), inplace=True)


class DownBlock3D(nn.Module):
    """使用 stride=2 将三维边长减半。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels, stride=2),
            ResidualBlock3D(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    """三线性 2× 上采样后用卷积细化。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels),
            ResidualBlock3D(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="trilinear",
            align_corners=False,
        )
        return self.block(x)


class AngleEmbedding(nn.Module):
    """把弧度角的 sin/cos 编码映射到指定特征维度。"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        angle_features = torch.stack(
            (torch.sin(angles), torch.cos(angles)), dim=-1
        )
        return self.network(angle_features)

