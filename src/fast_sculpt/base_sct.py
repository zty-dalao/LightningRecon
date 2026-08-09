"""Stage 1：将完整FDK-CBCT转换成可控的HU风格sCT基底B。"""

import torch
import torch.nn as nn

from .blocks import ConvBlock3D, UpBlock3D


class BaseSCTNet(nn.Module):
    """轻量3D残差U-Net。

    输出不是从零生成CT，而是在CBCT的logit空间预测有门控的修正量，因而
    尽量保留当前患者的真实解剖轮廓。默认通道数专门控制在24GB显存内。
    """

    def __init__(self, base_channels: int = 4):
        super().__init__()
        c = int(base_channels)
        self.enc0 = ConvBlock3D(1, c)
        self.enc1 = ConvBlock3D(c, c * 2, stride=2)
        self.enc2 = ConvBlock3D(c * 2, c * 4, stride=2)
        self.enc3 = ConvBlock3D(c * 4, c * 6, stride=2)
        self.bottleneck = ConvBlock3D(c * 6, c * 6)
        self.up2 = UpBlock3D(c * 6, c * 4, c * 4)
        self.up1 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up0 = UpBlock3D(c * 2, c, c)
        self.residual = nn.Conv3d(c, 1, 1)
        self.gate = nn.Conv3d(c, 1, 1)

        # 初始时接近恒等映射，避免训练开始就破坏患者结构。
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, cbct: torch.Tensor) -> dict[str, torch.Tensor]:
        if cbct.ndim != 5 or cbct.shape[1] != 1:
            raise ValueError("cbct必须是[B,1,D,H,W]")
        x0 = self.enc0(cbct)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.bottleneck(self.enc3(x2))
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        features = self.up0(x, x0)
        residual = self.residual(features)
        gate = torch.sigmoid(self.gate(features))
        cbct_logits = torch.logit(cbct.clamp(1e-4, 1.0 - 1e-4))
        sct = torch.sigmoid(cbct_logits + gate * residual)
        return {
            "base_sct": sct,
            "base_residual": residual,
            "base_gate": gate,
            "base_features": features,
        }
