"""
损失函数: 拉普拉斯金字塔频域损失 + VQ 损失。

L = λ₁·L_lap(Pred, pCT) + λ₂·L_struct(Pred, CBCT) + λ_vq·L_vq

L_lap: 拉普拉斯金字塔分解，HF比HF, MF比MF, 形成闭环频域监督
L_struct: 梯度 L1 (边缘结构对齐 CBCT)
L_vq: 码本学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# ★ 修正 4: 拉普拉斯金字塔频域损失
# =========================================================================

def laplacian_pyramid_loss(pred, gt, levels=2):
    """
    将 pred 和 gt 分别做拉普拉斯金字塔分解，逐级计算 L1 损失。

    金字塔构造:
      Level 0 (HF): residual = img - upsample(downsample(img))
      Level 1 (MF): downsample(img)
      ...

    L = Σ_i ||Pred_level[i] - GT_level[i]||₁
    """
    loss = 0.0
    p, g = pred, gt

    for i in range(levels):
        # 下采样
        p_down = F.avg_pool3d(p, kernel_size=2, stride=2)
        g_down = F.avg_pool3d(g, kernel_size=2, stride=2)

        # 上采样回原分辨率
        p_up = F.interpolate(p_down, size=p.shape[2:],
                             mode='trilinear', align_corners=False)
        g_up = F.interpolate(g_down, size=g.shape[2:],
                             mode='trilinear', align_corners=False)

        # HF 残差 (当前级的细节)
        loss += F.l1_loss(p - p_up, g - g_up)

        # 进入下一级
        p, g = p_down, g_down

    # 最粗糙级 (MF 骨架)
    loss += F.l1_loss(p, g)
    return loss


# =========================================================================
# 结构损失 (梯度 L1, 对齐 CBCT)
# =========================================================================

def structural_loss(pred, cbct):
    """三方向梯度 L1 差"""
    def _grad(x):
        dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        return dx.mean() + dy.mean() + dz.mean()
    return torch.abs(_grad(pred) - _grad(cbct))


# =========================================================================
# SSIM (监控)
# =========================================================================

def ssim_3d(img1, img2, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1 = img1.mean(dim=[2, 3, 4], keepdim=True)
    mu2 = img2.mean(dim=[2, 3, 4], keepdim=True)
    s1 = ((img1 - mu1) ** 2).mean(dim=[2, 3, 4], keepdim=True)
    s2 = ((img2 - mu2) ** 2).mean(dim=[2, 3, 4], keepdim=True)
    s12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3, 4], keepdim=True)
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return ssim.mean()


# =========================================================================
# 组合损失
# =========================================================================

class ReconstructionLoss(nn.Module):
    """
    L = λ_lap·L_lap + λ_struct·L_struct + λ_vq·L_vq
    """
    def __init__(self, w_lap=1.0, w_struct=0.3, w_vq=0.1, lap_levels=2):
        super().__init__()
        self.w_lap = w_lap
        self.w_struct = w_struct
        self.w_vq = w_vq
        self.lap_levels = lap_levels

    def forward(self, pred, cbct, pct, vq_loss):
        """
        pred: (B, 1, 256, 256, 256)
        cbct: (B, 1, 256, 256, 256)  — 结构监督
        pct:  (B, 1, 256, 256, 256)  — 频域+灰度监督
        """
        L_lap = laplacian_pyramid_loss(pred, pct, self.lap_levels)
        L_struct = structural_loss(pred, cbct)
        L_total = self.w_lap * L_lap + self.w_struct * L_struct + self.w_vq * vq_loss

        with torch.no_grad():
            ssim_val = ssim_3d(pred, pct)

        return {
            'total': L_total, 'lap': L_lap, 'struct': L_struct,
            'vq': vq_loss, 'ssim': ssim_val,
        }
