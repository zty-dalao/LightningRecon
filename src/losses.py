"""
损失函数: Charbonnier 图像损失 + 拉普拉斯金字塔 + 结构损失 + VQ 损失。

L = w_img·L_img + w_lap·L_lap + w_struct·L_struct + w_vq·L_vq

L_img:  Charbonnier (或 L1), 建议在 body mask 内计算
L_lap:  拉普拉斯金字塔分解，HF比HF, MF比MF
L_struct: 梯度 L1 (边缘结构对齐 CT)
L_vq:  码本学习 (commitment loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Charbonnier 损失 (平滑 L1, 比 L2 更鲁棒)
# =========================================================================

def charbonnier_loss(pred, gt, mask=None, eps=1e-3):
    """
    Charbonnier: sqrt((x-y)² + eps²) — L1 的平滑近似, 梯度更稳定。
    若提供 mask, 仅计算 mask>0 区域。
    """
    diff = pred - gt
    loss = torch.sqrt(diff * diff + eps * eps)
    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        loss = loss[mask].mean()
    else:
        loss = loss.mean()
    return loss


def masked_mae(pred, gt, mask):
    """mask 内的 MAE (HU 误差, 仅监控)"""
    mask = mask.bool()
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=pred.device)
    return (pred[mask] - gt[mask]).abs().mean()


# =========================================================================
# 拉普拉斯金字塔频域损失
# =========================================================================

def laplacian_pyramid_loss(pred, gt, levels=2):
    """
    将 pred 和 gt 分别做拉普拉斯金字塔分解，逐级计算 L1 损失。

    金字塔构造:
      Level 0 (HF): residual = img - upsample(downsample(img))
      Level 1 (MF): downsample(img)

    L = Σ_i ||Pred_level[i] - GT_level[i]||₁
    """
    loss = 0.0
    p, g = pred, gt

    for i in range(levels):
        p_down = F.avg_pool3d(p, kernel_size=2, stride=2)
        g_down = F.avg_pool3d(g, kernel_size=2, stride=2)
        p_up = F.interpolate(p_down, size=p.shape[2:],
                             mode='trilinear', align_corners=False)
        g_up = F.interpolate(g_down, size=g.shape[2:],
                             mode='trilinear', align_corners=False)
        loss += F.l1_loss(p - p_up, g - g_up)
        p, g = p_down, g_down

    loss += F.l1_loss(p, g)
    return loss


# =========================================================================
# 结构损失 (梯度 L1, 对齐 CT)
# =========================================================================

def structural_loss(pred, ref):
    """三方向梯度 L1 差 (pred vs reference)"""
    def _grad(x):
        dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        return dx.mean() + dy.mean() + dz.mean()
    return torch.abs(_grad(pred) - _grad(ref))


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


def ssim_3d_masked(img1, img2, mask, data_range=1.0):
    """mask 内 SSIM (仅监控)"""
    mask = mask.bool()
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=img1.device)
    # 简单做法：mask 外置零
    i1 = img1 * mask.float()
    i2 = img2 * mask.float()
    return ssim_3d(i1, i2, data_range)


# =========================================================================
# 组合损失 (支持三阶段不同权重)
# =========================================================================

class ReconstructionLoss(nn.Module):
    """
    L = w_img·L_img + w_lap·L_lap + w_struct·L_struct + w_vq·L_vq

    L_img 在 body mask 内计算 Charbonnier 损失。
    """
    def __init__(self, w_img=1.0, w_lap=0.05, w_struct=0.10, w_vq=0.05,
                 lap_levels=2, charbonnier_eps=1e-3):
        super().__init__()
        self.w_img = w_img
        self.w_lap = w_lap
        self.w_struct = w_struct
        self.w_vq = w_vq
        self.lap_levels = lap_levels
        self.charbonnier_eps = charbonnier_eps

    def forward(self, pred, ct, vq_loss, mask=None):
        """
        pred: (B, 1, D, H, W)
        ct:   (B, 1, D, H, W)
        mask: (B, 1, D, H, W) or None — body mask
        """
        L_img = charbonnier_loss(pred, ct, mask, self.charbonnier_eps)
        L_lap = laplacian_pyramid_loss(pred, ct, self.lap_levels)
        L_struct = structural_loss(pred, ct)
        L_total = (self.w_img * L_img +
                   self.w_lap * L_lap +
                   self.w_struct * L_struct +
                   self.w_vq * vq_loss)

        with torch.no_grad():
            ssim_val = ssim_3d(pred, ct)
            mask_mae = masked_mae(pred, ct, mask) if mask is not None else torch.tensor(float('nan'))
            mask_ssim = ssim_3d_masked(pred, ct, mask) if mask is not None else torch.tensor(float('nan'))

        return {
            'total': L_total, 'img': L_img, 'lap': L_lap, 'struct': L_struct,
            'vq': vq_loss, 'ssim': ssim_val,
            'mask_mae': mask_mae, 'mask_ssim': mask_ssim,
        }

    def set_weights(self, w_img=None, w_lap=None, w_struct=None, w_vq=None):
        """运行时动态调整损失权重 (用于阶段切换)"""
        if w_img is not None: self.w_img = w_img
        if w_lap is not None: self.w_lap = w_lap
        if w_struct is not None: self.w_struct = w_struct
        if w_vq is not None: self.w_vq = w_vq
