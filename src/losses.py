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
import numpy as np


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

def structural_loss(pred, ref, mask=None):
    """逐体素比较三个方向的一阶梯度，约束边缘位置和幅度。"""
    pred_grads = (
        pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :],
        pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :],
        pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1],
    )
    ref_grads = (
        ref[:, :, 1:, :, :] - ref[:, :, :-1, :, :],
        ref[:, :, :, 1:, :] - ref[:, :, :, :-1, :],
        ref[:, :, :, :, 1:] - ref[:, :, :, :, :-1],
    )
    if mask is None:
        return sum(
            F.l1_loss(pred_grad, ref_grad)
            for pred_grad, ref_grad in zip(pred_grads, ref_grads)
        ) / 3.0

    mask = mask.bool()
    pair_masks = (
        mask[:, :, 1:, :, :] & mask[:, :, :-1, :, :],
        mask[:, :, :, 1:, :] & mask[:, :, :, :-1, :],
        mask[:, :, :, :, 1:] & mask[:, :, :, :, :-1],
    )
    losses = []
    for pred_grad, ref_grad, pair_mask in zip(
        pred_grads, ref_grads, pair_masks
    ):
        if pair_mask.any():
            losses.append((pred_grad - ref_grad).abs()[pair_mask].mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


# =========================================================================
# SSIM (监控)
# =========================================================================

def _skimage_ssim_3d(img1, img2, data_range=1.0, return_map=False):
    """逐病例计算 skimage 3D Gaussian SSIM，不拆分为2D切片。"""
    from skimage.metrics import structural_similarity

    if img1.shape != img2.shape:
        raise ValueError(
            f'SSIM inputs must have the same shape, got '
            f'{tuple(img1.shape)} and {tuple(img2.shape)}'
        )
    if img1.ndim != 5 or img1.shape[1] != 1:
        raise ValueError(
            f'Expected single-channel (B,1,D,H,W), got {tuple(img1.shape)}'
        )
    if min(img1.shape[-3:]) < 11:
        raise ValueError(
            'skimage Gaussian SSIM with sigma=1.5 requires every spatial '
            f'dimension to be at least 11, got {tuple(img1.shape[-3:])}'
        )

    reference = img2.detach().float().cpu().numpy()[:, 0]
    prediction = img1.detach().float().cpu().numpy()[:, 0]
    scores = []
    maps = []
    for ref_volume, pred_volume in zip(reference, prediction):
        result = structural_similarity(
            ref_volume,
            pred_volume,
            data_range=float(data_range),
            gaussian_weights=True,
            sigma=1.5,
            win_size=11,
            use_sample_covariance=False,
            channel_axis=None,
            full=return_map,
        )
        if return_map:
            score, score_map = result
            maps.append(np.asarray(score_map, dtype=np.float32))
        else:
            score = result
        scores.append(float(score))
    score_tensor = img1.new_tensor(scores)
    if return_map:
        return score_tensor, maps
    return score_tensor


def ssim_3d(img1, img2, data_range=1.0):
    """逐病例 skimage 3D Gaussian SSIM，然后对病例求平均。"""
    return _skimage_ssim_3d(img1, img2, data_range).mean()


def ssim_3d_per_case(img1, img2, mask=None, data_range=1.0):
    """一次 skimage 调用返回逐病例 whole-volume 与可选 mask SSIM。"""
    if mask is None:
        return _skimage_ssim_3d(img1, img2, data_range), None
    if mask.shape != img1.shape:
        raise ValueError(
            f'SSIM mask must match image shape, got mask={tuple(mask.shape)} '
            f'and image={tuple(img1.shape)}'
        )
    whole_scores, score_maps = _skimage_ssim_3d(
        img1, img2, data_range, return_map=True
    )
    mask_np = mask.detach().bool().cpu().numpy()[:, 0]
    masked_scores = []
    # skimage Gaussian sigma=1.5 uses an effective 11-wide window. Match
    # skimage's scalar SSIM by excluding the five-voxel border.
    border = 5
    for score_map, sample_mask in zip(score_maps, mask_np):
        valid_mask = sample_mask[
            border:-border, border:-border, border:-border
        ]
        valid_map = score_map[
            border:-border, border:-border, border:-border
        ]
        if valid_mask.any():
            masked_scores.append(float(valid_map[valid_mask].mean()))
        else:
            masked_scores.append(float('nan'))
    return whole_scores, img1.new_tensor(masked_scores)


def ssim_3d_masked(img1, img2, mask, data_range=1.0):
    """在真实 body mask 内汇总 skimage 3D Gaussian SSIM map。"""
    _, masked_scores = ssim_3d_per_case(
        img1, img2, mask, data_range
    )
    valid = ~torch.isnan(masked_scores)
    if not valid.any():
        return img1.new_tensor(float('nan'))
    return masked_scores[valid].mean()


# =========================================================================
# 组合损失 (支持三阶段不同权重)
# =========================================================================

class ReconstructionLoss(nn.Module):
    """
    L = w_img·L_img + w_lap·L_lap + w_struct·L_struct + w_vq·L_vq

    L_img 在 body mask 内计算 Charbonnier 损失。
    """
    def __init__(self, w_img=1.0, w_lap=0.05, w_struct=0.05, w_vq=0.05,
                 lap_levels=2, charbonnier_eps=1e-3):
        super().__init__()
        self.w_img = w_img
        self.w_lap = w_lap
        self.w_struct = w_struct
        self.w_vq = w_vq
        self.lap_levels = lap_levels
        self.charbonnier_eps = charbonnier_eps

    def forward(self, pred, ct, vq_loss, mask=None, compute_metrics=True):
        """
        pred: (B, 1, D, H, W)
        ct:   (B, 1, D, H, W)
        mask: (B, 1, D, H, W) or None — body mask
        """
        L_img = charbonnier_loss(pred, ct, mask, self.charbonnier_eps)
        L_lap = laplacian_pyramid_loss(pred, ct, self.lap_levels)
        L_struct = structural_loss(pred, ct, mask)
        L_total = (self.w_img * L_img +
                   self.w_lap * L_lap +
                   self.w_struct * L_struct +
                   self.w_vq * vq_loss)

        with torch.no_grad():
            if compute_metrics:
                whole_ssim,masked_ssim = ssim_3d_per_case(
                    pred,ct,mask
                )
                ssim_val = whole_ssim.mean()
                mask_mae = (
                    masked_mae(pred, ct, mask)
                    if mask is not None else pred.new_tensor(float('nan'))
                )
                mask_ssim = (
                    masked_ssim[~torch.isnan(masked_ssim)].mean()
                    if masked_ssim is not None
                    and (~torch.isnan(masked_ssim)).any()
                    else pred.new_tensor(float('nan'))
                )
            else:
                ssim_val = pred.new_tensor(float('nan'))
                mask_mae = pred.new_tensor(float('nan'))
                mask_ssim = pred.new_tensor(float('nan'))

        return {
            'total': L_total, 'img': L_img, 'lap': L_lap, 'struct': L_struct,
            'vq': vq_loss, 'ssim': ssim_val,
            'mask_mae': mask_mae, 'mask_ssim': mask_ssim,
            'weighted_img': self.w_img * L_img,
            'weighted_lap': self.w_lap * L_lap,
            'weighted_struct': self.w_struct * L_struct,
            'weighted_vq': self.w_vq * vq_loss,
        }

    def set_weights(self, w_img=None, w_lap=None, w_struct=None, w_vq=None):
        """运行时动态调整损失权重 (用于阶段切换)"""
        if w_img is not None: self.w_img = w_img
        if w_lap is not None: self.w_lap = w_lap
        if w_struct is not None: self.w_struct = w_struct
        if w_vq is not None: self.w_vq = w_vq
