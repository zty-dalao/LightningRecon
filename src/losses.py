"""新两阶段sCT模型共用的体素、频率、结构和SSIM函数。"""

# 损失在 GPU 上使用 PyTorch 计算；SSIM 监控会显式转到 CPU/skimage。
import torch
import torch.nn.functional as F


# =========================================================================
# Charbonnier 损失 (平滑 L1, 比 L2 更鲁棒)
# =========================================================================

def charbonnier_loss(pred, gt, eps=1e-3):
    """Charbonnier: sqrt((x-y)² + eps²)，在完整体积上求平均。"""
    # eps 使零点附近可导，并降低极端误差相对 L2 的影响。
    diff = pred - gt
    return torch.sqrt(diff * diff + eps * eps).mean()


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
    # Python 标量会在第一次相加时自动提升为与输入同设备的张量。
    loss = 0.0
    p, g = pred, gt

    for i in range(levels):
        # 平均池化提取下一层低频体积。
        p_down = F.avg_pool3d(p, kernel_size=2, stride=2)
        g_down = F.avg_pool3d(g, kernel_size=2, stride=2)
        p_up = F.interpolate(p_down, size=p.shape[2:],
                             mode='trilinear', align_corners=False)
        g_up = F.interpolate(g_down, size=g.shape[2:],
                             mode='trilinear', align_corners=False)
        # 原体积减去重建低频，得到当前尺度的高频残差。
        loss += F.l1_loss(p - p_up, g - g_up)
        p, g = p_down, g_down

    # 金字塔最底层仍需比较低频主体，避免只约束边缘。
    loss += F.l1_loss(p, g)
    return loss


# =========================================================================
# 结构损失 (梯度 L1, 对齐 CT)
# =========================================================================

def structural_loss(pred, ref):
    """逐体素比较三个方向的一阶梯度，约束边缘位置和幅度。"""
    # 分别计算 D、H、W 三个方向的一阶前向差分。
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
    # 三个方向等权，保持结构损失尺度不随方向数增加。
    return sum(
        F.l1_loss(pred_grad, ref_grad)
        for pred_grad, ref_grad in zip(pred_grads, ref_grads)
    ) / 3.0


# =========================================================================
# SSIM (监控)
# =========================================================================

def _skimage_ssim_3d(img1, img2, data_range=1.0):
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

    # skimage 接收 NumPy 数组，因此先停止梯度并移到 CPU。
    reference = img2.detach().float().cpu().numpy()[:, 0]
    prediction = img1.detach().float().cpu().numpy()[:, 0]
    scores = []
    # 每个病例独立计算一个完整三维体积 SSIM，之后再由调用方平均。
    for ref_volume, pred_volume in zip(reference, prediction):
        score = structural_similarity(
            ref_volume,
            pred_volume,
            data_range=float(data_range),
            gaussian_weights=True,
            sigma=1.5,
            win_size=11,
            use_sample_covariance=False,
            channel_axis=None,
        )
        scores.append(float(score))
    return img1.new_tensor(scores)


def ssim_3d(img1, img2, data_range=1.0):
    """逐病例 skimage 3D Gaussian SSIM，然后对病例求平均。"""
    return _skimage_ssim_3d(img1, img2, data_range).mean()


def ssim_3d_per_case(img1, img2, data_range=1.0):
    """一次 skimage 调用返回逐病例完整体积 SSIM。"""
    return _skimage_ssim_3d(img1, img2, data_range)
