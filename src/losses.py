"""
损失函数: 双监督 (L_CBCT + L_pCT) + 辅助正则化。

来自 mymodel.md 的设计:
  L_total = L_cbct + L_pct + λ_smooth·L_smooth + λ_vq·L_vq + λ_sparse·L_sparse

L_cbct (解剖结构):
  - SSIM: 结构相似性 (关注边缘/纹理)
  - MSE:  均方误差 (关注整体强度)
  - L1:   绝对误差 (关注稀疏差异)
  - 梯度损失: 边缘一致性

L_pct (CT 值准确性):
  - MSE:      均方误差
  - 直方图损失: HU 分布对齐

辅助正则化:
  - 总变分 (TV): 抑制低频基座噪声
  - L1 稀疏:    鼓励基座特征稀疏
  - VQ 损失:    码本学习 (在 HighFreqCodebook 中计算)
"""

import torch                                                        # PyTorch 核心
import torch.nn as nn                                               # 神经网络模块
import torch.nn.functional as F                                     # 函数式 API
from math import exp                                                # 自然指数 (未直接使用，保留)


# =========================================================================
# 3D SSIM (结构相似性指数)
# =========================================================================

def _gaussian_window_3d(window_size, sigma, channel):
    """
    生成 3D 高斯卷积窗口，用于 SSIM 的局部统计量计算。

    通过三个 1D 高斯的张量积构造 3D 窗口:
      W(x,y,z) = g(x) ⊗ g(y) ⊗ g(z)

    Args:
        window_size: 窗口边长 (奇数)
        sigma:       高斯标准差
        channel:     输入通道数 (用于 groups=channel 的分组卷积)

    Returns:
        Tensor: shape (channel, 1, W, W, W) 的 3D 高斯窗口
    """
    coords = torch.arange(window_size, dtype=torch.float32)         # [0, 1, ..., W-1]
    coords -= window_size // 2                                      # 中心化: [-W/2, ..., W/2]

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))                # 1D 高斯: exp(-x²/(2σ²))
    g /= g.sum()                                                    # 归一化使窗口和为 1

    # 张量积构造 3D 窗口: g_3d[x,y,z] = g[x] * g[y] * g[z]
    g_1d = g.view(-1)                                               # (W,) 1D 高斯
    g_3d = (                                                        # 外积构造 3D 窗口
        g_1d[:, None, None] * g_1d[None, :, None] * g_1d[None, None, :]
    )                                                               # (W, W, W)
    g_3d = g_3d.unsqueeze(0).unsqueeze(0)                           # (1, 1, W, W, W) 加 batch 和通道维
    g_3d = g_3d.expand(channel, 1,                                   # 扩展到所有通道
                        window_size, window_size, window_size).contiguous()
    return g_3d                                                     # (channel, 1, W, W, W)


def ssim_3d(img1, img2, window_size=7, data_range=1.0):
    """
    计算两个 3D 体素之间的结构相似性指数 (SSIM)。

    SSIM(x, y) = (2μ_x·μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    值域 [0, 1]，越高表示结构越相似。

    Args:
        img1, img2:  (B, C, D, H, W) 的两个体素张量
        window_size: 高斯窗口大小
        data_range:  数据动态范围

    Returns:
        scalar: SSIM 均值 (全 batch 平均)
    """
    C1 = (0.01 * data_range) ** 2                                   # 亮度稳定常数
    C2 = (0.03 * data_range) ** 2                                   # 对比度稳定常数

    channel = img1.shape[1]                                         # 通道数
    window = _gaussian_window_3d(window_size, 1.5, channel)         # 生成 3D 高斯窗口
    window = window.to(img1.device)                                 # 移至相同设备

    # 局部均值: μ = conv3d(img, window)
    mu1 = F.conv3d(img1, window,                                    # 对 img1 做高斯平滑 → 局部均值
                    padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window,                                    # 对 img2 做高斯平滑 → 局部均值
                    padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)                                             # μ_x²
    mu2_sq = mu2.pow(2)                                             # μ_y²
    mu1_mu2 = mu1 * mu2                                             # μ_x·μ_y

    # 局部方差: σ² = E[X²] - μ²
    sigma1_sq = F.conv3d(img1 * img1, window,                       # E[X²] - μ_x² = σ_x²
                          padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window,                       # E[Y²] - μ_y² = σ_y²
                          padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window,                         # E[XY] - μ_x·μ_y = σ_xy
                        padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM 公式
    ssim_map = (                                                   # (2μ_x·μ_y+C1)(2σ_xy+C2) / ((μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2))
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )
    return ssim_map.mean()                                          # 全图平均


# =========================================================================
# 梯度损失: 边缘一致性
# =========================================================================

def gradient_loss(pred, target):
    """
    计算预测与标签的三方向梯度差异。

    惩罚边缘位置的重建差异，使预测的边缘与标签对齐。

    Args:
        pred, target: (B, C, D, H, W) 张量

    Returns:
        scalar: 梯度差异的 L1 范数
    """
    def _grad3d(x):
        """计算 3D 体素的三个方向梯度绝对值的均值。"""
        # x: (B, C, D, H, W)
        dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])     # D 方向差分
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])     # H 方向差分
        dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])     # W 方向差分
        return dx.mean() + dy.mean() + dz.mean()                    # 三方向均值求和

    return torch.abs(_grad3d(pred) - _grad3d(target))               # 预测与标签梯度差的绝对值


# =========================================================================
# 总变分损失 (Total Variation): 平滑正则
# =========================================================================

def total_variation_3d(x):
    """
    计算 3D 总变分，抑制高频噪声。

    TV(x) = Σ |∂x/∂d| + |∂x/∂h| + |∂x/∂w|

    Args:
        x: (B, C, D, H, W) 张量 (通常用于低频基座)

    Returns:
        scalar: 总变分值
    """
    dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()  # D 方向变化均值
    dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()  # H 方向变化均值
    dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()  # W 方向变化均值
    return dx + dy + dz                                              # 总变分


# =========================================================================
# 直方图匹配损失: CT 值分布对齐
# =========================================================================

def histogram_loss(pred, target, bins=64):
    """
    软直方图匹配损失: 使预测的 HU 分布逼近标签分布。

    通过分位数匹配 (2-Wasserstein 近似) 实现:
      对 pred 和 target 的分位数向量求 MSE。

    Args:
        pred, target: (B, C, D, H, W) 张量
        bins:         分位数采样点数

    Returns:
        scalar: 分位数 MSE 均值
    """
    B = pred.shape[0]                                               # batch 大小
    loss = 0.0                                                      # 累积损失
    for b in range(B):                                              # 逐样本计算 (分布不同)
        p = pred[b].detach().flatten()                              # 预测值展平 (detach 不反向传梯度到排序)
        t = target[b].flatten()                                     # 标签值展平

        # 排序后取分位数
        p_sorted = torch.sort(p)[0]                                 # 预测值升序排列
        t_sorted = torch.sort(t)[0]                                 # 标签值升序排列

        # 均匀采样 bins 个分位点的索引
        idx_p = torch.linspace(0, len(p_sorted) - 1,                # 在 [0, N-1] 内均匀取 bins 个点
                               bins, device=p.device).long()
        idx_t = torch.linspace(0, len(t_sorted) - 1,                # 同理对标签
                               bins, device=t.device).long()

        loss += F.mse_loss(p_sorted[idx_p], t_sorted[idx_t])        # 分位数向量 MSE

    return loss / B                                                 # 批平均


# =========================================================================
# 完整组合损失类
# =========================================================================

class ReconstructionLoss(nn.Module):
    """
    双监督重建损失，组合 CBCT 结构损失 + pCT 值损失 + 正则化项。

    用法:
        criterion = ReconstructionLoss()
        loss_dict = criterion(volume_pred, pct_gt, cbct_gt,
                              low_freq, vq_loss, perplexity)
        total_loss = loss_dict['total']
    """

    def __init__(
        self,
        w_cbct=1.0,                                                  # CBCT 解剖结构损失权重
        w_pct=1.0,                                                   # pCT CT 值损失权重
        w_smooth=0.01,                                               # 总变分平滑权重
        w_vq=0.1,                                                    # VQ 码本损失权重
        w_sparse=0.001,                                              # L1 稀疏正则权重
        w_grad=0.05,                                                 # 梯度边缘损失权重
        w_hist=0.1,                                                  # 直方图匹配权重
    ):
        super().__init__()
        # ---- 保存损失权重 ----
        self.w_cbct = w_cbct                                         # CBCT 损失权重
        self.w_pct = w_pct                                           # pCT 损失权重
        self.w_smooth = w_smooth                                     # 平滑权重
        self.w_vq = w_vq                                             # VQ 权重
        self.w_sparse = w_sparse                                     # 稀疏权重
        self.w_grad = w_grad                                         # 梯度权重
        self.w_hist = w_hist                                         # 直方图权重

        # ---- 基础损失函数 ----
        self.mse = nn.MSELoss()                                      # 均方误差
        self.l1 = nn.L1Loss()                                        # 绝对误差

    def forward(self, volume_pred, pct_gt, cbct_gt, low_freq, vq_loss, perplexity):
        """
        计算组合重建损失。

        Args:
            volume_pred: (B, 1, D, H, W)  预测体素
            pct_gt:      (B, 1, D, H, W)  规划 CT 标签
            cbct_gt:     (B, 1, D, H, W)  CBCT 标签
            low_freq:    (B, C, D, H, W)  低频基座特征
            vq_loss:     标量              VQ 承诺损失 (来自 HighFreqCodebook)
            perplexity:  标量              码本困惑度 (监控用)

        Returns:
            dict:
                'total':      总损失 (用于反向传播)
                'cbct':       CBCT 结构损失
                'pct':        pCT 值损失
                'ssim':       SSIM 值 (越高越好)
                'grad':       梯度损失
                'hist':       直方图损失
                'smooth':     总变分损失
                'vq':         VQ 损失
                'sparse':     稀疏损失
                'perplexity': 困惑度
        """
        # ============================================================
        # CBCT 解剖结构损失: 关注器官边界和组织结构
        # ============================================================
        ssim_val = ssim_3d(volume_pred, cbct_gt)                    # 结构相似性 (越高越好)
        cbct_mse = self.mse(volume_pred, cbct_gt)                   # 均方误差
        cbct_l1 = self.l1(volume_pred, cbct_gt)                     # 绝对误差
        grad_loss = gradient_loss(volume_pred, cbct_gt)             # 梯度边缘损失

        L_cbct = (1 - ssim_val) + cbct_mse + self.w_grad * grad_loss  # CBCT 总损失

        # ============================================================
        # pCT CT 值准确性损失: 关注 HU 值精度
        # ============================================================
        pct_mse = self.mse(volume_pred, pct_gt)                     # 均方误差
        hist_loss = histogram_loss(volume_pred, pct_gt)             # 直方图分布匹配

        L_pct = pct_mse + self.w_hist * hist_loss                   # pCT 总损失

        # ============================================================
        # 辅助正则化
        # ============================================================
        L_smooth = total_variation_3d(low_freq)                     # 低频基座平滑: 抑制噪声
        L_sparse = torch.norm(low_freq, p=1) / low_freq.numel()     # L1 稀疏: 鼓励特征稀疏

        # ============================================================
        # 总损失: 加权求和
        # ============================================================
        L_total = (                                                  # 加权组合
            self.w_cbct * L_cbct                                     # CBCT 结构
            + self.w_pct * L_pct                                     # pCT 值
            + self.w_smooth * L_smooth                               # 平滑正则
            + self.w_vq * vq_loss                                    # VQ 码本
            + self.w_sparse * L_sparse                               # 稀疏正则
        )

        return {                                                     # 返回所有分量 (监控用)
            'total': L_total,                                        # 总损失 (用于反向传播)
            'cbct': L_cbct,                                          # CBCT 结构损失
            'pct': L_pct,                                            # pCT 值损失
            'ssim': ssim_val,                                        # SSIM (监控指标)
            'grad': grad_loss,                                       # 梯度损失
            'hist': hist_loss,                                       # 直方图损失
            'smooth': L_smooth,                                      # 总变分
            'vq': vq_loss,                                           # VQ 损失
            'sparse': L_sparse,                                      # 稀疏损失
            'perplexity': perplexity,                                # 码本困惑度
        }
