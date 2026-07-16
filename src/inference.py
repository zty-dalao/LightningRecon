"""
推理脚本: 从 N 张稀疏投影重建 CT 体素并保存为 NIfTI。

支持两种模式:
  1. 单病例推理: 指定 --case_id，输出单个 .nii.gz
  2. 批量推理: 指定 --split，遍历整个划分输出到目录

用法:
  # 单病例
  python src/inference.py \
      --checkpoint ./logs/sparse_view/.../best_model.pth \
      --data_root /home/zty20020112/workspace/LightningRecon/data/paired \
      --case_id 2026-06-04_065713 \
      --output ./recon_065713.nii.gz

  # 批量
  python src/inference.py \
      --checkpoint ./logs/sparse_view/.../best_model.pth \
      --data_root /home/zty20020112/workspace/LightningRecon/data/paired \
      --split test \
      --output_dir ./outputs/
"""

import os                                                           # 文件路径操作
import sys                                                          # 系统路径
import argparse                                                     # 命令行参数解析

import torch                                                        # PyTorch 核心
import numpy as np                                                  # 数值计算
import nibabel as nib                                               # NIfTI 文件读写

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SparseViewReconstruction                  # 重建模型
from src.dataset import PairedCBCTDataset                        # 配对数据集


# =========================================================================
# 模型加载
# =========================================================================

def load_model(checkpoint_path, device):
    """
    从检查点加载训练好的模型。

    Args:
        checkpoint_path: .pth 检查点文件路径
        device:          'cuda' 或 'cpu'

    Returns:
        model: 加载好权重并设为 eval 模式的 SparseViewReconstruction
    """
    model = SparseViewReconstruction(                               # 实例化模型 (参数需与训练时一致)
        max_views=491,                                               # 最大视角数
        vol_ch=64,                                                   # 3D 特征体通道数
        base_ch=32,                                                  # 低频基座通道数
        codebook_size=1024,                                          # 码本大小
        codebook_dim=64,                                             # 码本向量维度
    ).to(device)                                                     # 移至设备

    ckpt = torch.load(checkpoint_path,                              # 加载检查点
                       map_location=device,                          # 映射到目标设备
                       weights_only=False)                           # 允许加载完整字典
    model.load_state_dict(ckpt['model_state'])                      # 加载模型权重
    model.eval()                                                     # 切换到评估模式

    print(f'Loaded model from epoch {ckpt.get("epoch", "?")}, '     # 打印来源信息
          f'best PSNR: {ckpt.get("best_psnr", "N/A")}')
    return model                                                     # 返回模型


# =========================================================================
# 单病例重建
# =========================================================================

def reconstruct_single(model, dataset, case_id, device, n_views=None,
                       ct_range=(-1000, 1000)):
    """
    对单个病例执行稀疏视角重建。

    Args:
        model:    已加载的模型
        dataset:  PairedCBCTDataset 实例
        case_id:  目标病例名 (如 "2026-06-04_065713")
        device:   'cuda' 或 'cpu'
        n_views:  推理使用的视角数 (None=使用全部)
        ct_range: CT 值反归一化范围 [min_hu, max_hu]

    Returns:
        volume: np.ndarray shape (D, H, W), float32, HU 值
    """
    # ---- 定位病例索引 ----
    try:
        idx = dataset.cases.index(case_id)                          # 在病例列表中查找
    except ValueError:                                              # 未找到
        raise ValueError(                                           # 抛出友好错误
            f'Case "{case_id}" not found. '
            f'Available: {dataset.cases[:5]}...'
        )

    batch = dataset[idx]                                            # 取出样本字典

    projs = batch['projs'].unsqueeze(0).to(device)                  # 加 batch 维 → (1, V, 1, H, W)

    # ---- 模型推理 ----
    with torch.no_grad():                                           # 禁用梯度 (省显存)
        volume_pred, _, _, _ = model(                               # 前向推理
            projs, n_sample_views=n_views                            # None=使用全部视角
        )

    # ---- 反归一化: [0, 1] → HU ----
    ct_min, ct_max = ct_range                                       # 拆解裁剪范围
    volume = volume_pred[0, 0].cpu().numpy()                        # 去 batch 和通道 → (D, H, W)
    volume = volume * (ct_max - ct_min) + ct_min                    # 线性映射回 HU
    volume = volume.astype(np.float32)                              # 确保 float32

    return volume                                                   # 返回 HU 体素


# =========================================================================
# NIfTI 保存
# =========================================================================

def save_nifti(volume, output_path, spacing=(0.9766, 0.9766, 1.9962)):
    """
    将 numpy 体素数组保存为 .nii.gz 文件。

    nibabel 的轴顺序是 (x, y, z) = (W, H, D)，
    而我们的 volume 是 (D, H, W)，需转置。

    Args:
        volume:      np.ndarray shape (D, H, W), float32
        output_path: 输出文件路径 (.nii.gz)
        spacing:     体素间距 (sx, sy, sz) mm，默认匹配原始数据
    """
    # 转置: (D, H, W) → (W, H, D) 以满足 nibabel 轴约定
    volume_nii = np.transpose(volume, (2, 1, 0))                    # → (W, H, D)

    # 构造仿射矩阵 (体素坐标 → 世界坐标)
    affine = np.eye(4)                                              # 4×4 单位矩阵
    affine[0, 0] = spacing[0]                                       # X 方向体素间距
    affine[1, 1] = spacing[1]                                       # Y 方向体素间距
    affine[2, 2] = spacing[2]                                       # Z 方向体素间距

    nii = nib.Nifti1Image(volume_nii, affine)                       # 创建 NIfTI 图像对象
    nib.save(nii, output_path)                                      # 保存到磁盘
    print(f'Saved volume [{volume_nii.shape}] to {output_path}')    # 打印确认


# =========================================================================
# 推理主函数
# =========================================================================

def inference(args):
    """执行推理流程 (单病例或批量)。"""
    # ---- 设备选择 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 优先
    print(f'Device: {device}')                                      # 打印设备

    # ---- 加载模型 ----
    model = load_model(args.checkpoint, device)                     # 从检查点加载

    # ---- 数据集 (加载指定数量的投影) ----
    n_views = args.n_views if args.n_views is not None else 491      # 推理视角数 (None→全部)
    dataset = PairedCBCTDataset(                                    # 推理数据集
        data_root=args.data_root,                                   # paired/ 路径
        split=args.split if args.case_id is None else 'test',       # 批量用指定划分，单病例用 test
        n_views=n_views,                                            # 加载的投影数
        proj_size=(256, 256),                                       # 投影 resize
        vol_size=(128, 128, 128),                                   # 体素 resize
    )

    if args.case_id:                                                # 单病例模式
        os.makedirs(                                                # 确保输出目录存在
            os.path.dirname(args.output) or '.', exist_ok=True
        )
        volume = reconstruct_single(                                # 执行重建
            model, dataset, args.case_id, device,
            n_views=args.n_views,                                    # 传入视角数 (None=全部)
            ct_range=(-1000, 1000)                                   # HU 窗口 [-1000, 1000]
        )
        save_nifti(volume, args.output)                             # 保存 NIfTI
    else:                                                           # 批量模式
        os.makedirs(args.output_dir, exist_ok=True)                 # 确保输出目录存在
        for case_id in dataset.cases:                               # 遍历所有病例
            print(f'Processing: {case_id}')                         # 打印进度
            volume = reconstruct_single(                            # 执行重建
                model, dataset, case_id, device,
                n_views=args.n_views,                                # 传入视角数
                ct_range=(-1000, 1000)
            )
            out_path = os.path.join(                                # 构造输出路径
                args.output_dir, f'{case_id}_recon.nii.gz'
            )
            save_nifti(volume, out_path)                            # 保存 NIfTI


# =========================================================================
# 命令行参数 & 入口
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(                               # 参数解析器
        description='Sparse-view CT reconstruction inference.'
    )

    # ---- 必需参数 ----
    parser.add_argument('--checkpoint', type=str, required=True,     # 模型检查点
                        help='模型检查点路径 (.pth)')
    parser.add_argument('--data_root', type=str, required=True,     # 数据根目录
                        help='paired/ 目录路径')

    # ---- 推理模式 ----
    parser.add_argument('--case_id', type=str, default=None,        # 单病例 ID
                        help='单个病例名 (如 2026-06-04_065713)，不指定则批量推理')
    parser.add_argument('--split', type=str, default='test',        # 批量推理划分
                        help='批量推理时的数据划分 (默认: test)')
    parser.add_argument('--n_views', type=int, default=None,        # 推理视角数
                        help='推理使用的视角数 (默认: None=使用全部, '
                             '如 --n_views 6 表示只用 6 个稀疏视角)')

    # ---- 输出路径 ----
    parser.add_argument('--output', type=str,                       # 单病例输出路径
                        default='./output.nii.gz',
                        help='单病例推理输出路径')
    parser.add_argument('--output_dir', type=str,                   # 批量输出目录
                        default='./outputs/',
                        help='批量推理输出目录')

    args = parser.parse_args()                                      # 解析命令行参数
    inference(args)                                                 # 执行推理
