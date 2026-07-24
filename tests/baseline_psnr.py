"""
基线评估: 计算 pCT 与 CBCT 之间的 PSNR/SSIM。

CBCT 是临床重建结果 (输入质量参考), pCT 是金标准 (Ground Truth)。
这个脚本给出"如果不做任何重建，直接用 CBCT 作为输出的 PSNR"。

用法:
    python tests/baseline_psnr.py --split test
    python tests/baseline_psnr.py --split train
"""

import os, sys, argparse, json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_nii(path):
    """加载 nii.gz 并返回 float32 numpy 数组。"""
    img = nib.load(path)
    data = img.get_fdata()
    return np.asarray(data, dtype=np.float32)


def compute_psnr(pred, target):
    """计算 PSNR (假设数值范围 [0, 1])。"""
    mse = np.mean((pred - target) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')


def compute_ssim(pred, target, L=1.0):
    """简化的 3D SSIM (体素级)。"""
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = np.mean((pred - mu_x) * (target - mu_y))

    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return float(ssim_val)


def main():
    parser = argparse.ArgumentParser(description='基线评估: pCT vs CBCT')
    parser.add_argument('--data_root', default='/root/autodl-tmp/thorax')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--value_range', type=float, nargs=2, default=[-1000, 1000],
                        help='HU 范围 [min, max], 用于归一化')
    args = parser.parse_args()

    overlap_dir = os.path.join(args.data_root, 'overlap')
    if not os.path.isdir(overlap_dir):
        print(f'错误: 缺少 overlap 目录 {overlap_dir}')
        sys.exit(1)

    # 加载数据划分
    splits_path = os.path.join(args.data_root, 'splits.json')
    if not os.path.exists(splits_path):
        meta_path = os.path.join(args.data_root, 'meta_info.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            cases = meta.get(args.split, [])
        else:
            print(f'错误: 缺少 {splits_path} 或 {meta_path}')
            sys.exit(1)
    else:
        with open(splits_path) as f:
            splits = json.load(f)
        cases = splits.get(args.split, [])

    print(f'[{args.split}] 共 {len(cases)} 例')
    print(f'{"Case":<30} {"PSNR(dB)":>10} {"SSIM":>8} {"CBCT_min":>10} {"CBCT_max":>10} {"pCT_min":>10} {"pCT_max":>10}')
    print('-' * 90)

    psnrs, ssims = [], []
    vmin, vmax = args.value_range

    for case in cases:
        cbct_path = os.path.join(overlap_dir, f'{case}_cbct.nii.gz')
        ct_path   = os.path.join(overlap_dir, f'{case}_ct.nii.gz')
        mask_cbct = os.path.join(overlap_dir, f'{case}_cbct_mask.nii.gz')
        mask_ct   = os.path.join(overlap_dir, f'{case}_ct_mask.nii.gz')

        if not all(os.path.exists(p) for p in [cbct_path, ct_path]):
            print(f'{case:<30} {"SKIP (文件缺失)":>50}')
            continue

        cbct = load_nii(cbct_path)
        pct  = load_nii(ct_path)

        # 归一化到 [0, 1]
        cbct_n = np.clip((cbct - vmin) / (vmax - vmin), 0, 1)
        pct_n  = np.clip((pct  - vmin) / (vmax - vmin), 0, 1)

        # Mask: 取 CBCT 和 CT 均有解剖结构的区域
        mask = np.ones(cbct.shape, dtype=bool)
        if os.path.exists(mask_cbct) and os.path.exists(mask_ct):
            m_cbct = load_nii(mask_cbct) > 0.5
            m_ct   = load_nii(mask_ct) > 0.5
            mask = m_cbct & m_ct
            if mask.sum() == 0:
                mask = np.ones(cbct.shape, dtype=bool)

        # 在 mask 区域内计算
        cbct_m = cbct_n[mask]
        pct_m  = pct_n[mask]

        psnr = compute_psnr(cbct_m, pct_m)
        ssim = compute_ssim(cbct_m, pct_m)

        psnrs.append(psnr)
        ssims.append(ssim)

        mask_pct = np.sum(mask) / mask.size * 100
        print(f'{case:<30} {psnr:>10.2f} {ssim:>8.4f} '
              f'{cbct.min():>10.1f} {cbct.max():>10.1f} '
              f'{pct.min():>10.1f} {pct.max():>10.1f} (mask {mask_pct:.1f}%)')

    print('-' * 90)
    print(f'{"均值":<30} {np.mean(psnrs):>10.2f} {np.mean(ssims):>8.4f}')
    print(f'{"标准差":<30} {np.std(psnrs):>10.2f} {np.std(ssims):>8.4f}')
    print(f'\nPSNR 范围: [{min(psnrs):.2f}, {max(psnrs):.2f}] dB')
    print(f'SSIM 范围: [{min(ssims):.4f}, {max(ssims):.4f}]')
    print(f'\n这个值是"直接用 CBCT 替代重建"的上限——任何重建模型应该超过这个 PSNR。')


if __name__ == '__main__':
    main()
