"""
数据集加载器: 从 thorax_fast/ 目录加载 CBCT 投影、CBCT 重建体和 pCT 体素。

目录结构 (data/thorax_fast/):
  projection_npy/{case_id}/Proj_00000.npy ... Proj_00490.npy  ← 491 张投影 (320,1280) float32
  cbct/{case_id}.nii.gz                                       ← CBCT 重建体 (512,512,110)
  ct/{case_id}.nii.gz                                         ← pCT 标签体 (512,512,110)

所有三个目录下同一病例使用完全相同的文件名 stem (如 2026-06-04_065713)。
"""

import os                                                       # 路径操作、文件枚举
import numpy as np                                              # 数组操作
import torch                                                    # PyTorch 张量
from torch.utils.data import Dataset                            # 数据集基类


class PairedCBCTDataset(Dataset):
    """
    配对 CBCT-pCT 数据集，用于稀疏视角 CBCT 重建 / CBCT→pCT 转换。

    每个样本包含:
      - projs:   (N_view, H_proj, W_proj) 投影正弦图 (已做 -log 归一化)
      - cbct_vol: (D, H, W) 全视角 CBCT 重建体素
      - ct_vol:   (D, H, W) 规划 CT (pCT) 体素标签

    Args:
        data_root:     thorax_fast/ 目录的绝对路径
        split:         'train' | 'val' | 'test' — 数据集划分
        n_views:       加载的投影数量 (训练=491, 推理可≤6)
        proj_size:     投影图 resize 目标 (H, W)，默认 (256, 256)
        vol_size:      体素 resize 目标 (D, H, W)，默认 (128, 128, 128)
        ct_range:      CT 值裁剪范围 [min_hu, max_hu] HU
        val_ratio:     验证集比例 (仅当无外部划分文件时生效)
        seed:          随机种子，保证划分可复现
    """

    def __init__(
        self,
        data_root,                                                # thorax_fast/ 根目录路径
        split='train',                                            # 数据集划分: train/val/test
        n_views=491,                                              # 每样本加载投影数
        proj_size=(256, 256),                                     # 投影 resize 目标 (H, W)
        vol_size=(128, 128, 128),                                 # 体素 resize 目标 (D, H, W)
        ct_range=(-1000, 1000),                                   # CT 值 HU 裁剪窗口
        val_ratio=0.1,                                            # 验证集占比
        seed=42,                                                  # 随机种子
    ):
        super().__init__()
        # ---- 保存配置 ----
        self.data_root = data_root                                # 数据根目录
        self.split = split                                        # 当前划分名
        self.n_views = n_views                                    # 目标投影数
        self.proj_size = proj_size                                # 投影尺寸 (H, W)
        self.vol_size = vol_size                                  # 体素尺寸 (D, H, W)
        self.ct_min, self.ct_max = ct_range                       # CT 值裁剪边界
        self.val_ratio = val_ratio                                # 验证集比例

        # ---- 定义三个子目录路径 ----
        proj_dir = os.path.join(data_root, 'projection_npy')      # 投影 .npy 目录
        cbct_dir = os.path.join(data_root, 'cbct')                # CBCT 体素目录
        ct_dir   = os.path.join(data_root, 'ct')                  # pCT 体素目录

        # ---- 检查数据目录是否存在 ----
        for d, name in [(proj_dir, 'projection_npy'),
                         (cbct_dir, 'cbct'),
                         (ct_dir, 'ct')]:
            if not os.path.isdir(d):                              # 目录不存在
                raise FileNotFoundError(
                    f'缺少数据目录: {d}。请确认 thorax_fast/ 下包含 '
                    f'projection_npy/、cbct/、ct/ 三个子目录。'
                )

        # ---- 发现所有病例 ----
        # 以 projection_npy 下的子目录名为准 (每个病例一个文件夹)
        all_cases = sorted(os.listdir(proj_dir))                  # 按字母排序，保证确定性

        # 过滤: 三个目录下都必须存在对应文件
        valid_cases = []                                          # 存放有效病例名
        for case in all_cases:                                    # 遍历每个候选病例
            proj_case_dir = os.path.join(proj_dir, case)          # 投影子目录路径
            if not os.path.isdir(proj_case_dir):                  # 跳过非目录项
                continue
            cbct_path = os.path.join(cbct_dir, f'{case}.nii.gz')  # CBCT 文件路径
            ct_path   = os.path.join(ct_dir,   f'{case}.nii.gz')  # pCT 文件路径
            if os.path.exists(cbct_path) and os.path.exists(ct_path):  # 两个体素文件都存在
                valid_cases.append(case)                          # 加入有效列表

        if not valid_cases:                                       # 一个有效病例都没有
            raise RuntimeError(
                f'未找到任何有效病例。请确认 projection_npy/ 下的每个子目录在 '
                f'cbct/ 和 ct/ 中都有对应的 .nii.gz 文件。'
            )

        # ---- 数据集划分 (train / val / test) ----
        # 优先使用外部 meta_info.json，否则自动按比例划分
        meta_path = os.path.join(data_root, 'meta_info.json')     # 外部划分文件路径
        if os.path.exists(meta_path):                             # 如果存在外部划分
            import json                                           # JSON 解析
            with open(meta_path, 'r') as f:
                meta = json.load(f)                               # 读取划分字典
            if split in meta:                                     # 存在该划分的键
                self.cases = [c for c in meta[split] if c in valid_cases]  # 取交集
                print(f'[PairedCBCTDataset] 从 meta_info.json 加载 '
                      f'[{split}]: {len(self.cases)} 例')
            else:                                                 # 键不存在则回退
                self.cases = valid_cases
                print(f'[PairedCBCTDataset] meta_info.json 中无 '
                      f'"{split}" 键，使用全部 {len(self.cases)} 例')
        else:                                                     # 无外部文件，自动划分
            np.random.seed(seed)                                  # 固定随机种子
            idx = np.random.permutation(len(valid_cases))         # 随机打乱索引
            n_val = max(1, int(len(valid_cases) * val_ratio))     # 验证集大小 (至少 1 例)
            n_test = max(1, int(len(valid_cases) * val_ratio))    # 测试集大小 (至少 1 例)
            if split == 'train':                                  # 训练集
                self.cases = [valid_cases[i] for i in idx[:-n_val - n_test]]
            elif split == 'val':                                  # 验证集
                self.cases = [valid_cases[i] for i in idx[-n_val - n_test:-n_test]]
            else:                                                 # 测试集
                self.cases = [valid_cases[i] for i in idx[-n_test:]]
            print(f'[PairedCBCTDataset] 自动划分 [{split}]: '
                  f'{len(self.cases)} 例 (总 {len(valid_cases)} 例)')

        # ---- 缓存子目录路径 (避免每次 __getitem__ 重新拼接) ----
        self._proj_dir = proj_dir                                 # 投影根目录
        self._cbct_dir = cbct_dir                                 # CBCT 体素目录
        self._ct_dir   = ct_dir                                   # pCT 体素目录

        # ---- 预扫描投影数量 ----
        if len(self.cases) > 0:                                   # 至少有一个病例
            sample_dir = os.path.join(proj_dir, self.cases[0])    # 第一个病例的投影目录
            self._total_projs = len(                              # 统计 .npy 文件数
                [f for f in os.listdir(sample_dir) if f.endswith('.npy')]
            )
            print(f'[PairedCBCTDataset] 每例投影数: {self._total_projs}')
        else:
            self._total_projs = 0

    # =====================================================================
    # 基础属性
    # =====================================================================

    def __len__(self):
        """返回数据集大小 (病例数)。"""
        return len(self.cases)                                    # 有效病例总数

    # =====================================================================
    # 投影加载
    # =====================================================================

    def _load_projections(self, case_id):
        """
        加载单个病例的所有投影图，均匀采样 n_views 张，resize 并归一化。

        投影原始格式: float32 (320, 1280)，已做 -log(I/I₀) 归一化，值域约 [0, ~10]。
        处理流程: 裁剪 → [0,1]归一化 → resize → 堆叠。

        Args:
            case_id: 病例目录名 (如 "2026-06-04_065713")

        Returns:
            np.ndarray: shape (n_views, H_proj, W_proj), float32, 值域 [0, 1]
        """
        # ---- 列出所有 .npy 投影文件 ----
        proj_case_dir = os.path.join(self._proj_dir, case_id)     # 病例投影目录
        files = sorted(                                           # 按文件名排序
            [f for f in os.listdir(proj_case_dir) if f.endswith('.npy')]
        )                                                         # 结果: Proj_00000 ~ Proj_00490
        total = len(files)                                        # 实际投影总数 (应为 491)

        # ---- 均匀采样 n_views 个视角 ----
        if self.n_views >= total:                                 # 请求数 ≥ 实际数
            indices = list(range(total))                          # 全部使用
        else:                                                     # 请求数 < 实际数
            # 等间隔采样，覆盖 360° 全角度范围
            indices = np.linspace(                                # 在 [0, total) 内均匀取 n_views 个点
                0, total, self.n_views, endpoint=False, dtype=int
            )

        # ---- 逐张加载 ----
        projs = []                                                # 投影列表
        for idx in indices:                                       # 遍历每个采样索引
            fpath = os.path.join(proj_case_dir, files[idx])       # 完整文件路径
            arr = np.load(fpath)                                  # 加载 float32 数组 (320, 1280)
            # arr 已经做过 -log(I/I0) 归一化，值域约 [0, ~10]
            arr = np.clip(arr, 0.0, 10.0)                         # 裁剪异常值到 [0, 10]
            arr = arr / 5.0 - 1.0                                  # 归一化到 [-1, 1] (与 sin/cos [-1,1] 对齐)
            # Resize 到目标尺寸 (PIL 比 cv2 更稳定)
            from PIL import Image                                 # PIL 图像处理库
            img = Image.fromarray(arr.astype(np.float32))         # numpy → PIL Image
            # PIL resize 参数顺序是 (W, H)，所以用 [::-1]
            img = img.resize(self.proj_size[::-1], Image.BILINEAR)  # 双线性插值 resize
            arr = np.array(img, dtype=np.float32)                 # PIL → numpy
            projs.append(arr)                                     # 加入列表

        return np.stack(projs, axis=0)                            # 堆叠为 (n_views, H, W)

    # =====================================================================
    # 体素加载 (CBCT / pCT)
    # =====================================================================

    def _load_volume(self, case_id, vol_type='ct'):
        """
        加载 .nii.gz 体素文件，裁剪 CT 值，resize 到统一尺寸。

        nibabel 返回 (x, y, z) 即 (W, H, D)，统一转为 (D, H, W) 方便 3D 卷积。

        Args:
            case_id:  病例名 (如 "2026-06-04_065713")
            vol_type: 'ct' (pCT 标签) 或 'cbct' (CBCT 重建)

        Returns:
            np.ndarray: shape (D, H, W), float32, 归一化到 [0, 1]
        """
        import nibabel as nib                                     # NIfTI 读写库

        # ---- 确定文件路径 ----
        vol_dir = self._ct_dir if vol_type == 'ct' else self._cbct_dir  # 选择目录
        path = os.path.join(vol_dir, f'{case_id}.nii.gz')         # 完整路径

        # ---- 加载体素数据 ----
        nii = nib.load(path)                                      # 读取 NIfTI 文件
        vol = nii.get_fdata().astype(np.float32)                  # 获取 numpy 数组

        # nibabel 返回 (x, y, z) 即 (W, H, D)，转为 (D, H, W)
        vol = np.transpose(vol, (2, 1, 0))                        # (W,H,D) → (D,H,W)

        # ---- 裁剪 CT 值到目标窗口 ----
        vol = np.clip(vol, self.ct_min, self.ct_max)              # 裁剪到 [ct_min, ct_max]

        # ---- 归一化到 [0, 1] ----
        vol = (vol - self.ct_min) / (self.ct_max - self.ct_min)   # 线性映射到 [0, 1]

        # ---- Resize 到统一尺寸 ----
        from scipy.ndimage import zoom                            # 3D 缩放 (需 scipy)
        current_shape = np.array(vol.shape, dtype=np.float32)     # 当前形状 (D, H, W)
        target_shape = np.array(self.vol_size, dtype=np.float32)  # 目标形状
        zoom_factors = target_shape / current_shape               # 各轴缩放因子
        vol = zoom(vol, zoom_factors, order=1)                    # 三线性插值缩放

        return vol.astype(np.float32)                             # 返回 float32

    # =====================================================================
    # 主接口: 取一个样本
    # =====================================================================

    def __getitem__(self, index):
        """
        返回第 index 个样本的完整数据字典。

        Returns:
            dict:
                'case_id':  str                          — 病例标识
                'projs':    Tensor (V, 1, H, W)          — 投影图
                'cbct':     Tensor (1, D, H, W)          — CBCT 重建体
                'ct':       Tensor (1, D, H, W)          — pCT 标签体
        """
        case_id = self.cases[index]                               # 当前病例名

        # ---- 加载投影 ----
        projs = self._load_projections(case_id)                   # (V, H, W) float32

        # ---- 加载体素 ----
        cbct_vol = self._load_volume(case_id, 'cbct')             # (D, H, W) float32
        ct_vol   = self._load_volume(case_id, 'ct')               # (D, H, W) float32

        # ---- 转为 PyTorch 张量并添加通道维度 ----
        return {
            'case_id': case_id,                                   # 病例名 (调试用)
            'projs': torch.from_numpy(projs).unsqueeze(1),        # (V, 1, H, W)
            'cbct':  torch.from_numpy(cbct_vol).unsqueeze(0),     # (1, D, H, W)
            'ct':    torch.from_numpy(ct_vol).unsqueeze(0),       # (1, D, H, W)
        }


# =====================================================================
# 测试入口: 直接运行此文件可验证数据加载是否正常
# =====================================================================

if __name__ == '__main__':
    data_root = '/home/zty20020112/workspace/LightningRecon/data/thorax_fast'  # 默认数据路径

    print('=' * 60)
    print('测试 PairedCBCTDataset')
    print('=' * 60)

    # ---- 创建训练集 (全 491 投影) ----
    train_set = PairedCBCTDataset(
        data_root=data_root,                                     # 数据根目录
        split='train',                                            # 训练集
        n_views=491,                                              # 使用全部投影
        proj_size=(256, 256),                                     # 投影 resize 尺寸
        vol_size=(128, 128, 128),                                 # 体素 resize 尺寸
    )
    print(f'训练集大小: {len(train_set)} 例')

    # ---- 取第一个样本验证形状 ----
    sample = train_set[0]                                         # 取第 0 个样本
    print(f'\n样本 case_id: {sample["case_id"]}')
    print(f'projs  形状: {sample["projs"].shape}   '
          f'dtype: {sample["projs"].dtype}')
    print(f'cbct   形状: {sample["cbct"].shape}    '
          f'dtype: {sample["cbct"].dtype}')
    print(f'ct     形状: {sample["ct"].shape}      '
          f'dtype: {sample["ct"].dtype}')
    print(f'projs  值域: [{sample["projs"].min():.4f}, '
          f'{sample["projs"].max():.4f}]')
    print(f'cbct   值域: [{sample["cbct"].min():.4f}, '
          f'{sample["cbct"].max():.4f}]')
    print(f'ct     值域: [{sample["ct"].min():.4f}, '
          f'{sample["ct"].max():.4f}]')

    # ---- 创建测试集 (仅 6 投影，模拟推理场景) ----
    test_set = PairedCBCTDataset(
        data_root=data_root,
        split='test',
        n_views=6,                                                # 推理用 6 视角
        proj_size=(256, 256),
        vol_size=(128, 128, 128),
    )
    print(f'\n测试集大小: {len(test_set)} 例')
    sample_test = test_set[0]
    print(f'测试样本 projs 形状: {sample_test["projs"].shape}')   # 应为 (6, 1, 256, 256)

    print('\n✓ 数据加载测试通过')
