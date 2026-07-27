"""
数据集加载器: 从 thorax/ 目录加载投影 (pickle) 和 CT 体素 (NIfTI)。

目录结构 (thorax/):
  projections/{case_id}.pickle        ← 投影数据 (uint8, 490帧, 1280×320)
  images/ct/{case_id}.nii.gz          ← CT 标签体 (256³, uint8 [0,255])
  meta_info.json                      ← 数据集划分 + 路径模板
  splits.json                         ← 备用划分文件

投影 pickle 格式:
  data['projs']      — uint8  [K, W, H]  (K≈490, W=1280, H=320)
  data['angles']     — float  [K,] 弧度
  data['projs_max']  — float  归一化系数
"""

# 标准库：路径处理、模块兼容、pickle 投影读取和 JSON 划分读取。
import os, sys, pickle, json
# NumPy 负责数组预处理，PyTorch 负责输出训练张量。
import numpy as np
import torch
from torch.utils.data import Dataset
# PIL 仅用于逐张二维投影的双线性缩放。
from PIL import Image
# 训练、验证和推理共享同一个等间隔索引公式。
from src.view_protocol import uniform_view_indices

# NumPy pickle 兼容 (旧版 numpy._core 别名)
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = np.core.numeric
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = np.core.multiarray


class ThoraxCTDataset(Dataset):
    """
    thorax 数据集: 投影 pickle → CT 重建。

    每个样本:
      - projs:  (V, 1, H_proj, W_proj)  投影图, float32 [-1, 1]
      - ct:     (1, D, H, W)             CT 标签体, float32 [0, 1]

    Args:
        data_root:  thorax/ 目录绝对路径
        split:      'train' | 'val' | 'test'
        n_views:    加载投影数; -1=全部
        proj_size:  投影 resize (H, W), 默认 (256, 256)
        vol_size:   体素 resize (D, H, W), 默认 (128, 128, 128)
        val_ratio:  验证集比例 (仅无 splits.json 时使用)
        seed:       随机种子
    """
    def __init__(self, data_root, split='train', n_views=-1,
                 proj_size=(256, 256), vol_size=(128, 128, 128),
                 val_ratio=0.1, seed=42, expected_source_views=None):
        """建立病例列表并预扫描每个病例的投影数量与角度。

        初始化阶段不会把全部投影和 CT 常驻内存；真正的数据读取发生在
        ``__getitem__``。``expected_source_views=None`` 表示允许逐病例视角数不同。
        """
        super().__init__()
        # 保存预处理配置，保证 train/val/test 使用相同的尺寸约定。
        self.data_root = data_root
        self.split = split
        self.n_views = n_views
        self.proj_size = proj_size
        self.vol_size = vol_size
        self.expected_source_views = expected_source_views

        # ---- 目录 ----
        proj_dir = os.path.join(data_root, 'projections')
        ct_dir   = os.path.join(data_root, 'images', 'ct')
        for d, n in [(proj_dir, 'projections'), (ct_dir, 'images/ct')]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f'缺少目录: {d} ({n}/)')

        # ---- 发现病例 (以 projections/ 下的 pickle 为准) ----
        # 排序保证同一数据目录在不同机器上的病例遍历顺序一致。
        all_cases = sorted(
            f.replace('.pickle', '')
            for f in os.listdir(proj_dir) if f.endswith('.pickle')
        )
        # 训练 Dataset 只接纳投影和 CT 标签同时存在的病例。
        valid_cases = []
        for case in all_cases:
            ct_path = os.path.join(ct_dir, f'{case}.nii.gz')
            if os.path.exists(ct_path):
                valid_cases.append(case)
        if not valid_cases:
            raise RuntimeError('未找到有效病例 (需同时存在 projection pickle 和 CT nii.gz)')

        # ---- 划分 ----
        splits_path = os.path.join(data_root, 'splits.json')
        meta_path   = os.path.join(data_root, 'meta_info.json')
        split_source = None
        metadata_found = False
        # 优先读取 splits.json；不存在对应键时再尝试 meta_info.json。
        for sp in [splits_path, meta_path]:
            if os.path.exists(sp):
                metadata_found = True
                with open(sp) as f:
                    d = json.load(f)
                if split in d:
                    # 元数据中缺少实体文件的病例会被排除。
                    self.cases = [c for c in d[split] if c in valid_cases]
                    split_source = os.path.basename(sp)
                    break
                # splits.json 可能用 'eval' 而非 'val'
                alias = 'eval' if split == 'val' else ('val' if split == 'eval' else None)
                if alias and alias in d:
                    self.cases = [c for c in d[alias] if c in valid_cases]
                    split_source = os.path.basename(sp)
                    break
        if split_source is None and metadata_found:
            raise KeyError(
                f'Existing split metadata does not define split={split!r}; '
                'refusing to silently create a different random partition.'
            )
        if split_source is None:
            # 自动划分
            # 仅在完全没有划分元数据时，按固定 seed 生成临时划分。
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(valid_cases))
            n_val = max(1, int(len(valid_cases) * val_ratio))
            n_test = max(1, int(len(valid_cases) * val_ratio))
            if n_val + n_test >= len(valid_cases):
                n_val = n_test = max(1, len(valid_cases) // 3)
            mapping = {
                'train': idx[:-n_val - n_test],
                'val':   idx[-n_val - n_test:-n_test],
                'test':  idx[-n_test:],
            }
            self.cases = [valid_cases[i] for i in mapping.get(split, [])]
            split_source = 'auto'
        if not self.cases:
            raise RuntimeError(f'Split {split!r} contains no valid cases')
        print(f'[ThoraxCTDataset] [{split}] {len(self.cases)} 例 (划分: {split_source})')

        self._proj_dir = proj_dir
        self._ct_dir = ct_dir

        # ---- 预扫描投影数 ----
        # 预扫描结果用于启动时验证课程最大视角数能否被所有病例支持。
        self._max_projs = 0
        self._min_projs = float('inf')
        self._case_angles = {}
        self._case_source_views = {}
        for case in self.cases:
            with open(os.path.join(proj_dir, f'{case}.pickle'), 'rb') as f:
                data = pickle.load(f)
            # 投影帧数和角度数必须一一对应。
            n = len(data['projs'])
            angles = np.asarray(data['angles'])
            if len(angles) != n:
                raise ValueError(
                    f'{case}: projections/angles length mismatch '
                    f'({n} vs {len(angles)})'
                )
            if expected_source_views is not None and n != expected_source_views:
                raise ValueError(
                    f'{case}: expected exactly {expected_source_views} source '
                    f'views, found {n}'
                )
            if self.n_views > n:
                raise ValueError(
                    f'{case}: requested {self.n_views} views from only {n}'
                )
            self._max_projs = max(self._max_projs, n)
            self._min_projs = min(self._min_projs, n)
            self._case_angles[case] = angles
            self._case_source_views[case] = n
        if self._min_projs == self._max_projs:
            source_summary = str(self._max_projs)
        else:
            source_summary = f'{self._min_projs}..{self._max_projs}（逐病例）'
        print(f'[ThoraxCTDataset] 原始投影数: {source_summary}')

    def __len__(self):
        return len(self.cases)

    @property
    def max_views(self):
        return self._max_projs

    @property
    def min_views(self):
        return self._min_projs

    @property
    def source_view_counts(self):
        """返回副本，防止外部代码意外修改 Dataset 内部统计。"""
        return dict(self._case_source_views)

    def ct_path(self, case_id):
        """返回当前 split 内病例的 CT 路径。"""
        if case_id not in self.cases:
            raise ValueError(f'Case {case_id!r} is not part of split {self.split!r}')
        return os.path.join(self._ct_dir, f'{case_id}.nii.gz')

    # =====================================================================
    # 投影加载 (pickle)
    # =====================================================================
    @staticmethod
    def _uniform_indices(total, n_views):
        """Return deterministic, approximately uniform indices over a full rotation."""
        if n_views < 0 or n_views >= total:
            return np.arange(total, dtype=np.int64)
        return np.asarray(
            uniform_view_indices(total,n_views),dtype=np.int64
        )

    def _load_projections(self, case_id):
        """读取、归一化并缩放一个病例的投影，同时保留真实角度和原始索引。"""
        path = os.path.join(self._proj_dir, f'{case_id}.pickle')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        projs = data['projs'].astype(np.float32)      # uint8 → float32 [K, W, H]
        angles = np.asarray(data['angles'], dtype=np.float32)
        projs_max = float(data['projs_max'])
        # total 是该病例实际拥有的源投影数，不假设固定为 491。
        total = len(projs)
        if len(angles) != total:
            raise ValueError(
                f'{case_id}: projections/angles length mismatch '
                f'({total} vs {len(angles)})'
            )

        # ---- 采样 ----
        # n_views=-1 时 indices 覆盖全部源投影；否则直接从源网格均匀采样。
        indices = self._uniform_indices(total, self.n_views)

        # ---- 逐帧处理 ----
        out = []
        for idx in indices:
            # 取出与 angles[idx] 配对的单张投影。
            arr = projs[idx]                            # (W, H) float32, 原始 uint8 值
            arr = arr / 255.0                           # [0, 255] → [0, 1]
            arr = arr * projs_max / 0.2                 # 反归一化 (与 DeepSparse 一致)
            arr = np.clip(arr, 0.0, 10.0)               # 裁剪
            arr = arr / 5.0 - 1.0                       # → [-1, 1]
            img = Image.fromarray(arr)
            img = img.resize(self.proj_size[::-1], Image.BILINEAR)
            out.append(np.array(img, dtype=np.float32))
        # 同时返回原始索引，后续可在不同源视角数病例中重建同一嵌套协议。
        return np.stack(out, axis=0), angles[indices], indices, total

    # =====================================================================
    # CT 体素加载 (NIfTI uint8)
    # =====================================================================
    def _load_ct(self, case_id):
        """读取 uint8-like CT 标签，转换到 [0,1] 并缩放到 ``vol_size``。"""
        path = self.ct_path(case_id)
        try:
            import nibabel as nib
            nii = nib.load(path)
            vol = nii.get_fdata().astype(np.float32)     # (W, H, D)
            vol = np.transpose(vol, (2, 1, 0))           # → (D, H, W)
        except ModuleNotFoundError:
            import SimpleITK as sitk
            vol = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

        # 在归一化前验证存储域，避免把原始 HU 错当作 uint8 再除以 255。
        value_min = float(np.nanmin(vol))
        value_max = float(np.nanmax(vol))
        if not np.isfinite(value_min) or not np.isfinite(value_max):
            raise ValueError(f'{case_id}: CT contains NaN or infinite values')
        if value_min < -1e-3 or value_max > 255.0 + 1e-3:
            raise ValueError(
                f'{case_id}: CT range [{value_min:.3f}, {value_max:.3f}] is '
                'not uint8-like [0,255]. Convert it explicitly or change the '
                'normalization instead of silently dividing HU values by 255.'
            )

        # uint8-like [0,255] → normalized [0,1]
        vol = vol / 255.0

        # resize
        from scipy.ndimage import zoom
        # 每个轴独立计算缩放倍率，输出顺序始终保持 (D,H,W)。
        zoom_factors = np.array(self.vol_size, dtype=np.float32) / np.array(vol.shape, dtype=np.float32)
        vol = zoom(vol, zoom_factors, order=1)
        return vol.astype(np.float32)

    def __getitem__(self, index):
        """按病例索引返回模型输入、物理角度、源网格索引和监督 CT。"""
        case_id = self.cases[index]
        (
            projs,
            angles,
            view_indices,
            source_views,
        ) = self._load_projections(case_id)
        ct = self._load_ct(case_id)                       # (D, H, W)
        return {
            'case_id': case_id,
            'projs': torch.from_numpy(projs).unsqueeze(1),  # (V, 1, H, W)
            'angles': torch.from_numpy(angles),              # (V,), radians
            'view_indices': torch.from_numpy(view_indices),  # (V,), original grid
            'source_views': torch.tensor(source_views, dtype=torch.long),
            'ct':    torch.from_numpy(ct).unsqueeze(0),      # (1, D, H, W)
        }
