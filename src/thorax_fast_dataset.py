"""Thorax Fast 配对数据加载器。

该加载器直接兼容 ``data/thorax_fast`` 的原始目录布局：

* ``processed/projections/{case}.pickle``：真实 XIM 预处理投影；
* ``processed/images/ct/{case}.nii.gz``：pCT 监督标签；
* ``processed/images/cbct/{case}.nii.gz``：与投影对应的 CBCT 体积；
* ``splits.json`` 或 ``meta_info.json``：train/eval/test 病例划分。

与旧加载器相比，本实现只解码选中的基准视角，并按 pickle 的真实公式
``uint8 / 255 * projs_max`` 恢复对数衰减值，不再额外除以 0.2。
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.view_protocol import resolve_view_curriculum, uniform_view_indices


def _resolve_layout(data_root: str | os.PathLike) -> tuple[Path, Path]:
    """返回 ``(dataset_root, processed_root)``，兼容根目录和 processed 目录。"""
    root = Path(data_root).expanduser().resolve()
    if (root / "processed").is_dir():
        return root, root / "processed"
    if (root / "projections").is_dir() and (root / "images").is_dir():
        return root.parent, root
    raise FileNotFoundError(
        f"无法识别 Thorax Fast 目录 {root}；需要包含 processed/，"
        "或直接指向同时包含 projections/ 与 images/ 的目录。"
    )


def _load_split_mapping(dataset_root: Path, processed_root: Path) -> dict:
    """按优先级读取官方划分文件，不静默重新随机划分。"""
    candidates = (
        dataset_root / "splits.json",
        dataset_root / "meta_info.json",
        processed_root / "splits.json",
        processed_root / "meta_info.json",
    )
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        "未找到 splits.json 或 meta_info.json；为避免数据泄漏，不自动生成新划分。"
    )


def _periodic_angle_distance(first: float, last: float) -> float:
    """计算两个弧度角在 2π 周期上的最短距离。"""
    return abs((last - first + np.pi) % (2.0 * np.pi) - np.pi)


class ThoraxFastDataset(Dataset):
    """读取配对 CT/CBCT 与真实投影。

    Args:
        data_root: ``data/thorax_fast`` 或其 ``processed`` 子目录。
        split: ``train``、``val/eval`` 或 ``test``。
        volume_keys: 需要加载的体积，可从 ``("ct", "cbct")`` 中选择。
        projection_views: ``None`` 根据 final_view 选60/64基准集；``-1`` 加载
            全部有效视角；正整数从源网格均匀选取。
        final_view: projection_views=None 时用于选择内置6/8/10-view课程。
        projection_size: 模型使用的投影 ``(H,W)``。
        projection_clip: 对数衰减固定归一化范围；默认映射 [0,10] 到 [-1,1]。
        projection_sampling: ``uniform`` 用于基准集/验证，``random`` 用于
            Phase B 在每次读取时随机选择真实角度。
        volume_size: 期望的训练体积尺寸；不匹配时直接报错，不做隐式插值。
        drop_duplicate_endpoint: 去除 -π/+π 这种同方向周期端点。
        require_projections: CT 先验预训练可设 False；重建和投影器训练设 True。
    """

    VALID_VOLUMES = {"ct", "cbct"}

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        volume_keys: Iterable[str] = ("ct",),
        projection_views: int | None = None,
        final_view: int = 6,
        projection_size: tuple[int, int] = (128, 128),
        projection_clip: tuple[float, float] = (0.0, 10.0),
        projection_sampling: str = "uniform",
        volume_size: tuple[int, int, int] = (256, 256, 256),
        drop_duplicate_endpoint: bool = True,
        require_projections: bool = True,
    ):
        super().__init__()
        self.dataset_root, self.processed_root = _resolve_layout(data_root)
        self.projection_dir = self.processed_root / "projections"
        self.image_root = self.processed_root / "images"

        self.volume_keys = tuple(volume_keys)
        invalid = set(self.volume_keys) - self.VALID_VOLUMES
        if invalid:
            raise ValueError(f"不支持的 volume_keys={sorted(invalid)}")
        if not self.volume_keys and not require_projections:
            raise ValueError("volume_keys 为空且不读取投影，样本将没有任何数据")
        if projection_views is None:
            projection_views = resolve_view_curriculum(final_view)[0]
        if projection_views == 0 or projection_views < -1:
            raise ValueError("projection_views 必须为 -1 或正整数")
        if min(projection_size) <= 0 or min(volume_size) <= 0:
            raise ValueError("projection_size 和 volume_size 的各维必须为正数")
        if projection_clip[0] >= projection_clip[1]:
            raise ValueError("projection_clip 必须严格递增")
        if projection_sampling not in {"uniform", "random"}:
            raise ValueError(
                "projection_sampling 只能是 'uniform' 或 'random'"
            )

        self.projection_views = int(projection_views)
        self.final_view = int(final_view)
        self.projection_size = tuple(int(v) for v in projection_size)
        self.projection_clip = tuple(float(v) for v in projection_clip)
        self.projection_sampling = projection_sampling
        self.volume_size = tuple(int(v) for v in volume_size)
        self.drop_duplicate_endpoint = bool(drop_duplicate_endpoint)
        self.require_projections = bool(require_projections)

        mapping = _load_split_mapping(self.dataset_root, self.processed_root)
        split_key = split
        if split == "val" and "val" not in mapping:
            split_key = "eval"
        if split == "eval" and "eval" not in mapping:
            split_key = "val"
        if split_key not in mapping:
            raise KeyError(f"划分文件中不存在 split={split!r}")

        requested_cases = list(dict.fromkeys(mapping[split_key]))
        self.cases = [
            case for case in requested_cases
            if self._case_has_required_files(case)
        ]
        if not self.cases:
            raise RuntimeError(f"split={split!r} 没有满足所需文件条件的病例")

        missing = len(requested_cases) - len(self.cases)
        print(
            f"[ThoraxFastDataset] split={split_key}: {len(self.cases)} 例"
            f"（缺少所需文件并跳过 {missing} 例）"
        )

    def _case_has_required_files(self, case: str) -> bool:
        """检查当前训练视图所需文件是否齐全。"""
        if self.require_projections and not (
            self.projection_dir / f"{case}.pickle"
        ).is_file():
            return False
        return all(
            (self.image_root / key / f"{case}.nii.gz").is_file()
            for key in self.volume_keys
        )

    def __len__(self) -> int:
        return len(self.cases)

    def _load_volume(self, case: str, key: str) -> tuple[torch.Tensor, dict]:
        """读取 uint8 NIfTI，并从 (X,Y,Z) 转换成模型的 (D,H,W)。"""
        import nibabel as nib

        path = self.image_root / key / f"{case}.nii.gz"
        nii = nib.load(path)
        array = np.asanyarray(nii.dataobj)
        if array.shape != self.volume_size:
            raise ValueError(
                f"{case}/{key}: 期望体积 {self.volume_size}，实际 {array.shape}；"
                "拒绝在 Dataset 中隐式插值。"
            )
        value_min = float(np.min(array))
        value_max = float(np.max(array))
        if value_min < 0.0 or value_max > 255.0:
            raise ValueError(
                f"{case}/{key}: 训练体积必须是 uint8-like [0,255]，"
                f"实际范围 [{value_min:.3f},{value_max:.3f}]"
            )

        # NIfTI 存储为 (X,Y,Z)，模型统一使用 (D=Z,H=Y,W=X)。
        volume = np.transpose(array.astype(np.float32), (2, 1, 0)) / 255.0
        metadata = {
            f"{key}_affine": torch.from_numpy(
                np.asarray(nii.affine, dtype=np.float32)
            ),
            f"{key}_spacing": torch.tensor(
                nii.header.get_zooms()[:3], dtype=torch.float32
            ),
        }
        return torch.from_numpy(volume).unsqueeze(0), metadata

    def _load_projections(self, case: str) -> dict[str, torch.Tensor]:
        """解码对数衰减投影，去重周期端点并只缩放最终选择的视角。"""
        path = self.projection_dir / f"{case}.pickle"
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        stored = np.asarray(payload["projs"])
        angles = np.asarray(payload["angles"], dtype=np.float32)
        if stored.ndim != 3:
            raise ValueError(f"{case}: projs 应为 [K,H,W]，实际 {stored.shape}")
        if stored.shape[0] != angles.shape[0]:
            raise ValueError(
                f"{case}: 投影/角度数量不一致 "
                f"{stored.shape[0]} vs {angles.shape[0]}"
            )

        raw_source_views = int(stored.shape[0])
        valid_count = raw_source_views
        endpoint_dropped = False
        if (
            self.drop_duplicate_endpoint
            and raw_source_views > 1
            and _periodic_angle_distance(
                float(angles[0]), float(angles[-1])
            ) < 1e-4
        ):
            # np.linspace(-π,+π,K) 会把同一物理方向保存两次。
            valid_count -= 1
            endpoint_dropped = True

        n_keep = (
            valid_count
            if self.projection_views == -1
            else self.projection_views
        )
        if n_keep > valid_count:
            raise ValueError(
                f"{case}: 请求 {n_keep} views，但只有 {valid_count} 个有效方向"
            )
        if self.projection_sampling == "random" and n_keep < valid_count:
            # DataLoader worker 会继承独立 NumPy seed，因此多进程下仍可复现。
            indices = np.sort(
                np.random.choice(valid_count, size=n_keep, replace=False)
            ).astype(np.int64)
        else:
            indices = np.asarray(
                uniform_view_indices(valid_count, n_keep), dtype=np.int64
            )

        # 先选视角再转 float32，避免为约 200MB 的完整 uint8 堆栈创建副本。
        selected = stored[indices].astype(np.float32)
        attenuation = selected / 255.0 * float(payload["projs_max"])
        low, high = self.projection_clip
        normalized = np.clip(
            (attenuation - low) / (high - low), 0.0, 1.0
        )
        normalized = normalized * 2.0 - 1.0

        projections = torch.from_numpy(normalized).unsqueeze(1)
        if tuple(projections.shape[-2:]) != self.projection_size:
            projections = F.interpolate(
                projections,
                size=self.projection_size,
                mode="bilinear",
                align_corners=False,
            )

        return {
            "projs": projections.contiguous(),
            "angles": torch.from_numpy(angles[indices].copy()),
            "view_indices": torch.from_numpy(indices),
            "source_views": torch.tensor(valid_count, dtype=torch.long),
            "raw_source_views": torch.tensor(
                raw_source_views, dtype=torch.long
            ),
            "duplicate_endpoint_dropped": torch.tensor(endpoint_dropped),
            "projs_max": torch.tensor(
                float(payload["projs_max"]), dtype=torch.float32
            ),
            "base_views": torch.tensor(n_keep, dtype=torch.long),
            "final_view": torch.tensor(self.final_view, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict:
        case = self.cases[index]
        sample: dict = {"case_id": case}
        for key in self.volume_keys:
            volume, metadata = self._load_volume(case, key)
            sample[key] = volume
            sample.update(metadata)
        if self.require_projections:
            sample.update(self._load_projections(case))
        return sample
