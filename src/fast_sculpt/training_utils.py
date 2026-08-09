"""新两阶段模型共用的轻量训练工具。"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def validate_run_version(version: str) -> str:
    """版本必须形如v1、v2，防止意外覆盖已有日志。"""
    if not re.fullmatch(r"v[1-9]\d*", version or ""):
        raise ValueError(f"run_version必须形如v1/v2，实际{version!r}")
    return version


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_loader(dataset, *, batch_size, shuffle, num_workers, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
        # 每轮重建worker，使dataset.set_projection_views立即生效。
        persistent_workers=False,
    )


def prepare_run_directory(log_root, run_name, *, resume):
    run_dir = Path(log_root).expanduser().resolve() / run_name
    if resume is None and run_dir.is_dir() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"运行目录非空：{run_dir}；请更换run_version或使用--resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def volume_psnr_per_case(prediction, target, data_range: float = 1.0):
    """在每个完整三维病例上求MSE，再转换为PSNR。"""
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    peak = prediction.new_tensor(float(data_range))
    return 20.0 * torch.log10(peak / torch.sqrt(mse.clamp_min(1e-12)))
