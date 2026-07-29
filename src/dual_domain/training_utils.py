"""Phase A/B/C 共用的可复现训练、采样、指标和 checkpoint 工具。"""

from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import ssim_3d_per_case
from src.view_protocol import uniform_view_indices


def validate_run_version(version: str) -> str:
    """要求显式使用 v1、v2 等运行版本，防止覆盖旧实验。"""
    if not re.fullmatch(r"v[1-9]\d*", version or ""):
        raise ValueError(f"run_version 必须形如 v1/v2，实际 {version!r}")
    return version


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """统一设置 Python、NumPy、CPU/CUDA PyTorch 随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """让每个 DataLoader worker 使用可复现且互不相同的 NumPy/Python seed。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """建立带独立 generator 和 worker seed 的 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        # 每个 epoch 重新按 generator 派生 worker seed。这样保存 generator
        # 状态后，Phase B 的随机角度采样也能从 checkpoint 精确续接。
        persistent_workers=False,
    )


def capture_loader_generator_state(loader: DataLoader) -> torch.Tensor:
    """保存 shuffle/worker seed 所使用的 DataLoader generator。"""
    if loader.generator is None:
        raise RuntimeError("DataLoader 未配置独立 generator")
    return loader.generator.get_state()


def restore_loader_generator_state(
    loader: DataLoader, state: torch.Tensor | None
) -> None:
    """恢复 DataLoader 的 shuffle 和 worker seed 序列。"""
    if state is None:
        return
    if loader.generator is None:
        raise RuntimeError("DataLoader 未配置独立 generator")
    loader.generator.set_state(state.cpu())


def prepare_run_directory(
    log_root: str | os.PathLike,
    run_name: str,
    *,
    resume: str | None,
) -> Path:
    """创建版本化目录；新训练拒绝复用非空目录。"""
    run_dir = Path(log_root).expanduser().resolve() / run_name
    if resume is None and run_dir.is_dir() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"运行目录非空：{run_dir}；请更换 run_version 或使用 --resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | os.PathLike, payload: dict) -> None:
    """使用 UTF-8 保存人类可读配置。"""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["cuda"]]
        )


def load_trusted_checkpoint(
    path: str | os.PathLike, map_location: str | torch.device = "cpu"
) -> dict:
    """加载本项目产生的完整 checkpoint。

    PyTorch 2.6 默认 ``weights_only=True``，完整 RNG/配置状态需要显式关闭。
    该函数只能用于用户自己训练、确认可信的 checkpoint。
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


def save_checkpoint(
    payload: dict,
    run_dir: Path,
    *,
    phase: str,
    version: str,
    kind: str,
    epoch: int | None = None,
) -> Path:
    """以 phase/version 命名保存 best、last 或周期 checkpoint。"""
    if kind == "epoch":
        if epoch is None:
            raise ValueError("周期 checkpoint 必须提供 epoch")
        name = f"phase_{phase}_epoch={epoch:04d}_{version}.pth"
    elif kind in {"best", "last"}:
        name = f"phase_{phase}_{kind}_{version}.pth"
    else:
        raise ValueError(f"不支持的 checkpoint kind={kind!r}")
    path = run_dir / name
    torch.save(payload, path)
    return path


def optimizer_step_due(
    batch_index: int, loader_length: int, accumulation_steps: int
) -> bool:
    """判断当前 micro-batch 是否结束一个梯度累积窗口。"""
    return (
        (batch_index + 1) % accumulation_steps == 0
        or batch_index + 1 == loader_length
    )


def accumulation_window_size(
    batch_index: int, loader_length: int, accumulation_steps: int
) -> int:
    """返回当前窗口真实 micro-batch 数，正确处理 epoch 尾部不足窗口。"""
    window_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, loader_length - window_start)


def volume_psnr_per_case(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """逐病例完整体积 PSNR。"""
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    peak = prediction.new_tensor(float(data_range))
    return 20.0 * torch.log10(peak / torch.sqrt(mse.clamp_min(1e-12)))


def projection_psnr_per_case(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """投影范围为 [-1,1]，因此 data_range=2。"""
    return volume_psnr_per_case(prediction, target, data_range=2.0)


@torch.no_grad()
def volume_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    compute_ssim: bool,
) -> dict[str, float]:
    """返回 batch 内逐病例平均的 PSNR 和可选 skimage 3D SSIM。"""
    metrics = {
        "psnr_sum": float(
            volume_psnr_per_case(prediction, target).sum()
        ),
        "cases": float(prediction.shape[0]),
    }
    if compute_ssim:
        metrics["ssim_sum"] = float(
            ssim_3d_per_case(prediction, target).sum()
        )
    return metrics


def select_views(
    projections: torch.Tensor,
    angles: torch.Tensor,
    n_views: int,
    *,
    random_subset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从已均匀建立的60/64-view基准集选择训练或固定部署子集。"""
    total = projections.shape[1]
    if not (0 < n_views <= total):
        raise ValueError(f"n_views={n_views} 不在 [1,{total}]")
    if n_views == total:
        indices = torch.arange(total, device=projections.device)
    elif random_subset:
        indices = torch.randperm(total, device=projections.device)[:n_views]
        indices = indices.sort().values
    else:
        indices = torch.tensor(
            uniform_view_indices(total, n_views),
            device=projections.device,
            dtype=torch.long,
        )
    return (
        projections.index_select(1, indices),
        angles.index_select(1, indices),
        indices,
    )


def select_cycle_views(
    base_projections: torch.Tensor,
    base_angles: torch.Tensor,
    input_indices: torch.Tensor,
    *,
    max_input_cycle_views: int,
    heldout_views: int,
    random_subset: bool = True,
) -> dict[str, torch.Tensor | None]:
    """为投影闭环选择少量输入角度和未输入角度，控制投影器开销。"""
    if max_input_cycle_views <= 0:
        raise ValueError("max_input_cycle_views 必须为正数")
    selected_count = min(max_input_cycle_views, input_indices.numel())
    if random_subset:
        positions = torch.randperm(
            input_indices.numel(), device=input_indices.device
        )[:selected_count]
    else:
        positions = torch.tensor(
            uniform_view_indices(input_indices.numel(), selected_count),
            device=input_indices.device,
            dtype=torch.long,
        )
    cycle_input_indices = input_indices.index_select(0, positions).sort().values

    all_indices = torch.arange(
        base_projections.shape[1], device=input_indices.device
    )
    available_mask = torch.ones(
        base_projections.shape[1],
        dtype=torch.bool,
        device=input_indices.device,
    )
    available_mask[input_indices] = False
    available = all_indices[available_mask]

    heldout_indices = None
    if heldout_views > 0 and available.numel() > 0:
        count = min(heldout_views, available.numel())
        if random_subset:
            held_positions = torch.randperm(
                available.numel(), device=available.device
            )[:count]
        else:
            held_positions = torch.tensor(
                uniform_view_indices(available.numel(), count),
                device=available.device,
                dtype=torch.long,
            )
        heldout_indices = available.index_select(
            0, held_positions
        ).sort().values

    result: dict[str, torch.Tensor | None] = {
        "input_projections": base_projections.index_select(
            1, cycle_input_indices
        ),
        "input_angles": base_angles.index_select(1, cycle_input_indices),
        "heldout_projections": None,
        "heldout_angles": None,
    }
    if heldout_indices is not None:
        result["heldout_projections"] = base_projections.index_select(
            1, heldout_indices
        )
        result["heldout_angles"] = base_angles.index_select(
            1, heldout_indices
        )
    return result


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """建立最小学习率为初始值1%的按 epoch 余弦调度器。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=min(group["lr"] for group in optimizer.param_groups) * 0.01,
    )


def mean_from_sums(sums: dict[str, float], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        raise ValueError("denominator 必须为正数")
    return {key: value / denominator for key, value in sums.items()}
