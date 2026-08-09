"""Stage 2训练入口：冻结HU基底网络，用稀疏投影门控雕刻sCT。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.fast_sculpt.training_utils import (
    build_loader, prepare_run_directory, set_global_seed,
    validate_run_version, volume_psnr_per_case,
)
from src.fast_sculpt import (
    BaseSCTNet, ProjectionGuidedSculptor, SculptingLoss,
)
from src.thorax_fast_dataset import ThoraxFastDataset
from src.view_protocol import resolve_view_curriculum


def view_for_epoch(epoch, schedule, epochs_per_view, final_epochs):
    """固定课程：每个阶段训练epochs_per_view，最终协议额外巩固。"""
    del final_epochs  # 总epoch由调用者计算；最后阶段自然延续。
    index = min(epoch // epochs_per_view, len(schedule) - 1)
    return schedule[index]


def _run_epoch(base_model, sculptor, loader, loss_fn, device,
               optimizer=None, scaler=None):
    training = optimizer is not None
    base_model.eval()
    sculptor.train(training)
    sums = {
        "Loss/total": 0.0,
        "Metrics/PSNR_final": 0.0,
        "Metrics/PSNR_base": 0.0,
        "Diagnostics/gate_mean": 0.0,
    }
    cases = 0
    for batch in loader:
        cbct = batch["cbct"].to(device, non_blocking=True)
        ct = batch["ct"].to(device, non_blocking=True)
        projs = batch["projs"].to(device, non_blocking=True)
        angles = batch["angles"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=scaler is not None and scaler.is_enabled(),
        ):
            base_sct = base_model(cbct)["base_sct"]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=scaler is not None and scaler.is_enabled(),
        ):
            outputs = sculptor(base_sct, projs, angles)
            losses = loss_fn(outputs, base_sct, ct)
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sculptor.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        n = cbct.shape[0]
        sums["Loss/total"] += float(losses["total"].detach()) * n
        sums["Metrics/PSNR_final"] += float(volume_psnr_per_case(
            outputs["final_sct"].detach(), ct
        ).sum())
        sums["Metrics/PSNR_base"] += float(
            volume_psnr_per_case(base_sct, ct).sum()
        )
        sums["Diagnostics/gate_mean"] += float(
            outputs["evidence_gate"].detach().mean()
        ) * n
        for group in ("raw", "weighted"):
            for name, value in losses[group].items():
                key = f"Loss/{group}/{name}"
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
        cases += n
    return {key: value / cases for key, value in sums.items()}


def train(args):
    validate_run_version(args.run_version)
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = resolve_view_curriculum(
        args.final_view, args.view_schedule or None
    )
    total_epochs = len(schedule) * args.epochs_per_view + args.final_epochs
    run_dir = prepare_run_directory(
        args.log_root,
        f"stage2_sculpt_finalview={args.final_view}_{args.run_version}",
        resume=args.resume,
    )
    writer = SummaryWriter(run_dir)

    dataset_common = dict(
        data_root=args.data_root, volume_keys=("cbct", "ct"),
        final_view=args.final_view, projection_base_views=schedule[0],
        projection_size=tuple(args.projection_size),
        volume_size=tuple(args.volume_size), require_projections=True,
    )
    train_set = ThoraxFastDataset(
        split="train", projection_views=schedule[0],
        projection_sampling="nested_random", **dataset_common
    )
    val_set = ThoraxFastDataset(
        split="val", projection_views=args.final_view,
        projection_sampling="nested_uniform", **dataset_common
    )
    train_loader = build_loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, seed=args.seed, device=device
    )
    val_loader = build_loader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, seed=args.seed + 1, device=device
    )

    stage1 = torch.load(
        args.stage1_checkpoint, map_location=device, weights_only=False
    )
    base_config = stage1.get("model_config", {"base_channels": 4})
    base_model = BaseSCTNet(**base_config).to(device)
    base_model.load_state_dict(stage1["model"])
    base_model.requires_grad_(False)
    sculptor = ProjectionGuidedSculptor(
        base_channels=args.base_channels,
        projection_channels=args.projection_channels,
        evidence_size=args.evidence_size,
    ).to(device)
    loss_fn = SculptingLoss(
        laplacian=args.laplacian_weight,
        structural=args.structural_weight,
    )
    optimizer = torch.optim.AdamW(
        sculptor.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.lr * 0.01
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_psnr = 0, float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        sculptor.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_psnr = state.get("best_psnr", best_psnr)

    for epoch in range(start_epoch, total_epochs):
        current_views = view_for_epoch(
            epoch, schedule, args.epochs_per_view, args.final_epochs
        )
        train_set.set_projection_views(current_views)
        train_result = _run_epoch(
            base_model, sculptor, train_loader, loss_fn, device,
            optimizer, scaler
        )
        with torch.no_grad():
            val_result = _run_epoch(
                base_model, sculptor, val_loader, loss_fn, device,
                scaler=scaler
            )
        scheduler.step()
        for key, value in train_result.items():
            writer.add_scalar(f"Train/{key}", value, epoch)
        for key, value in val_result.items():
            writer.add_scalar(f"Val_finalview={args.final_view}/{key}", value, epoch)
        writer.add_scalar("Train/Protocol/views", current_views, epoch)
        writer.add_scalar("Train/Optimizer/LR", optimizer.param_groups[0]["lr"], epoch)
        val_psnr = val_result["Metrics/PSNR_final"]
        payload = {
            "model": sculptor.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_psnr": max(best_psnr, val_psnr),
            "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
            "model_config": {
                "base_channels": args.base_channels,
                "projection_channels": args.projection_channels,
                "evidence_size": args.evidence_size,
            },
            "schedule": schedule, "args": vars(args),
        }
        torch.save(payload, run_dir / "stage2_last.pth")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(payload, run_dir / "stage2_best.pth")
        print(
            f"[Stage2 {epoch + 1:03d}/{total_epochs} views={current_views}] "
            f"loss={train_result['Loss/total']:.5f} "
            f"valPSNR={val_psnr:.2f}dB "
            f"base={val_result['Metrics/PSNR_base']:.2f}dB"
        )
    writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--run_version", default="v1")
    parser.add_argument("--log_root", default="logs")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--final_view", type=int, choices=(6,8,10), default=6)
    parser.add_argument("--view_schedule", default="")
    parser.add_argument("--epochs_per_view", type=int, default=30)
    parser.add_argument("--final_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=4)
    parser.add_argument("--projection_channels", type=int, default=24)
    parser.add_argument("--evidence_size", type=int, default=32)
    parser.add_argument("--projection_size", type=int, nargs=2, default=(128,128))
    parser.add_argument("--volume_size", type=int, nargs=3, default=(256,256,256))
    parser.add_argument("--laplacian_weight", type=float, default=0.12)
    parser.add_argument("--structural_weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_amp", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
