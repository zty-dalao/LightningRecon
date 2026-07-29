"""Phase B：用真实 CBCT 与 XIM 投影单独训练体素到投影模型。"""

from __future__ import annotations

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from src.dual_domain import (
    ApproximateProjectorGeometry,
    ForwardProjectorLoss,
    LearnedForwardProjector,
)
from src.dual_domain.training_utils import (
    accumulation_window_size,
    build_loader,
    capture_loader_generator_state,
    capture_rng_state,
    cosine_scheduler,
    load_trusted_checkpoint,
    optimizer_step_due,
    prepare_run_directory,
    projection_psnr_per_case,
    restore_loader_generator_state,
    restore_rng_state,
    save_checkpoint,
    save_json,
    set_global_seed,
    validate_run_version,
)
from src.thorax_fast_dataset import ThoraxFastDataset


PHASE = "B"
CHECKPOINT_FORMAT = 1


@torch.no_grad()
def evaluate(
    model: LearnedForwardProjector,
    loader,
    criterion: ForwardProjectorLoss,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    cases = 0
    for batch in loader:
        cbct = batch["cbct"].to(
            device, non_blocking=device.type == "cuda"
        )
        angles = batch["angles"].to(
            device, non_blocking=device.type == "cuda"
        )
        target = batch["projs"].to(
            device, non_blocking=device.type == "cuda"
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction = model(cbct, angles)
            losses = criterion(prediction, target)
        batch_size = cbct.shape[0]
        loss_sum += float(losses["total"]) * batch_size
        psnr_sum += float(
            projection_psnr_per_case(prediction, target).sum()
        )
        cases += batch_size
    return {
        "loss": loss_sum / cases,
        "psnr": psnr_sum / cases,
    }


def build_payload(
    *,
    epoch: int,
    model: LearnedForwardProjector,
    optimizer,
    scheduler,
    scaler,
    args,
    best_psnr: float,
    best_epoch: int,
    validation: dict | None,
    train_loader,
) -> dict:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "phase": PHASE,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "train_loader_generator_state": (
            capture_loader_generator_state(train_loader)
        ),
        "best_val_psnr": best_psnr,
        "best_epoch": best_epoch,
        "validation": validation,
        "run_version": args.run_version,
        "model_config": {
            "projection_size": list(args.projection_size),
            "integration_size": args.integration_size,
            "correction_channels": args.correction_channels,
            "geometry": {
                "dsd_mm": args.dsd,
                "dso_mm": args.dso,
                "detector_pixels": [1280, 320],
                "detector_spacing_mm": [0.336, 1.344],
                "voxel_spacing_mm": [2.0, 2.0, 2.0],
            },
        },
        "data_config": {
            "volume_source": "processed/images/cbct",
            "projection_views": args.views_per_case,
            "projection_clip": [0.0, 10.0],
            "projection_range": [-1.0, 1.0],
        },
    }


def train(args) -> None:
    validate_run_version(args.run_version)
    if min(
        args.epochs,
        args.batch_size,
        args.grad_accum,
        args.views_per_case,
        args.integration_size,
        args.eval_every,
        args.save_every,
    ) <= 0:
        raise ValueError(
            "训练轮数、batch、视角数、积分尺寸和间隔必须为正数"
        )
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    set_global_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    run_name = f"thorax_phaseB_projector_{args.run_version}"
    run_dir = prepare_run_directory(
        args.log_dir, run_name, resume=args.resume
    )
    writer = SummaryWriter(run_dir / "tensorboard")

    train_set = ThoraxFastDataset(
        args.data_root,
        split="train",
        volume_keys=("cbct",),
        projection_views=args.views_per_case,
        projection_size=tuple(args.projection_size),
        projection_sampling="random",
    )
    val_set = ThoraxFastDataset(
        args.data_root,
        split="val",
        volume_keys=("cbct",),
        projection_views=args.views_per_case,
        projection_size=tuple(args.projection_size),
        projection_sampling="uniform",
    )
    train_loader = build_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    val_loader = build_loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        device=device,
    )

    geometry = ApproximateProjectorGeometry(
        dsd_mm=args.dsd,
        dso_mm=args.dso,
    )
    model = LearnedForwardProjector(
        projection_size=tuple(args.projection_size),
        integration_size=args.integration_size,
        correction_channels=args.correction_channels,
        geometry=geometry,
    ).to(device)
    criterion = ForwardProjectorLoss(
        image_weight=1.0, gradient_weight=args.gradient_weight
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = cosine_scheduler(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_psnr = -float("inf")
    best_epoch = 0
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, device)
        if checkpoint.get("phase") != PHASE:
            raise ValueError("resume checkpoint 不是 Phase B")
        if checkpoint.get("run_version") != args.run_version:
            raise ValueError("resume run_version 与当前命令不一致")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if use_amp and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        restore_loader_generator_state(
            train_loader,
            checkpoint.get("train_loader_generator_state"),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_val_psnr", best_psnr))
        best_epoch = int(checkpoint.get("best_epoch", 0))

    config = vars(args).copy()
    config.update(
        {
            "phase": PHASE,
            "run_name": run_name,
            "train_cases": len(train_set),
            "val_cases": len(val_set),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    save_json(run_dir / "config.json", config)
    writer.add_text(
        "Run/Metadata",
        "\n".join(
            [
                "# Phase B: learned forward projector",
                f"- run_version: `{args.run_version}`",
                "- volume_source: `registered CBCT [0,1]`",
                "- target_projection: `real XIM attenuation [-1,1]`",
                f"- random_views_per_case: `{args.views_per_case}`",
                "- analytic_core: `differentiable parallel-beam approximation`",
            ]
        ),
        0,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        psnr_sum = 0.0
        cases = 0
        for batch_index, batch in enumerate(train_loader):
            cbct = batch["cbct"].to(
                device, non_blocking=device.type == "cuda"
            )
            angles = batch["angles"].to(
                device, non_blocking=device.type == "cuda"
            )
            target = batch["projs"].to(
                device, non_blocking=device.type == "cuda"
            )
            window = accumulation_window_size(
                batch_index, len(train_loader), args.grad_accum
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction = model(cbct, angles)
                losses = criterion(prediction, target)
                scaled_loss = losses["total"] / window
            scaler.scale(scaled_loss).backward()

            if optimizer_step_due(
                batch_index, len(train_loader), args.grad_accum
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = cbct.shape[0]
            cases += batch_size
            psnr_sum += float(
                projection_psnr_per_case(
                    prediction.detach(), target
                ).sum()
            )
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value) * batch_size
        scheduler.step()

        train_metrics = {
            name: value / cases for name, value in sums.items()
        }
        train_metrics["psnr"] = psnr_sum / cases
        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        writer.add_scalar(
            "Train/LearningRate", optimizer.param_groups[0]["lr"], epoch
        )

        validation = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(
                model, val_loader, criterion, device, use_amp
            )
            for name, value in validation.items():
                writer.add_scalar(f"Val/{name}", value, epoch)
            print(
                f"[B] E{epoch:04d} train={train_metrics['total']:.5f} "
                f"val_psnr={validation['psnr']:.3f}"
            )
            if validation["psnr"] > best_psnr:
                best_psnr = validation["psnr"]
                best_epoch = epoch
                payload = build_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    best_psnr=best_psnr,
                    best_epoch=best_epoch,
                    validation=validation,
                    train_loader=train_loader,
                )
                save_checkpoint(
                    payload,
                    run_dir,
                    phase=PHASE,
                    version=args.run_version,
                    kind="best",
                )
        else:
            print(
                f"[B] E{epoch:04d} train={train_metrics['total']:.5f} "
                f"train_psnr={train_metrics['psnr']:.3f}"
            )

        payload = build_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            best_psnr=best_psnr,
            best_epoch=best_epoch,
            validation=validation,
            train_loader=train_loader,
        )
        save_checkpoint(
            payload,
            run_dir,
            phase=PHASE,
            version=args.run_version,
            kind="last",
        )
        if epoch % args.save_every == 0:
            save_checkpoint(
                payload,
                run_dir,
                phase=PHASE,
                version=args.run_version,
                kind="epoch",
                epoch=epoch,
            )

    writer.close()
    print(
        f"Phase B 完成：best val PSNR={best_psnr:.3f}, epoch={best_epoch}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase B: train volume-to-projection model"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_version", required=True)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--views_per_case", type=int, default=6)
    parser.add_argument(
        "--projection_size", type=int, nargs=2, default=(128, 128)
    )
    parser.add_argument("--integration_size", type=int, default=96)
    parser.add_argument("--correction_channels", type=int, default=32)
    parser.add_argument("--dsd", type=float, default=1540.0)
    parser.add_argument("--dso", type=float, default=1000.0)
    parser.add_argument("--gradient_weight", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--resume", default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
