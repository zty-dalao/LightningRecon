"""Phase B：用真实 CBCT 与 XIM 投影单独训练体素到投影模型。"""

from __future__ import annotations

import argparse
import math
import shutil

import torch
import torch.nn.functional as F
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
from src.thorax_fast_dataset import (
    ThoraxFastDataset,
    resolve_thorax_fast_root,
)


PHASE = "B"
CHECKPOINT_FORMAT = 2


def _cosine_interpolate(
    start: float,
    end: float,
    index: int,
    length: int,
) -> float:
    """Return a finite half-cosine interpolation from ``start`` to ``end``.

    The value is derived only from the epoch index.  This makes an extended
    run independent of the stale ``T_max=150`` stored in a legacy scheduler.
    """
    if length <= 1:
        return float(end)
    progress = min(max(index / float(length - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end + (start - end) * cosine)


def phase_b_stage(epoch: int, base_epochs: int) -> int:
    """B1 fits projection intensity; B2 fine-tunes projection edges."""
    return 1 if int(epoch) <= int(base_epochs) else 2


def learning_rate_for_epoch(epoch: int, args) -> float:
    """Two independent cosine segments with a deliberate B2 LR restart."""
    if epoch <= args.base_epochs:
        return _cosine_interpolate(
            args.lr, args.min_lr, epoch - 1, args.base_epochs
        )
    edge_epochs = args.epochs - args.base_epochs
    return _cosine_interpolate(
        args.edge_lr,
        args.min_lr,
        epoch - args.base_epochs - 1,
        edge_epochs,
    )


def gradient_weight_for_epoch(epoch: int, args) -> float:
    """Half-cosine ramp from the B1 edge weight to the B2 target weight."""
    if epoch <= args.base_epochs:
        return float(args.gradient_weight)
    ramp_index = epoch - args.base_epochs - 1
    if ramp_index >= args.edge_gradient_ramp_epochs:
        return float(args.edge_gradient_weight)
    return _cosine_interpolate(
        args.gradient_weight,
        args.edge_gradient_weight,
        ramp_index,
        args.edge_gradient_ramp_epochs,
    )


def set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    """Apply the deterministic epoch learning rate to all parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def projection_visuals(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build deterministic first-case/first-view validation diagnostics.

    TensorBoard receives normalized target/prediction images plus intensity and
    first-derivative error maps.  The derivative maps are essential when a
    sharper result trades a small amount of pixel PSNR for better boundaries.
    """
    pred = prediction[0, 0].detach().float()
    true = target[0, 0].detach().float()

    def directional_gradients(image: torch.Tensor):
        dx = F.pad(
            image[..., :, 1:] - image[..., :, :-1],
            (0, 1, 0, 0),
        )
        dy = F.pad(
            image[..., 1:, :] - image[..., :-1, :],
            (0, 0, 0, 1),
        )
        return dx, dy

    pred_dx, pred_dy = directional_gradients(pred)
    true_dx, true_dy = directional_gradients(true)
    pred_gradient = 0.5 * (pred_dx.abs() + pred_dy.abs())
    true_gradient = 0.5 * (true_dx.abs() + true_dy.abs())
    gradient_error = 0.5 * (
        (pred_dx - true_dx).abs() + (pred_dy - true_dy).abs()
    )

    return {
        "target": ((true + 1.0) * 0.5).clamp(0.0, 1.0).cpu(),
        "prediction": ((pred + 1.0) * 0.5).clamp(0.0, 1.0).cpu(),
        "absolute_error": ((pred - true).abs() * 0.5).clamp(0.0, 1.0).cpu(),
        "target_gradient": (true_gradient * 0.5).clamp(0.0, 1.0).cpu(),
        "prediction_gradient": (pred_gradient * 0.5).clamp(0.0, 1.0).cpu(),
        "gradient_error": (gradient_error * 0.25).clamp(0.0, 1.0).cpu(),
    }


@torch.no_grad()
def evaluate(
    model: LearnedForwardProjector,
    loader,
    criterion: ForwardProjectorLoss,
    device: torch.device,
    use_amp: bool,
    selection_gradient_weight: float,
    volume_source: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    model.eval()
    sums = {"loss": 0.0, "image": 0.0, "gradient": 0.0}
    psnr_sum = 0.0
    cases = 0
    visuals = None
    for batch in loader:
        volume = batch[volume_source].to(
            device, non_blocking=device.type == "cuda"
        )
        angles = batch["angles"].to(
            device, non_blocking=device.type == "cuda"
        )
        target = batch["projs"].to(
            device, non_blocking=device.type == "cuda"
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            prediction = model(volume, angles)
            losses = criterion(prediction, target)
        if visuals is None:
            visuals = projection_visuals(prediction, target)
        batch_size = volume.shape[0]
        sums["loss"] += float(losses["total"]) * batch_size
        sums["image"] += float(losses["image"]) * batch_size
        sums["gradient"] += float(losses["gradient"]) * batch_size
        psnr_sum += float(
            projection_psnr_per_case(prediction, target).sum()
        )
        cases += batch_size
    result = {name: value / cases for name, value in sums.items()}
    result["psnr"] = psnr_sum / cases
    # Unlike Val/loss, this score keeps one fixed definition while the B2
    # training coefficient ramps, so checkpoints remain comparable.
    result["composite"] = (
        result["image"]
        + float(selection_gradient_weight) * result["gradient"]
    )
    if visuals is None:
        raise RuntimeError("validation loader produced no batches")
    return result, visuals


def build_payload(
    *,
    epoch: int,
    model: LearnedForwardProjector,
    optimizer,
    scaler,
    args,
    best_psnr: float,
    best_psnr_epoch: int,
    best_gradient: float,
    best_gradient_epoch: int,
    best_composite: float,
    best_composite_epoch: int,
    current_gradient_weight: float,
    current_learning_rate: float,
    validation: dict | None,
    train_loader,
) -> dict:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "phase": PHASE,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        # This is schedule metadata, not a stateful PyTorch scheduler.  Epoch-
        # derived values make old 150-epoch checkpoints safe to extend.
        "scheduler_state": {
            "type": "phase_b_two_stage_cosine",
            "base_epochs": args.base_epochs,
            "total_epochs": args.epochs,
            "lr": args.lr,
            "edge_lr": args.edge_lr,
            "min_lr": args.min_lr,
        },
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "train_loader_generator_state": (
            capture_loader_generator_state(train_loader)
        ),
        "best_val_psnr": best_psnr,
        "best_epoch": best_psnr_epoch,
        "best_psnr_epoch": best_psnr_epoch,
        "best_val_gradient": best_gradient,
        "best_gradient_epoch": best_gradient_epoch,
        "best_val_composite": best_composite,
        "best_composite_epoch": best_composite_epoch,
        "current_gradient_weight": current_gradient_weight,
        "current_learning_rate": current_learning_rate,
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
            "volume_source": args.volume_source,
            "projection_views": args.views_per_case,
            "projection_clip": [0.0, 10.0],
            "projection_range": [-1.0, 1.0],
        },
    }


def train(args) -> None:
    validate_run_version(args.run_version)
    args.data_root = str(resolve_thorax_fast_root(args.data_root))
    print(f"[Phase B] data_root={args.data_root}")
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
    if not (0 < args.base_epochs <= args.epochs):
        raise ValueError("base_epochs must be in [1, epochs]")
    if args.edge_gradient_ramp_epochs <= 0:
        raise ValueError("edge_gradient_ramp_epochs must be positive")
    if (
        args.epochs > args.base_epochs
        and args.edge_gradient_ramp_epochs
        > args.epochs - args.base_epochs
    ):
        raise ValueError(
            "edge_gradient_ramp_epochs cannot exceed the number of B2 epochs"
        )
    if min(
        args.gradient_weight,
        args.edge_gradient_weight,
        args.selection_gradient_weight,
    ) < 0.0:
        raise ValueError("gradient weights must be non-negative")
    if min(args.lr, args.edge_lr, args.min_lr) <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.min_lr > min(args.lr, args.edge_lr):
        raise ValueError("min_lr cannot exceed lr or edge_lr")
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
        volume_keys=(args.volume_source,),
        projection_views=args.views_per_case,
        projection_size=tuple(args.projection_size),
        projection_sampling="random",
    )
    val_set = ThoraxFastDataset(
        args.data_root,
        split="val",
        volume_keys=(args.volume_source,),
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_psnr = -float("inf")
    best_psnr_epoch = 0
    best_gradient = float("inf")
    best_gradient_epoch = 0
    best_composite = float("inf")
    best_composite_epoch = 0
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, device)
        if checkpoint.get("phase") != PHASE:
            raise ValueError("resume checkpoint 不是 Phase B")
        if checkpoint.get("run_version") != args.run_version:
            raise ValueError("resume run_version 与当前命令不一致")
        saved_source = checkpoint.get("data_config", {}).get(
            "volume_source", "cbct"
        )
        # 旧 checkpoint 固定使用 CBCT，历史字段保存的是目录描述。
        if saved_source == "processed/images/cbct":
            saved_source = "cbct"
        if saved_source != args.volume_source:
            raise ValueError(
                "resume volume_source 与当前命令不一致: "
                f"checkpoint={saved_source}, current={args.volume_source}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if use_amp and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        restore_loader_generator_state(
            train_loader,
            checkpoint.get("train_loader_generator_state"),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError(
                "resume checkpoint already reached/exceeded --epochs: "
                f"checkpoint epoch={start_epoch - 1}, epochs={args.epochs}"
            )
        best_psnr = float(checkpoint.get("best_val_psnr", best_psnr))
        best_psnr_epoch = int(
            checkpoint.get(
                "best_psnr_epoch", checkpoint.get("best_epoch", 0)
            )
        )
        best_gradient = float(
            checkpoint.get("best_val_gradient", best_gradient)
        )
        best_gradient_epoch = int(
            checkpoint.get("best_gradient_epoch", 0)
        )
        best_composite = float(
            checkpoint.get("best_val_composite", best_composite)
        )
        best_composite_epoch = int(
            checkpoint.get("best_composite_epoch", 0)
        )

        # Format-1 Phase B used phase_B_best_*.pth for PSNR selection.  Preserve
        # that historical best before the generic filename becomes the B2
        # composite alias.  Copying is intentional: the last checkpoint does
        # not contain the older best-PSNR model weights.
        if int(checkpoint.get("checkpoint_format", 1)) < 2:
            legacy_best = (
                run_dir / f"phase_{PHASE}_best_{args.run_version}.pth"
            )
            explicit_psnr = (
                run_dir
                / f"phase_{PHASE}_best_psnr_{args.run_version}.pth"
            )
            if legacy_best.is_file() and not explicit_psnr.exists():
                shutil.copy2(legacy_best, explicit_psnr)

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
                f"- volume_source: `registered {args.volume_source.upper()} [0,1]`",
                "- target_projection: `real XIM attenuation [-1,1]`",
                f"- random_views_per_case: `{args.views_per_case}`",
                "- analytic_core: `differentiable parallel-beam approximation`",
                f"- B1 epochs: `1..{args.base_epochs}`",
                f"- B2 epochs: `{args.base_epochs + 1}..{args.epochs}`",
                f"- gradient weight: `{args.gradient_weight}` -> "
                f"`{args.edge_gradient_weight}` over "
                f"`{args.edge_gradient_ramp_epochs}` B2 epochs",
                f"- B2 learning-rate restart: `{args.edge_lr}`",
                f"- fixed selection gradient weight: "
                f"`{args.selection_gradient_weight}`",
            ]
        ),
        0,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        stage = phase_b_stage(epoch, args.base_epochs)
        current_gradient_weight = gradient_weight_for_epoch(epoch, args)
        current_learning_rate = learning_rate_for_epoch(epoch, args)
        criterion.set_gradient_weight(current_gradient_weight)
        set_optimizer_learning_rate(optimizer, current_learning_rate)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        psnr_sum = 0.0
        cases = 0
        for batch_index, batch in enumerate(train_loader):
            volume = batch[args.volume_source].to(
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
                prediction = model(volume, angles)
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

            batch_size = volume.shape[0]
            cases += batch_size
            psnr_sum += float(
                projection_psnr_per_case(
                    prediction.detach(), target
                ).sum()
            )
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value) * batch_size
        train_metrics = {
            name: value / cases for name, value in sums.items()
        }
        train_metrics["psnr"] = psnr_sum / cases
        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        writer.add_scalar(
            "Train/LearningRate", current_learning_rate, epoch
        )
        writer.add_scalar("Train/Stage", stage, epoch)
        writer.add_scalar(
            "Train/gradient_weight", current_gradient_weight, epoch
        )

        validation = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation, validation_visuals = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                args.selection_gradient_weight,
                args.volume_source,
            )
            for name, value in validation.items():
                writer.add_scalar(f"Val/{name}", value, epoch)
            for name, image in validation_visuals.items():
                writer.add_image(f"ValImages/{name}", image, epoch)
            print(
                f"[B] E{epoch:04d} S{stage} "
                f"gw={current_gradient_weight:.4f} "
                f"train={train_metrics['total']:.5f} "
                f"val_psnr={validation['psnr']:.3f} "
                f"val_edge={validation['gradient']:.5f}"
            )
            improved_psnr = validation["psnr"] > best_psnr
            improved_edge = validation["gradient"] < best_gradient
            improved_composite = validation["composite"] < best_composite
            if improved_psnr:
                best_psnr = validation["psnr"]
                best_psnr_epoch = epoch
            if improved_edge:
                best_gradient = validation["gradient"]
                best_gradient_epoch = epoch
            if improved_composite:
                best_composite = validation["composite"]
                best_composite_epoch = epoch

            payload = build_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                best_psnr=best_psnr,
                best_psnr_epoch=best_psnr_epoch,
                best_gradient=best_gradient,
                best_gradient_epoch=best_gradient_epoch,
                best_composite=best_composite,
                best_composite_epoch=best_composite_epoch,
                current_gradient_weight=current_gradient_weight,
                current_learning_rate=current_learning_rate,
                validation=validation,
                train_loader=train_loader,
            )
            if improved_psnr:
                save_checkpoint(
                    payload,
                    run_dir,
                    phase=PHASE,
                    version=args.run_version,
                    kind="best_psnr",
                )
            if improved_edge:
                save_checkpoint(
                    payload,
                    run_dir,
                    phase=PHASE,
                    version=args.run_version,
                    kind="best_edge",
                )
            if improved_composite:
                save_checkpoint(
                    payload,
                    run_dir,
                    phase=PHASE,
                    version=args.run_version,
                    kind="best_composite",
                )
                # Keep the historical filename as an alias of the recommended
                # fixed-composite checkpoint.
                save_checkpoint(
                    payload,
                    run_dir,
                    phase=PHASE,
                    version=args.run_version,
                    kind="best",
                )
        else:
            print(
                f"[B] E{epoch:04d} S{stage} "
                f"gw={current_gradient_weight:.4f} "
                f"train={train_metrics['total']:.5f} "
                f"train_psnr={train_metrics['psnr']:.3f}"
            )

        payload = build_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            best_psnr=best_psnr,
            best_psnr_epoch=best_psnr_epoch,
            best_gradient=best_gradient,
            best_gradient_epoch=best_gradient_epoch,
            best_composite=best_composite,
            best_composite_epoch=best_composite_epoch,
            current_gradient_weight=current_gradient_weight,
            current_learning_rate=current_learning_rate,
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
        "Phase B finished: "
        f"best PSNR={best_psnr:.3f} @ {best_psnr_epoch}, "
        f"best edge={best_gradient:.6f} @ {best_gradient_epoch}, "
        f"best composite={best_composite:.6f} @ {best_composite_epoch}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase B: train volume-to-projection model"
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help=(
            "Thorax Fast根目录；省略时依次检查项目内data/thorax_fast和"
            "~/autodl-tmp/thorax"
        ),
    )
    parser.add_argument("--run_version", required=True)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Total B1+B2 epochs; default is 150 base + 100 edge fine-tune",
    )
    parser.add_argument(
        "--base_epochs",
        type=int,
        default=150,
        help="Last B1 intensity-fitting epoch; B2 starts on the next epoch",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--views_per_case", type=int, default=6)
    parser.add_argument(
        "--volume_source",
        choices=("cbct", "ct"),
        default="cbct",
        help=(
            "Phase B input volume. cbct is the same-acquisition baseline; "
            "ct fits registered planning CT to real projections as a "
            "CT-style projector experiment."
        ),
    )
    parser.add_argument(
        "--projection_size", type=int, nargs=2, default=(128, 128)
    )
    parser.add_argument("--integration_size", type=int, default=96)
    parser.add_argument("--correction_channels", type=int, default=32)
    parser.add_argument("--dsd", type=float, default=1540.0)
    parser.add_argument("--dso", type=float, default=1000.0)
    parser.add_argument("--gradient_weight", type=float, default=0.10)
    parser.add_argument(
        "--edge_gradient_weight",
        type=float,
        default=0.25,
        help="Target B2 gradient-loss coefficient",
    )
    parser.add_argument(
        "--edge_gradient_ramp_epochs",
        type=int,
        default=25,
        help="B2 epochs used for the smooth half-cosine coefficient ramp",
    )
    parser.add_argument(
        "--selection_gradient_weight",
        type=float,
        default=0.20,
        help="Fixed Val/composite gradient coefficient for checkpoint ranking",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--edge_lr",
        type=float,
        default=2e-5,
        help="Learning-rate restart at the first B2 epoch",
    )
    parser.add_argument("--min_lr", type=float, default=1e-6)
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
