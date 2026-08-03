"""Phase A：使用真实 CT 预训练分层解剖 codebook 与基础体积 Decoder。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.dual_domain import (
    AnatomyPriorLoss,
    HierarchicalAnatomyPrior,
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
    restore_loader_generator_state,
    restore_rng_state,
    save_checkpoint,
    save_json,
    set_global_seed,
    validate_run_version,
    volume_metrics,
)
from src.thorax_fast_dataset import (
    ThoraxFastDataset,
    resolve_thorax_fast_root,
)


PHASE = "A"
CHECKPOINT_FORMAT = 2


@torch.no_grad()
def initialize_codebooks(
    model: HierarchicalAnatomyPrior,
    loader,
    device: torch.device,
    use_amp: bool,
) -> None:
    """优化开始前用固定 CT Encoder 跨 batch 完成双码本 K-means。"""
    anatomy_ready = bool(
        model.anatomy_codebook.codebook.kmeans_initialized.item()
    )
    boundary_ready = bool(
        model.boundary_codebook.codebook.kmeans_initialized.item()
    )
    if anatomy_ready and boundary_ready:
        return

    print("[Phase A] 收集 CT latent，初始化 Anatomy/Boundary codebook...")
    model.train()
    while not (anatomy_ready and boundary_ready):
        made_progress = False
        for batch in loader:
            made_progress = True
            ct = batch["ct"].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                latents = model.encode_ct(ct)
                model.quantize(
                    latents["anatomy_latent"],
                    latents["boundary_latent"],
                    update_codebook=True,
                )
            anatomy_ready = bool(
                model.anatomy_codebook.codebook.kmeans_initialized.item()
            )
            boundary_ready = bool(
                model.boundary_codebook.codebook.kmeans_initialized.item()
            )
            anatomy_progress = float(
                model.anatomy_codebook.codebook.diagnostics().get(
                    "kmeans_init_progress", torch.tensor(0.0)
                )
            )
            boundary_progress = float(
                model.boundary_codebook.codebook.diagnostics().get(
                    "kmeans_init_progress", torch.tensor(0.0)
                )
            )
            print(
                f"  K-means: anatomy={anatomy_progress:.0%}, "
                f"boundary={boundary_progress:.0%}"
            )
            if anatomy_ready and boundary_ready:
                break
        if not made_progress:
            raise RuntimeError("训练集为空，无法初始化 codebook")


@torch.no_grad()
def evaluate(
    model: HierarchicalAnatomyPrior,
    loader,
    criterion: AnatomyPriorLoss,
    device: torch.device,
    use_amp: bool,
    compute_ssim: bool,
) -> dict[str, float]:
    model.eval()
    sums = {
        "loss": 0.0,
        "boundary_edge": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    cases = 0
    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=device.type == "cuda")
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(ct, update_codebook=False)
            losses = criterion(outputs, ct)
        target = F.interpolate(
            ct,
            size=outputs["base_volume"].shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        metrics = volume_metrics(
            outputs["base_volume"], target, compute_ssim=compute_ssim
        )
        batch_size = ct.shape[0]
        sums["loss"] += float(losses["total"]) * batch_size
        sums["boundary_edge"] += (
            float(losses["boundary_edge"]) * batch_size
        )
        sums["psnr"] += metrics["psnr_sum"]
        sums["ssim"] += metrics.get("ssim_sum", 0.0)
        cases += batch_size
    result = {
        "loss": sums["loss"] / cases,
        "boundary_edge": sums["boundary_edge"] / cases,
        "psnr": sums["psnr"] / cases,
    }
    if compute_ssim:
        result["ssim"] = sums["ssim"] / cases
    return result


def build_payload(
    *,
    epoch: int,
    model: HierarchicalAnatomyPrior,
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
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "train_loader_generator_state": (
            capture_loader_generator_state(train_loader)
        ),
        "best_val_psnr": best_psnr,
        "best_epoch": best_epoch,
        "validation": validation,
        "run_version": args.run_version,
        "model_config": {
            "architecture_version": 2,
            "anatomy_codebook_size": args.anatomy_codebook_size,
            "boundary_codebook_size": args.boundary_codebook_size,
            "anatomy_dim": args.anatomy_dim,
            "boundary_dim": args.boundary_dim,
            "base_channels": args.base_channels,
            "prior_feature_channels": args.prior_feature_channels,
            "kmeans_init_batches": args.kmeans_init_batches,
            "boundary_residual_blocks": args.boundary_residual_blocks,
            "anatomy_transformer_layers": (
                args.anatomy_transformer_layers
            ),
            "anatomy_transformer_heads": args.anatomy_transformer_heads,
            "anatomy_context_size": args.anatomy_context_size,
            "boundary_context_channels": args.boundary_context_channels,
            "boundary_edge_weight": args.boundary_edge_weight,
        },
        "data_config": {
            "volume_size": [256, 256, 256],
            "ct_range_hu": [-1000.0, 1000.0],
            "target": "processed/images/ct",
        },
    }


def train(args) -> None:
    validate_run_version(args.run_version)
    args.data_root = str(resolve_thorax_fast_root(args.data_root))
    print(f"[Phase A] data_root={args.data_root}")
    if min(
        args.epochs,
        args.batch_size,
        args.grad_accum,
        args.eval_every,
        args.save_every,
    ) <= 0:
        raise ValueError(
            "epochs、batch、梯度累积、验证和保存间隔必须为正数"
        )
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    if args.boundary_edge_weight < 0.0:
        raise ValueError("boundary_edge_weight cannot be negative")
    set_global_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    run_name = f"thorax_phaseA_prior_{args.run_version}"
    run_dir = prepare_run_directory(
        args.log_dir, run_name, resume=args.resume
    )

    train_set = ThoraxFastDataset(
        args.data_root,
        split="train",
        volume_keys=("ct",),
        require_projections=False,
    )
    val_set = ThoraxFastDataset(
        args.data_root,
        split="val",
        volume_keys=("ct",),
        require_projections=False,
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

    model = HierarchicalAnatomyPrior(
        anatomy_codebook_size=args.anatomy_codebook_size,
        boundary_codebook_size=args.boundary_codebook_size,
        anatomy_dim=args.anatomy_dim,
        boundary_dim=args.boundary_dim,
        base_channels=args.base_channels,
        prior_feature_channels=args.prior_feature_channels,
        kmeans_init_batches=args.kmeans_init_batches,
        boundary_residual_blocks=args.boundary_residual_blocks,
        anatomy_transformer_layers=args.anatomy_transformer_layers,
        anatomy_transformer_heads=args.anatomy_transformer_heads,
        anatomy_context_size=args.anatomy_context_size,
        boundary_context_channels=args.boundary_context_channels,
    ).to(device)
    criterion = AnatomyPriorLoss(
        boundary_edge_weight=args.boundary_edge_weight
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
            raise ValueError("resume checkpoint 不是 Phase A")
        if checkpoint.get("run_version") != args.run_version:
            raise ValueError("resume run_version 与当前命令不一致")
        if int(
            checkpoint.get("model_config", {}).get(
                "architecture_version", 1
            )
        ) != 2:
            raise ValueError(
                "该 Phase A checkpoint 来自旧解剖先验结构；新增全局 "
                "Transformer/Boundary 分支后必须重新训练 Phase A"
            )
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
    else:
        initialize_codebooks(model, train_loader, device, use_amp)

    # K-means 成功或 checkpoint 成功恢复后再创建事件文件。这样初始化阶段
    # 若报错，不会仅因残留 TensorBoard 文件阻止同一版本重新运行。
    writer = SummaryWriter(
        run_dir / "tensorboard",
        purge_step=start_epoch if args.resume else None,
    )
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
                "# Phase A: CT anatomy prior",
                f"- run_version: `{args.run_version}`",
                f"- train_cases: `{len(train_set)}`",
                f"- val_cases: `{len(val_set)}`",
                "- target: `clean pCT [0,1]`",
                "- output: `128^3 base volume`",
                "- anatomy_global_context: "
                f"`{args.anatomy_context_size}^3 tokens, "
                f"{args.anatomy_transformer_layers} layers, "
                f"{args.anatomy_transformer_heads} heads`",
                "- boundary: "
                f"`{args.boundary_residual_blocks} local residual blocks "
                "+ global context`",
                f"- boundary_edge_weight: `{args.boundary_edge_weight}`",
            ]
        ),
        0,
    )

    accumulation = args.grad_accum
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        cases = 0
        for batch_index, batch in enumerate(train_loader):
            ct = batch["ct"].to(
                device, non_blocking=device.type == "cuda"
            )
            window = accumulation_window_size(
                batch_index, len(train_loader), accumulation
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(ct, update_codebook=True)
                losses = criterion(outputs, ct)
                scaled_loss = losses["total"] / window
            scaler.scale(scaled_loss).backward()

            if optimizer_step_due(
                batch_index, len(train_loader), accumulation
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = ct.shape[0]
            cases += batch_size
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value) * batch_size
        scheduler.step()

        train_metrics = {name: value / cases for name, value in sums.items()}
        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        writer.add_scalar(
            "Train/LearningRate", optimizer.param_groups[0]["lr"], epoch
        )
        for prefix, quantizer in (
            ("anatomy", model.anatomy_codebook),
            ("boundary", model.boundary_codebook),
        ):
            for name, value in quantizer.diagnostics().items():
                writer.add_scalar(
                    f"Codebook/{prefix}_{name}", float(value), epoch
                )

        validation = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                args.compute_ssim,
            )
            for name, value in validation.items():
                writer.add_scalar(f"Val/{name}", value, epoch)
            print(
                f"[A] E{epoch:04d} train={train_metrics['total']:.5f} "
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
                f"[A] E{epoch:04d} train={train_metrics['total']:.5f}"
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
        f"Phase A 完成：best val PSNR={best_psnr:.3f}, epoch={best_epoch}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase A: pretrain CT anatomy codebooks"
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--anatomy_codebook_size", type=int, default=512)
    parser.add_argument("--boundary_codebook_size", type=int, default=256)
    parser.add_argument("--anatomy_dim", type=int, default=64)
    parser.add_argument("--boundary_dim", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--prior_feature_channels", type=int, default=32)
    parser.add_argument(
        "--boundary_residual_blocks", type=int, default=3
    )
    parser.add_argument(
        "--anatomy_transformer_layers", type=int, default=2
    )
    parser.add_argument(
        "--anatomy_transformer_heads", type=int, default=4
    )
    parser.add_argument("--anatomy_context_size", type=int, default=8)
    parser.add_argument(
        "--boundary_context_channels", type=int, default=8
    )
    parser.add_argument("--boundary_edge_weight", type=float, default=0.05)
    parser.add_argument("--kmeans_init_batches", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument(
        "--compute_ssim", action="store_true", default=True
    )
    parser.add_argument(
        "--no_ssim", action="store_false", dest="compute_ssim"
    )
    parser.add_argument("--resume", default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
