"""Phase C：训练投影→解剖先验→残差雕刻的双域主重建模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.dual_domain import (
    ApproximateProjectorGeometry,
    DualDomainLoss,
    DualDomainLossWeights,
    DualDomainReconstructionModel,
    HierarchicalAnatomyPrior,
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
    restore_loader_generator_state,
    restore_rng_state,
    save_checkpoint,
    save_json,
    select_cycle_views,
    select_views,
    set_global_seed,
    validate_run_version,
    volume_metrics,
    volume_psnr_per_case,
)
from src.thorax_fast_dataset import (
    ThoraxFastDataset,
    resolve_thorax_fast_root,
)
from src.view_protocol import resolve_view_curriculum


PHASE = "C"
CHECKPOINT_FORMAT = 1


def build_schedule(args) -> list[tuple[int, int, int, int]]:
    """返回 ``(start,end,n_views,stage)`` 的高到低训练日程。"""
    views = resolve_view_curriculum(
        args.final_view, args.view_schedule, max_views=64
    )
    schedule = [(1, args.stage1_epochs, views[0], 1)]
    cursor = args.stage1_epochs + 1
    for n_views in views[1:]:
        end = cursor + args.stage2_epochs_per_view - 1
        schedule.append((cursor, end, n_views, 2))
        cursor = end + 1
    if args.stage3_epochs > 0:
        schedule.append(
            (cursor, cursor + args.stage3_epochs - 1, args.final_view, 3)
        )
    return schedule


def stage_for_epoch(
    schedule: list[tuple[int, int, int, int]], epoch: int
) -> tuple[int, int]:
    for start, end, n_views, stage in schedule:
        if start <= epoch <= end:
            return stage, n_views
    raise RuntimeError(f"epoch={epoch} 不属于任何训练阶段")


def stage_weights(stage: int) -> DualDomainLossWeights:
    if stage == 1:
        return DualDomainLossWeights.stage1()
    if stage == 2:
        return DualDomainLossWeights.stage2()
    return DualDomainLossWeights.stage3()


def configure_optimizer(
    model: DualDomainReconstructionModel,
    stage: int,
    stage_epochs: int,
    weight_decay: float,
):
    """应用 Stage 冻结策略，并建立分组优化器和阶段内余弦调度器。"""
    # Stage 3 固定基础模型，只让投影编码和雕刻器适应厂家最终协议。
    prior_decoder_trainable = stage != 3
    for parameter in model.anatomy_prior.decoder.parameters():
        parameter.requires_grad = prior_decoder_trainable

    if stage == 1:
        lrs = {
            "projection_encoder": 1e-4,
            "prior_decoder": 5e-5,
            "sculptor": 1e-4,
        }
    elif stage == 2:
        lrs = {
            "projection_encoder": 5e-5,
            "prior_decoder": 2e-5,
            "sculptor": 5e-5,
        }
    else:
        lrs = {
            "projection_encoder": 1e-5,
            "prior_decoder": 0.0,
            "sculptor": 2e-5,
        }

    groups = []
    for name, parameters in model.trainable_parameter_groups().items():
        if parameters:
            groups.append(
                {"params": parameters, "lr": lrs[name], "name": name}
            )
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    scheduler = cosine_scheduler(optimizer, stage_epochs)
    return optimizer, scheduler


def _load_phase_a_prior(checkpoint: dict) -> HierarchicalAnatomyPrior:
    if checkpoint.get("phase") != "A":
        raise ValueError("--phase_a_checkpoint 不是 Phase A checkpoint")
    config = checkpoint["model_config"]
    prior = HierarchicalAnatomyPrior(
        anatomy_codebook_size=int(config["anatomy_codebook_size"]),
        boundary_codebook_size=int(config["boundary_codebook_size"]),
        anatomy_dim=int(config["anatomy_dim"]),
        boundary_dim=int(config["boundary_dim"]),
        base_channels=int(config["base_channels"]),
        prior_feature_channels=int(config["prior_feature_channels"]),
        kmeans_init_batches=int(config["kmeans_init_batches"]),
    )
    prior.load_state_dict(checkpoint["model_state"])
    if not (
        bool(prior.anatomy_codebook.codebook.kmeans_initialized.item())
        and bool(
            prior.boundary_codebook.codebook.kmeans_initialized.item()
        )
    ):
        raise ValueError("Phase A checkpoint 的 codebook 尚未完成 K-means")
    return prior


def _load_phase_b_projector(checkpoint: dict) -> LearnedForwardProjector:
    if checkpoint.get("phase") != "B":
        raise ValueError("--phase_b_checkpoint 不是 Phase B checkpoint")
    config = checkpoint["model_config"]
    geometry_config = config["geometry"]
    geometry = ApproximateProjectorGeometry(
        dsd_mm=float(geometry_config["dsd_mm"]),
        dso_mm=float(geometry_config["dso_mm"]),
        detector_pixels=tuple(geometry_config["detector_pixels"]),
        detector_spacing_mm=tuple(
            geometry_config["detector_spacing_mm"]
        ),
        voxel_spacing_mm=tuple(geometry_config["voxel_spacing_mm"]),
    )
    projector = LearnedForwardProjector(
        projection_size=tuple(config["projection_size"]),
        integration_size=int(config["integration_size"]),
        correction_channels=int(config["correction_channels"]),
        geometry=geometry,
    )
    projector.load_state_dict(checkpoint["model_state"])
    for parameter in projector.parameters():
        parameter.requires_grad = False
    projector.eval()
    return projector


def build_model(
    phase_a_checkpoint: dict,
    args,
) -> DualDomainReconstructionModel:
    prior = _load_phase_a_prior(phase_a_checkpoint)
    config = phase_a_checkpoint["model_config"]
    model = DualDomainReconstructionModel(
        anatomy_prior=prior,
        anatomy_dim=int(config["anatomy_dim"]),
        boundary_dim=int(config["boundary_dim"]),
        prior_feature_channels=int(config["prior_feature_channels"]),
        refinement_channels=args.refinement_channels,
        transformer_layers=args.transformer_layers,
        projection_seed_size=16,
        output_size=(256, 256, 256),
    )
    model.freeze_pretrained_prior(freeze_decoder=False)
    return model


def cycle_forward(
    projector: LearnedForwardProjector,
    final_volume: torch.Tensor,
    cycle_views: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """保持投影器参数冻结，但允许梯度穿过它回到 final_volume。"""
    input_prediction = projector(
        final_volume, cycle_views["input_angles"]
    )
    heldout_prediction = None
    if cycle_views["heldout_angles"] is not None:
        heldout_prediction = projector(
            final_volume, cycle_views["heldout_angles"]
        )
    return input_prediction, heldout_prediction


@torch.no_grad()
def evaluate(
    model: DualDomainReconstructionModel,
    projector: LearnedForwardProjector,
    loader,
    criterion: DualDomainLoss,
    device: torch.device,
    use_amp: bool,
    final_view: int,
    cycle_input_views: int,
    heldout_views: int,
    compute_ssim: bool,
) -> dict[str, float]:
    model.eval()
    projector.eval()
    sums = {
        "loss": 0.0,
        "final_psnr": 0.0,
        "base_psnr": 0.0,
        "ssim": 0.0,
        "input_cycle": 0.0,
        "heldout_cycle": 0.0,
    }
    cases = 0
    for batch in loader:
        base_projections = batch["projs"].to(
            device, non_blocking=device.type == "cuda"
        )
        base_angles = batch["angles"].to(
            device, non_blocking=device.type == "cuda"
        )
        ct = batch["ct"].to(
            device, non_blocking=device.type == "cuda"
        )
        projections, angles, input_indices = select_views(
            base_projections,
            base_angles,
            final_view,
            random_subset=False,
        )
        cycle_views = select_cycle_views(
            base_projections,
            base_angles,
            input_indices,
            max_input_cycle_views=cycle_input_views,
            heldout_views=heldout_views,
            random_subset=False,
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(projections, angles, teacher_ct=ct)
            input_reprojection, heldout_reprojection = cycle_forward(
                projector, outputs["final_volume"], cycle_views
            )
            losses = criterion(
                outputs,
                ct,
                input_projections=cycle_views["input_projections"],
                reconstructed_input_projections=input_reprojection,
                heldout_projections=cycle_views["heldout_projections"],
                reconstructed_heldout_projections=heldout_reprojection,
            )

        batch_size = ct.shape[0]
        cases += batch_size
        sums["loss"] += float(losses["total"]) * batch_size
        final_metrics = volume_metrics(
            outputs["final_volume"], ct, compute_ssim=compute_ssim
        )
        sums["final_psnr"] += final_metrics["psnr_sum"]
        sums["ssim"] += final_metrics.get("ssim_sum", 0.0)
        base_target = F.interpolate(
            ct,
            size=outputs["base_volume"].shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        sums["base_psnr"] += float(
            volume_psnr_per_case(
                outputs["base_volume"], base_target
            ).sum()
        )
        sums["input_cycle"] += (
            float(losses["raw/input_projection"]) * batch_size
        )
        sums["heldout_cycle"] += (
            float(losses["raw/heldout_projection"]) * batch_size
        )

    result = {
        key: value / cases
        for key, value in sums.items()
        if key != "ssim" or compute_ssim
    }
    return result


def build_payload(
    *,
    epoch: int,
    stage: int,
    n_views: int,
    model: DualDomainReconstructionModel,
    projector: LearnedForwardProjector,
    optimizer,
    scheduler,
    scaler,
    args,
    schedule,
    best_psnr: float,
    best_epoch: int,
    validation: dict | None,
    phase_a_model_config: dict,
    phase_b_model_config: dict,
    train_loader,
) -> dict:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "phase": PHASE,
        "epoch": epoch,
        "stage": stage,
        "current_views": n_views,
        "model_state": model.state_dict(),
        "projector_state": projector.state_dict(),
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
        "final_view": args.final_view,
        "schedule": schedule,
        "pretrained": {
            "phase_a_checkpoint": str(
                Path(args.phase_a_checkpoint).resolve()
            ),
            "phase_b_checkpoint": str(
                Path(args.phase_b_checkpoint).resolve()
            ),
        },
        "model_config": {
            "anatomy_prior": phase_a_model_config,
            "forward_projector": phase_b_model_config,
            "transformer_layers": args.transformer_layers,
            "refinement_channels": args.refinement_channels,
            "projection_seed_size": 16,
            "projection_size": list(args.projection_size),
            "output_size": [256, 256, 256],
        },
    }


def train(args) -> None:
    validate_run_version(args.run_version)
    args.data_root = str(resolve_thorax_fast_root(args.data_root))
    print(f"[Phase C] data_root={args.data_root}")
    positive = (
        args.stage1_epochs,
        args.stage2_epochs_per_view,
        args.batch_size,
        args.grad_accum,
        args.eval_every,
        args.save_every,
        args.cycle_input_views,
    )
    if (
        min(positive) <= 0
        or args.stage3_epochs < 0
        or args.heldout_cycle_views < 0
        or args.num_workers < 0
    ):
        raise ValueError("训练轮数、batch、保存间隔和循环视角参数不合法")

    set_global_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    schedule = build_schedule(args)
    total_epochs = schedule[-1][1]
    base_views = schedule[0][2]

    phase_a_checkpoint = load_trusted_checkpoint(
        args.phase_a_checkpoint, "cpu"
    )
    phase_b_checkpoint = load_trusted_checkpoint(
        args.phase_b_checkpoint, "cpu"
    )
    model = build_model(phase_a_checkpoint, args).to(device)
    projector = _load_phase_b_projector(
        phase_b_checkpoint
    ).to(device)

    saved_projection_size = tuple(
        phase_b_checkpoint["model_config"]["projection_size"]
    )
    if tuple(args.projection_size) != saved_projection_size:
        raise ValueError(
            f"Phase B projection_size={saved_projection_size}，"
            f"Phase C 请求 {tuple(args.projection_size)}"
        )

    run_name = (
        f"thorax_phaseC_finalview={args.final_view}_{args.run_version}"
    )
    run_dir = prepare_run_directory(
        args.log_dir, run_name, resume=args.resume
    )
    writer = SummaryWriter(run_dir / "tensorboard")

    dataset_kwargs = {
        "data_root": args.data_root,
        "volume_keys": ("ct",),
        "projection_views": base_views,
        "final_view": args.final_view,
        "projection_size": tuple(args.projection_size),
        "projection_sampling": "uniform",
    }
    train_set = ThoraxFastDataset(split="train", **dataset_kwargs)
    val_set = ThoraxFastDataset(split="val", **dataset_kwargs)
    test_set = ThoraxFastDataset(split="test", **dataset_kwargs)
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
    test_loader = build_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 2,
        device=device,
    )

    criterion = DualDomainLoss(stage_weights(1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = None
    scheduler = None
    previous_stage = None
    resume_checkpoint = None
    start_epoch = 1
    best_psnr = -float("inf")
    best_epoch = 0
    if args.resume:
        resume_checkpoint = load_trusted_checkpoint(args.resume, device)
        if resume_checkpoint.get("phase") != PHASE:
            raise ValueError("resume checkpoint 不是 Phase C")
        if resume_checkpoint.get("run_version") != args.run_version:
            raise ValueError("resume run_version 与当前命令不一致")
        if int(resume_checkpoint["final_view"]) != args.final_view:
            raise ValueError("resume final_view 与当前命令不一致")
        if resume_checkpoint["schedule"] != schedule:
            raise ValueError("resume 训练课程与当前命令不一致")
        model.load_state_dict(resume_checkpoint["model_state"])
        projector.load_state_dict(resume_checkpoint["projector_state"])
        restore_rng_state(resume_checkpoint.get("rng_state"))
        restore_loader_generator_state(
            train_loader,
            resume_checkpoint.get("train_loader_generator_state"),
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_psnr = float(
            resume_checkpoint.get("best_val_psnr", best_psnr)
        )
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))

    config = vars(args).copy()
    config.update(
        {
            "phase": PHASE,
            "run_name": run_name,
            "schedule": schedule,
            "total_epochs": total_epochs,
            "base_views": base_views,
            "train_cases": len(train_set),
            "val_cases": len(val_set),
            "test_cases": len(test_set),
            "parameters": sum(p.numel() for p in model.parameters()),
            "projector_parameters": sum(
                p.numel() for p in projector.parameters()
            ),
        }
    )
    save_json(run_dir / "config.json", config)
    writer.add_text(
        "Run/Metadata",
        "\n".join(
            [
                "# Phase C: dual-domain reconstruction",
                f"- final_view: `{args.final_view}`",
                f"- base_views: `{base_views}`",
                f"- schedule: `{schedule}`",
                "- Stage 2: `random subsets from fixed base`",
                "- Stage 3: `fixed manufacturer final-view subset`",
                "- codebooks: `frozen CT-domain EMA`",
                "- forward_projector: `frozen parameters, gradient to volume enabled`",
            ]
        ),
        0,
    )

    resume_optimizer_pending = resume_checkpoint is not None
    for epoch in range(start_epoch, total_epochs + 1):
        stage, current_views = stage_for_epoch(schedule, epoch)
        if stage != previous_stage:
            stage_start = min(
                start for start, _, _, item_stage in schedule
                if item_stage == stage
            )
            stage_end = max(
                end for _, end, _, item_stage in schedule
                if item_stage == stage
            )
            optimizer, scheduler = configure_optimizer(
                model,
                stage,
                stage_end - stage_start + 1,
                args.weight_decay,
            )
            criterion.set_weights(stage_weights(stage))
            if resume_optimizer_pending:
                if int(resume_checkpoint["stage"]) == stage:
                    optimizer.load_state_dict(
                        resume_checkpoint["optimizer_state"]
                    )
                    scheduler.load_state_dict(
                        resume_checkpoint["scheduler_state"]
                    )
                    if use_amp and resume_checkpoint.get("scaler_state"):
                        scaler.load_state_dict(
                            resume_checkpoint["scaler_state"]
                        )
                resume_optimizer_pending = False
            previous_stage = stage

        model.train()
        # 教师、EMA codebook 和冻结投影器保持确定性。
        model.anatomy_prior.encoder.eval()
        model.anatomy_prior.anatomy_codebook.eval()
        model.anatomy_prior.boundary_codebook.eval()
        projector.eval()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        cases = 0

        for batch_index, batch in enumerate(train_loader):
            base_projections = batch["projs"].to(
                device, non_blocking=device.type == "cuda"
            )
            base_angles = batch["angles"].to(
                device, non_blocking=device.type == "cuda"
            )
            ct = batch["ct"].to(
                device, non_blocking=device.type == "cuda"
            )
            random_subset = stage == 2
            projections, angles, input_indices = select_views(
                base_projections,
                base_angles,
                current_views,
                random_subset=random_subset,
            )
            cycle_views = select_cycle_views(
                base_projections,
                base_angles,
                input_indices,
                max_input_cycle_views=args.cycle_input_views,
                heldout_views=args.heldout_cycle_views,
                random_subset=True,
            )
            window = accumulation_window_size(
                batch_index, len(train_loader), args.grad_accum
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(projections, angles, teacher_ct=ct)
                input_reprojection, heldout_reprojection = cycle_forward(
                    projector, outputs["final_volume"], cycle_views
                )
                losses = criterion(
                    outputs,
                    ct,
                    input_projections=cycle_views["input_projections"],
                    reconstructed_input_projections=input_reprojection,
                    heldout_projections=cycle_views[
                        "heldout_projections"
                    ],
                    reconstructed_heldout_projections=heldout_reprojection,
                )
                scaled_loss = losses["total"] / window
            scaler.scale(scaled_loss).backward()

            if optimizer_step_due(
                batch_index, len(train_loader), args.grad_accum
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = ct.shape[0]
            cases += batch_size
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value) * batch_size
        scheduler.step()

        train_metrics = {
            name: value / cases for name, value in sums.items()
        }
        writer.add_scalar("Train/Stage", stage, epoch)
        writer.add_scalar("Train/n_views", current_views, epoch)
        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/Loss/{name}", value, epoch)
        for group in optimizer.param_groups:
            writer.add_scalar(
                f"Train/LearningRate/{group['name']}",
                group["lr"],
                epoch,
            )
        # 码字在 Phase C 中被冻结；这里记录的是投影特征对固定码本的
        # 实际命中分布，用于监测 encoder 是否退化到少数码字。
        for prefix, quantizer in (
            ("anatomy", model.anatomy_prior.anatomy_codebook),
            ("boundary", model.anatomy_prior.boundary_codebook),
        ):
            for name, value in quantizer.diagnostics().items():
                writer.add_scalar(
                    f"Codebook/{prefix}_{name}", float(value), epoch
                )
        writer.add_scalar(
            "Sculptor/residual_abs_mean",
            float(outputs["residual_logits"].detach().abs().mean()),
            epoch,
        )
        writer.add_scalar(
            "Sculptor/gate_mean",
            float(outputs["gate"].detach().mean()),
            epoch,
        )

        validation = None
        if epoch % args.eval_every == 0 or epoch == total_epochs:
            criterion.set_weights(stage_weights(stage))
            validation = evaluate(
                model,
                projector,
                val_loader,
                criterion,
                device,
                use_amp,
                args.final_view,
                args.cycle_input_views,
                args.heldout_cycle_views,
                args.compute_ssim,
            )
            for name, value in validation.items():
                writer.add_scalar(f"Val/{name}", value, epoch)
            print(
                f"[C] E{epoch:04d} S{stage} V={current_views} "
                f"train={train_metrics['total']:.5f} "
                f"val_psnr={validation['final_psnr']:.3f}"
            )
            if validation["final_psnr"] > best_psnr:
                best_psnr = validation["final_psnr"]
                best_epoch = epoch
                payload = build_payload(
                    epoch=epoch,
                    stage=stage,
                    n_views=current_views,
                    model=model,
                    projector=projector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    schedule=schedule,
                    best_psnr=best_psnr,
                    best_epoch=best_epoch,
                    validation=validation,
                    phase_a_model_config=phase_a_checkpoint[
                        "model_config"
                    ],
                    phase_b_model_config=phase_b_checkpoint[
                        "model_config"
                    ],
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
                f"[C] E{epoch:04d} S{stage} V={current_views} "
                f"train={train_metrics['total']:.5f}"
            )

        payload = build_payload(
            epoch=epoch,
            stage=stage,
            n_views=current_views,
            model=model,
            projector=projector,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            schedule=schedule,
            best_psnr=best_psnr,
            best_epoch=best_epoch,
            validation=validation,
            phase_a_model_config=phase_a_checkpoint["model_config"],
            phase_b_model_config=phase_b_checkpoint["model_config"],
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

    best_path = (
        run_dir / f"phase_C_best_{args.run_version}.pth"
    )
    if not best_path.is_file():
        raise FileNotFoundError("未生成 Phase C best checkpoint")
    best_checkpoint = load_trusted_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model_state"])
    projector.load_state_dict(best_checkpoint["projector_state"])
    best_stage = int(best_checkpoint["stage"])
    criterion.set_weights(stage_weights(best_stage))
    test_metrics = evaluate(
        model,
        projector,
        test_loader,
        criterion,
        device,
        use_amp,
        args.final_view,
        args.cycle_input_views,
        args.heldout_cycle_views,
        args.compute_ssim,
    )
    for name, value in test_metrics.items():
        writer.add_scalar(f"Test/{name}", value, total_epochs)
    save_json(
        run_dir / "test_metrics.json",
        {
            "best_checkpoint": str(best_path),
            "best_epoch": best_epoch,
            "final_view": args.final_view,
            "metrics": test_metrics,
        },
    )
    writer.close()
    print(
        f"Phase C 完成：best val PSNR={best_psnr:.3f}, "
        f"epoch={best_epoch}; test只评估一次。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase C: train dual-domain sparse-view reconstruction"
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help=(
            "Thorax Fast根目录；省略时依次检查项目内data/thorax_fast和"
            "~/autodl-tmp/thorax"
        ),
    )
    parser.add_argument("--phase_a_checkpoint", required=True)
    parser.add_argument("--phase_b_checkpoint", required=True)
    parser.add_argument("--run_version", required=True)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument(
        "--final_view", type=int, choices=(6, 8, 10), default=6
    )
    parser.add_argument("--view_schedule", default=None)
    parser.add_argument("--stage1_epochs", type=int, default=150)
    parser.add_argument(
        "--stage2_epochs_per_view", type=int, default=30
    )
    parser.add_argument("--stage3_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--refinement_channels", type=int, default=16)
    parser.add_argument(
        "--projection_size", type=int, nargs=2, default=(128, 128)
    )
    parser.add_argument("--cycle_input_views", type=int, default=6)
    parser.add_argument("--heldout_cycle_views", type=int, default=6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
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
