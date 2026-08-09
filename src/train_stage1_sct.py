"""Stage 1训练入口：配对CBCT -> HU风格sCT基底B。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

from src.fast_sculpt.training_utils import (
    build_loader,
    prepare_run_directory,
    set_global_seed,
    validate_run_version,
    volume_psnr_per_case,
)
from src.fast_sculpt import BaseSCTLoss, BaseSCTNet
from src.thorax_fast_dataset import ThoraxFastDataset


@dataclass
class EpochResult:
    loss: float
    psnr: float


def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = psnr_sum = 0.0
    cases = 0
    for batch in loader:
        cbct = batch["cbct"].to(device, non_blocking=True)
        ct = batch["ct"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler is not None and scaler.is_enabled(),
        ):
            outputs = model(cbct)
            losses = loss_fn(outputs, ct)
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        batch_size = cbct.shape[0]
        loss_sum += float(losses["total"].detach()) * batch_size
        psnr_sum += float(volume_psnr_per_case(
            outputs["base_sct"].detach(), ct
        ).sum())
        cases += batch_size
    return EpochResult(loss_sum / cases, psnr_sum / cases)


def train(args):
    validate_run_version(args.run_version)
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = prepare_run_directory(
        args.log_root, f"stage1_sct_{args.run_version}", resume=args.resume
    )
    writer = SummaryWriter(run_dir)

    common = dict(
        data_root=args.data_root,
        volume_keys=("cbct", "ct"),
        require_projections=False,
        volume_size=tuple(args.volume_size),
    )
    train_set = ThoraxFastDataset(split="train", **common)
    val_set = ThoraxFastDataset(split="val", **common)
    train_loader = build_loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, seed=args.seed, device=device
    )
    val_loader = build_loader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, seed=args.seed + 1, device=device
    )

    model = BaseSCTNet(args.base_channels).to(device)
    loss_fn = BaseSCTLoss(
        laplacian=args.laplacian_weight,
        structural=args.structural_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_psnr = 0, float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_psnr = state.get("best_psnr", best_psnr)

    for epoch in range(start_epoch, args.epochs):
        train_result = _run_epoch(
            model, train_loader, loss_fn, device, optimizer, scaler
        )
        with torch.no_grad():
            val_result = _run_epoch(
                model, val_loader, loss_fn, device, scaler=scaler
            )
        scheduler.step()
        writer.add_scalar("Train/Loss", train_result.loss, epoch)
        writer.add_scalar("Train/PSNR", train_result.psnr, epoch)
        writer.add_scalar("Val/Loss", val_result.loss, epoch)
        writer.add_scalar("Val/PSNR", val_result.psnr, epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_psnr": max(best_psnr, val_result.psnr),
            "model_config": {"base_channels": args.base_channels},
            "args": vars(args),
        }
        torch.save(payload, run_dir / "stage1_last.pth")
        if val_result.psnr > best_psnr:
            best_psnr = val_result.psnr
            torch.save(payload, run_dir / "stage1_best.pth")
        print(
            f"[Stage1 {epoch + 1:03d}/{args.epochs}] "
            f"train={train_result.loss:.5f} val={val_result.loss:.5f} "
            f"PSNR={val_result.psnr:.2f}dB"
        )
    writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--run_version", default="v1")
    parser.add_argument("--log_root", default="logs")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=4)
    parser.add_argument("--volume_size", type=int, nargs=3, default=(256,256,256))
    parser.add_argument("--laplacian_weight", type=float, default=0.10)
    parser.add_argument("--structural_weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_amp", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
