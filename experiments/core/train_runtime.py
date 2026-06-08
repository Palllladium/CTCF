from __future__ import annotations

import os
import time
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.core.cli_args import add_common_args
from experiments.core.data_loaders import (
    baseline_loader_builder,
    ixi_flip_axes_for,
    ixi_loaders,
    loaders_baseline,
    oasis_loaders,
)
from experiments.core.helpers import (
    build_val_dice_fn,
    resume_from_ckpt,
    save_ckpt,
    select_lr_policy,
)
from experiments.core.path_profiles import PATHS, print_experiment_header
from experiments.core.train_logging import (
    format_iter_log,
    format_metric_suffix,
    format_train_suffix,
    write_tb_images,
)
from utils import (
    AverageMeter,
    Grad3d,
    NCCVxm,
    RegisterModel,
    attach_stdout_logger,
    make_exp_dirs,
    mk_grid_img,
    perf_epoch_end,
    perf_epoch_start,
    validate,
)

__all__ = [
    "PATHS",
    "TrainContext",
    "add_common_args",
    "baseline_loader_builder",
    "ixi_flip_axes_for",
    "ixi_loaders",
    "loaders_baseline",
    "oasis_loaders",
    "run_train",
    "write_tb_images",
]


class TrainContext:
    """Shared training context: similarity loss, regulariser and spatial transformer."""

    def __init__(
        self,
        device: torch.device,
        vol_size: tuple[int, int, int],
        ncc_win: tuple[int, int, int] = (9, 9, 9),
        reg_penalty: str = "l2",
        interp: str = "bilinear",
    ):
        self.ncc: Any = NCCVxm(win=ncc_win)
        self.reg: Any = Grad3d(penalty=reg_penalty)
        self.reg_model: Any = RegisterModel(img_size=vol_size, mode=interp).to(device)


def _log_perf_scalars(writer: SummaryWriter, perf, epoch: int) -> None:
    perf_scalars = {
        "perf/epoch_time_sec": perf.epoch_time_sec,
        "perf/iter_time_ms": perf.mean_iter_time_ms,
        "perf/peak_gpu_mem_GB": perf.peak_gpu_mem_gib,
    }
    for tag, value in perf_scalars.items():
        if value is None:
            continue
        writer.add_scalar(tag, value, epoch)


def _log_val_scalars(
    writer: SummaryWriter,
    dsc: float,
    jacp: float,
    ndvp: float | None,
    sdlogj: float | None,
    jac_tb_tag: str,
    epoch: int,
) -> None:
    val_scalars: dict[str, float] = {"val/Dice": dsc, jac_tb_tag: jacp}
    if ndvp is not None:
        val_scalars["val/NDV%"] = ndvp
    if sdlogj is not None:
        val_scalars["val/SDlogJ"] = sdlogj
    for tag, value in val_scalars.items():
        writer.add_scalar(tag, value, epoch)


def run_train(args, runner, build_loaders=loaders_baseline) -> None:
    """Generic AMP training loop with periodic validation and optional checkpointing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    device = runner.device
    paths = make_exp_dirs(args.exp or "EXP")
    attach_stdout_logger(paths.log_dir, quiet=bool(args.quiet))

    save_ckpt_enabled = bool(args.save_ckpt)
    use_tb = bool(args.use_tb)
    ckpt_dir = os.path.join(paths.exp_dir, "ckpt")
    if save_ckpt_enabled:
        os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(paths.exp_dir, "vis"), exist_ok=True)
    writer = SummaryWriter(log_dir=paths.log_dir) if use_tb else None

    train_loader, val_loader = build_loaders(args)
    max_train_iters = args.max_train_iters
    raw_max_val = args.max_val_batches
    max_val_batches = None if raw_max_val <= 0 else raw_max_val
    scaler = torch.amp.GradScaler("cuda")
    train_total = len(train_loader) if max_train_iters <= 0 else min(len(train_loader), max_train_iters)

    ds_key = args.ds.upper()
    jac_name = "j<=0%"
    jac_tb_tag = "val/J<=0%"
    val_dice_fn = build_val_dice_fn(args, ds_key)

    epoch_start, best_dsc = resume_from_ckpt(
        args,
        runner,
        scaler,
        device,
        best_dsc_init=-1.0,
    )
    print_experiment_header(args, ds_key)

    nan_streak_limit = 3
    zero_dice_streak_limit = 3
    nan_streak = 0
    zero_dice_streak = 0

    for epoch in range(epoch_start, args.max_epoch):
        runner.model.train()
        t0 = perf_epoch_start()

        lr_now = select_lr_policy(
            runner,
            runner.optimizer,
            epoch,
            args.max_epoch,
            args.lr,
        )
        if writer is not None:
            writer.add_scalar("LR", lr_now, epoch)

        meters: dict[str, AverageMeter] = {}
        iter_time_sum = 0.0
        train_iters_done = 0
        print(f"Training Starts (epoch {epoch:03d})")

        for it, batch in enumerate(train_loader, start=1):
            if max_train_iters > 0 and it > max_train_iters:
                break

            t_it = time.perf_counter()
            runner.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss, logs = runner.train_step(batch, epoch)

            if not torch.isfinite(loss):
                nan_streak += 1
                print(
                    f"WARNING: non-finite loss at epoch {epoch} iter {it} (streak {nan_streak}/{nan_streak_limit})",
                )
                if nan_streak >= nan_streak_limit:
                    raise RuntimeError(
                        f"ABORT: {nan_streak_limit} consecutive non-finite losses. Check model/hyperparams.",
                    )
                runner.optimizer.zero_grad(set_to_none=True)
                continue
            nan_streak = 0

            scaler.scale(loss).backward()
            scaler.step(runner.optimizer)
            scaler.update()

            if logs:
                for k, v in logs.items():
                    meters.setdefault(k, AverageMeter()).update(float(v), 1)

            iter_time_sum += time.perf_counter() - t_it
            train_iters_done += 1

            if it % 10 == 0 and meters:
                print(format_iter_log(meters, it, train_total, lr_now))

        if writer is not None:
            for k, m in meters.items():
                writer.add_scalar(f"Loss/train_{k}", m.avg, epoch)

        perf = perf_epoch_end(t0, iters=train_iters_done, iter_time_sum=iter_time_sum)
        if writer is not None:
            _log_perf_scalars(writer, perf, epoch)

        runner.model.eval()
        need_vis = writer is not None and args.tb_images_every and epoch % args.tb_images_every == 0
        with torch.no_grad():
            val_res = validate(
                model=runner.model,
                val_loader=val_loader,
                device=device,
                forward_flow_fn=runner.forward_flow,
                dice_fn=val_dice_fn,
                register_model_cls=RegisterModel,
                mk_grid_img_fn=mk_grid_img if need_vis else None,
                grid_step=8,
                line_thickness=1,
                max_batches=max_val_batches,
                ds=args.ds,
            )

        dsc = val_res.dsc
        jacp = val_res.jac_nonpos_percent
        ndvp = val_res.ndv_percent
        sdlogj = val_res.sdlogj

        if dsc < 1e-6:
            zero_dice_streak += 1
            print(
                f"WARNING: val_dice~0 at epoch {epoch} (streak {zero_dice_streak}/{zero_dice_streak_limit})",
            )
            if zero_dice_streak >= zero_dice_streak_limit:
                raise RuntimeError(
                    f"ABORT: val_dice~0 for {zero_dice_streak_limit} consecutive epochs. Model is not learning.",
                )
        else:
            zero_dice_streak = 0

        if writer is not None:
            _log_val_scalars(writer, dsc, jacp, ndvp, sdlogj, jac_tb_tag, epoch)
        if hasattr(runner, "on_val_end"):
            runner.on_val_end(epoch, dsc, jacp)
        if need_vis:
            write_tb_images(writer, val_res.last_vis or {}, epoch)

        is_best = dsc > best_dsc
        if is_best:
            best_dsc = dsc

        state = {
            "epoch": epoch,
            "state_dict": runner.model.state_dict(),
            "optimizer": runner.optimizer.state_dict(),
            "best_dsc": best_dsc,
            "scaler": scaler.state_dict(),
        }
        if save_ckpt_enabled:
            save_ckpt(state, ckpt_dir, is_best)

        metric_suffix = format_metric_suffix(ndvp, sdlogj)
        train_suffix = format_train_suffix(meters)
        peak_mem = "n/a" if perf.peak_gpu_mem_gib is None else f"{perf.peak_gpu_mem_gib:.2f}GB"

        print(
            f"[epoch {epoch:03d}] val_dice={dsc:.4f} best={best_dsc:.4f} | "
            f"{jac_name}={jacp:.2f}{metric_suffix}{train_suffix}",
        )
        print(
            f"[perf  {epoch:03d}] epoch={perf.epoch_time_sec:.2f}s iter={perf.mean_iter_time_ms:.1f}ms peak={peak_mem}",
        )

    print(f">>> Training complete: {args.max_epoch} epochs, best_dice={best_dsc:.4f}")
    if writer is not None:
        writer.close()
