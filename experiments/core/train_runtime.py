from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from experiments.core.cli_common import add_common_args
from experiments.core.data_loaders import (
    baseline_loader_builder,
    ixi_flip_axes_for,
    ixi_loaders,
    loaders_baseline,
    oasis_loaders,
)
from experiments.core.distributed import (
    accumulation_steps,
    finite_on_all,
    reduce_epoch,
    restore_training_state,
    training_objective,
    training_state,
    world_info,
)
from experiments.core.helpers import (
    build_val_dice_fn,
    resume_from_ckpt,
    save_ckpt,
    select_lr_policy,
)
from experiments.core.path_profiles import PATHS, print_experiment_header
from experiments.core.precision import configure_precision, precision_autocast
from experiments.core.train_logging import (
    append_epoch_metrics,
    format_iter_log,
    format_metric_suffix,
    format_train_suffix,
    prepare_metrics_csv,
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
    rank, world = world_info()
    accumulation = accumulation_steps(args, world)
    global_batch = getattr(args, "global_batch_size", 0)
    experiment_name = args.exp or "EXP"
    paths = make_exp_dirs(experiment_name)
    log_dir = paths.log_dir if rank == 0 else os.path.join(paths.log_dir, f"rank_{rank}")
    attach_stdout_logger(log_dir, quiet=bool(args.quiet) or rank != 0)

    save_ckpt_enabled = bool(args.save_ckpt)
    use_tb = bool(args.use_tb)
    ckpt_dir = os.path.join(paths.exp_dir, "ckpt")
    if save_ckpt_enabled:
        os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(paths.exp_dir, "vis"), exist_ok=True)

    train_loader, val_loader = build_loaders(args)
    max_train_iters = args.max_train_iters
    raw_max_val = args.max_val_batches
    max_val_batches = None if raw_max_val <= 0 else raw_max_val
    precision = getattr(args, "precision", "fp16")
    if hasattr(args, "precision"):
        configure_precision(precision, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    print(f">>> precision={precision} scaler={scaler.is_enabled()}")
    train_total = len(train_loader) // accumulation
    if not global_batch and max_train_iters > 0:
        train_total = min(train_total, max_train_iters)
    if not train_total:
        raise ValueError("training loader does not contain a full optimizer update")
    warmup_updates = getattr(args, "perf_warmup_updates", 0)
    if not 0 <= warmup_updates < train_total:
        raise ValueError("perf_warmup_updates must leave at least one measured update")
    print(
        f">>> world_size={world} micro_batch={args.batch_size} global_batch={global_batch or args.batch_size} "
        f"accumulation={accumulation} updates_per_epoch={train_total}"
    )

    ds_key = args.ds.upper()
    jac_name = "j<=0%"
    jac_tb_tag = "val/J<=0%"
    val_dice_fn = build_val_dice_fn(args, ds_key)

    epoch_start, best_dsc, resumed_training = resume_from_ckpt(
        args,
        runner,
        scaler,
        device,
        best_dsc_init=-1.0,
    )
    objective = training_objective(runner)
    restore_training_state(resumed_training, args, device, epoch_start, train_loader.sampler)
    metrics_journal = (
        prepare_metrics_csv(
            os.path.join(paths.log_dir, "metrics.csv"),
            experiment=experiment_name,
            dataset=ds_key,
            epoch_start=epoch_start,
        )
        if rank == 0
        else None
    )
    writer = SummaryWriter(log_dir=paths.log_dir) if use_tb and rank == 0 else None
    print_experiment_header(args, ds_key)

    nonfinite_streak_limit = 3
    zero_dice_streak_limit = 3
    nan_streak = 0
    zero_dice_streak = 0
    amp_skip_streak = resumed_training.get("amp_skip_streak", 0)

    for epoch in range(epoch_start, args.max_epoch):
        runner.model.train()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if world > 1:
            dist.barrier()
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
        optimizer_updates = 0
        print(f"Training Starts (epoch {epoch:03d})")

        batches = iter(train_loader)
        measured_start = None
        for it in range(1, train_total + 1):
            if it == warmup_updates + 1:
                torch.cuda.synchronize(device)
                if world > 1 and warmup_updates:
                    dist.barrier()
                measured_start = time.perf_counter()
            t_it = time.perf_counter()
            runner.optimizer.zero_grad(set_to_none=True)
            micro_logs = []
            bad_loss = False
            for micro in range(accumulation):
                batch = next(batches)
                sync = objective.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
                with sync:
                    with precision_autocast(precision, device):
                        loss, logs = objective(batch, epoch)
                    if not finite_on_all(loss):
                        if global_batch or world > 1 or precision != "fp16":
                            raise FloatingPointError(
                                f"non-finite loss: seed={args.seed} epoch={epoch} update={it} "
                                f"rank={rank} local_loss={float(loss.detach())}; "
                                "GradScaler cannot repair a non-finite forward pass"
                            )
                        nan_streak += 1
                        print(
                            f"WARNING: non-finite loss at epoch {epoch} iter {it} (streak {nan_streak}/{nonfinite_streak_limit})"
                        )
                        if nan_streak >= nonfinite_streak_limit:
                            raise RuntimeError(f"ABORT: {nonfinite_streak_limit} consecutive non-finite losses")
                        bad_loss = True
                        break
                    nan_streak = 0
                    scaler.scale(loss / accumulation).backward()
                    micro_logs.append(logs or {})
            if bad_loss:
                continue
            if precision != "fp16":
                # The infinite threshold checks finiteness without clipping finite gradients.
                torch.nn.utils.clip_grad_norm_(
                    (p for group in runner.optimizer.param_groups for p in group["params"]),
                    float("inf"),
                    norm_type=float("inf"),
                    error_if_nonfinite=True,
                )
            scale_before_step = scaler.get_scale()
            scaler.step(runner.optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before_step
            optimizer_updates += int(not skipped)
            amp_skip_streak = amp_skip_streak + 1 if skipped else 0
            if skipped and (global_batch or world > 1):
                if rank == 0:
                    print(
                        f"WARNING: FP16 optimizer update skipped: seed={args.seed} epoch={epoch} update={it} "
                        f"scale={scale_before_step:g}->{scaler.get_scale():g} "
                        f"consecutive={amp_skip_streak}/{nonfinite_streak_limit}",
                        flush=True,
                    )
                if amp_skip_streak >= nonfinite_streak_limit:
                    raise FloatingPointError(
                        f"{nonfinite_streak_limit} consecutive FP16 updates skipped for nonfinite gradients; "
                        "diagnose the failing update before resuming"
                    )
            for logs in micro_logs:
                for k, v in logs.items():
                    meters.setdefault(k, AverageMeter()).update(float(v), 1)
            meters.setdefault("amp_skipped", AverageMeter()).update(float(skipped))

            iter_time_sum += time.perf_counter() - t_it
            train_iters_done += 1

            if rank == 0 and it % 10 == 0 and meters:
                print(format_iter_log(meters, it, train_total, lr_now))

        torch.cuda.synchronize(device)
        perf, measured_wall, allocated = reduce_epoch(
            meters,
            perf_epoch_end(t0, iters=train_iters_done, iter_time_sum=iter_time_sum),
            device,
            time.perf_counter() - measured_start,
        )
        if global_batch:
            for name, value in (
                ("global_batch_size", global_batch),
                ("world_size", world),
                ("accumulation_steps", accumulation),
                ("pairs_per_epoch", train_iters_done * global_batch),
                ("optimizer_updates", optimizer_updates),
                ("measured_updates", train_total - warmup_updates),
                ("measured_wall_seconds", measured_wall),
                ("measured_pairs_per_second", (train_total - warmup_updates) * global_batch / measured_wall),
                ("peak_allocated_gib", allocated),
            ):
                meters.setdefault(name, AverageMeter()).update(value)
            if rank == 0:
                print(
                    f"[throughput] measured_updates={train_total - warmup_updates} "
                    f"pairs_per_second={(train_total - warmup_updates) * global_batch / measured_wall:.3f} "
                    f"peak_allocated_gib={allocated:.3f}"
                )
        if writer is not None:
            for k, m in meters.items():
                writer.add_scalar(f"Loss/train_{k}", m.avg, epoch)
            _log_perf_scalars(writer, perf, epoch)

        runner.model.eval()
        need_vis = writer is not None and args.tb_images_every and epoch % args.tb_images_every == 0
        with torch.no_grad():
            val_res = None
            if rank == 0:
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
        values = [(val_res.dsc, val_res.jac_nonpos_percent, val_res.ndv_percent, val_res.sdlogj) if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(values, src=0)
        dsc, jacp, ndvp, sdlogj = values[0]

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

        saved_training = training_state(args, device, train_loader.sampler) if save_ckpt_enabled else None
        if saved_training is not None:
            saved_training["amp_skip_streak"] = amp_skip_streak
        if rank != 0:
            continue
        append_epoch_metrics(
            metrics_journal,
            experiment=experiment_name,
            dataset=ds_key,
            epoch=epoch,
            learning_rate=lr_now,
            train_iterations=train_iters_done,
            meters=meters,
            perf_epoch_time_sec=perf.epoch_time_sec,
            perf_iter_time_ms=perf.mean_iter_time_ms,
            perf_peak_gpu_mem_gib=perf.peak_gpu_mem_gib,
            val_dice=dsc,
            val_jac_nonpositive_percent=jacp,
            val_ndv_percent=ndvp,
            val_sdlogj=sdlogj,
            is_best=is_best,
            best_val_dice=best_dsc,
        )

        state = {
            "epoch": epoch,
            "state_dict": runner.model.state_dict(),
            "optimizer": runner.optimizer.state_dict(),
            "best_dsc": best_dsc,
            "scaler": scaler.state_dict(),
            "precision": precision,
            "training": saved_training,
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
