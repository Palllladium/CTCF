import argparse
import glob
import os
import time
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import OASIS, IXI
from utils import (
    adjust_learning_rate_poly, make_exp_dirs, attach_stdout_logger, load_checkpoint_if_exists,
    perf_epoch_start, perf_epoch_end, mk_grid_img, compute_fig, validate,
    AverageMeter, RegisterModel, NCCVxm, Grad3d, NumpyType, RandomFlip, SegNorm,
    dice_val, dice_val_VOI
)


PATHS = {
    1: {
        "OASIS": {
            "train_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/All",
            "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test",
        },
        "IXI": {
            "train_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Train",
            "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Val",
            "atlas_path": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/atlas.pkl",
        },
    },
    2: {
        "OASIS": {
            "train_dir": "/home/roman/P/OASIS_L2R_2021_task03/All",
            "val_dir": "/home/roman/P/OASIS_L2R_2021_task03/Test",
        },
        "IXI": {
            "train_dir": "/home/roman/P/IXI_data/Train",
            "val_dir": "/home/roman/P/IXI_data/Val",
            "atlas_path": "/home/roman/P/IXI_data/atlas.pkl",
        },
    },
}


class Ctx:
    """Shared train context with losses and spatial transformer."""
    def __init__(self, device, *, vol_size, ncc_win=(9, 9, 9), reg_penalty="l2", interp="bilinear"):
        self.ncc: Any = NCCVxm(win=ncc_win)
        self.reg: Any = Grad3d(penalty=reg_penalty)
        self.reg_model: Any = RegisterModel(tuple(vol_size), mode=interp).to(device)


def add_common_args(p: argparse.ArgumentParser, *, include_synth: bool = False, mode: str = "train"):
    """Add shared CLI args for train/infer experiment scripts."""
    if mode not in ("train", "infer"):
        raise ValueError(f"Unsupported mode='{mode}'. Expected 'train' or 'infer'.")
    ds_choices = ["OASIS", "IXI", "SYNTH"] if include_synth else ["OASIS", "IXI"]
    p.set_defaults(paths=1)
    p.add_argument("--ds", choices=ds_choices, default="OASIS", help="Dataset key to run.")
    p.add_argument("--1", dest="paths", action="store_const", const=1, help="Use path profile #1")
    p.add_argument("--2", dest="paths", action="store_const", const=2, help="Use path profile #2")
    p.add_argument("--paths", type=int, help="Path profile id (1/2/...)")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader worker processes.")

    if mode == "train":
        p.add_argument("--exp", default="", help="Experiment name used for logs/results directories.")
        p.add_argument("--max_epoch", type=int, default=400, help="Number of training epochs.")
        p.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
        p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
        p.add_argument("--max_train_iters", type=int, default=0, help="Limit train iterations per epoch; <=0 disables limit.")
        p.add_argument("--max_val_batches", type=int, default=0, help="Limit validation batches per epoch; <=0 disables limit.")
        p.add_argument("--resume", default="", help="Checkpoint path to resume training from.")
        p.add_argument("--save_ckpt", type=int, choices=[0, 1], default=1, help="Enable/disable checkpoint saving to disk.")
        p.add_argument("--use_tb", type=int, choices=[0, 1], default=1, help="Enable/disable TensorBoard logging.")
        p.add_argument("--tb_images_every", type=int, default=5, help="TensorBoard image logging period in epochs.")
    return p


def oasis_loaders(args, *, train_cls, val_cls, val_bs=1, drop_last_train=False, drop_last_val=True):
    """Build OASIS train/val loaders for a pair of dataset classes."""
    paths = PATHS[int(args.paths)][args.ds.upper()]
    
    train_dir = paths.get("train_dir", "")
    val_dir = paths.get("val_dir", "")
    if not os.path.isdir(train_dir): raise RuntimeError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir): raise RuntimeError(f"Validation dir not found: {val_dir}")

    tr_files = glob.glob(os.path.join(train_dir, "*.pkl"))
    va_files = glob.glob(os.path.join(val_dir, "*.pkl"))
    if not tr_files: raise RuntimeError(f"OASIS: no *.pkl in Train dir = {train_dir}")
    if not va_files: raise RuntimeError(f"OASIS: no *.pkl in Validation dir = {val_dir}")

    tfm = transforms.Compose([NumpyType((np.float32, np.int16))])
    tr = train_cls(tr_files, transforms=tfm)
    va = val_cls(va_files, transforms=tfm)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_train)
    val_loader = DataLoader(va, batch_size=val_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_val)
    return train_loader, val_loader


def ixi_loaders(args, *, train_cls, val_cls, val_bs=1, drop_last_train=True, drop_last_val=True):
    """Build IXI train/val loaders for a pair of dataset classes."""
    paths = PATHS[int(args.paths)][args.ds.upper()]
    
    train_dir = paths.get("train_dir", "")
    val_dir = paths.get("val_dir", "")
    atlas_path = str(paths.get("atlas_path", "")).rstrip("/\\")
    if not os.path.isdir(train_dir): raise RuntimeError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir): raise RuntimeError(f"Validation dir not found: {val_dir}")
    if not atlas_path or not os.path.exists(atlas_path): raise RuntimeError(f"Atlas path not found: {atlas_path}")

    tr_files = glob.glob(os.path.join(train_dir, "*.pkl"))
    va_files = glob.glob(os.path.join(val_dir, "*.pkl"))
    if not tr_files: raise RuntimeError(f"IXI: no *.pkl in Train dir = {train_dir}")
    if not va_files: raise RuntimeError(f"IXI: no *.pkl in Validation dir = {val_dir}")

    train_tfm = transforms.Compose([RandomFlip((1, 2, 3)), NumpyType((np.float32, np.float32))])
    val_tfm = transforms.Compose([SegNorm(), NumpyType((np.float32, np.int16))])
    tr = train_cls(tr_files, atlas_path, transforms=train_tfm)
    va = val_cls(va_files, atlas_path, transforms=val_tfm)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_train)
    val_loader = DataLoader(va, batch_size=val_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_val)
    return train_loader, val_loader


def loaders_baseline(args):
    """Dispatch baseline data loader factory by dataset name."""
    match args.ds:
        case "OASIS": 
            return oasis_loaders(
                args,
                train_cls=OASIS.OASISBrainDataset,
                val_cls=OASIS.OASISBrainInferDataset,
                val_bs=1,
                drop_last_train=False,
                drop_last_val=True)
        case "IXI":
            return ixi_loaders(
                args,
                train_cls=IXI.IXIBrainDataset,
                val_cls=IXI.IXIBrainInferDataset,
                val_bs=1,
                drop_last_train=True,
                drop_last_val=True)
        case _:
            raise ValueError(f"Unsupported dataset = '{args.ds}' for baseline loaders.")


def write_tb_images(writer: SummaryWriter, last_vis: dict, epoch: int):
    """Write segmentation and grid previews to TensorBoard."""
    if not last_vis: return
    def_out, def_grid, x_vis, y_vis = last_vis.get("def_seg"), last_vis.get("def_grid"), last_vis.get("x_seg"), last_vis.get("y_seg")
    if def_out is None or def_grid is None or x_vis is None or y_vis is None: return
   
    plt.switch_backend("agg")
    pred_fig, grid_fig, x_fig, tar_fig = compute_fig(def_out), compute_fig(def_grid), compute_fig(x_vis), compute_fig(y_vis)
    writer.add_figure("Grid", grid_fig, epoch); plt.close(grid_fig)
    writer.add_figure("Input", x_fig, epoch); plt.close(x_fig)
    writer.add_figure("Ground truth", tar_fig, epoch); plt.close(tar_fig)
    writer.add_figure("Prediction", pred_fig, epoch); plt.close(pred_fig)


def run_train(*, args, runner, build_loaders=loaders_baseline):
    """Run generic AMP training loop with periodic validation and optional checkpointing."""
    assert torch.cuda.is_available(), "CUDA required"
    device = runner.device
    paths = make_exp_dirs(args.exp or "EXP")
    attach_stdout_logger(paths.log_dir)
    ckpt_dir = os.path.join(paths.exp_dir, "ckpt")
    save_ckpt = bool(int(args.save_ckpt))
    use_tb = bool(int(args.use_tb))
    if save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(paths.exp_dir, "vis"), exist_ok=True)
    writer = SummaryWriter(log_dir=paths.log_dir) if use_tb else None

    train_loader, val_loader = build_loaders(args)
    max_train_iters = int(args.max_train_iters)
    max_val_batches = int(args.max_val_batches)
    max_val_batches = None if max_val_batches <= 0 else max_val_batches
    scaler = torch.amp.GradScaler("cuda")
    train_total = len(train_loader) if max_train_iters <= 0 else min(len(train_loader), max_train_iters)
    val_dice_fn = (lambda p, t: dice_val_VOI(p, t)) if args.ds == "OASIS" else (lambda p, t: dice_val(p, t, 46))

    epoch_start, best_dsc = 0, -1.0
    if args.resume:
        ckpt = load_checkpoint_if_exists(args.resume, model=runner.model, optimizer=runner.optimizer, map_location=device)
        if ckpt is not None:
            epoch_start = int(ckpt.get("epoch", -1)) + 1
            best_dsc = float(ckpt.get("best_dsc", best_dsc))
            if ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
            print(f">>> Resumed from {args.resume} @ epoch {epoch_start}, best={best_dsc:.4f}")

    print(f">>> Experiment: {args.exp} | ds={args.ds} | paths={args.paths}")
    if args.ds.upper() in ("OASIS", "IXI"):
        p = PATHS[int(args.paths)][args.ds.upper()]
        print(f"    Train dir = {p['train_dir']}")
        print(f"    Val dir   = {p['val_dir']}")
        if args.ds.upper() == "IXI":
            atlas_path = str(p["atlas_path"]).rstrip("/\\")
            print(f"    Atlas     = {atlas_path}")

    for epoch in range(epoch_start, int(args.max_epoch)):
        runner.model.train()
        t0 = perf_epoch_start()

        lr_now = adjust_learning_rate_poly(runner.optimizer, epoch, int(args.max_epoch), args.lr)
        if writer is not None: writer.add_scalar("LR", lr_now, epoch)

        meters, iter_time_sum = {}, 0.0
        train_iters_done = 0
        print(f"Training Starts (epoch {epoch:03d})")

        for it, batch in enumerate(train_loader, start=1):
            if max_train_iters > 0 and it > max_train_iters:
                break
            t_it = time.perf_counter()
            runner.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss, logs = runner.train_step(batch, epoch)
            scaler.scale(loss).backward()
            scaler.step(runner.optimizer)
            scaler.update()

            if logs:
                for k, v in logs.items():
                    meters.setdefault(k, AverageMeter()).update(float(v), 1)

            iter_time_sum += (time.perf_counter() - t_it)
            train_iters_done += 1

            if it % 10 == 0 and meters:
                main = "all" if "all" in meters else next(iter(meters.keys()))
                msg = f"Iter {it:4d} / {train_total:4d} | {main}(avg)={meters[main].avg:.4f}"

                if "ncc" in meters:  msg += f" | ncc={meters['ncc'].val:.4f}"
                if "reg" in meters:  msg += f" reg={meters['reg'].val:.4f}"
                if "icon" in meters: msg += f" icon={meters['icon'].val:.4f}"
                if "cyc" in meters:  msg += f" cyc={meters['cyc'].val:.4f}"
                if "jac" in meters:  msg += f" jac={meters['jac'].val:.4f}"
                if "dice_tr" in meters: msg += f" dice_tr={meters['dice_tr'].val:.4f}"

                msg += f" | lr={lr_now:.3e}"

                if "phase" in meters:
                    msg += (
                        f" | ctrl: ph={int(meters['phase'].val)}"
                        f" a3={meters['a3'].val:.2f}"
                        f" wJ={meters['wJ'].val:.2f} wI={meters['wI'].val:.2f} wC={meters['wC'].val:.2f}"
                    )
                print(msg)

        if writer is not None:
            for k, m in meters.items():
                writer.add_scalar(f"Loss/train_{k}", m.avg, epoch)

        perf = perf_epoch_end(t0, iters=train_iters_done, iter_time_sum=iter_time_sum)
        if writer is not None:
            writer.add_scalar("perf/epoch_time_sec", perf.epoch_time_sec, epoch)
            writer.add_scalar("perf/iter_time_ms", perf.mean_iter_time_ms, epoch)
            writer.add_scalar("perf/peak_gpu_mem_GB", perf.peak_gpu_mem_gib, epoch)

        runner.model.eval()
        need_vis = bool(writer is not None and args.tb_images_every and (epoch % int(args.tb_images_every) == 0))
        with torch.no_grad():
            val_res = validate(
                model=runner.model,
                val_loader=val_loader,
                device=device,
                forward_flow_fn=lambda x, y: runner.forward_flow(x, y),
                dice_fn=val_dice_fn,
                register_model_cls=RegisterModel,
                mk_grid_img_fn=mk_grid_img if need_vis else None,
                grid_step=8,
                line_thickness=1,
                max_batches=max_val_batches,
                fold_use_mask=(args.ds == "IXI"),
                fold_crop=1,
            )

        dsc, foldp = float(val_res.dsc), float(val_res.fold_percent)
        if writer is not None:
            writer.add_scalar("val/Dice", dsc, epoch)
            writer.add_scalar("val/Fold%", foldp, epoch)

        ctrl_suffix = ""
        ctrl = getattr(getattr(runner, "ctx", None), "ctcf_ctrl", None)
        if ctrl is not None:
            ctrl.on_val_end(epoch=epoch, val_dice=dsc, val_fold_percent=foldp)
            if writer is not None:
                for k, v in (ctrl.tb_scalars() or {}).items():
                    writer.add_scalar(k, float(v), epoch)
            a3 = float(ctrl.knobs.alpha_l3)
            ctrl_suffix = (
                f"ctrl: ph={ctrl.phase} a3={a3:.2f} "
                f"wJ={ctrl.knobs.w_jac_mul:.2f} wI={ctrl.knobs.w_icon_mul:.2f} wC={ctrl.knobs.w_cyc_mul:.2f}"
            )

        if writer is not None and args.tb_images_every and (epoch % int(args.tb_images_every) == 0):
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

        if save_ckpt:
            torch.save(state, os.path.join(ckpt_dir, "last.pth"))
            if is_best:
                torch.save(state, os.path.join(ckpt_dir, "best.pth"))

        print(f"[epoch {epoch:03d}] val_dice={dsc:.4f} best={best_dsc:.4f} | fold%={foldp:.2f}{ctrl_suffix}")
        peak_mem = "n/a" if perf.peak_gpu_mem_gib is None else f"{perf.peak_gpu_mem_gib:.2f}GB"
        print(f"[perf  {epoch:03d}] epoch={perf.epoch_time_sec:.2f}s iter={perf.mean_iter_time_ms:.1f}ms peak={peak_mem}")

    if writer is not None:
        writer.close()
