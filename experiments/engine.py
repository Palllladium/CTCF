import os, glob, time, argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass

from datasets import OASIS, IXI
from utils import (
    adjust_learning_rate_poly, setup_device, make_exp_dirs, attach_stdout_logger, load_checkpoint_if_exists, 
    perf_epoch_start, perf_epoch_end, dice_val_VOI, mk_grid_img, comput_fig, validate, save_checkpoint, 
    AverageMeter, register_model, NCC_vxm, Grad3d, NumpyType, trans as trans_utils
)


PATHS = {
    1: {
        "OASIS": {"train_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/All",
                 "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test"},
        "IXI":  {"train_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Train",
                 "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Val",
                 "atlas_path": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/atlas.pkl"},
    },
    2: {
        "OASIS": {"train_dir": "/home/roman/P/OASIS_L2R_2021_task03/All",
                 "val_dir": "/home/roman/P/OASIS_L2R_2021_task03/Test"},
        "IXI":  {"train_dir": "/home/roman/P/IXI_data/Train",
                 "val_dir": "/home/roman/P/IXI_data/Val",
                 "atlas_path": "/home/roman/P/IXI_data/atlas.pkl"},
    },
}


@dataclass
class Ctx:
    ncc: any
    reg: any
    reg_model: any


def make_ctx(device, *, vol_size, ncc_win=(9, 9, 9), reg_penalty="l2", interp="bilinear"):
    return Ctx(
        ncc=NCC_vxm(win=ncc_win),
        reg=Grad3d(penalty=reg_penalty),
        reg_model=register_model(tuple(vol_size), mode=interp).to(device),
    )


def _norm_dir(p: str) -> str:
    return p.rstrip("/\\") + os.sep


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--ds", choices=["OASIS", "IXI"], default="OASIS")
    p.add_argument("--1", dest="paths", action="store_const", const=1, help="Use path profile #1")
    p.add_argument("--2", dest="paths", action="store_const", const=2, help="Use path profile #2")
    p.add_argument("--paths", type=int, default=1, help="Path profile id (1/2/...)")
    p.add_argument("--train_dir", default="")
    p.add_argument("--val_dir", default="")
    p.add_argument("--atlas_path", default="")
    p.add_argument("--exp", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_epoch", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", default="")
    p.add_argument("--tb_images_every", type=int, default=5)
    p.add_argument("--grid_step", type=int, default=8)
    p.add_argument("--line_thickness", type=int, default=1)
    return p


def apply_paths(args):
    ds = args.ds.upper()
    prof = int(getattr(args, "paths", 1) or 1)
    prof_map = PATHS.get(prof, {}).get(ds, {})

    if not args.train_dir: args.train_dir = prof_map.get("train_dir", "")
    if not args.val_dir:   args.val_dir = prof_map.get("val_dir", "")
    if ds == "IXI" and not args.atlas_path: args.atlas_path = prof_map.get("atlas_path", "")

    args.train_dir, args.val_dir = _norm_dir(args.train_dir), _norm_dir(args.val_dir)
    if ds == "IXI": args.atlas_path = args.atlas_path.rstrip("/\\")

    if not os.path.isdir(args.train_dir): raise RuntimeError(f"Train dir not found: {args.train_dir}")
    if not os.path.isdir(args.val_dir): raise RuntimeError(f"Validation dir not found: {args.val_dir}")
    if ds == "IXI" and not os.path.exists(args.atlas_path): raise RuntimeError(f"Atlas path not found: {args.atlas_path}")

    return args


def oasis_loaders(args, *, train_cls, val_cls, val_bs=1, drop_last_train=False, drop_last_val=True):
    tfm = transforms.Compose([NumpyType((np.float32, np.int16))])
    
    tr_files = glob.glob(os.path.join(args.train_dir, "*.pkl"))
    va_files = glob.glob(os.path.join(args.val_dir, "*.pkl"))
    if not tr_files: raise RuntimeError(f"OASIS: no *.pkl in Train dir={args.train_dir}")
    if not va_files: raise RuntimeError(f"OASIS: no *.pkl in Validation dir={args.val_dir}")

    tr = train_cls(tr_files, transforms=tfm)
    va = val_cls(va_files, transforms=tfm)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_train)
    val_loader = DataLoader(va, batch_size=val_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_val)
    return train_loader, val_loader


def ixi_loaders(args, *, train_cls, val_cls, val_bs=1, drop_last_train=True, drop_last_val=True):
    train_tfm = transforms.Compose([trans_utils.RandomFlip(0), trans_utils.NumpyType((np.float32, np.float32))])
    val_tfm = transforms.Compose([trans_utils.Seg_norm(), trans_utils.NumpyType((np.float32, np.int16))])

    tr_files = glob.glob(os.path.join(args.train_dir, "*.pkl"))
    va_files = glob.glob(os.path.join(args.val_dir, "*.pkl"))
    if not tr_files: raise RuntimeError(f"IXI: no *.pkl in Train dir={args.train_dir}")
    if not va_files: raise RuntimeError(f"IXI: no *.pkl in Validation dir={args.val_dir}")

    tr = train_cls(tr_files, args.atlas_path, transforms=train_tfm)
    va = val_cls(va_files, args.atlas_path, transforms=val_tfm)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_train)
    val_loader = DataLoader(va, batch_size=val_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=drop_last_val)
    return train_loader, val_loader


def loaders_baseline(args):
    if args.ds == "OASIS":
        return oasis_loaders(args, train_cls=OASIS.OASISBrainDataset, val_cls=OASIS.OASISBrainInferDataset, val_bs=1, drop_last_train=False, drop_last_val=True)
    return ixi_loaders(args, train_cls=IXI.IXIBrainDataset, val_cls=IXI.IXIBrainInferDataset, val_bs=1, drop_last_train=True, drop_last_val=True)


def forward_flow_halfres(model, x, y, *, pool=2, amp=True):
    xh, yh = F.avg_pool3d(x, pool), F.avg_pool3d(y, pool)
    use_amp = amp and torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp): flow_h = model((xh, yh))
    return F.interpolate(flow_h.float(), scale_factor=pool, mode="trilinear", align_corners=False) * float(pool)


def write_tb_images(writer: SummaryWriter, last_vis: dict, epoch: int):
    if not last_vis: return
    def_out, def_grid, x_vis, y_vis = last_vis.get("def_seg"), last_vis.get("def_grid"), last_vis.get("x_seg"), last_vis.get("y_seg")
    if def_out is None or def_grid is None or x_vis is None or y_vis is None: return
    plt.switch_backend("agg")
    pred_fig, grid_fig, x_fig, tar_fig = comput_fig(def_out), comput_fig(def_grid), comput_fig(x_vis), comput_fig(y_vis)
    writer.add_figure("Grid", grid_fig, epoch); plt.close(grid_fig)
    writer.add_figure("input", x_fig, epoch); plt.close(x_fig)
    writer.add_figure("ground truth", tar_fig, epoch); plt.close(tar_fig)
    writer.add_figure("prediction", pred_fig, epoch); plt.close(pred_fig)


def run_train(*, args, runner, build_loaders=loaders_baseline):
    assert torch.cuda.is_available(), "CUDA required"
    device = runner.device
    paths = make_exp_dirs(args.exp or "EXP")
    attach_stdout_logger(paths.log_dir)
    ckpt_dir = os.path.join(paths.exp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(paths.exp_dir, "vis"), exist_ok=True)
    writer = SummaryWriter(log_dir=paths.log_dir)

    train_loader, val_loader = build_loaders(args)
    scaler = torch.amp.GradScaler("cuda")
    use_amp = True

    epoch_start, best_dsc = 0, -1.0
    if args.resume:
        ckpt = load_checkpoint_if_exists(args.resume, model=runner.model, optimizer=runner.optimizer, map_location=device)
        if ckpt is not None:
            epoch_start = int(ckpt.get("epoch", -1)) + 1
            best_dsc = float(ckpt.get("best_dsc", best_dsc))
            if ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
            print(f">>> Resumed from {args.resume} @ epoch {epoch_start}, best={best_dsc:.4f}")

    print(f">>> Experiment: {args.exp} | ds={args.ds} | paths={getattr(args,'paths',1)}")
    print(f"    train_dir = {args.train_dir}")
    print(f"    val_dir   = {args.val_dir}")
    if args.ds.upper() == "IXI": print(f"    atlas     = {args.atlas_path}")

    for epoch in range(epoch_start, int(args.max_epoch)):
        runner.model.train()
        t0 = perf_epoch_start()
        
        lr_now = adjust_learning_rate_poly(runner.optimizer, epoch, int(args.max_epoch), args.lr)
        writer.add_scalar("LR", lr_now, epoch)

        meters, iter_time_sum = {}, 0.0
        print(f"Training Starts (epoch {epoch:03d})")

        for it, batch in enumerate(train_loader, start=1):
            t_it = time.perf_counter()
            runner.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss, logs = runner.train_step(batch, epoch)
            scaler.scale(loss).backward()
            scaler.step(runner.optimizer)
            scaler.update()

            if logs:
                for k, v in logs.items():
                    meters.setdefault(k, AverageMeter()).update(float(v), 1)

            iter_time_sum += (time.perf_counter() - t_it)
            
            if it % 10 == 0 and meters:
                main = "all" if "all" in meters else next(iter(meters.keys()))
                msg = f"Iter {it:4d} / {len(train_loader):4d} | {main}(avg)={meters[main].avg:.4f}"

                if "ncc" in meters:  msg += f" | ncc={meters['ncc'].val:.4f}"
                if "reg" in meters:  msg += f" reg={meters['reg'].val:.4f}"
                if "icon" in meters: msg += f" icon={meters['icon'].val:.4f}"
                if "cyc" in meters:  msg += f" cyc={meters['cyc'].val:.4f}"
                if "jac" in meters:  msg += f" jac={meters['jac'].val:.4f}"

                msg += f" | lr={lr_now:.3e}"

                if "phase" in meters:
                    msg += (
                        f" | ctrl: ph={int(meters['phase'].val)}"
                        f" a3={meters['a3'].val:.2f}"
                        f" wJ={meters['wJ'].val:.2f} wI={meters['wI'].val:.2f} wC={meters['wC'].val:.2f}"
                    )
                print(msg)

        for k, m in meters.items(): 
            writer.add_scalar(f"Loss/train_{k}", m.avg, epoch)

        perf = perf_epoch_end(t0, iters=len(train_loader), iter_time_sum=iter_time_sum)
        writer.add_scalar("perf/epoch_time_sec", perf.epoch_time_sec, epoch)
        writer.add_scalar("perf/iter_time_ms", perf.mean_iter_time_ms, epoch)
        writer.add_scalar("perf/peak_gpu_mem_GB", perf.peak_gpu_mem_gib, epoch)

        runner.model.eval()
        with torch.no_grad():
            val_res = validate(
                model=runner.model, val_loader=val_loader, device=device,
                forward_flow_fn=lambda x, y: runner.forward_flow(x, y),
                dice_fn=dice_val_VOI, register_model_cls=register_model, mk_grid_img_fn=mk_grid_img,
                grid_step=int(args.grid_step), line_thickness=int(args.line_thickness)
            )

        dsc, foldp = float(val_res.dsc), float(val_res.fold_percent)
        writer.add_scalar("val/Dice", dsc, epoch)
        writer.add_scalar("val/Fold%", foldp, epoch)

        ctrl_suffix = ""
        ctrl = getattr(getattr(runner, "ctx", None), "ctcf_ctrl", None)
        if ctrl is not None:
            ctrl.on_val_end(epoch=epoch, val_dice=dsc, val_fold_percent=foldp)
            for k, v in (ctrl.tb_scalars() or {}).items():
                writer.add_scalar(k, float(v), epoch)
            ctrl_suffix = (
                f"ctrl: ph={ctrl.phase} a3={ctrl.knobs.alpha_l3:.2f} "
                f"wJ={ctrl.knobs.w_jac_mul:.2f} wI={ctrl.knobs.w_icon_mul:.2f} wC={ctrl.knobs.w_cyc_mul:.2f}"
            )

        if args.tb_images_every and (epoch % int(args.tb_images_every) == 0):
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

        torch.save(state, os.path.join(ckpt_dir, "last.pth"))
        if is_best: torch.save(state, os.path.join(ckpt_dir, "best.pth"))
        save_checkpoint(state, save_dir=os.path.join(ckpt_dir, "epochs"),
                        filename=f"epoch_{epoch:04d}.pth", max_model_num=8)

        print(f"[epoch {epoch:03d}] val_dice={dsc:.4f} best={best_dsc:.4f} | fold%={foldp:.2f}{ctrl_suffix}")

    writer.close()