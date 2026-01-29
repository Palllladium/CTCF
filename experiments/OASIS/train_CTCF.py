import os
import glob
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import optim

from experiments.OASIS import datasets
from models.CTCF.model import CTCF_CascadeA, CONFIGS as CONFIGS_CTCF

from utils import (
    AverageMeter,
    setup_device,
    make_exp_dirs,
    attach_stdout_logger,
    save_checkpoint,
    load_checkpoint_if_exists,
    perf_epoch_start,
    perf_epoch_end,
    dice_val_VOI,
    adjust_learning_rate_poly,
    NCC_vxm,
    Grad3d,
    NumpyType,
    register_model,
    mk_grid_img,
    comput_fig,
    validate_oasis,
    icon_loss,
    neg_jacobian_penalty,
    ctcf_schedule,
)


@torch.no_grad()
def forward_flow_ctcf(x: torch.Tensor, y: torch.Tensor, *, model: torch.nn.Module) -> torch.Tensor:
    use_amp = torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        _, flow_full = model(x, y)
    return flow_full


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--exp", default="CTCF")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--max_epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_dsc", type=float, default=0.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    p.add_argument("--w_icon", type=float, default=0.1)
    p.add_argument("--w_cyc", type=float, default=0.05)
    p.add_argument("--w_jac", type=float, default=0.01)

    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--unsup", action="store_true")
    p.add_argument("--resume", default="")

    p.add_argument("--no_level1", action="store_true")
    p.add_argument("--no_level3", action="store_true")

    p.add_argument("--recursive", action="store_true")
    p.add_argument("--fast_dev", action="store_true")

    p.add_argument("--tb_images_every", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    if args.fast_dev:
        args.num_workers = 0
        args.max_epoch = min(args.max_epoch, 30)

    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)

    train_dir = args.train_dir.rstrip("/\\")
    val_dir = args.val_dir.rstrip("/\\")
    print(f">>> Experiment: {args.exp}")
    print(f"    train_dir = {train_dir}{os.sep}")
    print(f"    val_dir   = {val_dir}{os.sep}")
    print(f"    max_epoch = {args.max_epoch}")
    print(f"    lr        = {args.lr}")
    print(f"    time_steps= {args.time_steps}")

    paths = make_exp_dirs(args.exp)
    attach_stdout_logger(paths.log_dir)

    ckpt_dir = os.path.join(paths.exp_dir, "ckpt")
    vis_dir = os.path.join(paths.exp_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=paths.log_dir)

    config = CONFIGS_CTCF["CTCF-CascadeA"]
    config.time_steps = int(args.time_steps)
    if args.no_level1:
        config.use_level1 = False
    if args.no_level3:
        config.use_level3 = False

    model = CTCF_CascadeA(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    train_files = glob.glob(os.path.join(train_dir, "*.pkl"))
    val_files = glob.glob(os.path.join(val_dir, "*.pkl"))

    train_set = datasets.OASISBrainDataset(
        train_files,
        transforms=transforms.Compose([NumpyType((np.float32, np.int16))]),
    )
    val_set = datasets.OASISBrainDataset(
        val_files,
        transforms=transforms.Compose([NumpyType((np.float32, np.int16))]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    criterion_ncc = NCC_vxm(win=(9, 9, 9))
    criterion_reg = Grad3d(penalty="l2")

    # warp helpers
    reg_bilin_half = register_model(config.img_size, mode="bilinear").to(device)
    reg_nearest_full = register_model(
        (config.img_size[0] * 2, config.img_size[1] * 2, config.img_size[2] * 2),
        mode="nearest",
    ).to(device)

    # resume
    epoch_start = 0
    best_dsc = -1.0
    if args.resume:
        ckpt = load_checkpoint_if_exists(
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        if ckpt is not None:
            epoch_start = int(ckpt.get("epoch", -1)) + 1
            best_dsc = float(ckpt.get("best_dsc", best_dsc))
            if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            print(f">>> Resumed from {args.resume} @ epoch {epoch_start}, best={best_dsc:.4f}")

    for epoch in range(epoch_start, int(args.max_epoch)):
        model.train()
        t0_epoch = perf_epoch_start()
        cur_lr = adjust_learning_rate_poly(optimizer, epoch, args.max_epoch, args.lr)

        # schedule (alpha for L1/L3, warm multiplier for icon/cyc/jac)
        alpha_l1, alpha_l3, warm = ctcf_schedule(epoch, args.max_epoch)

        W_icon = args.w_icon * warm
        W_cyc = args.w_cyc * warm
        W_jac = args.w_jac * warm

        loss_all = AverageMeter()
        loss_ncc_m = AverageMeter()
        loss_dsc_m = AverageMeter()
        loss_reg_m = AverageMeter()
        loss_icon_m = AverageMeter()
        loss_cyc_m = AverageMeter()
        loss_jac_m = AverageMeter()

        iter_time_sum = 0.0

        idx = 0
        print(f"Training Starts (epoch {epoch})")
        for _, data in enumerate(train_loader):
            t_it = time.perf_counter()
            idx += 1

            x, y, x_seg, y_seg = data
            x = x.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            use_amp = torch.cuda.is_available()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                def_xy, flow_xy, aux_xy = model(x, y, return_all=True, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
                def_yx, flow_yx, aux_yx = model(y, x, return_all=True, alpha_l1=alpha_l1, alpha_l3=alpha_l3)

                x_half = aux_xy["mov_half"]
                y_half = aux_xy["fix_half"]
                out_xy_h = aux_xy["mov_w_half_final"]
                out_yx_h = aux_yx["mov_w_half_final"]
                flow_xy_h = aux_xy["flow_half_final"]
                flow_yx_h = aux_yx["flow_half_final"]

                L_ncc = 0.5 * (criterion_ncc(out_xy_h, y_half) + criterion_ncc(out_yx_h, x_half))
                L_ncc = L_ncc * args.w_ncc

                # DSC supervised branch only if unsup=False
                if args.unsup:
                    L_dsc = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    x_seg_idx = x_seg.to(device).long()
                    y_seg_idx = y_seg.to(device).long()
                    x_seg_w = reg_nearest_full((x_seg_idx.float(), flow_xy))
                    y_seg_w = reg_nearest_full((y_seg_idx.float(), flow_yx))
                    L_dsc = 0.5 * (dice_val_VOI(x_seg_w, y_seg_idx) + dice_val_VOI(y_seg_w, x_seg_idx))
                    L_dsc = L_dsc * args.w_dsc

                L_reg = 0.5 * (criterion_reg(flow_xy_h) + criterion_reg(flow_yx_h))
                L_reg = L_reg * args.w_reg

                L_icon = icon_loss(flow_xy, flow_yx) * W_icon

                x_cycle = reg_bilin_half((out_xy_h, flow_yx_h))
                y_cycle = reg_bilin_half((out_yx_h, flow_xy_h))
                L_cyc = ((x_cycle - x_half).abs().mean() + (y_cycle - y_half).abs().mean()) * W_cyc

                L_jac = 0.5 * (neg_jacobian_penalty(flow_xy_h) + neg_jacobian_penalty(flow_yx_h))
                L_jac = L_jac * W_jac

                loss = L_ncc + L_dsc + L_reg + L_icon + L_cyc + L_jac

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_all.update(float(loss.detach().item()), 1)
            loss_ncc_m.update(float(L_ncc.detach().item()), 1)
            loss_dsc_m.update(float(L_dsc.detach().item()), 1)
            loss_reg_m.update(float(L_reg.detach().item()), 1)
            loss_icon_m.update(float(L_icon.detach().item()), 1)
            loss_cyc_m.update(float(L_cyc.detach().item()), 1)
            loss_jac_m.update(float(L_jac.detach().item()), 1)

            iter_time_sum += (time.perf_counter() - t_it)

            if idx % 10 == 0:
                print(
                    f"Iter {idx:4d} / {len(train_loader):4d} | "
                    f"loss(avg)={loss_all.avg:.4f} | "
                    f"last NCC={L_ncc.item():.4f} DSC={L_dsc.item():.4f} REG={L_reg.item():.4f} | "
                    f"ICON={L_icon.item():.4f} CYC={L_cyc.item():.4f} JAC={L_jac.item():.4f} | "
                    f"lr={cur_lr:.1e}"
                )

        # validation
        model.eval()
        with torch.no_grad():
            val_dsc = validate_oasis(val_loader, model, forward_flow_fn=forward_flow_ctcf, device=device)

        is_best = val_dsc > best_dsc
        if is_best:
            best_dsc = val_dsc

        # checkpoints:
        #  - best/last in ckpt_dir (never pruned)
        #  - epochs in ckpt_epochs_dir (pruned by save_checkpoint)
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dsc": best_dsc,
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "config": dict(config),
            "args": vars(args),
        }

        torch.save(state, os.path.join(ckpt_dir, "last.pth"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best.pth"))

        perf = perf_epoch_end(t0_epoch, iters=len(train_loader), iter_time_sum=iter_time_sum)

        # stdout
        print(
            f"[epoch {epoch:03d}] "
            f"alpha_l1={alpha_l1:.3f} alpha_l3={alpha_l3:.3f} warm={warm:.3f} | "
            f"loss={loss_all.avg:.4f} ncc={loss_ncc_m.avg:.4f} dsc={loss_dsc_m.avg:.4f} reg={loss_reg_m.avg:.4f} "
            f"icon={loss_icon_m.avg:.4f} cyc={loss_cyc_m.avg:.4f} jac={loss_jac_m.avg:.4f} | "
            f"val_dice={val_dsc:.4f} best={best_dsc:.4f} | "
            f"lr={cur_lr:.6g} it_ms={perf.mean_iter_time_ms:.2f} peakGiB={(perf.peak_gpu_mem_gib or 0.0):.2f}"
        )

        # TensorBoard scalars
        writer.add_scalar("train/loss", loss_all.avg, epoch)
        writer.add_scalar("train/ncc", loss_ncc_m.avg, epoch)
        writer.add_scalar("train/dsc", loss_dsc_m.avg, epoch)
        writer.add_scalar("train/reg", loss_reg_m.avg, epoch)
        writer.add_scalar("train/icon", loss_icon_m.avg, epoch)
        writer.add_scalar("train/cyc", loss_cyc_m.avg, epoch)
        writer.add_scalar("train/jac", loss_jac_m.avg, epoch)
        writer.add_scalar("val/dice", float(val_dsc), epoch)
        writer.add_scalar("lr", float(cur_lr), epoch)
        writer.add_scalar("sched/alpha_l1", float(alpha_l1), epoch)
        writer.add_scalar("sched/alpha_l3", float(alpha_l3), epoch)
        writer.add_scalar("sched/warm", float(warm), epoch)
        writer.add_scalar("perf/mean_iter_ms", float(perf.mean_iter_time_ms), epoch)
        if perf.peak_gpu_mem_gib is not None:
            writer.add_scalar("perf/peak_mem_gib", float(perf.peak_gpu_mem_gib), epoch)

        # TensorBoard images (optional)
        if int(args.tb_images_every) > 0 and (epoch % int(args.tb_images_every) == 0):
            try:
                xb, yb, _, _ = val_set[0]
                xb = xb.unsqueeze(0).to(device).float()
                yb = yb.unsqueeze(0).to(device).float()
                with torch.no_grad():
                    def_full, flow_full, _ = model(xb, yb, return_all=True, alpha_l1=1.0, alpha_l3=1.0)

                fig_mov = comput_fig(xb)
                fig_fix = comput_fig(yb)
                fig_def = comput_fig(def_full)

                writer.add_figure("vis/mov", fig_mov, epoch)
                writer.add_figure("vis/fix", fig_fix, epoch)
                writer.add_figure("vis/def", fig_def, epoch)

                grid = mk_grid_img(flow_full)
                reg_full = register_model(tuple(xb.shape[-3:]), mode="bilinear").to(device)
                def_grid = reg_full((grid, flow_full))
                fig_grid = comput_fig(def_grid)
                writer.add_figure("vis/grid_def", fig_grid, epoch)

                fig_mov.savefig(os.path.join(vis_dir, f"e{epoch:03d}_mov.png"), bbox_inches="tight")
                fig_fix.savefig(os.path.join(vis_dir, f"e{epoch:03d}_fix.png"), bbox_inches="tight")
                fig_def.savefig(os.path.join(vis_dir, f"e{epoch:03d}_def.png"), bbox_inches="tight")
                fig_grid.savefig(os.path.join(vis_dir, f"e{epoch:03d}_grid.png"), bbox_inches="tight")
            except Exception as e:
                print(f"[warn] TB images failed @ epoch {epoch}: {e}")

    writer.close()


if __name__ == "__main__":
    main()