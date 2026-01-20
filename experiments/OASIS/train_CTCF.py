from torch.utils.tensorboard import SummaryWriter
import os, glob, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
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
    DiceLoss,
    Grad3d,
    NumpyType,
    register_model,
    mk_grid_img,
    comput_fig,
    validate_oasis,
    icon_loss,
    neg_jacobian_penalty,
)


# -------------------- Validation adapter -------------------- #

@torch.no_grad()
def forward_flow_ctcf(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    model: torch.nn.Module
) -> torch.Tensor:
    use_amp = torch.cuda.is_available()
    with torch.amp.autocast("cuda", enabled=use_amp):
        _, flow_full = model(x, y)
    return flow_full


# ---------------------- CLI ---------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='Path to OASIS training .pkl files (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test .pkl files (e.g. /Test)')
    p.add_argument('--exp', default='CTCF_v2_CascadeA', help='Experiment name (results/<exp>/, logs/<exp>/)')
    p.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    p.add_argument('--max_epoch', type=int, default=500, help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    p.add_argument('--cont', action='store_true', help='Resume from results/<exp>/last.pth.tar')
    p.add_argument('--gpu', type=int, default=0, help='GPU id to use')

    # base losses
    p.add_argument('--w_ncc', type=float, default=1.0)
    p.add_argument('--w_dsc', type=float, default=1.0)
    p.add_argument('--w_reg', type=float, default=1.0)

    # CTCF regularizers
    p.add_argument('--w_icon', type=float, default=0.1)
    p.add_argument('--w_cyc', type=float, default=0.05)
    p.add_argument('--w_jac', type=float, default=0.01)

    # model
    p.add_argument('--time_steps', type=int, default=2, help='Level-2 integration steps (CTCF core on half-res).')
    p.add_argument('--unsup', action='store_true', help='Train without DSC loss (pure unsupervised).')

    # perf/memory
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='Enable AMP (recommended on GPU).')
    return p.parse_args()


# ---------------------- main ---------------------- #

def main():
    args = parse_args()

    dev = setup_device(args.gpu, seed=0, deterministic=False)
    device = dev.device

    # ---------- experiment dirs + logger ----------

    paths = make_exp_dirs(args.exp)
    attach_stdout_logger(paths.log_dir)
    writer = SummaryWriter(log_dir=paths.log_dir)

    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir   = args.val_dir   if args.val_dir.endswith(os.sep)   else args.val_dir   + os.sep

    lr = args.lr
    max_epoch = args.max_epoch
    batch_size = args.batch_size

    W_ncc  = args.w_ncc
    W_dsc  = args.w_dsc
    W_reg  = args.w_reg
    W_icon = args.w_icon
    W_cyc  = args.w_cyc
    W_jac  = args.w_jac

    time_steps = args.time_steps
    unsup = args.unsup

    print(f'>>> Experiment: {args.exp}')
    print(f'    train_dir = {train_dir}')
    print(f'    val_dir   = {val_dir}')
    print(f'    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={args.cont}')
    print(f'    time_steps={time_steps}')
    print(f'    loss weights: NCC={W_ncc}, DSC={W_dsc}, REG={W_reg}, ICON={W_icon}, CYC={W_cyc}, JAC={W_jac}')
    print(f'    UNSUPERVISED={"YES" if unsup else "NO"} (training)')

    # ---------- model config ----------

    config = CONFIGS_CTCF['CTCF-CascadeA']
    # config = CONFIGS_CTCF['CTCF-CascadeA-Debug']
    config.time_steps = int(time_steps)

    full_size = (160, 192, 224)
    half_size = (80, 96, 112)

    model = CTCF_CascadeA(config).to(device)

    # ---------- spatial transformers for visuals/seg warp ----------

    reg_nearest_full = register_model(full_size, 'nearest').to(device)
    reg_bilin_full   = register_model(full_size, 'bilinear').to(device)
    reg_bilin_half   = register_model(half_size, 'bilinear').to(device)

    # ---------- optimizer + losses ----------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=0.0, amsgrad=True
    )

    criterion_ncc = NCC_vxm()
    criterion_dsc = DiceLoss()
    criterion_reg = Grad3d(penalty='l2')

    use_amp = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ---------- resume ----------

    epoch_start = 0
    best_dsc = 0.0
    if args.cont:
        ckpt_path = os.path.join(paths.exp_dir, 'last.pth.tar')
        ckpt = load_checkpoint_if_exists(
            ckpt_path, model, optimizer,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        if ckpt:
            epoch_start = ckpt.get('epoch', 0)
            best_dsc = ckpt.get('best_dsc', 0.0)
            print(f"Loaded last checkpoint: epoch_start={epoch_start}, best_dsc={best_dsc:.4f}")
        else:
            print('No last.pth.tar found, starting from scratch.')

    # ---------- dataset ----------

    train_tf = transforms.Compose([NumpyType((np.float32, np.int16))])
    val_tf   = transforms.Compose([NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_tf)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    print(f'>>> #train={len(train_loader.dataset)}, #val={len(val_loader.dataset)}')

    # -------------------- training loop -------------------- #

    last_vis = {}

    for epoch in range(epoch_start, max_epoch):
        print(f'Training Starts (epoch {epoch})')

        t0 = perf_epoch_start()
        cur_lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, lr)

        loss_all = AverageMeter()
        loss_ncc_m = AverageMeter()
        loss_dsc_m = AverageMeter()
        loss_reg_m = AverageMeter()
        loss_icon_m = AverageMeter()
        loss_cyc_m = AverageMeter()
        loss_jac_m = AverageMeter()

        idx = 0
        iter_time_sum = 0.0

        for batch in train_loader:
            idx += 1
            model.train()
            iter_t0 = time.perf_counter()

            batch = [t.to(device, non_blocking=True) for t in batch]
            x, y, x_seg_idx, y_seg_idx = batch  # x: moving, y: fixed ; seg idx: [B,1,D,H,W]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                def_xy, flow_xy, aux_xy = model(x, y, return_all=True)
                def_yx, flow_yx, aux_yx = model(y, x, return_all=True)

                x_half = aux_xy["mov_half"]
                y_half = aux_xy["fix_half"]

                out_xy_h = aux_xy["mov_w_half_final"]
                out_yx_h = aux_yx["mov_w_half_final"]
                flow_xy_h = aux_xy["flow_half_final"]
                flow_yx_h = aux_yx["flow_half_final"]

                L_ncc = 0.5 * (criterion_ncc(out_xy_h, y_half) + criterion_ncc(out_yx_h, x_half))
                L_ncc = L_ncc * W_ncc

                if unsup:
                    L_dsc = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    x_seg_w = reg_nearest_full((x_seg_idx.float(), flow_xy))
                    y_seg_w = reg_nearest_full((y_seg_idx.float(), flow_yx))
                    L_dsc = 0.5 * (criterion_dsc(x_seg_w, y_seg_idx) + criterion_dsc(y_seg_w, x_seg_idx))
                    L_dsc = L_dsc * W_dsc

                L_reg = 0.5 * (criterion_reg(flow_xy) + criterion_reg(flow_yx))
                L_reg = L_reg * W_reg

                L_icon = icon_loss(flow_xy, flow_yx) * W_icon

                x_cycle = reg_bilin_half((out_xy_h, flow_yx_h))
                y_cycle = reg_bilin_half((out_yx_h, flow_xy_h))
                L_cyc = ((x_cycle - x_half).abs().mean() + (y_cycle - y_half).abs().mean()) * W_cyc

                L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx))
                L_jac = L_jac * W_jac

                loss = L_ncc + L_dsc + L_reg + L_icon + L_cyc + L_jac

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_all.update(loss.item(), 1)
            loss_ncc_m.update(float(L_ncc.detach().item()), 1)
            loss_dsc_m.update(float(L_dsc.detach().item()), 1)
            loss_reg_m.update(float(L_reg.detach().item()), 1)
            loss_icon_m.update(float(L_icon.detach().item()), 1)
            loss_cyc_m.update(float(L_cyc.detach().item()), 1)
            loss_jac_m.update(float(L_jac.detach().item()), 1)

            iter_time_sum += (time.perf_counter() - iter_t0)

            if idx % 50 == 0:
                with torch.no_grad():
                    grid = mk_grid_img(flow_xy, grid_step=8, line_thickness=1)

                    last_vis = {
                        "def_out": def_xy.detach().float().cpu(),
                        "def_grid": grid.detach().float().cpu(),
                        "x": x.detach().float().cpu(),
                        "y": y.detach().float().cpu(),
                    }

        # ---------- epoch end logs ----------

        perf = perf_epoch_end(t0, iters=idx, iter_time_sum=iter_time_sum)

        writer.add_scalar("train/loss", loss_all.avg, epoch)
        writer.add_scalar("train/loss_ncc", loss_ncc_m.avg, epoch)
        writer.add_scalar("train/loss_dsc", loss_dsc_m.avg, epoch)
        writer.add_scalar("train/loss_reg", loss_reg_m.avg, epoch)
        writer.add_scalar("train/loss_icon", loss_icon_m.avg, epoch)
        writer.add_scalar("train/loss_cyc", loss_cyc_m.avg, epoch)
        writer.add_scalar("train/loss_jac", loss_jac_m.avg, epoch)
        writer.add_scalar("train/lr", cur_lr, epoch)
        
        writer.add_scalar("train/iter_time_ms", perf.mean_iter_time_ms, epoch)
        writer.add_scalar("train/epoch_time_sec", perf.epoch_time_sec, epoch)
        if perf.peak_gpu_mem_gib is not None:
            writer.add_scalar("train/peak_gpu_mem_gib", perf.peak_gpu_mem_gib, epoch)

        peak_str = (
            f"{perf.peak_gpu_mem_gib:.2f}GiB"
            if perf.peak_gpu_mem_gib is not None else "NA"
        )

        print(
            f"[epoch {epoch}] "
            f"loss={loss_all.avg:.4f} "
            f"(ncc={loss_ncc_m.avg:.4f}, dsc={loss_dsc_m.avg:.4f}, "
            f"reg={loss_reg_m.avg:.4f}, icon={loss_icon_m.avg:.4f}, "
            f"cyc={loss_cyc_m.avg:.4f}, jac={loss_jac_m.avg:.4f}) "
            f"lr={cur_lr:.2e} "
            f"time={perf.epoch_time_sec:.1f}s "
            f"iter={perf.mean_iter_time_ms:.1f}ms "
            f"peak_mem={peak_str}"
        )

        # ---------- validation (OASIS Dice/VOI metrics) ----------

        def forward_flow(x, y):
            return forward_flow_ctcf(x, y, model=model)

        model.eval()

        val = validate_oasis(
            model=model,
            val_loader=val_loader,
            device=device,
            forward_flow_fn=forward_flow,
            dice_fn=dice_val_VOI,
            register_model_cls=register_model,
            mk_grid_img_fn=mk_grid_img,
            grid_step=8,
            line_thickness=1,
        )

        print(f"[epoch {epoch}] val_dsc={val.dsc:.4f} fold%={val.fold_percent:.2f}")

        # -------------------- Checkpoints (BEST + LAST) -------------------- #

        if val.dsc >= best_dsc:
            best_dsc = val.dsc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_dsc": best_dsc,
                    "optimizer": optimizer.state_dict(),
                },
                save_dir=paths.exp_dir,
                filename="best.pth.tar",
            )
            print(f"Saved new BEST checkpoint (DSC={best_dsc:.4f})")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_dsc": best_dsc,
                "optimizer": optimizer.state_dict(),
            },
            save_dir=paths.exp_dir,
            filename="last.pth.tar",
        )

        # -------------------- Visuals -------------------- #

        import matplotlib.pyplot as plt
        plt.switch_backend("agg")
        def_out = last_vis.get("def_out", None)
        def_grid = last_vis.get("def_grid", None)
        x_vis = last_vis.get("x", None)
        y_vis = last_vis.get("y", None)

        if def_out is not None and def_grid is not None and x_vis is not None and y_vis is not None:
            pred_fig = comput_fig(def_out)
            grid_fig = comput_fig(def_grid)
            x_fig = comput_fig(x_vis)
            tar_fig = comput_fig(y_vis)

            writer.add_figure("Grid", grid_fig, epoch); plt.close(grid_fig)
            writer.add_figure("input", x_fig, epoch); plt.close(x_fig)
            writer.add_figure("ground truth", tar_fig, epoch); plt.close(tar_fig)
            writer.add_figure("prediction", pred_fig, epoch); plt.close(pred_fig)

    writer.close()


if __name__ == '__main__':
    main()