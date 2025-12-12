from torch.utils.tensorboard import SummaryWriter
import os, glob, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from experiments.OASIS import datasets
from models.TransMorph_DCA.model import CONFIGS as CONFIGS_TM
import models.TransMorph_DCA.model as TransMorph

from utils import (
    AverageMeter,
    register_model,
    NCC_vxm,
    DiceLoss,
    Grad3d,
    NumpyType,
    setup_device,
    save_checkpoint,
    comput_fig,
    mk_grid_img,
    dice_val_VOI,
    perf_epoch_end,
    perf_epoch_start,
    make_exp_dirs,
    attach_stdout_logger,
    load_checkpoint_if_exists,
    adjust_learning_rate_poly,
    validate_oasis
)

# ---------- Adapter for validate_oasis() ---------- #

def forward_flow_tm_dca(model, x, y):
    x_half = F.avg_pool3d(x, 2)
    y_half = F.avg_pool3d(y, 2)

    flow_half = model((x_half.half(), y_half.half()))
    flow_full = F.interpolate(
        flow_half.float(),
        scale_factor=2,
        mode="trilinear",
        align_corners=False
    ) * 2.0
    return flow_full


# ---------------------- CLI ---------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='Path to OASIS training .pkl files (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test .pkl files (e.g. .../Test)')
    p.add_argument('--exp', default='TM_DCA_v3', help='Experiment name (results/<exp>/, logs/<exp>/)')
    p.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    p.add_argument('--max_epoch', type=int, default=500, help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    p.add_argument('--cont', action='store_true', help='Resume from results/<exp>/last.pth.tar')
    p.add_argument('--gpu', type=int, default=0, help='GPU id to use')

    p.add_argument('--w_ncc', type=float, default=1.0)
    p.add_argument('--w_dsc', type=float, default=1.0)
    p.add_argument('--w_reg', type=float, default=1.0)

    p.add_argument('--time_steps', type=int, default=12, help='Cascade time steps (original default: 12)')
    p.add_argument('--unsup', action='store_true', help='If set, train without DSC loss (pure unsupervised).')
    return p.parse_args()


# ---------------------- main ---------------------- #

def main():
    args = parse_args()

    # ---------- device ----------

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
    time_steps = args.time_steps

    W_ncc = args.w_ncc
    W_dsc = args.w_dsc
    W_reg = args.w_reg
    unsup = args.unsup

    print(f'>>> Experiment: {args.exp}')
    print(f'    train_dir = {train_dir}')
    print(f'    val_dir   = {val_dir}')
    print(f'    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={args.cont}')
    print(f'    time_steps={time_steps}')
    print(f'    loss weights: NCC={W_ncc}, DSC={W_dsc}, REG={W_reg}')
    print(f'    UNSUPERVISED={"YES" if unsup else "NO"} (training)')

    D, H, W = 160, 192, 224
    full_size = (D, H, W)
    half_size = (D // 2, H // 2, W // 2)

    # ---------- model ----------

    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = half_size
    config.dwin_kernel_size = (7, 5, 3)
    config.window_size = (D // 32, H // 32, W // 32)

    model = TransMorph.TransMorphCascadeAd(config, time_steps).to(device)
    spatial_trans = TransMorph.SpatialTransformer(full_size).to(device)

    # ---------- optimizer + losses ----------

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)

    criterion_ncc = NCC_vxm()
    criterion_dsc = DiceLoss()
    criterion_reg = Grad3d(penalty='l2')

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # ---------- resume ----------

    epoch_start = 0
    best_dsc = 0.0
    if args.cont:
        ckpt_path = os.path.join(paths.exp_dir, 'last.pth.tar')
        ckpt = load_checkpoint_if_exists(ckpt_path, model, optimizer, map_location="cuda")
        if ckpt:
            epoch_start = ckpt.get('epoch', 0)
            best_dsc = ckpt.get('best_dsc', 0.0)
            print(f"Loaded last checkpoint: epoch_start={epoch_start}, best_dsc={best_dsc:.4f}")
        else:
            print("No last.pth.tar found, starting from scratch.")

    # ---------- datasets ----------

    train_tf = transforms.Compose([NumpyType((np.float32, np.int16))])
    val_tf   = transforms.Compose([NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_tf)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    print(f'>>> #train={len(train_loader.dataset)}, #val={len(val_loader.dataset)}')

    # -------------------- training loop -------------------- 

    for epoch in range(epoch_start, max_epoch):
        print(f'Training Starts (epoch {epoch})')

        t0 = perf_epoch_start()
        cur_lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, lr)

        loss_all = AverageMeter()
        idx = 0
        iter_time_sum = 0.0

        for batch in train_loader:
            idx += 1
            model.train()
            iter_t0 = time.perf_counter()
            batch = [t.to(device, non_blocking=True) for t in batch]
            x, y, x_seg_idx, y_seg_idx = batch  # x: moving, y: fixed

            with torch.no_grad():
                x = x.half()
                y = y.half()

                x_half = F.avg_pool3d(x, 2).half()
                y_half = F.avg_pool3d(y, 2).half()

                if not unsup:
                    x_seg_oh = F.one_hot(x_seg_idx.long(), 36).float().squeeze(1).permute(0, 4, 1, 2, 3).half()
                    y_seg_oh = F.one_hot(y_seg_idx.long(), 36).float().squeeze(1).permute(0, 4, 1, 2, 3).half()
                else:
                    x_seg_oh, y_seg_oh = None, None

            # ---------------- x -> y (step 1) ----------------

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                flow_half = model((x_half, y_half))  # half-res flow
                flow = F.interpolate(flow_half, scale_factor=2, mode='trilinear', align_corners=False) * 2.0
                out = spatial_trans(x, flow.half())

                # NCC stability: compute in float32
                with torch.amp.autocast("cuda", enabled=False):
                    L_ncc = criterion_ncc(out.float(), y.float()) * W_ncc

                if not unsup:
                    def_seg = spatial_trans(x_seg_oh, flow.half())
                    L_dsc = criterion_dsc(def_seg, y_seg_oh) * W_dsc
                else:
                    L_dsc = torch.tensor(0.0, device=device, dtype=torch.float32)

                L_reg = criterion_reg(flow.half(), y) * W_reg
                loss = L_ncc + L_dsc + L_reg

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[NON-FINITE LOSS x->y] loss={loss.item()} "
                    f"ncc={L_ncc.item()} dsc={L_dsc.item()} reg={L_reg.item()}"
                )

            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_all.update(loss.item(), y.numel())

            # ---------------- y -> x (step 2) ----------------

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                flow_half = model((y_half, x_half))
                flow = F.interpolate(flow_half, scale_factor=2, mode='trilinear', align_corners=False) * 2.0
                out = spatial_trans(y, flow.half())

                with torch.amp.autocast("cuda", enabled=False):
                    L_ncc = criterion_ncc(out.float(), x.float()) * W_ncc

                if not unsup:
                    def_seg = spatial_trans(y_seg_oh, flow.half())
                    L_dsc = criterion_dsc(def_seg, x_seg_oh) * W_dsc
                else:
                    L_dsc = torch.tensor(0.0, device=device, dtype=torch.float32)

                L_reg = criterion_reg(flow.half(), x) * W_reg
                loss = L_ncc + L_dsc + L_reg

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[NON-FINITE LOSS y->x] loss={loss.item()} "
                    f"ncc={L_ncc.item()} dsc={L_dsc.item()} reg={L_reg.item()}"
                )

            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_all.update(loss.item(), x.numel())

            iter_time_sum += (time.perf_counter() - iter_t0)

            if idx % 10 == 0:
                print(
                    f"Iter {idx:4d} / {len(train_loader):4d} | "
                    f"loss(avg)={loss_all.avg:.4f} | "
                    f"last NCC={L_ncc.item():.4f} DSC={L_dsc.item():.4f} REG={L_reg.item():.4f} | "
                    f"lr={cur_lr:.1e}"
                )

        writer.add_scalar('Loss/train_total', loss_all.avg, epoch)
        print(f'Epoch {epoch} loss {loss_all.avg:.4f}')

        # ---------------- Performance ----------------

        perf = perf_epoch_end(t0, iters=idx, iter_time_sum=iter_time_sum)
        if perf.peak_gpu_mem_gib is not None:
            print(f"[PERF] Epoch {epoch}: time={perf.epoch_time_sec:.1f}s, iter={perf.mean_iter_time_ms:.1f} ms/iter, peak={perf.peak_gpu_mem_gib:.2f} GiB")
            writer.add_scalar('perf/epoch_time_sec', perf.epoch_time_sec, epoch)
            writer.add_scalar('perf/iter_time_ms', perf.mean_iter_time_ms, epoch)
            writer.add_scalar('perf/peak_gpu_mem_GB', perf.peak_gpu_mem_gib, epoch)
        else:
            print(f"[PERF] Epoch {epoch}: time={perf.epoch_time_sec:.1f}s, iter={perf.mean_iter_time_ms:.1f} ms/iter")

        # -------------------- Validation -------------------- #

        val = validate_oasis(
            model=model,
            val_loader=val_loader,
            device=device,
            forward_flow_fn=lambda x, y: forward_flow_tm_dca(model, x, y),
            dice_fn=dice_val_VOI,
            register_model_cls=register_model,
            mk_grid_img_fn=mk_grid_img,
        )
        print(f"val DSC: {val.dsc:.4f} | fold%: {val.fold_percent:.2f}")
        writer.add_scalar("DSC/validate", val.dsc, epoch)
        writer.add_scalar("Metric/validate_fold_percent", val.fold_percent, epoch)

        # take tensors for visualisation
        def_out  = val.last_vis.get("def_seg", None)
        def_grid = val.last_vis.get("def_grid", None)
        x_vis    = val.last_vis.get("x_seg", None)
        y_vis    = val.last_vis.get("y_seg", None)

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
        
        plt.switch_backend("agg")
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