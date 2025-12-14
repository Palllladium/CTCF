from torch.utils.tensorboard import SummaryWriter
import os, glob, argparse, time
import numpy as np
import torch
import contextlib
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from experiments.OASIS import datasets
from models.CTCF.model import CONFIGS as CONFIGS_CTCF
import models.CTCF.model as CTCF

from utils import (
    AverageMeter,
    setup_device,
    make_exp_dirs,
    attach_stdout_logger,
    save_checkpoint,
    load_checkpoint_if_exists,
    perf_epoch_start,
    perf_epoch_end,
    adjust_learning_rate_poly,
    NCC_vxm,
    DiceLoss,
    Grad3d,
    NumpyType,
    dice_val_VOI,
    register_model,
    mk_grid_img,
    comput_fig,
    validate_oasis,
    icon_loss,
    cycle_image_loss,
    neg_jacobian_penalty,
)


# -------------------- Validation adapter --------------------

@torch.no_grad()
def forward_flow_ctcf(model, x, y):
    """
    Expected:
      x,y: [B,1,D,H,W]
      model((x,y)) -> (out, flow) OR flow
    """
    use_amp = torch.cuda.is_available()
    with torch.amp.autocast("cuda", enabled=use_amp):
        _, flow = model((x, y))
    return flow

# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='Path to OASIS training .pkl files (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test .pkl files (e.g. .../Test)')
    p.add_argument('--exp', default='CTCF_v1', help='Experiment name (results/<exp>/, logs/<exp>/)')
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
    p.add_argument('--time_steps', type=int, default=12, help='Cascade steps (keep same as TM-DCA baseline unless testing)')
    p.add_argument('--unsup', action='store_true', help='Train without DSC loss (pure unsupervised).')

    # perf/memory
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='Enable AMP (recommended on GPU).')
    return p.parse_args()


# ---------------------- main ----------------------

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

    W_ncc = args.w_ncc
    W_dsc = args.w_dsc
    W_reg = args.w_reg
    W_icon = args.w_icon
    W_cyc = args.w_cyc
    W_jac = args.w_jac

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

    config = CONFIGS_CTCF['CTCF-DCA-SR']
    full_size = tuple(config.img_size)  # (D,H,W)
    model = CTCF.CTCF_DCA_SR(config, time_steps).to(device)

    # ---------- spatial transformers for visuals/seg warp ----------
    # For seg warp in training/validation (nearest) and grid (bilinear)

    reg_nearest = register_model(full_size, 'nearest').to(device)
    reg_bilin   = register_model(full_size, 'bilinear').to(device)

    # ---------- optimizer + losses ----------

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)

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
        ckpt = load_checkpoint_if_exists(ckpt_path, model, optimizer, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if ckpt:
            epoch_start = ckpt.get('epoch', 0)
            best_dsc = ckpt.get('best_dsc', 0.0)
            print(f"Loaded last checkpoint: epoch_start={epoch_start}, best_dsc={best_dsc:.4f}")
        else:
            print('No last.pth.tar found, starting from scratch.')

    # ---------- datasets ----------

    train_tf = transforms.Compose([NumpyType((np.float32, np.int16))])
    val_tf   = transforms.Compose([NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_tf)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

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
            x, y, x_seg_idx, y_seg_idx = batch  # x: moving, y: fixed ; seg are label indices [B,1,D,H,W]

            if not unsup:
                with torch.no_grad():
                    x_seg_oh = F.one_hot(x_seg_idx.long(), 36).float().squeeze(1).permute(0, 4, 1, 2, 3)
                    y_seg_oh = F.one_hot(y_seg_idx.long(), 36).float().squeeze(1).permute(0, 4, 1, 2, 3)
            else:
                x_seg_oh, y_seg_oh = None, None

            # ---------------- bidirectional step (x->y and y->x) ----------------

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()

            with autocast_ctx:
                out_xy, flow_xy = model((x, y))
                out_yx, flow_yx = model((y, x))

                # NCC in float32 for stability
                with torch.amp.autocast('cuda', enabled=False):
                    L_ncc = (criterion_ncc(out_xy.float(), y.float()) + criterion_ncc(out_yx.float(), x.float())) * 0.5
                    L_ncc = L_ncc * W_ncc

                # DSC (optional)
                if not unsup:
                    if hasattr(model, 'spatial_trans_full'):
                        def_xseg = model.spatial_trans_full(x_seg_oh, flow_xy)
                        def_yseg = model.spatial_trans_full(y_seg_oh, flow_yx)
                    else:
                        def_xseg = reg_bilin([x_seg_oh, flow_xy])
                        def_yseg = reg_bilin([y_seg_oh, flow_yx])
                    L_dsc = (criterion_dsc(def_xseg, y_seg_oh) + criterion_dsc(def_yseg, x_seg_oh)) * 0.5
                    L_dsc = L_dsc * W_dsc
                else:
                    L_dsc = torch.tensor(0.0, device=device, dtype=torch.float32)

                L_reg = (criterion_reg(flow_xy) + criterion_reg(flow_yx)) * 0.5
                L_reg = L_reg * W_reg
                L_icon = icon_loss(flow_xy, flow_yx) * W_icon
                L_cyc = cycle_image_loss(model, x, y, out_xy, out_yx, flow_xy, flow_yx) * W_cyc
                L_jac = (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * 0.5
                L_jac = L_jac * W_jac

                loss = L_ncc + L_dsc + L_reg + L_icon + L_cyc + L_jac

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[NON-FINITE LOSS] loss={loss.item()} "
                    f"ncc={float(L_ncc):.4f} dsc={float(L_dsc):.4f} reg={float(L_reg):.4f} "
                    f"icon={float(L_icon):.4f} cyc={float(L_cyc):.4f} jac={float(L_jac):.4f}"
                )

            if scaler is not None:
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
                    f"Iter {idx:4d}/{len(train_loader):4d} | "
                    f"loss(avg)={loss_all.avg:.4f} | "
                    f"NCC={float(L_ncc):.4f} DSC={float(L_dsc):.4f} REG={float(L_reg):.4f} "
                    f"ICON={float(L_icon):.4f} CYC={float(L_cyc):.4f} JAC={float(L_jac):.4f} | "
                    f"lr={cur_lr:.1e}"
                )

        writer.add_scalar('Loss/train_total', loss_all.avg, epoch)
        writer.add_scalar('Loss/train_ncc', float(L_ncc), epoch)
        writer.add_scalar('Loss/train_dsc', float(L_dsc), epoch)
        writer.add_scalar('Loss/train_reg', float(L_reg), epoch)
        writer.add_scalar('Loss/train_icon', float(L_icon), epoch)
        writer.add_scalar('Loss/train_cyc', float(L_cyc), epoch)
        writer.add_scalar('Loss/train_jac', float(L_jac), epoch)

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

        # -------------------- Validation --------------------

        # OOM on 32 GB config:
        torch.cuda.empty_cache()

        val = validate_oasis(
            model=model,
            val_loader=val_loader,
            device=device,
            forward_flow_fn=lambda a, b: forward_flow_ctcf(model, a, b),
            dice_fn=dice_val_VOI,
            register_model_cls=register_model,
            mk_grid_img_fn=mk_grid_img,
        )

        print(f"val DSC: {val.dsc:.4f} | fold%: {val.fold_percent:.2f}")
        writer.add_scalar('DSC/validate', val.dsc, epoch)
        writer.add_scalar('Metric/validate_fold_percent', val.fold_percent, epoch)

        # OOM on 32 GB config:
        del flow_xy, flow_yx
        torch.cuda.empty_cache()

        # -------------------- Checkpoints (BEST + LAST) --------------------

        if val.dsc >= best_dsc:
            best_dsc = val.dsc
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                },
                save_dir=paths.exp_dir,
                filename='best.pth.tar'
            )
            print(f"Saved new BEST checkpoint (DSC={best_dsc:.4f})")

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            },
            save_dir=paths.exp_dir,
            filename='last.pth.tar'
        )

        # -------------------- Visuals --------------------
        
        plt.switch_backend('agg')
        def_out  = val.last_vis.get('def_seg', None)
        def_grid = val.last_vis.get('def_grid', None)
        x_vis    = val.last_vis.get('x_seg', None)
        y_vis    = val.last_vis.get('y_seg', None)

        if def_out is not None and def_grid is not None and x_vis is not None and y_vis is not None:
            pred_fig = comput_fig(def_out)
            grid_fig = comput_fig(def_grid)
            x_fig = comput_fig(x_vis)
            tar_fig = comput_fig(y_vis)

            writer.add_figure('Grid', grid_fig, epoch); plt.close(grid_fig)
            writer.add_figure('input', x_fig, epoch); plt.close(x_fig)
            writer.add_figure('ground truth', tar_fig, epoch); plt.close(tar_fig)
            writer.add_figure('prediction', pred_fig, epoch); plt.close(pred_fig)

    writer.close()


if __name__ == '__main__':
    main()