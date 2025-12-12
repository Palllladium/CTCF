import warnings
warnings.filterwarnings(
    "error",
    message=r".*grid_sample.*align_corners.*",
)

from torch.utils.tensorboard import SummaryWriter
import os, glob, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
import time

from utils import ctcf_losses, field, utils, trans
import datasets

from models.CTCF.model import CONFIGS as CONFIGS_CTCF
from models.CTCF.model import CTCF_DCA_SR_Cascade
from utils import losses


# ---------------------- logger ---------------------- #

class Logger(object):
    """
    Mirrors stdout to a log file.
    """
    def __init__(self, save_dir: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.log = open(save_dir + "logfile.log", "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ---------------------- helpers ---------------------- #

def comput_fig(img: torch.Tensor) -> plt.Figure:
    """
    Visualize 16 axial slices from [B,1,D,H,W] (z=48..63).
    """
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer: optim.Optimizer,
                         epoch: int,
                         max_epochs: int,
                         init_lr: float,
                         power: float = 0.9) -> float:
    """
    Polynomial LR schedule:
        lr(epoch) = init_lr * (1 - epoch / max_epochs)^power
    """
    new_lr = round(init_lr * np.power(1 - (epoch / max_epochs), power), 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def mk_grid_img(grid_step: int,
                line_thickness: int = 1,
                grid_sz=(160, 192, 224)) -> torch.Tensor:
    """
    3D binary grid [1,1,D,H,W] for deformation visualization.
    """
    grid_img = np.zeros(grid_sz, dtype=np.float32)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1.0
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1.0
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state: dict,
                    save_dir: str,
                    filename: str,
                    max_model_num: int = 8) -> None:
    """
    Save checkpoint and keep only last N files.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))


# ---------------------- CLI ---------------------- #

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--train_dir', required=True, help='Path to OASIS training *.pkl (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test *.pkl (e.g. .../Test)')
    p.add_argument('--exp', default='CTCF_DCA_SR_Cascade', help='Experiment name for results/ and logs/')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_epoch', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--cont', action='store_true')
    p.add_argument('--gpu', type=int, default=0)

    # config key in CONFIGS
    p.add_argument('--config', type=str, default='CTCF-DCA-SR',
                   choices=['CTCF-DCA-SR', 'CTCF-DCA-SR-Debug'],
                   help='Config key in TransMorph.models.CTCF_DCA_SR.configs.CONFIGS')

    # cascade steps in the model
    p.add_argument('--time_steps', type=int, default=4, help='Cascade time steps')

    # loss weights (UNSUPERVISED objective)
    p.add_argument('--w_ncc', type=float, default=1.0)
    p.add_argument('--w_reg', type=float, default=1.0)
    p.add_argument('--w_icon', type=float, default=0.1)
    p.add_argument('--w_cyc', type=float, default=0.1)
    p.add_argument('--w_jac', type=float, default=0.02)

    return p.parse_args()


# ---------------------- main ---------------------- #

def main():
    args = parse_args()

    # ---------- GPU info (same style as your trainers) ----------
    GPU_iden = args.gpu
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('Is GPU available? ' + str(GPU_avai))

    torch.manual_seed(0)

    # ---------- experiment paths ----------
    exp_root = args.exp.rstrip('/') + '/'
    exp_dir = os.path.join('results', exp_root)
    log_dir = os.path.join('logs', exp_root)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- logger + TensorBoard ----------
    sys.stdout = Logger(log_dir + os.sep if not log_dir.endswith(os.sep) else log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir   = args.val_dir   if args.val_dir.endswith(os.sep)   else args.val_dir   + os.sep

    batch_size = args.batch_size
    lr = args.lr
    max_epoch = args.max_epoch
    epoch_start = 0

    W_ncc  = args.w_ncc
    W_reg  = args.w_reg
    W_icon = args.w_icon
    W_cyc  = args.w_cyc
    W_jac  = args.w_jac

    print(f'>>> Experiment: {args.exp}')
    print(f'    train_dir = {train_dir}')
    print(f'    val_dir   = {val_dir}')
    print(f'    cfg       = {args.config}')
    print(f'    time_steps= {args.time_steps}')
    print(f'    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={args.cont}')
    print(f'    weights: NCC={W_ncc}, REG={W_reg}, ICON={W_icon}, CYC={W_cyc}, JAC={W_jac}')
    print('    NOTE: Training objective is UNSUPERVISED. Dice is logged as metric only.')

    # ---------- data ----------
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed   = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    print(f'>>> #train={len(train_loader.dataset)}, #val={len(val_loader.dataset)}')

    # ---------- model + config ----------
    config = CONFIGS_CTCF[args.config]
    model = CTCF_DCA_SR_Cascade(config, time_steps=args.time_steps).cuda()

    H, W, D = 160, 192, 224

    reg_model_nearest = utils.register_model((H, W, D), 'nearest').cuda()
    reg_model_bilin   = utils.register_model((H, W, D), 'bilinear').cuda()

    # ---------- optimizer + losses ----------
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')

    scaler = torch.amp.GradScaler("cuda")

    # ---------- resume ----------
    best_dsc = 0.0
    if args.cont:
        ckpts = natsorted(os.listdir(exp_dir))
        if len(ckpts) == 0:
            print('No checkpoints found, starting from scratch.')
        else:
            last_ckpt = ckpts[-1]
            ckpt_path = os.path.join(exp_dir, last_ckpt)
            ckpt = torch.load(ckpt_path, map_location='cuda')
            epoch_start = ckpt.get('epoch', 1)
            best_dsc = ckpt.get('best_dsc', 0.0)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f'Model: {last_ckpt} loaded! best_dsc={best_dsc:.4f}, resume_epoch={epoch_start}')

    # -------------------- training loop -------------------- #
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts (epoch {})'.format(epoch))

        epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        loss_meter = utils.AverageMeter()
        ncc_meter  = utils.AverageMeter()
        reg_meter  = utils.AverageMeter()
        icon_meter = utils.AverageMeter()
        cyc_meter  = utils.AverageMeter()
        jac_meter  = utils.AverageMeter()
        dice_meter = utils.AverageMeter()
        fold_meter = utils.AverageMeter()

        idx = 0
        iter_time_sum = 0.0

        for data in train_loader:
            idx += 1
            model.train()

            iter_start_time = time.perf_counter()
            cur_lr = adjust_learning_rate(optimizer, epoch, max_epoch, lr)

            data = [t.cuda(non_blocking=True) for t in data]
            x, y, x_seg, y_seg = data

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                # ---------- forward: x->y ----------
                x_in = torch.cat((x, y), dim=1)     # [B,2,D,H,W]
                out_xy, flow_xy = model(x_in)        # out_xy: warped x->y (full-res), flow_xy: full-res

                # ---------- forward: y->x ----------
                y_in = torch.cat((y, x), dim=1)
                out_yx, flow_yx = model(y_in)

                # NCC (float32 for stability)
                with torch.amp.autocast("cuda", enabled=False):
                    L_ncc_xy = criterion_ncc(out_xy.float(), y.float())
                    L_ncc_yx = criterion_ncc(out_yx.float(), x.float())
                L_ncc = 0.5 * (L_ncc_xy + L_ncc_yx)

                # Grad regularization
                L_reg_xy = criterion_reg(flow_xy, y)
                L_reg_yx = criterion_reg(flow_yx, x)
                L_reg = 0.5 * (L_reg_xy + L_reg_yx)

                # ICON
                L_icon = ctcf_losses.icon_loss(flow_xy, flow_yx)
                # Cycle (image space)
                L_cyc = ctcf_losses.cycle_image_loss(model, x, y, out_xy, out_yx, flow_xy, flow_yx)
                # Jacobian folding penalty (loss)
                L_jac = field.neg_jacobian_penalty(flow_xy) + field.neg_jacobian_penalty(flow_yx)

                # Total UNSUP loss
                loss = (W_ncc * L_ncc +
                        W_reg * L_reg +
                        W_icon * L_icon +
                        W_cyc * L_cyc +
                        W_jac * L_jac)

            if not torch.isfinite(loss):
                print("===== NON-FINITE LOSS DETECTED (CTCF_DCA_SR_Cascade) =====")
                print(f"loss={loss.item()} NCC={L_ncc.item()} REG={L_reg.item()} ICON={L_icon.item()} CYC={L_cyc.item()} JAC={L_jac.item()}")
                raise RuntimeError("Non-finite loss detected.")

            # backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # --- metrics for logging (Dice + fold%) ---
            # warp segs with nearest sampler using external register_model (as in train_CTCF_v2 validation)
            with torch.no_grad():
                def_xy_seg = reg_model_nearest([x_seg.float(), flow_xy.float()])
                def_yx_seg = reg_model_nearest([y_seg.float(), flow_yx.float()])

                dsc_xy = utils.dice_val_VOI(def_xy_seg.long(), y_seg.long())
                dsc_yx = utils.dice_val_VOI(def_yx_seg.long(), x_seg.long())
                dsc = 0.5 * (dsc_xy + dsc_yx)

                fold_xy = ctcf_losses.percent_nonpositive_jacobian(flow_xy.float())
                fold_yx = ctcf_losses.percent_nonpositive_jacobian(flow_yx.float())
                fold = 0.5 * (fold_xy + fold_yx)

            # meters
            loss_meter.update(loss.item(), x.numel())
            ncc_meter.update(L_ncc.item(), x.numel())
            reg_meter.update(L_reg.item(), x.numel())
            icon_meter.update(L_icon.item(), x.numel())
            cyc_meter.update(L_cyc.item(), x.numel())
            jac_meter.update(L_jac.item(), x.numel())
            dice_meter.update(dsc.item(), x.size(0))
            fold_meter.update(fold.item(), x.size(0))

            iter_time = time.perf_counter() - iter_start_time
            iter_time_sum += iter_time

            if idx % 1 == 0:
                print(
                    f"Epoch {epoch:03d} | Iter {idx:04d} / {len(train_loader):04d} | "
                    f"loss={loss_meter.avg:.4f} "
                    f"(NCC={ncc_meter.avg:.4f}, REG={reg_meter.avg:.4f}, "
                    f"ICON={icon_meter.avg:.4f}, CYC={cyc_meter.avg:.4f}, JAC={jac_meter.avg:.4f}) | "
                    f"Dice={dice_meter.avg:.4f} | %J<=0={fold_meter.avg:.2f} | lr={cur_lr:.1e}"
                )

        # ---------- epoch scalars ----------
        writer.add_scalar('Loss/train_total', loss_meter.avg, epoch)
        writer.add_scalar('Loss/train_ncc', ncc_meter.avg, epoch)
        writer.add_scalar('Loss/train_reg', reg_meter.avg, epoch)
        writer.add_scalar('Loss/train_icon', icon_meter.avg, epoch)
        writer.add_scalar('Loss/train_cycle', cyc_meter.avg, epoch)
        writer.add_scalar('Loss/train_jac', jac_meter.avg, epoch)
        writer.add_scalar('Metric/train_dice', dice_meter.avg, epoch)
        writer.add_scalar('Metric/train_fold_percent', fold_meter.avg, epoch)

        print('Epoch {} training loss {:.4f}'.format(epoch, loss_meter.avg))

        # ---------- perf ----------
        epoch_time = time.perf_counter() - epoch_start_time
        mean_iter_time = iter_time_sum / max(1, idx)

        if torch.cuda.is_available():
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            peak_mem_gib = peak_mem_bytes / (1024 ** 3)
            print(f"[PERF] Epoch {epoch}: time={epoch_time:.1f}s, iter={mean_iter_time*1000:.1f} ms/iter, peak GPU mem={peak_mem_gib:.2f} GiB")
            writer.add_scalar('perf/epoch_time_sec', epoch_time, epoch)
            writer.add_scalar('perf/iter_time_ms', mean_iter_time * 1000.0, epoch)
            writer.add_scalar('perf/peak_gpu_mem_GB', peak_mem_gib, epoch)
        else:
            print(f"[PERF] Epoch {epoch}: time={epoch_time:.1f}s, iter={mean_iter_time*1000:.1f} ms/iter")

        # -------------------- validation -------------------- #
        eval_dsc = utils.AverageMeter()
        eval_fold = utils.AverageMeter()

        # for visualization (keep last batch of validation)
        def_out = None
        def_grid = None
        x_vis = None
        y_vis = None

        with torch.no_grad():
            model.eval()
            for data in val_loader:
                data = [t.cuda(non_blocking=True) for t in data]
                x, y, x_seg, y_seg = data

                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, (D, H, W))

                out, flow = model(x_in)
                def_out = reg_model_nearest([x_seg.float(), flow.float()])
                def_grid = reg_model_bilin([grid_img.float(), flow.float()])

                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))

                fold = ctcf_losses.percent_nonpositive_jacobian(flow.float())
                eval_fold.update(fold.item(), x.size(0))

                x_vis = x_seg
                y_vis = y_seg

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        writer.add_scalar('Metric/validate_fold_percent', eval_fold.avg, epoch)
        print('Validation DSC avg: {:.4f} | %J<=0 avg: {:.2f}'.format(eval_dsc.avg, eval_fold.avg))

        # ---------- checkpoints ----------
        if eval_dsc.avg >= best_dsc:
            best_dsc = eval_dsc.avg
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                },
                save_dir=exp_dir,
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
            save_dir=exp_dir,
            filename='last.pth.tar'
        )

        # ---------- TensorBoard figures ----------
        plt.switch_backend('agg')
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