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

from TransMorph import utils, losses
from TransMorph.data import datasets, trans
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
import TransMorph.models.TransMorph as TransMorph


class Logger(object):
    """
    Simple stdout logger that mirrors everything printed to both console and a log file.
    """
    def __init__(self, save_dir: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        # single logfile inside given directory
        self.log = open(save_dir + "logfile.log", "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def comput_fig(img: torch.Tensor) -> plt.Figure:
    """
    Build a matplotlib figure with 16 axial slices from a 5D tensor [B, C, D, H, W].
    - Assumes B=1, C=1.
    - Uses slices z = 48..63.
    - Returns a matplotlib Figure object for TensorBoard logging.
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
    Polynomial learning-rate schedule:
        lr(epoch) = init_lr * (1 - epoch / max_epochs)^power
    This is the same schedule used in the original TransMorph code.
    Returns the updated learning rate.
    """
    new_lr = round(init_lr * np.power(1 - (epoch / max_epochs), power), 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def mk_grid_img(grid_step: int,
                line_thickness: int = 1,
                grid_sz=(160, 192, 224)) -> torch.Tensor:
    """
    Create a binary 3D grid image [1, 1, D, H, W] used for visualizing deformations.
    - grid_step: spacing between grid lines (in voxels).
    - line_thickness: thickness of each grid line (in voxels).
    - grid_sz: (D, H, W) size of the volume.
    The result is moved to CUDA, since it will be warped on GPU.
    """
    grid_img = np.zeros(grid_sz, dtype=np.float32)
    # horizontal lines along H
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1.0
    # vertical lines along W
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1.0
    grid_img = grid_img[None, None, ...]  # [1, 1, D, H, W]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state: dict,
                    save_dir: str = 'models/',
                    filename: str = 'checkpoint.pth.tar',
                    max_model_num: int = 8) -> None:
    """
    Save a checkpoint and keep only the last `max_model_num` files in the directory.
    - state: dictionary with model/optimizer state.
    - save_dir: directory where checkpoints are stored.
    - filename: filename for the current checkpoint.
    - max_model_num: how many recent checkpoints to retain.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))


def parse_args():
    """
    Parse command-line arguments for training on the OASIS dataset.
    Important flags:
      --train_dir : directory with *.pkl training volumes (OASIS "All")
      --val_dir   : directory with *.pkl validation volumes (OASIS "Test")
      --exp       : experiment name; controls 'experiments/<exp>/' and 'logs/<exp>/'
      --lr        : initial learning rate
      --max_epoch : number of training epochs
      --batch_size: batch size (TransMorph uses 1 by default)
      --cont      : continue training from the latest checkpoint in experiments/<exp>/
      --gpu       : GPU id to use
    """
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='Path to OASIS training .pkl files (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test .pkl files (e.g. .../Test)')
    p.add_argument('--exp', default='TransMorph_V2', help='Experiment name (used for experiments/ and logs/ sub-folders)')
    p.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-4)')
    p.add_argument('--max_epoch', type=int, default=500, help='Number of training epochs (default: 500)')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    p.add_argument('--cont', action='store_true', help='Resume training from latest checkpoint in experiments/<exp>/')
    p.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    return p.parse_args()


def main():
    args = parse_args()

    GPU_iden = args.gpu
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)

    # ---------- experiment directories ----------
    exp_root = args.exp.rstrip('/') + '/'
    exp_dir = os.path.join('experiments', exp_root)
    log_dir = os.path.join('logs', exp_root)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- logger + TensorBoard ----------
    # All print() calls will be duplicated to logs/<exp>/logfile.log
    sys.stdout = Logger(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    batch_size = args.batch_size
    # Ensure trailing separator for glob
    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir   = args.val_dir if args.val_dir.endswith(os.sep) else args.val_dir + os.sep

    weights = [1, 1, 1]  # [w_ncc, w_dsc, w_reg]
    lr = args.lr
    max_epoch = args.max_epoch
    epoch_start = 0
    cont_training = args.cont

    print(f'>>> Experiment: {args.exp}')
    print(f'    train_dir = {train_dir}')
    print(f'    val_dir   = {val_dir}')
    print(f'    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={cont_training}')
    print(f'    loss weights: NCC={weights[0]}, DSC={weights[1]}, REG={weights[2]}')

    # ---------- model definition ----------
    config = CONFIGS_TM['TransMorph-Large']
    model = TransMorph.TransMorph(config).cuda()

    # ---------- spatial transformers for evaluation/visualization ----------
    reg_model = utils.register_model(config.img_size, 'nearest').cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear').cuda()

    # ---------- optional resume from checkpoint ----------
    best_dsc = 0.0
    if cont_training:
        model_dir = exp_dir
        ckpts = natsorted(os.listdir(model_dir))
        if len(ckpts) == 0:
            print('No checkpoints found, starting from scratch.')
            updated_lr = lr
        else:
            last_ckpt = ckpts[-1]
            ckpt_path = os.path.join(model_dir, last_ckpt)
            ckpt = torch.load(ckpt_path)
            epoch_start = ckpt.get('epoch', 1)  # continue from this epoch index
            best_dsc = ckpt.get('best_dsc', 0.0)
            model.load_state_dict(ckpt['state_dict'])
            updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
            print(f'Model: {last_ckpt} loaded! best_dsc={best_dsc:.4f}, resume_epoch={epoch_start}')
    else:
        updated_lr = lr

    # ---------- datasets and dataloaders ----------
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    print(f'>>> #train={len(train_loader.dataset)}, #val={len(val_loader.dataset)}')

    # ---------- optimizer and loss functions ----------
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_dsc = losses.DiceLoss()
    criterion_reg = losses.Grad3d(penalty='l2')

    loss_all = utils.AverageMeter()

    # ---------- training loop ----------
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts (epoch {})'.format(epoch))
        
        # --- perf: start timing & reset peak memory ---
        epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            # Reset CUDA peak memory stats at the beginning of each epoch
            torch.cuda.reset_peak_memory_stats()

        loss_all.reset()
        idx = 0
        iter_time_sum = 0.0

        for data in train_loader:
            idx += 1
            model.train()

            # --- perf: iteration start time ---
            iter_start_time = time.perf_counter()
            
            # Polynomial LR schedule update
            cur_lr = adjust_learning_rate(optimizer, epoch, max_epoch, lr)

            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            # ===== Forward direction: x -> y =====
            # One-hot encode source labels (36 anatomical labels)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)         # [B, D, H, W, 36]
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3)    # [B, 36, D, H, W]

            # Concatenate source and target images along channel dimension
            x_in = torch.cat((x, y), dim=1)               # [B, 2, D, H, W]
            output, flow = model(x_in)                    # output: warped x, flow: deformation field

            # Warp each anatomical label channel with the same flow
            def_segs = []
            for i in range(36):
                def_seg = model.spatial_trans(x_seg_oh[:, i:i+1, ...].float(), flow.float())
                def_segs.append(def_seg)
            def_seg = torch.cat(def_segs, dim=1)          # [B, 36, D, H, W]

            # Compute image similarity, segmentation alignment and regularization
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_dsc = criterion_dsc(def_seg, y_seg.long()) * weights[1]
            loss_reg = criterion_reg(flow, y) * weights[2]
            loss = loss_ncc + loss_dsc + loss_reg
            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_seg_oh, x_in, def_segs, def_seg, loss

            # ===== Backward direction: y -> x =====
            y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=36)
            y_seg_oh = torch.squeeze(y_seg_oh, 1)         # [B, D, H, W, 36]
            y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3)    # [B, 36, D, H, W]

            y_in = torch.cat((y, x), dim=1)               # [B, 2, D, H, W]
            output, flow = model(y_in)

            def_segs = []
            for i in range(36):
                def_seg = model.spatial_trans(y_seg_oh[:, i:i+1, ...].float(), flow.float())
                def_segs.append(def_seg)
            def_seg = torch.cat(def_segs, dim=1)

            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_dsc = criterion_dsc(def_seg, x_seg.long()) * weights[1]
            loss_reg = criterion_reg(flow, x) * weights[2]
            loss = loss_ncc + loss_dsc + loss_reg
            loss_all.update(loss.item(), x.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del y_seg_oh, y_in, def_segs, def_seg

            # --- perf: accumulate iteration time ---
            iter_time = time.perf_counter() - iter_start_time
            iter_time_sum += iter_time

            if idx % 10 == 0:
                print('Iter {} of {} | loss {:.4f}, ImgSim {:.6f}, DSC {:.6f}, Reg {:.6f}, lr {:.1e}'.format(
                    idx, len(train_loader),
                    loss_all.avg,
                    loss_ncc.item(),
                    loss_dsc.item(),
                    loss_reg.item(),
                    cur_lr
                ))

        # Log average training loss per epoch
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        # --- perf: end of epoch (time + GPU memory) ---
        epoch_time = time.perf_counter() - epoch_start_time
        mean_iter_time = iter_time_sum / max(1, idx)  # seconds per iteration

        if torch.cuda.is_available():
            # Peak memory allocated on this device during the epoch (bytes -> GiB)
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            peak_mem_gib = peak_mem_bytes / (1024 ** 3)

            print(
                f"[PERF] Epoch {epoch}: "
                f"time={epoch_time:.1f}s, "
                f"iter={mean_iter_time * 1000:.1f} ms/iter, "
                f"peak GPU mem={peak_mem_gib:.2f} GiB"
            )

            writer.add_scalar('perf/epoch_time_sec', epoch_time, epoch)
            writer.add_scalar('perf/iter_time_ms', mean_iter_time * 1000.0, epoch)
            writer.add_scalar('perf/peak_gpu_mem_GB', peak_mem_gib, epoch)
        else:
            print(
                f"[PERF] Epoch {epoch}: "
                f"time={epoch_time:.1f}s, "
                f"iter={mean_iter_time * 1000:.1f} ms/iter"
            )

        # ---------- Validation ----------
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)                     # (warped, flow)
                def_out = reg_model([x_seg.float(), output[1].float()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].float()])

                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print('val DSC running avg: {:.4f}'.format(eval_dsc.avg))

        best_dsc = max(eval_dsc.avg, best_dsc)

        # Save checkpoint with current validation DSC in filename
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=exp_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)

        # Log sample figures to TensorBoard
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch); plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch); plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch); plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch); plt.close(pred_fig)

    writer.close()


if __name__ == '__main__':
    main()