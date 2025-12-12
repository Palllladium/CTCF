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
from natsort import natsorted
import time
import random

from utils import utils, trans
from experiments.OASIS import datasets
from models.UTSRMorph.model import CONFIGS as CONFIGS_UM, UTSRMorph
from utils import losses


# ---------------------- Logger & helpers ---------------------- #

class Logger(object):
    """
    Simple stdout logger that mirrors everything printed to both console and a log file.
    All print(...) calls go to terminal AND to logs/<exp>/logfile.log.
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


def adjust_learning_rate(optimizer: optim.Optimizer,
                         epoch: int,
                         max_epochs: int,
                         init_lr: float,
                         power: float = 0.9) -> float:
    """
    Polynomial LR schedule (same as original UTSRMorph and TransMorph trainers):
        lr(epoch) = init_lr * (1 - epoch / max_epochs)^power
    """
    new_lr = round(init_lr * np.power(1 - (epoch / max_epochs), power), 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def save_checkpoint(state, save_dir, filename='model.pth', max_model_num=50):
    """
    Save a checkpoint and keep only the latest N models in the directory.
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
    """
    CLI for UTSRMorph_v2 training on OASIS (unsupervised).

    Important flags:
      --train_dir : directory with *.pkl training volumes (OASIS "All")
      --val_dir   : directory with *.pkl validation volumes (OASIS "Test")
      --exp       : experiment name; controls 'results/<exp>/' and 'logs/<exp>/'
      --lr        : initial learning rate
      --max_epoch : number of training epochs
      --batch_size: batch size (UTSRMorph uses 1 by default)
      --cont      : continue training from the latest checkpoint in results/<exp>/
      --gpu       : GPU id to use
    """
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='Path to OASIS training .pkl files (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test .pkl files (e.g. .../Test)')
    p.add_argument('--exp', default='UTSRMorph_v2', help='Experiment name (used for results/ and logs/ sub-folders)')
    p.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-4)')
    p.add_argument('--max_epoch', type=int, default=500, help='Number of training epochs (default: 500)')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    p.add_argument('--cont', action='store_true', help='Resume training from latest checkpoint in results/<exp>/')
    p.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    return p.parse_args()


# ---------------------- main ---------------------- #

def main():
    args = parse_args()

    # ---------- GPU setup ----------
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
    exp_dir = os.path.join('results', exp_root)
    log_dir = os.path.join('logs', exp_root)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- logger + TensorBoard ----------
    sys.stdout = Logger(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    batch_size = args.batch_size
    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir   = args.val_dir if args.val_dir.endswith(os.sep) else args.val_dir + os.sep

    weights = [1.0, 1.0]
    lr = args.lr
    max_epoch = args.max_epoch
    epoch_start = 0
    cont_training = args.cont

    print(f'>>> Experiment: {args.exp}')
    print(f'    train_dir = {train_dir}')
    print(f'    val_dir   = {val_dir}')
    print(f'    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={cont_training}')
    print(f'    loss weights: NCC={weights[0]}, REG={weights[1]}')

    # ---------- model ----------
    config = CONFIGS_UM['UTSRMorph-Large']
    model = UTSRMorph(config).cuda()

    # ---------- spatial transformer for evaluation ----------
    reg_model = utils.register_model(config.img_size, 'nearest').cuda()

    # ---------- optional resume ----------
    best_dsc = 0.0
    if cont_training:
        ckpts = natsorted(os.listdir(exp_dir))
        if len(ckpts) == 0:
            print('No checkpoints found, starting from scratch.')
            updated_lr = lr
        else:
            last_ckpt = ckpts[-1]
            ckpt_path = os.path.join(exp_dir, last_ckpt)
            ckpt = torch.load(ckpt_path)
            epoch_start = ckpt.get('epoch', 1)
            best_dsc = ckpt.get('best_dsc', 0.0)
            model.load_state_dict(ckpt['state_dict'])
            updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
            print(f'Model: {last_ckpt} loaded! best_dsc={best_dsc:.4f}, resume_epoch={epoch_start}')
    else:
        updated_lr = lr

    # ---------- datasets ----------
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed   = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # ---------- losses & optimizer ----------
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=1e-5)
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')

    # ---------- training loop ----------
    for epoch in range(epoch_start, max_epoch):
        epoch_start_time = time.perf_counter()
        iter_time_sum = 0.0

        loss_all = utils.AverageMeter()
        model.train()

        for iter_idx, data in enumerate(train_loader):
            iter_start_time = time.perf_counter()

            cur_lr = adjust_learning_rate(optimizer, epoch, max_epoch, lr)

            data = [t.cuda(non_blocking=True) for t in data]
            x = data[0]   # first image
            y = data[1]   # second image
            # segmentations are available but not used in unsupervised loss
            # x_seg = data[2]
            # y_seg = data[3]

            optimizer.zero_grad(set_to_none=True)

            # Random direction (x->y or y->x) for unsupervised registration
            if random.randint(0, 1) == 0:
                src = x
                tgt = y
            else:
                src = y
                tgt = x

            # Concatenate moving and fixed as 2-channel input
            input_pair = torch.cat((src, tgt), dim=1)  # [B,2,D,H,W]
            output, flow = model(input_pair)

            sim_loss = criterion_ncc(output, tgt)
            reg_loss = criterion_reg(flow, tgt)

            loss = weights[0] * sim_loss + weights[1] * reg_loss

            if not torch.isfinite(loss):
                print("===== NON-FINITE LOSS DETECTED =====")
                print(f"  sim_loss={sim_loss.item():.4e}, reg_loss={reg_loss.item():.4e}")
                print(f"  flow stats: min={flow.min().item():.4e}, max={flow.max().item():.4e}")
                raise RuntimeError("Non-finite loss in UTSRMorph_v2 training.")

            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), src.numel())

            iter_time = time.perf_counter() - iter_start_time
            iter_time_sum += iter_time

            if iter_idx % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Iter {iter_idx:04d} / {len(train_loader):04d} | "
                    f"loss={loss_all.avg:.4f} "
                    f"(NCC={sim_loss.item():.4f}, REG={reg_loss.item():.4f}) | "
                    f"lr={cur_lr:.1e}"
                )

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} training loss {:.4f}'.format(epoch, loss_all.avg))

        # --- perf: end of epoch (time + GPU memory) ---
        epoch_time = time.perf_counter() - epoch_start_time
        mean_iter_time = iter_time_sum / max(1, len(train_loader))

        if torch.cuda.is_available():
            peak_mem_bytes = torch.cuda.max_memory_allocated(device=GPU_iden)
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

        # ---------- validation (Dice) ----------
        model.eval()
        dsc_all = utils.AverageMeter()

        with torch.no_grad():
            for data in val_loader:
                data = [t.cuda(non_blocking=True) for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                # x -> y direction for evaluation (fixed)
                input_pair = torch.cat((x, y), dim=1)
                output, flow = model(input_pair)

                # warp segmentation with external registration model (nearest)
                y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=36)
                y_seg_oh = y_seg_oh.squeeze(1).permute(0, 4, 1, 2, 3).float()

                def_segs = []
                for i in range(36):
                    def_seg = reg_model(y_seg_oh[:, i:i+1, ...].float(), flow)
                    def_segs.append(def_seg)
                def_seg = torch.cat(def_segs, dim=1)

                dsc = utils.dice_val_VOI(def_seg.long(), x_seg.long(), use_gpu=True)
                dsc_all.update(dsc.item(), x.size(0))

        mean_dsc = dsc_all.avg
        writer.add_scalar('DSC/validate', mean_dsc, epoch)
        print('Validation Dice at epoch {}: {:.4f}'.format(epoch, mean_dsc))

        # Save last and (if improved) best checkpoint
        ckpt_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': max(best_dsc, mean_dsc),
        }
        save_checkpoint(ckpt_state, exp_dir, filename=f'epoch_{epoch:03d}.pth')

        if mean_dsc > best_dsc:
            best_dsc = mean_dsc
            save_checkpoint(ckpt_state, exp_dir, filename='best.pth')
            print('Model saved (new best DSC: {:.4f})'.format(best_dsc))

    writer.close()


if __name__ == '__main__':
    main()
