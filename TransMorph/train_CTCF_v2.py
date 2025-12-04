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


# ---------------------- Logger & small helpers ---------------------- #

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


def comput_fig(img: torch.Tensor) -> plt.Figure:
    """
    Build a matplotlib figure with 16 axial slices from [B, C, D, H, W] tensor.
    Assumes B=1, C=1. Uses slices z = 48..63 for visualization.
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

    This matches the original TransMorph trainer for OASIS.
    """
    new_lr = round(init_lr * np.power(1 - (epoch / max_epochs), power), 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def mk_grid_img(grid_step: int,
                line_thickness: int = 1,
                grid_sz=(160, 192, 224)) -> torch.Tensor:
    """
    Build a binary 3D grid [1, 1, D, H, W] used to visualize deformation fields.
    Ones mark the grid lines, zeros elsewhere.
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
                    save_dir: str = 'models/',
                    filename: str = 'checkpoint.pth.tar',
                    max_model_num: int = 8) -> None:
    """
    Save a checkpoint in `save_dir/filename` and keep only last `max_model_num` files
    (older ones are removed in order).
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))


# ---------------------- CTCF-specific helpers: ICON / Cycle / Jacobian ---------------------- #

def warp_with_model(model, img, flow):
    """
    Warp an image using the model's internal spatial transformer.

    Args:
        model: registration network that exposes model.spatial_trans(img, flow).
        img  : [B, C, D, H, W] input volume.
        flow : [B, 3, D, H, W] displacement field (dz, dy, dx).

    Returns:
        Warped image with the same shape as img.
    """
    return model.spatial_trans(img, flow)


def compose_flow(model, flow_ab, flow_ba):
    """
    Compute the coordinate composition of two displacement fields.

    We represent deformation as:
        phi(x) = x + u(x),
    where u(x) is the displacement.

    If phi_ab(x) = x + u_ab(x)  (A → B)
       phi_ba(x) = x + u_ba(x)  (B → A),

    then the composed displacement field for phi_ab ∘ phi_ba can be approximated by:
        (phi_ab ∘ phi_ba)(x) - x ≈ u_ab(x) + warp(u_ba, u_ab)(x),

    where warp(u_ba, u_ab) evaluates u_ba at coordinates displaced by u_ab.

    Args:
        flow_ab: [B,3,D,H,W] displacement from A→B.
        flow_ba: [B,3,D,H,W] displacement from B→A.

    Returns:
        Displacement field [B,3,D,H,W] for phi_ab ∘ phi_ba.
    """
    flow_ba_warped = warp_with_model(model, flow_ba, flow_ab)
    return flow_ab + flow_ba_warped


def icon_loss(model, flow_ab, flow_ba):
    """
    Inverse-consistency loss (ICON).

    Ideally, forward and backward deformations should be mutual inverses:
        phi_ab ∘ phi_ba ≈ Id
        phi_ba ∘ phi_ab ≈ Id

    This loss penalizes deviations from invertibility by composing flows in both
    directions and forcing the resulting displacement to be close to zero.

    Args:
        flow_ab: displacement field A→B.
        flow_ba: displacement field B→A.

    Returns:
        Scalar ICON penalty.
    """
    comp_ab_ba = compose_flow(model, flow_ab, flow_ba)  # A→B→A
    comp_ba_ab = compose_flow(model, flow_ba, flow_ab)  # B→A→B
    return (comp_ab_ba.pow(2).mean() + comp_ba_ab.pow(2).mean())


def cycle_image_loss(model, x, y, x_warp, y_warp, flow_xy, flow_yx):
    """
    Cycle-consistency loss in image space.

    We expect the following cycles to be close to identity:
        x → y (via flow_xy) → x   (via flow_yx)
        y → x (via flow_yx) → y   (via flow_xy)

    This loss compares the reconstructed volumes x_cycle, y_cycle with the
    original x, y in L1 sense.

    Args:
        x, y      : original volumes   [B,1,D,H,W].
        x_warp    : x warped toward y  [B,1,D,H,W].
        y_warp    : y warped toward x  [B,1,D,H,W].
        flow_xy   : displacement x→y   [B,3,D,H,W].
        flow_yx   : displacement y→x   [B,3,D,H,W].

    Returns:
        Scalar L1 cycle-consistency loss.
    """
    x_cycle = warp_with_model(model, x_warp, flow_yx)
    y_cycle = warp_with_model(model, y_warp, flow_xy)

    return (x_cycle - x).abs().mean() + (y_cycle - y).abs().mean()


def jacobian_det_3d(flow):
    """
    Compute the 3D Jacobian determinant of a deformation field.

    A deformation is:
        phi(x) = x + u(x),
    where u(x) is the displacement.

    The Jacobian matrix of phi is:
        J = I + ∇u,

    where ∇u is the spatial gradient of the displacement components.
    If det(J) <= 0, folding or topology violations occur (non-diffeomorphic warp).

    Args:
        flow: [B,3,D,H,W], channels ordered as (dz, dy, dx).

    Returns:
        detJ: [B,1,D,H,W] Jacobian determinant field.
    """
    B, C, D, H, W = flow.shape
    assert C == 3, "Flow must be [B,3,D,H,W] with channels (z,y,x)."

    # Split displacement into u_x, u_y, u_z components.
    u = flow[:, 2:3]  # dx
    v = flow[:, 1:2]  # dy
    w = flow[:, 0:1]  # dz

    def grad_central(u_comp):
        """
        Compute central finite differences with boundary replication.
        Returns gradients along x (W), y (H), z (D).
        """
        # gradient along W (x)
        u_pad = torch.nn.functional.pad(u_comp, (1, 1, 0, 0, 0, 0), mode='replicate')
        ux = (u_pad[..., 2:] - u_pad[..., :-2]) * 0.5

        # gradient along H (y)
        u_pad = torch.nn.functional.pad(u_comp, (0, 0, 1, 1, 0, 0), mode='replicate')
        uy = (u_pad[..., 2:, :] - u_pad[..., :-2, :]) * 0.5

        # gradient along D (z)
        u_pad = torch.nn.functional.pad(u_comp, (0, 0, 0, 0, 1, 1), mode='replicate')
        uz = (u_pad[:, :, 2:, :, :] - u_pad[:, :, :-2, :, :]) * 0.5

        return ux, uy, uz

    # Compute gradients for each component.
    ux, uy, uz = grad_central(u)
    vx, vy, vz = grad_central(v)
    wx, wy, wz = grad_central(w)

    # Assemble Jacobian = I + ∇u.
    J11 = 1.0 + ux; J12 = uy;      J13 = uz
    J21 = vx;      J22 = 1.0 + vy; J23 = vz
    J31 = wx;      J32 = wy;      J33 = 1.0 + wz

    # Determinant of 3×3 matrix.
    detJ = (
        J11 * (J22 * J33 - J23 * J32)
        - J12 * (J21 * J33 - J23 * J31)
        + J13 * (J21 * J32 - J22 * J31)
    )

    return detJ.unsqueeze(1)


def neg_jacobian_penalty(flow, eps=0.0):
    """
    Penalize non-positive Jacobian determinants (foldings).

    We compute detJ for the deformation field and enforce:
        detJ > eps  almost everywhere.

    Using:
        L_jac = E[ ReLU( -(detJ - eps) ) ]

    If eps > 0, we introduce a safety margin (detJ must be greater than eps).

    Args:
        flow: [B,3,D,H,W] displacement field.
        eps : safety margin for detJ > eps.

    Returns:
        Scalar penalty value.
    """
    # For numerical stability, compute Jacobian in full fp32
    detJ = jacobian_det_3d(flow.float())
    return torch.relu(-(detJ - eps)).mean()


def multiclass_soft_dice_loss(pred, target, num_classes=36, eps=1e-5):
    """
    Memory-efficient multi-class soft Dice loss.

    Args:
        pred  : [B, C, D, H, W] soft class maps (e.g., warped one-hot segmentations),
                C must equal num_classes.
        target: [B, 1, D, H, W] integer labels in [0, num_classes-1].

    We avoid building a full one-hot volume for target at full resolution,
    which would be very memory-hungry for 3D MRI.

    Instead, we loop over classes and construct binary masks on the fly:
        T_c(x) = 1 if target(x) == c else 0
    Dice_c = (2 * sum(P_c * T_c) + eps) / (sum(P_c) + sum(T_c) + eps)

    We aggregate Dice over foreground classes c = 1..num_classes-1
    (background c=0 is ignored, similar to VOI evaluation), then define:
        Loss = 1 - mean_c(Dice_c)
    """
    B, C, D, H, W = pred.shape
    assert C == num_classes, f"pred has {C} channels, expected {num_classes}"

    # Ensure target has shape [B, D, H, W] of integers.
    target = target.long().squeeze(1)  # [B, D, H, W]

    device = pred.device
    dice_sum = pred.new_zeros(1, device=device)

    # Loop over foreground classes; background is class 0.
    valid_classes = 0
    for c in range(1, num_classes):
        p_c = pred[:, c, ...]                  # [B, D, H, W]
        t_c = (target == c).float()            # [B, D, H, W]

        # If this class does not appear at all, skip it to avoid division by ~0.
        if t_c.sum() < 1.0 and p_c.sum() < 1.0:
            continue

        inter = (p_c * t_c).sum()
        denom = p_c.sum() + t_c.sum()

        dice_c = (2.0 * inter + eps) / (denom + eps)
        dice_sum = dice_sum + dice_c
        valid_classes += 1

    if valid_classes == 0:
        # If there is no foreground at all, define loss as 0 (perfect by convention).
        return pred.new_zeros(1, device=device)

    mean_dice = dice_sum / float(valid_classes)
    return 1.0 - mean_dice


# ---------------------- CLI ---------------------- #

def parse_args():
    """
    CLI for CTCF_v2 training on OASIS.

    Differences vs TransMorph baseline:
      - additional ICON / cycle / Jacobian penalties.
      - AMP + gradient checkpointing for memory efficiency.
      - custom memory-friendly multi-class Dice loss.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='Path to OASIS training .pkl files (e.g. .../All)')
    p.add_argument('--val_dir', required=True, help='Path to OASIS validation/test .pkl files (e.g. .../Test)')
    p.add_argument('--exp', default='CTCF_v2', help='Experiment name (for experiments/ and logs/)')
    p.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    p.add_argument('--max_epoch', type=int, default=500, help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    p.add_argument('--cont', action='store_true', help='Resume from latest checkpoint in experiments/<exp>/')
    p.add_argument('--gpu', type=int, default=0, help='GPU id to use')

    # extra loss weights
    p.add_argument('--w_ncc', type=float, default=1.0)
    p.add_argument('--w_dsc', type=float, default=1.0)
    p.add_argument('--w_reg', type=float, default=1.0)
    p.add_argument('--w_icon', type=float, default=0.1)
    p.add_argument('--w_cyc', type=float, default=0.1)
    p.add_argument('--w_jac', type=float, default=0.02)
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

    # ---------- experiment dirs ----------
    exp_root = args.exp.rstrip('/') + '/'
    exp_dir = os.path.join('experiments', exp_root)
    log_dir = os.path.join('logs', exp_root)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- logger + TensorBoard ----------
    sys.stdout = Logger(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    batch_size = args.batch_size
    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir   = args.val_dir if args.val_dir.endswith(os.sep) else args.val_dir + os.sep

    lr = args.lr
    max_epoch = args.max_epoch
    epoch_start = 0
    cont_training = args.cont

    W_ncc  = args.w_ncc
    W_dsc  = args.w_dsc
    W_reg  = args.w_reg
    W_icon = args.w_icon
    W_cyc  = args.w_cyc
    W_jac  = args.w_jac

    print(f'>>> Experiment: {args.exp}')
    print(f'    train_dir = {train_dir}')
    print(f'    val_dir   = {val_dir}')
    print(f'    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={cont_training}')
    print(f'    loss weights: NCC={W_ncc}, DSC={W_dsc}, REG={W_reg}, ICON={W_icon}, CYC={W_cyc}, JAC={W_jac}')

    # ---------- model ----------
    config = CONFIGS_TM['TransMorph-Large']
    # Enable gradient checkpointing inside TransMorph blocks to save memory.
    if hasattr(config, "use_checkpoint"):
        config.use_checkpoint = True
    model = TransMorph.TransMorph(config).cuda()

    # Spatial transformers (used only in validation / visualization).
    reg_model = utils.register_model(config.img_size, 'nearest').cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear').cuda()

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
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set   = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    print(f'>>> #train={len(train_loader.dataset)}, #val={len(val_loader.dataset)}')

    # ---------- optimizer + base losses ----------
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')

    scaler = torch.amp.GradScaler("cuda")  # AMP gradient scaler
    loss_all = utils.AverageMeter()

    # ---------- training loop ----------
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts (epoch {})'.format(epoch))
        
         # --- perf: start timing & reset peak memory ---
        epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        loss_all.reset()
        idx = 0
        iter_time_sum = 0.0

        for data in train_loader:
            idx += 1
            model.train()

            # --- perf: iteration start time ---
            iter_start_time = time.perf_counter()

            cur_lr = adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda(non_blocking=True) for t in data]
            x = data[0]      # moving / fixed images
            y = data[1]
            x_seg = data[2]  # segmentation labels
            y_seg = data[3]

            optimizer.zero_grad(set_to_none=True)

            # ----- forward pass with AMP -----
            with torch.amp.autocast("cuda"):
                # ===== direction x -> y =====
                x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
                x_seg_oh = torch.squeeze(x_seg_oh, 1)
                x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3)  # [B,36,D,H,W]

                x_in = torch.cat((x, y), dim=1)
                out_xy, flow_xy = model(x_in)  # warped x toward y + flow

                def_segs_xy = []
                for i in range(36):
                    def_seg = model.spatial_trans(x_seg_oh[:, i:i+1, ...].float(), flow_xy)
                    def_segs_xy.append(def_seg)
                def_seg_xy = torch.cat(def_segs_xy, dim=1)  # [B,36,D,H,W]

                L_ncc_xy = criterion_ncc(out_xy, y)
                L_dsc_xy = multiclass_soft_dice_loss(def_seg_xy, y_seg, num_classes=36)
                L_reg_xy = criterion_reg(flow_xy, y)

                # ===== direction y -> x =====
                y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=36)
                y_seg_oh = torch.squeeze(y_seg_oh, 1)
                y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3)  # [B,36,D,H,W]

                y_in = torch.cat((y, x), dim=1)
                out_yx, flow_yx = model(y_in)

                def_segs_yx = []
                for i in range(36):
                    def_seg = model.spatial_trans(y_seg_oh[:, i:i+1, ...].float(), flow_yx)
                    def_segs_yx.append(def_seg)
                def_seg_yx = torch.cat(def_segs_yx, dim=1)

                L_ncc_yx = criterion_ncc(out_yx, x)
                L_dsc_yx = multiclass_soft_dice_loss(def_seg_yx, x_seg, num_classes=36)
                L_reg_yx = criterion_reg(flow_yx, x)

                # ===== CTCF-specific losses =====
                L_icon = icon_loss(model, flow_xy, flow_yx)
                L_cyc  = cycle_image_loss(model, x, y, out_xy, out_yx, flow_xy, flow_yx)
                L_jac  = neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)

                # Symmetrize basic losses over two directions.
                L_ncc = 0.5 * (L_ncc_xy + L_ncc_yx)
                L_dsc = 0.5 * (L_dsc_xy + L_dsc_yx)
                L_reg = 0.5 * (L_reg_xy + L_reg_yx)

                loss = (W_ncc * L_ncc +
                        W_dsc * L_dsc +
                        W_reg * L_reg +
                        W_icon * L_icon +
                        W_cyc * L_cyc +
                        W_jac * L_jac)

            # ---------- NaN diagnostics ----------
            if not torch.isfinite(loss):
                print("===== NON-FINITE LOSS DETECTED =====")
                def safe_item(t):
                    return t.item() if torch.isfinite(t) else 'NaN/Inf'
                print(f"  L_ncc={safe_item(L_ncc)}")
                print(f"  L_dsc={safe_item(L_dsc)}")
                print(f"  L_reg={safe_item(L_reg)}")
                print(f"  L_icon={safe_item(L_icon)}")
                print(f"  L_cyc={safe_item(L_cyc)}")
                print(f"  L_jac={safe_item(L_jac)}")
                print(f"  flow_xy stats: min={flow_xy.min().item():.4e}, max={flow_xy.max().item():.4e}")
                print(f"  flow_yx stats: min={flow_yx.min().item():.4e}, max={flow_yx.max().item():.4e}")
                raise RuntimeError("Non-finite loss in CTCF_v2 training (see logs for details).")

            loss_all.update(loss.item(), x.numel())

            # ----- backward + optimizer step with AMP -----
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # free some intermediate tensors explicitly
            del x_seg_oh, y_seg_oh, def_segs_xy, def_seg_xy, def_segs_yx, def_seg_yx

            # --- perf: accumulate iteration time ---
            iter_time = time.perf_counter() - iter_start_time
            iter_time_sum += iter_time

            if idx % 10 == 0:
                print(
                    f"Iter {idx:4d} / {len(train_loader):4d} | "
                    f"loss={loss_all.avg:.4f} | "
                    f"NCC={L_ncc.item():.4f} DSC={L_dsc.item():.4f} REG={L_reg.item():.4f} "
                    f"ICON={L_icon.item():.4f} CYC={L_cyc.item():.4f} JAC={L_jac.item():.4f} "
                    f"lr={cur_lr:.1e}"
                )

        # log scalars per epoch
        writer.add_scalar('Loss/train_total', loss_all.avg, epoch)
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
                data = [t.cuda(non_blocking=True) for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                def_out = reg_model([x_seg.float(), output[1].float()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].float()])

                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print('val DSC running avg: {:.4f}'.format(eval_dsc.avg))

        best_dsc = max(eval_dsc.avg, best_dsc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=exp_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)

        # visualizations
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