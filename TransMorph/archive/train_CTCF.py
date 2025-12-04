import os, argparse, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from TransMorph.models.utils_torch import setup_torch, amp_context_and_scaler
from TransMorph.models.utils_train import get_logger, snapshot_env, validate
from TransMorph.models.configs_CTCF import CONFIGS_CTCF
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
from TransMorph.models.wrappers import BaseRegModel
from TransMorph.data import datasets, trans
from TransMorph.models.cascade import CascadeCTCF
from TransMorph.models.utils_field import warp, compose_flows, sdlogj_metric
from TransMorph.models.losses_ctcf import LNCCLoss, Grad3dLoss, icon_flow_loss, cycle_image_loss, neg_jac_loss

NUM_WORKERS = 8

def build_dataloaders(train_dir, val_dir, batch_size=1, n_workers=NUM_WORKERS):
    tf = lambda: transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_paths = glob.glob(os.path.join(train_dir, '*.pkl'))
    val_paths   = glob.glob(os.path.join(val_dir, '*.pkl'))
    ds_train = datasets.OASISBrainDataset(train_paths, transforms=tf())
    ds_val   = datasets.OASISBrainDataset(val_paths,   transforms=tf())
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, persistent_workers=(n_workers>0),
                              pin_memory=True, drop_last=True, prefetch_factor=2)
    val_loader   = DataLoader(ds_val, batch_size=1, shuffle=False,
                              num_workers=max(1,n_workers//2), persistent_workers=(n_workers>0),
                              pin_memory=True, prefetch_factor=2)
    return train_loader, val_loader

def main():
    # ---------- CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--val_dir',   required=True)
    ap.add_argument('--exp',       default='CTCF_LargeGPU')
    ap.add_argument('--cfg',       default='CTCF-LargeGPU')
    ap.add_argument('--precision', choices=['tf32','ieee','none'], default='tf32')
    args = ap.parse_args()

    # ---------- torch bootstrap ----------
    setup_torch(precision=args.precision)

    # ---------- конфиг ----------
    C = CONFIGS_CTCF[args.cfg]
    hp = C.hp                         # epochs, batch_size, lr, weight_decay, val_every, use_amp
    loss_sched = C.loss_schedule      # callable(epoch) -> dict весов

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- лог/тб ----------
    log_dir = os.path.join('logs', args.exp)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(os.path.join(log_dir, "train_console.log"), name=args.exp)
    writer = SummaryWriter(log_dir=log_dir)
    env = snapshot_env(os.path.join(log_dir, "run_env.json"))
    logger.info(f">> env: {env}")
    logger.info(f">> cfg={args.cfg}  img_size={tuple(C.img_size)}  hp={hp}")

    # ---------- data ----------
    train_loader, val_loader = build_dataloaders(args.train_dir, args.val_dir, batch_size=hp.batch_size)
    logger.info(f">> sanity: #train={len(train_loader.dataset)}  #val={len(val_loader.dataset)}")
    sample = next(iter(train_loader))
    logger.info(f"  sample shapes: {tuple(sample[0].shape)} {tuple(sample[1].shape)}")

    # ---------- model ----------
    base_cfg = CONFIGS_TM[C.base_backbone]
    base_cfg.img_size = tuple(C.img_size)
    model = CascadeCTCF(lambda cfg: BaseRegModel(cfg), base_cfg, levels=C.levels).to(device)
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    logger.info(f">> torch.compile disabled (SM {cc[0]}.{cc[1]} or not requested)")

    # ---------- losses/optim ----------
    lncc = LNCCLoss(win=(9,9,9)).to(device)
    reg_grad = Grad3dLoss(penalty='l2').to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=hp.epochs, eta_min=1e-6)

    # ---------- AMP ----------
    autocast_ctx, scaler = amp_context_and_scaler(hp.use_amp)

    best_val = -1.0
    global_step = 0

    for epoch in range(1, hp.epochs + 1):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        W = loss_sched(epoch)
        writer.add_scalar('opt/lr', sched.get_last_lr()[0], epoch)
        for k, v in W.items(): writer.add_scalar(f'opt/{k}', v, epoch)

        logger.info(f">> epoch {epoch} start; batches: {len(train_loader)}")

        for i, data in enumerate(train_loader):
            F_img = data[0].to(device).float()
            M_img = data[1].to(device).float()

            target_DHW = tuple(C.img_size)
            if F_img.shape[-3:] != target_DHW:
                F_img = torch.nn.functional.interpolate(F_img, size=target_DHW, mode='trilinear', align_corners=True)
                M_img = torch.nn.functional.interpolate(M_img, size=target_DHW, mode='trilinear', align_corners=True)

            optim.zero_grad(set_to_none=True)
            with autocast_ctx:
                M_warp, phi_ab = model(F_img, M_img)
                F_warp, phi_ba = model(M_img, F_img)

                L_sim = lncc(F_img, M_warp)
                L_reg = reg_grad(phi_ab) + reg_grad(phi_ba)
                L_icon = icon_flow_loss(phi_ab, phi_ba, compose_flows)
                L_jac  = neg_jac_loss(phi_ab) + neg_jac_loss(phi_ba)
                L_cyc  = cycle_image_loss(F_img, M_img, warp, phi_ab, phi_ba)

                loss = (W['w_sim']*L_sim + W['w_reg']*L_reg + W['w_icon']*L_icon +
                        W['w_jac']*L_jac + W['w_cyc']*L_cyc)

            if not torch.isfinite(loss):
                logger.warning(f"[WARN] non-finite loss at step {global_step}; skip")
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            if global_step % 10 == 0:
                logger.info(
                    f"step {global_step}: loss={loss.item():.4f} | "
                    f"L_sim={L_sim.item():.4f} L_reg={L_reg.item():.4f} "
                    f"L_icon={L_icon.item():.4f} L_jac={L_jac.item():.4f} L_cyc={L_cyc.item():.4f}"
                )
                writer.add_scalar('train/loss',   loss.item(), global_step)
                writer.add_scalar('train/L_sim',  L_sim.item(),  global_step)
                writer.add_scalar('train/L_reg',  L_reg.item(),  global_step)
                writer.add_scalar('train/L_icon', L_icon.item(), global_step)
                writer.add_scalar('train/L_jac',  L_jac.item(),  global_step)
                writer.add_scalar('train/L_cyc',  L_cyc.item(),  global_step)

            global_step += 1

        # --- validate/save ---
        if hp.val_every and (epoch % hp.val_every == 0) and len(val_loader.dataset) > 0:
            val_dice = validate(model, val_loader, warp_fn=warp, sdlogj_fn=sdlogj_metric,
                                target_shape=tuple(C.img_size), device=device)
            writer.add_scalar('val/dice', val_dice, epoch)
            exp_dir = os.path.join('experiments', args.exp)
            os.makedirs(exp_dir, exist_ok=True)
            ckpt = os.path.join(exp_dir, 'best.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_dice': float(val_dice)}, ckpt)
            logger.info(f"Saved checkpoint to {ckpt} (val_dice={val_dice:.4f})")

        sched.step()

if __name__ == "__main__":
    main()