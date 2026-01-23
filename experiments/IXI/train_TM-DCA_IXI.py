from torch.utils.tensorboard import SummaryWriter
import os, glob, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from experiments.IXI import datasets
from utils import trans as trans_utils

from models.TransMorph_DCA.model import CONFIGS as CONFIGS_TM
import models.TransMorph_DCA.model as TransMorph

from utils import (
    AverageMeter,
    register_model,
    NCC_vxm,
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
    validate_oasis,
)

# ---------- Adapter for validate_oasis() ---------- #
def forward_flow_tm_dca(model, x, y):
    """
    Exactly as in your OASIS TM-DCA trainer:
      - network runs on half-res inputs (avg_pool /2)
      - upsample flow back to full-res and scale vectors *2
    """
    x_half = F.avg_pool3d(x, 2)
    y_half = F.avg_pool3d(y, 2)

    use_amp = torch.cuda.is_available()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        flow_half = model((x_half, y_half))

    flow_full = F.interpolate(
        flow_half.float(),
        scale_factor=2,
        mode="trilinear",
        align_corners=False
    ) * 2.0

    return flow_full


def parse_args():
    p = argparse.ArgumentParser()

    # IXI paths
    p.add_argument("--train_dir", type=str, required=True, help="Folder with IXI Train/*.pkl")
    p.add_argument("--val_dir", type=str, required=True, help="Folder with IXI Val/*.pkl")
    p.add_argument("--atlas_path", type=str, required=True, help="Path to atlas.pkl (img, seg)")

    # run
    p.add_argument("--exp", type=str, default="TM_DCA_IXI", help="Experiment name (results/<exp>, logs/<exp>)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cont", action="store_true", help="Resume from results/<exp>/last.pth.tar")

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    # losses
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)

    # image size (IXI default in original repo)
    p.add_argument("--full_D", type=int, default=160)
    p.add_argument("--full_H", type=int, default=192)
    p.add_argument("--full_W", type=int, default=224)

    # Dwin kernel params for config (как в оригинальном IXI тренере)
    p.add_argument("--dwin_z", type=int, default=7)
    p.add_argument("--dwin_y", type=int, default=5)
    p.add_argument("--dwin_x", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()

    dev = setup_device(args.gpu, seed=0, deterministic=False)
    device = dev.device

    paths = make_exp_dirs(args.exp)
    attach_stdout_logger(paths.log_dir)
    writer = SummaryWriter(log_dir=paths.log_dir)

    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir = args.val_dir if args.val_dir.endswith(os.sep) else args.val_dir + os.sep

    full_size = (args.full_D, args.full_H, args.full_W)
    half_size = (args.full_D // 2, args.full_H // 2, args.full_W // 2)

    # ---------- model ----------
    config = CONFIGS_TM["TransMorph-3-LVL"]
    config.img_size = half_size
    config.dwin_kernel_size = (args.dwin_z, args.dwin_y, args.dwin_x)
    config.window_size = (args.full_D // 32, args.full_H // 32, args.full_W // 32)

    model = TransMorph.TransMorphCascadeAd(config, time_steps=12).to(device)

    # ---------- optimizer + losses ----------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty="l2")

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # ---------- resume ----------
    epoch_start = 0
    best_dsc = 0.0
    if args.cont:
        ckpt_path = os.path.join(paths.exp_dir, "last.pth.tar")
        ckpt = load_checkpoint_if_exists(ckpt_path, model, optimizer, map_location="cuda")
        if ckpt:
            epoch_start = int(ckpt.get("epoch", 0))
            best_dsc = float(ckpt.get("best_dsc", 0.0))
            print(f"[resume] epoch_start={epoch_start} best_dsc={best_dsc:.4f}")

    # ---------- datasets (exactly as original IXI trainer) ----------
    train_composed = transforms.Compose([
        trans_utils.RandomFlip(0),
        trans_utils.NumpyType((np.float32, np.float32)),
    ])

    val_composed = transforms.Compose([
        trans_utils.Seg_norm(),                 # rearrange labels to 0..N
        trans_utils.NumpyType((np.float32, np.int16)),
    ])

    train_set = datasets.IXIBrainDataset(
        glob.glob(train_dir + "*.pkl"),
        args.atlas_path,
        transforms=train_composed
    )
    val_set = datasets.IXIBrainInferDataset(
        glob.glob(val_dir + "*.pkl"),
        args.atlas_path,
        transforms=val_composed
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
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f">>> Experiment: {args.exp}")
    print(f"    train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")
    print(f"    full_size={full_size} half_size={half_size}")
    print(f"    weights: NCC={args.w_ncc}, REG={args.w_reg}")

    # ---------- training loop ----------
    for epoch in range(epoch_start, args.max_epoch):
        print(f"Training Starts (epoch {epoch})")

        perf_epoch_start()

        # lr schedule (poly)
        cur_lr = adjust_learning_rate_poly(optimizer, epoch, args.max_epoch, args.lr)
        writer.add_scalar("LR", cur_lr, epoch)

        model.train()
        loss_all = AverageMeter()
        loss_ncc_meter = AverageMeter()
        loss_reg_meter = AverageMeter()

        for (x, y) in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # forward flow (half-res net, full-res flow)
            flow_full = forward_flow_tm_dca(model, x, y)

            # warp moving (atlas) to fixed (subject) using existing register_model()
            reg_bilin = register_model(full_size, mode="bilinear").to(device)
            x_warp = reg_bilin((x.float(), flow_full.float()))

            # losses (float32 for stability)
            loss_ncc = criterion_ncc(y.float(), x_warp.float()) * args.w_ncc
            loss_reg = criterion_reg(flow_full.float()) * args.w_reg
            loss = loss_ncc + loss_reg

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[NON-FINITE LOSS] loss={loss.item()} ncc={loss_ncc.item()} reg={loss_reg.item()}"
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_all.update(loss.item(), y.numel())
            loss_ncc_meter.update(loss_ncc.item(), y.numel())
            loss_reg_meter.update(loss_reg.item(), y.numel())

        writer.add_scalar("Loss/train", loss_all.avg, epoch)
        writer.add_scalar("Loss/train_ncc", loss_ncc_meter.avg, epoch)
        writer.add_scalar("Loss/train_reg", loss_reg_meter.avg, epoch)

        # ---------- validate ----------
        val_res = validate_oasis(
            model=model,
            val_loader=val_loader,
            device=device,
            forward_flow_fn=lambda a, b: forward_flow_tm_dca(model, a, b),
            dice_fn=dice_val_VOI,
            register_model_cls=register_model,
            mk_grid_img_fn=mk_grid_img,
            grid_step=8,
            line_thickness=1,
        )

        dsc = float(val_res.dsc)
        fold_percent = float(val_res.fold_percent)

        writer.add_scalar("DSC/validate", dsc, epoch)
        writer.add_scalar("Metric/validate_fold_percent", fold_percent, epoch)

        # visuals (reuse your comput_fig exactly)
        if val_res.last_vis:
            plt.switch_backend("agg")
            def_grid = val_res.last_vis.get("def_grid", None)
            x_seg = val_res.last_vis.get("x_seg", None)
            y_seg = val_res.last_vis.get("y_seg", None)
            def_seg = val_res.last_vis.get("def_seg", None)

            if def_grid is not None:
                fig = comput_fig(def_grid)
                writer.add_figure("Grid", fig, epoch)
                plt.close(fig)

            if x_seg is not None:
                fig = comput_fig(x_seg)
                writer.add_figure("input", fig, epoch)
                plt.close(fig)

            if y_seg is not None:
                fig = comput_fig(y_seg)
                writer.add_figure("ground truth", fig, epoch)
                plt.close(fig)

            if def_seg is not None:
                fig = comput_fig(def_seg)
                writer.add_figure("prediction", fig, epoch)
                plt.close(fig)

        # checkpoints
        is_best = dsc >= best_dsc
        if is_best:
            best_dsc = dsc

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

        if is_best:
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

        perf_epoch_end(writer=writer, epoch=epoch)

        print(f"[epoch {epoch}] train_loss={loss_all.avg:.4f} val_dsc={dsc:.4f} fold%={fold_percent:.2f} best={best_dsc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
