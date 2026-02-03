import os
import glob
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import optim

from experiments.IXI import datasets
from models.UTSRMorph.model import CONFIGS as CONFIGS_UM, UTSRMorph

from utils import (
    trans,
    AverageMeter,
    setup_device,
    make_exp_dirs,
    attach_stdout_logger,
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
)


@torch.no_grad()
def forward_flow_utsrmorph(x: torch.Tensor, y: torch.Tensor, *, model: torch.nn.Module) -> torch.Tensor:
    inp = torch.cat((x, y), dim=1)
    use_amp = torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        _, flow = model(inp)
    return flow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help=".../IXI_data/Train (folder with *.pkl)")
    p.add_argument("--val_dir", required=True, help=".../IXI_data/Val (folder with *.pkl)")
    p.add_argument("--atlas_path", required=True, help=".../IXI_data/atlas.pkl")
    p.add_argument("--exp", type=str, default="UTSRMorph_IXI")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cont", action="store_true")
    p.add_argument("--config", type=str, default="UTSRMorph-Large")
    p.add_argument("--max_epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=4.0)
    p.add_argument("--resume", default="")
    p.add_argument("--tb_images_every", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)

    train_dir = args.train_dir.rstrip("/\\") + os.sep
    val_dir = args.val_dir.rstrip("/\\") + os.sep
    print(f">>> Experiment: {args.exp}")
    print(f"    train_dir = {train_dir}{os.sep}")
    print(f"    val_dir   = {val_dir}{os.sep}")
    print(f"    max_epoch = {args.max_epoch}")
    print(f"    lr        = {args.lr}")
    print(f"    atlas     ={args.atlas_path}")
    
    paths = make_exp_dirs(args.exp)
    attach_stdout_logger(paths.log_dir)

    ckpt_dir = os.path.join(paths.exp_dir, "ckpt")
    vis_dir = os.path.join(paths.exp_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=paths.log_dir)

    config = CONFIGS_UM[args.config]
    use_amp = torch.cuda.is_available()
    model = UTSRMorph(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_tf = transforms.Compose([trans.RandomFlip(0),NumpyType((np.float32, np.float32))])
    val_tf = transforms.Compose([trans.Seg_norm(),NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + "*.pkl"),args.atlas_path,transforms=train_tf)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + "*.pkl"),args.atlas_path,transforms=val_tf)
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    val_loader = DataLoader(val_set,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=True)

    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty="l2")

    # resume
    epoch_start = 0
    best_dsc = -1.0
    if args.resume:
        ckpt = load_checkpoint_if_exists(args.resume,model=model,optimizer=optimizer,map_location=device)
        if ckpt is not None:
            epoch_start = int(ckpt.get("epoch", -1)) + 1
            best_dsc = float(ckpt.get("best_dsc", best_dsc))
            if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            print(f">>> Resumed from {args.resume} @ epoch {epoch_start}, best={best_dsc:.4f}")

    for epoch in range(epoch_start, args.max_epoch):
        model.train()
        t0_epoch = perf_epoch_start()
        cur_lr = adjust_learning_rate_poly(optimizer, epoch, args.max_epoch, args.lr)
        writer.add_scalar("LR", cur_lr, epoch)

        meters = {
            "all": AverageMeter(),
            "ncc": AverageMeter(),
            "reg": AverageMeter(),
        }

        iter_time_sum = 0.0

        print(f"Training Starts (epoch {epoch:03d})")
        for it, (x, y) in enumerate(train_loader, start=1):
            t_iter0 = time.perf_counter()

            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            
            inp = torch.cat((x, y), dim=1)  # [B,2,D,H,W]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out, flow = model(inp)

                with torch.autocast(device_type="cuda", enabled=False):
                    L_ncc = criterion_ncc(out.float(), y.float()) * args.w_ncc

                L_reg = criterion_reg(flow.float()) * args.w_reg
                loss = L_ncc + L_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            meters["all"].update(loss.item(), y.numel())
            meters["ncc"].update(L_ncc.item(), y.numel())
            meters["reg"].update(L_reg.item(), y.numel())
            iter_time_sum += (time.perf_counter() - t_iter0)

            if it % 10 == 0:
                print(
                    f"Iter {it:4d} / {len(train_loader):4d} | "
                    f"loss(avg)={meters['all'].avg:.4f} | "
                    f"last NCC={L_ncc.item():.4f} REG={L_reg.item():.4f} | "
                    f"lr={cur_lr:.2e}"
                )

        model.eval()
        with torch.no_grad():
            val = validate_oasis(
                model=model,
                val_loader=val_loader,
                device=device,
                forward_flow_fn=lambda x_, y_: forward_flow_utsrmorph(x_, y_, model=model),
                dice_fn=dice_val_VOI,
                register_model_cls=register_model,
                mk_grid_img_fn=mk_grid_img,
                grid_step=8,
                line_thickness=1,
            )
        val_dsc = float(val.dsc)
        fold_percent = float(val.fold_percent)

        is_best = val_dsc >= best_dsc
        if is_best:
            best_dsc = val_dsc

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dsc": best_dsc,
            "scaler": scaler.state_dict() if use_amp else None,
            "config": dict(config),
            "args": vars(args),
        }

        torch.save(state, os.path.join(ckpt_dir, "last.pth"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best.pth"))

        perf = perf_epoch_end(t0_epoch, iters=len(train_loader), iter_time_sum=iter_time_sum)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={meters['all'].avg:.4f} ncc={meters['ncc'].avg:.4f} reg={meters['reg'].avg:.4f} | "
            f"val_dice={val_dsc:.4f} best={best_dsc:.4f} | "
            f"fold%={fold_percent:.2f} | "
            f"lr={cur_lr:.6g} it_ms={perf.mean_iter_time_ms:.2f} peakGiB={(perf.peak_gpu_mem_gib or 0.0):.2f}"
        )

        writer.add_scalar("Loss/train_all", meters["all"].avg, epoch)
        writer.add_scalar("Loss/train_ncc", meters["ncc"].avg, epoch)
        writer.add_scalar("Loss/train_reg", meters["reg"].avg, epoch)
        writer.add_scalar("DSC/val", val_dsc, epoch)
        writer.add_scalar("Metric/fold_percent", fold_percent, epoch)

        plt.switch_backend("agg")
        if (epoch % max(1, int(args.tb_images_every))) == 0:
            last_vis = val.last_vis or {}
            def_out = last_vis.get("def_seg", None)
            def_grid = last_vis.get("def_grid", None)
            x_vis = last_vis.get("x_seg", None)
            y_vis = last_vis.get("y_seg", None)

            if def_out is not None and x_vis is not None and y_vis is not None:
                pred_fig = comput_fig(def_out.float())
                x_fig = comput_fig(x_vis.float())
                tar_fig = comput_fig(y_vis.float())

                writer.add_figure("prediction", pred_fig, epoch); plt.close(pred_fig)
                writer.add_figure("input", x_fig, epoch); plt.close(x_fig)
                writer.add_figure("ground truth", tar_fig, epoch); plt.close(tar_fig)

                if def_grid is not None:
                    grid_fig = comput_fig(def_grid.float())
                    writer.add_figure("Grid", grid_fig, epoch); plt.close(grid_fig)

    writer.close()


if __name__ == "__main__":
    main()