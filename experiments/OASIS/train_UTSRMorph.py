from torch.utils.tensorboard import SummaryWriter
import os, glob, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from experiments.OASIS import datasets
from models.UTSRMorph.model import CONFIGS as CONFIGS_UM, UTSRMorph

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

def forward_flow_utsrmorph(model, x, y):
    """
    Expected:
      x,y: [B,1,D,H,W]
      model(inp=[B,2,D,H,W]) -> (out, flow) where flow: [B,3,D,H,W]
    """
    inp = torch.cat((x, y), dim=1)  # [B,2,D,H,W]
    _, flow = model(inp)
    return flow


# ---------------------- CLI ---------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="Path to OASIS training .pkl files (e.g. .../All)")
    p.add_argument("--val_dir", required=True, help="Path to OASIS validation/test .pkl files (e.g. .../Test)")
    p.add_argument("--exp", default="UTSRMorph_v3", help="Experiment name (results/<exp>/, logs/<exp>/)")
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    p.add_argument("--max_epoch", type=int, default=500, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--cont", action="store_true", help="Resume from results/<exp>/last.pth.tar")
    p.add_argument("--gpu", type=int, default=0, help="GPU id to use")

    # loss weights (unsupervised)
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)

    # model config key
    p.add_argument(
        "--config",
        type=str,
        default="UTSRMorph-Large",
        help="Key in CONFIGS_UM (e.g., UTSRMorph-Large)",
    )

    # determinism
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (slower)")

    return p.parse_args()


# ---------------------- main ---------------------- #

def main():
    args = parse_args()

    # ---------- device ----------

    dev = setup_device(args.gpu, seed=args.seed, deterministic=args.deterministic)
    device = dev.device

    # ---------- experiment dirs + logger ----------

    paths = make_exp_dirs(args.exp)
    attach_stdout_logger(paths.log_dir)
    writer = SummaryWriter(log_dir=paths.log_dir)

    train_dir = args.train_dir if args.train_dir.endswith(os.sep) else args.train_dir + os.sep
    val_dir = args.val_dir if args.val_dir.endswith(os.sep) else args.val_dir + os.sep

    lr = args.lr
    max_epoch = args.max_epoch
    batch_size = args.batch_size

    W_ncc = args.w_ncc
    W_reg = args.w_reg

    print(f">>> Experiment: {args.exp}")
    print(f"    train_dir = {train_dir}")
    print(f"    val_dir   = {val_dir}")
    print(f"    cfg       = {args.config}")
    print(f"    lr={lr}, max_epoch={max_epoch}, batch_size={batch_size}, cont={args.cont}")
    print(f"    loss weights: NCC={W_ncc}, REG={W_reg}")

    # ---------- model ----------

    if args.config not in CONFIGS_UM:
        raise KeyError(f"Unknown UTSRMorph config '{args.config}'. Available: {list(CONFIGS_UM.keys())}")

    config = CONFIGS_UM[args.config]
    model = UTSRMorph(config).to(device)

    # ---------- optimizer + losses ----------

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty="l2")

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---------- resume ----------

    epoch_start = 0
    best_dsc = 0.0
    if args.cont:
        ckpt_path = os.path.join(paths.exp_dir, "last.pth.tar")
        ckpt = load_checkpoint_if_exists(ckpt_path, model, optimizer, map_location="cuda" if use_amp else "cpu")
        if ckpt:
            epoch_start = ckpt.get("epoch", 0)
            best_dsc = ckpt.get("best_dsc", 0.0)
            print(f"Loaded last checkpoint: epoch_start={epoch_start}, best_dsc={best_dsc:.4f}")
        else:
            print("No last.pth.tar found, starting from scratch.")

    # ---------- datasets ----------

    train_tf = transforms.Compose([NumpyType((np.float32, np.int16))])
    val_tf = transforms.Compose([NumpyType((np.float32, np.int16))])

    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + "*.pkl"), transforms=train_tf)
    val_set = datasets.OASISBrainInferDataset(glob.glob(val_dir + "*.pkl"), transforms=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    print(f">>> #train={len(train_loader.dataset)}, #val={len(val_loader.dataset)}")

    # -------------------- training loop -------------------- 

    for epoch in range(epoch_start, max_epoch):
        print(f"Training Starts (epoch {epoch})")

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
            x, y, x_seg, y_seg = batch  # [B,1,D,H,W] + segs

            if random.randint(0, 1) == 0:
                src = x
                tgt = y
            else:
                src = y
                tgt = x

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                inp = torch.cat((src, tgt), dim=1)  # [B,2,D,H,W]
                out, flow = model(inp)              # out ~ warped src->tgt, flow: [B,3,D,H,W]

                # NCC stability: compute in float32 (same idea as TM-DCA)
                with torch.amp.autocast("cuda", enabled=False):
                    L_ncc = criterion_ncc(out.float(), tgt.float()) * W_ncc

                # regularization
                L_reg = criterion_reg(flow, tgt) * W_reg

                loss = L_ncc + L_reg

            if not torch.isfinite(loss):
                print("===== NON-FINITE LOSS DETECTED (UTSRMorph) =====")
                print(f"loss={loss.item()} NCC={L_ncc.item()} REG={L_reg.item()}")
                print(f"flow stats: min={flow.min().item():.4e}, max={flow.max().item():.4e}")
                raise RuntimeError("Non-finite loss in UTSRMorph training.")

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_all.update(loss.item(), tgt.numel())
            iter_time_sum += (time.perf_counter() - iter_t0)

            if idx % 10 == 0:
                print(
                    f"Iter {idx:4d} / {len(train_loader):4d} | "
                    f"loss(avg)={loss_all.avg:.4f} | "
                    f"last NCC={L_ncc.item():.4f} REG={L_reg.item():.4f} | "
                    f"lr={cur_lr:.1e}"
                )

        writer.add_scalar("Loss/train_total", loss_all.avg, epoch)
        print(f"Epoch {epoch} loss {loss_all.avg:.4f}")

        # ---------------- Performance ----------------

        perf = perf_epoch_end(t0, iters=idx, iter_time_sum=iter_time_sum)
        if perf.peak_gpu_mem_gib is not None:
            print(
                f"[PERF] Epoch {epoch}: time={perf.epoch_time_sec:.1f}s, "
                f"iter={perf.mean_iter_time_ms:.1f} ms/iter, "
                f"peak={perf.peak_gpu_mem_gib:.2f} GiB"
            )
            writer.add_scalar("perf/epoch_time_sec", perf.epoch_time_sec, epoch)
            writer.add_scalar("perf/iter_time_ms", perf.mean_iter_time_ms, epoch)
            writer.add_scalar("perf/peak_gpu_mem_GB", perf.peak_gpu_mem_gib, epoch)
        else:
            print(f"[PERF] Epoch {epoch}: time={perf.epoch_time_sec:.1f}s, iter={perf.mean_iter_time_ms:.1f} ms/iter")

        # -------------------- Validation (unified) -------------------- #

        val = validate_oasis(
            model=model,
            val_loader=val_loader,
            device=device,
            forward_flow_fn=lambda x, y: forward_flow_utsrmorph(model, x, y),
            dice_fn=dice_val_VOI,
            register_model_cls=register_model,
            mk_grid_img_fn=mk_grid_img,
        )
        print(f"val DSC: {val.dsc:.4f} | fold%: {val.fold_percent:.2f}")
        writer.add_scalar("DSC/validate", val.dsc, epoch)
        writer.add_scalar("Metric/validate_fold_percent", val.fold_percent, epoch)

        # tensors for visualization
        def_out = val.last_vis.get("def_seg", None)
        def_grid = val.last_vis.get("def_grid", None)
        x_vis = val.last_vis.get("x_seg", None)
        y_vis = val.last_vis.get("y_seg", None)

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


if __name__ == "__main__":
    main()