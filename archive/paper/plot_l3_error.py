"""
L3 before/after error-map visualization (for Reviewer 1, Comment #4).

Runs CTCF with return_all=True on a few OASIS pairs, renders |warped - fixed| maps
for (a) L2-only output and (b) L2+L3 output, plus the improvement = before - after.

Usage:
  python tools/visualize_l3_error.py --ckpt /home/roman/P/CTCF/results/CTCF_UPD_OASIS_E500/best.pth \
                                     --3 --gpu 0 --n_cases 3 --out results/figs/l3_error
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import OASIS
from experiments.core.model_adapters import CtcfAdapter
from experiments.core.train_runtime import PATHS, add_common_args
from experiments.inference import load_checkpoint_state
from utils import NumpyType, setup_device


def _ortho_slices(vol: np.ndarray):
    d, h, w = vol.shape
    return vol[d // 2], vol[:, h // 2, :], vol[:, :, w // 2]


def _panel(ax, img, vmin=None, vmax=None, cmap="gray", title=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if title is not None:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_figure(
    case_id: str, fixed: np.ndarray, moving: np.ndarray, def_l2: np.ndarray, def_full: np.ndarray, out_path: str
) -> None:
    err_before = np.abs(def_l2 - fixed)
    err_after = np.abs(def_full - fixed)
    improvement = err_before - err_after  # positive = L3 helped

    vmax_err = float(np.percentile(err_before, 99))
    vmax_imp = float(np.percentile(np.abs(improvement), 99))

    slices = {
        "fixed": _ortho_slices(fixed),
        "moving": _ortho_slices(moving),
        "err_before": _ortho_slices(err_before),
        "err_after": _ortho_slices(err_after),
        "improve": _ortho_slices(improvement),
    }

    rows = [
        ("Fixed", slices["fixed"], {"cmap": "gray"}),
        ("Moving", slices["moving"], {"cmap": "gray"}),
        ("|L2-fix|", slices["err_before"], {"cmap": "hot", "vmin": 0, "vmax": vmax_err}),
        ("|L2+L3-fix|", slices["err_after"], {"cmap": "hot", "vmin": 0, "vmax": vmax_err}),
        ("Improvement (L3)", slices["improve"], {"cmap": "RdBu_r", "vmin": -vmax_imp, "vmax": vmax_imp}),
    ]
    view_names = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.2, 2.2 * len(rows)))
    im_err = None
    im_imp = None
    for r, (title, imgs, kw) in enumerate(rows):
        for c in range(3):
            im = axes[r, c].imshow(imgs[c], interpolation="nearest", **kw)
            if r == 0 or c == 0:
                axes[r, c].set_title(f"{title} ({view_names[c]})" if r == 0 or c == 0 else None, fontsize=9)
            axes[r, c].axis("off")
            if title in ("|L2-fix|", "|L2+L3-fix|"):
                im_err = im
            elif title == "Improvement (L3)":
                im_imp = im

    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    if im_err is not None:
        cbar_ax_err = fig.add_axes([0.90, 0.46, 0.014, 0.34])
        fig.colorbar(im_err, cax=cbar_ax_err, label="|intensity diff|")
    if im_imp is not None:
        cbar_ax_imp = fig.add_axes([0.90, 0.08, 0.014, 0.16])
        fig.colorbar(im_imp, cax=cbar_ax_imp, label="improvement (L3)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pil_kwargs={"compress_level": 1})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, mode="infer")
    ap.add_argument("--ckpt", required=True, help="CTCF checkpoint (.pth / .pth.tar).")
    ap.add_argument("--strict_ckpt", type=int, default=0)
    ap.add_argument("--n_cases", type=int, default=3, help="Number of cases to render.")
    ap.add_argument("--out", default="results/figs/l3_error", help="Output directory.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctcf_config", default="CTCF-CascadeA")
    ap.add_argument("--time_steps", type=int, default=6)
    args = ap.parse_args()

    device = setup_device(args.gpu, seed=args.seed, deterministic=False)
    ds_paths = PATHS[int(args.paths)][args.ds.upper()]
    test_dir = ds_paths.get("test_dir", ds_paths["val_dir"])
    files = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))[: args.n_cases]
    if not files:
        raise RuntimeError(f"No .pkl files under {test_dir}")

    ds = OASIS.OASISBrainInferDataset(files, transforms=transforms.Compose([NumpyType((np.float32, np.int16))]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    adapter = CtcfAdapter()
    model = adapter.build(time_steps=args.time_steps, config_key=args.ctcf_config).to(device).eval()
    load_checkpoint_state(model, args.ckpt, strict=bool(args.strict_ckpt))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing to: {out_dir}")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x, y, _x_seg, _y_seg = [t.to(device, non_blocking=True) for t in batch]
            cid = Path(files[idx]).stem
            if cid.startswith("p_"):
                cid = cid[2:]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                def_full, flow_full, aux = model(x, y, return_all=True, alpha_l1=1.0)

            # Reconstruct L2-only deformation by applying flow_half_l2 upsampled (matches training path)
            flow_half_l2 = aux["flow_half_l2"].float()
            if getattr(model, "l2_full_res", False):
                flow_l2 = flow_half_l2  # already full
            else:
                from models.CTCF.blocks import upsample_flow

                flow_l2 = upsample_flow(flow_half_l2, scale_factor=2)
            def_l2 = model.st_full(x, flow_l2)

            fixed_np = y.detach().float().cpu().numpy()[0, 0]
            moving_np = x.detach().float().cpu().numpy()[0, 0]
            def_l2_np = def_l2.detach().float().cpu().numpy()[0, 0]
            def_full_np = def_full.detach().float().cpu().numpy()[0, 0]

            out_png = out_dir / f"l3_error_{cid}.png"
            make_figure(cid, fixed_np, moving_np, def_l2_np, def_full_np, str(out_png))
            print(f"  [{idx + 1}/{len(files)}] {cid} -> {out_png}")

    print("[DONE]")


if __name__ == "__main__":
    main()
