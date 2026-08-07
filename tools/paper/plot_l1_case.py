"""
L1 case study for Reviewer 1, Comment #5: illustrate a case where the coarse
Level-1 flow is essential for handling large initial misalignment.

Pipeline:
  1. Optional --find_worst: rank test cases by pre-registration residual |F-M|.mean()
     and report the top-N most misaligned case indices. Use the result to pick --case-index.
  2. With --case-index K, render a figure of K OASIS pair:
       Row per case × 6 cols: fixed, moving, |F-M|, L1-warped, |F-L1|, |F-full|
     Error maps share a single colormap; rightmost colorbar.

Usage:
  # 1) Find worst cases first (no figure rendered)
  python tools/visualize_l1_case.py --ckpt <CTCF_OASIS_ckpt> --3 --gpu 0 --find_worst --top 5

  # 2) Render the case you picked
  python tools/visualize_l1_case.py --ckpt <CTCF_OASIS_ckpt> --3 --gpu 0 \
      --case-index 14 --out results/figs/l1_case.png
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import OASIS
from experiments.core.model_adapters import CtcfAdapter
from experiments.core.train_runtime import PATHS, add_common_args
from experiments.inference import load_checkpoint_state
from models.CTCF.blocks import upsample_flow
from utils import NumpyType, setup_device


def build_ctcf(ckpt: str, config_key: str, time_steps: int, strict: bool, device):
    adapter = CtcfAdapter()
    model = adapter.build(time_steps=time_steps, config_key=config_key).to(device).eval()
    load_checkpoint_state(model, ckpt, strict=strict)
    return model


def l1_only_flow(model, mov_full, fix_full):
    """Run only Level-1 and return full-res flow (no L2, no L3)."""
    mov_quarter = F.interpolate(mov_full, scale_factor=0.25, mode="trilinear", align_corners=False)
    fix_quarter = F.interpolate(fix_full, scale_factor=0.25, mode="trilinear", align_corners=False)
    if getattr(model, "l1_half_res", False):
        mov_half = F.interpolate(mov_full, scale_factor=0.5, mode="trilinear", align_corners=False)
        fix_half = F.interpolate(fix_full, scale_factor=0.5, mode="trilinear", align_corners=False)
        flow_half_l1 = model.level1(mov_half, fix_half, return_features=False)
        flow_full = upsample_flow(flow_half_l1, scale_factor=2)
    else:
        flow_quarter = model.level1(mov_quarter, fix_quarter, return_features=False)
        flow_half = upsample_flow(flow_quarter, scale_factor=2)
        flow_full = upsample_flow(flow_half, scale_factor=2)
    return flow_full


def ortho_slices(vol: np.ndarray):
    d, h, w = vol.shape
    return vol[d // 2], vol[:, h // 2, :], vol[:, :, w // 2]


def find_worst_cases(files, top: int):
    """Rank by |fixed - moving| mean (higher = worse alignment)."""
    from utils import pkload

    scores = []
    for idx, p in enumerate(files):
        x, y, _, _ = pkload(p)
        diff = float(np.abs(x.astype(np.float32) - y.astype(np.float32)).mean())
        scores.append((idx, Path(p).stem, diff))
    scores.sort(key=lambda t: -t[2])
    print(f"\n{'rank':<6} {'idx':<6} {'case_id':<20} {'|F-M|.mean':<12}")
    print("-" * 46)
    for r, (idx, cid, d) in enumerate(scores[:top]):
        print(f"{r + 1:<6} {idx:<6} {cid:<20} {d:.4f}")
    return scores


def find_best_l1_cases(loader, files, model, device, top: int):
    """Rank by L1 improvement Δ = mean(|F-M|) - mean(|F-L1|) (higher = L1 helped more)."""
    scores = []
    for idx, batch in enumerate(loader):
        x, y, _, _ = [t.to(device) for t in batch]
        with torch.no_grad():
            flow_l1 = l1_only_flow(model, x, y)
            def_l1 = model.st_full(x, flow_l1)
        fixed = y.float().cpu().numpy()[0, 0]
        moving = x.float().cpu().numpy()[0, 0]
        warp = def_l1.float().cpu().numpy()[0, 0]
        e0 = float(np.abs(fixed - moving).mean())
        e1 = float(np.abs(fixed - warp).mean())
        scores.append((idx, Path(files[idx]).stem, e0, e1, e0 - e1))
    scores.sort(key=lambda t: -t[4])
    print(f"\n{'rank':<6} {'idx':<6} {'case_id':<20} {'|F-M|':<10} {'|F-L1|':<10} {'Δ (gain)':<10}")
    print("-" * 66)
    for r, (idx, cid, e0, e1, d) in enumerate(scores[:top]):
        print(f"{r + 1:<6} {idx:<6} {cid:<20} {e0:<10.4f} {e1:<10.4f} {d:<+10.4f}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, mode="infer")
    ap.add_argument("--ckpt", required=True, help="CTCF OASIS checkpoint.")
    ap.add_argument("--strict_ckpt", type=int, default=0)
    ap.add_argument(
        "--find_worst", action="store_true", help="Print top-N cases by initial |F-M| residual (no figure)."
    )
    ap.add_argument(
        "--find_best_l1",
        action="store_true",
        help="Print top-N cases by L1 improvement Δ=mean(|F-M|)-mean(|F-L1|) (no figure). Requires running L1.",
    )
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--case-index", type=int, nargs="+", default=[0], help="One or more case indices to render as rows."
    )
    ap.add_argument("--ctcf-config", default="CTCF-CascadeA")
    ap.add_argument("--time_steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figs/l1_case.png")
    ap.add_argument("--view", default="coronal", choices=["axial", "coronal", "sagittal"])
    args = ap.parse_args()

    ds_paths = PATHS[int(args.paths)][args.ds]
    test_dir = ds_paths.get("test_dir", ds_paths["val_dir"])
    files = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))
    if not files:
        raise RuntimeError(f"No .pkl in {test_dir}")

    if args.find_worst:
        find_worst_cases(files, args.top)
        return

    device = setup_device(args.gpu, seed=args.seed, deterministic=False)
    ds = OASIS.OASISBrainInferDataset(files, transforms=transforms.Compose([NumpyType((np.float32, np.int16))]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_ctcf(args.ckpt, args.ctcf_config, args.time_steps, bool(args.strict_ckpt), device)
    adapter = CtcfAdapter()

    if args.find_best_l1:
        find_best_l1_cases(loader, files, model, device, args.top)
        return

    rows = []
    case_ids = []
    for idx, batch in enumerate(loader):
        if idx not in args.case_index:
            continue
        x, y, _, _ = [t.to(device) for t in batch]
        cid = Path(files[idx]).stem
        case_ids.append(cid)

        with torch.no_grad():
            flow_l1 = l1_only_flow(model, x, y)
            def_l1 = model.st_full(x, flow_l1)
            flow_full = adapter.forward(model, x, y)
            def_full = model.st_full(x, flow_full)

        fixed = y.float().cpu().numpy()[0, 0]
        moving = x.float().cpu().numpy()[0, 0]
        warp_l1 = def_l1.float().cpu().numpy()[0, 0]
        warp_full = def_full.float().cpu().numpy()[0, 0]
        # Displacement magnitude (voxels) of L1 flow — proves L1 is non-trivial
        # even when warped intensities look similar to the moving image.
        mag_l1 = torch.linalg.norm(flow_l1, dim=1).float().cpu().numpy()[0]
        rows.append(
            {
                "cid": cid,
                "fixed": fixed,
                "moving": moving,
                "err_initial": np.abs(fixed - moving),
                "warp_l1": warp_l1,
                "err_l1": np.abs(fixed - warp_l1),
                "err_full": np.abs(fixed - warp_full),
                "mag_l1": mag_l1,
            }
        )

        if len(rows) == len(args.case_index):
            break

    if not rows:
        raise RuntimeError(f"No case matched --case-index {args.case_index} (have {len(files)} files).")

    # Unified color scale for all error maps across all rows
    vmax = float(max(np.percentile(r["err_initial"], 99) for r in rows))
    vmax_mag = float(max(np.percentile(r["mag_l1"], 99) for r in rows))

    view_idx = {"axial": 0, "coronal": 1, "sagittal": 2}[args.view]
    col_titles = ["|F-M|", "|F-L1|", "|F-full|", "|L1 flow|"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11.5, 2.6 * len(rows)), squeeze=False)
    im_err = None
    im_mag = None
    for r, row in enumerate(rows):
        panels = [
            (row["err_initial"], {"cmap": "hot", "vmin": 0, "vmax": vmax}),
            (row["err_l1"], {"cmap": "hot", "vmin": 0, "vmax": vmax}),
            (row["err_full"], {"cmap": "hot", "vmin": 0, "vmax": vmax}),
            (row["mag_l1"], {"cmap": "viridis", "vmin": 0, "vmax": vmax_mag}),
        ]
        for c, (img, kw) in enumerate(panels):
            ax = axes[r, c]
            sl = ortho_slices(img)[view_idx]
            im = ax.imshow(sl, interpolation="nearest", **kw)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            ax.axis("off")
            if c in (0, 1, 2):
                im_err = im
            if c == 3:
                im_mag = im

    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    if im_err is not None:
        cbar_ax = fig.add_axes([0.86, 0.12, 0.014, 0.76])
        fig.colorbar(im_err, cax=cbar_ax, label="|intensity diff|")
    if im_mag is not None:
        cbar_ax2 = fig.add_axes([0.945, 0.12, 0.014, 0.76])
        fig.colorbar(im_mag, cax=cbar_ax2, label="|L1 flow| (vox)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pil_kwargs={"compress_level": 1})
    plt.close(fig)
    print(f"[SAVED] {out_path}  (cases: {case_ids})")


if __name__ == "__main__":
    main()
