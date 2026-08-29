"""
Jacobian determinant maps for CTCF vs TM-DCA vs UTSRMorph (Reviewer 1, Comment #7).

Produces one figure per dataset: 3 rows (models) × 3 cols (axial/coronal/sagittal),
all panels share the same log|det J| colormap so folding is visually comparable.

Usage:
  python tools/visualize_jacobian_maps.py \
    --ckpt-ctcf  /home/roman/P/CTCF/results/CTCF_UPD_OASIS_E500/best.pth \
    --ckpt-tmdca /home/roman/P/CTCF/results/TM_DCA_unsup_OASIS/best.pth.tar \
    --ckpt-utsr  /home/roman/P/CTCF/results/UTSRMorph_OASIS/best.pth.tar \
    --utsr-config UTSRMorph-Large \
    --ds OASIS --3 --gpu 0 --case-index 0 --out results/figs/jacobian_oasis.png

  python tools/visualize_jacobian_maps.py \
    --ckpt-ctcf  /home/roman/P/CTCF/results/CTCF_IXI_TUNED/best.pth \
    --ckpt-tmdca /home/roman/P/CTCF/results/TM_DCA_IXI/best.pth.tar \
    --ckpt-utsr  /home/roman/P/CTCF/results/UTSR_IXI_WREG4_E500/best.pth \
    --utsr-config UTSRMorph-IXI-Large \
    --ds IXI --use_test --3 --gpu 0 --case-index 0 --out results/figs/jacobian_ixi.png
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

from datasets import OASIS, IXI
from experiments.core.model_adapters import CtcfAdapter, TmDcaAdapter, UtsrMorphAdapter
from experiments.core.train_runtime import PATHS, add_common_args
from experiments.inference import load_checkpoint_state
from utils import NumpyType, SegNorm, jacobian_det, setup_device


MODEL_NAMES = ["CTCF", "TM-DCA", "UTSRMorph"]


def build_loader(ds_key: str, paths_profile: int, use_test: bool):
    ds_paths = PATHS[paths_profile][ds_key]
    test_dir = ds_paths["test_dir"] if (use_test and "test_dir" in ds_paths) else ds_paths["val_dir"]
    files = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))
    if not files:
        raise RuntimeError(f"No .pkl under {test_dir}")
    if ds_key == "OASIS":
        ds = OASIS.OASISBrainInferDataset(files, transforms=transforms.Compose([NumpyType((np.float32, np.int16))]))
    else:
        atlas = str(ds_paths["atlas_path"]).rstrip("/\\")
        ds = IXI.IXIBrainInferDataset(
            files, atlas, transforms=transforms.Compose([SegNorm(), NumpyType((np.float32, np.int16))])
        )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), files


def run_model(name: str, ckpt: str, config_key: str, x, y, device, time_steps: int, strict: bool):
    if name == "ctcf":
        adapter = CtcfAdapter()
        model = adapter.build(time_steps=time_steps, config_key=config_key).to(device).eval()
    elif name == "tm-dca":
        adapter = TmDcaAdapter()
        # TM-DCA checkpoints were trained with time_steps=12 (adapter default);
        # overriding to match CTCF's 6 would leave 30 unexpected decoder keys.
        model = adapter.build(config_key=config_key).to(device).eval()
    elif name == "utsrmorph":
        adapter = UtsrMorphAdapter()
        model = adapter.build(config_key=config_key).to(device).eval()
    else:
        raise ValueError(name)
    load_checkpoint_state(model, ckpt, strict=strict)
    with torch.no_grad():
        flow = adapter.forward(model, x, y)
    del model
    torch.cuda.empty_cache()
    return flow.float()


def ortho_slices(vol: np.ndarray):
    d, h, w = vol.shape
    return vol[d // 2], vol[:, h // 2, :], vol[:, :, w // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, mode="infer")
    ap.add_argument("--ckpt-ctcf", required=True)
    ap.add_argument("--ckpt-tmdca", required=True)
    ap.add_argument("--ckpt-utsr", required=True)
    ap.add_argument("--use_test", action="store_true")
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--strict_ckpt", type=int, default=0)
    ap.add_argument("--ctcf-config", default="CTCF-CascadeA")
    ap.add_argument("--tm-config", default="TransMorph-3-LVL")
    ap.add_argument("--utsr-config", default="UTSRMorph-Large")
    ap.add_argument("--time_steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=False, help="Output PNG path (not required when --rank_cases).")
    ap.add_argument("--clip", type=float, default=1.0, help="Colormap clip for log|det J| (symmetric).")
    ap.add_argument(
        "--rank_cases",
        action="store_true",
        help="Score every case on folds%% and SDlogJ for all 3 models, print a table, exit.",
    )
    ap.add_argument(
        "--rank_limit", type=int, default=0, help="Restrict --rank_cases to first N cases (0 = all). Useful for IXI."
    )
    args = ap.parse_args()

    device = setup_device(args.gpu, seed=args.seed, deterministic=False)
    loader, files = build_loader(args.ds, int(args.paths), args.use_test)

    ckpts = [
        ("ctcf", args.ckpt_ctcf, args.ctcf_config),
        ("tm-dca", args.ckpt_tmdca, args.tm_config),
        ("utsrmorph", args.ckpt_utsr, args.utsr_config),
    ]

    # ── Ranking mode: print per-case metrics for all 3 models, exit ───────
    if args.rank_cases:
        n_total = len(files) if args.rank_limit <= 0 else min(args.rank_limit, len(files))
        per_case: list[dict] = []
        for idx, b in enumerate(loader):
            if idx >= n_total:
                break
            cid = Path(files[idx]).stem
            xi, yi = b[0].to(device), b[1].to(device)
            row = {"idx": idx, "cid": cid}
            for name, ckpt, cfg in ckpts:
                flow = run_model(name, ckpt, cfg, xi, yi, device, args.time_steps, bool(args.strict_ckpt))
                det = jacobian_det(flow).detach().cpu().numpy()[0, 0]
                det_shift = np.clip(det + 3.0, 1e-6, None)
                row[f"{name}_folds"] = 100.0 * float((det <= 0).mean())
                row[f"{name}_sdlog"] = float(np.log(det_shift).std())
                del flow
                torch.cuda.empty_cache()
            # CTCF advantage: how much smaller are CTCF's folds vs mean of baselines
            row["ctcf_adv_folds"] = 0.5 * (row["tm-dca_folds"] + row["utsrmorph_folds"]) - row["ctcf_folds"]
            row["ctcf_adv_sdlog"] = 0.5 * (row["tm-dca_sdlog"] + row["utsrmorph_sdlog"]) - row["ctcf_sdlog"]
            per_case.append(row)
            print(
                f"  [{idx + 1}/{n_total}] {cid}  ctcf folds={row['ctcf_folds']:.3f} sdlog={row['ctcf_sdlog']:.4f}  "
                f"tm folds={row['tm-dca_folds']:.3f} sdlog={row['tm-dca_sdlog']:.4f}  "
                f"utsr folds={row['utsrmorph_folds']:.3f} sdlog={row['utsrmorph_sdlog']:.4f}"
            )

        per_case.sort(key=lambda r: -(r["ctcf_adv_folds"] + r["ctcf_adv_sdlog"]))
        print(f"\nTop cases ranked by CTCF topology advantage (folds% + SDlogJ vs mean of baselines):")
        print(
            f"{'rank':<6}{'idx':<6}{'case_id':<18} {'adv_folds':<11}{'adv_sdlog':<11}  | ctcf fld/sdl   tm fld/sdl   utsr fld/sdl"
        )
        print("-" * 110)
        for r, row in enumerate(per_case):
            print(
                f"{r + 1:<6}{row['idx']:<6}{row['cid']:<18} "
                f"{row['ctcf_adv_folds']:+.3f}      {row['ctcf_adv_sdlog']:+.4f}     | "
                f"{row['ctcf_folds']:.3f}/{row['ctcf_sdlog']:.4f}  "
                f"{row['tm-dca_folds']:.3f}/{row['tm-dca_sdlog']:.4f}  "
                f"{row['utsrmorph_folds']:.3f}/{row['utsrmorph_sdlog']:.4f}"
            )
        return

    if not args.out:
        raise SystemExit("--out is required when not using --rank_cases")

    # Grab the chosen case
    batch = None
    for idx, b in enumerate(loader):
        if idx == args.case_index:
            batch = b
            case_id = Path(files[idx]).stem
            break
    if batch is None:
        raise RuntimeError(f"case-index {args.case_index} out of range (N={len(files)})")
    x, y = batch[0].to(device), batch[1].to(device)

    # Collect log|det J| maps for all three methods
    logdet_maps = []
    stats = []
    for name, ckpt, cfg in ckpts:
        flow = run_model(name, ckpt, cfg, x, y, device, args.time_steps, bool(args.strict_ckpt))
        det = jacobian_det(flow)
        det_np = det.detach().cpu().numpy()[0, 0]
        # log of clamped det (det+3 is the shift used in SDlogJ)
        det_shift = np.clip(det_np + 3.0, 1e-6, None)
        logdet = np.log(det_shift) - np.log(3.0)  # re-center so untransformed region (det=1) ~ 0
        logdet_maps.append(logdet)
        neg_pct = 100.0 * float((det_np <= 0).mean())
        sdlog = float(np.log(det_shift).std())
        stats.append({"folds_pct": neg_pct, "sdlogj": sdlog})
        print(f"  {name:<10} folds%={neg_pct:.3f}  SDlogJ={sdlog:.4f}")
        del flow
        torch.cuda.empty_cache()

    # Unified colormap: symmetric around 0, clipped to +-args.clip
    vmax = float(args.clip)
    vmin = -vmax

    view_names = ["axial", "coronal", "sagittal"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.5))
    im_last = None
    for r, (nm, lgd) in enumerate(zip(MODEL_NAMES, logdet_maps)):
        slices = ortho_slices(lgd)
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(slices[c], cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(view_names[c], fontsize=10)
            if c == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f"{nm}\nfolds={stats[r]['folds_pct']:.2f}%\nSDlogJ={stats[r]['sdlogj']:.3f}",
                    transform=ax.transAxes,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
            im_last = im

    fig.tight_layout(rect=(0.06, 0.0, 0.92, 1.0))

    cbar_ax = fig.add_axes([0.935, 0.1, 0.015, 0.8])
    fig.colorbar(im_last, cax=cbar_ax, label="log|det J|")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pil_kwargs={"compress_level": 1})
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
