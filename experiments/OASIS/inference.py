#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified inference for OASIS Task-3 for 3 models in this repo:
  - TM-DCA  (models.TransMorph_DCA.model)
  - UTSRMorph (models.UTSRMorph.model)
  - CTCF    (models.CTCF.model)

Outputs (per run):
  out_dir/
    per_case.csv               # per-case metrics (+ per-label dice columns)
    summary.json               # mean/std/sem/ci95 + metadata
    summary.csv                # same but CSV
    png/                       # optional visuals
    flows/                     # optional displacement fields (.npz)

Metric set (recommended for paper):
  - Dice mean over VOI labels 1..35 (and per-label dice columns)
  - Fold% (detJ <= 0) percent
  - std(log(detJ)) (smoothness / non-physicality proxy)
  - optional HD95 mean over labels 1..35 (surface-distance)
  - time per case (seconds)

IMPORTANT:
  - Uses the SAME forward adapters as in your training scripts:
      * TM-DCA: half-res inference, then upsample flow x2 and multiply by 2
      * CTCF:   same
      * UTSRMorph: concat(x,y) -> model -> flow
"""

from __future__ import annotations

import os
import glob
import json
import time
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from experiments.OASIS import datasets

# repo utils (you already use these in training scripts)
from utils import (
    pkload,
    NumpyType,
    register_model,
    dice_val_VOI,
    jacobian_det,
    setup_device,
)

# ------------------------------ Helpers ------------------------------ #

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_checkpoint(path_or_dir: str, prefer_best: bool = True) -> str:
    """
    Accepts:
      - path to checkpoint file
      - OR path to directory containing checkpoints

    Priority:
      1) "best" in filename (if prefer_best=True)
      2) otherwise newest by modification time
    """
    if os.path.isfile(path_or_dir):
        return path_or_dir

    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"Checkpoint path not found: {path_or_dir}")

    exts = (".pth", ".pt", ".pth.tar")
    files = [
        os.path.join(path_or_dir, f)
        for f in os.listdir(path_or_dir)
        if f.endswith(exts)
    ]
    if not files:
        raise RuntimeError(f"No checkpoint files found in directory: {path_or_dir}")

    if prefer_best:
        best = [f for f in files if "best" in os.path.basename(f).lower()]
        if best:
            best.sort()
            return best[-1]

    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def load_state_dict(model: torch.nn.Module, ckpt_path: str) -> Dict:
    """
    Loads ckpt in common formats:
      - {"state_dict": ...}
      - {"model": ...}
      - raw state_dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            # Might already be a state dict-like mapping
            sd = ckpt
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    return ckpt if isinstance(ckpt, dict) else {"state_dict": sd}


def case_id_from_path(pkl_path: str) -> str:
    base = os.path.basename(pkl_path)
    # In original baselines they did [2:] after split('.')[0] (strip "p_")
    stem = os.path.splitext(base)[0]
    # keep both forms safe
    return stem[2:] if stem.startswith("p_") else stem


def dice_per_label_1to35(def_seg: torch.Tensor, y_seg: torch.Tensor) -> np.ndarray:
    """
    Compute per-label Dice for labels 1..35 on CPU numpy.
    Inputs are [B,1,D,H,W] integer tensors.
    Returns shape (35,) float64.
    """
    pred = def_seg.detach().cpu().numpy()[0, 0]
    true = y_seg.detach().cpu().numpy()[0, 0]
    out = np.zeros((35,), dtype=np.float64)
    for i, lbl in enumerate(range(1, 36)):
        p = (pred == lbl)
        t = (true == lbl)
        inter = np.sum(p & t)
        union = np.sum(p) + np.sum(t)
        out[i] = (2.0 * inter) / (union + 1e-5)
    return out


def fold_percent_from_flow(flow: torch.Tensor) -> float:
    detJ = jacobian_det(flow.float())  # [B,1,D,H,W]
    return float((detJ <= 0.0).float().mean().item() * 100.0)


def logdet_std_from_flow(flow: torch.Tensor) -> float:
    detJ = jacobian_det(flow.float())  # [B,1,D,H,W]
    # clamp to avoid log(<=0)
    detJ = torch.clamp(detJ, min=1e-9, max=1e9)
    logdet = torch.log(detJ)
    return float(torch.std(logdet).item())


def compute_ci95(mean: float, std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    sem = std / math.sqrt(n)
    return 1.96 * sem


def require_surface_distance():
    try:
        from surface_distance import compute_robust_hausdorff, compute_surface_distances  # noqa
    except Exception as e:
        raise RuntimeError(
            "HD95 requested but 'surface-distance' is not installed.\n"
            "Install it in your environment:\n"
            "  pip install surface-distance\n"
        ) from e


def hd95_mean_1to35(def_seg: torch.Tensor, y_seg: torch.Tensor, spacing=(1.0, 1.0, 1.0)) -> float:
    """
    Mean HD95 over labels 1..35 using surface-distance.
    If either label absent in pred or gt -> hd for that label = 0 (matches many baseline scripts).
    """
    require_surface_distance()
    from surface_distance import compute_robust_hausdorff, compute_surface_distances

    pred = def_seg.detach().cpu().numpy()[0, 0]
    true = y_seg.detach().cpu().numpy()[0, 0]

    hds = []
    for lbl in range(1, 36):
        p = (pred == lbl)
        t = (true == lbl)
        if p.sum() == 0 or t.sum() == 0:
            hds.append(0.0)
        else:
            sd = compute_surface_distances(t, p, spacing)
            hd = compute_robust_hausdorff(sd, 95.0)
            hds.append(float(hd))
    return float(np.mean(hds))


def save_flow_npz(flow: torch.Tensor, path: str):
    # store float16 to reduce size, same idea as baselines
    arr = flow.detach().cpu().numpy().astype(np.float16)
    np.savez_compressed(path, flow=arr)


def save_png_triplet(out_png: str, x: torch.Tensor, y: torch.Tensor, x_seg: torch.Tensor, y_seg: torch.Tensor,
                     def_seg: torch.Tensor, def_grid: Optional[torch.Tensor] = None):
    """
    Minimal but paper-useful visualization:
      3 orthogonal slices for: fixed(y), moving(x), fixed_seg(y_seg), warped_seg(def_seg), optional def_grid
    """
    import matplotlib.pyplot as plt

    # tensors are [B,1,D,H,W]
    x = x.detach().cpu().numpy()[0, 0]
    y = y.detach().cpu().numpy()[0, 0]
    xs = x_seg.detach().cpu().numpy()[0, 0]
    ys = y_seg.detach().cpu().numpy()[0, 0]
    ds = def_seg.detach().cpu().numpy()[0, 0]
    dg = None
    if def_grid is not None:
        dg = def_grid.detach().cpu().numpy()[0, 0]

    D, H, W = y.shape
    cz, cy, cx = D // 2, H // 2, W // 2

    # (axial=z fixed), (coronal=y), (sagittal=x)
    def slices(vol):
        return (vol[cz, :, :], vol[:, cy, :], vol[:, :, cx])

    y_ax, y_cor, y_sag = slices(y)
    x_ax, x_cor, x_sag = slices(x)
    ys_ax, ys_cor, ys_sag = slices(ys)
    ds_ax, ds_cor, ds_sag = slices(ds)
    if dg is not None:
        dg_ax, dg_cor, dg_sag = slices(dg)

    fig = plt.figure(figsize=(12, 10))

    # Row 1: fixed/moving
    axs = []
    axs.append(fig.add_subplot(4, 3, 1)); axs[-1].imshow(y_ax, cmap="gray"); axs[-1].set_title("Fixed (ax)")
    axs.append(fig.add_subplot(4, 3, 2)); axs[-1].imshow(y_cor, cmap="gray"); axs[-1].set_title("Fixed (cor)")
    axs.append(fig.add_subplot(4, 3, 3)); axs[-1].imshow(y_sag, cmap="gray"); axs[-1].set_title("Fixed (sag)")

    axs.append(fig.add_subplot(4, 3, 4)); axs[-1].imshow(x_ax, cmap="gray"); axs[-1].set_title("Moving (ax)")
    axs.append(fig.add_subplot(4, 3, 5)); axs[-1].imshow(x_cor, cmap="gray"); axs[-1].set_title("Moving (cor)")
    axs.append(fig.add_subplot(4, 3, 6)); axs[-1].imshow(x_sag, cmap="gray"); axs[-1].set_title("Moving (sag)")

    # Row 3: segs
    axs.append(fig.add_subplot(4, 3, 7)); axs[-1].imshow(ys_ax); axs[-1].set_title("Fixed seg (ax)")
    axs.append(fig.add_subplot(4, 3, 8)); axs[-1].imshow(ys_cor); axs[-1].set_title("Fixed seg (cor)")
    axs.append(fig.add_subplot(4, 3, 9)); axs[-1].imshow(ys_sag); axs[-1].set_title("Fixed seg (sag)")

    # Row 4: warped seg (+ grid if available)
    axs.append(fig.add_subplot(4, 3, 10)); axs[-1].imshow(ds_ax); axs[-1].set_title("Warped seg (ax)")
    axs.append(fig.add_subplot(4, 3, 11)); axs[-1].imshow(ds_cor); axs[-1].set_title("Warped seg (cor)")
    axs.append(fig.add_subplot(4, 3, 12)); axs[-1].imshow(ds_sag); axs[-1].set_title("Warped seg (sag)")

    for a in axs:
        a.axis("off")

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ------------------------------ Model builders + forward adapters ------------------------------ #

@dataclass
class ModelBundle:
    name: str
    model: torch.nn.Module
    forward_flow_fn: callable


def build_tm_dca(time_steps: int, vol_size=(160, 192, 224), dwin=(7, 5, 3)) -> ModelBundle:
    from models.TransMorph_DCA.model import CONFIGS as CONFIGS_TM
    import models.TransMorph_DCA.model as TransMorph

    D, H, W = vol_size
    half_size = (D // 2, H // 2, W // 2)

    config = CONFIGS_TM["TransMorph-3-LVL"]
    config.img_size = half_size
    config.dwin_kernel_size = tuple(dwin)
    config.window_size = (D // 32, H // 32, W // 32)

    model = TransMorph.TransMorphCascadeAd(config, time_steps)

    @torch.no_grad()
    def forward_flow(model_, x, y):
        x_half = F.avg_pool3d(x, 2)
        y_half = F.avg_pool3d(y, 2)
        use_amp = torch.cuda.is_available()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            flow_half = model_((x_half, y_half))
        flow_full = F.interpolate(flow_half.float(), scale_factor=2, mode="trilinear", align_corners=False) * 2.0
        return flow_full

    return ModelBundle("tm-dca", model, forward_flow)


def build_ctcf(time_steps: int, vol_size=(160, 192, 224), dwin=(7, 5, 3), config_key="CTCF-3-LVL") -> ModelBundle:
    from models.CTCF.model import CONFIGS as CONFIGS_CTCF
    import models.CTCF.model as CTCF

    D, H, W = vol_size
    half_size = (D // 2, H // 2, W // 2)

    if config_key not in CONFIGS_CTCF:
        raise KeyError(f"Unknown CTCF config '{config_key}'. Available: {list(CONFIGS_CTCF.keys())}")

    config = CONFIGS_CTCF[config_key]
    config.img_size = half_size
    # if your CTCF config also uses dwin/window_size, keep parity with TM-DCA
    if hasattr(config, "dwin_kernel_size"):
        config.dwin_kernel_size = tuple(dwin)
    if hasattr(config, "window_size"):
        config.window_size = (D // 32, H // 32, W // 32)

    # In your code, main class is CTCF_DCA_SR
    if hasattr(CTCF, "CTCF_DCA_SR"):
        model = CTCF.CTCF_DCA_SR(config, time_steps)
    else:
        raise AttributeError("models.CTCF.model does not expose CTCF_DCA_SR; check your model module.")

    @torch.no_grad()
    def forward_flow(model_, x, y):
        x_half = F.avg_pool3d(x, 2)
        y_half = F.avg_pool3d(y, 2)
        use_amp = torch.cuda.is_available()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out_h, flow_h = model_((x_half, y_half))
        flow = F.interpolate(flow_h.float(), scale_factor=2, mode="trilinear", align_corners=False) * 2.0
        return flow

    return ModelBundle("ctcf", model, forward_flow)


def build_utsrmorph(config_key: str = "UTSRMorph-Large") -> ModelBundle:
    from models.UTSRMorph.model import CONFIGS as CONFIGS_UM, UTSRMorph

    if config_key not in CONFIGS_UM:
        raise KeyError(f"Unknown UTSRMorph config '{config_key}'. Available: {list(CONFIGS_UM.keys())}")

    config = CONFIGS_UM[config_key]
    model = UTSRMorph(config)

    @torch.no_grad()
    def forward_flow(model_, x, y):
        inp = torch.cat((x, y), dim=1)
        use_amp = torch.cuda.is_available()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, flow = model_(inp)
        return flow

    return ModelBundle("utsrmorph", model, forward_flow)


def build_model(model_name: str, args) -> ModelBundle:
    model_name = model_name.lower().strip()
    if model_name in ("tm-dca", "tm_dca", "tmdca"):
        return build_tm_dca(time_steps=args.time_steps, vol_size=args.vol_size, dwin=args.dwin)
    if model_name in ("ctcf",):
        return build_ctcf(time_steps=args.time_steps, vol_size=args.vol_size, dwin=args.dwin, config_key=args.ctcf_config)
    if model_name in ("utsrmorph", "utsr"):
        return build_utsrmorph(config_key=args.utsr_config)
    raise ValueError("Unknown --model. Use one of: tm-dca, utsrmorph, ctcf")


# ------------------------------ Inference ------------------------------ #

def run_inference(args):
    # device setup (consistent with training scripts)
    dev = setup_device(args.gpu, seed=args.seed, deterministic=args.deterministic)
    device = dev.device

    test_dir = args.test_dir
    if not test_dir.endswith(os.sep):
        test_dir += os.sep

    out_dir = ensure_dir(args.out_dir)
    png_dir = ensure_dir(os.path.join(out_dir, "png")) if args.save_pngs else None
    flow_dir = ensure_dir(os.path.join(out_dir, "flows")) if args.save_flow else None

    # dataset
    test_files = sorted(glob.glob(test_dir + "*.pkl"))
    if not test_files:
        raise RuntimeError(f"No .pkl files found in test_dir: {test_dir}")

    test_tf = transforms.Compose([NumpyType((np.float32, np.int16))])
    test_set = datasets.OASISBrainInferDataset(test_files, transforms=test_tf)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    bundle = build_model(args.model, args)
    model = bundle.model.to(device)
    forward_flow_fn = bundle.forward_flow_fn

    ckpt_path = resolve_checkpoint(args.ckpt, prefer_best=not args.prefer_last)
    print(f"[INFO] Model: {bundle.name}")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    ckpt = load_state_dict(model, ckpt_path)
    model.eval()

    # warpers (created lazily once we know vol_shape)
    reg_nearest = None
    reg_bilin = None

    # grid for def_grid (optional)
    mk_grid_img_fn = None
    if args.save_pngs:
        # you already have mk_grid_img in utils used in training scripts
        from utils import mk_grid_img as _mk_grid_img
        mk_grid_img_fn = _mk_grid_img

    # output CSV
    per_case_path = os.path.join(out_dir, "per_case.csv")
    header = (
        ["case_id", "dice_mean", "fold_percent", "logdet_std", "time_sec"]
        + (["hd95_mean"] if args.hd95 else [])
        + [f"dice_lbl_{i}" for i in range(1, 36)]
    )

    rows = []
    times = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x, y, x_seg, y_seg = [t.to(device, non_blocking=True) for t in batch]
            vol_shape = tuple(x.shape[2:])  # (D,H,W)

            if reg_nearest is None:
                reg_nearest = register_model(vol_shape, mode="nearest").to(device)
                reg_bilin = register_model(vol_shape, mode="bilinear").to(device)

            cid = case_id_from_path(test_files[idx])

            t0 = time.perf_counter()
            flow = forward_flow_fn(model, x, y)  # [1,3,D,H,W]
            # warp seg (nearest)
            def_seg = reg_nearest([x_seg.float(), flow.float()])
            dt = time.perf_counter() - t0
            times.append(dt)

            dice_mean = float(dice_val_VOI(def_seg.long(), y_seg.long()).item())
            dice_lbl = dice_per_label_1to35(def_seg.long(), y_seg.long())
            foldp = fold_percent_from_flow(flow)
            logdet_std = logdet_std_from_flow(flow)

            row = {
                "case_id": cid,
                "dice_mean": dice_mean,
                "fold_percent": foldp,
                "logdet_std": logdet_std,
                "time_sec": dt,
            }

            if args.hd95:
                # spacing for OASIS typically treated as (1,1,1) in many baselines unless stated otherwise
                row["hd95_mean"] = hd95_mean_1to35(def_seg.long(), y_seg.long(), spacing=(1.0, 1.0, 1.0))

            # per-label dice columns
            for i in range(35):
                row[f"dice_lbl_{i+1}"] = float(dice_lbl[i])

            rows.append(row)

            # optional save flow
            if args.save_flow:
                save_flow_npz(flow, os.path.join(flow_dir, f"flow_{cid}.npz"))

            # optional pngs (save either all or only first N)
            if args.save_pngs and (args.png_limit < 0 or idx < args.png_limit):
                def_grid = None
                if mk_grid_img_fn is not None:
                    grid_img = mk_grid_img_fn(args.grid_step, args.line_thickness, vol_shape, device=device)
                    def_grid = reg_bilin([grid_img.float(), flow.float()])
                save_png_triplet(
                    out_png=os.path.join(png_dir, f"{cid}.png"),
                    x=x, y=y, x_seg=x_seg, y_seg=y_seg, def_seg=def_seg, def_grid=def_grid
                )

            if (idx + 1) % max(1, args.print_every) == 0:
                msg = f"[{idx+1:03d}/{len(test_loader):03d}] {cid} dice={dice_mean:.4f} fold%={foldp:.2f} time={dt:.3f}s"
                if args.hd95:
                    msg += f" hd95={row['hd95_mean']:.3f}"
                print(msg)

    # write per_case.csv
    with open(per_case_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary
    def agg(key: str) -> Tuple[float, float]:
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    n = len(rows)
    summary = {
        "model": bundle.name,
        "ckpt_path": ckpt_path,
        "test_dir": args.test_dir,
        "n_cases": n,
        "metrics": {},
    }

    for key in ["dice_mean", "fold_percent", "logdet_std", "time_sec"] + (["hd95_mean"] if args.hd95 else []):
        m, s = agg(key)
        sem = s / math.sqrt(n) if n > 1 else 0.0
        ci95 = compute_ci95(m, s, n)
        summary["metrics"][key] = {"mean": m, "std": s, "sem": sem, "ci95": ci95}

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_csv_path = os.path.join(out_dir, "summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "sem", "ci95"])
        for k, v in summary["metrics"].items():
            w.writerow([k, f"{v['mean']:.6f}", f"{v['std']:.6f}", f"{v['sem']:.6f}", f"{v['ci95']:.6f}"])

    print(f"\n[SAVED] per-case:  {per_case_path}")
    print(f"[SAVED] summary:   {summary_path}")
    print(f"[SAVED] summary:   {summary_csv_path}")
    if args.save_pngs:
        print(f"[SAVED] png dir:   {png_dir}")
    if args.save_flow:
        print(f"[SAVED] flow dir:  {flow_dir}")


# ------------------------------ Aggregation (optional, for paper table) ------------------------------ #

def run_aggregate(args):
    """
    Aggregate multiple model summaries into a single comparison CSV + LaTeX.
    Usage:
      python -m experiments.OASIS.infer_oasis aggregate \
        --runs TM-DCA=.../summary.json UTSRMorph=.../summary.json CTCF=.../summary.json
    """
    runs = []
    for item in args.runs:
        name, path = item.split("=", 1)
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        runs.append((name, s))

    metrics_order = ["dice_mean", "hd95_mean", "fold_percent", "logdet_std", "time_sec"]

    out_csv = args.out_csv
    out_tex = args.out_tex
    ensure_dir(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".")

    # CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["metric"] + [name for name, _ in runs]
        w.writerow(header)
        for m in metrics_order:
            row = [m]
            for name, s in runs:
                if m not in s["metrics"]:
                    row.append("")
                else:
                    v = s["metrics"][m]
                    row.append(f"{v['mean']:.4f} ± {v['std']:.4f} (CI95 ± {v['ci95']:.4f})")
            w.writerow(row)

    # LaTeX (mean ± std)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * len(runs) + "}\n")
        f.write("\\hline\n")
        f.write("Metric & " + " & ".join([name for name, _ in runs]) + " \\\\\n")
        f.write("\\hline\n")
        for m in metrics_order:
            line = [m]
            for name, s in runs:
                if m not in s["metrics"]:
                    line.append("--")
                else:
                    v = s["metrics"][m]
                    line.append(f"{v['mean']:.4f} $\\pm$ {v['std']:.4f}")
            f.write(" & ".join(line) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print(f"[SAVED] comparison csv:   {out_csv}")
    print(f"[SAVED] comparison latex: {out_tex}")


# ------------------------------ CLI ------------------------------ #

def build_parser():
    p = argparse.ArgumentParser(description="Unified OASIS inference for TM-DCA / UTSRMorph / CTCF")
    sub = p.add_subparsers(dest="cmd", required=True)

    # infer
    pinf = sub.add_parser("infer", help="Run inference for one model and export metrics/visuals")
    pinf.add_argument("--model", required=True, choices=["tm-dca", "utsrmorph", "ctcf"])
    pinf.add_argument("--ckpt", required=True, help="Checkpoint FILE or directory containing checkpoints")
    pinf.add_argument("--test_dir", required=True, help="Path to OASIS Test/*.pkl")
    pinf.add_argument("--out_dir", required=True, help="Output directory (will be created)")
    pinf.add_argument("--gpu", type=int, default=0)
    pinf.add_argument("--seed", type=int, default=0)
    pinf.add_argument("--deterministic", action="store_true")
    pinf.add_argument("--num_workers", type=int, default=4)
    pinf.add_argument("--print_every", type=int, default=1)
    pinf.add_argument("--prefer_last", action="store_true", help="Prefer last/newest instead of best")
    pinf.add_argument("--save_flow", action="store_true", help="Save flow fields as .npz")
    pinf.add_argument("--save_pngs", action="store_true", help="Save PNG visuals (requires matplotlib)")
    pinf.add_argument("--png_limit", type=int, default=5, help="How many cases to save PNG for (-1 = all)")
    pinf.add_argument("--grid_step", type=int, default=8)
    pinf.add_argument("--line_thickness", type=int, default=1)
    pinf.add_argument("--hd95", action="store_true", help="Compute HD95 mean over labels 1..35 (requires surface-distance)")

    # model-specific knobs (kept consistent with training scripts defaults)
    pinf.add_argument("--time_steps", type=int, default=12, help="TM-DCA/CTCF cascade steps")
    pinf.add_argument("--vol_size", type=int, nargs=3, default=[160, 192, 224], help="Volume size D H W")
    pinf.add_argument("--dwin", type=int, nargs=3, default=[7, 5, 3], help="TM-DCA/CTCF dwin kernel size")
    pinf.add_argument("--utsr_config", type=str, default="UTSRMorph-Large", help="UTSRMorph config key")
    pinf.add_argument("--ctcf_config", type=str, default="CTCF-3-LVL", help="CTCF config key")

    # aggregate
    pagg = sub.add_parser("aggregate", help="Aggregate multiple summary.json into comparison tables")
    pagg.add_argument("--runs", nargs="+", required=True, help="List like NAME=path/to/summary.json")
    pagg.add_argument("--out_csv", type=str, default="comparison_table.csv")
    pagg.add_argument("--out_tex", type=str, default="comparison_table_latex.txt")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "infer":
        run_inference(args)
    elif args.cmd == "aggregate":
        run_aggregate(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()