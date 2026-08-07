"""
Publication-quality qualitative comparison figure for registration paper.

Layout (3 rows x 5 cols, ALL 15 cells filled):
  Row 0:  Fixed img   | Moving img  | Checker CTCF    | Checker TM-DCA    | Checker UTSRMorph
  Row 1:  Fixed seg   | |F - M|     | |F-W| CTCF      | |F-W| TM-DCA     | |F-W| UTSRMorph
  Row 2:  Moving seg  | Warped CTCF | Def.grid CTCF   | Def.grid TM-DCA   | Def.grid UTSRMorph

Usage:
    python -m tools.paper.plot_qualitative_v2 --ds OASIS --case 0446_0447
    python -m tools.paper.plot_qualitative_v2 --ds IXI   --case subject_131
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as Fn


METHODS = [
    ("ctcf/best", "CTCF (ours)"),
    ("tm-dca/best.pth", "TM-DCA"),
    ("utsrmorph/best", "UTSRMorph"),
]

METHODS_FALLBACK = {
    "utsrmorph/best": "utsrmorph/best.pth",
}

DATASET_PATHS = {
    1: {
        "OASIS": {
            "test_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test",
        },
        "IXI": {
            "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Val",
            "test_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Test",
            "atlas_path": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/atlas.pkl",
        },
    },
}


def pkload(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def warp_tensor(src, flow):
    """Warp src (1,1,D,H,W) by flow (1,3,D,H,W)."""
    _, _, d, h, w = src.shape
    zz = torch.arange(d, dtype=torch.float32)
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=0).unsqueeze(0)
    new_locs = grid + flow
    new_locs[:, 0] = 2.0 * (new_locs[:, 0] / (d - 1) - 0.5)
    new_locs[:, 1] = 2.0 * (new_locs[:, 1] / (h - 1) - 0.5)
    new_locs[:, 2] = 2.0 * (new_locs[:, 2] / (w - 1) - 0.5)
    return Fn.grid_sample(src, new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]], mode="bilinear", align_corners=False)


def get_slice(vol, view, frac=0.5):
    d, h, w = vol.shape[:3]
    if view == "ax":
        return vol[int(d * frac)]
    if view == "cor":
        return vol[:, int(h * frac)]
    if view == "sag":
        return vol[:, :, int(w * frac)]


def _match_histogram(source, reference):
    """Match the intensity histogram of `source` to `reference` over nonzero
    voxels (skull-stripped brain region). Equalizes not only the dynamic range
    but also the shape of the distribution, so that a checkerboard overlay of
    (reference, matched_source) shows no brightness discontinuity at tile
    boundaries when the images are correctly aligned.

    Needed for IXI where the atlas (moving) and the subject (fixed) come from
    different pools and have different tissue-contrast profiles; percentile
    clipping alone does not equalize distribution shape.
    """
    src_mask = source > 0
    ref_mask = reference > 0
    if not (np.any(src_mask) and np.any(ref_mask)):
        return source.astype(np.float32)

    src_vals = source[src_mask].ravel().astype(np.float64)
    ref_vals = reference[ref_mask].ravel().astype(np.float64)
    ref_sorted = np.sort(ref_vals)

    # Rank-normalize source voxels to [0, 1], then look up the matching value
    # from the reference's sorted distribution.
    src_argsort = np.argsort(src_vals)
    src_rank = np.empty_like(src_argsort, dtype=np.float64)
    src_rank[src_argsort] = np.arange(len(src_vals)) / max(1, len(src_vals) - 1)
    ref_percentiles = np.linspace(0.0, 1.0, len(ref_sorted))
    matched = np.interp(src_rank, ref_percentiles, ref_sorted)

    result = source.astype(np.float32).copy()
    result[src_mask] = matched.astype(np.float32)
    return result


def _percentile_scale(img, lo_p=1.0, hi_p=99.0, ref=None):
    """Scale `img` to [0, 1] using percentiles from `ref` (or from img itself
    if ref is None), using nonzero voxels only."""
    base = ref if ref is not None else img
    pos = base[base > 0]
    if pos.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(pos, [lo_p, hi_p])
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def checkerboard(img_a, img_b, block=16):
    """Checkerboard overlay with histogram-matched intensities so that a
    correctly aligned registration produces no visible tile boundaries.
    `img_b` is histogram-matched to `img_a`; the pair is then scaled to [0, 1]
    using percentiles of `img_a`'s intensity distribution."""
    img_b_matched = _match_histogram(img_b, img_a)
    a_n = _percentile_scale(img_a)
    b_n = _percentile_scale(img_b_matched, ref=img_a)
    h, w = a_n.shape
    mask = ((np.arange(h)[:, None] // block) + (np.arange(w)[None, :] // block)) % 2
    return np.where(mask, a_n, b_n)


def seg_to_rgb(seg_slice):
    labels = np.unique(seg_slice)
    labels = labels[labels > 0]
    np.random.seed(42)
    rgb = np.zeros((*seg_slice.shape, 3), dtype=np.float32)
    for lb in labels:
        rgb[seg_slice == lb] = np.random.rand(3) * 0.7 + 0.3
    return rgb


def make_deformed_grid(flow_sl, grid_step=4, line_thickness=1):
    """Deformed grid from a 2D flow slice (H, W, C>=2)."""
    h, w = flow_sl.shape[:2]
    grid = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, grid_step):
        grid[max(0, i - line_thickness // 2) : min(h, i + line_thickness // 2 + 1), :] = 1.0
    for j in range(0, w, grid_step):
        grid[:, max(0, j - line_thickness // 2) : min(w, j + line_thickness // 2 + 1)] = 1.0

    grid_t = torch.from_numpy(grid[None, None]).float()
    fy = flow_sl[..., 0]
    fx = flow_sl[..., 1] if flow_sl.shape[-1] > 1 else np.zeros_like(fy)

    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    new_y = 2.0 * ((gy + torch.from_numpy(fy).float()) / (h - 1) - 0.5)
    new_x = 2.0 * ((gx + torch.from_numpy(fx).float()) / (w - 1) - 0.5)
    sample_grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
    warped = Fn.grid_sample(grid_t, sample_grid, mode="bilinear", align_corners=False)
    return warped[0, 0].numpy()


def load_case_data(ds, case_id, paths_profile=1, use_test=False):
    paths = DATASET_PATHS[paths_profile]
    if ds == "OASIS":
        x, y, x_seg, y_seg = pkload(os.path.join(paths["OASIS"]["test_dir"], f"p_{case_id}.pkl"))
        return dict(moving=x, fixed=y, moving_seg=x_seg.astype(np.int32), fixed_seg=y_seg.astype(np.int32))
    atlas_img, atlas_seg = pkload(paths["IXI"]["atlas_path"])
    ixi_dir = paths["IXI"]["test_dir"] if use_test else paths["IXI"]["val_dir"]
    subj_img, subj_seg = pkload(os.path.join(ixi_dir, f"{case_id}.pkl"))
    return dict(
        moving=atlas_img.astype(np.float32),
        fixed=subj_img.astype(np.float32),
        moving_seg=atlas_seg.astype(np.int32),
        fixed_seg=subj_seg.astype(np.int32),
    )


def load_flow(ds, method_dir, case_id):
    base = os.path.join("results", "infer", ds, method_dir, "flows")
    for prefix in [f"flow_{case_id}.npz", f"flow_p_{case_id}.npz"]:
        p = os.path.join(base, prefix)
        if os.path.exists(p):
            return np.load(p)["flow"][0]
    raise FileNotFoundError(f"Flow not found in {base} for {case_id}")


def make_figure(ds, case_id, view, out_path, checker_block, dpi, use_test=False):
    print(f"[INFO] ds={ds} case={case_id} view={view} dpi={dpi} use_test={use_test}")

    data = load_case_data(ds, case_id, use_test=use_test)
    mov, fix = data["moving"], data["fixed"]
    mov_seg, fix_seg = data["moving_seg"], data["fixed_seg"]
    fix_sl = get_slice(fix, view)
    mov_sl = get_slice(mov, view)
    fix_seg_sl = get_slice(fix_seg, view)
    mov_seg_sl = get_slice(mov_seg, view)

    # Per-method data
    mdata = {}
    for mdir, mname in METHODS:
        try:
            fl = load_flow(ds, mdir, case_id)
        except FileNotFoundError:
            fl = load_flow(ds, METHODS_FALLBACK.get(mdir, mdir), case_id)
        warped = warp_tensor(torch.from_numpy(mov[None, None]).float(), torch.from_numpy(fl[None]).float())[
            0, 0
        ].numpy()
        flow_sl = np.stack([get_slice(fl[c], view) for c in range(3)], axis=-1)
        mdata[mname] = dict(warped_sl=get_slice(warped, view), flow_sl=flow_sl)
        print(f"  [{mname}] ok")

    mnames = [n for _, n in METHODS]
    ctcf_name = mnames[0]

    # ── Style ──
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    vmax = max(fix_sl.max(), mov_sl.max())
    INTERP = "bilinear"
    GR = dict(cmap="gray", vmin=0, vmax=vmax, interpolation=INTERP, aspect="equal")
    # Checkerboard output is already percentile-normalized to [0, 1]; use a
    # matching display range so the tile populations appear at the same
    # brightness level regardless of dataset-specific intensity distributions.
    GR_CHK = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation=INTERP, aspect="equal")

    n_rows, n_cols = 3, 5
    fig = plt.figure(figsize=(3.2 * n_cols, 2.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.06, wspace=0.04)

    def mk(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        return a

    # ── Row 0: Fixed, Moving, Checkerboards ──
    a = mk(0, 0)
    a.imshow(fix_sl, **GR)
    a.set_title("Fixed", fontsize=11, fontweight="bold", pad=4)
    a = mk(0, 1)
    a.imshow(mov_sl, **GR)
    a.set_title("Moving", fontsize=11, fontweight="bold", pad=4)
    for j, nm in enumerate(mnames):
        a = mk(0, 2 + j)
        a.imshow(checkerboard(fix_sl, mdata[nm]["warped_sl"], block=checker_block), **GR_CHK)
        a.set_title(nm, fontsize=11, fontweight="bold", pad=4)

    # ── Row 1: Fixed seg, |F-M|, |F-W| per method ──
    diff_fm = np.abs(fix_sl.astype(np.float32) - mov_sl.astype(np.float32))
    diff_max = diff_fm.max()
    DM = dict(cmap="hot", vmin=0, vmax=diff_max, interpolation=INTERP, aspect="equal")

    a = mk(1, 0)
    a.imshow(seg_to_rgb(fix_seg_sl), interpolation=INTERP, aspect="equal")
    a.set_title("Fixed seg", fontsize=10, pad=4)
    a = mk(1, 1)
    a.imshow(diff_fm, **DM)
    a.set_title("|F \u2212 M|", fontsize=10, pad=4)
    for j, nm in enumerate(mnames):
        diff_fw = np.abs(fix_sl.astype(np.float32) - mdata[nm]["warped_sl"].astype(np.float32))
        a = mk(1, 2 + j)
        a.imshow(diff_fw, **DM)
        a.set_title("|F \u2212 W|", fontsize=10, pad=4)

    # ── Row 2: Moving seg, Warped CTCF, Deformed grids ──
    a = mk(2, 0)
    a.imshow(seg_to_rgb(mov_seg_sl), interpolation=INTERP, aspect="equal")
    a.set_title("Moving seg", fontsize=10, pad=4)
    a = mk(2, 1)
    a.imshow(mdata[ctcf_name]["warped_sl"], **GR)
    a.set_title("Warped (CTCF)", fontsize=10, pad=4)

    print("  Rendering deformed grids...")
    for j, nm in enumerate(mnames):
        defgrid = make_deformed_grid(mdata[nm]["flow_sl"], grid_step=4)
        a = mk(2, 2 + j)
        a.imshow(defgrid, cmap="gray", vmin=0, vmax=1, interpolation=INTERP, aspect="equal")
        a.set_title("Def. grid", fontsize=10, pad=4)

    # ── Save ──
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {out} ({dpi} DPI)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", choices=["OASIS", "IXI"], default="OASIS")
    p.add_argument("--case", default="auto")
    p.add_argument("--view", default="cor", choices=["ax", "cor", "sag"])
    p.add_argument("--out", default=None)
    p.add_argument("--checker_block", type=int, default=16)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--use_test", action="store_true", help="Use test_dir for IXI")
    a = p.parse_args()

    if a.out is None:
        a.out = f"figures/qualitative_{a.ds.lower()}_v2.png"
    make_figure(a.ds, a.case, a.view, a.out, a.checker_block, a.dpi, use_test=a.use_test)


if __name__ == "__main__":
    main()
