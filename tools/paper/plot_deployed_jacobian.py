"""Deployed-warp Jacobian figure (Paper 3) — the topology story the Paper-1 log|det J| map cannot show.

Paper 1's map used the central-difference determinant under log|.|, which (a) is the wrong scheme (it does not
test the trilinear interpolant grid_sample deploys) and (b) HIDES the sign (a fold, det<0, is invisible under
|.|). This figure reads SAVED flow fields (produced by `experiments.inference --save_flow`) and renders, for
one slice, three panels per field:

  1. signed trilinear det J   -- the DEPLOYED grid_sample determinant (min over 5^3 samples per cell), on a
     diverging colormap centred at 0, so a fold (det<0) is a distinct RED region, not hidden by log|.|;
  2. witnessed-fold mask      -- cells the SOUND detector proves fold (min sampled trilinear det < 0);
  3. Bernstein certificate    -- the per-cell sound lower bound (the number we certify): red below eps.

Give --ff (feed-forward) and optionally --rep (the repaired / collared field) to get the before/after: the FF
row shows the folds the central log|det| map misses; the REP row shows them gone (mask empty, margin >= eps).

Usage (anywhere, no GPU -- reads saved fields):
  python tools/paper/plot_deployed_jacobian.py \
      --ff  results/jacfig_ff/flows/flow_0440_0441.npz \
      --rep results/jacfig_rep/flows/flow_0440_0441.npz \
      --eps 0.001 --out results/figs/deployed_jacobian_0440_0441.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from utils.field import _trilinear_cell_cert_bound, _trilinear_cell_min_det


def load_flow(path: str) -> torch.Tensor:
    """Load a saved flow .npz ([1,3,D,H,W] under key 'flow') as a float tensor."""
    with np.load(path) as d:
        arr = d["flow"]
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.dim() == 4:
        t = t.unsqueeze(0)
    if t.dim() != 5 or t.shape[1] != 3:
        raise ValueError(f"{path}: expected flow [1,3,D,H,W], got {tuple(t.shape)}")
    return t


def cell_maps(flow: torch.Tensor, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell worst sampled trilinear det and sound Bernstein lower bound, each [D-1,H-1,W-1]."""
    with torch.no_grad():
        det = _trilinear_cell_min_det(flow, samples=5).cpu().numpy()
        bnd = _trilinear_cell_cert_bound(flow, 0, eps).cpu().numpy()
    return det, bnd


def pick_slice(det: np.ndarray) -> int:
    """Axial (axis-0) slice with the most folded cells; middle slice if the field is fold-free."""
    folds_per_slice = (det < 0.0).sum(axis=(1, 2))
    return int(folds_per_slice.argmax()) if folds_per_slice.max() > 0 else det.shape[0] // 2


def _row(axes, det: np.ndarray, bnd: np.ndarray, sl: int, eps: float, label: str, det_clip: float) -> None:
    det_s, bnd_s = det[sl], bnd[sl]
    fold_pct = 100.0 * float((det < 0.0).mean())

    # 1) signed trilinear det J (diverging at 0): negative = fold = red
    ax = axes[0]
    ax.imshow(
        det_s, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-det_clip, vmax=det_clip), interpolation="nearest"
    )
    ax.set_ylabel(f"{label}\nfold%={fold_pct:.3f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # 2) witnessed-fold mask (cells with a proven negative sampled det)
    ax = axes[1]
    ax.imshow(det_s < 0.0, cmap="Reds", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    # 3) Bernstein certificate margin (sound lower bound), diverging at eps: below eps = not certified = red
    ax = axes[2]
    ax.imshow(
        bnd_s,
        cmap="RdBu",
        norm=TwoSlopeNorm(vcenter=eps, vmin=min(bnd_s.min(), -eps), vmax=max(bnd_s.max(), eps + 1e-6)),
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", required=True, help="Feed-forward flow .npz (row 1).")
    ap.add_argument("--rep", default=None, help="Repaired/collared flow .npz (row 2, optional).")
    ap.add_argument("--eps", type=float, default=0.001, help="Certificate margin for the Bernstein panel.")
    ap.add_argument("--slice", type=int, default=-1, help="Axial slice; -1 = the most-folded FF slice.")
    ap.add_argument("--det_clip", type=float, default=1.0, help="Symmetric colour clip for signed det J.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    args = ap.parse_args()

    ff_det, ff_bnd = cell_maps(load_flow(args.ff), args.eps)
    sl = pick_slice(ff_det) if args.slice < 0 else args.slice
    print(
        f"[slice] axial {sl} | FF fold%={100.0 * float((ff_det < 0).mean()):.4f} "
        f"min_det={ff_det.min():.4f} min_bnd={ff_bnd.min():.4f}"
    )

    rows = [("feed-forward", ff_det, ff_bnd)]
    if args.rep:
        rep_det, rep_bnd = cell_maps(load_flow(args.rep), args.eps)
        print(
            f"[rep] fold%={100.0 * float((rep_det < 0).mean()):.4f} "
            f"min_det={rep_det.min():.4f} min_bnd={rep_bnd.min():.4f}"
        )
        rows.append(("repaired (certified)", rep_det, rep_bnd))

    col_titles = ["signed trilinear det J", "witnessed folds", "Bernstein margin"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.2 * len(rows)), squeeze=False)
    for c, t in enumerate(col_titles):
        axes[0][c].set_title(t, fontsize=10)
    for r, (label, det, bnd) in enumerate(rows):
        _row(axes[r], det, bnd, sl, args.eps, label, args.det_clip)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
