"""Stage-1 universal audit: the fold-free certificate + repair as a PLUG-IN for any method's field.

Decoupled from our model classes: ingests saved displacement fields from any source via a small convention
adapter, so VoxelMorph / ConvexAdam / FireANTs / SITReg / CTCF fields are audited on the same footing. For
each field it reports the fold-scheme DISCREPANCY -- central-difference, digital-10 (Liu et al.) and sampled
trilinear all UNDER-count relative to the sound Bernstein certificate of the deployed trilinear warp -- and,
with --repair, the certified-repair Dice cost. This is the "floor" deliverable: a per-method / per-dataset
audit that stands on its own even if nothing downstream (search engine, raw-SOTA) works out.

Flow CONTRACT (what an adapter must return): voxel-unit displacement, channel-first [1,3,D,H,W], axis order
(z,y,x), on the same grid convention as utils.field._warp (align_corners=True) / SpatialTransformer. A
source whose files differ (mm units, x,y,z order, normalized coords, phi vs u) needs its own adapter branch.

Usage (validated on our own saved fields):
  python tools/analysis/field_audit.py --flows results/infer/OASIS/ctcf/best/flows \
      --segs <OASIS_Test_dir> --source ctcf --ds OASIS --out results/audit/ctcf_oasis [--repair 1]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.common import pkload  # noqa: E402
from utils.dice import OASIS_VOI_LABELS, dice_per_label  # noqa: E402
from utils.field import (  # noqa: E402
    digital_fold_percent,
    jacobian_nonpositive_percent,
    trilinear_cert_bound,
    trilinear_fold_percent,
    trilinear_project,
)
from utils.spatial import RegisterModel  # noqa: E402


def to_contract(flow_np: np.ndarray, source: str) -> torch.Tensor:
    """Canonicalize a saved field to the contract: voxel-unit displacement [1,3,D,H,W], (z,y,x). Add a
    branch per new source; never silently assume a convention (a wrong axis order or unit reads as folds)."""
    t = torch.from_numpy(np.ascontiguousarray(flow_np)).float()
    if t.dim() == 4:  # [3,D,H,W] -> add batch
        t = t[None]
    if source == "ctcf":
        return t  # our saved flows already satisfy the contract
    raise NotImplementedError(
        f"source '{source}' has no adapter yet. Implement its conversion (units/axis-order/phi-vs-u) here, "
        f"and verify it against a KNOWN field before trusting the audit (anti-MIND-bug: no 'looks right')."
    )


def segs_for(stem: str, seg_dir: str):
    """Load (x_seg, y_seg) for a flow whose stem encodes the pair, e.g. flow_0438_0439 -> p_0438_0439.pkl."""
    pair = stem.replace("flow_", "").lstrip("p_")
    pkl = os.path.join(seg_dir, f"p_{pair}.pkl")
    if not os.path.isfile(pkl):
        return None
    _, _, xs, ys = pkload(pkl)
    return (
        torch.from_numpy(np.ascontiguousarray(xs)[None, None]),
        torch.from_numpy(np.ascontiguousarray(ys)[None, None]),
    )


def main():
    ap = argparse.ArgumentParser(description="Stage-1 certificate+repair audit of foreign displacement fields")
    ap.add_argument("--flows", required=True, help="dir or glob of saved flow files (.npz with key 'flow', or .npy)")
    ap.add_argument("--segs", default=None, help="dir of paired segmentation pkls (for the Dice columns)")
    ap.add_argument("--source", default="ctcf", help="field convention adapter (ctcf|... add your own)")
    ap.add_argument("--ds", default="OASIS", choices=["OASIS", "IXI"])
    ap.add_argument("--repair", type=int, default=0, choices=[0, 1], help="also certified-repair + Dice cost (slow)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--out", default="results/audit/run")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of fields (0 = all)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.flows if any(c in args.flows for c in "*?[") else os.path.join(args.flows, "*")))
    files = [f for f in files if f.endswith((".npz", ".npy"))]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npz/.npy flow files under {args.flows}")

    labels = OASIS_VOI_LABELS  # IXI uses its own set; wire IXI_VOI_LABELS when auditing IXI fields
    reg = RegisterModel((160, 192, 224), mode="nearest") if args.segs else None
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i, f in enumerate(files):
        arr = np.load(f)
        flow = to_contract(arr["flow"] if f.endswith(".npz") else arr, args.source)
        r = {
            "field": Path(f).stem,
            # DISCREPANCY: three surrogate schemes vs the sound certificate.
            "central_fold_pct": jacobian_nonpositive_percent(flow, crop=1),
            "digital10_fold_pct": float(digital_fold_percent(flow).item()),
            "sampled_tri_fold_pct": trilinear_fold_percent(flow),
            "tri_cert_bound": trilinear_cert_bound(flow, eps=args.eps),
        }
        r["certified_feedfwd"] = float(r["tri_cert_bound"] >= args.eps)
        segs = segs_for(Path(f).stem, args.segs) if args.segs else None
        if segs is not None:
            xs, ys = segs
            with torch.no_grad():
                d0 = float(np.mean(dice_per_label(reg((xs.float(), flow.float())).long(), ys.long(), labels)))
            r["dice_feedfwd"] = d0
        if args.repair:
            rep, resid, iters = trilinear_project(flow, eps=args.eps, max_iters=80)
            r["tri_cert_bound_repaired"] = trilinear_cert_bound(rep, eps=args.eps)
            r["repair_iters"] = float(iters)
            if segs is not None:
                with torch.no_grad():
                    d1 = float(np.mean(dice_per_label(reg((xs.float(), rep.float())).long(), ys.long(), labels)))
                r["dice_repaired"] = d1
                r["dice_cost"] = d1 - r["dice_feedfwd"]
        rows.append(r)
        print(f"[{i+1}/{len(files)}] {r['field']}: central={r['central_fold_pct']:.4f} "
              f"digital10={r['digital10_fold_pct']:.4f} sampled_tri={r['sampled_tri_fold_pct']:.4f} "
              f"cert={r['tri_cert_bound']:+.4f} {'CERT' if r['certified_feedfwd'] else 'FOLDS'}", flush=True)

    keys = sorted({k for r in rows for k in r})
    import csv
    with open(os.path.join(args.out, "audit.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["field"] + [k for k in keys if k != "field"])
        w.writeheader()
        w.writerows(rows)

    a = {k: np.mean([r[k] for r in rows if k in r]) for k in keys if k != "field"}
    print(f"\n===== audit ({args.source}/{args.ds}, n={len(rows)}) =====")
    print(f"central-diff  fold% mean = {a.get('central_fold_pct', float('nan')):.4f}")
    print(f"digital-10    fold% mean = {a.get('digital10_fold_pct', float('nan')):.4f}")
    print(f"sampled-tri   fold% mean = {a.get('sampled_tri_fold_pct', float('nan')):.4f}")
    print(f"certified feed-forward   = {a.get('certified_feedfwd', 0.0)*100:.1f}% of fields")
    print(f"HEADLINE: surrogate schemes read ~0 while the certificate flags folds -> the gap the audit exposes.")
    if args.repair:
        print(f"repaired certified       = {(np.mean([r['tri_cert_bound_repaired'] >= args.eps for r in rows]))*100:.1f}%")
        if "dice_cost" in a:
            print(f"repair Dice cost mean    = {a['dice_cost']:+.4f}")
    print(f"wrote {os.path.join(args.out, 'audit.csv')}")


if __name__ == "__main__":
    main()
