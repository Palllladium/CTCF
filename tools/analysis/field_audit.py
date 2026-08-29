"""Cross-method audit of the Bernstein screen, sampled determinants and optional repair.

Decoupled from our model classes: ingests saved displacement fields from any source via a small convention
adapter, so fields from unrelated methods can be audited on the same footing. It reports central-difference,
digital-10, sampled-trilinear and float64 Bernstein quantities without assuming in advance that a cross-method
gap exists.  The float64 quantities are operational screens; ``utils.cert_exact`` supplies the separate
machine-sound verdict on saved float32 bytes. A failed sufficient predicate is not itself a witnessed fold.

Flow CONTRACT (what an adapter must return): voxel-unit displacement, channel-first [1,3,D,H,W], axis order
(z,y,x), on the canonical voxel grid used by utils.field._warp (align_corners=True). A
source whose files differ (mm units, x,y,z order, normalized coords, phi vs u) needs its own adapter branch.

The checkpoint-compatible CTCF SpatialTransformer is a legacy exception: it uses align_corners=False with
shape-1 normalization. ``--source ctcf`` materializes that sampler's effective pull map before audit.

Usage (validated on our own saved fields):
  python tools/analysis/field_audit.py --flows results/infer/OASIS/ctcf/best/flows \
      --segs <OASIS_Test_dir> --source ctcf --ds OASIS --out results/audit/ctcf_oasis [--repair 1]
"""

import argparse
import csv
import glob
import hashlib
import json
import numbers
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.core.path_profiles import get_dataset_paths
from utils.common import pkload
from utils.dice import OASIS_VOI_LABELS, dice_per_label
from utils.field import (
    _warp,
    digital_fold_percent,
    jacobian_nonpositive_percent,
    trilinear_cert_bound,
    trilinear_fold_percent,
    trilinear_project,
)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _materialize_ctcf_legacy_sampler(flow: torch.Tensor) -> torch.Tensor:
    """Convert a stored CTCF flow to the source-index pull map actually sampled by SpatialTransformer."""
    _, _, d, h, w = flow.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, dtype=flow.dtype),
        torch.arange(h, dtype=flow.dtype),
        torch.arange(w, dtype=flow.dtype),
        indexing="ij",
    )
    grid = torch.stack((zz, yy, xx), dim=0).unsqueeze(0)
    scale = flow.new_tensor((d / (d - 1), h / (h - 1), w / (w - 1))).view(1, 3, 1, 1, 1)
    return scale * (grid + flow) - 0.5 - grid


def to_contract(flow_np: np.ndarray, source: str) -> torch.Tensor:
    """Canonicalize a saved field to the contract: voxel-unit displacement [1,3,D,H,W], (z,y,x). Add a
    branch per new source; never silently assume a convention (a wrong axis order or unit reads as folds)."""
    if flow_np.dtype != np.float32:
        raise TypeError(f"stored flow must be float32, got {flow_np.dtype}; refusing an implicit cast")
    t = torch.from_numpy(np.ascontiguousarray(flow_np))
    if t.dim() == 4:  # [3,D,H,W] -> add batch
        t = t[None]
    if t.dim() != 5 or t.shape[0] != 1 or t.shape[1] != 3 or min(t.shape[-3:]) < 2:
        raise ValueError(f"expected flow [1,3,D,H,W], got {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("flow contains NaN or Inf")
    if source == "canonical":
        return t
    if source == "ctcf":
        return _materialize_ctcf_legacy_sampler(t)
    raise NotImplementedError(
        f"source '{source}' has no adapter yet. Implement its conversion (units/axis-order/phi-vs-u) here, "
        f"and verify it against a KNOWN field before trusting the audit (anti-MIND-bug: no 'looks right')."
    )


def segs_for(stem: str, seg_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (x_seg, y_seg) for a flow whose stem encodes the pair, e.g. flow_0438_0439 -> p_0438_0439.pkl."""
    pair = stem.replace("flow_", "").lstrip("p_")
    pkl = os.path.join(seg_dir, f"p_{pair}.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"missing segmentation pair {pkl}")
    _, _, xs, ys = pkload(pkl)
    xs, ys = np.asarray(xs), np.asarray(ys)
    if xs.ndim != 3 or ys.ndim != 3 or xs.shape != ys.shape:
        raise ValueError(f"{pkl}: expected equal 3D segmentation shapes, got {xs.shape} and {ys.shape}")
    return (
        torch.from_numpy(np.ascontiguousarray(xs)[None, None]),
        torch.from_numpy(np.ascontiguousarray(ys)[None, None]),
    )


def _manifest_files(manifest_path: str, flow_dir: str) -> tuple[list[str], dict[str, dict]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != 2 or not manifest.get("cases"):
        raise ValueError(f"{manifest_path}: expected a non-empty schema-2 external-field manifest")
    cases = manifest["cases"]
    for pair, case in cases.items():
        pair_path = Path(case["pair_file"])
        if not pair_path.is_file():
            raise FileNotFoundError(f"{pair}: manifest pair file is missing: {pair_path}")
        pair_hash = hashlib.sha256(pair_path.read_bytes()).hexdigest()
        if pair_hash != case.get("pair_file_sha256"):
            raise RuntimeError(f"{pair}: segmentation/input pkl hash differs from manifest")
    files = [str(Path(cases[pair]["flow_file"])) for pair in sorted(cases)]
    expected = {str(Path(path).resolve()) for path in files}
    actual = {str(path.resolve()) for path in Path(flow_dir).glob("flow_*.npz")}
    if expected != actual:
        raise RuntimeError(
            f"manifest/flow-directory mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return files, cases


def _save_flow_atomic(path: Path, flow: torch.Tensor) -> tuple[str, str]:
    array = flow.detach().cpu().numpy().astype(np.float32, copy=False)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, flow=array)
    temporary.replace(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return _sha256_array(array), file_hash


def main():
    ap = argparse.ArgumentParser(description="Cross-method topology screen and optional repair")
    ap.add_argument("--flows", required=True, help="dir or glob of saved flow files (.npz with key 'flow', or .npy)")
    ap.add_argument("--manifest", help="schema-2 generator manifest; enforces exact case set and array hashes")
    ap.add_argument("--segs", default=None, help="dir of paired segmentation pkls (for the Dice columns)")
    ap.add_argument("--paths", type=int, choices=[1, 2, 3], help="derive --segs from this CTCF path profile")
    ap.add_argument(
        "--source",
        default="canonical",
        choices=["canonical", "ctcf"],
        help="canonical pull field, or stored CTCF field materialized to its deployed legacy sampler map",
    )
    ap.add_argument("--ds", default="OASIS", choices=["OASIS"])
    ap.add_argument("--repair", type=int, default=0, choices=[0, 1], help="also repair + Dice cost (slow)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--work-eps", type=float, default=1.1e-3, help="repair target; should exceed --eps")
    ap.add_argument("--out", default="results/audit/run")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of fields (0 = all)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (float64 cert is far faster on a strong-FP64 GPU)")
    args = ap.parse_args()

    if args.paths is not None:
        derived = get_dataset_paths(args.paths, "OASIS")["val_dir"]
        if args.segs is not None and Path(args.segs).resolve() != Path(derived).resolve():
            raise ValueError(f"--segs {args.segs} disagrees with profile {args.paths}: {derived}")
        args.segs = derived
    if args.segs is not None and not Path(args.segs).is_dir():
        raise FileNotFoundError(f"segmentation directory does not exist: {args.segs}")
    if args.repair and args.work_eps <= args.eps:
        raise ValueError("--work-eps must exceed --eps so float32 materialisation has a guard band")

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    manifest_cases = None
    if args.manifest:
        files, manifest_cases = _manifest_files(args.manifest, args.flows)
    else:
        files = sorted(glob.glob(args.flows if any(c in args.flows for c in "*?[") else os.path.join(args.flows, "*")))
        files = [f for f in files if f.endswith((".npz", ".npy"))]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npz/.npy flow files under {args.flows}")

    labels = OASIS_VOI_LABELS
    os.makedirs(args.out, exist_ok=True)
    repaired_dir = Path(args.out) / "repaired_flows"
    if args.repair:
        repaired_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, f in enumerate(files):
        if f.endswith(".npz"):
            with np.load(f, allow_pickle=False) as data:
                if set(data.files) != {"flow"}:
                    raise ValueError(f"{f}: expected only the 'flow' array, got {data.files}")
                stored = np.asarray(data["flow"])
        else:
            stored = np.load(f, allow_pickle=False)
        pair = Path(f).stem.removeprefix("flow_")
        if manifest_cases is not None:
            expected_hash = manifest_cases[pair]["row"]["flow_array_sha256"]
            actual_hash = _sha256_array(stored)
            if actual_hash != expected_hash:
                raise RuntimeError(f"{pair}: flow array hash differs from manifest")
        flow = to_contract(stored, args.source).to(dev)
        r = {
            "field": Path(f).stem,
            "source_array_sha256": _sha256_array(stored),
            # Report all schemes; the data, not the script, decides whether a discrepancy exists.
            "central_fold_pct": jacobian_nonpositive_percent(flow, crop=1),
            "digital10_fold_pct": float(digital_fold_percent(flow).item()),
            "sampled_tri_fold_pct": trilinear_fold_percent(flow),
            "tri_cert_bound": trilinear_cert_bound(flow, eps=args.eps),
        }
        r["bernstein_pass_float64"] = float(np.isfinite(r["tri_cert_bound"]) and r["tri_cert_bound"] >= args.eps)
        r["sampled_negative_float32"] = float(r["sampled_tri_fold_pct"] > 0.0)
        if r["bernstein_pass_float64"] and r["sampled_negative_float32"]:
            raise RuntimeError(f"{pair}: inconsistent operational screen (Bernstein pass with negative sample)")
        r["audit_state"] = (
            "PREDICATE_PASS_FLOAT64"
            if r["bernstein_pass_float64"]
            else "SAMPLED_NEGATIVE_FLOAT32"
            if r["sampled_negative_float32"]
            else "UNRESOLVED_FLOAT64"
        )
        segs = segs_for(Path(f).stem, args.segs) if args.segs else None
        if segs is not None:
            xs, ys = (s.to(dev) for s in segs)
            if tuple(xs.shape[-3:]) != tuple(flow.shape[-3:]):
                raise ValueError(f"{pair}: segmentation shape {tuple(xs.shape[-3:])} != flow {tuple(flow.shape[-3:])}")
            with torch.no_grad():
                d0 = float(
                    np.mean(dice_per_label(_warp(xs.float(), flow.float(), mode="nearest").long(), ys.long(), labels))
                )
            r["dice_feedfwd"] = d0
        if args.repair:
            rep, repair_report = trilinear_project(flow, eps=args.work_eps, max_iters=80)
            r["tri_cert_bound_repaired"] = trilinear_cert_bound(rep, eps=args.work_eps)
            r["repair_pass_float64"] = float(repair_report.certified)
            r["repair_uncertified_cells"] = float(repair_report.n_uncertified_cells)
            r["repair_iters"] = float(repair_report.iterations)
            array_hash, file_hash = _save_flow_atomic(repaired_dir / f"flow_{pair}.npz", rep)
            r["repaired_array_sha256"] = array_hash
            r["repaired_file_sha256"] = file_hash
            if segs is not None:
                with torch.no_grad():
                    d1 = float(
                        np.mean(
                            dice_per_label(_warp(xs.float(), rep.float(), mode="nearest").long(), ys.long(), labels)
                        )
                    )
                r["dice_repaired"] = d1
                r["dice_cost"] = d1 - r["dice_feedfwd"]
        rows.append(r)
        print(
            f"[{i + 1}/{len(files)}] {r['field']}: central={r['central_fold_pct']:.4f} "
            f"digital10={r['digital10_fold_pct']:.4f} sampled_tri={r['sampled_tri_fold_pct']:.4f} "
            f"cert={r['tri_cert_bound']:+.4f} {r['audit_state']}",
            flush=True,
        )

    if args.repair:
        expected_repaired = {f"flow_{Path(f).stem.removeprefix('flow_')}.npz" for f in files}
        actual_repaired = {path.name for path in repaired_dir.glob("flow_*.npz")}
        if expected_repaired != actual_repaired:
            raise RuntimeError(
                f"repaired field-set mismatch: missing={sorted(expected_repaired - actual_repaired)}, "
                f"extra={sorted(actual_repaired - expected_repaired)}"
            )

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "audit.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["field"] + [k for k in keys if k != "field"])
        w.writeheader()
        w.writerows(rows)

    numeric_keys = [
        k for k in keys if any(k in r for r in rows) and all(isinstance(r[k], numbers.Real) for r in rows if k in r)
    ]
    a = {k: np.mean([r[k] for r in rows if k in r]) for k in numeric_keys}
    print(f"\n===== audit ({args.source}/{args.ds}, n={len(rows)}) =====")
    print(f"central-diff  fold% mean = {a.get('central_fold_pct', float('nan')):.4f}")
    print(f"digital-10    fold% mean = {a.get('digital10_fold_pct', float('nan')):.4f}")
    print(f"sampled-tri   fold% mean = {a.get('sampled_tri_fold_pct', float('nan')):.4f}")
    print(f"float64 predicate pass   = {a.get('bernstein_pass_float64', 0.0) * 100:.1f}% of fields")
    states = {
        state: sum(r["audit_state"] == state for r in rows)
        for state in ("PREDICATE_PASS_FLOAT64", "SAMPLED_NEGATIVE_FLOAT32", "UNRESOLVED_FLOAT64")
    }
    print(f"states: {states}")
    if states["SAMPLED_NEGATIVE_FLOAT32"]:
        print("RESULT: float32 sampled determinants are negative in at least one field; inspect margins separately.")
    elif states["UNRESOLVED_FLOAT64"]:
        print("RESULT: some fields fail the float64 sufficient screen without a negative sampled value.")
    else:
        print("RESULT: every audited field passes the float64 Bernstein screen; exact reports remain authoritative.")
    if args.repair:
        print(
            f"repaired float64 pass    = {(np.mean([r['tri_cert_bound_repaired'] >= args.work_eps for r in rows])) * 100:.1f}%"
        )
        if "dice_cost" in a:
            print(f"repair Dice cost mean    = {a['dice_cost']:+.4f}")
    print(f"wrote {os.path.join(args.out, 'audit.csv')}")
    if args.repair and not all(r["repair_pass_float64"] == 1.0 for r in rows):
        raise RuntimeError("at least one repaired field failed the operational Bernstein work-margin screen")


if __name__ == "__main__":
    main()
