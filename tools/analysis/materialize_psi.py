"""Materialize the source-index pull map Psi of the checkpoint-compatible sampler as a float32 displacement field.

The CTCF SpatialTransformer normalizes by (N_j-1) and samples with align_corners=False, so a stored canonical flow
u is read at Psi_j = N_j/(N_j-1) * Phi_j - 1/2 = A o Phi, with Phi(x) = x + u(x) and A an orientation-preserving
diagonal affine (the transform of tools/analysis/field_audit.py:_materialize_ctcf_legacy_sampler).
With --report every saved field is reloaded and checked twice on the same bytes: utils.cert_exact decides the
cellwise Bernstein predicate of its trilinear interpolant, and utils.cert_boundary decides whether its boundary
values are separable and strictly monotone. Only both together give a global homeomorphism of that interpolant
onto a box close to A(Omega); the materialized boundary equals A only up to float32 rounding, so failing the
boundary check proves nothing either way. Neither verifies the coordinate arithmetic inside the grid_sample kernel.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import certify_flow_exact
from utils.field import legacy_sampling_displacement
from utils.field_io import load_flow_npz, save_flow_npz_atomic


def materialize_psi_displacement(flow: torch.Tensor) -> torch.Tensor:
    """Phi displacement u [1,3,D,H,W] (voxel units, z,y,x) -> Psi displacement u_Psi = Psi - x."""
    return legacy_sampling_displacement(flow)


def certify_stored(path: Path, eps: str, max_exact: int) -> dict:
    """Cell certificate, boundary check and their combination for the bytes stored at ``path``."""
    flow = load_flow_npz(path)
    cell = certify_flow_exact(flow, eps=eps, max_exact=max_exact)
    boundary = boundary_report(flow)
    return {"file": str(path), **global_verdict(cell, boundary), "cell": cell, "boundary": boundary}


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize the deployed (align_corners=False) Psi displacement.")
    ap.add_argument("--flows", required=True, help="Directory or glob of canonical Phi flow .npz files.")
    ap.add_argument("--out", required=True, help="Output directory for the materialized Psi .npz files.")
    ap.add_argument("--report", help="Also certify every saved Psi field and write the JSON reports here.")
    ap.add_argument("--eps", default="0.001", help="Exact rational threshold of the cell predicate.")
    ap.add_argument("--max-exact", type=int, default=100_000, help="Ambiguous cells decided exactly per field.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.flows, "*.npz")) if os.path.isdir(args.flows) else glob.glob(args.flows))
    if not paths:
        raise SystemExit(f"no .npz flows matched {args.flows!r}")
    os.makedirs(args.out, exist_ok=True)
    reports = []
    for p in paths:
        with np.load(p) as d:
            arr = d["flow"]
        if arr.dtype != np.float32:
            raise TypeError(f"{p}: stored flow must be float32, got {arr.dtype}")
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.dim() == 4:
            t = t.unsqueeze(0)
        target = Path(args.out) / os.path.basename(p)
        save_flow_npz_atomic(target, materialize_psi_displacement(t))
        if args.report:
            reports.append(certify_stored(target, args.eps, args.max_exact))
            print(f"{reports[-1]['status']:<28} {target}")
    print(f"materialized {len(paths)} Psi displacement fields -> {args.out}")
    if not args.report:
        return 0
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
    if all(report["global_homeomorphism"] for report in reports):
        return 0
    return 2 if any(report["status"] == "INCONCLUSIVE_RESOURCE_LIMIT" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
