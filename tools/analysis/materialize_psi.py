"""Materialize the DEPLOYED source-index pull map Psi as a displacement field, so the artifact-bound
verifier (utils.cert_exact) certifies the map grid_sample actually applies, not only the canonical Phi.

The checkpoint-compatible CTCF SpatialTransformer combines normalization by (N_j-1) with align_corners=False.
The canonical certified map is Phi(x)=x+u(x); the effective sampler map is Psi_j = N_j/(N_j-1) * Phi_j - 1/2
= A o Phi with A an orientation-preserving diagonal affine (identical transform to
tools/analysis/field_audit.py:_materialize_ctcf_legacy_sampler). Since A preserves orientation,
det D Psi = (prod_j N_j/(N_j-1)) det D Phi > 0 iff Phi is fold-free, so a Psi certificate at eps carries the
same fold-free conclusion for the deployed coordinates. Psi has an AFFINE (not identity) boundary, so its
report must NOT be run with --require-zero-boundary.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.field import legacy_sampling_displacement


def materialize_psi_displacement(flow: torch.Tensor) -> torch.Tensor:
    """Phi displacement u [1,3,D,H,W] (voxel units, z,y,x) -> Psi displacement u_Psi = Psi - x."""
    return legacy_sampling_displacement(flow)


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize the deployed (align_corners=False) Psi displacement.")
    ap.add_argument("--flows", required=True, help="Directory or glob of canonical Phi flow .npz files.")
    ap.add_argument("--out", required=True, help="Output directory for the materialized Psi .npz files.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.flows, "*.npz")) if os.path.isdir(args.flows) else glob.glob(args.flows))
    if not paths:
        raise SystemExit(f"no .npz flows matched {args.flows!r}")
    os.makedirs(args.out, exist_ok=True)
    for p in paths:
        with np.load(p) as d:
            arr = d["flow"]
        if arr.dtype != np.float32:
            raise TypeError(f"{p}: stored flow must be float32, got {arr.dtype}")
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.dim() == 4:
            t = t.unsqueeze(0)
        psi = materialize_psi_displacement(t).numpy().astype(np.float32)  # [1,3,D,H,W]
        np.savez_compressed(os.path.join(args.out, os.path.basename(p)), flow=psi)
    print(f"materialized {len(paths)} Psi displacement fields -> {args.out}")


if __name__ == "__main__":
    main()
