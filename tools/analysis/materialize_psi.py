"""Materialize the source-index pull map Psi of the checkpoint-compatible sampler as a float32 displacement field.

The CTCF SpatialTransformer normalizes by (N_j-1) and samples with align_corners=False, so a stored canonical flow
u is read at Psi_j = N_j/(N_j-1) * Phi_j - 1/2 = A o Phi, with Phi(x) = x + u(x) and A an orientation-preserving
diagonal affine (the transform of tools/analysis/field_audit.py:_materialize_ctcf_legacy_sampler).
utils.cert_exact on the output decides the cellwise Bernstein predicate for the trilinear interpolant of these
float32 values. That is not a global homeomorphism of the deployed map: the materialized boundary equals A only up
to float32 rounding, so the global statement rests on the zero-boundary Phi certificate and the exact affine A.
Nor does it verify the coordinate arithmetic inside the grid_sample kernel. Psi has an approximately affine (not
identity) boundary, so its report must NOT be run with --require-zero-boundary.
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
