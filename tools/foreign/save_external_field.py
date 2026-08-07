"""Canonical saver for EXTERNAL registration fields, so the deployed-warp topology audit can consume any
method's output uniformly. Run YOUR method (FireANTs / ConvexAdam / ...) to register a moving->fixed OASIS
pair, then call this to store the RAW displacement field + the method's OWN warped moving image + metadata.

We do NOT convert conventions here. The adapter on our side (field_audit) converts sign/axis-order/units/
align_corners and VALIDATES itself by checking warp(moving, adapted_field) ~= your warped_moving -- which is
why we need your warped image, not a promise about the convention. Storing raw + metadata keeps that check honest.

Usage (from .nii.gz, needs nibabel) -- one call per registered pair:
    python tools/foreign/save_external_field.py \
        --method fireants --pair 0440_0441 \
        --disp disp_0440_0441.nii.gz --moving mov_0440.nii.gz --fixed fix_0441.nii.gz \
        --warped warped_mov_0440_0441.nii.gz \
        --units voxel --axis_order DHW --sign fwd --align_corners true \
        --note "FireANTs SyN preset, disp = moving->fixed in voxels"

Or import save_field(...) and pass numpy arrays directly from your Python pipeline.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def save_field(out_dir: str, method: str, pair: str, disp: np.ndarray, moving: np.ndarray,
               fixed: np.ndarray, warped_moving: np.ndarray, units: str, axis_order: str,
               sign: str, align_corners: str, note: str = "") -> str:
    """Store one external field + images + metadata as results/foreign/<method>/<pair>.npz.

    disp: the method's RAW displacement field (any shape/convention — recorded, not converted).
    units: 'voxel' or 'mm'.  axis_order: e.g. 'DHW' or 'WHD'.  sign: 'fwd' (moving->fixed) or 'back'.
    align_corners: 'true'/'false'/'unknown'. note: free text (which function produced disp, presets, etc.).
    """
    dst = os.path.join(out_dir, method)
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, f"{pair}.npz")
    np.savez_compressed(
        path,
        disp=np.asarray(disp, dtype=np.float32),
        moving=np.asarray(moving, dtype=np.float32),
        fixed=np.asarray(fixed, dtype=np.float32),
        warped_moving=np.asarray(warped_moving, dtype=np.float32),
        meta=np.array({"method": method, "pair": pair, "units": units, "axis_order": axis_order,
                       "sign": sign, "align_corners": align_corners, "note": note,
                       "disp_shape": tuple(np.asarray(disp).shape)}, dtype=object),
    )
    print(f"[SAVED] {path}  disp{tuple(np.asarray(disp).shape)} units={units} axes={axis_order} sign={sign}")
    return path


def _load(path: str) -> np.ndarray:
    """Load a .nii/.nii.gz (nibabel) or .npy/.npz(single-array) into a numpy array."""
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib  # only needed for NIfTI inputs

        return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    if path.endswith(".npz"):
        with np.load(path) as d:
            return np.asarray(d[list(d.keys())[0]], dtype=np.float32)
    return np.asarray(np.load(path), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="results/foreign")
    ap.add_argument("--method", required=True, help="e.g. fireants, convexadam")
    ap.add_argument("--pair", required=True, help="e.g. 0440_0441 (moving_fixed), matching our OASIS test pairs")
    ap.add_argument("--disp", required=True, help="RAW displacement field (.nii.gz/.npy/.npz)")
    ap.add_argument("--moving", required=True)
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--warped", required=True, help="the method's OWN warped moving image (for convention check)")
    ap.add_argument("--units", required=True, choices=["voxel", "mm"])
    ap.add_argument("--axis_order", required=True, help="e.g. DHW or WHD")
    ap.add_argument("--sign", required=True, choices=["fwd", "back"])
    ap.add_argument("--align_corners", default="unknown", choices=["true", "false", "unknown"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    save_field(args.out_dir, args.method, args.pair, _load(args.disp), _load(args.moving),
               _load(args.fixed), _load(args.warped), args.units, args.axis_order, args.sign,
               args.align_corners, args.note)


if __name__ == "__main__":
    main()
