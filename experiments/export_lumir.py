from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from datasets.LUMIR import LUMIR_INTENSITY_MAX, LUMIR_SHAPE, load_lumir_volume, lumir_disp_name, read_lumir_json
from experiments.core.cli_common import add_common_args
from experiments.core.cli_ctcf import add_ctcf_model_args, add_ctcf_override_args, ctcf_overrides_from_args
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.path_profiles import resolve_lumir_paths
from utils.field import legacy_sampling_displacement

# Which images/JSON a pair list comes from: the 2024 archive keeps its validation images next to imagesTr.
_PAIR_SOURCES = {"lumir24": ("train_root", "train_json"), "lumir25": ("val_root", "val_json")}


def evaluator_displacement(flow: torch.Tensor) -> np.ndarray:
    """[D,H,W,3] float32 displacement for the official evaluator (scipy map_coordinates, voxel units, array axes).

    It reproduces the map the checkpoint-compatible SpatialTransformer samples for a stored [1,3,D,H,W] flow.
    """
    if flow.ndim != 5 or flow.shape[:2] != (1, 3) or not bool(torch.isfinite(flow).all()):
        raise ValueError("export requires a finite [1,3,D,H,W] field")
    displacement = legacy_sampling_displacement(flow.float())[0].permute(1, 2, 3, 0)
    return displacement.cpu().numpy().astype(np.float32)


def save_displacement(path: Path, displacement: np.ndarray) -> None:
    """Write a [D,H,W,3] float32 displacement as NIfTI with identity affine and unit spacing, like the baselines."""
    if displacement.ndim != 4 or displacement.shape[-1] != 3 or displacement.dtype != np.float32:
        raise ValueError(f"expected a [D,H,W,3] float32 displacement, got {displacement.shape} {displacement.dtype}")
    nib.save(nib.Nifti1Image(displacement, np.eye(4)), path)


def _load_pair(root: Path, fixed: str, moving: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = load_lumir_volume(root / moving) / LUMIR_INTENSITY_MAX
    y = load_lumir_volume(root / fixed) / LUMIR_INTENSITY_MAX
    return torch.from_numpy(x)[None, None].to(device), torch.from_numpy(y)[None, None].to(device)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Export official LUMIR validation displacements of a CTCF checkpoint.")
    add_common_args(p, mode="infer", include_lumir=True)
    add_ctcf_model_args(p)
    add_ctcf_override_args(p)
    p.set_defaults(ds="LUMIR")
    p.add_argument("--ckpt", required=True, help="CTCF checkpoint, loaded strictly.")
    p.add_argument("--pairs", choices=sorted(_PAIR_SOURCES), required=True, help="Official pair list to export.")
    p.add_argument("--out", type=Path, required=True, help="Fresh folder; gets <name>/, <name>.zip, manifest.json.")
    p.add_argument("--name", default="predictions", help="Top-level folder inside the zip, as the evaluator expects.")
    p.add_argument("--disable_l1", type=int, choices=[0, 1], default=0)
    p.add_argument("--disable_l3", type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    if not args.name or args.name in (".", "..") or any(char in args.name for char in "/\\"):
        p.error("--name must be a single nonempty folder name")
    if args.out.exists():
        raise FileExistsError(f"use a fresh output folder: {args.out}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root_key, json_key = _PAIR_SOURCES[args.pairs]
    paths = resolve_lumir_paths(args, required=(root_key, json_key))
    _, pairs = read_lumir_json(paths[json_key])
    model = get_model_adapter("ctcf").build(
        time_steps=args.time_steps,
        config_key=args.config,
        img_size=LUMIR_SHAPE,
        dwin_size=volume_window_size(LUMIR_SHAPE),
        **ctcf_overrides_from_args(args),
    )
    load_report = load_checkpoint_state(model, args.ckpt, strict=True)
    model = model.to(device).eval()

    folder = args.out / args.name
    folder.mkdir(parents=True)
    files = []
    for fixed, moving in pairs:
        x, y = _load_pair(paths[root_key], fixed, moving, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, flow = model(x, y, alpha_l1=0.0 if args.disable_l1 else 1.0, alpha_l3=0.0 if args.disable_l3 else 1.0)
        target = folder / f"{lumir_disp_name(fixed, moving)}.nii.gz"
        save_displacement(target, evaluator_displacement(flow))
        files.append({"fixed": fixed, "moving": moving, "file": target.name, "sha256": _sha256(target)})
        print(f"{target.name} <- {Path(moving).name} onto {Path(fixed).name}")

    archive = args.out / f"{args.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for entry in files:
            handle.write(folder / entry["file"], arcname=f"{args.name}/{entry['file']}")
    manifest = {
        "checkpoint": args.ckpt,
        "checkpoint_load": load_report,
        "config": args.config,
        "time_steps": args.time_steps,
        "overrides": ctcf_overrides_from_args(args),
        "disable_l1": args.disable_l1,
        "disable_l3": args.disable_l3,
        "pairs": args.pairs,
        "pairs_json": str(paths[json_key]),
        "convention": "fixed->moving displacement in voxels along the stored array axes, [D,H,W,3] float32, "
        "as sampled by the checkpoint-compatible SpatialTransformer (legacy_sampling_displacement)",
        "archive": archive.name,
        "archive_sha256": _sha256(archive),
        "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(files)} displacements -> {archive}")


if __name__ == "__main__":
    main()
