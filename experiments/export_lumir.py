from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from datasets.LUMIR import (
    LUMIR_INTENSITY_MAX,
    LUMIR_SHAPE,
    labelled_t1_pairs,
    load_lumir_volume,
    lumir_disp_name,
    read_lumir_json,
)
from experiments.core.cli_common import add_common_args
from experiments.core.cli_ctcf import add_ctcf_model_args, add_ctcf_override_args, ctcf_overrides_from_args
from experiments.core.field_scoring import IDENTITY_BOUNDARY, PROFILE, REPAIR, repair_to_identity_boundary, warp_labels
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.path_profiles import resolve_lumir_paths
from experiments.core.precision import add_precision_arg, configure_precision, precision_autocast
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import certify_flow_exact
from utils.dice import dice_per_label
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


def _label_map(path: Path, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.rint(load_lumir_volume(path)).astype(np.int64))[None, None].to(device)


def _label_scores(flow: torch.Tensor, x_seg: torch.Tensor, y_seg: torch.Tensor) -> dict:
    labels = [int(value) for value in torch.unique(y_seg) if value > 0]
    dice = dice_per_label(warp_labels(x_seg, flow, "canonical").long(), y_seg.long(), labels=labels)
    return {"coarse_dice": float(np.mean(dice)), **PROFILE.jac_metrics(flow, x_seg)}


def certified_entry(
    target: Path, displacement: np.ndarray, fixed: str, moving: str, labels: tuple | None, device: torch.device
) -> dict:
    """Repair one canonical field to an identity boundary, write it, and certify the bytes reread from that file."""
    raw = torch.from_numpy(displacement).permute(3, 0, 1, 2)[None].to(device)
    started = time.perf_counter()
    repaired, columns = repair_to_identity_boundary(raw)
    seconds = time.perf_counter() - started
    save_displacement(target, repaired[0].permute(1, 2, 3, 0).cpu().numpy().astype(np.float32))
    stored = nib.load(target).get_fdata(dtype=np.float32)
    stored = torch.from_numpy(np.ascontiguousarray(stored)).permute(3, 0, 1, 2)[None]
    cell = certify_flow_exact(stored, eps=IDENTITY_BOUNDARY["claim_eps"])
    boundary = boundary_report(stored)
    change = (repaired - raw).abs()
    entry = {
        "fixed": fixed,
        "moving": moving,
        "file": target.name,
        "sha256": sha256_file(target),
        "repair": columns,
        "repair_seconds": seconds,
        "changed_voxel_fraction": float((change > 0).any(dim=1).float().mean()),
        "max_component_change_voxels": float(change.max()),
        "global": global_verdict(cell, boundary),
        "cell": cell,
        "boundary": boundary,
    }
    if labels is not None:
        entry["labelled"] = {"raw": _label_scores(raw, *labels), "certified": _label_scores(repaired, *labels)}
    return entry


def bind_recipe(out: Path, recipe: dict) -> None:
    """A resumed certified export reuses fields only from the same weights and settings."""
    path = out / "recipe.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(recipe)):
        raise ValueError(f"{out} belongs to another checkpoint or export recipe; use a fresh folder")
    atomic_write_json(path, recipe)


def main() -> None:
    p = argparse.ArgumentParser(description="Export official LUMIR validation displacements of a CTCF checkpoint.")
    add_common_args(p, mode="infer", include_lumir=True)
    add_precision_arg(p, default=None)
    add_ctcf_model_args(p)
    add_ctcf_override_args(p)
    p.set_defaults(ds="LUMIR")
    p.add_argument("--ckpt", required=True, help="CTCF checkpoint, loaded strictly.")
    p.add_argument("--pairs", choices=sorted(_PAIR_SOURCES), required=True, help="Official pair list to export.")
    p.add_argument("--out", type=Path, required=True, help="Fresh folder; gets <name>/, <name>.zip, manifest.json.")
    p.add_argument("--name", default="predictions", help="Top-level folder inside the zip, as the evaluator expects.")
    p.add_argument("--disable_l1", type=int, choices=[0, 1], default=0)
    p.add_argument("--disable_l3", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--certified",
        type=int,
        choices=[0, 1],
        default=0,
        help="Also repair to an identity boundary and certify the written bytes; a killed export resumes per pair.",
    )
    p.add_argument("--limit_pairs", type=int, default=0, help="Certified check on the first pairs, labelled first.")
    args = p.parse_args()
    if not args.name or args.name in (".", "..") or any(char in args.name for char in "/\\"):
        p.error("--name must be a single nonempty folder name")
    if args.limit_pairs and not args.certified:
        p.error("--limit_pairs is a check of the certified export")
    if args.out.exists() and not args.certified:
        raise FileExistsError(f"use a fresh output folder: {args.out}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root_key, json_key = _PAIR_SOURCES[args.pairs]
    paths = resolve_lumir_paths(args, required=(root_key, json_key))
    _, pairs = read_lumir_json(paths[json_key])
    labelled = (
        set(labelled_t1_pairs(pairs, paths["label_dir"])) if args.certified and paths["label_dir"].exists() else ()
    )
    model = get_model_adapter("ctcf").build(
        time_steps=args.time_steps,
        config_key=args.config,
        img_size=LUMIR_SHAPE,
        dwin_size=volume_window_size(LUMIR_SHAPE),
        **ctcf_overrides_from_args(args),
    )
    load_report = load_checkpoint_state(model, args.ckpt, strict=True)
    precision = args.precision or load_report["precision"]
    configure_precision(precision, device)
    model = model.to(device).eval()
    if args.limit_pairs:
        pairs = sorted(pairs, key=lambda pair: pair not in labelled)[: args.limit_pairs]
    recipe = {
        "checkpoint_sha256": sha256_file(Path(args.ckpt)),
        "precision": precision,
        "config": args.config,
        "time_steps": args.time_steps,
        "overrides": ctcf_overrides_from_args(args),
        "disable_l1": args.disable_l1,
        "disable_l3": args.disable_l3,
        "pairs": args.pairs,
        "pairs_sha256": sha256_file(paths[json_key]),
    }
    if args.certified:
        recipe |= {"identity_boundary": IDENTITY_BOUNDARY, "repair": REPAIR, "limit_pairs": args.limit_pairs}
        bind_recipe(args.out, recipe)

    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=bool(args.certified))
    certificates = args.out / "certificates"
    files = []
    for fixed, moving in pairs:
        target = folder / f"{lumir_disp_name(fixed, moving)}.nii.gz"
        report = certificates / f"{target.name}.json"
        if args.certified and report.exists() and target.exists():
            entry = json.loads(report.read_text(encoding="utf-8"))
            if entry["sha256"] != sha256_file(target):
                raise ValueError(f"certified field bytes changed: {target}")
            files.append({key: entry[key] for key in ("fixed", "moving", "file", "sha256")})
            continue
        x, y = _load_pair(paths[root_key], fixed, moving, device)
        with torch.no_grad(), precision_autocast(precision, device):
            _, flow = model(x, y, alpha_l1=0.0 if args.disable_l1 else 1.0, alpha_l3=0.0 if args.disable_l3 else 1.0)
        if args.certified:
            labels = None
            if (fixed, moving) in labelled:
                labels = (
                    _label_map(paths["label_dir"] / Path(moving).name, device),
                    _label_map(paths["label_dir"] / Path(fixed).name, device),
                )
            with torch.no_grad():
                entry = certified_entry(target, evaluator_displacement(flow), fixed, moving, labels, device)
            atomic_write_json(report, entry)
            files.append({key: entry[key] for key in ("fixed", "moving", "file", "sha256")})
            print(f"{target.name}: {entry['global']['status']}")
            continue
        save_displacement(target, evaluator_displacement(flow))
        files.append({"fixed": fixed, "moving": moving, "file": target.name, "sha256": sha256_file(target)})
        print(f"{target.name} <- {Path(moving).name} onto {Path(fixed).name}")

    archive = args.out / f"{args.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for entry in files:
            handle.write(folder / entry["file"], arcname=f"{args.name}/{entry['file']}")
    manifest = {
        "checkpoint": args.ckpt,
        "checkpoint_load": load_report,
        **recipe,
        "pairs_json": str(paths[json_key]),
        "convention": "fixed->moving displacement in voxels along the stored array axes, [D,H,W,3] float32, "
        "as sampled by the checkpoint-compatible SpatialTransformer (legacy_sampling_displacement)",
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "files": files,
    }
    if args.certified:
        verdicts = [json.loads((certificates / f"{e['file']}.json").read_text(encoding="utf-8")) for e in files]
        manifest["certified"] = {
            "claim_eps": IDENTITY_BOUNDARY["claim_eps"],
            "collar_voxels": IDENTITY_BOUNDARY["collar"],
            "global_homeomorphism_pairs": sum(v["global"]["global_homeomorphism"] for v in verdicts),
            "scope": "verdicts of the stored fields' trilinear interpolants; see certificates/ for each pair",
        }
    atomic_write_json(args.out / "manifest.json", manifest)
    print(f"{len(files)} displacements -> {archive}")


if __name__ == "__main__":
    main()
