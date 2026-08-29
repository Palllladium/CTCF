"""Generate canonical external registration fields directly from official Python APIs.

This bridge deliberately does not accept user-supplied axis/sign/unit flags.  Each method adapter owns and
tests its convention.  The output contract is a float32 pull displacement ``flow`` with shape
``[1,3,D,H,W]``, components ``(z,y,x)``, voxel units, on the fixed grid:

    warped_moving(i) = moving(i + flow(i)).

OASIS pair files are the source of truth.  In ``p_0440_0441.pkl``, for example, tuple item ``x`` is moving
subject 0441 and ``y`` is fixed subject 0440; the filename is ``fixed_moving``, not ``moving_fixed``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.core.path_profiles import get_dataset_paths
from utils.common import pkload
from utils.field import _warp

_CONVEXADAM_PARAMS = {
    "mind_r": 1,
    "mind_d": 2,
    "lambda_weight": 1.25,
    "grid_sp": 6,
    "disp_hw": 4,
    "selected_niter": 80,
    "selected_smooth": 0,
    "grid_sp_adam": 2,
    "ic": True,
    "use_mask": False,
    "dtype": "float16",
}
_CONVEXADAM_COMMIT = "b229e52e44b114e2040a503334c92269750c16b2"
_SEED = 0


def _configure_numeric_policy(device: torch.device) -> None:
    """Reduce avoidable variation without claiming unsupported CUDA bitwise determinism.

    ConvexAdam's Adam refinement differentiates through 3-D ``grid_sample``.  PyTorch has no
    deterministic CUDA implementation of that backward operation, so strict deterministic mode would
    reject the official algorithm.  We therefore use the upstream test policy (warn-only) plus fixed
    seeds and explicit TF32/cuDNN settings.  The remaining limitation is recorded in the manifest and a
    separate two-run smoke gate checks stability of the scientific verdict.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type != "cuda":
        raise RuntimeError("the pinned ConvexAdam audit protocol requires a CUDA device")


def _reset_case_seed() -> None:
    """Make any future random use independent of resume order."""
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect git worktree {path}") from exc
    return bool(result.stdout.strip())


def _package_root(module_file: str) -> Path:
    # Scientific runs require the editable checkout so its exact commit and dirty state can be verified.
    path = Path(module_file).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent
    return path.parent


def _convexadam_provenance() -> dict:
    try:
        import convexAdam
    except ImportError as exc:
        raise RuntimeError("ConvexAdam is unavailable; run `python -m pip install -e ../convexAdam`") from exc
    root = _package_root(convexAdam.__file__)
    git_sha = _git_sha(root)
    if git_sha != _CONVEXADAM_COMMIT:
        raise RuntimeError(f"ConvexAdam must be exactly {_CONVEXADAM_COMMIT}, got {git_sha!r} at {root}")
    if _git_dirty(root):
        raise RuntimeError(f"ConvexAdam has tracked local changes at {root}; refuse a scientific run")
    try:
        version = importlib.metadata.version("convexAdam")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "name": "convexadam-mindssc",
        "preset": "MIND-SSC public Python API defaults (not the semantic Task3 leaderboard pipeline)",
        "package_version": version,
        "module_path": str(Path(convexAdam.__file__).resolve()),
        "git_sha": git_sha,
        "params": _CONVEXADAM_PARAMS,
    }


def _environment_provenance(device: torch.device) -> dict:
    versions = {}
    for package in ("numpy", "scipy", "SimpleITK", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    gpu = None
    gpu_capability = None
    gpu_uuid = None
    gpu_driver = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = properties.name
        gpu_capability = list(torch.cuda.get_device_capability(device))
        gpu_uuid = str(getattr(properties, "uuid", "unavailable"))
        try:
            driver_lines = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            gpu_driver = driver_lines[device_index].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            gpu_driver = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_git_version": torch.version.git_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver": gpu_driver,
        "device": str(device),
        "gpu": gpu,
        "gpu_uuid": gpu_uuid,
        "gpu_compute_capability": gpu_capability,
        "numeric_policy": {
            "name": "controlled-cuda-upstream-test-policy-plus",
            "seed": _SEED,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "bitwise_reproducible_claim": False,
            "known_limitation": (
                "CUDA grid_sampler_3d_backward and AvgPool3d backward have no deterministic PyTorch implementation"
            ),
        },
    }


def _validate_volume_pair(moving: np.ndarray, fixed: np.ndarray, pair: str) -> None:
    if moving.ndim != 3 or fixed.ndim != 3 or moving.shape != fixed.shape:
        raise RuntimeError(f"{pair}: expected equal 3D image shapes, got moving={moving.shape}, fixed={fixed.shape}")
    if not np.isfinite(moving).all() or not np.isfinite(fixed).all():
        raise RuntimeError(f"{pair}: input image contains NaN or Inf")


def _validate_canonical_flow(flow: np.ndarray, spatial_shape: tuple[int, ...], pair: str) -> None:
    if flow.dtype != np.float32 or flow.shape != (1, 3, *spatial_shape):
        raise RuntimeError(f"{pair}: invalid canonical flow dtype/shape: {flow.dtype} {flow.shape}")
    if not np.isfinite(flow).all():
        raise RuntimeError(f"{pair}: flow contains NaN or Inf")


def _canonical_warp(moving: np.ndarray, flow: np.ndarray) -> np.ndarray:
    image = torch.from_numpy(np.ascontiguousarray(moving))[None, None].float()
    field = torch.from_numpy(np.ascontiguousarray(flow)).float()
    with torch.no_grad():
        return _warp(image, field, mode="bilinear")[0, 0].cpu().numpy()


def _valid_pull_mask(flow: np.ndarray) -> np.ndarray:
    _, _, d, h, w = flow.shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    return (
        (zz + flow[0, 0] >= 0)
        & (zz + flow[0, 0] <= d - 1)
        & (yy + flow[0, 1] >= 0)
        & (yy + flow[0, 1] <= h - 1)
        & (xx + flow[0, 2] >= 0)
        & (xx + flow[0, 2] <= w - 1)
    )


def _run_convexadam(fixed: np.ndarray, moving: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict]:
    try:
        from convexAdam.apply_convex import apply_convex
        from convexAdam.convex_adam_MIND import convex_adam_pt
    except ImportError as exc:
        raise RuntimeError("ConvexAdam is unavailable; run `python -m pip install -e ../convexAdam`") from exc

    kwargs = {k: v for k, v in _CONVEXADAM_PARAMS.items() if k != "dtype"}
    displacement = convex_adam_pt(
        img_fixed=np.asarray(fixed, dtype=np.float32),
        img_moving=np.asarray(moving, dtype=np.float32),
        dtype=torch.float16,
        device=device,
        verbose=True,
        **kwargs,
    )
    displacement = np.asarray(displacement)
    expected_shape = (*fixed.shape, 3)
    if displacement.shape != expected_shape:
        raise RuntimeError(f"ConvexAdam returned {displacement.shape}, expected {expected_shape}")

    # ConvexAdam's apply_convex adds displacement[...,axis] to the matching array index.  It is already a
    # fixed-grid pull field in voxel units; only channel placement changes.
    flow = np.moveaxis(displacement, -1, 0)[None].astype(np.float32, copy=False)
    _validate_canonical_flow(flow, fixed.shape, "ConvexAdam output")
    native_warp = np.asarray(apply_convex(displacement, moving), dtype=np.float32)
    canonical_warp = _canonical_warp(moving, flow)
    if not np.isfinite(native_warp).all() or not np.isfinite(canonical_warp).all():
        raise RuntimeError("ConvexAdam convention parity produced NaN or Inf")
    valid = _valid_pull_mask(flow)
    if not valid.any():
        raise RuntimeError("ConvexAdam field has no in-bounds pull coordinates")
    error = np.abs(native_warp - canonical_warp)
    parity = {
        "valid_fraction": float(valid.mean()),
        "warp_max_abs_valid": float(error[valid].max()),
        "warp_mean_abs_valid": float(error[valid].mean()),
    }
    # SciPy map_coordinates(order=1) and torch grid_sample should agree up to small implementation rounding.
    if parity["warp_max_abs_valid"] > 2e-4 or parity["warp_mean_abs_valid"] > 2e-6:
        raise RuntimeError(f"ConvexAdam convention parity failed: {parity}")

    return flow, parity


def _load_existing_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema": 2, "contract": {}, "cases": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=["convexadam"])
    parser.add_argument("--paths", type=int, default=2, choices=[1, 2, 3], help="CTCF path profile")
    parser.add_argument("--out", default="results/external/convexadam")
    parser.add_argument("--limit", type=int, default=0, help="0 = all OASIS pair files")
    parser.add_argument("--pair", help="single pair stem, e.g. 0440_0441")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--expected-count", type=int, help="fail unless exactly this many pairs are selected")
    parser.add_argument("--allow-dirty", action="store_true", help="development only; never use for reported runs")
    args = parser.parse_args()

    test_dir = Path(get_dataset_paths(args.paths, "OASIS")["val_dir"])
    files = sorted(test_dir.glob("p_*.pkl"))
    if args.pair:
        wanted = f"p_{args.pair}.pkl"
        files = [path for path in files if path.name == wanted]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no matching OASIS pair files in {test_dir}")
    if args.expected_count is not None and len(files) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} pair files, selected {len(files)} in {test_dir}")

    output = Path(args.out)
    flow_dir = output / "flows"
    flow_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = _load_existing_manifest(manifest_path)
    if manifest.get("schema") != 2:
        raise RuntimeError(f"unsupported manifest schema {manifest.get('schema')!r}; use a fresh output directory")
    method_provenance = _convexadam_provenance()
    if manifest.get("cases") and manifest.get("method") != method_provenance:
        raise RuntimeError(
            "ConvexAdam version/commit/parameters changed; choose a new --out directory rather than mixing runs"
        )
    manifest["method"] = method_provenance
    manifest["contract"] = {
        "array": "flow",
        "dtype": "float32",
        "shape": "[1,3,D,H,W]",
        "components": "z,y,x",
        "units": "voxel",
        "mapping": "fixed-grid pull: phi(i)=i+flow(i)",
        "sampler": "canonical align_corners=True voxel coordinates",
    }
    repo_root = Path(__file__).resolve().parents[2]
    ctcf_dirty = _git_dirty(repo_root)
    if not args.allow_dirty and ctcf_dirty:
        raise RuntimeError(
            f"CTCF has local changes or untracked files at {repo_root}; commit/clean before a scientific run"
        )
    device = torch.device(args.device)
    _configure_numeric_policy(device)
    ctcf_git_sha = _git_sha(repo_root)
    environment = _environment_provenance(device)
    if manifest.get("cases") and manifest.get("ctcf_git_sha") != ctcf_git_sha:
        raise RuntimeError("CTCF commit changed since these fields were generated; choose a fresh --out directory")
    if manifest.get("cases") and manifest.get("environment") != environment:
        raise RuntimeError(
            "runtime environment changed since these fields were generated; choose a fresh --out directory"
        )
    if manifest.get("cases") and manifest.get("ctcf_dirty") != ctcf_dirty:
        raise RuntimeError("CTCF dirty state changed since these fields were generated; choose a fresh --out directory")
    manifest["ctcf_git_sha"] = ctcf_git_sha
    manifest["ctcf_dirty"] = ctcf_dirty
    manifest["environment"] = environment

    rows: list[dict] = []
    generation_run_id = str(uuid.uuid4())
    selected_pairs = {path.stem.removeprefix("p_") for path in files}
    stale_manifest = set(manifest["cases"]) - selected_pairs
    if stale_manifest:
        raise RuntimeError(
            f"output manifest contains stale/unselected cases: {sorted(stale_manifest)}; use a fresh --out"
        )
    for index, path in enumerate(files, 1):
        pair = path.stem.removeprefix("p_")
        moving, fixed, _, _ = pkload(path)
        moving = np.asarray(moving, dtype=np.float32)
        fixed = np.asarray(fixed, dtype=np.float32)
        _validate_volume_pair(moving, fixed, pair)
        input_hashes = {"moving": _sha256_array(moving), "fixed": _sha256_array(fixed)}
        pair_file_hash = _sha256_file(path)
        flow_path = flow_dir / f"flow_{pair}.npz"

        previous = manifest["cases"].get(pair)
        if flow_path.exists() and previous and not args.force:
            if previous.get("input_sha256") != input_hashes:
                raise RuntimeError(f"{pair}: inputs changed; refuse stale resume (use --force after investigation)")
            if previous.get("pair_file_sha256") != pair_file_hash:
                raise RuntimeError(f"{pair}: complete pair pkl hash changed; refuse stale resume")
            with np.load(flow_path, allow_pickle=False) as data:
                if set(data.files) != {"flow"}:
                    raise RuntimeError(f"{pair}: expected only the 'flow' array in {flow_path}, got {data.files}")
                existing_flow = np.asarray(data["flow"])
            _validate_canonical_flow(existing_flow, fixed.shape, pair)
            if _sha256_array(existing_flow) != previous["row"].get("flow_array_sha256"):
                raise RuntimeError(f"{pair}: stored flow array hash differs from manifest")
            if _sha256_file(flow_path) != previous["row"].get("flow_file_sha256"):
                raise RuntimeError(f"{pair}: stored npz file hash differs from manifest")
            print(f"[SKIP {index}/{len(files)}] {pair}: fingerprint matches")
            rows.append(previous["row"])
            continue

        print(f"[RUN  {index}/{len(files)}] {pair}: tuple x=moving, y=fixed")
        _reset_case_seed()
        started = time.perf_counter()
        flow, parity = _run_convexadam(fixed=fixed, moving=moving, device=device)
        runtime = time.perf_counter() - started
        _validate_canonical_flow(flow, fixed.shape, pair)
        temporary = flow_path.with_name(flow_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, flow=flow)
        temporary.replace(flow_path)
        row = {
            "pair": pair,
            "runtime_sec": runtime,
            "flow_array_sha256": _sha256_array(flow),
            "flow_file_sha256": _sha256_file(flow_path),
            **parity,
        }
        manifest["cases"][pair] = {
            "generation_run_id": generation_run_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pair_semantics": "filename=fixed_moving; tuple x=moving, y=fixed",
            "pair_file": str(path),
            "pair_file_sha256": pair_file_hash,
            "input_sha256": input_hashes,
            "flow_file": str(flow_path),
            "row": row,
        }
        _write_manifest(manifest_path, manifest)
        rows.append(row)
        print(f"[PASS {index}/{len(files)}] {pair}: parity max={row['warp_max_abs_valid']:.3e}")

    actual_files = {path.stem.removeprefix("flow_") for path in flow_dir.glob("flow_*.npz")}
    if actual_files != selected_pairs or set(manifest["cases"]) != selected_pairs:
        raise RuntimeError(
            f"output case-set mismatch: selected={sorted(selected_pairs)}, files={sorted(actual_files)}, "
            f"manifest={sorted(manifest['cases'])}"
        )

    with (output / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest_path}, {output / 'per_case.csv'} and {len(rows)} canonical fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
