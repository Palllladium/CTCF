"""Prepare individual LPBA40 images and inventory unprocessed HCP structural acquisitions."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import struct
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from scipy import ndimage

from tools.analysis.run_artifacts import atomic_write_json, sha256_file


def unpack_archives(root: Path, *, move_archives: bool = False) -> list[dict]:
    root = root.resolve()
    archives, raw = root / "archives", root / "raw"
    archives.mkdir(parents=True, exist_ok=True)
    if move_archives:
        for source in sorted(root.glob("*.zip*")):
            target = archives / source.name
            if source.resolve().parent != root or target.resolve().parent != archives.resolve():
                raise ValueError("Archive move leaves the selected dataset root")
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
    rows = []
    for path in sorted(archives.glob("*.zip")):
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        row = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "md5": digest.hexdigest()}
        checksum = path.with_suffix(".zip.md5")
        if checksum.exists():
            parts = checksum.read_text(encoding="utf-8").split()
            if len(parts) != 2 or parts[1].lstrip("*") != path.name or parts[0].lower() != row["md5"]:
                raise ValueError(f"Provider MD5 does not match {path.name}")
            row["provider_md5_verified"] = True
        row["members"] = extract_zip(path, raw)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No ZIP archives in {archives}")
    for nested in sorted(raw.glob("*.zip")):
        rows.append(
            {
                "file": nested.relative_to(root).as_posix(),
                "sha256": sha256_file(nested),
                "members": extract_zip(nested, raw),
            }
        )
    return rows


def extract_zip(path: Path, out: Path) -> int:
    """Preserve archive payloads; reject traversal, symlinks and conflicting existing files."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as archive:
        members = archive.infolist()
        for entry in members:
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            target = out.joinpath(*name.parts).resolve()
            if name.is_absolute() or ":" in entry.filename or not target.is_relative_to(out):
                raise ValueError(f"Unsafe ZIP member {entry.filename!r}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"ZIP symlink is not supported: {entry.filename}")
        if bad := archive.testzip():
            raise ValueError(f"ZIP CRC failed: {path.name}: {bad}")
        for entry in members:
            target = out / entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                crc = 0
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != entry.file_size or crc != entry.CRC:
                    raise ValueError(f"Existing extracted bytes differ: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as stream:
                shutil.copyfileobj(source, stream)
    return len(members)


def read_analyze(header: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    image = nib.AnalyzeImage.from_file_map(
        {
            "header": nib.FileHolder(filename=str(header)),
            "image": nib.FileHolder(filename=str(header.with_suffix(".img.gz"))),
        }
    )
    values = np.asarray(image.dataobj).squeeze()
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid Analyze volume {header}")
    return values, tuple(float(v) for v in image.header.get_zooms()[:3])


def air_pull_matrix(data: bytes) -> tuple[np.ndarray, tuple[int, ...], tuple[float, ...]]:
    """Read AIR16 voxel transforms; coordinate conventions were cross-checked against SITReg."""
    # Reference: SITReg a665a46a, src/data/lpba40/air.py (Joel Honkamaa).
    if len(data) != 720:
        raise ValueError("Expected the 720-byte AIR16 structure")
    for endian in ("<", ">"):
        standard = struct.unpack_from(endian + "4I3d", data, 256)
        reslice = struct.unpack_from(endian + "4I3d", data, 424)
        if standard[0] in (8, 16, 32, 64) and all(0 < d < 4096 for d in (*standard[1:4], *reslice[1:4])):
            break
    else:
        raise ValueError("Invalid AIR16 dimensions or byte order")
    matrix = np.frombuffer(data, dtype=endian + "f8", count=16).reshape(4, 4).T.copy()
    spacing = np.asarray(standard[4:7])
    if np.any(spacing <= 0) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("AIR transform is not a finite affine map")
    pull = np.diag([*(spacing.min() / spacing), 1.0]) @ np.linalg.inv(matrix)
    return pull, tuple(reslice[1:4]), tuple(reslice[4:7])


def resample(values, pull, shape, *, order):
    return ndimage.affine_transform(
        values.astype(np.float32),
        pull[:3, :3],
        offset=pull[:3, 3],
        output_shape=shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def target_grid(source_shape, source_spacing, shape=(160, 192, 224), spacing=1.15):
    """A shared centered physical grid: LIA tensor axes, isotropic spacing, no anatomical fitting."""
    # Original LPBA arrays run RAS; SITReg's exported diagonal affine uses another coordinate convention.
    source = np.diag([*source_spacing, 1.0])
    axes = np.asarray([[-spacing, 0, 0], [0, 0, spacing], [0, -spacing, 0]])
    center = source[:3, :3] @ ((np.asarray(source_shape) - 1) / 2)
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = axes, center - axes @ ((np.asarray(shape) - 1) / 2)
    pull = np.linalg.solve(source, affine)
    extent = np.abs(np.linalg.solve(affine[:3, :3], source[:3, :3])) @ (np.asarray(source_shape) - 1)
    if np.any(extent > np.asarray(shape) - 1 + 1e-6):
        raise ValueError("Target grid truncates the original delineation field of view")
    return pull, affine


def balanced_pairs(ids: list[str], seed: int = 0) -> list[list[str]]:
    if len(ids) < 4 or len(ids) % 2:
        raise ValueError("The symmetric degree-three graph requires an even number of at least four subjects")
    order = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    n = len(order)
    return [[subject, order[(i + offset) % n]] for i, subject in enumerate(order) for offset in (1, n - 1, n // 2)]


def lpba(root: Path, report: Path, shape: tuple[int, ...], spacing: float) -> None:
    archives = unpack_archives(root)
    native, delineation = root / "raw/LPBA40/native_space", root / "raw/LPBA40/delineation_space"
    definitions = ET.parse(delineation / "lpba40.label.xml").getroot()
    names = {int(item.attrib["id"]): item.attrib["fullname"] for item in definitions if int(item.attrib["id"]) != 0}
    if len(names) != 56:
        raise ValueError("Expected the 56 nonbackground LPBA40 labels")
    original_ids = sorted(names)
    lut = np.zeros(max(original_ids) + 1, dtype=np.int16)
    lut[original_ids] = np.arange(1, 57)
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    rows, started, common_affine = [], time.monotonic(), None
    for subject in [f"S{i:02d}" for i in range(1, 41)]:
        directory = native / subject
        image, voxel_size = read_analyze(directory / f"{subject}.native.brain.bfc.hdr")
        tissue, _ = read_analyze(directory / "tissue" / f"{subject}.native.tissue.hdr")
        labels, label_spacing = read_analyze(delineation / subject / f"{subject}.delineation.structure.label.hdr")
        air_path = directory / "transforms" / f"{subject}.delineation.to.native.air"
        native_pull, source_shape, source_spacing = air_pull_matrix(air_path.read_bytes())
        if labels.shape != source_shape or not np.allclose(label_spacing, source_spacing):
            raise ValueError(f"AIR and delineation geometry disagree for {subject}")
        physical_pull = np.diag(voxel_size) @ native_pull[:3, :3] @ np.diag(1 / np.asarray(source_spacing))
        rigid_error = float(np.max(np.abs(physical_pull.T @ physical_pull - np.eye(3))))
        if rigid_error > 1e-8 or np.linalg.det(physical_pull) <= 0:
            raise ValueError(f"Expected an orientation-preserving rigid AIR alignment for {subject}")
        grid_pull, affine = target_grid(source_shape, source_spacing, shape, spacing)
        if common_affine is not None and not np.array_equal(affine, common_affine):
            raise ValueError("LPBA subjects do not share the same physical target grid")
        common_affine = affine
        delineation_image = resample(image, native_pull, source_shape, order=1)
        delineation_tissue = resample(tissue, native_pull, source_shape, order=1)
        wm = np.abs(delineation_tissue - 3) < 1e-5
        if not wm.any():
            raise ValueError(f"No automatic WM tissue found for {subject}")
        wm_mean = float(delineation_image[wm].mean())
        delineation_image = delineation_image / wm_mean * (delineation_tissue > 0)
        values = resample(delineation_image, grid_pull, shape, order=1)
        scale = float(np.percentile(values[values > 0], 99.5))
        values = np.clip(values / scale, 0, 1).astype(np.float32)
        raw_labels = resample(labels, grid_pull, shape, order=0).astype(np.int16)
        if not set(np.unique(raw_labels)) <= {0, *original_ids}:
            raise ValueError(f"Unexpected anatomical labels for {subject}")
        dense_labels = lut[raw_labels]
        counts = np.bincount(dense_labels.ravel(), minlength=57)
        sums = np.bincount(
            dense_labels.ravel(), weights=np.repeat(np.arange(shape[0]), shape[1] * shape[2]), minlength=57
        )
        centers = sums / np.maximum(counts, 1)
        margins = [
            float((centers[i + 1] - centers[i + 2]) * spacing)
            for i, label in enumerate(original_ids)
            if names[label].startswith("L ")
        ]
        if not np.all(counts[1:] > 0) or min(margins) <= 0:
            raise ValueError(f"Label coverage/laterality does not agree with the declared LIA grid for {subject}")
        file = out / f"{subject}.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as saved:
                if not np.array_equal(saved["image"], values) or not np.array_equal(saved["labels"], dense_labels):
                    raise ValueError(
                        f"Prepared bytes differ from the requested recipe; use another output directory: {file}"
                    )
        else:
            np.savez_compressed(file, image=values, labels=dense_labels)
        reference, _ = read_analyze(delineation / subject / f"{subject}.delineation.skullstripped.hdr")
        foreground = (reference > 0) & (delineation_image > 0)
        correlation = float(np.corrcoef(reference[foreground], delineation_image[foreground])[0, 1])
        rows.append(
            {
                "id": subject,
                "file": file.name,
                "sha256": sha256_file(file),
                "air_sha256": sha256_file(air_path),
                "native_shape": list(image.shape),
                "native_spacing_mm": list(voxel_size),
                "wm_mean": wm_mean,
                "intensity_scale": scale,
                "provided_delineation_correlation": correlation,
                "air_rigid_metric_error": rigid_error,
                "minimum_left_minus_right_mm": min(margins),
            }
        )
        print(f"[LPBA] {subject} {len(rows)}/40 correlation={correlation:.5f}", flush=True)
    ras_to_lps = np.diag([-1, -1, 1])
    direction = ras_to_lps @ common_affine[:3, :3][:, ::-1] / spacing
    manifest = {
        "dataset": "LPBA40",
        "subjects": rows,
        "pairs": balanced_pairs([row["id"] for row in rows]),
        "shape": list(shape),
        "spacing_mm": [spacing] * 3,
        "tensor_axes": "LIA",
        "world_frame": "RAS-aligned LPBA array frame; source voxel-zero origin, not absolute MNI coordinates",
        "affine_ras": common_affine.tolist(),
        "sitk_geometry": {
            "spacing": [spacing] * 3,
            "origin": (ras_to_lps @ common_affine[:3, 3]).tolist(),
            "direction": direction.ravel().tolist(),
        },
        "label_ids": list(range(1, 57)),
        "original_label_ids": original_ids,
        "labels_remapped": True,
        "label_names": {str(i + 1): names[label] for i, label in enumerate(original_ids)},
        "label_mapping": {str(label): i + 1 for i, label in enumerate(original_ids)},
        "recipe": "SITReg AIR native-to-delineation; centered physical LIA grid; WM mean then foreground p99.5 clipping",
        "source_revision": "SITReg a665a46a3553b26ade6ff5f57815e5e8ec43a948",
        "mask_disclosure": "Official brain masks derive from manual delineations; tissue classification is automatic. No additional label-based fitting, centering or masking.",
        "source_fov_preserved": True,
        "source_archives": archives,
        "pair_seed": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(out / "manifest.json", manifest, allow_nan=False)
    atomic_write_json(
        report,
        {
            "dataset": "LPBA40",
            "archives": archives,
            "manifest": str(out / "manifest.json"),
            "subjects": rows,
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
        allow_nan=False,
    )


def hcp(root: Path, report: Path) -> None:
    archives = unpack_archives(root, move_archives=True)
    rows = []
    for path in sorted((root / "raw").glob("*/unprocessed/3T/*/*_3T_*.nii.gz")):
        if not (
            (path.parent.name.startswith("T1w_MPR") and "_T1w_MPR" in path.name)
            or (path.parent.name.startswith("T2w_SPC") and "_T2w_SPC" in path.name)
        ):
            continue
        image = nib.load(path)
        values = np.asarray(image.dataobj)
        if values.ndim != 3 or not np.isfinite(values).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
            raise ValueError(f"Invalid raw HCP MRI: {path}")
        rows.append(
            {
                "id": path.relative_to(root / "raw").parts[0],
                "acquisition": path.parent.name,
                "file": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "spacing_mm": list(map(float, image.header.get_zooms()[:3])),
                "affine_ras": image.affine.tolist(),
                "axis_codes": list(nib.aff2axcodes(image.affine)),
                "finite": True,
                "intensity_range": [float(values.min()), float(values.max())],
            }
        )
    ids = sorted({row["id"] for row in rows})
    if not ids or any(
        not {"T1w_MPR1", "T2w_SPC1"} <= {r["acquisition"] for r in rows if r["id"] == subject} for subject in ids
    ):
        raise ValueError("Each HCP QC subject must contain at least T1w_MPR1 and T2w_SPC1")
    atomic_write_json(
        report,
        {
            "dataset": "HCP-YA2025",
            "status": "RAW_HEADER_INVENTORY_ONLY",
            "archives": archives,
            "subjects": ids,
            "acquisitions": rows,
            "prepared_for_training": False,
            "limitations": "No T1-only bias correction, skull stripping, T2-to-T1 fitting or intersubject preprocessing has been performed.",
        },
        allow_nan=False,
    )
    print(f"[HCP RAW INVENTORY] {len(ids)} subjects, {len(rows)} T1/T2 acquisitions: {report}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("lpba", "hcp"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shape", nargs=3, type=int, default=(160, 192, 224))
    parser.add_argument("--spacing", type=float, default=1.15)
    args = parser.parse_args()
    if args.spacing <= 0 or min(args.shape) < 2:
        parser.error("Spacing must be positive and each target dimension at least two")
    if args.dataset == "lpba":
        lpba(args.root.resolve(), args.report, tuple(args.shape), args.spacing)
    else:
        hcp(args.root.resolve(), args.report)


if __name__ == "__main__":
    main()
