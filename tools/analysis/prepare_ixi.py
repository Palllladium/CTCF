"""Recover paired IXI T2 geometry, write alignment previews, then finalise a reviewed Train/Val cohort."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
from scipy import ndimage

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import pkload

MANUAL_MAPPING_REVIEW = {40, 127, 144, 192, 254, 341, 407, 558, 576, 356, 357}
LATERAL_LABEL_PAIRS = ((2, 41), (3, 42), (10, 49), (11, 50), (12, 51), (13, 52), (17, 53), (18, 54))


def laterality_report(labels, reference, processed_to_t1):
    """Train-only integrity check in raw-T1 LPS, including the fitted transform; absolute origin is irrelevant."""
    rows = []
    for left, right in LATERAL_LABEL_PAIRS:
        locations = [np.argwhere(labels == label) for label in (left, right)]
        if any(not len(points) for points in locations):
            continue
        points = [
            processed_to_t1.TransformPoint(reference.TransformContinuousIndexToPhysicalPoint(tuple(p.mean(0)[::-1])))
            for p in locations
        ]
        rows.append({"labels": [left, right], "left_minus_right_lps_mm": (np.asarray(points[0]) - points[1]).tolist()})
    if len(rows) < 2:
        raise ValueError("Laterality QC needs at least two bilateral raw FreeSurfer label pairs")
    return {"pairs": rows, "passed": all(row["left_minus_right_lps_mm"][0] > 0 for row in rows)}


def rigid_motion(transform, fixed):
    center = np.asarray(fixed.TransformContinuousIndexToPhysicalPoint(tuple((n - 1) / 2 for n in fixed.GetSize())))
    matrix = linear_matrix(transform)
    return {
        "rigid_shift_mm": float(np.linalg.norm(np.asarray(transform.TransformPoint(tuple(center))) - center)),
        "rigid_angle_deg": float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1)))),
    }


def linear_matrix(transform):
    if not transform.IsLinear():
        raise ValueError("Preparation accepts affine/rigid transforms, not nonlinear warps")
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.column_stack([np.asarray(transform.TransformPoint(tuple(axis))) - origin for axis in np.eye(3)])
    if np.linalg.det(matrix) <= 0:
        raise ValueError("An intra-subject transform reflects orientation; check array geometry and subject identity")
    return matrix


def processed_image(values, geometry):
    image = sitk.GetImageFromArray(np.asarray(values, dtype=np.float32))
    image.SetDirection(geometry["direction"])
    image.SetSpacing(geometry["spacing"])
    image.SetOrigin(geometry["origin"])
    return image


def register(fixed, moving, *, affine: bool, initial=None):
    """SimpleITK returns a pull transform: fixed physical coordinates -> moving physical coordinates."""
    if initial is None:
        if affine:
            initial = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        else:
            initial = sitk.Euler3DTransform()
            initial.SetCenter(
                fixed.TransformContinuousIndexToPhysicalPoint(tuple((n - 1) / 2 for n in fixed.GetSize()))
            )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=2026)
    method.SetInterpolator(sitk.sitkLinear)
    if affine:
        method.SetMetricFixedMask(sitk.Cast(fixed > 0, sitk.sitkUInt8))
    method.SetOptimizerAsGradientDescentLineSearch(1.0, 150, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    result = method.Execute(fixed, moving)
    linear_matrix(result)
    return result, {
        "metric": method.GetMetricValue(),
        "stop": method.GetOptimizerStopConditionDescription(),
        "initialisation": "geometry" if affine else "header_identity",
    }


def prepare_arrays(t1, raw_t1, raw_t2, geometry, *, processed_to_t1=None, t1_to_t2=None):
    fixed = processed_image(t1, geometry)
    if processed_to_t1 is None:
        processed_to_t1, fit1 = register(fixed, raw_t1, affine=True)
    else:
        fit1 = {"source": "supplied processed-physical -> raw-T1-physical pull transform"}
    if t1_to_t2 is None:
        t1_to_t2, fit2 = register(raw_t1, raw_t2, affine=False)
    else:
        fit2 = {"source": "supplied raw-T1-physical -> raw-T2-physical pull transform"}
    # CompositeTransform applies its last transform first: processed -> T1 -> T2.
    for index, transform in enumerate((processed_to_t1, t1_to_t2)):
        matrix = linear_matrix(transform)
        if index == 1 and not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
            raise ValueError("The raw T1-to-T2 pull transform must be rigid")
    pull = sitk.CompositeTransform([t1_to_t2, processed_to_t1])
    aligned_t1 = sitk.GetArrayFromImage(sitk.Resample(raw_t1, fixed, processed_to_t1, sitk.sitkLinear, 0.0))
    t2 = sitk.GetArrayFromImage(sitk.Resample(raw_t2, fixed, pull, sitk.sitkLinear, 0.0))
    coverage_image = sitk.Image(raw_t2.GetSize(), sitk.sitkUInt8) + 1
    coverage_image.CopyInformation(raw_t2)
    coverage = sitk.GetArrayFromImage(sitk.Resample(coverage_image, fixed, pull, sitk.sitkNearestNeighbor, 0))
    mask = ndimage.binary_fill_holes(t1 > 0)
    if not (mask & (coverage > 0)).any() or not np.isfinite(t1).all() or not np.isfinite(t2).all():
        raise ValueError("IXI preparation requires finite images and a non-empty T1 brain support")
    lo, hi = np.percentile(t2[mask & (coverage > 0)], [1, 99])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("T2 has no usable intensity range inside the covered T1 brain")
    normalised = np.where(mask, np.clip((t2 - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    a, b = t1[mask].astype(np.float64), aligned_t1[mask].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        raise ValueError("T1 identity check has a constant brain image")
    qc = {
        "brain_coverage": float(coverage[mask].mean()),
        "t1_correlation": float(np.corrcoef(a, b)[0, 1]),
        "t2_percentiles": [float(lo), float(hi)],
        "n4": False,
        "t1_fit": fit1,
        "t2_fit": fit2,
        **rigid_motion(t1_to_t2, raw_t1),
    }
    return normalised, aligned_t1, qc, processed_to_t1, t1_to_t2


def preview(path, t1, aligned_t1, t2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, (label, values) in enumerate(
        (("Processed T1", t1), ("Raw T1 aligned", aligned_t1), ("T2 aligned/masked", t2))
    ):
        bounds = np.percentile(values[values > 0], [1, 99]) if np.any(values > 0) else (0, 1)
        for axis in range(3):
            ax = axes[row, axis]
            ax.imshow(
                np.take(values, values.shape[axis] // 2, axis=axis).T,
                cmap="gray",
                origin="lower",
                vmin=bounds[0],
                vmax=bounds[1],
            )
            ax.contour(np.take(t1 > 0, t1.shape[axis] // 2, axis=axis).T, levels=[0.5], colors="red", linewidths=0.3)
            ax.set_title(f"{label}; array axis {axis}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def cohort(mapping):
    with Path(mapping).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["split"] in {"Train", "Val"} and row["t2_file"]]
    ids = [row["ixi_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Mapping repeats original IXI IDs")
    return sorted(rows, key=lambda row: (row["split"], row["processed_file"]))


def calibration_subjects(rows):
    training = [row for row in rows if row["split"] == "Train"]
    chosen = []
    for site in ("Guys", "HH", "IOP"):
        candidates = [row for row in training if f"-{site}-" in row["t1_file"]]
        if candidates:
            chosen.append(candidates[0])
    chosen += [row for row in training if row not in chosen][: 8 - len(chosen)]
    return chosen


def calibrate(args, rows):
    """Shortlist both handedness classes, fit affine maps, and verify Train laterality before ranking."""
    samples = calibration_subjects(rows)
    directions = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.eye(3)[:, order] * np.asarray(signs)
            directions.append(tuple(matrix.ravel()))
    raw = {row["ixi_id"]: sitk.ReadImage(str(args.raw / row["t1_file"]), sitk.sitkFloat32) for row in samples}
    processed = {row["ixi_id"]: pkload(args.processed / "Train" / row["processed_file"]) for row in samples}
    ranked = []
    for direction in directions:
        geometry = {"direction": direction, "spacing": [1, 1, 1], "origin": [0, 0, 0]}
        scores = []
        for row in samples:
            values = processed[row["ixi_id"]][0]
            fixed = sitk.Shrink(processed_image(values, geometry), [4] * 3)
            moving = sitk.Shrink(raw[row["ixi_id"]], [4] * 3)
            transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3))
            method = sitk.ImageRegistrationMethod()
            method.SetMetricAsMattesMutualInformation(32)
            method.SetInterpolator(sitk.sitkLinear)
            method.SetInitialTransform(transform)
            scores.append(method.MetricEvaluate(fixed, moving))
        ranked.append({"mean_metric": float(np.mean(scores)), **geometry})
    ranked.sort(key=lambda row: row["mean_metric"])
    shortlist = [
        candidate
        for sign in (-1, 1)
        for candidate in [g for g in ranked if np.linalg.det(np.asarray(g["direction"]).reshape(3, 3)) * sign > 0][:2]
    ]
    fitted = []
    for geometry in shortlist:
        checks = []
        for row in samples:
            t1, labels = processed[row["ixi_id"]]
            fixed, moving = processed_image(t1, geometry), raw[row["ixi_id"]]
            try:
                transform, _ = register(sitk.Shrink(fixed, [4] * 3), sitk.Shrink(moving, [4] * 3), affine=True)
                aligned = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0))
                mask = t1 > 0
                correlation = float(np.corrcoef(t1[mask], aligned[mask])[0, 1])
                check = laterality_report(labels, fixed, transform)
                checks.append({"id": row["ixi_id"], "correlation": correlation, **check})
            except (RuntimeError, ValueError) as error:
                checks.append({"id": row["ixi_id"], "passed": False, "error": str(error)})
        passed = all(check["passed"] and np.isfinite(check.get("correlation", np.nan)) for check in checks)
        fitted.append(
            geometry
            | {
                "orientation_qc_passed": passed,
                "train_checks": checks,
                "fitted_correlation": float(np.mean([check["correlation"] for check in checks])) if passed else None,
            }
        )
    fitted.sort(key=lambda g: g["fitted_correlation"] if g["orientation_qc_passed"] else -np.inf, reverse=True)
    eligible = [g for g in fitted if g["orientation_qc_passed"]]
    for index, geometry in enumerate(eligible):
        geometry.update(
            calibration_subjects=[row["ixi_id"] for row in samples],
            coverage_min=0.99,
            correlation_min=0.8,
            rigid_shift_mm_max=20.0,
            rigid_angle_deg_max=20.0,
        )
        atomic_write_json(args.out / f"geometry_{index}.json", geometry)
    atomic_write_json(
        args.out / "orientation_scores.json",
        {
            "initial_candidates": ranked,
            "affine_candidates": fitted,
            "scope": "Train label laterality integrity QC; fits and ranking use intensity, never Val/Test labels",
        },
    )
    if not eligible:
        raise ValueError(
            "No candidate passed Train laterality after affine fitting; recover/check preprocessing geometry"
        )
    print(
        "Inspect eligible Train alignments and freeze QC limits before preparing Val; labels were used for Train laterality QC only."
    )


def prepare(args, rows):
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    if not geometry.get("orientation_qc_passed"):
        raise ValueError("Geometry lacks successful Train laterality QC; recalibrate before preparing T2")
    selected = rows if not args.limit else calibration_subjects(rows)[: args.limit]
    selected = selected[args.shard :: args.shards]
    if not selected:
        raise ValueError("The selected preparation shard is empty")
    for row in selected:
        started = time.perf_counter()
        folder = args.out / row["split"] / Path(row["processed_file"]).stem
        folder.mkdir(parents=True, exist_ok=True)
        processed = args.processed / row["split"] / row["processed_file"]
        sources = [processed, args.raw / row["t1_file"], args.raw / row["t2_file"]]
        binding = {"sources": {str(p.resolve()): sha256_file(p) for p in sources}, "geometry": geometry}
        transform_paths = (
            []
            if args.transforms is None
            else [args.transforms / f"{row['ixi_id']}_{suffix}.tfm" for suffix in ("processed_to_t1", "t1_to_t2")]
        )
        binding["transforms"] = {str(p.resolve()): sha256_file(p) for p in transform_paths if p.is_file()}
        report = folder / "qc.json"
        if report.exists():
            previous = json.loads(report.read_text())
            if previous["binding"] != binding:
                raise ValueError(f"Preparation recipe changed for {row['ixi_id']}; use a new output directory")
            if (
                (row["split"] != "Train" or "laterality" in previous["qc"])
                and (folder / "t2.npy").is_file()
                and sha256_file(folder / "t2.npy") == previous["t2_sha256"]
            ):
                continue
        transforms = [sitk.ReadTransform(str(p)) if p.is_file() else None for p in transform_paths] or [None, None]
        t1, labels = pkload(processed)
        t1 = np.asarray(t1, dtype=np.float32)
        raw_t1, raw_t2 = [sitk.ReadImage(str(p), sitk.sitkFloat32) for p in sources[1:]]
        t2, aligned, qc, fit1, fit2 = prepare_arrays(
            t1, raw_t1, raw_t2, geometry, processed_to_t1=transforms[0], t1_to_t2=transforms[1]
        )
        if row["split"] == "Train":
            qc["laterality"] = laterality_report(labels, processed_image(t1, geometry), fit1)
        with (folder / "t2.tmp").open("wb") as stream:
            np.save(stream, t2, allow_pickle=False)
        (folder / "t2.tmp").replace(folder / "t2.npy")
        sitk.WriteTransform(fit1, str(folder / "processed_to_t1.tfm"))
        sitk.WriteTransform(fit2, str(folder / "t1_to_t2.tfm"))
        preview(folder / "alignment.png", t1, aligned, t2)
        flags = [
            name
            for name in ("brain_coverage", "t1_correlation")
            if qc[name] < geometry["coverage_min" if name == "brain_coverage" else "correlation_min"]
        ]
        flags += [name for name in ("rigid_shift_mm", "rigid_angle_deg") if qc[name] > geometry[name + "_max"]]
        if "laterality" in qc and not qc["laterality"]["passed"]:
            flags.append("laterality")
        qc["wall_seconds"] = time.perf_counter() - started
        atomic_write_json(
            report,
            {
                "id": Path(row["processed_file"]).stem,
                "ixi_id": row["ixi_id"],
                "split": row["split"],
                "t1": os.path.relpath(processed, args.out),
                "t1_sha256": binding["sources"][str(processed.resolve())],
                "t2": os.path.relpath(folder / "t2.npy", args.out),
                "t2_sha256": sha256_file(folder / "t2.npy"),
                "shape": list(t1.shape),
                "flags": flags,
                "binding": binding,
                "qc": qc,
                "match": row["match"],
            },
        )
        print(f"[IXI PREPARED] {row['ixi_id']} {qc} flags={flags}", flush=True)
    times = [
        json.loads((args.out / row["split"] / Path(row["processed_file"]).stem / "qc.json").read_text())["qc"][
            "wall_seconds"
        ]
        for row in selected
    ]
    atomic_write_json(
        args.out / f"preparation_profile_{args.shard}.json",
        {
            "subjects": len(times),
            "mean_seconds": float(np.mean(times)),
            "estimated_cohort_seconds": float(np.mean(times)) * len(rows),
            "scope": "Short CPU preparation estimate; thread count and node contention affect extrapolation",
        },
    )


def identity_controls(args, rows):
    """Compare matched T1 fits with wrong-subject affine fits; similarity alone does not establish identity."""
    geometry = json.loads(args.geometry.read_text())
    controls = []
    for row in calibration_subjects(rows):
        site = row["t1_file"].split("-")[1]
        wrong = next(
            other
            for other in rows
            if other["split"] == "Train" and other["ixi_id"] != row["ixi_id"] and other["t1_file"].split("-")[1] == site
        )
        t1 = np.asarray(pkload(args.processed / "Train" / row["processed_file"])[0], dtype=np.float32)
        reference = processed_image(t1, geometry)
        raw = sitk.ReadImage(str(args.raw / wrong["t1_file"]), sitk.sitkFloat32)
        transform, _ = register(reference, raw, affine=True)
        fitted = sitk.GetArrayFromImage(sitk.Resample(raw, reference, transform, sitk.sitkLinear, 0))
        mask = t1 > 0
        correlation = float(np.corrcoef(t1[mask], fitted[mask])[0, 1])
        folder = args.out / "Train" / Path(row["processed_file"]).stem
        correct = json.loads((folder / "qc.json").read_text())
        preview(folder / f"wrong_{wrong['ixi_id']}.png", t1, fitted, np.load(folder / "t2.npy", allow_pickle=False))
        controls.append(
            {
                "id": correct["id"],
                "correct": row["ixi_id"],
                "wrong": wrong["ixi_id"],
                "correct_correlation": correct["qc"]["t1_correlation"],
                "wrong_correlation": correlation,
            }
        )
    atomic_write_json(
        args.out / "identity_controls.json",
        {
            "geometry": geometry,
            "mapping_sha256": sha256_file(args.mapping),
            "controls": controls,
            "scope": "Train-only wrong-ID controls; inspect images, not only correlations",
        },
    )


def finalise(args, rows):
    reports = [
        json.loads((args.out / row["split"] / Path(row["processed_file"]).stem / "qc.json").read_text()) for row in rows
    ]
    groups = [[row for row in reports if row["split"] == split] for split in ("Train", "Val")]
    if [len(group) for group in groups] != [args.expected_train, args.expected_val]:
        raise ValueError("Prepared cohort differs from the expected Train/Val counts")
    if len({tuple(row["shape"]) for row in reports}) != 1:
        raise ValueError("Prepared subjects have different array shapes")
    if not reports[0]["binding"]["geometry"].get("orientation_qc_passed"):
        raise ValueError("Preparation has no successful Train orientation QC; regenerate the cohort")
    if any(not row["qc"].get("laterality", {}).get("passed") for row in groups[0]):
        raise ValueError("Every Train subject needs successful laterality QC; rerun prepare")
    if any(row["binding"]["geometry"] != reports[0]["binding"]["geometry"] for row in reports):
        raise ValueError("Prepared subjects used different geometry/QC recipes")
    mandatory = {Path(row["processed_file"]).stem for row in calibration_subjects(rows)}
    mandatory |= {row["id"] for row in reports if row["flags"] or row["match"] != "demographics"}
    mandatory |= {row["id"] for row in groups[1][:3]}
    mandatory |= {row["id"] for row in reports if int(row["id"].rsplit("_", 1)[-1]) in MANUAL_MAPPING_REVIEW}
    controls = json.loads((args.out / "identity_controls.json").read_text())
    if controls["geometry"] != reports[0]["binding"]["geometry"] or controls["mapping_sha256"] != sha256_file(
        args.mapping
    ):
        raise ValueError("Identity controls belong to another mapping or preparation geometry")
    mandatory |= {row["id"] for row in controls["controls"]}
    if not args.reviewed:
        atomic_write_json(
            args.out / "review_required.json",
            {
                "subjects": sorted(mandatory),
                "note": "Also inspect flagged alignment/ID cases and representative Val subjects; correct failed alignment before finalising.",
            },
        )
        raise ValueError("Inspect alignment previews and supply --reviewed with the reviewed subject IDs")
    reviewed = set(args.reviewed.read_text().split())
    if not mandatory <= reviewed or any(row["flags"] for row in reports):
        raise ValueError(
            "Required image review is incomplete or QC flags remain; correct preparation, do not drop subjects"
        )
    for row in reports:
        if any(sha256_file(Path(path)) != digest for path, digest in row["binding"]["sources"].items()):
            raise ValueError(f"Preparation source changed for {row['id']}")
        if sha256_file(args.out / row["t2"]) != row["t2_sha256"]:
            raise ValueError(f"Prepared T2 changed for {row['id']}")
    ids = [row["id"] for row in groups[1]]
    if len(ids) % 2:
        raise ValueError("Validation needs an even number of subjects")
    order = np.random.default_rng(0).permutation(len(ids))
    pairs = [(ids[a], ids[b]) for a, b in zip(order[::2], order[1::2], strict=True)]
    pairs = [direction for a, b in pairs for direction in ((a, b), (b, a))]
    atomic_write_json(
        args.out / "manifest.json",
        {
            "reviewed": True,
            "reviewed_ids": sorted(reviewed),
            "shape": reports[0]["shape"],
            "training": groups[0],
            "validation": groups[1],
            "validation_pairs": pairs,
            "expected_counts": {"training": args.expected_train, "validation": args.expected_val},
            "mapping_sha256": sha256_file(args.mapping),
            "scope": "Train labels used only for laterality integrity QC; intensity-only registration/normalisation; Val labels for evaluation",
        },
    )
    print(f"[IXI READY] {args.out / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "prepare", "identity", "finalise"))
    for name in ("mapping", "processed", "raw", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--transforms", type=Path, help="Optional supplied pull transforms in SimpleITK .tfm format.")
    parser.add_argument("--reviewed", type=Path, help="Text file containing image-reviewed processed subject IDs.")
    parser.add_argument("--limit", type=int, default=0, help="Prepare only this many Train cases for initial QC.")
    parser.add_argument("--expected_train", type=int, default=399)
    parser.add_argument("--expected_val", type=int, default=58)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.limit < 0 or args.limit > 8:
        parser.error("Use 0<=shard<shards and limit0..8 (the fixed Train calibration subjects)")
    if args.stage != "prepare" and args.shards != 1:
        parser.error("Sharding is supported only for prepare")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = cohort(args.mapping)
    if args.stage == "calibrate":
        calibrate(args, rows)
    elif args.stage in {"prepare", "identity"}:
        if args.geometry is None:
            parser.error("prepare requires explicit --geometry chosen and checked on Train")
        if args.stage == "prepare":
            prepare(args, rows)
        else:
            identity_controls(args, rows)
    else:
        finalise(args, rows)


if __name__ == "__main__":
    main()
