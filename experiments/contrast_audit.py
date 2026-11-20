"""Frozen-field completion, stable-loss controls and anatomical diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from scipy import ndimage

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from datasets.LPBA40 import read_lpba_manifest
from experiments.contrast_diagnostics import read_csv
from experiments.contrast_learning import options_for, resample_t1_control
from experiments.contrast_probes import match_regularity
from experiments.contrasts import sync
from experiments.core.data_loaders import ixi_paired_loaders
from experiments.core.field_scoring import score
from experiments.core.inference_metrics import metric_profile_for
from experiments.core.precision import configure_precision
from experiments.core.train_runtime import run_train
from experiments.eval_matching import cases, certified_scores, comparison, evaluate, report as matching_report
from experiments.matching_diagnostics import supported_points
from experiments.repair_diagnostics import project, repair_one
from experiments.train_CTCF import Runner
from experiments.train_matching import load_base, load_encoder, refine_config, source_image, teacher_subjects
from models.Matching import (
    FeatureEncoder,
    Matching,
    base_flow,
    candidate_logits,
    matching_energy,
    overlap_support,
    sample_voxels,
    voxel_grid,
)
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import Grad3d, SegNorm, pkload
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import certify_flow_exact
from utils.field import _trilinear_cell_cert_bound, enforce_identity_boundary, identity_collar
from utils.field_io import load_flow_npz, save_flow_npz_atomic
from utils.losses import local_ncc_map, stable_ncc_map


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def endpoints(args):
    result = {
        "base": (args.base_checkpoint, None, None),
        "parent": (args.checkpoint, None, None),
        "fine_b0": (args.b0_checkpoint, None, None),
    }
    for kind in ("scratch_b0", "scratch_b1", "resampled_t1", "shuffled_t2"):
        report = args.source / "registration" / kind / "report.json"
        saved = read(report)
        checkpoint = Path(saved["checkpoint"])
        if sha256_file(checkpoint) != saved["checkpoint_sha256"]:
            raise ValueError(f"Completed registration checkpoint changed: {kind}")
        result[kind] = checkpoint, None, report
    for name in ("t1_value", "t1_solver", "paired_value", "paired_solver", "paired_solver_small", "paired_solver_half"):
        result[name] = (
            args.base_checkpoint,
            args.source / "matching" / ("paired_solver" if name.endswith("_half") else name) / "ckpt/last.pth",
            None,
        )
    return result


def preflight(args):
    manifest = read_ixi_manifest(args.manifest, verify_bytes=True)
    expected = read(args.source / "preflight.json")
    external = read_lpba_manifest(args.external_manifest, verify_bytes=True)
    if external["fingerprint"] not in expected["files"].values():
        raise ValueError("The external cohort differs from the source run")
    files = {str(args.manifest.resolve()): manifest["fingerprint"]}
    for base, metric, endpoint in endpoints(args).values():
        for path in (base, metric, endpoint):
            if path:
                files[str(path.resolve())] = sha256_file(path)
    for name in ("t1", "paired"):
        path = args.source / "teachers" / name / "ckpt/last.pth"
        files[str(path.resolve())] = sha256_file(path)
    if manifest["fingerprint"] not in expected["files"].values():
        raise ValueError("The source run used another paired manifest")
    for path, digest in expected["files"].items():
        if Path(path).exists() and sha256_file(Path(path)) != digest:
            raise ValueError(f"A source-run input changed: {path}")
    binding = {
        "files": files,
        "source": str(args.source.resolve()),
        "source_head": (args.source / "head.txt").read_text().strip(),
        "epochs": 10,
        "ncc_mode": "stable",
        "seeds": [0, 1],
    }
    path = args.out / "inputs.json"
    if path.exists() and read(path) != binding:
        raise ValueError("Closure inputs changed; use another RUN_ID")
    atomic_write_json(path, binding)
    for dataset, expected_cases in (("LPBA", 120), ("OASIS", 100)):
        for name, (base, metric, _endpoint) in endpoints(args).items():
            source = args.source / dataset / name
            saved = read(source / "report.json")
            if saved.get("cases") != expected_cases or saved.get("failures") or saved.get("control_failures"):
                raise ValueError(f"Historical external readout is incomplete: {source}")
            recipe = saved["binding"]
            if recipe["base_sha256"] != sha256_file(base) or recipe["metric_sha256"] != (
                sha256_file(metric) if metric else None
            ):
                raise ValueError(f"Historical external readout belongs to other weights: {source}")
            rows = read_csv(source / "FF/per_case.csv")
            if len(rows) != expected_cases or len({row["case_id"] for row in rows}) != expected_cases:
                raise ValueError(f"Historical external table is incomplete or duplicated: {source}")
            destination = args.out / dataset / name
            destination.mkdir(parents=True, exist_ok=True)
            for file in ("recipe.json", "report.json", "diagnostics.json", "trace.csv"):
                if (source / file).is_file():
                    shutil.copy2(source / file, destination / file)
            (destination / "FF").mkdir(exist_ok=True)
            for file in (source / "FF").iterdir():
                if file.suffix in (".csv", ".json"):
                    shutil.copy2(file, destination / "FF" / file.name)
            atomic_write_json(
                destination / "readout_origin.json",
                {
                    "source": str(source),
                    "per_case_sha256": sha256_file(source / "FF/per_case.csv"),
                    "recomputed": False,
                },
            )


def targets(args):
    manifest = read_ixi_manifest(args.manifest, verify_bytes=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    original_sources = read(args.source / "targets/inputs.json")["sources"]
    for kind in ("resampled_t1", "shuffled_t2"):
        source = args.source / "targets/targets" / kind / "manifest.json"
        document = read_ixi_manifest(source, verify_bytes=True)
        if document["source_manifest_sha256"] != manifest["fingerprint"]:
            raise ValueError(f"Control targets belong to another original cohort: {kind}")
        if document["fingerprint"] != read(args.source / "registration" / kind / "report.json")["manifest_sha256"]:
            raise ValueError(f"The historical control manifest changed: {kind}")
        if [(r["id"], r["t1_sha256"]) for r in document["training"]] != [
            (r["id"], r["t1_sha256"]) for r in manifest["training"]
        ]:
            raise ValueError("Target controls changed T1 subjects or their sampling order")
        document.pop("fingerprint")
        atomic_write_json(args.out / "targets" / kind / "manifest.json", document)
    normalised = copy.deepcopy(manifest)
    normalised.pop("fingerprint")
    normalised.update(source_manifest_sha256=manifest["fingerprint"], target_control="normalised_t1")
    for row in normalised["training"]:
        a = np.asarray(pkload(row["t1"])[0], dtype=np.float32)
        folder = Path(row["t2"]).parent
        qc_path = folder / "qc.json"
        if sha256_file(qc_path) != original_sources[str(qc_path.resolve())]:
            raise ValueError(f"The frozen normalization QC changed: {row['id']}")
        qc = read(qc_path)
        sources = [
            (Path(p), digest) for p, digest in qc["binding"]["sources"].items() if p.lower().endswith("-t2.nii.gz")
        ]
        if len(sources) != 1:
            raise ValueError(f"Cannot identify one native T2 reference: {row['id']}")
        raw, expected = sources[0]
        raw = args.raw_root / raw.name if args.raw_root is not None else raw
        transforms = [folder / f"{name}.tfm" for name in ("processed_to_t1", "t1_to_t2")]
        if sha256_file(raw) != expected or any(
            sha256_file(p) != original_sources[str(p.resolve())] for p in transforms
        ):
            raise ValueError(f"The frozen normalization geometry changed: {row['id']}")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(raw))
        reader.ReadImageInformation()
        grid = sitk.Image(reader.GetSize(), sitk.sitkUInt8)
        grid.SetSpacing(reader.GetSpacing())
        grid.SetOrigin(reader.GetOrigin())
        grid.SetDirection(reader.GetDirection())
        values, quality = resample_t1_control(
            a,
            grid,
            qc["binding"]["geometry"],
            *[sitk.ReadTransform(str(p)) for p in transforms],
            normalisation_only=True,
        )
        path = args.out / "targets/normalised_t1/Train" / row["id"] / "aux.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".tmp").open("wb") as stream:
            np.save(stream, values)
        path.with_suffix(".tmp").replace(path)
        row.update(t2=str(path.resolve()), t2_sha256=sha256_file(path))
        atomic_write_json(
            path.with_suffix(".json"),
            {"quality": quality, "raw_reference_sha256": expected, "normalisation_only": True},
        )
    atomic_write_json(args.out / "targets/normalised_t1/manifest.json", normalised)
    atomic_write_json(
        args.out / "report.json",
        {
            "subjects": len(normalised["training"]),
            "scope": "Only auxiliary targets change. Normalisation and roundtrip use the same filled T1 mask, native-T2 coverage support and percentile rule; only the roundtrip interpolates T1. Native T2 intensities are never read.",
        },
    )


def complete(args):
    failures = []
    for index, (name, (base, metric, endpoint)) in enumerate(endpoints(args).items()):
        if index % args.lanes != args.lane:
            continue
        common = vars(args) | {
            "dataset": "IXI",
            "base_checkpoint": base,
            "metric_checkpoint": metric,
            "endpoint": endpoint,
            "descriptor_stride": 2 if name.endswith("_half") else None,
            "solver_steps": None,
            "limit": 0,
            "controls": 0,
            "cache_from": args.source / "IXI" / name,
        }
        raw = args.out / "raw" / name
        try:
            raw.mkdir(parents=True, exist_ok=True)
            evaluate(SimpleNamespace(**(common | {"out": raw, "certify": 0, "save_raw": True, "raw_from": None})))
        except Exception as error:
            failures.append({"endpoint": name, "stage": "FF", "error": repr(error)})
            continue
        destination = args.out / "IXI" / name
        destination.mkdir(parents=True, exist_ok=True)
        try:
            evaluate(
                SimpleNamespace(**(common | {"out": destination, "certify": 1, "save_raw": True, "raw_from": raw}))
            )
        except Exception as error:
            failures.append({"endpoint": name, "stage": "REP", "error": repr(error)})
    atomic_write_json(args.out / f"completion_{args.lane}.json", {"failures": failures})
    if failures:
        raise RuntimeError("Some endpoints are incomplete; all available raw readouts were retained")


def recovery(args):
    selected = {"subject_557__subject_85", "subject_92__subject_478"}
    reports = []
    base = None
    for x, y, xs, ys, case in cases(SimpleNamespace(dataset="IXI", manifest=args.manifest, limit=0))[0]:
        if case not in selected:
            continue
        directory = args.out / case
        if (directory / "report.json").exists():
            reports.append(read(directory / "report.json"))
            continue
        source = args.source / "IXI/base/pairs" / case / "FF.npz"
        if source.exists():
            raw = load_flow_npz(source).to(args.device)
        else:
            if args.base_checkpoint is None:
                raise FileNotFoundError(f"Required saved recovery field: {source}")
            if base is None:
                base = load_base(args.base_checkpoint, x.shape[1:], args.device)
            raw = base_flow(base, x[None].to(args.device), y[None].to(args.device))
        labels = [v[None].to(args.device) for v in (xs, ys)]
        report = repair_one(raw, None, {"dataset": "IXI", "case_id": case, "old_tri_iters": 80.0}, labels, directory)
        reports.append(report)
    if len(reports) != 2:
        raise ValueError("The two frozen recovery cases are absent")
    atomic_write_json(args.out / "report.json", {"cases": reports})


def profile(args):
    options = options_for(
        SimpleNamespace(
            **(
                vars(args)
                | {
                    "kind": "b1",
                    "exp": f"{args.out.parent.name}_{args.out.name}",
                    "seed": 0,
                    "ncc_mode": "stable",
                    "epochs": 10,
                    "workers": 8,
                    "precision": "bf16",
                }
            )
        )
    )
    options.max_epoch, options.max_train_iters, options.max_val_batches, options.save_ckpt = 1, 20, 1, 0
    runner = Runner(options, args.device)
    counter, tick, measurements = 0, None, {}
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)

    def observe_step(_optimizer, _args, _kwargs):
        nonlocal counter, tick
        counter += 1
        if counter == 5:
            tick = sync(args.device)
        if counter == 20:
            measurements["update_seconds"] = (sync(args.device) - tick) / 15

    runner.optimizer.register_step_post_hook(observe_step)
    run_train(options, runner, build_loaders=ixi_paired_loaders)
    measurements.update(
        measured_updates=15,
        completed_updates=counter,
        peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2**30 if args.device.type == "cuda" else None,
        ncc_mode="stable",
        scope="Real full-cascade B1 forward/backward/Adam; five warmup updates excluded; no runtime gate",
    )
    atomic_write_json(args.out / "report.json", measurements)
    if "update_seconds" in measurements:
        print(
            f"[STABLE PROFILE] update_seconds={measurements['update_seconds']:.3f} estimated_seven_training_gpu_hours={7 * 3990 * measurements['update_seconds'] / 3600:.2f}; validation/readout excluded; no limit",
            flush=True,
        )
    if counter != 20 or not np.isfinite(measurements.get("update_seconds", np.nan)):
        raise RuntimeError("Stable NCC training profile did not finish its twenty updates")


def train_pairs(manifest, device):
    _, heldout = teacher_subjects(manifest)
    dataset = IXIPairedDataset(manifest, "b1")
    dataset.rows = [row for row in dataset.rows if row["id"] in heldout]
    for index in range(0, len(dataset.rows) - 1, 2):
        rows = dataset.rows[index : index + 2]
        values = [v.unsqueeze(0).to(device, torch.float32) for v in dataset[(index, index + 1)]]
        labels = []
        for row in rows:
            image, segmentation = pkload(row["t1"])
            labels.append(
                torch.as_tensor(np.ascontiguousarray(SegNorm()([image, segmentation])[1]), device=device)[None, None]
            )
        yield "__".join(row["id"] for row in rows), values, labels, rows


def point_reference(a, b, points, width=9):
    """Independent centered binary64 window calculation, including zero padding."""
    pad = width // 2
    a, b = [torch.nn.functional.pad(v.double(), (pad,) * 6) for v in (a, b)]
    values = []
    for z, y, x in points.T.tolist():
        av, bv = [v[0, 0, z : z + width, y : y + width, x : x + width].flatten() for v in (a, b)]
        da, db = av - av.mean(), bv - bv.mean()
        va, vb = da.square().sum(), db.square().sum()
        values.append((da * db).sum().square() / (va.clamp_min(1e-5) * vb.clamp_min(1e-5)))
    return torch.stack(values)


def inputs(args):
    manifest = read_ixi_manifest(args.manifest)
    controls = {
        name: read_ixi_manifest(args.prepared / "targets" / name / "manifest.json")
        for name in ("normalised_t1", "resampled_t1", "shuffled_t2")
    }
    rows = []
    for index, subject in enumerate(manifest["training"]):
        t1 = np.asarray(pkload(subject["t1"])[0], dtype=np.float32)
        support = ndimage.binary_fill_holes(t1 > 0)
        for name, document in {"t2": manifest, **controls}.items():
            path = Path(document["training"][index]["t2"])
            a = np.load(path, allow_pickle=False)
            finite = bool(np.isfinite(a).all())
            v = a[support]
            correlation = (
                float(np.corrcoef(t1[support], v)[0, 1]) if finite and min(t1[support].std(), v.std()) > 0 else None
            )
            values = a[np.isfinite(a)]
            low, high = (float(values.min()), float(values.max())) if values.size else (None, None)
            valid_range = finite and low >= -1e-6 and high <= 1.000001
            coverage = (
                read(path.parent / "qc.json").get("qc", {}).get("brain_coverage")
                if (path.parent / "qc.json").exists()
                else None
            )
            if name in ("resampled_t1", "normalised_t1") and path.with_suffix(".json").exists():
                coverage = read(path.with_suffix(".json"))["quality"]["coverage"]
            rows.append(
                {
                    "subject": subject["id"],
                    "target": name,
                    "finite": finite,
                    "min": low,
                    "max": high,
                    "valid_range": valid_range,
                    "zero_fraction_in_t1": float((v == 0).mean()),
                    "one_fraction_in_t1": float((v == 1).mean()),
                    "correlation_with_t1": correlation,
                    "coverage_qc": coverage,
                }
            )
    atomic_write_json(args.out / "target_audit.json", {"subjects": len(manifest["training"]), "rows": rows})
    if any(not row["valid_range"] for row in rows):
        raise FloatingPointError(
            "Target audit found nonfinite or out-of-range auxiliary targets; see target_audit.json"
        )
    base = load_base(args.base_checkpoint, manifest["shape"], args.device)
    other = load_base(args.b0_checkpoint, manifest["shape"], args.device)
    numerical, passed = [], True
    for case, values, _labels, subjects in train_pairs(manifest, args.device):
        record = args.out / "ncc" / f"{case}.json"
        if record.exists():
            saved = read(record)
            numerical.extend(saved["rows"])
            passed &= saved["passed"]
            continue
        x, y, tx, ty = values
        fields = {"b0": base_flow(other, x, y), "b1": base_flow(base, x, y)}
        target_pairs = {"t1": (x[:, :1], y[:, :1]), "t2": (tx, ty)}
        for name in ("normalised_t1", "resampled_t1"):
            table = {row["id"]: row for row in controls[name]["training"]}
            target_pairs[name] = tuple(
                torch.from_numpy(np.load(table[s["id"]]["t2"])).to(args.device)[None, None] for s in subjects
            )
        case_rows, case_passed = [], True
        for arm, field in fields.items():
            for target, (a, b) in target_pairs.items():
                variable = field.detach().requires_grad_(True)
                warped = sample_voxels(a, voxel_grid(field.shape[2:], device=args.device) + variable)
                tick = sync(args.device)
                stable = stable_ncc_map(warped, b)
                gradient = torch.autograd.grad(stable.mean(), variable)[0]
                elapsed = sync(args.device) - tick
                old_variable = field.detach().requires_grad_(True)
                legacy = local_ncc_map(
                    sample_voxels(a, voxel_grid(field.shape[2:], device=args.device) + old_variable), b
                )
                old_gradient = torch.autograd.grad(legacy.mean(), old_variable)[0]
                # Sample the strongest legacy violations as well as uniformly spaced windows.
                flat = legacy.detach().flatten()
                positions = torch.cat(
                    (
                        flat.topk(min(32, flat.numel())).indices,
                        torch.linspace(0, flat.numel() - 1, 32, device=args.device).long(),
                    )
                ).unique()
                coordinates = torch.stack(torch.unravel_index(positions, field.shape[2:]))
                reference = point_reference(warped.detach(), b, coordinates)
                gap = float((stable.detach().flatten()[positions].double() - reference).abs().max())
                # Independent direct-convolution values and field gradients on an interior patch.
                start = [max(0, n // 2 - 12) for n in field.shape[2:]]
                slices = tuple(slice(v, min(v + 24, n)) for v, n in zip(start, field.shape[2:], strict=True))
                patch = field[(slice(None), slice(None), *slices)].double().detach().requires_grad_(True)
                grid = voxel_grid(patch.shape[2:], device=args.device, dtype=torch.double) + patch.new_tensor(
                    start
                ).view(1, 3, 1, 1, 1)
                image = sample_voxels(a.double(), grid + patch)
                fixed = b[(slice(None), slice(None), *slices)].double()
                direct, separable = local_ncc_map(image, fixed), stable_ncc_map(image, fixed)
                first = torch.autograd.grad(direct.mean(), patch, retain_graph=True)[0]
                second = torch.autograd.grad(separable.mean(), patch)[0]
                error_norm = float((first - second).norm())
                reference_norm_value = float(first.norm())
                relative = error_norm / max(reference_norm_value, 1e-10)
                production_patch = patch.detach().float().requires_grad_(True)
                production_image = sample_voxels(a, grid.float() + production_patch)
                stable_patch = stable_ncc_map(production_image, fixed.float())
                old_patch = local_ncc_map(production_image, fixed.float())
                stable_patch_gradient = torch.autograd.grad(stable_patch.mean(), production_patch, retain_graph=True)[0]
                old_patch_gradient = torch.autograd.grad(old_patch.mean(), production_patch)[0]
                reference_norm = first.double().norm().clamp_min(1e-10)
                finite = bool(torch.isfinite(gradient).all())
                acceptable = finite and gap <= 2e-5 and error_norm <= 1e-8 + 1e-3 * reference_norm_value
                case_passed &= acceptable
                case_rows.append(
                    {
                        "case_id": case,
                        "field": arm,
                        "target": target,
                        "legacy_max": float(flat.max()),
                        "legacy_above_one": int((flat > 1.00001).sum()),
                        "stable_max": float(stable.detach().max()),
                        "reference_max_abs_gap": gap,
                        "patch_field_gradient_relative_gap": relative,
                        "patch_field_gradient_error_norm": error_norm,
                        "patch_field_gradient_reference_norm": reference_norm_value,
                        "canonical_fp32_stable_patch_gradient_relative_gap": float(
                            (stable_patch_gradient.double() - first).norm() / reference_norm
                        ),
                        "canonical_fp32_legacy_patch_gradient_relative_gap": float(
                            (old_patch_gradient.double() - first).norm() / reference_norm
                        ),
                        "legacy_finite_gradient": bool(torch.isfinite(old_gradient).all()),
                        "legacy_full_gradient_rms": float(old_gradient.double().square().mean().sqrt())
                        if bool(torch.isfinite(old_gradient).all())
                        else None,
                        "stable_full_gradient_rms": float(gradient.double().square().mean().sqrt()),
                        "finite_gradient": finite,
                        "stable_forward_backward_seconds": elapsed,
                        "gradient_coordinates": "canonical source-index displacement psi; border padding",
                        "passed": acceptable,
                    }
                )
                del (
                    stable,
                    legacy,
                    warped,
                    gradient,
                    old_gradient,
                    direct,
                    separable,
                    first,
                    second,
                    stable_patch,
                    old_patch,
                    stable_patch_gradient,
                    old_patch_gradient,
                )
        numerical.extend(case_rows)
        passed &= case_passed
        atomic_write_json(record, {"rows": case_rows, "passed": case_passed})
        print(f"[NCC AUDIT] {case} passed={case_passed}", flush=True)
    atomic_write_json(
        args.out / "numerics.json",
        {
            "rows": numerical,
            "passed": passed,
            "scope": "Native-grid values/gradients in canonical psi coordinates with border padding; independent centered binary64 windows and direct-convolution patch gradients. Training instead differentiates legacy u through RegisterModel/grid_sample with zeros padding; that path is checked separately by the native training profile. No Dice-based selection.",
        },
    )
    if not passed:
        raise FloatingPointError("Stable NCC did not pass its predeclared Train numerical tolerances")


def class_agreement(probability, candidate_labels, fixed, center, ambiguous, *, prediction=None):
    probability = probability[0]
    prediction = probability.argmax(0) if prediction is None else prediction
    correct = candidate_labels == fixed[None]
    before, after = center == fixed, candidate_labels.gather(0, prediction[None])[0] == fixed
    groups = {
        "foreground": fixed > 0,
        "cortex": (fixed == 2) | (fixed == 22),
        "deep_gray": torch.isin(fixed, fixed.new_tensor((7, 8, 9, 10, 27, 28, 29, 30))),
        "ambiguous": (fixed > 0) & ambiguous,
    }
    result = {}
    for name, mask in groups.items():
        if not bool(mask.any()):
            result[name] = {"queries": 0}
            continue
        result[name] = {
            "queries": int(mask.sum()),
            "before": float(before[mask].float().mean()),
            "after": float(after[mask].float().mean()),
            "correct_to_wrong": float((before & ~after)[mask].float().mean()),
            "wrong_to_correct": float((~before & after)[mask].float().mean()),
            "candidate_class_coverage": float(correct[:, mask].any(0).float().mean()),
            "correct_class_probability_mass": float((probability * correct).sum(0)[mask].mean()),
        }
    return result


@torch.no_grad()
def teachers(args):
    manifest = read_ixi_manifest(args.manifest)
    base = load_base(args.base_checkpoint, manifest["shape"], args.device)
    names = {
        "t1_teacher": args.source / "teachers/t1/ckpt/last.pth",
        "paired_teacher": args.source / "teachers/paired/ckpt/last.pth",
        "t1_student": args.source / "matching/t1_solver/ckpt/last.pth",
        "paired_student": args.source / "matching/paired_solver/ckpt/last.pth",
    }
    fit_ids, heldout_ids = teacher_subjects(manifest)
    results = []
    for pair_index, (case, values, labels, _subjects) in enumerate(train_pairs(manifest, args.device)):
        output = args.out / "pairs" / case / "report.json"
        if output.exists():
            results.extend(read(output)["rows"])
            continue
        x, y, tx, ty = values
        parent = base_flow(base, x, y)
        torch.manual_seed(20261117 + pair_index)
        points = supported_points(y[:, :1], parent, args.points, margin=4)
        query_path = output.parent / "queries.json"
        atomic_write_json(
            query_path,
            {
                "points_zyx": points[0].T.cpu().tolist(),
                "seed": 20261117 + pair_index,
                "selection": "T1 support and initial coordinates with four-voxel margin; no labels or teacher",
            },
        )
        fixed = sample_voxels(labels[1].float(), points)[:, 0][0].long()
        centers = points + sample_voxels(parent, points)
        center = sample_voxels(labels[0].float(), centers.round())[0, 0].long()
        populations, refined = {}, []
        for radius in (2, 4):
            offsets = torch.tensor(list(product((-float(radius), 0.0, float(radius)), repeat=3)), device=args.device)
            candidate_labels = torch.stack(
                [
                    sample_voxels(labels[0].float(), (centers + offset.view(1, 3, 1)).round())[0, 0].long()
                    for offset in offsets
                ]
            )
            intensity = torch.stack(
                [sample_voxels(x[:, :1], centers + offset.view(1, 3, 1))[0, 0] for offset in offsets]
            )
            costs = (intensity - sample_voxels(y[:, :1], points)[0, 0]).square().sort(0).values
            ambiguous = costs[1] - costs[0] <= 0.0025
            populations[radius] = offsets, candidate_labels, ambiguous
            uniform = torch.full((1, 27, points.shape[-1]), 1 / 27, device=args.device)
            for name, prediction in (
                ("center", torch.full((points.shape[-1],), 13, device=args.device, dtype=torch.long)),
                ("random_candidate", torch.randint(27, (points.shape[-1],), device=args.device)),
            ):
                probability = uniform
                if name == "center":
                    probability = torch.zeros_like(uniform)
                    probability[:, 13] = 1
                results.append(
                    {
                        "case_id": case,
                        "model": name,
                        "radius": radius,
                        "groups": class_agreement(
                            probability, candidate_labels, fixed, center, ambiguous, prediction=prediction
                        ),
                    }
                )
        pair_rows = results[-4:]
        main_metadata = None
        for name, path in (*names.items(), ("random_encoder", None)):
            if path:
                encoder, saved = load_encoder(path, args.device)
                if saved["step"] != saved["recipe"]["updates"] or saved["manifest_sha256"] != manifest["fingerprint"]:
                    raise ValueError(f"Incomplete or mismatched descriptor checkpoint: {name}")
                if saved["teacher_fit_ids"] != fit_ids or saved["teacher_heldout_ids"] != heldout_ids:
                    raise ValueError(f"Descriptor monitoring subjects differ: {name}")
                if name.endswith("student") and saved["base_sha256"] != sha256_file(args.base_checkpoint):
                    raise ValueError(f"Student belongs to another parent: {name}")
                if name == "paired_student":
                    main_metadata = saved
            else:
                saved = main_metadata
                torch.manual_seed(saved["recipe"]["seed"])
                encoder = FeatureEncoder(**saved["encoder_options"]).to(args.device)
            encoder.eval().requires_grad_(False)
            moving, fixed_image = (
                (source_image(x, tx, saved["source"]), source_image(y, ty, saved["source"]))
                if name.endswith("teacher")
                else (x[:, :1], y[:, :1])
            )
            tick = sync(args.device)
            fm, ff = encoder(moving), encoder(fixed_image)
            feature_seconds = sync(args.device) - tick
            for radius, (offsets, candidate_labels, ambiguous) in populations.items():
                probability = candidate_logits(
                    ff, fm, parent, offsets, points=points, temperature=saved["recipe"]["temperature"]
                ).softmax(1)
                pair_rows.append(
                    {
                        "case_id": case,
                        "model": name,
                        "radius": radius,
                        "groups": class_agreement(probability, candidate_labels, fixed, center, ambiguous),
                    }
                )
            config = refine_config(main_metadata or read(args.source / "matching/paired_solver/recipe.json"))
            model = Matching(encoder, config).eval()
            for multiplier in (0.5, 1.0, 2.0):
                model.solver_config = config | {"step_size": config["step_size"] * multiplier}
                tick = sync(args.device)
                flow, energy = model(x[:, :1], y[:, :1], parent, features=(fm, ff), return_diagnostics=True)
                row = score(
                    flow,
                    case,
                    feature_seconds + sync(args.device) - tick,
                    *labels,
                    0,
                    "canonical",
                    profile=metric_profile_for("IXI"),
                )
                refined.append(
                    {
                        "model": name,
                        "step_multiplier": multiplier,
                        **energy,
                        "time_scope": "Descriptor extraction, solver and energy diagnostics; parent excluded",
                        "correction_rms_voxels": float((flow - parent).square().mean().sqrt()),
                        "correction_max_voxels": float((flow - parent).abs().max()),
                        **row,
                    }
                )
            del model, encoder, fm, ff
        atomic_write_json(
            output, {"rows": pair_rows, "refinements": refined, "queries_sha256": sha256_file(query_path)}
        )
        results = [row for row in results if row["case_id"] != case] + pair_rows
        print(f"[MATCHING SOURCE AUDIT] {case}", flush=True)
    atomic_write_json(
        args.out / "report.json",
        {
            "rows": results,
            "scope": "Frozen Train subjects held out from teacher pretraining; students saw all Train subjects. T1-only query selection; labels measure class coverage/ranking, not true correspondence. Random candidate is a seeded draw with a uniform distribution; center has unit mass at the unmodified correspondence. Per-pair subject dependence is retained.",
        },
    )


def ncc_steps(moving, fixed, initial, config, count):
    grid = voxel_grid(initial.shape[2:], device=initial.device)
    support = overlap_support(moving, fixed, initial)
    current = initial.detach().requires_grad_(True)
    upper = initial.new_tensor(initial.shape[2:]).view(1, 3, 1, 1, 1) - 1
    with torch.enable_grad():
        for _ in range(count):
            positions = grid + current
            cc = stable_ncc_map(sample_voxels(moving, positions), fixed)
            outside = (torch.relu(-positions).square() + torch.relu(positions - upper).square()).sum(1, keepdim=True)
            loss = ((-cc + config["outside_weight"] * outside) * support).sum() / support.sum().clamp_min(1)
            loss += config["reg_weight"] * Grad3d(penalty="l2")(current)
            grad = torch.autograd.grad(loss, current)[0] * np.prod(initial.shape[2:])
            current = (
                (current - config["max_update"] * torch.tanh(config["step_size"] * grad / config["max_update"]))
                .detach()
                .requires_grad_(True)
            )
    return current.detach()


def solvers(args):
    encoder, saved = load_encoder(args.source / "matching/paired_solver/ckpt/last.pth", args.device)
    config = refine_config(saved)
    ncc_config = config | (saved.get("ncc_config") or {})
    model = Matching(encoder.eval().requires_grad_(False), config).eval()
    items, profile, _signature, count = cases(args)
    base, outputs = None, []
    for index, (x, y, xs, ys, case) in enumerate(items):
        if index >= count:
            break
        output = args.out / "pairs" / case / "report.json"
        if output.exists():
            outputs.extend(read(output)["rows"])
            continue
        x, y, xs, ys = [v.unsqueeze(0).to(args.device) for v in (x, y, xs, ys)]
        if base is None:
            base = load_base(args.base_checkpoint, x.shape[2:], args.device)
        tick = sync(args.device)
        initial = base_flow(base, x, y)
        base_seconds = sync(args.device) - tick
        rows, fields, candidates = [], {"base": initial}, []
        for steps in (1, 3):
            tick = sync(args.device)
            field = model(x, y, initial, steps=steps)
            seconds = sync(args.device) - tick
            fields[f"matching{steps}"] = field
            candidates.append(("Matching", steps, field, base_seconds + seconds))
        candidates.append(("base", 0, initial, base_seconds))
        for steps in (1, 2, 3, 5):
            tick = sync(args.device)
            field = ncc_steps(x, y, initial, ncc_config, steps)
            candidates.append(("NCC", steps, field, base_seconds + sync(args.device) - tick))
        with torch.no_grad():
            features = encoder(x), encoder(y)
            support = overlap_support(x, y, initial)
            for method, steps, field, elapsed in candidates:
                row = score(field, case, elapsed, xs, ys, 0, "canonical", profile=profile)
                row.update(objective_readout(field, x, y, features, support, config, ncc_config))
                rows.append({"method": method, "steps": steps, **row})
            del features
        regularity = Grad3d(penalty="l2")
        target = min(float(regularity(f)) for f in (initial, fields["matching3"]))
        matched = []
        for name in ("base", "matching3"):
            field, detail = match_regularity(fields[name].cpu(), target)
            matched.append(
                {
                    "method": name,
                    "match": detail,
                    **score(field.to(args.device), case, 0, xs, ys, 0, "canonical", profile=profile),
                }
            )
        atomic_write_json(
            output,
            {
                "rows": rows,
                "common_R": matched,
                "target_R": target,
                "matching_config": config,
                "ncc_config": ncc_config,
            },
        )
        outputs.extend(rows)
        print(f"[SOLVER CURVES] {args.dataset} {index + 1}/{count} {case}", flush=True)
    atomic_write_json(
        args.out / "report.json",
        {
            "cases": count,
            "rows": outputs,
            "scope": "Measured fixed-step curves, including feature extraction at each Matching endpoint; posthoc objective/Dice/geometry calculations excluded from inference timing. No iso-time interpolation or Val-based step selection. Common-R cases control diffusion only.",
        },
    )


@torch.no_grad()
def objective_readout(field, moving, fixed, features, support, config, ncc_config):
    positions = voxel_grid(field.shape[2:], device=field.device) + field
    cc = stable_ncc_map(sample_voxels(moving, positions), fixed)
    count = support.sum().clamp_min(1)
    data = -(cc * support).sum() / count
    upper = field.new_tensor(field.shape[2:]).view(1, 3, 1, 1, 1) - 1
    outside = (torch.relu(-positions).square() + torch.relu(positions - upper).square()).sum(1, keepdim=True)
    regularity = Grad3d(penalty="l2")(field)
    energy = (
        data + ncc_config["outside_weight"] * (outside * support).sum() / count + ncc_config["reg_weight"] * regularity
    )
    fm, ff = features
    learned = matching_energy(
        ff, fm, field, mask=support, reg_weight=config["reg_weight"], outside_weight=config["outside_weight"]
    )
    return {
        "ncc_data_loss": float(data),
        "ncc_energy": float(energy),
        "matching_energy": float(learned),
        "diffusion_R": float(regularity),
    }


def region_weight(labels, width=4):
    cortex = np.isin(labels.detach().cpu().numpy()[0, 0], (2, 22))
    distance = ndimage.distance_transform_edt(~cortex) if cortex.any() else np.full(cortex.shape, width)
    t = np.clip(distance / width, 0, 1)
    return torch.from_numpy((3 * t * t - 2 * t * t * t).astype(np.float32)).to(labels.device)[None, None]


def regional(args):
    manifest = read_ixi_manifest(args.manifest)
    if args.family == "scratch":
        weights = [
            Path(read(args.source / "registration" / f"scratch_b{i}" / "report.json")["checkpoint"]) for i in (0, 1)
        ]
    else:
        weights = [args.b0_checkpoint, args.base_checkpoint]

    # Stream one pair at a time; do not retain the full validation cohort on the GPU.
    def data():
        for case, values, labels, _subjects in train_pairs(manifest, args.device):
            yield "Train", case, values[:2], labels
        if args.family == "scratch":
            local = SimpleNamespace(dataset="IXI", manifest=args.manifest, limit=0)
            for x, y, xs, ys, case in cases(local)[0]:
                yield (
                    "Val",
                    case,
                    [v.unsqueeze(0).to(args.device) for v in (x, y)],
                    [v.unsqueeze(0).to(args.device) for v in (xs, ys)],
                )

    models = [load_base(p, manifest["shape"], args.device) for p in weights]
    rows, failures = [], []
    for split, case, (x, y), labels in data():
        destination = args.out / split / case
        path = destination / "report.json"
        if path.exists() and not read(path)["failed"]:
            rows.extend(read(path)["rows"])
            continue
        fields = [base_flow(model, x[:, :1], y[:, :1]) for model in models]
        case_rows = []
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            field = fields[0] + alpha * (fields[1] - fields[0])
            case_rows.append(
                {
                    "split": split,
                    "alpha": alpha,
                    "mode": "FF",
                    **score(field, case, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI")),
                }
            )
        field = fields[0] + region_weight(labels[1]) * (fields[1] - fields[0])
        raw = score(field, case, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI"))
        case_rows.append({"split": split, "alpha": "label_oracle", "mode": "FF", **raw})
        failed = False
        if args.family == "scratch":
            for name, candidate in (("base", fields[0]), ("label_oracle", field)):
                try:
                    scored, report = certified_scores(
                        candidate, case, 0, *labels, destination / name, metric_profile_for("IXI"), "IXI"
                    )
                    failed |= bool(report["errors"] or not report["global_homeomorphism"])
                    if "REP" in scored:
                        case_rows.append({"split": split, "alpha": name, "mode": "REP", **scored["REP"]})
                except Exception as error:
                    failed = True
                    failures.append({"case_id": case, "split": split, "variant": name, "error": repr(error)})
        atomic_write_json(path, {"rows": case_rows, "failed": failed})
        rows.extend(case_rows)
        if failed:
            failures.append({"case_id": case, "split": split})
        print(f"[REGIONAL ORACLE] {args.family} {split} {case}", flush=True)
    atomic_write_json(
        args.out / "report.json",
        {
            "rows": rows,
            "failures": failures,
            "oracle_criteria": regional_criteria(rows, failures),
            "scope": "Fixed-label oracle, never a label-free model or an upper bound on all spatial mechanisms. Alpha is zero on fixed cortex, not a guarantee of preserved cortex Dice. Family/endpoints and alpha grid fixed before evaluation.",
        },
    )
    if failures:
        raise RuntimeError("Some regional repaired endpoints are incomplete")


def regional_criteria(rows, failures):
    result = {}
    for split in ("Train", "Val"):
        tables = [
            {
                row["case_id"]: row
                for row in rows
                if row["split"] == split and row["mode"] == "REP" and row["alpha"] == name
            }
            for name in ("base", "label_oracle")
        ]
        if not tables[0] or tables[0].keys() != tables[1].keys():
            result[split] = {"status": "NOT_MEASURED_OR_INCOMPLETE"}
            continue
        delta = float(
            np.mean([float(tables[1][case]["dice_mean"]) - float(tables[0][case]["dice_mean"]) for case in tables[0]])
        )
        cortex = float(
            np.mean(
                [
                    float(tables[1][case][f"dice_lbl_{label}"]) - float(tables[0][case][f"dice_lbl_{label}"])
                    for case in tables[0]
                    for label in (2, 22)
                ]
            )
        )
        complete = not any(row.get("split") == split for row in failures)
        result[split] = {
            "cases": len(tables[0]),
            "macro_REP_delta": delta,
            "cortex_REP_delta": cortex,
            "crosses_predeclared_criterion": bool(complete and delta >= 0.005 and cortex >= -0.001),
            "complete_certification": complete,
        }
    return result


def exact_saved(field, path, epsilon=1 / 1024):
    tick = time.perf_counter()
    save_flow_npz_atomic(path, field)
    stored = load_flow_npz(path)
    cell = certify_flow_exact(stored, eps=epsilon)
    boundary = boundary_report(stored)
    return {
        "sha256": sha256_file(path),
        "cells": cell,
        "boundary": boundary,
        "global": global_verdict(cell, boundary),
        "seconds": time.perf_counter() - tick,
    }


def seqvp_cells(field, module, *, coordinates=None):
    if coordinates is None:
        bound = _trilinear_cell_cert_bound(field).detach().cpu().flatten()
        selected = torch.cat(
            (
                bound.topk(min(16, bound.numel()), largest=False).indices,
                torch.linspace(0, bound.numel() - 1, min(16, bound.numel())).long(),
            )
        ).unique()
        coordinates = [
            [int(v) for v in torch.unravel_index(index, tuple(n - 1 for n in field.shape[2:]))] for index in selected
        ]
    output = []
    for z, y, x in coordinates:
        cube = field[:, :, z : z + 2, y : y + 2, x : x + 2].cpu()
        vertices = []
        for dz, dy, dx in product((0, 1), repeat=3):
            u = cube[0, :, dz, dy, dx].numpy().astype(np.float64)
            # The external cell API adds identity internally and expects displacement, in xyz order.
            vertices.append(u[::-1])
        tick = time.perf_counter()
        theirs = module.certify_cell(
            np.asarray(vertices),
            (1.0, 1.0, 1.0),
            config=module.CertificateConfig(delta=1 / 1024, max_depth=0, max_boxes=1),
        )
        elapsed = time.perf_counter() - tick
        tick = time.perf_counter()
        adaptive = module.certify_cell(
            np.asarray(vertices), (1.0, 1.0, 1.0), config=module.AdaptiveCertificateConfig(delta=1 / 1024)
        )
        adaptive_elapsed = time.perf_counter() - tick
        tick = time.perf_counter()
        ours = certify_flow_exact(cube, eps=1 / 1024)
        output.append(
            {
                "cell_zyx": [z, y, x],
                "SeqVP": asdict(theirs),
                "SeqVP_seconds": elapsed,
                "SeqVP_adaptive": asdict(adaptive),
                "SeqVP_adaptive_seconds": adaptive_elapsed,
                "ours": ours,
                "ours_seconds": time.perf_counter() - tick,
            }
        )
    return output


def certificates(args):
    commit = subprocess.check_output(["git", "-C", str(args.seqvp_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != "ffef26513c88182caf3add70c4b5185bbc6dbb75":
        raise ValueError("SeqVP checkout differs from the reviewed pinned reference")
    sys.path.insert(0, str(args.seqvp_root.resolve()))
    from jd_curl.three_d import feasibility

    from experiments.core.field_scoring import repair_to_identity_boundary

    manifest = read_ixi_manifest(args.manifest, require_t2=False)
    selected = {"subject_557__subject_85", "__".join(manifest["validation_pairs"][0])}
    if len(selected) != 2:
        raise ValueError("Certificate comparison needs two distinct frozen pairs")
    rows, failures = [], []
    for _x, _y, xs, ys, case in cases(SimpleNamespace(dataset="IXI", manifest=args.manifest, limit=0))[0]:
        if case not in selected:
            continue
        labels = [v.unsqueeze(0).to(args.device) for v in (xs, ys)]
        for arm in ("base", "scratch_b1", "paired_solver"):
            destination = args.out / case / arm
            report_path = destination / "report.json"
            if report_path.exists():
                rows.append(read(report_path))
                continue
            raw_path = args.completion_root / "raw" / arm / "pairs" / case / "FF.npz"
            raw = load_flow_npz(raw_path).to(args.device)
            report = {"case_id": case, "arm": arm, "raw_sha256": sha256_file(raw_path), "SeqVP_commit": commit}
            report["comparison_policy"] = {
                "claim_epsilon": "1/1024",
                "local_work_epsilon": 0.0011,
                "collar_width_voxels": 4,
                "cell_selection": "Up to sixteen lowest raw CTCF floating bounds plus sixteen uniformly spaced cells; same coordinates for raw and local. Stress sample, not a whole-map comparison.",
                "comparators": "CTCF >= epsilon; SeqVP > delta. Equality is not an equivalent predicate.",
            }
            report["cell_comparison_raw"] = seqvp_cells(raw.cpu(), feasibility)
            tick = time.perf_counter()
            collared = enforce_identity_boundary(identity_collar(raw, 4))
            local, _, detail = project(collared, tri_iters=640)
            local_seconds = time.perf_counter() - tick
            local_certificate = exact_saved(local, destination / "local.npz")
            report["local"] = {"repair_seconds": local_seconds, "repair": detail, "certificate": local_certificate}
            global_trials, candidate = [], None
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.0):
                trial = collared * alpha
                checked = exact_saved(trial, destination / f"global_{alpha}.npz")
                global_trials.append({"alpha": alpha, "certificate": checked})
                if checked["global"]["global_homeomorphism"]:
                    candidate = trial
                    break
            report["global_reduction"] = global_trials
            report["collar_metrics"] = score(
                collared, case, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI")
            )
            if candidate is None:
                failures.append(f"{case}/{arm}: global reduction could not certify identity")
            for name, field in (("raw", raw), ("local", local), ("global", candidate)):
                if field is None:
                    continue
                change = field - raw
                report[name + "_metrics"] = {
                    **score(field, case, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI")),
                    "change_rms_voxels": float(change.square().mean().sqrt()),
                    "change_max_voxels": float(change.abs().max()),
                    "changed_node_fraction": float((change.abs().amax(1) > 0).float().mean()),
                }
            coordinates = [row["cell_zyx"] for row in report["cell_comparison_raw"]]
            report["cell_comparison_local"] = seqvp_cells(
                load_flow_npz(destination / "local.npz"), feasibility, coordinates=coordinates
            )
            # Include the standard article endpoint separately from the dyadic comparison.
            standard, columns = repair_to_identity_boundary(raw)
            report["standard"] = {
                "repair": columns,
                "certificate": exact_saved(standard, destination / "standard.npz", 1 / 1000),
            }
            atomic_write_json(report_path, report)
            rows.append(report)
            print(f"[CERTIFICATE COMPARISON] {case} {arm}", flush=True)
    atomic_write_json(
        args.out / "report.json",
        {
            "fields": len(rows),
            "rows": rows,
            "failures": failures,
            "scope": "SeqVP check on preselected cells only; not full-map SeqVP or its registration solver. Full-field local/global repair uses our common dyadic predicate, the same collar and saved bytes. Identity reduction is labelled, not nontrivial success. Main protocol epsilon remains 1/1000.",
        },
    )
    if len(rows) != 6 or failures:
        raise RuntimeError("Certificate comparison is incomplete")


def summary(args):
    results, errors, trajectories = {}, [], {}
    try:
        matching_report(
            SimpleNamespace(out=args.out, datasets=("IXI", "LPBA", "OASIS"), required_arms=list(endpoints(args)))
        )
    except (RuntimeError, OSError, ValueError) as error:
        errors.append(repr(error))
    for name in ("b0_s0", "b1_s0", "normalised_t1_s0", "resampled_t1_s0", "shuffled_t2_s0", "b0_s1", "b1_s1"):
        path = args.out / "stable/IXI" / name / "report.json"
        if not path.exists() or not read(path).get("complete_certified_endpoint"):
            errors.append(str(path))
    initials = {}
    for name in ("b0_s0", "b1_s0", "normalised_t1_s0", "resampled_t1_s0", "shuffled_t2_s0", "b0_s1", "b1_s1"):
        path = args.out / "registration" / name / "report.json"
        if path.exists():
            initials[name] = read(path)["initial_model_sha256"]
    for seed in (0, 1):
        if len({value for name, value in initials.items() if name.endswith(f"_s{seed}")}) > 1:
            errors.append(f"Stable controls have different initial model states within seed {seed}")
    for name in initials:
        trajectory = {}
        for epoch in (1, 3, 8, 9, 10):
            path = args.out / "registration" / name / f"epoch{epoch}/per_case.csv"
            if not path.exists():
                errors.append(f"Missing fixed regional observation: {path}")
                continue
            values = read_csv(path)
            if len(values) != 58:
                errors.append(f"Incomplete regional observation: {path}")
                continue
            trajectory[epoch] = {
                "macro_FF": float(np.mean([float(row["dice_mean"]) for row in values])),
                "cortex_FF": float(np.mean([float(row[f"dice_lbl_{label}"]) for row in values for label in (2, 22)])),
            }
        trajectories[name] = trajectory
    for mode in ("FF", "REP"):
        tables = {}
        for name in ("b0_s0", "b1_s0", "normalised_t1_s0", "resampled_t1_s0", "shuffled_t2_s0", "b0_s1", "b1_s1"):
            path = args.out / "stable/IXI" / name / mode / "per_case.csv"
            if path.exists():
                tables[name] = read_csv(path)
        for first, second in (
            ("b1_s0", "b0_s0"),
            ("b1_s1", "b0_s1"),
            ("normalised_t1_s0", "b0_s0"),
            ("resampled_t1_s0", "normalised_t1_s0"),
            ("shuffled_t2_s0", "b1_s0"),
        ):
            if first in tables and second in tables and len(tables[first]) == len(tables[second]) == 58:
                results[f"{mode}_{first}-{second}"] = comparison(
                    tables[first], tables[second], "IXI", certified=mode == "REP"
                )
    atomic_write_json(
        args.out / "closure_summary.json",
        {
            "comparisons": results,
            "incomplete_stable_endpoints": errors,
            "initial_model_sha256": initials,
            "regional_learning_curves": trajectories,
            "scope": "Predeclared ten-epoch endpoints, readouts at 1/3/10. No convergence guarantee or Val-based extension. Other diagnostic blocks carry their own completion status and reports.",
        },
    )
    if errors:
        raise RuntimeError("Stable comparison is incomplete; available comparisons were saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "targets",
            "complete",
            "recovery",
            "profile",
            "inputs",
            "teachers",
            "solvers",
            "regional",
            "certificates",
            "summary",
        ),
    )
    for name in (
        "manifest",
        "source",
        "checkpoint",
        "base_checkpoint",
        "b0_checkpoint",
        "prepared",
        "completion_root",
        "seqvp_root",
        "out",
        "oasis_root",
        "external_manifest",
        "raw_root",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--dataset", choices=("IXI", "LPBA", "OASIS"), default="IXI")
    parser.add_argument("--family", choices=("scratch", "finetune"), default="scratch")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    args = parser.parse_args()
    if args.lanes < 1 or not 0 <= args.lane < args.lanes:
        parser.error("Require 0 <= lane < lanes")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage not in ("preflight", "targets", "summary"):
        configure_precision("bf16", args.device)
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
