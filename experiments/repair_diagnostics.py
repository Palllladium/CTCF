"""Bounded repair experiments on immutable saved canonical fields."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.IXI import read_paired_manifest
from experiments.core.field_scoring import score
from experiments.core.inference_metrics import metric_profile_for
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import SegNorm, pkload, setup_device
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import _M, _cell_corners_exact, certify_flow_exact
from utils.field import (
    _trilinear_cell_cert_bound,
    boundary_vertex_mask,
    digital_project,
    enforce_identity_boundary,
    identity_collar,
    trilinear_project,
)
from utils.field_io import load_flow_npz, save_flow_npz_atomic


def inventory(source):
    paths = []
    for root in (
        "parent",
        "b0",
        "b1",
        "b2",
        "consensus/direct",
        "consensus/smooth",
        "consensus/bank0_k16",
        "consensus/bank1_k16",
    ):
        paths.extend(sorted((source / root / "pairs").glob("*/report.json")))
    cases = []
    for path in paths:
        report = json.loads(path.read_text())
        metrics = json.loads((path.parent / "metrics.json").read_text())
        if "global_REP" not in report:
            raise ValueError(f"Source lacks its exact REP verdict: {path}")
        if report["certificate_REP"]["boundary_nonzero_count"] != 0:
            raise ValueError(f"Source is not the original identity-boundary endpoint: {path}")
        relative = path.parent.relative_to(source).as_posix()
        cases.append(
            {
                "directory": relative,
                "case_id": path.parent.name,
                "dataset": "OASIS" if relative.startswith("consensus/") else "IXI",
                "old_pass": report["global_REP"]["global_homeomorphism"],
                "bound": report["certificate_REP"]["interval_lo_min"],
                "old_tri_iters": int(report["repair"]["tri_proj_iters"]),
                "old_digital_iters": int(report["repair"]["proj_iters"]),
                "report_sha256": sha256_file(path),
                "metrics_sha256": sha256_file(path.parent / "metrics.json"),
                "field_sha256": report["field_sha256"],
                "ff_dice": metrics["FF"]["dice_mean"],
                "old_rep_dice": metrics["REP"]["dice_mean"],
            }
        )
    if len(cases) != 632:
        raise ValueError(f"Expected the complete 632-field certified endpoint, found {len(cases)}")
    failed = [case for case in cases if not case["old_pass"]]
    passed = [case for case in cases if case["old_pass"]]
    controls = sorted(passed, key=lambda c: c["bound"])[:6]
    controls += sorted(
        (c for c in passed if c not in controls), key=lambda c: hashlib.sha256(c["directory"].encode()).hexdigest()
    )[:6]
    representatives = [min(failed, key=lambda c: c["bound"])]
    for subset in ([c for c in failed if c["bound"] > 0], [c for c in failed if c["dataset"] == "OASIS"]):
        if subset:
            representatives.append(min(subset, key=lambda c: c["bound"]))
    for case in cases:
        case["control"] = case in controls
        case["representative"] = case in representatives
    return cases


def normal_band(flow, width):
    """Interpolate normal displacement and taper transverse components; coupled edges can still fold."""
    if width < 1 or any(2 * width >= n for n in flow.shape[2:]):
        raise ValueError("Normal band needs two non-overlapping boundary layers")
    windows = []
    for n in flow.shape[2:]:
        i = torch.arange(n, device=flow.device, dtype=flow.dtype)
        t = torch.minimum(i, n - 1 - i).div(width).clamp(0, 1)
        windows.append(t.square() * (3 - 2 * t))
    out = flow.detach().clone()
    for axis, n in enumerate(flow.shape[2:]):
        component = flow[:, axis : axis + 1]
        low, high = component.select(axis + 2, width), component.select(axis + 2, n - 1 - width)
        for index in range(width + 1):
            out[:, axis : axis + 1].select(axis + 2, index).copy_(low * (index / width))
            out[:, axis : axis + 1].select(axis + 2, n - 1 - index).copy_(high * (index / width))
        for other in range(3):
            if other != axis:
                view = [1, 1, 1, 1, 1]
                view[other + 2] = flow.shape[other + 2]
                out[:, axis : axis + 1].mul_(windows[other].view(view))
    return enforce_identity_boundary(out)


def exact_det(corners, point):
    columns = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        column = [Fraction(0)] * 3
        for bits in itertools.product((0, 1), repeat=2):
            lo = [0, 0, 0]
            weight = Fraction(1)
            for other, bit in zip(others, bits, strict=True):
                lo[other] = bit
                weight *= point[other] if bit else 1 - point[other]
            hi = lo.copy()
            hi[axis] = 1
            column = [
                value + weight * (corners[tuple(hi)][i] - corners[tuple(lo)][i]) for i, value in enumerate(column)
            ]
        columns.append(column)
    a, b, c = columns
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def cell_probe(field, cell):
    corners = _cell_corners_exact(field[0].detach().cpu().numpy(), tuple(cell))
    points = list(itertools.product((Fraction(0), Fraction(1, 2), Fraction(1)), repeat=3))
    values = [exact_det(corners, point) for point in points]
    index = min(range(len(values)), key=values.__getitem__)
    result = {
        "cell": cell,
        "point": [str(x) for x in points[index]],
        "det_fraction": str(values[index]),
        "negative_witness": values[index] < 0,
        "below_epsilon_witness": values[index] < Fraction(1, 1000),
        "scope": "Exact binary32 cell and rational point; absence of a witness does not prove positivity",
    }
    if min(values) >= Fraction(1, 1000):
        tensor = np.asarray(values, dtype=object).reshape(3, 3, 3)
        for axis in range(3):
            tensor = np.apply_along_axis(
                lambda v: np.asarray([sum(_M[i][j] * v[j] for j in range(3)) for i in range(3)], dtype=object),
                axis,
                tensor,
            )
        pieces = [tensor]
        result["subdivision_bounds"] = {"0": str(min(tensor.flat))}
        if min(tensor.flat) >= Fraction(1, 1000):
            return result
        for depth in (1, 2):
            for axis in range(3):
                next_pieces = []
                for block in pieces:
                    left = np.apply_along_axis(
                        lambda v: np.asarray([v[0], (v[0] + v[1]) / 2, (v[0] + 2 * v[1] + v[2]) / 4], dtype=object),
                        axis,
                        block,
                    )
                    right = np.apply_along_axis(
                        lambda v: np.asarray([(v[0] + 2 * v[1] + v[2]) / 4, (v[1] + v[2]) / 2, v[2]], dtype=object),
                        axis,
                        block,
                    )
                    next_pieces.extend((left, right))
                pieces = next_pieces
            result["subdivision_bounds"][str(depth)] = str(min(min(block.flat) for block in pieces))
        result["subdivision_scope"] = "Local exact diagnostic only; not a replacement for the adopted depth0 report"
    return result


@torch.no_grad()
def spatial_report(flow, labels=None, reference=None, band_width=16):
    bound = _trilinear_cell_cert_bound(flow, 0, 0.001)
    bad = ~torch.isfinite(bound) | (bound < 0.001)
    coords = torch.nonzero(bad)
    report = {
        "min_bound_float": float(bound.min()),
        "failed_cells_float": len(coords),
        "boundary_max_displacement": float(flow.masked_select(boundary_vertex_mask(flow).expand_as(flow)).abs().max()),
        "boundary_band_width": band_width,
        "bad_near_multiple_faces": 0,
        "bad_near_three_faces": 0,
    }
    if reference is not None:
        prior = _trilinear_cell_cert_bound(reference, 0, 0.001)
        before = ~torch.isfinite(prior) | (prior < 0.001)
        report["introduced_bad_cells_float"] = int((bad & ~before).sum())
        report["removed_bad_cells_float"] = int((~bad & before).sum())
        if report["introduced_bad_cells_float"]:
            index = torch.where(bad & ~before, bound, torch.inf).argmin().item()
            cell = [int(x) for x in np.unravel_index(index, tuple(bound.shape))]
            report["introduced_probe_before"] = cell_probe(reference, cell)
            report["introduced_probe_after"] = cell_probe(flow, cell)
    if not len(coords):
        return report
    shape = torch.tensor(bound.shape, device=flow.device)
    faces_in_band = (torch.minimum(coords, shape - 1 - coords) < band_width).sum(1)
    report["bad_near_multiple_faces"] = int((faces_in_band >= 2).sum())
    report["bad_near_three_faces"] = int((faces_in_band == 3).sum())
    distances = torch.stack(
        [coords[:, a] if side == 0 else shape[a] - 1 - coords[:, a] for a in range(3) for side in (0, 1)], dim=1
    )
    distance, face = distances.min(1)
    report["nearest_face_counts"] = torch.bincount(face, minlength=6).tolist()
    report["distance_histogram"] = torch.bincount(distance).tolist()
    report["bad_in_first10_layers"] = int((distance < 10).sum())
    vectors = flow[0, :, coords[:, 0], coords[:, 1], coords[:, 2]]
    components = []
    for side in range(6):
        selected = face == side
        if bool(selected.any()):
            normal = vectors[side // 2, selected]
            tangent = (vectors[:, selected].square().sum(0) - normal.square()).clamp_min(0).sqrt()
            components.append(
                {
                    "normal_mean_inward": float(normal.mean()) * (1 if side % 2 == 0 else -1),
                    "tangent_rms": float(tangent.square().mean().sqrt()),
                }
            )
        else:
            components.append(None)
    report["displacement_at_bad_cells_by_face"] = components
    if labels is not None:
        supported = F.max_pool3d((labels > 0).float(), kernel_size=2, stride=1)[0, 0] > 0
        report["bad_cells_touching_fixed_labels"] = int((bad & supported).sum())
    worst = torch.argmin(bound).item()
    cell = list(np.unravel_index(worst, tuple(bound.shape)))
    report["exact_probe"] = cell_probe(flow, [int(x) for x in cell])
    return report


def labels_for(case, manifest, oasis, device):
    subjects = {row["id"]: row for row in manifest["validation"]} if case["dataset"] == "IXI" else None
    labels = []
    for subject in case["case_id"].split("__"):
        path = subjects[subject]["t1"] if subjects is not None else oasis / (subject + ".pkl")
        image, seg = pkload(path)
        if subjects is not None:
            seg = SegNorm()([image, seg])[1]
        labels.append(torch.as_tensor(np.ascontiguousarray(seg), device=device)[None, None])
    return labels


def certify_candidate(field, directory, case, labels, reference=None, *, verify=True):
    started = time.perf_counter()
    verified = {}
    if verify:
        save_flow_npz_atomic(directory / "REP.npz", field)
        stored = load_flow_npz(directory / "REP.npz")
        field = stored.to(field.device)
        verification_started = time.perf_counter()
        cell = certify_flow_exact(stored, eps="0.001")
        verified = {
            "certificate_REP": cell,
            "boundary_REP": boundary_report(stored),
            "certificate_seconds": time.perf_counter() - verification_started,
            "npz_sha256": sha256_file(directory / "REP.npz"),
        }
        verified["global_REP"] = global_verdict(cell, verified["boundary_REP"])
    metrics = score(field, case["case_id"], 0, *labels, 0.001, "canonical", profile=metric_profile_for(case["dataset"]))
    metrics.pop("time_sec")
    result = {
        **verified,
        "metrics": metrics,
        "exact_verification_performed": verify,
        "persist_certify_score_wall_seconds": time.perf_counter() - started,
        "timing_scope": "Saved-field experiment; no network inference latency is claimed",
    }
    if reference is not None:
        difference = field - reference
        result["change_rms_voxels"] = float(difference.square().mean().sqrt())
        result["change_max_voxels"] = float(difference.abs().max())
    atomic_write_json(directory / "report.json", result)
    return result


def project(field, digital_iters=80, tri_iters=640):
    started = time.perf_counter()
    fixed = boundary_vertex_mask(field)
    field, folds, digital = digital_project(field, eps=0, damp=0.6, max_iters=digital_iters, fixed_mask=fixed)
    after_digital = field.clone()
    field, tri = trilinear_project(field, eps=0.0011, damp=0.6, max_iters=tri_iters, fixed_mask=fixed)
    return (
        enforce_identity_boundary(field),
        after_digital,
        {
            "digital_iters": digital,
            "digital_residual": folds,
            "tri_iters": tri.iterations,
            "tri_certified_float": tri.certified,
            "repair_wall_seconds": time.perf_counter() - started,
        },
    )


def repair_one(raw, old, case, labels, directory):
    started = time.perf_counter()
    stages = {
        "FF": spatial_report(raw, labels[1]),
        "old_REP": spatial_report(old, labels[1]) if old is not None else None,
    }
    candidates = []
    selected = None
    current = old.clone() if old is not None else project(identity_collar(raw, 4), tri_iters=80)[0]
    applied = case.get("old_tri_iters", 80)
    for limit in (160, 320, 640):
        current, detail = trilinear_project(
            current, eps=0.0011, damp=0.6, max_iters=limit - applied, fixed_mask=boundary_vertex_mask(current)
        )
        applied += detail.iterations
        stage = spatial_report(current, labels[1])
        stage["tri_iterations_total"] = applied
        stages[f"continue_{limit}"] = stage
        if detail.certified or limit == 640:
            name = f"continue_{limit}"
            result = certify_candidate(current, directory / name, case, labels, raw)
            candidates.append({"name": name, "report": result})
            if result["global_REP"]["global_homeomorphism"]:
                selected = name
                break
    for width in (4, 8, 16):
        collared = identity_collar(raw, width)
        field, after_digital, detail = project(collared, tri_iters=640)
        stages[f"collar_{width}"] = spatial_report(collared, labels[1], raw)
        stages[f"digital_{width}"] = spatial_report(after_digital, labels[1])
        stages[f"project_{width}"] = spatial_report(field, labels[1]) | detail
        if width == 4:
            tri80, _ = trilinear_project(
                after_digital.clone(),
                eps=0.0011,
                damp=0.6,
                max_iters=80,
                fixed_mask=boundary_vertex_mask(after_digital),
            )
            stages["fresh_tri80"] = spatial_report(tri80, labels[1])
            if case.get("representative"):
                for name, value in (("collar4", collared), ("digital80", after_digital), ("tri80", tri80)):
                    save_flow_npz_atomic(directory / "intermediate" / f"{name}.npz", value)
        eligible = stages[f"project_{width}"]["failed_cells_float"] == 0
        verify = bool(case.get("representative") or (selected is None and eligible))
        result = certify_candidate(field, directory / f"width{width}", case, labels, raw, verify=verify)
        candidates.append({"name": f"width{width}", "selection_eligible": eligible, "report": result})
        if selected is None and eligible and result.get("global_REP", {}).get("global_homeomorphism"):
            selected = f"width{width}"
    if case.get("representative"):
        field, _, detail = project(identity_collar(raw, 4), digital_iters=640, tri_iters=640)
        stages["both_stages_640"] = spatial_report(field, labels[1]) | detail
        result = certify_candidate(field, directory / "both640", case, labels, raw)
        candidates.append({"name": "both640", "selection_eligible": False, "report": result})
    if case.get("representative") or selected is None:
        field, _, detail = project(normal_band(raw, 16), tri_iters=640)
        stages["normal16"] = spatial_report(field, labels[1]) | detail
        result = certify_candidate(field, directory / "normal16", case, labels, raw)
        candidates.append({"name": "normal16", "report": result})
        if selected is None and result["global_REP"]["global_homeomorphism"]:
            selected = "normal16"
    if selected is None and case.get("representative"):
        extended = []
        for padding in (8, 16):
            padded = identity_collar(F.pad(raw, (padding,) * 6, mode="replicate"), padding)
            expanded = project(padded, tri_iters=640)[0]
            location = directory / f"extended{padding}"
            save_flow_npz_atomic(location / "REP.npz", expanded)
            stored = load_flow_npz(location / "REP.npz")
            certificate = certify_flow_exact(stored, eps="0.001")
            boundary = boundary_report(stored)
            cropped = expanded[..., padding:-padding, padding:-padding, padding:-padding]
            diagnostic = {
                "padding": padding,
                "scope": "Extended box only; not the original-box C-04 endpoint",
                "certificate": certificate,
                "boundary": boundary,
                "global_extended_box": global_verdict(certificate, boundary),
                "cropped_metrics": score(
                    cropped,
                    case["case_id"],
                    0,
                    *labels,
                    0.001,
                    "canonical",
                    profile=metric_profile_for(case["dataset"]),
                ),
            }
            diagnostic["cropped_metrics"].pop("time_sec")
            atomic_write_json(location / "report.json", diagnostic)
            extended.append(diagnostic)
        stages["extended_box_diagnostics"] = extended
    fallback = selected is None
    if fallback:
        for scale in (0.75, 0.5, 0.25, 0.125, 0.0625, 0.0):
            reduced = enforce_identity_boundary(field * scale)
            if float(_trilinear_cell_cert_bound(reduced, 0, 0.0011).min()) >= 0.0011:
                name = f"fallback_{scale:g}"
                result = certify_candidate(reduced, directory / name, case, labels, raw)
                candidates.append({"name": name, "report": result})
                if result["global_REP"]["global_homeomorphism"]:
                    selected = name
                    break
    if selected is None:
        raise RuntimeError("Even the explicit identity fallback did not certify")
    chosen = next(value["report"] for value in candidates if value["name"] == selected)
    record = {
        "case": case,
        "stages": stages,
        "candidates": candidates,
        "selected": selected,
        "fallback": fallback,
        "selected_metrics": chosen["metrics"],
        "global_REP": chosen["global_REP"],
        "source_sha256": case.get("source_sha256"),
        "whole_search_wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(directory / "report.json", record)
    return record


def load_repair_result(directory, source_sha=None):
    record = json.loads((directory / "report.json").read_text())
    if source_sha is not None and record.get("source_sha256") != source_sha:
        raise ValueError(f"Short-control field changed after repair: {directory}")
    chosen = directory / record["selected"]
    report = json.loads((chosen / "report.json").read_text())
    if sha256_file(chosen / "REP.npz") != report["npz_sha256"] or record["global_REP"] != report["global_REP"]:
        raise ValueError(f"Selected repaired bytes or verdict changed: {directory}")
    return record


def repair_saved(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    manifest = read_paired_manifest(args.manifest, require_t2=False)
    device = setup_device()
    cases = [case for case in inputs["cases"] if not case["old_pass"] or case["control"]]
    errors = []
    for case in cases[args.shard :: args.shards]:
        source = args.source / case["directory"]
        destination = args.out / "repair" / case["directory"].replace("/pairs/", "/")
        if (destination / "report.json").exists():
            load_repair_result(destination)
            continue
        if sha256_file(source / "report.json") != case["report_sha256"]:
            raise ValueError(f"Source report changed: {source}")
        for mode in ("FF", "REP"):
            if sha256_file(source / f"{mode}.npz") != case["field_sha256"][mode]:
                raise ValueError(f"Source field changed: {source}/{mode}")
        raw, old = [load_flow_npz(source / f"{mode}.npz").to(device) for mode in ("FF", "REP")]
        try:
            labels = labels_for(case, manifest, args.oasis_root, device)
            record = repair_one(raw, old, case, labels, destination)
        except (RuntimeError, ValueError, OSError, FloatingPointError) as error:
            errors.append({"case": case["directory"], "error": repr(error)})
            atomic_write_json(destination / "error.json", errors[-1])
            continue
        print(
            f"[REPAIR CASE] {case['directory']} selected={record['selected']} fallback={record['fallback']}", flush=True
        )
    if errors:
        raise RuntimeError(f"{len(errors)} saved-field repairs failed; see their error.json files")


def repair_short(args):
    manifest = read_paired_manifest(args.manifest, require_t2=False)
    device = setup_device()
    reports = []
    epoch = getattr(args, "epoch", 3)
    suffix = "" if epoch == 3 else f"_epoch{epoch}"
    for source in sorted((args.arm_dir / f"epoch{epoch}/pairs").glob("*/FF.npz")):
        case = {"case_id": source.parent.name, "dataset": "IXI", "source_sha256": sha256_file(source)}
        destination = args.arm_dir / f"certified{suffix}" / case["case_id"]
        if (destination / "report.json").exists():
            record = load_repair_result(destination, sha256_file(source))
        else:
            raw = load_flow_npz(source).to(device)
            labels = labels_for(case, manifest, args.oasis_root, device)
            repaired, _, detail = project(identity_collar(raw, 4), tri_iters=80)
            case["old_tri_iters"], case["old_digital_iters"] = detail["tri_iters"], detail["digital_iters"]
            candidate = certify_candidate(repaired, destination / "standard80", case, labels, raw)
            if candidate["global_REP"]["global_homeomorphism"]:
                record = {
                    "case": case,
                    "selected": "standard80",
                    "fallback": False,
                    "selected_metrics": candidate["metrics"],
                    "global_REP": candidate["global_REP"],
                    "source_sha256": case["source_sha256"],
                }
                atomic_write_json(destination / "report.json", record)
            else:
                record = repair_one(raw, repaired, case, labels, destination)
        reports.append(record)
    if len(reports) != 58:
        raise ValueError("Final short-control certification requires all 58 directions")
    atomic_write_json(
        args.arm_dir / f"repair_summary{suffix}.json",
        {
            "directions": len(reports),
            "fallbacks": sum(row["fallback"] for row in reports),
            "identity_fallbacks": sum(row["selected"] == "fallback_0" for row in reports),
            "functional_repair_complete_without_fallback": not any(row["fallback"] for row in reports),
            "rows": [row["selected_metrics"] for row in reports],
            "certified": all(row["global_REP"]["global_homeomorphism"] for row in reports),
        },
    )


def finish(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    rows, unresolved = [], []
    groups, expected_counts = {}, {}
    for case in inputs["cases"]:
        group = case["directory"].split("/pairs/")[0]
        groups.setdefault(group, [])
        expected_counts[group] = expected_counts.get(group, 0) + 1
        source = args.source / case["directory"]
        destination = args.out / "repair" / case["directory"].replace("/pairs/", "/") / "report.json"
        try:
            if sha256_file(source / "REP.npz") != case["field_sha256"]["REP"]:
                raise ValueError(f"Cannot reuse changed certified bytes: {source}")
            if sha256_file(source / "report.json") != case["report_sha256"]:
                raise ValueError(f"Cannot reuse changed source verdict: {source}")
            if sha256_file(source / "metrics.json") != case["metrics_sha256"]:
                raise ValueError(f"Cannot reuse changed source metrics: {source}")
            if case["old_pass"]:
                report = json.loads((source / "report.json").read_text())
                metrics = json.loads((source / "metrics.json").read_text())["REP"]
                row = {"case": case, "reused": True, "metrics": metrics, "global_REP": report["global_REP"]}
            else:
                report = load_repair_result(destination.parent)
                row = {
                    "case": case,
                    "reused": False,
                    "metrics": report["selected_metrics"],
                    "fallback": report["fallback"],
                    "selected": report["selected"],
                    "global_REP": report["global_REP"],
                }
            if not row["global_REP"]["global_homeomorphism"]:
                raise ValueError("Endpoint lacks a global certificate")
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            unresolved.append({"case": case["directory"], "error": repr(error)})
            continue
        rows.append(row)
        groups[group].append(row)
    group_summary = {
        key: {
            "directions": len(values),
            "expected_directions": expected_counts[key],
            "complete": len(values) == expected_counts[key],
            "FF": float(np.mean([r["case"]["ff_dice"] for r in values])) if values else None,
            "old_REP": float(np.mean([r["case"]["old_rep_dice"] for r in values])) if values else None,
            "new_REP": float(np.mean([r["metrics"]["dice_mean"] for r in values])) if values else None,
            "worst_dice_change": min(
                (r["metrics"]["dice_mean"] - r["case"]["old_rep_dice"] for r in values), default=None
            ),
            "fallbacks": sum(r.get("fallback", False) for r in values),
        }
        for key, values in groups.items()
    }
    atomic_write_json(
        args.out / "repair_summary.json",
        {
            "fields": len(inputs["cases"]),
            "resolved_fields": len(rows),
            "complete": not unresolved,
            "unresolved": unresolved,
            "metrics_scope": "Resolved fields only; expected and resolved counts must agree for a complete endpoint",
            "reused": sum(r["reused"] for r in rows),
            "fallbacks": sum(r.get("fallback", False) for r in rows),
            "identity_fallbacks": sum(r.get("selected") == "fallback_0" for r in rows),
            "functional_repair_complete_without_fallback": not unresolved
            and not any(r.get("fallback", False) for r in rows),
            "groups": group_summary,
            "global_certified": sum(r["global_REP"]["global_homeomorphism"] for r in rows),
            "rows": rows,
        },
    )
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} endpoint fields unresolved; partial repair_summary.json was saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("saved", "short", "finish"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oasis-root", type=Path, default=Path("/data/mooncake/P/OASIS_L2R_2021_task03/All"))
    parser.add_argument("--arm-dir", type=Path)
    parser.add_argument("--epoch", type=int, choices=(3, 10), default=3)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Invalid shard")
    {"saved": repair_saved, "short": repair_short, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
