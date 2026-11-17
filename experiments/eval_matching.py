"""Evaluate fixed CTCF fields and T1-only matching refinements with the shared field scorer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from experiments.contrast_probes import match_regularity
from experiments.contrasts import sync
from experiments.core.data_loaders import oasis_subject_paths, oasis_validation_pairs
from experiments.core.field_scoring import load_saved_scores, paired_dice_delta, score, score_saved_field, warp_labels
from experiments.core.inference_metrics import MetricProfile, metric_profile_for, write_results, write_trace
from experiments.core.precision import configure_precision
from experiments.repair_diagnostics import repair_one
from experiments.train_matching import load_base, load_encoder, refine_config
from models.Matching import Matching, base_flow, overlap_support, resize_flow, sample_voxels, voxel_grid
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import Grad3d, pkload
from utils.dice import dice_per_label
from utils.field_io import load_flow_npz, save_flow_npz_atomic
from utils.losses import local_ncc_map


def cases(args):
    if args.dataset == "IXI":
        manifest = read_ixi_manifest(args.manifest, require_t2=False)
        dataset = IXIPairedDataset(manifest, "b0", validation=True)
        profile = metric_profile_for("IXI")
        names = ["__".join(pair) for pair in manifest["validation_pairs"]]
        items = ((x[:1], y[:1], xs, ys, name) for name, (x, y, xs, ys) in zip(names, dataset, strict=True))
        signature = manifest["fingerprint"]
        count = len(names)
    elif args.dataset == "LPBA":
        from datasets.LPBA40 import LPBAPairDataset, read_lpba_manifest

        manifest = read_lpba_manifest(args.manifest)
        dataset = LPBAPairDataset(manifest)
        common = metric_profile_for("IXI")
        profile = MetricProfile(tuple(manifest["label_ids"]), common.jac_metrics, common.jac_log)
        items, signature = iter(dataset), sha256_file(args.manifest)
        count = len(dataset)
    else:
        split = Path("datasets/splits/oasis_subject_disjoint_v2/split_manifest.json")
        _, development = oasis_subject_paths(args.oasis_root, split)
        pairs = oasis_validation_pairs(development, args.manifest)

        def read_pairs():
            for moving, fixed in pairs:
                values = []
                for path in (moving, fixed):
                    subject = pkload(path)
                    image, labels = subject if len(subject) == 2 else (subject[0], subject[2])
                    values.extend(torch.from_numpy(np.ascontiguousarray(a))[None] for a in (image, labels))
                x, xs, y, ys = values
                yield x.float(), y.float(), xs.long(), ys.long(), f"{Path(moving).stem}__{Path(fixed).stem}"

        items, profile, signature = read_pairs(), metric_profile_for("OASIS"), sha256_file(args.manifest)
        count = len(pairs)
    return items, profile, signature, min(count, args.limit) if args.limit else count


def timed_ncc(moving, fixed, parent, seconds, config, *, max_steps=100):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("The NCC time allowance must be finite and nonnegative")
    size = tuple(parent.shape[2:])
    a, b = [F.interpolate(v, size=size, mode="trilinear", align_corners=True) for v in (moving, fixed)]
    start = resize_flow(parent, size)
    current = start.detach().requires_grad_(True)
    grid = voxel_grid(size, device=parent.device)
    support, regularity = overlap_support(a, b, start), Grad3d(penalty="l2")
    upper = parent.new_tensor(size).view(1, 3, 1, 1, 1) - 1
    tick, count, step_seconds = sync(parent.device), 0, 0.0
    while count < max_steps and sync(parent.device) - tick + step_seconds < seconds:
        step_tick = sync(parent.device)
        with torch.enable_grad():
            positions = grid + current
            similarity = local_ncc_map(sample_voxels(a, positions), b, (9, 9, 9), 1e-5)
            outside = (F.relu(-positions).square() + F.relu(positions - upper).square()).sum(1, keepdim=True)
            loss = ((-similarity + config["outside_weight"] * outside) * support).sum() / support.sum().clamp_min(1)
            loss = loss + config["reg_weight"] * regularity(current)
            gradient = torch.autograd.grad(loss, current)[0] * np.prod(size)
            current = (
                (current - config["max_update"] * torch.tanh(config["step_size"] * gradient / config["max_update"]))
                .detach()
                .requires_grad_(True)
            )
        count += 1
        step_seconds = sync(parent.device) - step_tick
    elapsed = sync(parent.device) - tick
    return parent + resize_flow(current.detach() - start, parent.shape[2:]), {
        "steps": count,
        "seconds": elapsed,
        "target_seconds": seconds,
        "time_ratio": elapsed / seconds if seconds > 0 else None,
        "matched_time": bool(seconds > 0 and abs(elapsed - seconds) / seconds <= 0.1),
    }


def certified_scores(flow, case, seconds, xs, ys, directory, profile, dataset):
    started = time.perf_counter()
    options = dict(profile=profile, certify_modes=("REP",), cache=False, repair_columns=False)
    rows, report = score_saved_field(flow, case, seconds, xs, ys, directory, **options)
    recovery = {"used": False, "fallback": False, "selected": "standard80"}
    if dataset == "IXI" and not report["global_homeomorphism"]:
        recovery["used"] = True
        atomic_write_json(directory / "recovery" / "standard_report.json", report)
        try:
            old = load_flow_npz(directory / "REP.npz").to(flow.device) if "REP" in rows else None
            record = repair_one(
                flow,
                old,
                {
                    "dataset": dataset,
                    "case_id": case,
                    "old_tri_iters": report.get("repair", {}).get("tri_proj_iters", 80),
                },
                (xs, ys),
                directory / "recovery",
            )
            chosen = load_flow_npz(directory / "recovery" / record["selected"] / "REP.npz").to(flow.device)
            rows, report = score_saved_field(
                flow,
                case,
                seconds,
                xs,
                ys,
                directory,
                repair_fn=lambda _raw: (chosen, {}),
                **options,
            )
            recovery.update(fallback=bool(record["fallback"]), selected=record["selected"])
            # Recovery includes certificate-based candidate selection, persistence and scoring.
            rows["REP"]["time_sec"] = seconds + time.perf_counter() - started
        except (RuntimeError, ValueError, OSError, FloatingPointError) as error:
            report["errors"].append(f"accepted_repair_recovery: {error!r}")
            recovery["error"] = repr(error)
    if "REP" in rows:
        identity = not bool(torch.count_nonzero(load_flow_npz(directory / "REP.npz")))
        rows["REP"].update(
            recovery_used=int(recovery["used"]),
            recovery_fallback=int(recovery["fallback"]),
            identity_used=int(identity),
        )
    report.update(repair_policy="standard80_then_accepted_extended_IXI_recovery", recovery=recovery)
    atomic_write_json(directory / "metrics.json", rows)
    atomic_write_json(directory / "report.json", report)
    return rows, report


def evaluate(args):
    if args.controls and args.metric_checkpoint is None:
        raise ValueError("Matching controls require a refinement checkpoint")
    if getattr(args, "raw_from", None) is not None and args.controls:
        raise ValueError("Repair-only evaluation does not rerun optional inference controls")
    items, profile, data_hash, expected_count = cases(args)
    binding = {
        "base_sha256": sha256_file(args.base_checkpoint),
        "metric_sha256": sha256_file(args.metric_checkpoint) if args.metric_checkpoint else None,
        "manifest_sha256": data_hash,
        "dataset": args.dataset,
        "limit": args.limit,
        "certify": args.certify,
        "controls": args.controls,
        "expected_cases": expected_count,
        "endpoint_sha256": sha256_file(args.endpoint) if getattr(args, "endpoint", None) else None,
        "repair_policy": "standard80_then_accepted_extended_IXI_recovery" if args.certify else None,
        "descriptor_stride_override": getattr(args, "descriptor_stride", None),
        "solver_steps_override": getattr(args, "solver_steps", None),
        "save_raw": bool(getattr(args, "save_raw", False)),
    }
    if getattr(args, "endpoint", None):
        endpoint = json.loads(args.endpoint.read_text())
        if endpoint["checkpoint_sha256"] != binding["base_sha256"]:
            raise ValueError("Base checkpoint differs from the completed training endpoint")
        if args.dataset == "IXI" and endpoint["source_manifest_sha256"] != data_hash:
            raise ValueError("Endpoint and evaluation use different original validation manifests")
    recipe = args.out / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != binding:
        raise ValueError("Evaluation inputs or policy changed; use another output directory")
    atomic_write_json(recipe, binding)
    raw_from = getattr(args, "raw_from", None)
    cache_from = getattr(args, "cache_from", None)
    for source in (raw_from, cache_from):
        if source is not None:
            previous = json.loads((source / "recipe.json").read_text())
            keys = (
                "base_sha256",
                "metric_sha256",
                "manifest_sha256",
                "dataset",
                "limit",
                "expected_cases",
                "endpoint_sha256",
                "descriptor_stride_override",
                "solver_steps_override",
            )
            if any(previous.get(key) != binding.get(key) for key in keys):
                raise ValueError(f"Cached fields belong to another endpoint or dataset: {source}")
            if source.resolve() == args.out.resolve():
                raise ValueError("Source cache must be a separate output directory")
    encoder, metadata = (
        load_encoder(args.metric_checkpoint, args.device)
        if args.metric_checkpoint and raw_from is None
        else (None, None)
    )
    if metadata and (metadata["kind"] != "student" or metadata["base_sha256"] != binding["base_sha256"]):
        raise ValueError("Refinement checkpoint belongs to another base model")
    if metadata and metadata["step"] != metadata["recipe"]["updates"]:
        raise ValueError("Refinement checkpoint has not completed its fixed training budget")
    config = refine_config(metadata) if metadata else None
    if config and getattr(args, "solver_steps", None) is not None:
        config["steps"] = args.solver_steps
    if encoder is not None:
        encoder.eval().requires_grad_(False)
        if getattr(args, "descriptor_stride", None) is not None:
            encoder.stride = args.descriptor_stride
    base = model = None
    rows, trace, diagnostics = {"FF": [], "REP": [], "NCC": [], "R_MATCH": []}, [], []
    failures, control_failures = [], []
    for index, (x, y, xs, ys, case) in enumerate(items):
        if args.limit and index >= args.limit:
            break
        directory = args.out / "pairs" / case
        cache = directory / "evaluation.json"
        if cache_from is not None and not cache.exists():
            adopt_case(cache_from / "pairs" / case, directory, certified=bool(args.certify))
        if cache.exists():
            saved = json.loads(cache.read_text())
            if not saved["failed"]:
                required_modes = (
                    {"FF"} | ({"REP"} if args.certify else set()) | ({"NCC", "R_MATCH"} if args.controls else set())
                )
                if saved["rows"].keys() != required_modes or any(
                    row["case_id"] != case for row in saved["rows"].values()
                ):
                    raise ValueError(f"Cached evaluation modes/case differ: {case}")
                if args.certify:
                    stored, _ = load_saved_scores(directory)
                    for mode, row in stored.items():
                        if any(saved["rows"][mode].get(key) != value for key, value in row.items()):
                            raise ValueError(f"Cached metrics differ from saved field scores: {case}/{mode}")
                if binding["save_raw"] and saved.get("raw_sha256") != sha256_file(directory / "FF.npz"):
                    raise ValueError(f"Cached raw field changed: {case}")
                for mode, row in saved["rows"].items():
                    rows[mode].append(row)
                trace.extend(saved["trace"])
                diagnostics.append(saved["diagnostics"])
                continue
        x, y, xs, ys = [v.unsqueeze(0).to(args.device) for v in (x, y, xs, ys)]
        imported = (raw_from / "pairs" / case) if raw_from is not None else None
        if imported is not None:
            saved = json.loads((imported / "evaluation.json").read_text())
            raw_path = imported / "FF.npz"
            if saved.get("raw_sha256") != sha256_file(raw_path) or saved["rows"]["FF"]["case_id"] != case:
                raise ValueError(f"Raw field cache is missing or modified: {case}")
            flow = load_flow_npz(raw_path).to(args.device)
            case_trace, diagnostic = saved["trace"], dict(saved["diagnostics"])
            base_seconds, extra_seconds = diagnostic["base_seconds"], diagnostic["extra_seconds"]
            case_rows = {"FF": saved["rows"]["FF"]}
        elif base is None:
            base = load_base(args.base_checkpoint, x.shape[2:], args.device)
            if encoder is not None:
                model = Matching(encoder, config, base=base).eval()
        if imported is None:
            started = sync(args.device)
            parent = model.initial_field(x, y) if model is not None else base_flow(base, x, y)
            if not bool(torch.isfinite(parent).all()):
                raise FloatingPointError(f"Nonfinite base field: {case}")
            base_seconds = sync(args.device) - started
            flow, detail, extra_seconds, case_trace = parent, {}, 0.0, []
            if encoder is not None:
                started = sync(args.device)
                flow, detail = model(x, y, parent, return_diagnostics=True)
                if not bool(torch.isfinite(flow).all()):
                    raise FloatingPointError(f"Nonfinite refined field: {case}")
                extra_seconds = sync(args.device) - started
                for step in (0, 1):
                    probe = parent if step == 0 else model(x, y, parent, steps=1)
                    dice = dice_per_label(warp_labels(xs, probe, "canonical").long(), ys.long(), labels=profile.labels)
                    case_trace.append({"case_id": case, "tto_step": step, "dice_mean": float(np.mean(dice))})
            case_rows = {"FF": score(flow, case, base_seconds + extra_seconds, xs, ys, 0, "canonical", profile=profile)}
            case_trace.append(
                {
                    "case_id": case,
                    "tto_step": config["steps"] if config else 0,
                    "dice_mean": case_rows["FF"]["dice_mean"],
                }
            )
            diagnostic = {"case_id": case, "base_seconds": base_seconds, "extra_seconds": extra_seconds, **detail}
        failed = False
        directory.mkdir(parents=True, exist_ok=True)
        if binding["save_raw"] or args.certify:
            save_flow_npz_atomic(directory / "FF.npz", flow)
        snapshot = {"rows": case_rows, "trace": case_trace, "diagnostics": diagnostic, "failed": True}
        if (directory / "FF.npz").exists():
            snapshot["raw_sha256"] = sha256_file(directory / "FF.npz")
        atomic_write_json(cache, snapshot)
        if args.certify:
            try:
                scored, report = certified_scores(
                    flow, case, base_seconds + extra_seconds, xs, ys, directory, profile, args.dataset
                )
                case_rows.update(scored)
                failed = bool(report["errors"] or not report["global_homeomorphism"])
            except Exception as error:  # Isolate repair failures; retain FF and continue the remaining frozen cases.
                diagnostic["repair_error"] = repr(error)
                failed = True
            if failed:
                failures.append(case)
        # Keep the primary readout if an optional control fails or execution is interrupted.
        atomic_write_json(cache, {"rows": case_rows, "trace": case_trace, "diagnostics": diagnostic, "failed": True})
        if encoder is not None and args.controls:
            try:
                ncc_config = config | (metadata.get("ncc_config") or {})
                ncc_flow, budget = timed_ncc(x, y, parent, extra_seconds, ncc_config)
                case_rows["NCC"] = score(
                    ncc_flow, case, base_seconds + budget["seconds"], xs, ys, 0, "canonical", profile=profile
                )
                regularity = Grad3d(penalty="l2")
                search_started = sync(args.device)
                r_flow, r_detail = match_regularity(parent.cpu(), float(regularity(flow.detach().cpu())))
                r_flow = r_flow.to(args.device)
                search_seconds = sync(args.device) - search_started
                r_detail.update(
                    search_seconds=search_seconds,
                    timing_scope="CPU target regularity, Gaussian search and field transfers",
                )
                case_rows["R_MATCH"] = score(
                    r_flow, case, base_seconds + search_seconds, xs, ys, 0, "canonical", profile=profile
                )
                diagnostic.update(ncc_budget=budget, ncc_config=ncc_config, smoothing=r_detail)
            except (RuntimeError, ValueError, OSError, FloatingPointError) as error:
                diagnostic["control_error"] = repr(error)
                control_failures.append(case)
        if (directory / "FF.npz").exists():
            snapshot["raw_sha256"] = sha256_file(directory / "FF.npz")
        atomic_write_json(cache, snapshot | {"failed": failed or case in control_failures})
        for mode, row in case_rows.items():
            rows[mode].append(row)
        trace.extend(case_trace)
        diagnostics.append(diagnostic)
        print(f"[MATCHING CASE] {args.dataset} {index + 1} {case}", flush=True)
    for mode, values in rows.items():
        if values:
            destination = args.out / mode
            destination.mkdir(exist_ok=True)
            write_results(
                values,
                str(destination),
                args.out.name,
                str(args.metric_checkpoint or args.base_checkpoint),
                args.dataset,
            )
            if args.dataset == "LPBA":
                summary_path = destination / "summary.json"
                summary = json.loads(summary_path.read_text())
                summary["scope"] = (
                    "Descriptive case dispersion; graph pairs share subjects, no independent-case interval"
                )
                for metric in summary["metrics"].values():
                    for key in ("sem", "ci95"):
                        metric.pop(key, None)
                atomic_write_json(summary_path, summary)
                with (destination / "summary.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("metric", "mean", "std", "min", "max"))
                    writer.writerows(
                        (key, *(metric[name] for name in ("mean", "std", "min", "max")))
                        for key, metric in summary["metrics"].items()
                    )
    write_trace(trace, str(args.out))
    atomic_write_json(args.out / "diagnostics.json", {"cases": diagnostics})
    atomic_write_json(
        args.out / "report.json",
        {
            "cases": len(rows["FF"]),
            "failures": failures,
            "control_failures": control_failures,
            "binding": binding,
            "complete_certified_endpoint": bool(args.certify and len(rows["REP"]) == expected_count and not failures),
        },
    )
    if len(rows["FF"]) != expected_count:
        raise RuntimeError("Evaluation is missing frozen cases")
    if failures:
        raise RuntimeError("Raw scores saved; some repaired fields lack a global certificate")
    if control_failures:
        raise RuntimeError("Primary scores saved; some optional controls failed")


def adopt_case(source, destination, *, certified):
    """Reuse saved field bytes, never modifying the source attempt's JSON or arrays."""
    if not (source / "FF.npz").is_file() or not (source / "report.json").is_file():
        return
    report = json.loads((source / "report.json").read_text())
    if report.get("field_sha256", {}).get("FF") != sha256_file(source / "FF.npz"):
        raise ValueError(f"Source FF bytes do not match their report: {source}")
    saved = json.loads((source / "evaluation.json").read_text()) if (source / "evaluation.json").exists() else {}
    rows = json.loads((source / "metrics.json").read_text())
    if certified:
        if not report.get("global_homeomorphism") or report.get("errors"):
            return
        load_saved_scores(source)
    modes = ("FF", "REP") if certified else ("FF",)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name in ("metrics.json", "report.json"):
            shutil.copy2(path, destination / path.name)
        elif path.name in {f"{mode}.npz" for mode in modes}:
            target = destination / path.name
            if target.exists():
                if sha256_file(path) != sha256_file(target):
                    raise ValueError(f"Interrupted cache adoption left different field bytes: {target}")
                continue
            try:
                os.link(path, target)
            except OSError:
                temporary = target.with_name(f".{target.name}.tmp")
                shutil.copy2(path, temporary)
                temporary.replace(target)
    atomic_write_json(
        destination / "evaluation.json",
        {
            "rows": {mode: rows[mode] for mode in modes},
            "trace": saved.get("trace", []),
            "diagnostics": saved.get(
                "diagnostics",
                {
                    "base_seconds": rows["FF"]["time_sec"],
                    "extra_seconds": 0,
                    "timing_scope": "historical total; component times unavailable",
                },
            ),
            "failed": False,
            "raw_sha256": sha256_file(destination / "FF.npz"),
        },
    )


def comparison(first, second, dataset, *, certified=False):
    result = paired_dice_delta(first, second)
    result["threshold_crossed"] = result["resource_signal"]
    result["resource_signal"] = bool(certified and result["threshold_crossed"])
    result["certified_comparison"] = certified
    if dataset == "LPBA":
        result.pop("ci95", None)
        result["scope"] = "Descriptive graph-pair average; graph pairs share subjects, no independent-pair interval"
    tables = [{row["case_id"]: row for row in rows} for rows in (first, second)]
    columns = sorted(
        key for key in first[0] if key.startswith("dice_lbl_") or key in ("sdlogj", "sdlogj_plus3", "ndv_percent")
    )
    result["metric_deltas"] = {
        key: float(np.mean([float(tables[0][case][key]) - float(tables[1][case][key]) for case in tables[0]]))
        for key in columns
        if all(key in row for rows in tables for row in rows.values())
    }
    if dataset == "IXI":
        groups = {"cortex": (2, 22), "deep_gray": (7, 8, 9, 10, 27, 28, 29, 30), "cerebellum": (5, 6, 25, 26)}
        result["region_deltas"] = {
            name: {
                "labels": labels,
                "mean": float(np.mean([result["metric_deltas"][f"dice_lbl_{label}"] for label in labels])),
            }
            for name, labels in groups.items()
            if all(f"dice_lbl_{label}" in result["metric_deltas"] for label in labels)
        }
    return result


def report(args):
    result, errors = {}, []
    for dataset in args.datasets:
        endpoints = {}
        for arm in args.required_arms:
            path = args.out / dataset / arm / "report.json"
            if path.exists():
                endpoints[arm] = json.loads(path.read_text())
            else:
                errors.append(f"Missing endpoint: {path}")
        for mode in ("FF", "REP") if dataset == "IXI" else ("FF",):
            tables = {}
            for arm in args.required_arms:
                path = args.out / dataset / arm / mode / "per_case.csv"
                try:
                    with path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    endpoint = endpoints.get(arm, {})
                    expected = endpoint.get("binding", {}).get("expected_cases")
                    if not rows or expected != len(rows):
                        raise ValueError(f"Incomplete readout: expected {expected}, found {len(rows)}")
                    if mode == "REP" and not endpoint.get("complete_certified_endpoint"):
                        raise ValueError("Endpoint is not completely certified")
                    tables[arm] = rows
                except (OSError, ValueError) as error:
                    errors.append(f"{path}: {error}")
            pairs = {
                ("paired_solver", "paired_value"),
                ("paired_value", "t1_value"),
                ("paired_solver", "t1_solver"),
                ("t1_solver", "t1_value"),
                ("scratch_b1", "scratch_b0"),
                ("paired_solver_small", "paired_solver"),
                ("paired_solver_half", "paired_solver"),
            }
            pairs |= {(arm, control) for arm in tables for control in ("base", "parent") if arm != control}
            for first, second in sorted(pairs):
                if first in tables and second in tables:
                    try:
                        result[f"{dataset}_{mode}_{first}-{second}"] = comparison(
                            tables[first], tables[second], dataset, certified=mode == "REP"
                        )
                    except ValueError as error:
                        errors.append(f"{dataset}/{mode}/{first}-{second}: {error}")
            matrix = ("paired_solver", "paired_value", "t1_solver", "t1_value")
            if all(arm in tables for arm in matrix):
                try:
                    values = [{row["case_id"]: row for row in tables[arm]} for arm in matrix]
                    if any(table.keys() != values[0].keys() for table in values):
                        raise ValueError("Interaction requires the same case IDs in all four arms")
                    contrasts = [
                        [
                            {"case_id": case, "dice_mean": float(a[case]["dice_mean"]) - float(b[case]["dice_mean"])}
                            for case in a
                        ]
                        for a, b in ((values[0], values[1]), (values[2], values[3]))
                    ]
                    result[f"{dataset}_{mode}_interaction"] = comparison(*contrasts, dataset, certified=mode == "REP")
                except ValueError as error:
                    errors.append(f"{dataset}/{mode}/interaction: {error}")
            if mode == "FF" and "paired_solver" in tables:
                for control in ("NCC", "R_MATCH"):
                    path = args.out / dataset / "paired_solver" / control / "per_case.csv"
                    if not path.exists():
                        if endpoints.get("paired_solver", {}).get("binding", {}).get("controls"):
                            errors.append(f"Missing requested control: {path}")
                        continue
                    try:
                        with path.open(newline="") as stream:
                            control_rows = list(csv.DictReader(stream))
                        delta = comparison(tables["paired_solver"], control_rows, dataset)
                        diagnostics = json.loads((path.parent.parent / "diagnostics.json").read_text())["cases"]
                        key, flag = ("ncc_budget", "matched_time") if control == "NCC" else ("smoothing", "matched")
                        delta["control_matched_fraction"] = float(
                            np.mean([bool(row.get(key, {}).get(flag)) for row in diagnostics])
                        )
                        delta["all_controls_matched"] = delta["control_matched_fraction"] == 1.0
                        delta["control_scope"] = (
                            "Timing or diffusion regularity matching only; mismatches are not a controlled exclusion"
                        )
                        result[f"{dataset}_{mode}_paired_solver-{control}"] = delta
                    except (OSError, ValueError, KeyError) as error:
                        errors.append(f"{path}: {error}")
    if not result:
        errors.append("No complete comparisons")
    atomic_write_json(
        args.out / "matching_summary.json",
        {
            "comparisons": result,
            "errors": errors,
            "scope": "One initialization; common base is already T2-finetuned. T1 source isolates the additional teacher contrast, not a T2-free training history.",
        },
    )
    if errors:
        raise RuntimeError("Some matching comparisons are incomplete; available results were saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "report"))
    parser.add_argument("--dataset", choices=("IXI", "LPBA", "OASIS"), default="IXI")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--oasis_root", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--metric_checkpoint", type=Path)
    parser.add_argument("--descriptor_stride", type=int, choices=(1, 2))
    parser.add_argument("--solver_steps", type=int, choices=(0, 1, 3))
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--raw_from", type=Path)
    parser.add_argument("--cache_from", type=Path)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--certify", type=int, choices=(0, 1), default=0)
    parser.add_argument("--controls", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--datasets", nargs="+", choices=("IXI", "LPBA", "OASIS"), default=("IXI", "LPBA", "OASIS"))
    parser.add_argument(
        "--required_arms",
        nargs="+",
        default=(
            "base",
            "parent",
            "fine_b0",
            "scratch_b0",
            "scratch_b1",
            "resampled_t1",
            "shuffled_t2",
            "t1_value",
            "t1_solver",
            "paired_value",
            "paired_solver",
            "paired_solver_small",
            "paired_solver_half",
        ),
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.stage == "evaluate" and (args.manifest is None or args.base_checkpoint is None):
        parser.error("evaluate needs --manifest and --base_checkpoint")
    if args.stage == "evaluate" and args.dataset == "OASIS" and args.oasis_root is None:
        parser.error("OASIS evaluation needs --oasis_root")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report(args)
    else:
        configure_precision("bf16", args.device)
        evaluate(args)


if __name__ == "__main__":
    main()
