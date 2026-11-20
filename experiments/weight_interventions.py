"""Label-guided weight interventions with subject-disjoint fitting, selection and readout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from datasets.LPBA40 import read_lpba_manifest
from experiments.contrast_audit import region_weight
from experiments.contrast_probes import edited_state, match_regularity
from experiments.core.field_scoring import load_saved_scores, repair_to_identity_boundary, score, warp_labels
from experiments.core.inference_metrics import metric_profile_for
from experiments.core.precision import configure_precision
from experiments.core.weight_editing import (
    CORTEX,
    HEADS,
    error_counts,
    feature_moments,
    fit_direction,
    group_mean,
    joint_criterion,
    matched_features,
    project_weight,
    projection_report,
    protected_weight,
)
from experiments.eval_matching import cases, certified_scores
from experiments.repair_diagnostics import project
from experiments.train_matching import load_base
from models.Matching import base_flow
from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, sha256_file
from utils import Grad3d
from utils.field import _trilinear_cell_cert_bound, _warp
from utils.field_io import load_flow_npz
from utils.losses import stable_ncc_map


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen inputs or procedure changed: {path}")
    atomic_write_json(path, value)


def split_subjects(ids, previously_examined, seed=20261120):
    if len(ids) != 397 or len(set(ids)) != 397:
        raise ValueError("This experiment requires the common 397-subject IXI Train cohort")
    order = sorted(ids, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    sealed = [name for name in order if name not in previously_examined][:80]
    if len(sealed) != 80:
        raise ValueError("Too few subjects remain outside known previous anatomical probes")
    remaining = [name for name in order if name not in sealed]
    return {"discovery": remaining[80:], "selection": remaining[:80], "sealed": sealed}


def reciprocal_pairs(ids):
    return [[a, b] for i in range(0, len(ids) - 1, 2) for a, b in (ids[i : i + 2], ids[i : i + 2][::-1])]


def prepare(args):
    manifest = read_ixi_manifest(args.manifest, verify_bytes=True)
    external = read_lpba_manifest(args.external_manifest, verify_bytes=True)
    known, history = set(), {}
    paths = [args.history_run / "probe_inputs.json", args.source / "teacher_audit/report.json"]
    paths += [args.source / "regional" / family / "report.json" for family in ("finetune", "scratch")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"History needed to define the sealed subjects: {path}")
        document = read(path)
        history[str(path)] = sha256_file(path)
        for row in document.get("train_pairs", []):
            known.update(row)
        for row in document.get("rows", []):
            if row.get("split", "Train") == "Train":
                known.update(row.get("case_id", "").split("__"))
    groups = split_subjects([row["id"] for row in manifest["training"]], known)
    weights, initial = {}, {}
    files = {
        str(args.manifest): manifest["fingerprint"],
        str(args.external_manifest): external["fingerprint"],
        str(args.checkpoint): sha256_file(args.checkpoint),
        str(args.oasis_pairs): sha256_file(args.oasis_pairs),
        "datasets/splits/oasis_subject_disjoint_v2/split_manifest.json": sha256_file(
            Path("datasets/splits/oasis_subject_disjoint_v2/split_manifest.json")
        ),
    }
    for seed in (0, 1):
        for arm in ("b0", "b1"):
            name = f"{arm}_s{seed}"
            folder = args.source / "registration" / name
            endpoint, recipe = read(folder / "report.json"), read(folder / "recipe.json")
            options = endpoint["arguments"]
            if (
                any(
                    options[k] != value
                    for k, value in {
                        "seed": seed,
                        "ixi_arm": arm,
                        "ncc_mode": "stable",
                        "precision": "bf16",
                        "w_reg": 1,
                        "w_dice": 0,
                        "image_channels": 2,
                        "finetune": 1,
                    }.items()
                )
                or endpoint["updates"] != 3990
            ):
                raise ValueError(f"The source is not the fixed stable ten-epoch endpoint: {name}")
            if (
                recipe["parent_sha256"] != files[str(args.checkpoint)]
                or recipe["manifest_sha256"] != manifest["fingerprint"]
            ):
                raise ValueError("Source training parent/cohort differs")
            path = Path(endpoint["checkpoint"])
            if sha256_file(path) != endpoint["checkpoint_sha256"]:
                raise ValueError(f"Source checkpoint changed: {path}")
            weights[name] = str(path)
            initial[name] = recipe["initial_model_sha256"]
            for p in (path, folder / "report.json", folder / "recipe.json"):
                files[str(p)] = sha256_file(p)
        if initial[f"b0_s{seed}"] != initial[f"b1_s{seed}"]:
            raise ValueError("B0/B1 did not start from the same expanded parent")
    for family, names in (("stable/IXI", ("b0_s0", "b1_s0", "b0_s1", "b1_s1")), ("IXI", ("scratch_b0", "scratch_b1"))):
        for name in names:
            path = args.source / family / name / "report.json"
            endpoint = read(path)
            if endpoint.get("cases") != len(manifest["validation_pairs"]) or not endpoint.get(
                "complete_certified_endpoint"
            ):
                raise ValueError(f"Oracle source evaluation is incomplete: {path}")
            if name in weights and endpoint["binding"]["base_sha256"] != files[weights[name]]:
                raise ValueError(f"Oracle fields do not belong to the frozen checkpoint: {name}")
            if name in weights and endpoint["binding"]["manifest_sha256"] != manifest["fingerprint"]:
                raise ValueError(f"Oracle fields use another validation cohort: {name}")
            files[str(path)] = sha256_file(path)
    pairs = {key: reciprocal_pairs(ids[:40] if key == "discovery" else ids) for key, ids in groups.items()}
    freeze(
        args.out / "inputs.json",
        {
            "files": files,
            "history": history,
            "weights": weights,
            "initial": initial,
            "manifest": str(args.manifest),
            "external_manifest": str(args.external_manifest),
            "parent": str(args.checkpoint),
            "shape": manifest["shape"],
            "scope": "Label-guided diagnostic; sealed Train subjects were seen without labels in pretraining, not a new project-level test",
        },
    )
    freeze(
        args.out / "splits.json",
        {"subjects": groups, "pairs": pairs, "previous_anatomical_probes": sorted(known - {""}), "seed": 20261120},
    )
    text = "\n".join(groups["discovery"]) + "\n"
    ids_file = args.out / "discovery_ids.txt"
    if ids_file.exists() and ids_file.read_text() != text:
        raise ValueError("Discovery label permissions changed")
    atomic_write_text(ids_file, text)


def verify(args):
    inputs = read(args.out / "inputs.json")
    for path, digest in inputs["files"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return inputs


def data(args, split, *, lane=0, lanes=1):
    if split in ("discovery", "selection", "sealed"):
        manifest = read_ixi_manifest(args.manifest, require_t2=False)
        pairs = read(args.out / "splits.json")["pairs"][split][lane::lanes]
        local = copy.deepcopy(manifest)
        local["validation"], local["validation_pairs"] = local["training"], pairs
        dataset = IXIPairedDataset(local, "b0", validation=True)
        return (
            (x[:1], y[:1], xs, ys, "__".join(pair)) for pair, (x, y, xs, ys) in zip(pairs, dataset, strict=True)
        ), metric_profile_for("IXI")
    options = SimpleNamespace(
        dataset=split,
        manifest={"IXI": args.manifest, "LPBA": args.external_manifest, "OASIS": args.oasis_pairs}[split],
        oasis_root=args.oasis_root,
        limit=0,
    )
    items, profile, _, _ = cases(options, lane=lane, lanes=lanes)
    return items, profile


def tensors(values, device):
    return [value[None].to(device) for value in values]


def measure(field, case, x, y, xs, ys, profile, *, row=None):
    if not torch.isfinite(field).all():
        raise FloatingPointError(f"Non-finite field: {case}")
    row = dict(row) if row is not None else score(field, case, 0, xs, ys, 0, "canonical", profile=profile)
    row["regularity"] = float(Grad3d(penalty="l2")(field))
    row["ncc"] = float(stable_ncc_map(_warp(x[:, :1], field), y[:, :1]).mean())
    if profile.labels == metric_profile_for("IXI").labels:
        row.update(error_counts(warp_labels(xs, field, "canonical").long(), ys))
    return row


def provisional_repair(field):
    repaired, _ = repair_to_identity_boundary(field)
    passed = bool((_trilinear_cell_cert_bound(repaired) > 0.001).all())
    return repaired, passed


def boundary_repair(field):
    repaired, _, detail = project(field, tri_iters=80)
    return repaired, {
        **detail,
        "tri_proj_iters": detail["tri_iters"],
        "proj_iters": detail["digital_iters"],
        "scope": "Repair an already identity-boundary blend without a second collar",
    }


@torch.no_grad()
def collect(args):
    inputs, moments, reports = verify(args), {key: [] for key in HEADS}, []
    models = [load_base(inputs["weights"][f"{arm}_s{args.seed}"], inputs["shape"], args.device) for arm in ("b0", "b1")]
    features, snippets, hooks = {}, {}, []
    for key, path in HEADS.items():

        def capture(_module, values, name=key):
            features[name] = values[0].detach()

        hooks.append(models[1].get_submodule(path).register_forward_pre_hook(capture))
    try:
        for index, (x, y, xs, ys, case) in enumerate(data(args, "discovery")[0]):
            x, y, xs, ys = tensors((x, y, xs, ys), args.device)
            first, second = [base_flow(model, x, y) for model in models]
            predictions = [
                torch.isin(warp_labels(xs, field, "canonical").long(), xs.new_tensor(CORTEX))
                for field in (first, second)
            ]
            truth = torch.isin(ys, ys.new_tensor(CORTEX))
            error = predictions[1] & ~predictions[0] & ~truth
            correct = ~predictions[1] & ~predictions[0] & ~truth
            counts = {}
            for head in HEADS:
                samples = matched_features(features[head], error, correct, y, ys, seed=20261120 + index)
                counts[head] = len(samples[0]) if samples else 0
                if samples:
                    moments[head].append(feature_moments(*samples))
                    snippets[head] = features[head][:, :, :8, :8, :8].clone()
            reports.append({"case_id": case, "new_cortex_fp": int(error.sum()), "matched": counts})
            features.clear()
            print(f"[DISCOVERY FEATURES] seed={args.seed} {index + 1}/40 {case} {counts}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    directions, diagnostics = {}, {}
    for head in HEADS:
        direction = fit_direction(moments[head])
        directions[head] = direction.tolist() if direction is not None else None
        diagnostics[head] = {
            "pairs": len(moments[head]),
            "matched_points": sum(m["count"] for m in moments[head]),
            "status": "FITTED" if direction is not None else "NO_DIRECTION",
        }
        if direction is not None:
            contrasts = [(m["means"][0] - m["means"][1]) @ direction.double() for m in moments[head]]
            diagnostics[head]["mean_error_minus_control_projection"] = float(torch.stack(contrasts).mean())
            diagnostics[head]["compilation"] = projection_report(
                models[1].get_submodule(HEADS[head]), snippets[head], direction
            )
    atomic_write_json(
        args.out / f"directions_s{args.seed}.json",
        {
            "directions": directions,
            "diagnostics": diagnostics,
            "rows": reports,
            "ridge_trace_fraction": 0.001,
            "weighting": "Equal directional-pair contributions; intensity/distance matched; adaptive head footprints, not exact full-grid feature correspondence",
            "inputs_sha256": sha256_file(args.out / "inputs.json"),
            "splits_sha256": sha256_file(args.out / "splits.json"),
        },
    )


def save_weights(path, source, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        key: source[key]
        for key in ("arguments", "precision", "ncc_mode", "input_representation", "intervention")
        if key in source
    }
    saved["state_dict"] = state
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def edits(args):
    inputs, rows = verify(args), []
    fitted = {seed: read(args.out / f"directions_s{seed}.json")["directions"] for seed in (0, 1)}
    for seed in (0, 1):
        directions = fitted[seed]
        saved = torch.load(inputs["weights"][f"b1_s{seed}"], map_location="cpu", weights_only=False)
        state = saved["state_dict"]
        for head, path in HEADS.items():
            if any(fitted[s][head] is None for s in (0, 1)):
                continue
            weight = state[f"{path}.weight"]
            direction = torch.tensor(directions[head])
            for strength in (0.5, 1.0):
                target = project_weight(weight, direction, strength)
                norm = (target - weight).double().norm()
                for random in (-1, 0, 1, 2):
                    vector = direction
                    scale = strength
                    if random >= 0:
                        vector = torch.randn(
                            len(direction),
                            generator=torch.Generator().manual_seed(
                                20261120 + 100 * seed + 10 * list(HEADS).index(head) + random
                            ),
                        )
                        unit_norm = (project_weight(weight, vector, 1) - weight).double().norm()
                        if unit_norm <= 0:
                            raise ValueError("Random direction has zero effect on the head")
                        scale = float(norm / unit_norm)
                    name = f"{head}_{int(100 * strength)}_{'target' if random < 0 else f'random{random}'}"
                    updated = dict(state)
                    updated[f"{path}.weight"] = project_weight(weight, vector, scale)
                    output = args.out / "weights" / f"{name}_s{seed}.pth"
                    annotation = {
                        "kind": "rank_one_head_projection",
                        "head": head,
                        "direction": vector.tolist(),
                        "strength": scale,
                        "random": random,
                        "label_guidance": True,
                        "scope": "Discovery labels fit the target direction and its matched edit norm; labelled selection chooses the final procedure. Original arguments describe source B1 training.",
                    }
                    save_weights(output, saved | {"intervention": annotation}, updated)
                    rows.append(
                        {
                            "name": name,
                            "seed": seed,
                            "head": head,
                            "strength": strength,
                            "random": random,
                            "actual_strength": scale,
                            "edit_norm": float((updated[f"{path}.weight"] - weight).double().norm()),
                            "target_edit_norm": float(norm),
                            "checkpoint": str(output),
                            "sha256": sha256_file(output),
                        }
                    )
    freeze(
        args.out / "edits.json",
        {
            "rows": rows,
            "scope": "Uncentred rank-one input-channel erasure; bias unchanged; random tensor-edit Frobenius norm matched to each target",
        },
    )


@torch.no_grad()
def evaluate_model(args, name, checkpoint, split, *, exact=False, controls=False, lane=0, lanes=1):
    directory = args.out / "readouts" / split / name
    freeze(
        directory / "inputs.json",
        {
            "checkpoint_sha256": sha256_file(checkpoint),
            "splits_sha256": sha256_file(args.out / "splits.json"),
            "root_inputs_sha256": sha256_file(args.out / "inputs.json"),
            "exact": exact,
            "controls": controls,
        },
    )
    model, baseline, failures = None, None, []
    profile = data(args, split)[1]
    for index, (x, y, xs, ys, case) in enumerate(data(args, split, lane=lane, lanes=lanes)[0]):
        path = directory / "pairs" / case / "readout.json"
        if path.exists() and read(path).get("complete"):
            if exact:
                adopted = read(path).get("adopted_from")
                load_saved_scores(Path(adopted) if adopted else path.parent)
                if controls:
                    for label in ("amplitude", "gaussian"):
                        load_saved_scores(path.parent / label)
            continue
        x, y, xs, ys = tensors((x, y, xs, ys), args.device)
        if split == "IXI" and name in ("b0_s0", "b1_s0", "b0_s1", "b1_s1"):
            source = args.source / "stable/IXI" / name / "pairs" / case
            rows, certificate = load_saved_scores(source)
            for mode in ("FF", "REP"):
                values = measure(load_flow_npz(source / f"{mode}.npz").to(args.device), case, x, y, xs, ys, profile)
                values.pop("time_sec")
                rows[mode].update(values)
            complete = bool(certificate["global_homeomorphism"] and not certificate["errors"])
            atomic_write_json(
                path,
                {
                    "rows": rows,
                    "certificate": certificate,
                    "controls": {},
                    "complete": complete,
                    "adopted_from": str(source),
                },
            )
            if not complete:
                failures.append(case)
            continue
        if model is None:
            model = load_base(checkpoint, x.shape[2:], args.device)
        field = base_flow(model, x, y)
        raw = measure(field, case, x, y, xs, ys, profile)
        if split in ("sealed", "IXI") and name not in ("b0_s0", "b1_s0", "b0_s1", "b1_s1"):
            seed = int(name.rsplit("_s", 1)[1])
            baseline_pair = args.out / "readouts" / split / f"b1_s{seed}" / "pairs" / case
            origin = read(baseline_pair / "readout.json").get("adopted_from")
            reference_raw = load_flow_npz((Path(origin) if origin else baseline_pair) / "FF.npz").to(args.device)
            difference = field - reference_raw
            raw.update(
                delta_b1_rms_voxels=float(difference.square().mean().sqrt()),
                delta_b1_max_voxels=float(difference.abs().max()),
            )
        atomic_write_json(
            path, {"rows": {"FF": raw}, "certificate": {"status": "PENDING"}, "controls": {}, "complete": False}
        )
        if exact:
            rows, certificate = certified_scores(field, case, 0, xs, ys, path.parent, profile, "IXI")
            if "REP" in rows:
                repaired = load_flow_npz(path.parent / "REP.npz").to(args.device)
                values = measure(repaired, case, x, y, xs, ys, profile, row=rows["REP"])
                values.pop("time_sec")
                rows["REP"].update(values)
            complete = bool(certificate["global_homeomorphism"] and not certificate["errors"])
        elif split in ("selection", "discovery"):
            repaired, passed = provisional_repair(field)
            rows = {"FF": raw, "REP_FLOAT_UNCERTIFIED": measure(repaired, case, x, y, xs, ys, profile)}
            rows["REP_FLOAT_UNCERTIFIED"]["float_bound_passed"] = int(passed)
            certificate, complete = (
                {"float_bound_passed": passed, "scope": "Provisional selection only; no exact/global certificate"},
                True,
            )
        else:
            rows, certificate, complete = {"FF": raw}, {}, True
        rows["FF"] = raw
        common = {}
        if controls:
            seed = int(name.rsplit("_s", 1)[1])
            if baseline is None:
                inputs = read(args.out / "inputs.json")
                baseline = load_base(inputs["weights"][f"b1_s{seed}"], x.shape[2:], args.device)
            reference = base_flow(baseline, x, y)
            before = float(Grad3d(penalty="l2")(reference))
            target = min(raw["regularity"], before)
            smoothed, detail = match_regularity(reference.cpu(), target)
            amplitude = reference * (np.sqrt(target / before) if before > 0 else 1)
            common = {
                "target_R": target,
                "gaussian_match": detail,
                "amplitude": measure(amplitude, case, x, y, xs, ys, profile),
                "gaussian": measure(smoothed.to(args.device), case, x, y, xs, ys, profile),
                "scope": "FF-only controls; chosen per field by diffusion R, never by Dice; target is min(edit R, B1 R)",
            }
            if exact:
                common["scope"] = "R chosen without labels; final FF and exact-certified REP controls"
                for label, control in (("amplitude", amplitude), ("gaussian", smoothed.to(args.device))):
                    control_rows, proof = certified_scores(
                        control, case, 0, xs, ys, path.parent / label, profile, "IXI"
                    )
                    complete &= proof["global_homeomorphism"] and not proof["errors"]
                    if "REP" in control_rows:
                        stored = load_flow_npz(path.parent / label / "REP.npz").to(args.device)
                        values = measure(stored, case, x, y, xs, ys, profile, row=control_rows["REP"])
                        values.pop("time_sec")
                        common[f"{label}_REP"] = control_rows["REP"] | values
                    common[f"{label}_certified"] = bool(proof["global_homeomorphism"] and not proof["errors"])
        atomic_write_json(path, {"rows": rows, "certificate": certificate, "controls": common, "complete": complete})
        print(f"[WEIGHT READOUT] {split} {name} {index + 1} {case} complete={complete}", flush=True)
        if not complete:
            failures.append(case)
    if failures:
        raise RuntimeError(f"Final certification failed for {name}: {failures}; all raw Dice retained")


def readout(args, split, name, mode):
    rows = []
    for case in case_names(args, split):
        saved = read(args.out / "readouts" / split / name / "pairs" / case / "readout.json")
        valid = mode == "FF" or saved["complete"]
        if mode == "REP":
            valid = bool(saved["certificate"].get("global_homeomorphism") and not saved["certificate"].get("errors"))
        if not valid or mode not in saved["rows"]:
            raise ValueError(f"Incomplete readout: {split}/{name}/{case}/{mode}")
        rows.append(saved["rows"][mode])
    return group_mean(rows, regions={} if split in ("LPBA", "OASIS") else None)


def case_names(args, split):
    if split in ("discovery", "selection", "sealed"):
        return ["__".join(pair) for pair in read(args.out / "splits.json")["pairs"][split]]
    if split == "IXI":
        return ["__".join(pair) for pair in read(args.manifest)["validation_pairs"]]
    if split == "LPBA":
        return ["__".join(pair) for pair in read(args.external_manifest)["pairs"]]
    return [
        f"p_{r['moving_subject_id'].rsplit('_P', 1)[1]}__p_{r['fixed_subject_id'].rsplit('_P', 1)[1]}"
        for r in read(args.oasis_pairs)["cases"]
    ]


def selection(args):
    inputs = verify(args)
    for arm in ("b0", "b1"):
        name = f"{arm}_s{args.seed}"
        evaluate_model(args, name, Path(inputs["weights"][name]), "selection")
    for edit in read(args.out / "edits.json")["rows"]:
        if edit["seed"] != args.seed:
            continue
        if sha256_file(Path(edit["checkpoint"])) != edit["sha256"]:
            raise ValueError("Compiled edit checkpoint changed")
        evaluate_model(
            args, f"{edit['name']}_s{args.seed}", Path(edit["checkpoint"]), "selection", controls=edit["random"] < 0
        )


def select(args):
    verify(args)
    edits = read(args.out / "edits.json")["rows"]
    names = sorted({row["name"] for row in edits})
    comparisons = {}
    mode = "REP_FLOAT_UNCERTIFIED"
    for name in names:
        comparisons[name] = {}
        for seed in (0, 1):
            a, b, c = [readout(args, "selection", f"{key}_s{seed}", mode) for key in ("b0", "b1", name)]
            comparisons[name][str(seed)] = joint_criterion(c, a, b)
            feasible = c["means"].get("float_bound_passed", 1) == 1
            comparisons[name][str(seed)].update(provisional_geometry_passed=feasible)
            comparisons[name][str(seed)]["passed"] &= feasible

    def rank(name):
        values = comparisons[name].values()
        return (
            all(value["passed"] for value in values),
            min(value["cortex_vs_b0"] for value in values),
            min(value["macro_vs_b1"] for value in values),
            name,
        )

    targets = sorted((name for name in names if name.endswith("target")), key=rank, reverse=True)[:2]
    randoms = sorted(
        {
            control
            for name in targets
            for control in names
            if control.startswith(name.removesuffix("target")) and "random" in control
        }
    )
    freeze(
        args.out / "selection.json",
        {
            "targets": targets,
            "random_controls": randoms,
            "comparisons": comparisons,
            "direction_status": "FITTED" if names else "NO_USABLE_DIRECTION",
            "selection_mode": mode,
            "edits_sha256": sha256_file(args.out / "edits.json"),
            "splits_sha256": sha256_file(args.out / "splits.json"),
            "scope": "One shared head/nominal strength across seeds, ranked by both-seed joint criterion, worst cortex gap, then worst macro gap. Top two retained even if negative; sealed readout cannot change selection. All three norm-matched random directions for each retained target are carried forward; no favourable random direction is selected.",
        },
    )


def final(args):
    inputs, chosen = verify(args), read(args.out / "selection.json")
    if chosen["edits_sha256"] != sha256_file(args.out / "edits.json") or chosen["splits_sha256"] != sha256_file(
        args.out / "splits.json"
    ):
        raise ValueError("Selected procedures or subject partition changed")
    selected = {f"{arm}_s{seed}": Path(inputs["weights"][f"{arm}_s{seed}"]) for seed in (0, 1) for arm in ("b0", "b1")}
    for edit in read(args.out / "edits.json")["rows"]:
        if edit["name"] in chosen["targets"] + chosen["random_controls"]:
            path = Path(edit["checkpoint"])
            if sha256_file(path) != edit["sha256"]:
                raise ValueError("Selected checkpoint changed")
            selected[f"{edit['name']}_s{edit['seed']}"] = path
    if args.supervised_checkpoint:
        endpoint = read(args.out / "supervision/report.json")
        if sha256_file(args.supervised_checkpoint) != endpoint["checkpoint_sha256"]:
            raise ValueError("Supervised endpoint changed")
        selected["supervised_s0"] = args.supervised_checkpoint
    failures = []
    for name, path in selected.items():
        target = name.rsplit("_s", 1)[0] in chosen["targets"]
        try:
            evaluate_model(
                args,
                name,
                path,
                args.split,
                exact=args.split in ("sealed", "IXI"),
                controls=target and args.split in ("sealed", "IXI"),
                lane=args.lane,
                lanes=args.lanes,
            )
        except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
            failures.append({"name": name, "error": repr(error)})
    atomic_write_json(
        args.out / f"final_{args.split}_{args.lane}.json",
        {
            "models": list(selected),
            "lane": args.lane,
            "lanes": args.lanes,
            "selection_sha256": sha256_file(args.out / "selection.json"),
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError("Some final candidates failed; other candidate readouts were retained")


@torch.no_grad()
def fields(args):
    verify(args)
    failures, all_rows = [], []
    families = (("stable0", "b0_s0", "b1_s0"), ("stable1", "b0_s1", "b1_s1"), ("scratch0", "scratch_b0", "scratch_b1"))
    for family, first, second in families:
        for index, (x, y, xs, ys, case) in enumerate(data(args, "IXI", lane=args.lane, lanes=args.lanes)[0]):
            path = args.out / "fields" / family / case / "readout.json"
            if path.exists() and read(path).get("complete"):
                for variant in ("half", "fixed_only", "two_sided"):
                    load_saved_scores(path.parent / variant)
                all_rows.extend(read(path)["rows"])
                continue
            x, y, xs, ys = tensors((x, y, xs, ys), args.device)
            folders = [
                args.source
                / ("stable" if family.startswith("stable") else "IXI")
                / ("IXI" if family.startswith("stable") else "")
                / name
                / "pairs"
                / case
                for name in (first, second)
            ]
            fields_pair = []
            for folder in folders:
                _, certificate = load_saved_scores(folder)
                if not certificate["global_homeomorphism"] or certificate["errors"]:
                    raise ValueError(f"Oracle baseline is not certified: {folder}")
                fields_pair.append(load_flow_npz(folder / "REP.npz").to(args.device))
            a, b = fields_pair
            fixed = torch.isin(ys, ys.new_tensor(CORTEX))
            candidates = {
                "b0": a,
                "b1": b,
                "half": (a + b) / 2,
                "fixed_only": a + region_weight(ys) * (b - a),
                "two_sided": a + protected_weight(xs, ys, a, b) * (b - a),
            }
            rows, complete = [], True
            for name, field in candidates.items():
                if name in ("b0", "b1"):
                    raw = load_flow_npz(folders[0 if name == "b0" else 1] / "FF.npz").to(args.device)
                    scored = {
                        "FF": measure(raw, case, x, y, xs, ys, metric_profile_for("IXI")),
                        "REP": measure(field, case, x, y, xs, ys, metric_profile_for("IXI")),
                    }
                else:
                    scored, certificate = certified_scores(
                        field,
                        case,
                        0,
                        xs,
                        ys,
                        path.parent / name,
                        metric_profile_for("IXI"),
                        "IXI",
                        repair_fn=boundary_repair,
                    )
                    complete &= certificate["global_homeomorphism"] and not certificate["errors"]
                    for mode in scored:
                        stored = load_flow_npz(path.parent / name / f"{mode}.npz").to(args.device)
                        scored[mode].update(
                            measure(stored, case, x, y, xs, ys, metric_profile_for("IXI"), row=scored[mode])
                        )
                for mode, row in scored.items():
                    origin = "blend_of_saved_REP"
                    if name in ("b0", "b1"):
                        origin = {"FF": "original_FF", "REP": "saved_REP"}[mode]
                    rows.append({"family": family, "variant": name, "mode": mode, "origin": origin, **row})
            for fraction in (0.25, 0.5, 0.75, 1):
                candidate = a + fraction * (candidates["two_sided"] - a)
                predictions = [warp_labels(xs, value, "canonical").long() for value in (candidate, a)]
                cortex_maps = [
                    torch.where(torch.isin(value, value.new_tensor(CORTEX)), value, 0) for value in predictions
                ]
                changed = cortex_maps[0] != cortex_maps[1]
                difference = (candidate - a).masked_select(fixed.expand_as(a))
                if changed.any() or (difference.numel() and difference.abs().max() > 1e-6):
                    raise RuntimeError("The two-sided pre-repair protection failed its actual label check")
            atomic_write_json(
                path,
                {
                    "rows": rows,
                    "complete": bool(complete),
                    "scope": "Label oracle, starting from certified REP. Baseline FF is the original raw field; oracle FF is a blend of saved REP. Primary comparisons use REP. Pre-repair moving cortex equality checked at four fractions; final anatomy is measured after repair, not assumed.",
                },
            )
            all_rows.extend(rows)
            if not complete:
                failures.append({"family": family, "case_id": case})
            print(f"[FIELD ORACLE] {family} {index + 1}/58 {case}", flush=True)
    atomic_write_json(args.out / f"fields_{args.lane}.json", {"rows": all_rows, "failures": failures})
    if failures:
        raise RuntimeError("Some oracle fields did not receive a global certificate")


def hybrids(args):
    inputs = verify(args)
    endpoint = read(args.out / "supervision/report.json")
    first = torch.load(inputs["weights"]["b1_s0"], map_location="cpu", weights_only=False)
    second = torch.load(endpoint["checkpoint"], map_location="cpu", weights_only=False)
    model = load_base(inputs["weights"]["b1_s0"], inputs["shape"], torch.device("cpu"))
    names = set(dict(model.named_parameters()))
    del model
    for bits in product((0, 1), repeat=3):
        name = "su_" + "".join(map(str, bits))
        levels = [f"level{i + 1}" for i, bit in enumerate(bits) if bit]
        state = edited_state(first["state_dict"], second["state_dict"], levels=levels, parameter_names=names)
        path = args.out / "weights" / f"{name}.pth"
        annotation = {
            "kind": "supervised_module_hybrid",
            "levels_from_S": levels,
            "S_sha256": endpoint["checkpoint_sha256"],
            "label_guidance": True,
            "scope": "S uses discovery-only anatomical supervision; original arguments describe U training",
        }
        save_weights(path, first | {"intervention": annotation}, state)
        evaluate_model(args, name, path, "discovery")


def summary(args):
    chosen = read(args.out / "selection.json")
    output, missing = {"selected": chosen, "readouts": {}, "raw_readouts": {}, "criteria": {}}, []
    if not (args.out / "readouts").is_dir():
        raise ValueError("No final readouts exist")
    expected = [
        f"{arm}_s{seed}" for arm in ("b0", "b1", *chosen["targets"], *chosen["random_controls"]) for seed in (0, 1)
    ]
    if (args.out / "supervision/report.json").is_file():
        expected.append("supervised_s0")
    else:
        missing.append({"stage": "supervision", "error": "S has no fixed-last10 endpoint"})
    for split in ("sealed", "IXI", "LPBA", "OASIS"):
        output["readouts"][split] = {}
        output["raw_readouts"][split] = {}
        mode = "REP" if split in ("sealed", "IXI") else "FF"
        for name in expected:
            try:
                output["raw_readouts"][split][name] = readout(args, split, name, "FF")
            except (OSError, KeyError, ValueError) as error:
                missing.append({"split": split, "name": name, "mode": "FF", "error": repr(error)})
            try:
                output["readouts"][split][name] = readout(args, split, name, mode)
            except (OSError, KeyError, ValueError) as error:
                missing.append({"split": split, "name": name, "error": repr(error)})
        if split in ("sealed", "IXI"):
            table = output["readouts"][split]
            output["criteria"][split] = {}
            for name in chosen["targets"]:
                outcome = {}
                for seed in (0, 1):
                    keys = [f"{key}_s{seed}" for key in (name, "b0", "b1")]
                    if all(key in table for key in keys):
                        outcome[str(seed)] = joint_criterion(*(table[key] for key in keys))
                        controls = {}
                        for control in chosen["random_controls"]:
                            if not control.startswith(name.removesuffix("target")):
                                continue
                            control_name = f"{control}_s{seed}"
                            if control_name in table:
                                controls[control] = table[control_name]
                        for label in ("amplitude", "gaussian"):
                            rows = []
                            for case in case_names(args, split):
                                saved = read(args.out / "readouts" / split / keys[0] / "pairs" / case / "readout.json")
                                if not saved["controls"].get(f"{label}_certified"):
                                    missing.append(
                                        {
                                            "split": split,
                                            "name": keys[0],
                                            "control": label,
                                            "case_id": case,
                                            "error": "Control REP certificate missing or refused",
                                        }
                                    )
                                    break
                                rows.append(saved["controls"][f"{label}_REP"])
                            else:
                                controls[label] = group_mean(rows)
                        outcome[str(seed)]["control_deltas"] = {
                            control: {
                                k: table[keys[0]]["means"][k] - value["means"][k]
                                for k in ("dice_mean", "cortex", "deep_gray")
                            }
                            for control, value in controls.items()
                        }
                        outcome[str(seed)]["controls_joint_criterion"] = {
                            control: joint_criterion(value, table[keys[1]], table[keys[2]])
                            for control, value in controls.items()
                        }
                output["criteria"][split][name] = outcome
    output["field_oracle"] = {}
    for family in ("stable0", "stable1", "scratch0"):
        rows = []
        for case in case_names(args, "IXI"):
            try:
                saved = read(args.out / "fields" / family / case / "readout.json")
                if not saved["complete"]:
                    raise ValueError("Uncertified oracle endpoint")
                for variant in ("half", "fixed_only", "two_sided"):
                    load_saved_scores(args.out / "fields" / family / case / variant)
                rows.extend(saved["rows"])
            except (OSError, ValueError) as error:
                missing.append({"family": family, "case_id": case, "error": repr(error)})
        table = {
            name: group_mean([r for r in rows if r["variant"] == name and r["mode"] == "REP"])
            for name in ("b0", "b1", "half", "fixed_only", "two_sided")
            if any(r["variant"] == name and r["mode"] == "REP" for r in rows)
        }
        criteria = {}
        if "b0" in table and "b1" in table:
            for name, value in table.items():
                outcome = joint_criterion(value, table["b0"], table["b1"])
                outcome["complete_cohort"] = value["directions"] == len(case_names(args, "IXI"))
                outcome["passed"] &= outcome["complete_cohort"]
                criteria[name] = outcome
        output["field_oracle"][family] = {"means": table, "criteria": criteria}
    output["supervision_hybrids"] = {}
    for bits in product((0, 1), repeat=3):
        name = "su_" + "".join(map(str, bits))
        try:
            output["supervision_hybrids"][name] = readout(args, "discovery", name, "FF")
        except (OSError, ValueError, KeyError) as error:
            missing.append({"hybrid": name, "error": repr(error)})
    output.update(
        missing=missing,
        supervision="S and fitted directions use anatomical labels; no new label-free claim",
        scope="Reciprocal directions averaged by unordered pair. Exact certificates only on final IXI/sealed and field oracle arrays; external readouts FF only. Passing a Dice criterion does not establish novelty.",
    )
    atomic_write_json(args.out / "report.json", output)
    if missing:
        raise RuntimeError("Some final readouts are incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "recovery",
            "collect",
            "edits",
            "selection",
            "select",
            "final",
            "fields",
            "hybrids",
            "summary",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--external_manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("results/contrasts/BC_20261118T105250Z"))
    parser.add_argument("--history_run", type=Path, default=Path("results/contrasts/BM_20261113T143420Z"))
    parser.add_argument("--oasis_root", type=Path, default=Path("/data/mooncake/P/OASIS_L2R_2021_task03/All"))
    parser.add_argument(
        "--oasis_pairs", type=Path, default=Path("datasets/splits/oasis_subject_disjoint_v2/pair_manifest.json")
    )
    parser.add_argument("--supervised_checkpoint", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--split", choices=("sealed", "IXI", "LPBA", "OASIS"), default="sealed")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.lane < args.lanes:
        parser.error("Require 0 <= lane < lanes")
    cpu_stage = args.stage in ("prepare", "edits", "select", "summary")
    args.device = torch.device(
        "cuda" if not cpu_stage and torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    )
    configure_precision("bf16" if args.device.type == "cuda" else "fp32", args.device)
    if args.stage == "recovery":
        verify(args)
        for case in ("subject_557__subject_85", "subject_92__subject_478"):
            load_saved_scores(args.source / "IXI/base/pairs" / case)
        args.base_checkpoint, args.out = None, args.out / "recovery"
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
