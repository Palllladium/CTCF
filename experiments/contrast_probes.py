"""Fixed Train probes of contrast dependence, cascade weight changes and geometry-matched smoothing."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from datasets.IXI import IXIPairedDataset, read_paired_manifest
from experiments.contrast_diagnostics import read_csv
from experiments.contrasts import RECIPE, build_runner, tensors
from experiments.core.field_scoring import paired_dice_delta, score
from experiments.core.inference_metrics import metric_profile_for, write_results
from experiments.core.precision import configure_precision
from experiments.dispatch1.eval_consensus import smooth
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import Grad3d, SegNorm, pkload, setup_device
from utils.field import legacy_sampling_displacement
from utils.field_io import load_flow_npz


def train_pairs(manifest, count=30):
    rows = manifest["training"]
    if 2 * count > len(rows):
        raise ValueError("Train probes require distinct subjects in every pair")
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261112)).tolist()[: 2 * count]
    return [(order[i], order[i + 1]) for i in range(0, len(order), 2)]


def source_weights(source, arm):
    report = json.loads((source / f"{arm}_w1_s0/report.json").read_text())
    path = Path("results") / report["exp"] / "ckpt/last.pth"
    if sha256_file(path) != report["checkpoint_sha256"]:
        raise ValueError(f"Source weights changed: {path}")
    saved = torch.load(path, map_location="cpu")
    expected = RECIPE | {"ixi_arm": arm, "w_reg": 1, "seed": 0, "max_epoch": 3, "precision": "bf16"}
    if any(saved["arguments"].get(key) != value for key, value in expected.items()) or saved["epoch"] != 2:
        raise ValueError(f"Not the fixed three-epoch w1 endpoint: {path}")
    return path, saved["state_dict"]


def prepare(args):
    manifest = read_paired_manifest(args.manifest, verify_bytes=True)
    source_inputs = json.loads((args.source / "inputs.json").read_text())
    if (
        source_inputs["parent_sha256"] != sha256_file(args.checkpoint)
        or source_inputs["manifest_sha256"] != manifest["fingerprint"]
    ):
        raise ValueError("The short controls use another parent or manifest")
    sources = {arm: str(source_weights(args.source, arm)[0]) for arm in ("b0", "b1", "b2")}
    fields = {}
    for name, directory in (
        ("b0", args.source / "b0_w1_s0/epoch3/pairs"),
        ("b1", args.source / "b1_w1_s0/epoch3/pairs"),
        ("parent", args.original / "parent/pairs"),
    ):
        values = sorted(directory.glob("*/FF.npz"))
        if len(values) != 58:
            raise ValueError(f"{name}: all 58 saved FF directions are required")
        fields[name] = {}
        for path in values:
            case = path.parent.name
            report_path = (
                args.source / f"{name}_w1_s0/certified" / case if name != "parent" else path.parent
            ) / "report.json"
            report = json.loads(report_path.read_text())
            expected = report["source_sha256"] if name != "parent" else report["field_sha256"]["FF"]
            fields[name][case] = sha256_file(path)
            if fields[name][case] != expected:
                raise ValueError(f"Saved FF bytes changed: {path}")
        if set(fields[name]) != {"__".join(pair) for pair in manifest["validation_pairs"]}:
            raise ValueError(f"{name}: saved fields differ from the validation manifest")
    for arm in ("b0", "b1", "b2"):
        report = json.loads((args.source / f"{arm}_w1_s0/report.json").read_text())
        if report["manifest_sha256"] != manifest["fingerprint"]:
            raise ValueError("Source readout uses a different IXI manifest")
    if json.loads((args.original / "parent/recipe.json").read_text())["checkpoint_sha256"] != sha256_file(
        args.checkpoint
    ):
        raise ValueError("Parent differs from the original experiment")
    binding = {
        "source": str(args.source.resolve()),
        "original": str(args.original.resolve()),
        "manifest_sha256": manifest["fingerprint"],
        "parent_sha256": sha256_file(args.checkpoint),
        "weights": {arm: {"path": path, "sha256": sha256_file(Path(path))} for arm, path in sources.items()},
        "fields": fields,
        "train_pairs": [
            [manifest["training"][a]["id"], manifest["training"][b]["id"]] for a, b in train_pairs(manifest)
        ],
        "interpolation": [0, 0.5, 1, 1.5, 2],
        "smoothing_sigma_range": [0, 4],
        "scope": "Fixed Train probes; no hybrid selection on validation, no hybrid certificates",
    }
    path = args.out / "probe_inputs.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Probe inputs changed; use a fresh RUN_ID")
    atomic_write_json(path, binding)


def edited_state(first, second, *, levels=(), fraction=None, parameter_names):
    if first.keys() != second.keys():
        raise ValueError("Weight endpoints have different architectures")
    output = {}
    for key, value in first.items():
        other = second[key]
        if value.shape != other.shape:
            raise ValueError(f"Weight endpoints have different shapes: {key}")
        if key not in parameter_names:
            if not torch.equal(value, other):
                raise ValueError(f"Non-parameter buffer differs: {key}")
            output[key] = value
        elif fraction is not None:
            output[key] = value + fraction * (other - value)
        else:
            output[key] = other if key.split(".", 1)[0] in levels else value
    return output


def weight_spectra(first, second, parameter_names):
    rows = []
    for key in sorted(parameter_names):
        base, difference = first[key].double(), second[key].double() - first[key].double()
        row = {"parameter": key, "relative_l2": float(difference.norm() / base.norm().clamp_min(1e-30))}
        if difference.ndim >= 2:
            singular = torch.linalg.svdvals(difference.flatten(1))
            energy = singular.square()
            total = energy.sum()
            row.update(
                top_singular_values=singular[:8].tolist(),
                rank90=int(torch.searchsorted(energy.cumsum(0), 0.9 * total)) + 1 if total > 0 else 0,
                top_energy_fraction=float(energy[0] / total) if total > 0 else 0,
            )
        rows.append(row)
    return rows


def probe_batch(dataset, pair, device):
    x, y, tx, ty = tensors(dataset[pair], device)
    labels = []
    for index in pair:
        image, segmentation = pkload(dataset.rows[index]["t1"])
        segmentation = SegNorm()([image, segmentation])[1]
        labels.append(torch.from_numpy(np.ascontiguousarray(segmentation))[None, None].to(device))
    name = "__".join(dataset.rows[index]["id"] for index in pair)
    return x, y, tx, ty, labels, name


@torch.no_grad()
def measure(runner, batch, inputs=None, reference=None):
    x, y, tx, ty, labels, name = batch
    a, b = (x, y) if inputs is None else inputs
    forward, backward = [runner._forward_flow(p, q) for p, q in ((a, b), (b, a))]
    warped = [runner.ctx.reg_model((p, f)) for p, f in ((x[:, :1], forward), (y[:, :1], backward))]
    d1, d2 = runner.contrast_terms(x, y, *warped, forward, backward, (tx, ty))
    canonical = legacy_sampling_displacement(forward)
    row = score(canonical, name, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI"))
    row.update(d1=float(d1), d2=float(d2), regularity=float(Grad3d(penalty="l2")(canonical)))
    if reference is not None:
        change = canonical - reference
        row.update(field_change_rms=float(change.square().mean().sqrt()), field_change_max=float(change.abs().max()))
    return row, canonical


def weights(args):
    manifest = read_paired_manifest(args.manifest)
    runner = build_runner(args, "b1")
    runner.model.eval()
    first, second = [source_weights(args.source, arm)[1] for arm in ("b0", "b1")]
    names = set(dict(runner.model.named_parameters()))
    atomic_write_json(args.out / "weight_spectra.json", weight_spectra(first, second, names))
    states = [
        ("levels_" + "".join(map(str, bits)), {"levels": tuple(f"level{i + 1}" for i, bit in enumerate(bits) if bit)})
        for bits in itertools.product((0, 1), repeat=3)
    ]
    states += [(f"mix_{fraction:g}", {"fraction": fraction}) for fraction in (0, 0.5, 1, 1.5, 2)]
    dataset = IXIPairedDataset(manifest, "b1")
    outputs = {name: [] for name, _ in states}
    for pair in train_pairs(manifest):
        batch = probe_batch(dataset, pair, args.device)
        reference = None
        for name, edit in states:
            runner.model.load_state_dict(edited_state(first, second, parameter_names=names, **edit), strict=True)
            row, field = measure(runner, batch, reference=reference)
            if reference is None:
                reference = field
            outputs[name].append(row)
        print(f"[WEIGHT PROBE] {batch[-1]}", flush=True)
    for name, rows in outputs.items():
        directory = args.out / "weights" / name
        directory.mkdir(parents=True, exist_ok=True)
        write_results(rows, str(directory), name, "B0/B1 w1 seed0", "Train only; uncertified diagnostic fields")


def t2_inputs(args):
    manifest = read_paired_manifest(args.manifest)
    runner = build_runner(args, "b2")
    runner.model.load_state_dict(source_weights(args.source, "b2")[1], strict=True)
    runner.model.eval()
    dataset = IXIPairedDataset(manifest, "b2")
    pairs = train_pairs(manifest)
    outputs = {name: [] for name in ("true", "t1_copy", "zero", "foreign")}
    for index, pair in enumerate(pairs):
        batch = probe_batch(dataset, pair, args.device)
        x, y = batch[:2]
        foreign_pair = pairs[(index + 1) % len(pairs)]
        foreign_x, foreign_y, _, _ = tensors(dataset[foreign_pair], args.device)
        reference = None
        for name, a2, b2 in (
            ("true", x[:, 1:], y[:, 1:]),
            ("t1_copy", x[:, :1], y[:, :1]),
            ("zero", torch.zeros_like(x[:, :1]), torch.zeros_like(y[:, :1])),
            ("foreign", foreign_x[:, 1:], foreign_y[:, 1:]),
        ):
            inputs = torch.cat((x[:, :1], a2), 1), torch.cat((y[:, :1], b2), 1)
            row, field = measure(runner, batch, inputs, reference)
            if reference is None:
                reference = field
            outputs[name].append(row)
        print(f"[T2 PROBE] {batch[-1]}", flush=True)
    for name, rows in outputs.items():
        directory = args.out / "t2_inputs" / name
        directory.mkdir(parents=True, exist_ok=True)
        write_results(rows, str(directory), name, "B2 w1 seed0", "Train input intervention; D2 always uses true T2")


def match_regularity(flow, target):
    regularity = Grad3d(penalty="l2")
    grid = (0, 0.125, 0.25, 0.5, 1, 2, 4)
    samples, best = {}, None

    def evaluate(sigma):
        nonlocal best
        candidate = smooth(flow, sigma)
        value = float(regularity(candidate))
        samples[sigma] = value
        if best is None or abs(value - target) < best[0]:
            best = abs(value - target), sigma, value, candidate

    for sigma in grid:
        evaluate(sigma)
    brackets = [(a, b) for a, b in itertools.pairwise(grid) if (samples[a] - target) * (samples[b] - target) <= 0]
    if brackets:
        left, right = min(brackets, key=lambda pair: abs(samples[pair[0]] - target) + abs(samples[pair[1]] - target))
        for _ in range(12):
            middle = (left + right) / 2
            evaluate(middle)
            if (samples[left] - target) * (samples[middle] - target) <= 0:
                right = middle
            else:
                left = middle
    gap, sigma, value, candidate = best
    relative = gap / max(abs(target), 1e-30)
    return candidate, {
        "sigma": sigma,
        "regularity": value,
        "target_regularity": target,
        "relative_R_gap": relative,
        "matched": relative <= 0.005,
        "search_status": "BRACKET_REFINED" if brackets else "NO_BRACKET_IN_0_4",
        "R_monotone_on_grid": all(samples[a] >= samples[b] for a, b in itertools.pairwise(grid)),
        "evaluations": [{"sigma": s, "R": r} for s, r in sorted(samples.items())],
    }


def smoothing(args):
    manifest = read_paired_manifest(args.manifest, require_t2=False)
    subjects = {row["id"]: row for row in manifest["validation"]}
    inputs = json.loads((args.out / "probe_inputs.json").read_text())
    outputs, traces = {name: [] for name in ("b0", "parent")}, []
    for target_path in sorted((args.source / "b1_w1_s0/epoch3/pairs").glob("*/FF.npz")):
        case = target_path.parent.name
        target_flow = load_flow_npz(target_path)
        if sha256_file(target_path) != inputs["fields"]["b1"][case]:
            raise ValueError(f"Target field changed: {case}")
        target = float(Grad3d(penalty="l2")(target_flow))
        labels = []
        for subject in case.split("__"):
            image, segmentation = pkload(subjects[subject]["t1"])
            labels.append(
                torch.as_tensor(np.ascontiguousarray(SegNorm()([image, segmentation])[1]), device=args.device)[
                    None, None
                ]
            )
        for name in outputs:
            path = (
                (args.source / "b0_w1_s0/epoch3/pairs" if name == "b0" else args.original / "parent/pairs")
                / case
                / "FF.npz"
            )
            if sha256_file(path) != inputs["fields"][name][case]:
                raise ValueError(f"Source field changed: {path}")
            candidate, detail = match_regularity(load_flow_npz(path), target)
            row = score(candidate.to(args.device), case, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI"))
            row.update({key: value for key, value in detail.items() if key not in ("evaluations", "search_status")})
            outputs[name].append(row)
            traces.append({"source": name, "case": case, **detail})
        print(f"[R MATCH] {case}", flush=True)
    for name, rows in outputs.items():
        if len(rows) != 58:
            raise ValueError("Geometry matching requires every validation direction")
        directory = args.out / "smoothing" / name
        directory.mkdir(parents=True, exist_ok=True)
        write_results(rows, str(directory), name, "saved w1 fields", "Geometry-only bounded search; no Dice selection")
    atomic_write_json(args.out / "smoothing/search.json", traces)


def summary(args):
    tables = {
        name: read_csv(args.out / name / "epoch3/per_case.csv") for name in ("half_t1", "full_t1_t2", "b0_s1", "b1_s1")
    }
    tables.update({name: read_csv(args.source / f"{name}_w1_s0/epoch3/per_case.csv") for name in ("b0", "b1")})
    if any(len(rows) != 58 for rows in tables.values()):
        raise ValueError("Factorial comparison requires all 58 directions in every arm")
    result = {}
    comparisons = (
        ("T2_at_half_T1", "b1", "half_t1"),
        ("T2_at_full_T1", "full_t1_t2", "b0"),
        ("halve_T1_without_T2", "half_t1", "b0"),
        ("seed0_B1_B0", "b1", "b0"),
        ("seed1_B1_B0", "b1_s1", "b0_s1"),
    )
    for name, a, b in comparisons:
        result[name] = paired_dice_delta(tables[a], tables[b])
        left, right = [{row["case_id"]: row for row in tables[arm]} for arm in (a, b)]
        keys = [
            key
            for key in tables[a][0]
            if key.startswith("dice_lbl_") or key in ("regularity", "ndv_percent", "sdlogj_plus3")
        ]
        result[name]["metric_deltas"] = {
            key: float(np.mean([float(left[case][key]) - float(right[case][key]) for case in left])) for key in keys
        }
    result["seed1_epoch10_B1_B0"] = paired_dice_delta(
        read_csv(args.out / "b1_s1/epoch10/per_case.csv"), read_csv(args.out / "b0_s1/epoch10/per_case.csv")
    )
    result["scope"] = "Exploratory fixed readouts; seed1 epoch10 has no matching seed0 epoch10"
    by_case = {arm: {row["case_id"]: row for row in rows} for arm, rows in tables.items()}
    added = {
        dose: [
            {
                "case_id": case,
                "dice_mean": float(by_case[yes][case]["dice_mean"]) - float(by_case[no][case]["dice_mean"]),
            }
            for case in by_case["b0"]
        ]
        for dose, yes, no in (("full", "full_t1_t2", "b0"), ("half", "b1", "half_t1"))
    }
    result["T1_T2_interaction"] = paired_dice_delta(added["full"], added["half"])
    result["certificates"] = {
        str(path.relative_to(args.out)): json.loads(path.read_text())
        for path in args.out.glob("*/repair_summary*.json")
    }
    repaired = {}
    for name in ("half_t1", "full_t1_t2", "b0", "b1"):
        directory = args.source / f"{name}_w1_s0" if name in ("b0", "b1") else args.out / name
        path = directory / "repair_summary.json"
        if path.exists():
            repaired[name] = json.loads(path.read_text())
    result["factor_REP"] = {"missing": sorted(set(("half_t1", "full_t1_t2", "b0", "b1")) - repaired.keys())}
    if not result["factor_REP"]["missing"]:
        result["factor_REP"].update(
            {name: paired_dice_delta(repaired[a]["rows"], repaired[b]["rows"]) for name, a, b in comparisons[:3]}
        )
        result["factor_REP"]["certified"] = all(report["certified"] for report in repaired.values())
        result["factor_REP"]["fallbacks"] = {name: report["fallbacks"] for name, report in repaired.items()}
    for epoch in (3, 10):
        suffix = "" if epoch == 3 else f"_epoch{epoch}"
        repairs = {}
        for arm in ("b0", "b1"):
            path = args.out / f"{arm}_s1/repair_summary{suffix}.json"
            if path.exists():
                repairs[arm] = json.loads(path.read_text())
        if len(repairs) == 2:
            result[f"seed1_epoch{epoch}_REP_B1_B0"] = {
                **paired_dice_delta(repairs["b1"]["rows"], repairs["b0"]["rows"]),
                "both_certified": all(repairs[arm]["certified"] for arm in repairs),
                "fallbacks": {arm: repairs[arm]["fallbacks"] for arm in repairs},
            }
    for mode in ("b0", "parent"):
        path = args.out / f"smoothing/{mode}/per_case.csv"
        if path.exists():
            smoothed = read_csv(path)
            result[f"B1_minus_smoothed_{mode}"] = {
                **paired_dice_delta(tables["b1"], smoothed),
                "matched_directions": sum(row["matched"] == "True" for row in smoothed),
                "max_relative_R_gap": max(float(row["relative_R_gap"]) for row in smoothed),
                "scope": "All directions; unmatched targets do not support a geometry-matched interpretation",
            }
    atomic_write_json(args.out / "mechanism_summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "weights", "t2", "smooth", "summary"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.precision, args.device = "bf16", setup_device()
    configure_precision(args.precision, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "weights": weights, "t2": t2_inputs, "smooth": smoothing, "summary": summary}[args.stage](args)


if __name__ == "__main__":
    main()
