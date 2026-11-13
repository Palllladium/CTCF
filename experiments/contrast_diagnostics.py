"""Short, matched regularisation controls using the unchanged CTCF training runtime."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from datasets.IXI import IXIPairedDataset, read_paired_manifest
from experiments.contrasts import RECIPE, sync, tensors, training_args
from experiments.core.data_loaders import ixi_paired_loaders
from experiments.core.field_scoring import paired_dice_delta, score
from experiments.core.inference_metrics import metric_profile_for, write_results
from experiments.core.precision import configure_precision
from experiments.core.train_runtime import run_train
from experiments.dispatch1.eval_consensus import smooth
from experiments.train_CTCF import Runner
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import Grad3d, setup_device
from utils.field import legacy_sampling_displacement
from utils.field_io import load_flow_npz, save_flow_npz_atomic

BASE_HEAD = "c18e497d3059e7163249814d1861f1ef6235efea"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def frozen_training_diff():
    roots = ["experiments/train_CTCF.py", "experiments/contrasts.py", "experiments/core", "models", "datasets", "utils"]
    baseline = set(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", BASE_HEAD, "--", *roots], text=True
        ).splitlines()
    )
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE_HEAD, "--", *roots], text=True).splitlines()
    return sorted(baseline.intersection(changed))


def anchor_comparison(previous, current, tolerance=0.001):
    old = {int(row["epoch"]): float(row["val_dice"]) for row in previous}
    new = {int(row["epoch"]): float(row["val_dice"]) for row in current}
    delta = {str(epoch + 1): new[epoch] - old[epoch] for epoch in (0, 2)}
    return {
        "epoch_differences": delta,
        "tolerance": tolerance,
        "passed": all(abs(value) <= tolerance for value in delta.values()),
        "scope": "Operational same-epoch anchor, not a statistical estimate of repeatability",
    }


def prepare(args):
    from experiments.repair_diagnostics import inventory

    if (args.source / "head.txt").read_text().strip() != BASE_HEAD:
        raise ValueError("Follow-up requires the reviewed source revision")
    source = json.loads((args.source / "parent/recipe.json").read_text())
    manifest = read_paired_manifest(args.manifest, verify_bytes=True)
    if (
        source["checkpoint_sha256"] != sha256_file(args.checkpoint)
        or source["manifest_sha256"] != manifest["fingerprint"]
    ):
        raise ValueError("Source parent or paired manifest differs from the completed experiment")
    cases = inventory(args.source)
    missing = []
    for case in cases:
        needs_raw = not case["old_pass"] or case["control"] or case["directory"].split("/")[0] in ("b0", "b1", "b2")
        for mode in ("FF", "REP") if needs_raw else ("REP",):
            path = args.source / case["directory"] / f"{mode}.npz"
            if not path.is_file():
                missing.append(str(path))
    for arm in ("b0", "b1", "b2"):
        path = Path("logs") / f"{args.source.name}_{arm}" / "metrics.csv"
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Source heavy fields are required; missing {len(missing)}: {missing[:5]}")
    inputs = {
        "source": str(args.source.resolve()),
        "parent_sha256": source["checkpoint_sha256"],
        "manifest_sha256": manifest["fingerprint"],
        "base_head": BASE_HEAD,
        "changed_training_files": frozen_training_diff(),
        "recipe": RECIPE,
        "short_epochs": 3,
        "anchor_tolerance": 0.001,
        "cases": cases,
    }
    if (args.out / "inputs.json").exists() and json.loads((args.out / "inputs.json").read_text()) != inputs:
        raise ValueError("Continuation belongs to different inputs or training code")
    atomic_write_json(args.out / "inputs.json", inputs)


@torch.no_grad()
def observe(runner, manifest, directory, *, save_fields=False):
    directory.mkdir(parents=True, exist_ok=True)
    previous_mode = runner.model.training
    device = runner.device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    rows = []
    try:
        with torch.random.fork_rng(devices=devices):
            runner.model.eval()
            dataset = IXIPairedDataset(manifest, runner.args.ixi_arm, validation=True)
            for index in range(len(dataset)):
                x, y, xs, ys = tensors(dataset[index], device)
                moving, fixed = dataset.pairs[index]
                name = f"{dataset.rows[moving]['id']}__{dataset.rows[fixed]['id']}"
                started = sync(device)
                flow = legacy_sampling_displacement(runner._forward_flow(x, y))
                seconds = sync(device) - started
                rows.append(score(flow, name, seconds, xs, ys, 0.0, "canonical", profile=metric_profile_for("IXI")))
                rows[-1]["regularity"] = float(Grad3d(penalty="l2")(flow))
                if save_fields:
                    save_flow_npz_atomic(directory / "pairs" / name / "FF.npz", flow)
            write_results(
                rows,
                str(directory),
                runner.args.ixi_arm,
                f"live:{runner.args.exp}/{directory.name}",
                "IXI exploratory short control",
            )
    finally:
        runner.model.train(previous_mode)
    return rows


class DiagnosticRunner(Runner):
    def __init__(self, args, device, manifest, out, updates=0):
        super().__init__(args, device)
        self.observation_manifest, self.observation_out = manifest, out
        self.completed_updates = updates
        self.optimizer.register_step_post_hook(self.observe_step)

    def observe_step(self, _optimizer, _args, _kwargs):
        self.completed_updates += 1
        if self.args.ixi_arm == "b0" and self.completed_updates in (20, 100):
            observe(self, self.observation_manifest, self.observation_out / f"step{self.completed_updates}")

    def on_val_end(self, epoch, _dice, _jac):
        if epoch + 1 in (1, 3, self.args.max_epoch):
            observe(self, self.observation_manifest, self.observation_out / f"epoch{epoch + 1}", save_fields=epoch >= 2)


def short_train(args):
    manifest = read_paired_manifest(args.manifest, require_t2=args.arm != "b0")
    options = training_args(args, args.arm)
    epochs = getattr(args, "epochs", 3)
    options.max_epoch, options.w_reg, options.seed, options.exp = epochs, args.w_reg, args.seed, args.exp
    options.ixi_t1_weight, options.ixi_t2_weight = getattr(args, "t1_weight", None), getattr(args, "t2_weight", None)
    checkpoint = Path("results") / args.exp / "ckpt/last.pth"
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu")
        expected = {
            "w_reg": args.w_reg,
            "ixi_arm": args.arm,
            "seed": args.seed,
            "max_epoch": epochs,
            "ixi_t1_weight": options.ixi_t1_weight,
            "ixi_t2_weight": options.ixi_t2_weight,
        }
        if any(saved["arguments"].get(key) != value for key, value in expected.items()):
            raise ValueError("Short control checkpoint uses another recipe")
        start = int(saved["epoch"]) + 1
        options.init_ckpt, options.resume = "", str(checkpoint)
    device = setup_device(seed=args.seed, deterministic=False)
    configure_precision("bf16", device)
    runner = DiagnosticRunner(options, device, manifest, args.out, updates=start * 399)
    if not start:
        observe(runner, manifest, args.out / "epoch0")
    run_train(options, runner, build_loaders=ixi_paired_loaders)
    if not all((args.out / f"epoch{epoch}/per_case.csv").exists() for epoch in {3, epochs}):
        raise ValueError("Short control lacks its final observation")
    atomic_write_json(
        args.out / "report.json",
        {
            "arm": args.arm,
            "w_reg": args.w_reg,
            "seed": args.seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "updates": epochs * 399,
            "epochs": epochs,
            "t1_weight": options.ixi_t1_weight,
            "t2_weight": options.ixi_t2_weight,
            "exp": args.exp,
            "manifest_sha256": manifest["fingerprint"],
        },
    )


def anchor(args):
    source_rows = read_csv(Path("logs") / args.exp / "metrics.csv")
    historic = Path("logs") / f"{args.source.name}_b0" / "metrics.csv"
    if not historic.exists():
        raise FileNotFoundError(f"Historical training curve missing: {historic}")
    historic_rows = read_csv(historic)
    result = anchor_comparison(historic_rows, source_rows)
    result["changed_training_files"] = frozen_training_diff()
    result["reuse_old_curves"] = result["passed"] and not result["changed_training_files"]
    result["add_b1_w4"] = not result["reuse_old_curves"]
    result["first_epoch_losses"] = {}
    for name, rows in (("historic", historic_rows), ("fresh", source_rows)):
        epoch0 = next(row for row in rows if int(row["epoch"]) == 0)
        meters = json.loads(epoch0.get("loss_meters_json") or "{}")
        result["first_epoch_losses"][name] = {
            key: meters.get(key, {}).get("avg") for key in ("reg", "ncc", "icon", "jac")
        }
    result["epoch0_vs_historic_parent"] = {}
    reference = args.source / "parent/FF/per_case.csv"
    if reference.exists():
        parent = {row["case_id"]: float(row["dice_mean"]) for row in read_csv(reference)}
        for name in ("b0_w1_s0", "b0_w4_s0"):
            path = args.out / name / "epoch0/per_case.csv"
            if not path.exists():
                result["epoch0_vs_historic_parent"][name] = {"status": "MISSING_READOUT"}
                continue
            current = {row["case_id"]: float(row["dice_mean"]) for row in read_csv(path)}
            delta = {case: current[case] - parent[case] for case in sorted(current.keys() & parent.keys())}
            result["epoch0_vs_historic_parent"][name] = {
                "missing_directions": sorted(parent.keys() - current.keys()),
                "extra_directions": sorted(current.keys() - parent.keys()),
                "dice_delta_by_direction": delta,
                "max_absolute_dice_delta": max(map(abs, delta.values()), default=None),
            }
    else:
        result["epoch0_vs_historic_parent"]["status"] = "MISSING_REFERENCE"
    result["diagnostic_scope"] = "Losses and initial readout agreement are descriptive, without extra acceptance gates"
    atomic_write_json(args.out / "anchor.json", result)


def profile(args):
    options = training_args(args, "b0")
    options.max_epoch, options.max_train_iters, options.max_val_batches = 1, 60, 2
    options.num_workers, options.exp, options.save_ckpt = args.workers, args.exp, 0
    device = setup_device()
    configure_precision("bf16", device)
    torch.backends.cudnn.benchmark = bool(args.benchmark)
    runner = Runner(options, device)
    timing, counter, started = {}, 0, None

    def measured_step(_optimizer, _args, _kwargs):
        nonlocal counter, started
        counter += 1
        if counter in (10, 60):
            torch.cuda.synchronize(device)
            if counter == 10:
                started = time.perf_counter()
            else:
                timing["wall_seconds"] = time.perf_counter() - started

    runner.optimizer.register_step_post_hook(measured_step)
    run_train(options, runner, build_loaders=ixi_paired_loaders)
    atomic_write_json(
        args.out / "report.json",
        {
            "workers": args.workers,
            "cudnn_benchmark": bool(args.benchmark),
            "measured_updates": 50,
            "update_seconds": timing["wall_seconds"] / 50,
            "scope": "Actual setup_device, optimiser and training loop; ten warmup updates excluded",
        },
    )


def smoothing_controls(args):
    manifest = read_paired_manifest(args.manifest, require_t2=False)
    subjects = {row["id"]: row for row in manifest["validation"]}
    from utils import SegNorm, pkload

    regularity = Grad3d(penalty="l2")
    outputs = {arm: [] for arm in ("b1", "b2")}
    for path in sorted((args.source / "b0/pairs").glob("*/FF.npz")):
        case_id = path.parent.name
        for arm in ("b0", "b1", "b2"):
            directory = args.source / arm / "pairs" / case_id
            expected = json.loads((directory / "report.json").read_text())["field_sha256"]["FF"]
            if sha256_file(directory / "FF.npz") != expected:
                raise ValueError(f"Source smoothing-control field changed: {directory}")
        flow = load_flow_npz(path).to(args.device)
        targets = {
            arm: float(regularity(load_flow_npz(args.source / arm / "pairs" / case_id / "FF.npz").to(args.device)))
            for arm in outputs
        }
        moving, fixed = case_id.split("__")
        labels = []
        for subject in (moving, fixed):
            image, seg = pkload(subjects[subject]["t1"])
            seg = SegNorm()([image, seg])[1]
            labels.append(torch.as_tensor(np.ascontiguousarray(seg), device=args.device)[None, None])
        choices = {arm: None for arm in outputs}
        for sigma in (0, 0.25, 0.5, 1, 2):
            candidate = smooth(flow, sigma)
            value = float(regularity(candidate))
            for arm in outputs:
                distance = abs(value - targets[arm])
                if choices[arm] is None or distance < choices[arm][0]:
                    choices[arm] = (distance, sigma, value, candidate.clone())
        for arm, (distance, sigma, value, candidate) in choices.items():
            row = score(candidate, case_id, 0, *labels, 0, "canonical", profile=metric_profile_for("IXI"))
            row.pop("time_sec")
            row.update(
                {"sigma": sigma, "regularity": value, "target_regularity": targets[arm], "regularity_gap": distance}
            )
            outputs[arm].append(row)
    for arm, rows in outputs.items():
        if len(rows) != 58:
            raise ValueError("Smoothing control needs all 58 directions")
        (args.out / arm).mkdir(parents=True, exist_ok=True)
        write_results(rows, str(args.out / arm), "B0 smoothing matched to " + arm, "saved fields", "No Dice selection")


def decide(args):
    rows = {arm: read_csv(args.out / f"{arm}_w1_s0/epoch3/per_case.csv") for arm in ("b0", "b1", "b2")}
    if any(len(value) != 58 for value in rows.values()):
        raise ValueError("Conditional replication requires all 58 directions in every short control")
    comparisons = {arm: paired_dice_delta(rows[arm], rows["b0"]) for arm in ("b1", "b2")}
    repaired = {}
    for arm in rows:
        path = args.out / f"{arm}_w1_s0/repair_summary.json"
        if path.exists():
            repaired[arm] = json.loads(path.read_text())
    rep_comparisons = {
        arm: {
            **paired_dice_delta(repaired[arm]["rows"], repaired["b0"]["rows"]),
            "both_arms_fallback_free": repaired[arm]["fallbacks"] == repaired["b0"]["fallbacks"] == 0,
            "both_arms_certified": repaired[arm]["certified"] and repaired["b0"]["certified"],
        }
        for arm in ("b1", "b2")
        if arm in repaired and "b0" in repaired
    }
    chosen = next((arm for arm in ("b1", "b2") if comparisons[arm]["resource_signal"]), None)
    atomic_write_json(
        args.out / "decision.json",
        {
            "FF_comparisons": comparisons,
            "REP_comparisons": rep_comparisons,
            "REP_scope": "Endpoint-algorithm contrast; model-quality attribution requires the fallback/certificate flags",
            "repeat_arm": chosen,
            "repeat_trigger": "FF >=0.005; certified REP and fallback price are reported separately",
            "interpretation": "Three-epoch exploratory signal only; not a convergence or universal negative claim",
        },
    )


def summary(args):
    result = {
        "anchor": json.loads((args.out / "anchor.json").read_text()),
        "decision": json.loads((args.out / "decision.json").read_text()),
        "arms": {},
    }
    for path in sorted(args.out.glob("b*_w*_s*/report.json")):
        arm = path.parent.name
        result["arms"][arm] = json.loads(path.read_text())
        result["arms"][arm]["FF"] = json.loads((path.parent / "epoch3/summary.json").read_text())
        repaired = path.parent / "repair_summary.json"
        if repaired.exists():
            result["arms"][arm]["REP"] = json.loads(repaired.read_text())
    for epoch in (1, 3):
        lo = read_csv(args.out / f"b0_w1_s0/epoch{epoch}/per_case.csv")
        hi = read_csv(args.out / f"b0_w4_s0/epoch{epoch}/per_case.csv")
        lo, hi = [sorted(rows, key=lambda row: row["case_id"]) for rows in (lo, hi)]
        result[f"regularisation_epoch{epoch}"] = paired_dice_delta(lo, hi)
        result[f"regularisation_epoch{epoch}"]["per_label"] = {
            key: float(np.mean([float(a[key]) - float(b[key]) for a, b in zip(lo, hi, strict=True)]))
            for key in lo[0]
            if key.startswith("dice_lbl_")
        }
        result[f"regularisation_epoch{epoch}"]["regularity_change"] = float(
            np.mean([float(a["regularity"]) - float(b["regularity"]) for a, b in zip(lo, hi, strict=True)])
        )
    result["historic_w4"] = {}
    repeated = result["decision"]["repeat_arm"]
    if repeated and (args.out / f"{repeated}_w1_s1/report.json").exists():
        result["replication_FF"] = paired_dice_delta(
            read_csv(args.out / f"{repeated}_w1_s1/epoch3/per_case.csv"),
            read_csv(args.out / "b0_w1_s1/epoch3/per_case.csv"),
        )
    if result["anchor"]["reuse_old_curves"]:
        for arm in ("b0", "b1", "b2"):
            rows = read_csv(Path("logs") / f"{args.source.name}_{arm}" / "metrics.csv")
            by_epoch = {int(row["epoch"]): float(row["val_dice"]) for row in rows}
            result["historic_w4"][arm] = {str(epoch + 1): by_epoch[epoch] for epoch in (0, 2)}
    result["historical_readout_scope"] = (
        "Only native scalar curves reused, not missing early per-label or certified REP scores"
    )
    atomic_write_json(args.out / "short_summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "anchor", "smooth", "decide", "summary", "profile"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=("b0", "b1", "b2"), default="b0")
    parser.add_argument("--w_reg", type=float, choices=(1, 4), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, choices=(3, 10), default=3)
    parser.add_argument("--t1_weight", type=float)
    parser.add_argument("--t2_weight", type=float)
    parser.add_argument("--exp")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.precision, args.device = "bf16", torch.device("cuda:0")
    args.out.mkdir(parents=True, exist_ok=True)
    actions = {
        "prepare": prepare,
        "train": short_train,
        "anchor": anchor,
        "smooth": smoothing_controls,
        "decide": decide,
        "summary": summary,
        "profile": profile,
    }
    actions[args.stage](args)


if __name__ == "__main__":
    main()
