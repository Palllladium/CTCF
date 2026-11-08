"""Readiness, final-field evaluation and paired summaries for the IXI contrast pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch

from datasets.IXI import PAIRED_TRAIN_EXCLUSIONS, IXIPairedDataset, IXIPairSampler, read_paired_manifest
from experiments.core.field_scoring import paired_dice_delta, score_saved_field
from experiments.core.helpers import build_val_dice_fn
from experiments.core.inference_metrics import metric_profile_for, write_results
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from experiments.core.precision import configure_precision, precision_autocast
from experiments.train_CTCF import Runner, main as train_main, parse_args
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import RegisterModel, validate
from utils.field import legacy_sampling_displacement

RECIPE = dict(
    ds="IXI",
    config="CTCF-CascadeA-Mamba",
    l3_svf=0,
    time_steps=6,
    image_channels=2,
    finetune=1,
    lr=2e-5,
    w_reg=4,
    w_ncc=1,
    w_icon=0.05,
    w_jac=0.005,
    jac_mode="central",
    max_epoch=100,
    max_train_iters=399,
    batch_size=1,
    seed=0,
)


def training_args(args, arm):
    train = parse_args([])
    vars(train).update(
        RECIPE,
        ixi_pairs=str(args.manifest),
        ixi_arm=arm,
        init_ckpt=str(args.checkpoint),
        precision=args.precision,
        use_tb=0,
        quiet=1,
        paths=3,
    )
    return train


def build_runner(args, arm):
    return Runner(training_args(args, arm), args.device)


def expansion_check(scalar, expanded, inputs, device, precision):
    configure_precision(precision, device)
    x, y, b2_x, b2_y = inputs
    with torch.no_grad(), precision_autocast(precision, device):
        reference = scalar(x[:, :1], y[:, :1], alpha_l1=1, alpha_l3=1)[1].float()
        fields = [expanded(a, b, alpha_l1=1, alpha_l3=1)[1].float() for a, b in ((x, y), (b2_x, b2_y))]
    gaps = [field - reference for field in fields]
    return {
        "rms_voxels": [float(g.square().mean().sqrt()) for g in gaps],
        "max_voxels": [float(g.abs().max()) for g in gaps],
    }, fields[0]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def tensors(batch, device):
    return [value.unsqueeze(0).to(device, torch.float32) for value in batch]


def gradient_dose(runner, dataset, count=8):
    rows = []
    sampler = IXIPairSampler(len(dataset), 2026, "readiness")
    runner.model.eval()
    for index, pair in enumerate(sampler):
        if index == count:
            break
        x, y, t2_x, t2_y = tensors(dataset[pair], runner.device)
        fields = [runner._forward_flow(a, b).detach().requires_grad_(True) for a, b in ((x, y), (y, x))]
        warped = [runner.ctx.reg_model((a, f)) for a, f in zip((x, y), fields, strict=True)]
        d1, d2 = runner.contrast_terms(x, y, *warped, *fields, (t2_x, t2_y))
        g1 = torch.autograd.grad(d1, fields, retain_graph=True)
        g2 = torch.autograd.grad(d2, fields)
        n1 = sum(float(g.square().sum()) for g in g1) ** 0.5
        n2 = sum(float(g.square().sum()) for g in g2) ** 0.5
        mixed = sum(float(((a + b) / 2).square().sum()) for a, b in zip(g1, g2, strict=True)) ** 0.5
        if not all(math.isfinite(n) for n in (n1, n2, mixed)) or min(n1, n2) <= 0:
            raise FloatingPointError(f"Invalid contrast gradients on Train pair {pair}: {n1}, {n2}, {mixed}")
        rows.append(
            {
                "moving": dataset.rows[pair[0]]["id"],
                "fixed": dataset.rows[pair[1]]["id"],
                "t2_over_t1": n2 / n1,
                "mixed_over_t1": mixed / n1,
                "cosine": sum(float((a * b).sum()) for a, b in zip(g1, g2, strict=True)) / (n1 * n2),
            }
        )
    if len(rows) != count:
        raise ValueError(f"Readiness needs {count} fixed Train pairs")
    medians = {key: statistics.median(row[key] for row in rows) for key in ("t2_over_t1", "mixed_over_t1", "cosine")}
    return {
        "pairs": rows,
        "medians": medians,
        "passed": all(0.5 <= medians[key] <= 2 for key in ("t2_over_t1", "mixed_over_t1")),
    }


def check(args, manifest):
    if (
        manifest["expected_counts"] != {"training": 397, "validation": 58}
        or manifest["shape"] != [160, 192, 224]
        or manifest.get("training_exclusions") != list(PAIRED_TRAIN_EXCLUSIONS)
    ):
        raise ValueError("Readiness requires 397 Train / 58 Val subjects and the two documented raw-T2 exclusions")
    source = torch.load(args.checkpoint, map_location="cpu")
    if int(source["epoch"]) + 1 != 250 or source.get("precision") != "bf16":
        raise ValueError("The pilot parent must be Mamba294 last after 250 epochs, recorded BF16")
    runner = build_runner(args, "b1")
    runner.model.eval()
    scalar = (
        get_model_adapter("ctcf")
        .build(config_key="CTCF-CascadeA-Mamba", l3_svf=False, time_steps=6)
        .to(args.device)
        .eval()
    )
    load_checkpoint_state(scalar, str(args.checkpoint), strict=True)
    dataset = IXIPairedDataset(manifest, "b1", validation=True)
    x, y, xs, ys = tensors(dataset[0], args.device)
    b2_x, b2_y, _, _ = tensors(IXIPairedDataset(manifest, "b2", validation=True)[0], args.device)
    inputs = (x, y, b2_x, b2_y)
    structural, _ = expansion_check(scalar, runner.model, inputs, args.device, "fp32")
    numerical, expanded = expansion_check(scalar, runner.model, inputs, args.device, args.precision)
    expansion = {
        "fp32_structural": structural,
        "inference_precision": numerical,
        "precision": args.precision,
        "passed": max(structural["rms_voxels"]) <= 1e-4 and max(structural["max_voxels"]) <= 1e-3,
    }
    dose = gradient_dose(runner, IXIPairedDataset(manifest, "b1"))
    started = sync(args.device)
    _, pair_report = score_saved_field(
        legacy_sampling_displacement(expanded.float()),
        "readiness",
        0,
        xs,
        ys,
        args.out / "pair",
        profile=metric_profile_for("IXI"),
    )
    pair_seconds = sync(args.device) - started
    del scalar, runner, source
    profiles = {}
    for arm in ("b0", "b1", "b2"):
        runner = build_runner(args, arm)
        train = IXIPairedDataset(manifest, arm)
        pairs = iter(IXIPairSampler(len(train), 0, "profile"))
        runner.model.train()
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        timings = []
        for index in range(8):
            started = sync(args.device)
            batch = [value.unsqueeze(0) for value in train[next(pairs)]]
            runner.optimizer.zero_grad(set_to_none=True)
            with precision_autocast(args.precision, args.device):
                loss, _ = runner.train_step(batch, 0)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite {arm} profile loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), float("inf"), error_if_nonfinite=True)
            runner.optimizer.step()
            elapsed = sync(args.device) - started
            if index >= 3:
                timings.append(elapsed)
        started = sync(args.device)
        loader = torch.utils.data.DataLoader(IXIPairedDataset(manifest, arm, validation=True), batch_size=1)
        validate(
            runner.model,
            loader,
            args.device,
            runner._forward_flow,
            build_val_dice_fn(runner.args, "IXI"),
            RegisterModel,
            max_batches=2,
            ds="IXI",
        )
        validation_seconds = (sync(args.device) - started) / 2
        profiles[arm] = {
            "update_seconds": statistics.mean(timings),
            "validation_case_seconds": validation_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3
            if args.device.type == "cuda"
            else None,
        }
        del runner
    training_hours = sum(p["update_seconds"] * 39900 for p in profiles.values()) / 3600
    validation_hours = sum(p["validation_case_seconds"] * 58 * 100 for p in profiles.values()) / 3600
    final_hours = pair_seconds * 58 * 4 / 3600
    estimate = training_hours + validation_hours + final_hours + args.consensus_reserve_hours
    report = {
        "expansion": expansion,
        "gradient_dose": dose,
        "pair": pair_report,
        "profiles": profiles,
        "estimated_gpu_hours": estimate,
        "budget_gpu_hours": args.budget_hours,
        "budget_components_hours": {
            "training": training_hours,
            "epoch_validation": validation_hours,
            "final_fields": final_hours,
            "consensus_reserve": args.consensus_reserve_hours,
        },
        "budget_note": "Estimate from short full-size probes, not a measured complete run",
        "passed": expansion["passed"]
        and dose["passed"]
        and not pair_report["errors"]
        and estimate <= args.budget_hours,
    }
    atomic_write_json(args.out / "report.json", report)
    if not report["passed"]:
        raise RuntimeError("IXI readiness held the training; inspect report.json, do not silently alter the recipe")


def evaluate(args, manifest):
    source = torch.load(args.checkpoint, map_location="cpu")
    if "arguments" in source:
        trained = source["arguments"]
        expected = RECIPE | {
            "ixi_arm": args.arm,
            "precision": args.precision,
            "ixi_manifest_sha256": manifest["fingerprint"],
        }
        if any(trained.get(key) != value for key, value in expected.items()):
            raise ValueError("Evaluation arm, architecture, precision or manifest differs from the trained checkpoint")
    elif args.arm != "b0" or int(source["epoch"]) + 1 != 250 or source.get("precision") != args.precision:
        raise ValueError("Only the fixed Mamba294 parent may be scored without paired-IXI training metadata")
    del source
    runner = build_runner(args, args.arm)
    runner.model.eval()
    dataset = IXIPairedDataset(manifest, args.arm, validation=True)
    binding = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": manifest["fingerprint"],
        "arm": args.arm,
        "precision": args.precision,
        "limit": args.limit,
    }
    recipe = args.out / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != binding:
        raise ValueError("Evaluation belongs to other weights, data or contrast settings")
    atomic_write_json(recipe, binding)
    rows, failures = {"FF": [], "REP": []}, {}
    for index in range(min(len(dataset), args.limit or len(dataset))):
        moving, fixed = dataset.pairs[index]
        case_id = f"{dataset.rows[moving]['id']}__{dataset.rows[fixed]['id']}"
        directory = args.out / "pairs" / case_id
        x, y, xs, ys = tensors(dataset[index], args.device)
        started = sync(args.device)
        flow = runner._forward_flow(x, y)
        seconds = sync(args.device) - started
        values, report = score_saved_field(
            legacy_sampling_displacement(flow), case_id, seconds, xs, ys, directory, profile=metric_profile_for("IXI")
        )
        for mode, row in values.items():
            rows[mode].append(row)
        if report["errors"] or not report["global_homeomorphism"]:
            failures[case_id] = report
        print(f"[IXI CASE] {args.arm} {index + 1}/{len(dataset)} {case_id}", flush=True)
    for mode, values in rows.items():
        if values:
            (args.out / mode).mkdir(exist_ok=True)
            write_results(values, str(args.out / mode), args.arm, str(args.checkpoint), "IXI Val subject pairs")
    atomic_write_json(
        args.out / "report.json",
        {
            "pairs": len(rows["FF"]),
            "failures": failures,
            "complete_certified_endpoint": len(rows["REP"]) == 58 and not failures,
            "scope": "Exploratory Val, two directions per pair",
        },
    )
    if failures:
        raise RuntimeError("Raw scores are saved; one or more repaired fields did not receive a global certificate")


def summary(args):
    result, errors = {}, []
    certified = {
        arm: json.loads((args.out / arm / "report.json").read_text()).get("complete_certified_endpoint", False)
        if (args.out / arm / "report.json").exists()
        else False
        for arm in ("b0", "b1", "b2", "parent")
    }
    for mode in ("FF", "REP"):
        tables = {}
        for arm in ("b0", "b1", "b2", "parent"):
            path = args.out / arm / mode / "per_case.csv"
            tables[arm] = list(csv.DictReader(path.open(newline=""))) if path.exists() else []
        for arm, control in (("b1", "b0"), ("b2", "b0"), ("b2", "b1"), ("b0", "parent")):
            try:
                result[f"{mode}_{arm}-{control}"] = paired_dice_delta(tables[arm], tables[control])
                comparison = result[f"{mode}_{arm}-{control}"]
                comparison["complete_endpoint"] = comparison["pairs"] == 29
                if mode == "REP":
                    comparison["certified_comparison"] = certified[arm] and certified[control]
                    comparison["resource_signal"] &= comparison["certified_comparison"]
                comparison["resource_signal"] &= comparison["complete_endpoint"]
            except ValueError as error:
                errors.append(f"{mode}_{arm}-{control}: {error}")
    result["errors"] = errors
    atomic_write_json(args.out / "paired_summary.json", result)
    if errors:
        raise RuntimeError("Some paired comparisons are incomplete; valid raw summaries have been saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "check", "evaluate", "summary"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("b0", "b1", "b2"), default="b0")
    parser.add_argument("--exp")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp32", "tf32"), default="bf16")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget_hours", type=float, default=72)
    parser.add_argument("--consensus_reserve_hours", type=float, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        summary(args)
        return
    if args.manifest is None or args.checkpoint is None:
        parser.error("check/evaluate need --manifest and --checkpoint")
    configure_precision(args.precision, args.device)
    manifest = read_paired_manifest(
        args.manifest, verify_bytes=args.stage == "check", require_t2=args.stage == "check" or args.arm == "b2"
    )
    if args.stage == "train":
        if not args.exp:
            parser.error("train requires --exp")
        train = training_args(args, args.arm)
        train.exp = args.exp
        if args.resume:
            train.init_ckpt, train.resume = "", str(args.resume)
        train_main(train)
    elif args.stage == "check":
        check(args, manifest)
    else:
        evaluate(args, manifest)


if __name__ == "__main__":
    main()
