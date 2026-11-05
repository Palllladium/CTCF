"""Frozen, canonical bridge averaging on the subject-disjoint OASIS Dev cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from experiments.core.data_loaders import oasis_subject_paths, oasis_validation_pairs
from experiments.core.field_scoring import paired_dice_delta, score_saved_field, warp_labels
from experiments.core.inference_metrics import write_results
from experiments.core.precision import configure_precision
from experiments.dispatch1.bridges import Parent
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import pkload
from utils.dice import OASIS_VOI_LABELS, dice_val_subset
from utils.field import compose_flows, legacy_sampling_displacement

SPLIT = Path("datasets/splits/oasis_subject_disjoint_v2/split_manifest.json")
PRIMARY = ("direct", "smooth", "bank0_k16", "bank1_k16")


def certificate_modes(arm):
    return ("FF", "REP") if arm == "direct" else ("REP",) if arm in PRIMARY else ()


@lru_cache(maxsize=32)
def image(path):
    return torch.from_numpy(np.ascontiguousarray(pkload(path)[0], dtype=np.float32))[None, None]


def bridge(parent, moving, fixed, intermediate, device):
    first = legacy_sampling_displacement(parent.flow(image(intermediate).to(device), image(fixed).to(device)))
    second = legacy_sampling_displacement(parent.flow(image(moving).to(device), image(intermediate).to(device)))
    return compose_flows(first, second)


def roughness(flow):
    return sum(float(torch.diff(flow, dim=axis).square().mean()) for axis in (-3, -2, -1)) / 3


def smooth(flow, sigma):
    values = gaussian_filter(flow.detach().cpu().numpy(), sigma=(0, 0, sigma, sigma, sigma), mode="nearest")
    return torch.from_numpy(values).to(flow.device)


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def smoothing_control(parent, training, banks, device):
    started = clock(device)
    pool = sorted(set(training) - set(banks[0]) - set(banks[1]))
    chosen = random.Random(2026).sample(pool, 16)
    pairs = list(zip(chosen[::2], chosen[1::2], strict=True))
    directs, targets = [], []
    for moving, fixed in pairs:
        direct = legacy_sampling_displacement(parent.flow(image(moving).to(device), image(fixed).to(device)))
        total = torch.zeros_like(direct)
        for intermediate in banks[0]:
            total.add_(bridge(parent, moving, fixed, intermediate, device))
        targets.append(roughness(total / len(banks[0])))
        directs.append(direct.cpu())
    target = float(np.mean(targets))
    values = {
        str(sigma): float(np.mean([roughness(smooth(f, sigma)) for f in directs])) for sigma in (0, 0.25, 0.5, 1, 2, 4)
    }
    sigma = min(values, key=lambda value: abs(values[value] - target))
    return {
        "sigma": float(sigma),
        "target_roughness": target,
        "roughness_by_sigma": values,
        "training_pairs": pairs,
        "matched_relative_error": abs(values[sigma] - target) / max(target, 1e-12),
        "wall_seconds": clock(device) - started,
        "scope": "Train-only roughness matching, no Dice selection",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget_hours", type=float, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_precision("bf16", args.device)
    source = torch.load(args.checkpoint, map_location="cpu")
    if int(source["epoch"]) + 1 != 250 or source.get("precision") != "bf16":
        raise ValueError("Consensus requires Mamba294 last after 250 epochs, recorded BF16")
    del source
    training, dev = oasis_subject_paths(args.data, SPLIT)
    pairs = oasis_validation_pairs(dev, SPLIT.with_name("pair_manifest.json"))
    banks = [random.Random(seed).sample(sorted(training), 16) for seed in (0, 1)]
    binding = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split_sha256": sha256_file(SPLIT),
        "pairs_sha256": sha256_file(SPLIT.with_name("pair_manifest.json")),
        "precision": "bf16",
        "banks": banks,
        "limit": args.limit,
        "dev_bank": "pair-excluded seed2, separately transductive",
        "certification": "primary REP, direct FF; other variants diagnostic without exact certification",
    }
    recipe = args.out / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != binding:
        raise ValueError("Consensus output belongs to another checkpoint, split or recipe")
    atomic_write_json(recipe, binding)
    parent = Parent.load(str(args.checkpoint), "CTCF-CascadeA-Mamba", False, 6, args.device, precision="bf16")
    if parent.load_report["precision"] != "bf16":
        raise ValueError("Consensus requires the BF16 Mamba294 parent")
    control_file = args.out / "smoothing.json"
    if control_file.exists():
        control = json.loads(control_file.read_text())
    else:
        control = smoothing_control(parent, training, banks, args.device)
        atomic_write_json(control_file, control)
    rows, failures, elapsed_cases, individuals = {}, {}, [], {}
    budget_held = False
    selected = pairs[: args.limit] if args.limit else pairs
    for index, (moving, fixed) in enumerate(selected):
        started = time.perf_counter()
        case_id = f"{Path(moving).stem}__{Path(fixed).stem}"
        xs, ys = [
            torch.from_numpy(np.ascontiguousarray(pkload(p)[1]))[None, None].to(args.device) for p in (moving, fixed)
        ]

        def record(arm, flow, generation_seconds, case_id=case_id, xs=xs, ys=ys):
            directory = args.out / arm / "pairs" / case_id
            values, report = score_saved_field(
                flow, case_id, generation_seconds, xs, ys, directory, certify_modes=certificate_modes(arm)
            )
            for mode, row in values.items():
                rows.setdefault((arm, mode), []).append(row)
            if report["errors"] or (arm in PRIMARY and not report["global_homeomorphism"]):
                failures[f"{arm}/{case_id}"] = report

        x, y = image(moving).to(args.device), image(fixed).to(args.device)
        generated = clock(args.device)
        direct = legacy_sampling_displacement(parent.flow(x, y))
        direct_seconds = clock(args.device) - generated
        record("direct", direct, direct_seconds)
        generated = clock(args.device)
        smoothed = smooth(direct, control["sigma"])
        record("smooth", smoothed, direct_seconds + clock(args.device) - generated)
        member_rows = []
        for bank_index, bank in enumerate(banks):
            total = torch.zeros_like(direct)
            generation_seconds = 0.0
            for k, intermediate in enumerate(bank, start=1):
                generated = clock(args.device)
                field = bridge(parent, moving, fixed, intermediate, args.device)
                total.add_(field)
                generation_seconds += clock(args.device) - generated
                warped = warp_labels(xs, field, "canonical").long()
                dice = float(dice_val_subset(warped, ys.long(), OASIS_VOI_LABELS))
                member_rows.append(
                    {"bank": bank_index, "k": k, "intermediate": Path(intermediate).stem, "dice_mean": dice}
                )
                if k in (1, 4, 16):
                    record(f"bank{bank_index}_k{k}", total / k, generation_seconds)
            member_dice = [r["dice_mean"] for r in member_rows if r["bank"] == bank_index]
            individuals.setdefault(bank_index, []).append(
                {"case_id": case_id, "dice_mean": float(np.mean(member_dice))}
            )
        seed = int(hashlib.sha256("|".join(sorted((moving, fixed))).encode()).hexdigest()[:8], 16)
        dev_bank = random.Random(seed + 2).sample(sorted(set(dev) - {moving, fixed}), 16)
        total = torch.zeros_like(direct)
        generated = clock(args.device)
        for intermediate in dev_bank:
            total.add_(bridge(parent, moving, fixed, intermediate, args.device))
        record("dev_transductive_k16", total / 16, clock(args.device) - generated)
        member_file = args.out / "members" / f"{case_id}.json"
        elapsed = time.perf_counter() - started
        if member_file.exists():
            elapsed = max(elapsed, json.loads(member_file.read_text()).get("wall_seconds", 0))
        atomic_write_json(member_file, {"train_bridges": member_rows, "dev_bank": dev_bank, "wall_seconds": elapsed})
        elapsed_cases.append(elapsed)
        atomic_write_json(
            args.out / "progress.json",
            {
                "completed_directions": index + 1,
                "total": len(selected),
                "seconds_per_direction": elapsed_cases,
                "failures": sorted(failures),
            },
        )
        print(f"[CONSENSUS CASE] {index + 1}/{len(selected)} {case_id}", flush=True)
        if index == 1 and control["wall_seconds"] + np.mean(elapsed_cases) * len(selected) > args.budget_hours * 3600:
            budget_held = True
            break
    for (arm, mode), values in rows.items():
        destination = args.out / arm / mode
        destination.mkdir(parents=True, exist_ok=True)
        write_results(values, str(destination), arm, str(args.checkpoint), "OASIS subject-disjoint Dev")
    comparisons = {}
    for (arm, mode), values in rows.items():
        controls = ["direct", "smooth"]
        if arm.startswith("bank") and not arm.endswith("_k1"):
            controls.append(arm.split("_")[0] + "_k1")
        for reference in controls:
            if arm == reference:
                continue
            try:
                result = paired_dice_delta(values, rows.get((reference, mode), []))
            except ValueError as error:
                result = {"status": "INCOMPLETE", "error": str(error)}
            if "mean" in result:
                result["complete_endpoint"] = len(elapsed_cases) == len(pairs)
                result["resource_signal"] &= result["complete_endpoint"]
                if mode == "REP":
                    result["certified_comparison"] = (
                        arm in PRIMARY
                        and reference in PRIMARY
                        and not any(key.startswith((arm + "/", reference + "/")) for key in failures)
                    )
                    result["resource_signal"] &= result["certified_comparison"]
            comparisons[f"{mode}_{arm}-{reference}"] = result
    for bank, values in individuals.items():
        try:
            result = paired_dice_delta(rows[(f"bank{bank}_k16", "FF")], values)
        except ValueError as error:
            result = {"status": "INCOMPLETE", "error": str(error)}
        comparisons[f"FF_bank{bank}_k16-mean_individual_dice"] = result
    atomic_write_json(args.out / "paired_summary.json", comparisons)
    atomic_write_json(
        args.out / "report.json",
        {
            "directions": len(elapsed_cases),
            "failures": failures,
            "budget_held": budget_held,
            "certification_policy": {arm: list(certificate_modes(arm)) for arm in PRIMARY},
            "diagnostic_variants": "K1/K4 and Dev bank: repaired Dice only, no certified claim",
            "timing": "per-direction wall time includes every arm, repair, disk and CPU certification",
            "interpretation": "K1 may benefit from composition; Dev-bank is transductive, not the primary result",
        },
    )
    if budget_held:
        raise RuntimeError(
            "Consensus profile exceeds its reserve; saved partial results are not a negative hypothesis verdict"
        )
    if failures:
        raise RuntimeError("Raw consensus scores saved; some final fields were not globally certified")


if __name__ == "__main__":
    main()
