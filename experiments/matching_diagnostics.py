"""Train-only teacher probes and shared voxel-unit solver calibration."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from experiments.core.precision import configure_precision
from experiments.eval_matching import timed_ncc
from experiments.train_matching import (
    load_base,
    load_encoder,
    query_points,
    smooth_perturbation,
    source_image,
    teacher_subjects,
    training_pair,
)
from models.Matching import FeatureEncoder, candidate_logits, refine, sample_voxels, voxel_grid
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import SegNorm, pkload, setup_device
from utils.field import _warp


def supported_points(image, field, count, *, margin=2):
    positions = voxel_grid(field.shape[2:], device=field.device) + field
    support = image[:, :1] > 0
    for axis, n in enumerate(field.shape[2:]):
        support &= (positions[:, axis : axis + 1] >= margin) & (positions[:, axis : axis + 1] <= n - 1 - margin)
    return query_points(support, count)


def bind_output(args, binding):
    path = args.out / "recipe.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Diagnostic inputs changed; use another output directory")
    atomic_write_json(path, binding)


@torch.no_grad()
def teacher_probes(args, manifest, device):
    teacher, saved = load_encoder(args.teacher, device)
    fit_ids, heldout_ids = teacher_subjects(manifest)
    if (
        saved["kind"] != "teacher"
        or saved["step"] != saved["recipe"]["updates"]
        or saved["manifest_sha256"] != manifest["fingerprint"]
        or saved["teacher_fit_ids"] != fit_ids
        or saved["teacher_heldout_ids"] != heldout_ids
    ):
        raise ValueError("Teacher probes require the completed teacher with the frozen Train monitoring split")
    teacher.eval().requires_grad_(False)
    binding = {
        "teacher_sha256": sha256_file(args.teacher),
        "manifest_sha256": manifest["fingerprint"],
        "base_sha256": sha256_file(args.base_checkpoint) if args.base_checkpoint else None,
        "points": args.points,
        "seed": args.seed,
        "pairs": args.pairs,
    }
    bind_output(args, binding)
    dataset = IXIPairedDataset(manifest, "b1" if saved["source"] == "paired" else "b0")
    dataset.rows = [row for row in dataset.rows if row["id"] in heldout_ids]
    base = load_base(args.base_checkpoint, manifest["shape"], device) if args.base_checkpoint else None
    offsets = torch.tensor(list(product((-2.0, 0.0, 2.0), repeat=3)), device=device)
    rows = []
    count = min(args.pairs, max(1, len(dataset) // 2))
    for index in range(count):
        first, second = 2 * index % len(dataset), (2 * index + 1) % len(dataset)
        x, y, tx, ty = [value.unsqueeze(0).to(device, torch.float32) for value in dataset[(first, second)]]
        truth = smooth_perturbation(manifest["shape"], device, 1.0)
        points = supported_points(_warp(x[:, :1], truth), truth, args.points)
        original = source_image(x, tx, saved["source"])
        permutation = torch.randperm(np.prod(manifest["shape"]), device=device)
        variants = {
            "true": original,
            "copy_t1": source_image(x, x[:, :1], saved["source"]),
            "zero_t2": source_image(x, torch.zeros_like(tx), saved["source"]),
            "foreign_t2": source_image(x, ty, saved["source"]),
            "blank": torch.zeros_like(original),
            "spatial_shuffle": original.flatten(2)[:, :, permutation].reshape_as(original),
        }
        for name, inputs in variants.items():
            fixed, moving = teacher(_warp(inputs, truth)), teacher(inputs)
            logits = candidate_logits(fixed, moving, truth, offsets, points=points)
            probability = logits.softmax(1)
            prediction = logits.argmax(1)
            row = {
                "pair": index,
                "subjects": [dataset.rows[i]["id"] for i in (first, second)],
                "variant": name,
                "known_local_top1": float((prediction == 13).float().mean()),
                "known_offset_error_vox": float(offsets[prediction].norm(dim=-1).mean()),
                "entropy": float(-(probability * probability.clamp_min(1e-8).log()).sum(1).mean()),
                "positive_margin": float(
                    (logits[:, 13] - logits[:, torch.arange(27, device=device) != 13].amax(1)).mean()
                ),
                "feature_spatial_std": float(moving.flatten(2).std(-1).mean()),
            }
            rows.append(row)
        wrong = teacher(source_image(y, ty, saved["source"]))
        logits = candidate_logits(teacher(_warp(original, truth)), wrong, truth, offsets, points=points)
        rows.append(
            {
                "pair": index,
                "variant": "nonmatching_image",
                "known_local_top1": float((logits.argmax(1) == 13).float().mean()),
            }
        )
        if base is not None and first != second:
            from models.Matching import base_flow

            parent = base_flow(base, x[:, :1], y[:, :1])
            points = supported_points(y[:, :1], parent, args.points)
            labels = []
            for subject in (first, second):
                image, segmentation = pkload(dataset.rows[subject]["t1"])
                _, segmentation = SegNorm()([image, segmentation])
                labels.append(torch.as_tensor(np.ascontiguousarray(segmentation), device=device)[None, None].float())
            fixed_labels = sample_voxels(labels[1], points.round())[:, 0]
            for name, (side_x, side_y) in {
                "true": (tx, ty),
                "copy_t1": (x[:, :1], y[:, :1]),
                "zero_t2": (torch.zeros_like(tx), torch.zeros_like(ty)),
                "foreign_t2": (ty, tx),
            }.items():
                fm = teacher(source_image(x, side_x, saved["source"]))
                ff = teacher(source_image(y, side_y, saved["source"]))
                logits = candidate_logits(ff, fm, parent, offsets, points=points)
                target = points + sample_voxels(parent, points) + offsets[logits.argmax(1)].transpose(1, 2)
                predicted = sample_voxels(labels[0], target.round())[:, 0]
                foreground = fixed_labels > 0
                rows.append(
                    {
                        "pair": index,
                        "variant": name,
                        "readout": "intersubject_label_measurement_only",
                        "same_class_fraction": float((predicted == fixed_labels).float().mean()),
                        "foreground_same_class_fraction": float(
                            (predicted[foreground] == fixed_labels[foreground]).float().mean()
                        )
                        if bool(foreground.any())
                        else None,
                        "foreground_queries": int(foreground.sum()),
                    }
                )
        print(f"[TEACHER PROBE] {saved['source']} {index + 1}/{count}", flush=True)
    report = {
        "binding": binding,
        "fit_subjects": len(fit_ids),
        "heldout_subjects": heldout_ids,
        "rows": rows,
        "scope": "Train monitoring only. Synthetic retrieval is not anatomical correspondence; labels measure same-class agreement and never select a checkpoint or solver setting. No diagnostic is an automatic rejection gate.",
    }
    atomic_write_json(args.out / "report.json", report)
    return report


def calibrate(args, manifest, device):
    options = {"width": 16, "channels": 16, "stride": args.stride, "precision": "bf16", "checkpoint_blocks": True}
    torch.manual_seed(0)
    encoder = FeatureEncoder(**options).to(device).eval().requires_grad_(False)
    binding = {
        "manifest_sha256": manifest["fingerprint"],
        "encoder_options": options,
        "seed": args.seed,
        "pairs": args.pairs,
    }
    bind_output(args, binding)
    dataset = IXIPairedDataset(manifest, "b0")
    trials = {(step, reg): [] for step, reg in product((0.05, 0.1, 0.25, 0.5), (0.003, 0.01, 0.03))}
    ncc_trials = {pair: [] for pair in trials}
    before = []
    for index in range(args.pairs):
        torch.manual_seed(args.seed + index)
        x = training_pair(dataset, index, args.seed, device)[0][:, :1]
        with torch.no_grad():
            truth = smooth_perturbation(manifest["shape"], device, 1.0)
            warped = _warp(x, truth)
            initial = truth + smooth_perturbation(manifest["shape"], device, 0.5)
            support = (warped > 0).float()
            fm, ff = encoder(x), encoder(warped)
            before.append(
                float(((initial - truth).square().sum(1, keepdim=True).sqrt() * support).sum() / support.sum())
            )
        for (step, reg), values in trials.items():
            output = refine(
                ff,
                fm,
                initial,
                steps=3,
                step_size=step,
                reg_weight=reg,
                max_update=0.5,
                mask=support,
                create_graph=False,
            )
            error = ((output - truth).square().sum(1, keepdim=True).sqrt() * support).sum() / support.sum()
            if not bool(torch.isfinite(error)):
                raise FloatingPointError("Nonfinite Train-only solver calibration error")
            values.append(float(error))
            ncc, detail = timed_ncc(
                x,
                warped,
                initial,
                60.0,
                {"step_size": step, "reg_weight": reg, "outside_weight": 1.0, "max_update": 0.5},
                max_steps=3,
            )
            if detail["steps"] != 3:
                raise RuntimeError("Train NCC calibration did not complete its fixed three updates")
            error = ((ncc - truth).square().sum(1, keepdim=True).sqrt() * support).sum() / support.sum()
            if not bool(torch.isfinite(error)):
                raise FloatingPointError("Nonfinite Train NCC calibration error")
            ncc_trials[(step, reg)].append(float(error))
        print(f"[SOLVER CALIBRATION] {index + 1}/{args.pairs}", flush=True)
    winner = min(trials, key=lambda pair: (np.mean(trials[pair]), pair[0], -pair[1]))
    ncc_winner = min(ncc_trials, key=lambda pair: (np.mean(ncc_trials[pair]), pair[0], -pair[1]))
    report = {
        "binding": binding,
        "selected": {"step_size": winner[0], "reg_weight": winner[1]},
        "ncc_selected": {"step_size": ncc_winner[0], "reg_weight": ncc_winner[1]},
        "initial_endpoint_error": float(np.mean(before)),
        "trials": [
            {"step_size": step, "reg_weight": reg, "endpoint_errors": values, "mean": float(np.mean(values))}
            for (step, reg), values in trials.items()
        ],
        "scope": "One common randomly initialized main student; native voxel-unit synthetic endpoint error on Train. No anatomical labels, Val Dice or per-arm retuning. Not proof of anatomical quality.",
        "ncc_trials": [
            {"step_size": step, "reg_weight": reg, "endpoint_errors": values, "mean": float(np.mean(values))}
            for (step, reg), values in ncc_trials.items()
        ],
    }
    atomic_write_json(args.out / "calibration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "calibrate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args()
    if min(args.points, args.pairs) < 1 or (args.stage == "teacher" and args.teacher is None):
        parser.error("Positive points/pairs are required; teacher probes also require --teacher")
    device = torch.device("cpu") if args.device == "cpu" else setup_device(seed=args.seed)
    configure_precision("bf16", device)
    torch.manual_seed(args.seed)
    manifest = read_ixi_manifest(args.manifest, require_t2=args.stage == "teacher")
    args.out.mkdir(parents=True, exist_ok=True)
    (teacher_probes if args.stage == "teacher" else calibrate)(args, manifest, device)


if __name__ == "__main__":
    main()
