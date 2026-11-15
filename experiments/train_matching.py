"""Learn correspondence costs from Train-only contrasts and differentiable refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from experiments.core.distributed import restore_training_state, restored_state_digest, training_state
from experiments.core.helpers import save_ckpt
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from experiments.core.precision import configure_precision, precision_autocast
from models.Matching import (
    FeatureEncoder,
    candidate_logits,
    feature_points,
    flow_inside_fraction,
    matching_energy,
    refine,
    sample_voxels,
    voxel_grid,
)
from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, sha256_file
from utils import setup_device
from utils.field import _warp, legacy_sampling_displacement


def load_encoder(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("architecture") != "residual_fpn_native_v1":
        raise ValueError("Unsupported Matching architecture; the old prototype is not a multiscale checkpoint")
    model = FeatureEncoder(**saved["encoder_options"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, saved


def load_base(path, shape, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    options = saved.get("arguments", {})
    model = (
        get_model_adapter("ctcf")
        .build(
            config_key=options.get("config", "CTCF-CascadeA-Mamba"),
            time_steps=options.get("time_steps", 6),
            img_size=tuple(shape),
            image_channels=options.get("image_channels") or 1,
            l3_svf=bool(options.get("l3_svf", 0)),
        )
        .to(device)
    )
    load_checkpoint_state(model, str(path), strict=True)
    model.requires_grad_(False).eval()
    return model


@torch.no_grad()
def base_flow(model, moving, fixed, precision="bf16"):
    channels = model.cfg.image_channels
    x, y = [value[:, :1].expand(-1, channels, -1, -1, -1) for value in (moving, fixed)]
    with precision_autocast(precision, moving.device):
        flow = model(x, y, alpha_l1=1.0, alpha_l3=1.0)[1]
    return legacy_sampling_displacement(flow.float())


def refine_config(saved):
    recipe = saved["recipe"]
    return {
        "steps": recipe["solver_steps"],
        **{key: recipe[key] for key in ("step_size", "reg_weight", "outside_weight", "max_update")},
    }


def smooth_perturbation(shape, device, amplitude):
    noise = torch.randn(1, 3, 3, 3, 3, device=device)
    return amplitude * F.interpolate(noise, size=shape, mode="trilinear", align_corners=True).tanh()


def query_points(mask, count):
    indices = torch.nonzero(mask[0, 0], as_tuple=False)
    if len(indices) < count:
        raise ValueError(f"Foreground/query support has {len(indices)} positions, needs {count}")
    chosen = indices[torch.randperm(len(indices), device=indices.device)[:count]]
    return chosen.T.unsqueeze(0).to(torch.float32)


def known_warp_loss(encoder, image, *, points, temperature, amplitude=1.0):
    shape = image.shape[2:]
    flow = smooth_perturbation(shape, image.device, amplitude)
    with torch.no_grad():
        warped = _warp(image, flow)
        gamma = torch.exp(0.15 * torch.randn(1, image.shape[1], 1, 1, 1, device=image.device))
        warped = torch.where(warped > 0, warped.clamp_min(1e-6).pow(gamma), warped)
        support = warped[:, :1] > 0
        grid = voxel_grid(shape, device=image.device) + flow
        for axis, n in enumerate(shape):
            support &= (grid[:, axis : axis + 1] > 0) & (grid[:, axis : axis + 1] < n - 1)
        selected = query_points(support, points)
        corresponding = selected + sample_voxels(flow, selected)
    source_features, warped_features = encoder(image), encoder(warped)
    first = F.normalize(
        sample_voxels(warped_features, feature_points(selected, shape, warped_features.shape[2:])), dim=1
    ).transpose(1, 2)
    second = F.normalize(
        sample_voxels(source_features, feature_points(corresponding, shape, source_features.shape[2:])), dim=1
    ).transpose(1, 2)
    distances = first.square().sum(-1).unsqueeze(2) + second.square().sum(-1).unsqueeze(1)
    scores = -(distances - 2 * first @ second.transpose(1, 2)) / temperature
    target = torch.arange(points, device=image.device)
    return 0.5 * (F.cross_entropy(scores[0], target) + F.cross_entropy(scores[0].T, target))


def source_image(inputs, side, source):
    return torch.cat((inputs[:, :1], side if source == "paired" else inputs[:, :1]), dim=1)


def training_pair(dataset, step, seed, device):
    generator = torch.Generator().manual_seed(seed + step)
    moving = int(torch.randint(len(dataset), (), generator=generator))
    fixed = int(torch.randint(len(dataset) - 1, (), generator=generator))
    pair = moving, fixed + int(fixed >= moving)
    return [value.unsqueeze(0).to(device, torch.float32) for value in dataset[pair]]


def teacher_subjects(manifest):
    rows = manifest["training"]
    if len(rows) < 3:
        raise ValueError("Teacher fitting and monitoring need at least three Train subjects")
    ordered = sorted(rows, key=lambda row: hashlib.sha256(f"CTCF-teacher-20261115:{row['id']}".encode()).hexdigest())
    heldout = {row["id"] for row in ordered[: min(16, len(rows) - 2)]}
    return [row["id"] for row in rows if row["id"] not in heldout], sorted(heldout)


def matching_step(encoder, teacher, base, batch, args):
    x, y, t2_x, t2_y = batch
    if args.kind == "teacher":
        loss = known_warp_loss(
            encoder,
            source_image(x, t2_x, args.source),
            points=args.points,
            temperature=args.temperature,
            amplitude=args.known_amplitude,
        )
        return loss, {"known": float(loss.detach())}
    shape = x.shape[2:]
    initial = base_flow(base, x[:, :1], y[:, :1], args.base_precision)
    query_flow = initial + smooth_perturbation(shape, x.device, args.query_amplitude)
    with torch.no_grad():
        teacher_x = teacher(source_image(x, t2_x, args.source))
        teacher_y = teacher(source_image(y, t2_y, args.source))
        coordinates = voxel_grid(shape, device=x.device) + query_flow
        support = (y[:, :1] > 0) & (sample_voxels(x[:, :1], coordinates) > 0)
        query_support = support.clone()
        for axis, n in enumerate(shape):
            query_support &= (coordinates[:, axis : axis + 1] >= 2) & (coordinates[:, axis : axis + 1] <= n - 3)
        points = query_points(query_support, args.points)
        offsets = torch.cartesian_prod(*[torch.tensor([-2.0, 0.0, 2.0], device=x.device)] * 3)
        teacher_logits = candidate_logits(
            teacher_y, teacher_x, query_flow, offsets, temperature=args.temperature, points=points
        )
        targets = F.softmax(teacher_logits, dim=1)
    student_x, student_y = encoder(x[:, :1]), encoder(y[:, :1])
    logits = candidate_logits(student_y, student_x, query_flow, offsets, temperature=args.temperature, points=points)
    value = F.kl_div(F.log_softmax(logits, dim=1), targets, reduction="none").sum(1).mean()
    known = known_warp_loss(
        encoder, x[:, :1], points=args.points, temperature=args.temperature, amplitude=args.known_amplitude
    )
    loss, outer = value + args.known_weight * known, value.new_zeros(())
    if args.mode == "solver":
        refined = refine(student_y, student_x, query_flow, mask=support, **refine_config({"recipe": vars(args)}))
        outer = matching_energy(
            teacher_y, teacher_x, refined, reg_weight=args.reg_weight, mask=support, outside_weight=args.outside_weight
        )
        loss = loss + args.outer_weight * outer
    return loss, {
        "value": float(value.detach()),
        "known": float(known.detach()),
        "outer": float(outer.detach()),
        "teacher_entropy": float(-(targets * targets.clamp_min(1e-8).log()).sum(1).mean()),
        "teacher_peak": float(targets.amax(1).mean()),
        "inside": float(flow_inside_fraction(query_flow)),
        "query_support": int(query_support.sum()),
        "refinement_grid_voxels": math.prod(shape),
    }


def _recipe(args):
    return {
        key: getattr(args, key)
        for key in (
            "updates",
            "lr",
            "seed",
            "points",
            "temperature",
            "solver_steps",
            "step_size",
            "reg_weight",
            "outside_weight",
            "max_update",
            "known_weight",
            "outer_weight",
            "query_amplitude",
            "known_amplitude",
            "base_precision",
        )
    } | {"optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}}


def run(args):
    args.kind = args.stage if args.stage != "profile" else args.kind
    args.max_train_iters = args.updates
    if min(args.updates, args.points, args.save_every, args.log_every, args.profile_updates) < 1:
        raise ValueError("Training/update/count arguments must be positive")
    if args.stage == "profile" and args.profile_updates < 2:
        raise ValueError("Profiling needs at least one warmup and one measured update")
    for name in ("lr", "temperature", "step_size", "max_update", "known_amplitude", "max_peak_gib"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("reg_weight", "outside_weight", "known_weight", "outer_weight", "query_amplitude"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.kind == "student" and args.base_checkpoint is None:
        raise ValueError("Student/profile requires --base_checkpoint")
    if args.kind == "student" and args.stage != "profile" and args.teacher is None:
        raise ValueError("Student training requires --teacher")
    device = torch.device("cpu") if args.device == "cpu" else setup_device(args.gpu, args.seed)
    torch.manual_seed(args.seed)
    configure_precision("fp32", device)
    manifest = read_ixi_manifest(args.manifest, require_t2=args.source == "paired")
    calibration_sha = None
    ncc_config = None
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text())
        if calibration["binding"]["manifest_sha256"] != manifest["fingerprint"]:
            raise ValueError("Solver calibration belongs to another Train manifest")
        for key in ("step_size", "reg_weight"):
            setattr(args, key, calibration["selected"][key])
        calibration_sha = sha256_file(args.calibration)
        ncc_config = calibration["ncc_selected"]
    dataset = IXIPairedDataset(manifest, "b1" if args.source == "paired" else "b0")
    fit_ids, heldout_ids = teacher_subjects(manifest)
    if args.kind == "teacher":
        dataset.rows = [row for row in dataset.rows if row["id"] in fit_ids]
    teacher = base = None
    if args.kind == "student":
        base = load_base(args.base_checkpoint, manifest["shape"], device)
        if args.teacher is not None:
            teacher, saved_teacher = load_encoder(args.teacher, device)
            if saved_teacher["kind"] != "teacher" or saved_teacher["source"] != args.source:
                raise ValueError("The teacher checkpoint source/kind differs")
            if saved_teacher["step"] != saved_teacher["recipe"]["updates"]:
                raise ValueError("The teacher checkpoint has not completed its fixed training budget")
            if saved_teacher["manifest_sha256"] != manifest["fingerprint"]:
                raise ValueError("The teacher was trained on another manifest")
            if saved_teacher["encoder_options"]["channels"] != args.channels:
                raise ValueError("Teacher and student descriptor channel counts differ")
            if saved_teacher["teacher_fit_ids"] != fit_ids or saved_teacher["teacher_heldout_ids"] != heldout_ids:
                raise ValueError("Teacher monitoring/fitting subjects differ")
        else:
            teacher = FeatureEncoder(
                2,
                width=args.teacher_width,
                channels=args.channels,
                stride=args.stride,
                precision=args.encoder_precision,
            ).to(device)
        teacher.requires_grad_(False).eval()
    torch.manual_seed(args.seed)
    options = {
        "input_channels": 2 if args.kind == "teacher" else 1,
        "width": args.teacher_width if args.kind == "teacher" else args.width,
        "channels": args.channels,
        "stride": args.stride,
        "precision": args.encoder_precision,
        "checkpoint_blocks": True,
    }
    encoder = FeatureEncoder(**options).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    binding = {
        "architecture": "residual_fpn_native_v1",
        "kind": args.kind,
        "source": args.source,
        "mode": args.mode if args.kind == "student" else None,
        "encoder_options": options,
        "recipe": _recipe(args),
        "manifest_sha256": manifest["fingerprint"],
        "base_sha256": sha256_file(args.base_checkpoint) if args.kind == "student" else None,
        "teacher_sha256": sha256_file(args.teacher) if args.kind == "student" and args.teacher else None,
        "calibration_sha256": calibration_sha,
        "ncc_config": ncc_config,
        "teacher_fit_ids": fit_ids,
        "teacher_heldout_ids": heldout_ids,
        "initial_model_sha256": restored_state_digest(encoder, optimizer, device)["model"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    previous_seconds = 0.0
    if args.stage == "profile":
        start, history = 0, []
    else:
        recipe_path, checkpoint = args.out / "recipe.json", args.out / "ckpt/last.pth"
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != binding:
            raise ValueError("Matching run inputs/recipe changed; use a new output directory")
        atomic_write_json(recipe_path, binding)
        start, history = 0, []
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if any(saved[key] != value for key, value in binding.items()):
                raise ValueError("Matching checkpoint inputs/recipe changed")
            encoder.load_state_dict(saved["state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            start, history = saved["step"], saved["history"]
            previous_seconds = saved.get("wall_seconds", 0.0)
            restore_training_state(saved["training"], args, device, 0)
        if history:
            atomic_write_text(args.out / "metrics.csv", rows_to_csv(list(history[0]), history))
        (args.out / "ckpt").mkdir(exist_ok=True)
    encoder.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started, timings = time.perf_counter(), []
    endpoint = args.profile_updates if args.stage == "profile" else args.updates
    for step in range(start, endpoint):
        tick = time.perf_counter()
        batch = training_pair(dataset, step, args.seed, device)
        optimizer.zero_grad(set_to_none=True)
        loss, logs = matching_step(encoder, teacher, base, batch, args)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite matching loss at update {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), float("inf"), error_if_nonfinite=True)
        if args.stage == "profile" and not bool(norm > 0):
            raise FloatingPointError("The profiled matching encoder has no parameter gradient")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - tick)
        if (step + 1) % args.log_every == 0 or step + 1 == endpoint:
            row = {"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm), **logs}
            history.append(row)
            atomic_write_text(args.out / "metrics.csv", rows_to_csv(list(row), history))
            print(
                f"[matching {args.kind}/{args.source}/{args.mode}] step={step + 1}/{endpoint} loss={row['loss']:.6g}",
                flush=True,
            )
        if args.stage != "profile" and ((step + 1) % args.save_every == 0 or step + 1 == endpoint):
            save_ckpt(
                binding
                | {
                    "state_dict": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "history": history,
                    "training": training_state(args, device),
                    "wall_seconds": previous_seconds + time.perf_counter() - started,
                },
                str(args.out / "ckpt"),
                False,
            )
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    memory_ok = peak is None or peak <= args.max_peak_gib
    report = binding | {
        "completed_updates": endpoint,
        "resumed_update": start,
        "wall_seconds": previous_seconds + elapsed,
        "attempt_wall_seconds": elapsed,
        "measured_updates": len(timings[1:]),
        "update_seconds": sum(timings[1:]) / max(1, len(timings[1:])),
        "peak_allocated_gib": peak,
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "descriptor_stride": args.stride,
        "field_shape": manifest["shape"],
        "random_teacher_profile": args.stage == "profile" and args.kind == "student" and args.teacher is None,
        "scope": "Train-only correspondence pseudo-targets; no anatomical labels or validation selection",
    }
    if args.stage == "profile":
        report.update(passed=memory_ok, reason="OK" if memory_ok else "MEMORY_LIMIT", max_peak_gib=args.max_peak_gib)
    atomic_write_json(
        args.out / ("profile.json" if args.stage == "profile" else "report.json"), report, allow_nan=False
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "student", "profile"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("teacher", "student"), default="student")
    parser.add_argument("--source", choices=("t1", "paired"), default="paired")
    parser.add_argument("--mode", choices=("value", "solver"), default="value")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--teacher_width", type=int, default=24)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--encoder_precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--solver_steps", type=int, default=3)
    parser.add_argument("--step_size", type=float, default=0.25)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--outside_weight", type=float, default=1.0)
    parser.add_argument("--max_update", type=float, default=0.5)
    parser.add_argument("--known_weight", type=float, default=0.1)
    parser.add_argument("--outer_weight", type=float, default=1.0)
    parser.add_argument("--query_amplitude", type=float, default=0.5)
    parser.add_argument("--known_amplitude", type=float, default=1.0)
    parser.add_argument("--base_precision", choices=("bf16", "fp32", "tf32"), default="bf16")
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--profile_updates", type=int, default=3)
    parser.add_argument("--max_peak_gib", type=float, default=60.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except torch.cuda.OutOfMemoryError as error:
        if args.stage != "profile" or args.device == "cpu" or not torch.cuda.is_available():
            raise
        atomic_write_json(args.out / "profile.json", {"passed": False, "reason": "CUDA_OOM", "error": str(error)})
        return 86
    return 86 if args.stage == "profile" and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
