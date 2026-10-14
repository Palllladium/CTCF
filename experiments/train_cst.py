"""Train the corrected CST adaptation on OASIS without labels: CST (Bigalke et al., MICCAI 2023) alone (--arm cst) or
with CoRe's affine-equivariance term (--arm cst_eq). Rules: PILOTS.md, "CST/CoRe pilot", in the memory registry.

--init_only builds the shared start once: the (fixed, moving) pair list, the stage-0 pseudo labels from a random
feature net, the start check on the 19 test pairs and the measured cost of one arm. Each arm then trains a freshly
initialised net from that start for `stages` x `iters_per_stage` iterations and rebuilds the pseudo labels after every
stage but the last. Labels of the 19 test pairs are read only for the start check and the monitoring log.

    python -m experiments.train_cst --init_only --out results/cst/<stamp>/init --paths 3 --gpu 0
    python -m experiments.train_cst --arm cst_eq --init results/cst/<stamp>/init --out results/cst/<stamp>/cst_eq_s0 \
        --seed 0 --paths 3 --gpu 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.core.path_profiles import get_dataset_paths
from models.CST.adaptation import (
    augment_pair,
    equivariance_loss,
    half,
    make_pairs,
    random_affine_pair,
    refine,
    sampling_weights,
    solver_offsets,
    teacher_fields,
    to_canonical,
    voxel_error,
)
from models.CST.core import coupled_convex, make_feature_net
from utils import pkload, setup_device
from utils.dice import OASIS_VOI_LABELS, dice_per_label
from utils.field import _warp

MONITOR_SEED = 900_001  # random projections of the monitoring refinement, fixed per test pair
BRAIN_MARGIN = 1e-2  # brain = normalized intensity above the background value by more than this
AMP_WARMUP = 50  # GradScaler lowers its initial scale over the first steps; their skips are not a failure


def center_crop(t: torch.Tensor, crop) -> torch.Tensor:
    if not crop:
        return t
    starts = [(n - c) // 2 for n, c in zip(t.shape[2:], crop, strict=True)]
    return t[(slice(None), slice(None), *(slice(s, s + c) for s, c in zip(starts, crop, strict=True)))]


def normalise(image: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Z-score over the brain (image > 0); the background becomes one value, returned for padding warps."""
    brain = image > 0
    mean, std = image[brain].mean(), image[brain].std()
    return (image - mean) / std, float(-mean / std)


def volume(array, crop) -> torch.Tensor:
    return center_crop(torch.as_tensor(np.ascontiguousarray(array))[None, None], crop)


def load_training_images(args, device):
    files = sorted(glob.glob(os.path.join(get_dataset_paths(args.paths, "OASIS")["train_dir"], "*.pkl")))
    files = files[: args.limit_files or None]
    images, pads = [], []
    for path in files:
        image, pad = normalise(volume(pkload(path)[0], args.crop).float())
        images.append(image.half())
        pads.append(pad)
    return torch.cat(images).to(device), pads, [Path(f).name for f in files]


def load_test_pairs(args, device):
    """(stem, fixed, moving, moving background, moving labels, fixed labels); pkl item x is moving, y fixed."""
    files = sorted(glob.glob(os.path.join(get_dataset_paths(args.paths, "OASIS")["val_dir"], "*.pkl")))
    pairs = []
    for path in files[: args.limit_test or None]:
        x, y, x_seg, y_seg = pkload(path)
        moving, pad = normalise(volume(x, args.crop).float())
        fixed, _ = normalise(volume(y, args.crop).float())
        segs = (volume(x_seg, args.crop).to(device), volume(y_seg, args.crop).to(device))
        pairs.append((Path(path).stem.removeprefix("p_"), fixed.to(device), moving.to(device), pad, *segs))
    return pairs


def monitor(net, test_pairs, reg_fac) -> dict[str, float]:
    """Mean Dice35 of the identity, solver and refined fields on the test pairs (labels for this log only)."""
    scores = {"identity": [], "solver": [], "refined": []}
    for k, (_, fixed, moving, pad, x_seg, y_seg) in enumerate(test_pairs):
        solver = solver_offsets(net, fixed, moving, pad=pad)
        refined = refine(net, fixed, moving, solver, reg_fac=reg_fac, proj_seed=MONITOR_SEED + k)
        for name, offsets in (("identity", torch.zeros_like(solver)), ("solver", solver), ("refined", refined)):
            warped = _warp(x_seg.float(), to_canonical(offsets), mode="nearest").long()
            scores[name].append(float(np.mean(dice_per_label(warped, y_seg.long(), OASIS_VOI_LABELS))))
    return {name: float(np.mean(values)) for name, values in scores.items()}


def rebuild(net, images, pads, pairs, stage, reg_fac, seed):
    """Pseudo labels (refined offsets at half resolution, float16) and the difficulty of every pair."""
    full = tuple(images.shape[2:])
    fields = torch.empty((len(pairs), 3, *(n // 2 for n in full)), dtype=torch.float16, device=images.device)
    difficulty = torch.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        solver, refined = teacher_fields(
            net,
            images[i : i + 1].float(),
            images[j : j + 1].float(),
            reg_fac=reg_fac,
            proj_seed=seed * 7919 + stage * 100_003 + k,
            pad=pads[j],
        )
        solver, refined = half(solver), half(refined)
        fields[k] = refined[0].half()
        difficulty[k] = float(voxel_error(solver, refined, full).mean())
    return fields, difficulty


def train_step(net, batch, pairs, images, pads, fields, args, it, total, generators, optimizer, scaler):
    full = tuple(images.shape[2:])
    with torch.no_grad():
        views = [
            augment_pair(
                images[i : i + 1].float(),
                images[j : j + 1].float(),
                fields[k : k + 1].float(),
                args.augment,
                generators["aug"],
                (pads[i], pads[j]),
            )
            for k, (i, j) in ((k, pairs[k]) for k in batch)
        ]
    fixed, moving, target = (torch.cat(parts) for parts in zip(*views, strict=True))
    weight = args.cl_weight * (1.0 - it / total) if args.arm == "cst_eq" else 0.0
    with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=images.is_cuda):
        prediction = coupled_convex(net(fixed), net(moving), use_ice=False, img_shape=tuple(n // 2 for n in full))
        loss_reg = voxel_error(prediction.float(), target, full).mean()
        loss_cl = torch.zeros((), device=images.device)
        if weight > 0.0:
            terms = []
            for k in batch:
                for index, affine in zip(pairs[k], random_affine_pair(args.cl_strength, generators["cl"]), strict=True):
                    image = images[index : index + 1].float()
                    brain = image > pads[index] + BRAIN_MARGIN
                    terms.append(
                        equivariance_loss(
                            net,
                            image,
                            brain,
                            affine,
                            args.cl_points,
                            args.cl_temperature,
                            generators["cl"],
                            pads[index],
                        )
                    )
            loss_cl = torch.stack(terms).mean()
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss_reg + weight * loss_cl).backward()
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # GradScaler skips the update and lowers its scale when a gradient is not finite
    skipped = scaler.is_enabled() and scaler.get_scale() < scale
    return float(loss_reg), float(loss_cl), weight, float(skipped or not torch.isfinite(loss_reg).item())


def new_training(args, device):
    torch.manual_seed(1000 + args.seed)  # the student: a fresh net, the same for both arms of one seed
    net = make_feature_net().to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.iters_per_stage, 1, args.lr_min)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    generators = {
        name: torch.Generator().manual_seed(10 * args.seed + k) for k, name in enumerate(("pick", "aug", "cl"))
    }
    return net, optimizer, scheduler, scaler, generators


def time_steps(args, images, pads, pairs, fields, device) -> tuple[float, float]:
    """Median seconds of a training step of the heavier arm, and the peak GiB."""
    probe = argparse.Namespace(**{**vars(args), "arm": "cst_eq", "seed": 0})
    net, optimizer, _, scaler, generators = new_training(probe, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seconds = []
    for step in range(args.timing_steps + 2):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        batch = [step % len(pairs), (step + 1) % len(pairs)][: args.batch]
        train_step(net, batch, pairs, images, pads, fields, probe, 0, 1, generators, optimizer, scaler)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= 2:
            seconds.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    return float(np.median(seconds)), peak


def run_init(args, images, pads, files, test_pairs, device) -> int:
    torch.manual_seed(args.init_seed)
    pairs = make_pairs(len(images), args.pairs_per_subject, args.init_seed)
    teacher = make_feature_net().to(device)
    t0 = time.perf_counter()
    fields, difficulty = rebuild(teacher, images, pads, pairs, 0, args.reg_fac, args.init_seed)
    teacher_sec = (time.perf_counter() - t0) / len(pairs)
    t0 = time.perf_counter()
    start = monitor(teacher, test_pairs, args.reg_fac)
    monitor_sec = time.perf_counter() - t0
    step_sec, peak = time_steps(args, images, pads, pairs, fields, device)
    rebuilds = args.stages - 1
    arm_hours = (
        rebuilds * len(pairs) * teacher_sec + args.stages * (args.iters_per_stage * step_sec + monitor_sec)
    ) / 3600
    report = {
        "pairs": len(pairs),
        "images": len(images),
        "teacher_sec_per_pair": teacher_sec,
        "monitor_sec": monitor_sec,
        "step_sec": step_sec,
        "peak_gib": peak,
        "projected_arm_hours": arm_hours,
        "start_dice": start,
        "start_ok": start["refined"] > start["identity"],
        "args": vars(args),
    }
    out = Path(args.out)
    torch.save({"pairs": pairs, "files": files, "fields": fields.cpu(), "difficulty": difficulty}, out / "init.pth")
    (out / "init_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"[init] {len(pairs)} pairs, teacher {teacher_sec:.2f}s/pair, step {step_sec:.3f}s, peak {peak:.1f} GiB, "
        f"projected {arm_hours:.1f} h per arm; start Dice identity {start['identity']:.4f} solver "
        f"{start['solver']:.4f} refined {start['refined']:.4f}",
        flush=True,
    )
    if not report["start_ok"]:
        print("[init] FAIL: the random-feature teacher does not improve on the identity; check the normalisation")
        return 2
    return 0


def run_arm(args, images, pads, files, test_pairs, device) -> int:
    init = torch.load(Path(args.init) / "init.pth", map_location="cpu", weights_only=False)
    if init["files"] != files:
        raise RuntimeError("the training images differ from the ones the shared start was built on")
    pairs = [tuple(p) for p in init["pairs"]]
    fields, difficulty = init["fields"].to(device), init["difficulty"]
    net, optimizer, scheduler, scaler, generators = new_training(args, device)
    out = Path(args.out)
    total = args.stages * args.iters_per_stage
    history, t_start = [], time.perf_counter()
    for stage in range(1, args.stages + 1):
        weights = sampling_weights(difficulty)
        window, skipped, counted, t_stage = [], 0, 0, time.perf_counter()
        for local in range(args.iters_per_stage):
            it = (stage - 1) * args.iters_per_stage + local
            batch = torch.multinomial(weights, args.batch, replacement=True, generator=generators["pick"]).tolist()
            window.append(
                train_step(net, batch, pairs, images, pads, fields, args, it, total, generators, optimizer, scaler)
            )
            skipped += int(window[-1][3])
            counted += int(window[-1][3] and it >= AMP_WARMUP)
            scheduler.step()
            if (local + 1) % args.log_every == 0:
                reg, cl, weight, skips = (float(np.mean(v)) for v in zip(*window, strict=True))
                print(
                    f"[train] stage {stage} it {it + 1} reg {reg:.3f} cl {cl:.3f} w {weight:.2f} skipped "
                    f"{skips * len(window):.0f}/{len(window)} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"{(time.perf_counter() - t_stage) / (local + 1):.3f}s/it",
                    flush=True,
                )
                window = []
        torch.save(
            {"net": net.state_dict(), "arm": args.arm, "seed": args.seed, "stage": stage}, out / f"stage{stage}.pth"
        )
        entry = {
            "stage": stage,
            "updates": args.iters_per_stage - skipped,
            "skipped_steps": skipped,
            "skipped_after_warmup": counted,
            "train_hours": (time.perf_counter() - t_start) / 3600,
        }
        if counted > args.max_skip_fraction * args.iters_per_stage:
            history.append(entry)
            (out / "history.json").write_text(
                json.dumps({"args": vars(args), "stages": history}, indent=1), encoding="utf-8"
            )
            print(f"[stage] FAIL: {skipped} of {args.iters_per_stage} steps skipped (non-finite loss or gradients)")
            return 4
        if stage < args.stages:
            fields, difficulty = rebuild(net, images, pads, pairs, stage, args.reg_fac, args.seed)
        entry["monitor"] = monitor(net, test_pairs, args.reg_fac)
        net.train()
        entry["hours"] = (time.perf_counter() - t_start) / 3600
        history.append(entry)
        (out / "history.json").write_text(
            json.dumps({"args": vars(args), "stages": history}, indent=1), encoding="utf-8"
        )
        print(f"[stage] {stage}: {entry['monitor']}, {skipped} steps skipped, {entry['hours']:.2f} h", flush=True)
        if stage == 1 and args.max_hours and entry["hours"] * args.stages > args.max_hours:
            print(f"[stage] STOP: projected {entry['hours'] * args.stages:.1f} h exceeds --max_hours {args.max_hours}")
            return 3
    torch.save({"net": net.state_dict(), "arm": args.arm, "seed": args.seed, "stage": args.stages}, out / "final.pth")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--init_only", action="store_true", help="build the shared start (pairs, stage-0 pseudo labels)")
    p.add_argument("--init", help="directory of the shared start, for an arm")
    p.add_argument("--arm", choices=["cst", "cst_eq"], default="cst")
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="arm seed: student initialisation, pair picks, augmentation")
    p.add_argument("--init_seed", type=int, default=0, help="pair list and the random stage-0 teacher")
    p.add_argument("--pairs_per_subject", type=int, default=2)
    p.add_argument("--stages", type=int, default=8)
    p.add_argument("--iters_per_stage", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--augment", type=float, default=0.05, help="affine input augmentation strength")
    p.add_argument("--reg_fac", type=float, default=1.0, help="regularisation of the teacher's Adam refinement")
    p.add_argument("--cl_strength", type=float, default=0.02, help="affine strength of the equivariance views")
    p.add_argument("--cl_points", type=int, default=1000, help="equivariance points per image")
    p.add_argument("--cl_temperature", type=float, default=0.1)
    p.add_argument("--cl_weight", type=float, default=10.0, help="equivariance weight, decaying linearly to 0")
    p.add_argument("--max_hours", type=float, default=0.0, help="stop after stage 1 if the arm would exceed this")
    p.add_argument("--max_skip_fraction", type=float, default=0.1, help="fail a stage that skips more updates")
    p.add_argument("--timing_steps", type=int, default=5)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--limit_files", type=int, default=0, help="first N training images (smoke)")
    p.add_argument("--limit_test", type=int, default=0, help="first N test pairs (smoke)")
    p.add_argument("--crop", type=int, nargs=3, default=None, help="centre crop D H W, multiples of 32 (smoke)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.init_only and not args.init:
        raise SystemExit("an arm needs --init (the shared start); build it with --init_only first")
    device = setup_device(gpu_id=args.gpu, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    images, pads, files = load_training_images(args, device)
    test_pairs = load_test_pairs(args, device)
    print(f">>> {'init' if args.init_only else args.arm} on {len(images)} images {tuple(images.shape[2:])}", flush=True)
    run = run_init if args.init_only else run_arm
    raise SystemExit(run(args, images, pads, files, test_pairs, device))


if __name__ == "__main__":
    main()
