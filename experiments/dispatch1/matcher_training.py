"""The OASIS recipe shared by the im2grid-Lite and MultiMatch entry points: profile, lambda calibration, training with
resumable checkpoints, and evaluation in the common FF/REP protocol (canonical sampling)."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.core.field_scoring import score_ff_rep
from experiments.core.inference_metrics import write_results
from experiments.core.path_profiles import get_dataset_paths
from utils import pkload

WARMUP_UPDATES = 500  # discarded calibration warm-up at lambda 1
CALIBRATION_PAIRS = 8
CALIBRATION_SEED = 2026
AMP_WARMUP = 50  # updates before skipped AMP steps count
LOG_EVERY, CKPT_EVERY = 50, 1000


def parameter_gradient_norm(loss: torch.Tensor, parameters, retain_graph=True) -> float:
    """Gradient norm over the parameters; parameters the loss does not reach count as zero."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))
    if not math.isfinite(norm):
        raise FloatingPointError("non-finite parameter gradient during lambda calibration")
    return norm


class TrainingPairs(Dataset):
    """Pairs and their common flips derive from the index, so a resumed run continues the same stream."""

    def __init__(self, files, seed, start, length, crop=None):
        if len(files) < 2:
            raise RuntimeError("matcher training requires at least two OASIS subjects")
        self.files, self.seed, self.start, self.length, self.crop = files, seed, start, length, crop

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(index)
        rng = random.Random(self.seed * 1_000_003 + self.start + index)
        pair = rng.sample(self.files, 2)
        images = [torch.from_numpy(np.ascontiguousarray(pkload(path)[0], dtype=np.float32))[None] for path in pair]
        if self.crop:
            shape = images[0].shape[1:]
            box = (slice(None), *(slice((n - c) // 2, (n + c) // 2) for n, c in zip(shape, self.crop, strict=True)))
            images = [image[box] for image in images]
        for axis in (1, 2, 3):
            if rng.random() < 0.5:
                images = [image.flip(axis) for image in images]
        return tuple(images)


def training_files(args):
    root = Path(get_dataset_paths(args.paths, "OASIS")["train_dir"])
    return sorted(root.glob("*.pkl"))[: args.limit_files or None]


def append_metrics(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_metrics(path: Path, step: int) -> None:
    """Drop rows past the resumed checkpoint."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, kept = reader.fieldnames, [row for row in reader if int(row["step"]) <= step]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept)


def train(args, device, lam, steps, build_model, losses, method, directory=None):
    """``steps`` successful updates (AMP-skipped steps do not count) from the seed's start; with ``directory`` it
    checkpoints every CKPT_EVERY updates and resumes from its last.pth."""
    torch.manual_seed(args.seed)
    model = build_model(args).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    files = training_files(args)
    configuration = {
        "seed": args.seed,
        "lambda": lam,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "crop": args.crop,
        "files": [p.name for p in files],
    }
    counts = {"step": 0, "attempts": 0, "skipped": 0, "skipped_after_warmup": 0}
    path = directory / "last.pth" if directory else None
    metrics = directory.parent / "metrics.csv" if directory else None
    if path and path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        if state["configuration"] != configuration or state["method"] != method:
            raise RuntimeError("checkpoint method or training configuration differs from this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        counts = {key: state[key] for key in counts}
        if metrics.exists():
            truncate_metrics(metrics, counts["step"])
        print(f"[matcher] resumed at update {counts['step']}, attempt {counts['attempts']}", flush=True)
    start_counts = dict(counts)
    stream = TrainingPairs(files, args.seed, counts["attempts"], max(steps - counts["step"], 0) * 2 + 1000, args.crop)
    loader = DataLoader(stream, batch_size=1, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    window, consecutive = [], 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for moving, fixed in loader:
        if counts["step"] >= steps:
            break
        moving, fixed = moving.to(device, non_blocking=True), fixed.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            similarity, regularity = losses(model, moving, fixed)
            loss = similarity + lam * regularity
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite matcher objective at attempt {counts['attempts'] + 1}")
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        counts["attempts"] += 1
        dropped = scaler.is_enabled() and scaler.get_scale() < scale
        counts["skipped"] += int(dropped)
        counts["skipped_after_warmup"] += int(dropped and counts["step"] >= AMP_WARMUP)
        consecutive = consecutive + 1 if dropped else 0
        if consecutive >= 100:
            raise FloatingPointError("100 consecutive AMP overflows: training cannot make progress")
        if dropped:
            continue
        counts["step"] += 1
        window.append((float(similarity.detach()), float(regularity.detach())))
        if counts["step"] % LOG_EVERY == 0 or counts["step"] == steps:
            sim, reg = (statistics.mean(values) for values in zip(*window, strict=True))
            row = {**counts, "similarity": sim, "reg": reg}
            print(f"[matcher] {row}", flush=True)
            if metrics:
                append_metrics(metrics, row)
            window.clear()
        if path and (counts["step"] % CKPT_EVERY == 0 or counts["step"] == steps):
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "lambda": lam,
                "configuration": configuration,
                "method": method,
                **counts,
            }
            torch.save(payload, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").replace(path)
    if counts["step"] != steps:
        raise RuntimeError(f"matcher data stream exhausted after {counts['step']} of {steps} updates")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    done = {key: counts[key] - start_counts[key] for key in ("step", "attempts")}
    return model, {
        "seconds": time.perf_counter() - start,
        "updates": done["step"],
        "attempts": done["attempts"],
        "skipped": counts["skipped"],
        "skipped_after_warmup": counts["skipped_after_warmup"],
        **peak_memory(device),
    }


def peak_memory(device) -> dict:
    """Peak allocated and reserved GiB since the last reset; reserved is what CTCF logs as perf_peak_gpu_mem_gib."""
    if device.type != "cuda":
        return {"peak_allocated_gib": None, "peak_reserved_gib": None}
    allocated, reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
    return {"peak_allocated_gib": allocated / 2**30, "peak_reserved_gib": reserved / 2**30}


def calibrate(args, device, build_model, losses, method) -> dict:
    """lambda* = median over fixed training pairs of |grad similarity| / |grad regularity| over the parameters, after a
    discarded warm-up at lambda 1 whose mean update time is reported as step_sec and whose peak memory is in warmup."""
    model, warmup = train(args, device, 1.0, WARMUP_UPDATES, build_model, losses, method)
    pairs = TrainingPairs(
        training_files(args), seed=CALIBRATION_SEED, start=0, length=CALIBRATION_PAIRS, crop=args.crop
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    ratios = []
    for moving, fixed in pairs:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            similarity, regularity = losses(model, moving[None].to(device), fixed[None].to(device))
        sim = parameter_gradient_norm(similarity, parameters)
        reg = parameter_gradient_norm(regularity, parameters, retain_graph=False)
        if sim <= 0 or reg <= 0:
            raise RuntimeError("calibration requires nonzero parameter gradients of both losses")
        ratios.append(sim / reg)
    step_sec = warmup["seconds"] / warmup["updates"]
    return {
        "lambda": statistics.median(ratios),
        "ratios": ratios,
        "warmup": warmup,
        "step_sec": step_sec,
        "method": method,
    }


@torch.no_grad()
def evaluate(args, device, build_model, predict, method) -> None:
    """FF and REP rows of the test pairs, summary.json per mode, and summary_arm.json with the model diagnostics."""
    state = torch.load(args.eval_ckpt, map_location=device, weights_only=False)
    if state["method"] != method:
        raise RuntimeError("checkpoint belongs to a different matcher")
    model = build_model(args).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    test_dir = get_dataset_paths(args.paths, "OASIS")["val_dir"]
    files = sorted(Path(test_dir).glob("*.pkl"))[: args.limit_test or None]
    if not files:
        raise RuntimeError(f"no OASIS test pairs in {test_dir}")
    rows, diagnostics = {"FF": [], "REP": []}, []
    for path in files:
        moving, fixed, moving_seg, fixed_seg = (
            torch.from_numpy(np.ascontiguousarray(a)).to(device)[None, None] for a in pkload(path)
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            flow = predict(model, moving.float(), fixed.float()).float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        case_id = path.stem.removeprefix("p_")
        ff, rep = score_ff_rep(flow, case_id, time.perf_counter() - start, moving_seg, fixed_seg, "canonical")
        rows["FF"].append(ff)
        rows["REP"].append(rep)
        diagnostics.append({"case_id": case_id, **getattr(model, "diagnostics", {})})
        print(f"[matcher] {path.stem}: FF {ff['dice_mean']:.5f}, REP {rep['dice_mean']:.5f}", flush=True)
    out = Path(args.out)
    for mode, scored in rows.items():
        (out / mode).mkdir(parents=True, exist_ok=True)
        write_results(scored, str(out / mode), str(method["name"]), args.eval_ckpt, test_dir)
    summary = {
        "FF": float(np.mean([r["dice_mean"] for r in rows["FF"]])),
        "REP": float(np.mean([r["dice_mean"] for r in rows["REP"]])),
        "sdlogj_FF": float(np.mean([r["sdlogj"] for r in rows["FF"]])),
        "certified_REP_cases": sum(bool(r["tri_proj_certified"]) for r in rows["REP"]),
        "sampling": "canonical",
        "training": {key: state[key] for key in ("lambda", "step", "attempts", "skipped", "skipped_after_warmup")},
        "method": method,
        "diagnostics": diagnostics,
    }
    (out / "summary_arm.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics = Path(args.eval_ckpt).parent.parent / "metrics.csv"
    if metrics.exists():
        shutil.copy2(metrics, out / "metrics.csv")


def add_arguments(parser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile", default="", help="JSON with the seconds per update of --profile_steps updates")
    mode.add_argument("--calibrate", default="", help="JSON with lambda*")
    mode.add_argument("--eval_ckpt", default="", help="checkpoint to evaluate into --out")
    parser.add_argument("--profile_steps", type=int, default=100)
    parser.add_argument("--lambda_file", default="")
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=39_400)
    parser.add_argument("--exp", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-9)
    parser.add_argument("--crop", type=int, nargs=3)
    parser.add_argument("--limit_files", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)


def run(args, device, build_model, losses, predict, method) -> None:
    if args.iters <= 0 or args.profile_steps <= 0:
        raise ValueError("update counts must be positive")
    if args.eval_ckpt:
        if not args.out:
            raise ValueError("--eval_ckpt needs --out")
        evaluate(args, device, build_model, predict, method)
        return
    if args.profile:
        _, result = train(args, device, 1.0, args.profile_steps, build_model, losses, method)
        result.update(step_sec=result["seconds"] / result["updates"], method=method)
        output = Path(args.profile)
    elif args.calibrate:
        result, output = calibrate(args, device, build_model, losses, method), Path(args.calibrate)
    else:
        if not args.exp or not args.lambda_file:
            raise ValueError("training requires --exp and --lambda_file")
        calibration = json.loads(Path(args.lambda_file).read_text(encoding="utf-8"))
        if calibration["method"] != method:
            raise ValueError("lambda calibration belongs to a different matcher")
        lam = calibration["lambda"] * args.lambda_scale
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError("lambda must be finite and positive")
        directory = Path("results") / args.exp / "ckpt"
        directory.mkdir(parents=True, exist_ok=True)
        _, result = train(args, device, lam, args.iters, build_model, losses, method, directory)
        print(f"[matcher] completed: {result}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[matcher] {output}: {result}", flush=True)
