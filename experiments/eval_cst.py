"""Evaluate a trained CST checkpoint on the 19 OASIS test pairs in our protocol, and calibrate its test-time refinement.

  --control_out FILE  the P1 control's SDlogJ on 20 fixed training pairs (our operator, on its own stored flows); the
                      pairs are drawn once and written with it
  --calibrate FILE    the refinement weight (reg_fac) at which the CST refined fields of the same 20 training pairs reach
                      the control's mean SDlogJ, by log bisection; no test pair is used
  default             fields of the 19 test pairs in --mode solver|refined, scored as experiments.inference scores them:
                      feed-forward (<out>/<mode>__FF) and after its repair chain, digital projection eps 0 then
                      trilinear projection eps 0.001 (<out>/<mode>__REP); per_case.csv and summary files

    python -m experiments.eval_cst --control_out RUN/control_sdlogj.json --paths 3 --gpu 0
    python -m experiments.eval_cst --ckpt RUN/cst_s0/final.pth --calibrate RUN/cst_s0/reg_fac.json \
        --control RUN/control_sdlogj.json --paths 3 --gpu 0
    python -m experiments.eval_cst --ckpt RUN/cst_s0/final.pth --mode refined --reg_fac_file RUN/cst_s0/reg_fac.json \
        --out RUN/cst_s0/eval --paths 3 --gpu 0
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.core.inference_metrics import metric_profile_for, write_results
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from experiments.core.path_profiles import get_dataset_paths
from experiments.train_cst import normalise, volume
from models.CST.adaptation import refine, solver_offsets, to_canonical
from models.CST.core import make_feature_net
from utils import logdet_std_from_flow, pkload, setup_device
from utils.dice import dice_per_label
from utils.field import (
    _warp,
    boundary_max_disp,
    boundary_nonzero_count,
    boundary_tangential_lip,
    digital_min_det,
    digital_project,
    displacement_grad_norm_max,
    trilinear_cert_bound,
    trilinear_fold_percent,
    trilinear_min_det,
    trilinear_project,
)

CONTROL_CONFIG = "CTCF-CascadeA-VM-Unified"  # the P1 control: VXM-100 from scratch, seed 0
CALIBRATION_SEED = 2026
PROJECTION_SEED = 800_001  # random projections of the refinement, fixed per pair
CALIBRATION_TOLERANCE = 0.03  # relative SDlogJ gap at which the refined fields count as geometry-matched
REPAIR = {"project_eps": 0.0, "tri_eps": 0.001, "damp": 0.6, "iters": 80}  # experiments.inference REP settings
PROFILE = metric_profile_for("OASIS")


def training_files(args) -> list[str]:
    files = sorted(glob.glob(os.path.join(get_dataset_paths(args.paths, "OASIS")["train_dir"], "*.pkl")))
    return files[: args.limit_files or None]


def calibration_pairs(n: int, count: int) -> list[tuple[int, int]]:
    generator = torch.Generator().manual_seed(CALIBRATION_SEED)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < count:
        i, j = torch.randperm(n, generator=generator)[:2].tolist()
        if (i, j) not in pairs:
            pairs.append((i, j))
    return pairs


def load_net(path: str, device: torch.device):
    net = make_feature_net()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["net"], strict=True)
    return net.to(device).eval()


def run_control(args, device) -> None:
    files = training_files(args)
    pairs = calibration_pairs(len(files), args.calibration_pairs)
    adapter = get_model_adapter("ctcf")
    model = adapter.build(time_steps=12, config_key=CONTROL_CONFIG, l3_svf=True, synth_img_size=args.crop)
    if not args.smoke:
        load_checkpoint_state(model, args.control_ckpt, strict=True)
    model = model.to(device).eval()
    values = []
    for i, j in pairs:
        fixed, moving = (volume(pkload(files[k])[0], args.crop).float().to(device) for k in (i, j))
        with torch.no_grad():
            flow = adapter.forward(model, moving, fixed)
        values.append(float(logdet_std_from_flow(flow)))
    result = {
        "ckpt": None if args.smoke else args.control_ckpt,
        "pairs": [[Path(files[i]).name, Path(files[j]).name] for i, j in pairs],
        "sdlogj": values,
        "mean": float(np.mean(values)),
    }
    Path(args.control_out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[control] mean SDlogJ {result['mean']:.4f} over {len(pairs)} training pairs", flush=True)


def calibrate_weight(mean_sdlogj, target: float, reg_min: float, reg_max: float, steps: int) -> dict:
    """Log bisection of the refinement weight for a mean SDlogJ that falls as the weight grows. The weight counts as
    geometry-matched only if the SDlogJ re-measured at it lies within CALIBRATION_TOLERANCE of the target; a target
    outside the reachable range is reported as such, never as a match."""
    lo, hi = math.log10(reg_min), math.log10(reg_max)
    if mean_sdlogj(lo) <= target:
        chosen, bound = lo, "target_above_range"  # even the weakest regularisation is smoother than the target
    elif mean_sdlogj(hi) >= target:
        chosen, bound = hi, "target_below_range"  # even the strongest regularisation is rougher than the target
    else:
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mean_sdlogj(mid) > target else (lo, mid)
        chosen, bound = 0.5 * (lo + hi), None
    achieved = mean_sdlogj(chosen)
    reached = abs(achieved - target) <= CALIBRATION_TOLERANCE * target
    return {
        "reg_fac": 10.0**chosen,
        "target": target,
        "achieved": achieved,
        "status": "reached" if reached else bound or "not_converged",
        "tolerance": CALIBRATION_TOLERANCE,
    }


def run_calibration(args, net, device) -> None:
    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    train_dir = get_dataset_paths(args.paths, "OASIS")["train_dir"]
    cases = []
    for k, names in enumerate(control["pairs"]):
        (fixed, _), (moving, pad) = (
            normalise(volume(pkload(os.path.join(train_dir, name))[0], args.crop).float()) for name in names
        )
        fixed, moving = fixed.to(device), moving.to(device)
        cases.append((k, fixed, moving, solver_offsets(net, fixed, moving, pad=pad)))
    trace = []

    def mean_sdlogj(log_reg):
        reg_fac = 10.0**log_reg
        value = float(
            np.mean(
                [
                    float(
                        logdet_std_from_flow(
                            to_canonical(refine(net, f, m, s, reg_fac=reg_fac, proj_seed=PROJECTION_SEED + k))
                        )
                    )
                    for k, f, m, s in cases
                ]
            )
        )
        trace.append({"reg_fac": reg_fac, "sdlogj": value})
        print(f"[calibrate] reg_fac {reg_fac:.4g}: SDlogJ {value:.4f} (target {control['mean']:.4f})", flush=True)
        return value

    result = calibrate_weight(mean_sdlogj, control["mean"], args.reg_min, args.reg_max, args.bisection_steps)
    solver = float(np.mean([float(logdet_std_from_flow(to_canonical(s))) for _, _, _, s in cases]))
    result.update(solver_sdlogj=solver, trace=trace)
    Path(args.calibrate).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        f"[calibrate] {result['status']}: reg_fac {result['reg_fac']:.4g}, SDlogJ {result['achieved']:.4f} for target "
        f"{result['target']:.4f} (solver SDlogJ {solver:.4f})",
        flush=True,
    )


def score(flow, case_id, seconds, x_seg, y_seg, eps) -> dict:
    """The per-case columns of experiments.inference on a canonical flow."""
    with torch.no_grad():
        warped = _warp(x_seg.float(), flow, mode="nearest").long()
        dice = dice_per_label(warped, y_seg.long(), labels=PROFILE.labels)
    tri_bound = trilinear_cert_bound(flow, 0, eps)
    tri_fold = trilinear_fold_percent(flow)
    nonzero = boundary_nonzero_count(flow)
    row = {
        "case_id": case_id,
        "time_sec": seconds,
        "cert_min_det": digital_min_det(flow),
        "tri_min_det": trilinear_min_det(flow),
        "tri_cert_bound": tri_bound,
        "bernstein_pass_float64": float(math.isfinite(tri_bound) and tri_bound >= eps),
        "tri_fold_pct": tri_fold,
        "tri_case_folds": float(tri_fold > 0.0),
        "disp_grad_norm": displacement_grad_norm_max(flow),
        "boundary_tan_lip": boundary_tangential_lip(flow),
        "boundary_max_disp": boundary_max_disp(flow),
        "boundary_nonzero_count": float(nonzero),
        "identity_boundary_exact": float(nonzero == 0),
        "dice_mean": float(np.mean(dice)),
        **PROFILE.jac_metrics(flow, x_seg),
    }
    row.update({f"dice_lbl_{label}": float(v) for label, v in zip(PROFILE.labels, dice, strict=True)})
    return row


def run_evaluation(args, net, device) -> None:
    reg_fac, calibration = None, None
    if args.mode == "refined":
        calibration = json.loads(Path(args.reg_fac_file).read_text(encoding="utf-8"))
        reg_fac = calibration["reg_fac"]
    test_dir = get_dataset_paths(args.paths, "OASIS")["val_dir"]
    files = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))[: args.limit_test or None]
    rows = {"FF": [], "REP": []}
    for k, path in enumerate(files):
        x, y, x_seg, y_seg = pkload(path)
        moving, pad = normalise(volume(x, args.crop).float().to(device))
        fixed, _ = normalise(volume(y, args.crop).float().to(device))
        x_seg, y_seg = (volume(s, args.crop).to(device) for s in (x_seg, y_seg))
        case_id = Path(path).stem.removeprefix("p_")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        offsets = solver_offsets(net, fixed, moving, pad=pad)
        if reg_fac is not None:
            offsets = refine(net, fixed, moving, offsets, reg_fac=reg_fac, proj_seed=PROJECTION_SEED + k)
        flow = to_canonical(offsets).float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward = time.perf_counter() - t0
        rows["FF"].append(score(flow, case_id, forward, x_seg, y_seg, 0.0))
        t0 = time.perf_counter()
        repaired, folds, iters = digital_project(
            flow, eps=REPAIR["project_eps"], damp=REPAIR["damp"], max_iters=REPAIR["iters"]
        )
        repaired, report = trilinear_project(
            repaired, eps=REPAIR["tri_eps"], damp=REPAIR["damp"], max_iters=REPAIR["iters"], subdiv_depth=0
        )
        row = score(repaired, case_id, forward + time.perf_counter() - t0, x_seg, y_seg, REPAIR["tri_eps"])
        row.update(
            {
                "proj_folds_end": folds,
                "proj_iters": float(iters),
                "tri_proj_certified": float(report.certified),
                "tri_proj_uncertified_cells": float(report.n_uncertified_cells),
                "tri_proj_sampled_negative_pct": report.sampled_negative_cell_percent,
                "tri_proj_iters": float(report.iterations),
            }
        )
        rows["REP"].append(row)
        print(
            f"[{k + 1}/{len(files)}] {case_id} dice FF {rows['FF'][-1]['dice_mean']:.4f} REP {row['dice_mean']:.4f} "
            f"sdlogj {rows['FF'][-1]['sdlogj']:.4f} {forward:.1f}s",
            flush=True,
        )
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    for scoring, scored in rows.items():
        out = Path(args.out) / f"{args.mode}__{scoring}"
        out.mkdir(parents=True, exist_ok=True)
        write_results(scored, str(out), f"cst_{args.mode}", args.ckpt, test_dir)
    (Path(args.out) / f"{args.mode}__cost.json").write_text(
        json.dumps(
            {
                "reg_fac": reg_fac,
                "calibration_status": calibration["status"] if reg_fac is not None else None,
                "peak_gib": peak,
                "pairs": len(files),
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", help="CST checkpoint (stage<k>.pth or final.pth from experiments.train_cst)")
    p.add_argument("--mode", choices=["solver", "refined"], default="solver")
    p.add_argument("--reg_fac_file", help="calibration result for --mode refined")
    p.add_argument("--out", help="evaluation directory")
    p.add_argument("--control_out", help="write the control's SDlogJ on the calibration pairs here")
    p.add_argument("--control_ckpt", default="results/PILOT_P1_CONTROL/ckpt/last.pth")
    p.add_argument("--calibrate", help="write the calibrated refinement weight here")
    p.add_argument("--control", help="the control's SDlogJ file, for --calibrate")
    p.add_argument("--calibration_pairs", type=int, default=20)
    p.add_argument("--reg_min", type=float, default=0.01)
    p.add_argument("--reg_max", type=float, default=1000.0)
    p.add_argument("--bisection_steps", type=int, default=10)
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit_files", type=int, default=0, help="first N training images (smoke)")
    p.add_argument("--limit_test", type=int, default=0, help="first N test pairs (smoke)")
    p.add_argument("--crop", type=int, nargs=3, default=None, help="centre crop D H W, multiples of 32 (smoke)")
    p.add_argument("--smoke", action="store_true", help="random-init control model (local smoke runs)")
    args = p.parse_args()
    device = setup_device(gpu_id=args.gpu, seed=0)
    if args.crop:
        args.crop = tuple(args.crop)
    if args.control_out:
        run_control(args, device)
        return
    net = load_net(args.ckpt, device)
    if args.calibrate:
        run_calibration(args, net, device)
    else:
        run_evaluation(args, net, device)


if __name__ == "__main__":
    main()
