"""Subject MAP, canonical pair scoring, and heldout importance diagnostics of a population atlas."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.core.field_scoring import REPAIR, repair, score
from experiments.core.inference_metrics import write_results
from experiments.population.appearance import appearance_marginal
from experiments.population.geometry import (
    canonical_flow,
    coefficient_bands,
    haar_forward,
    haar_inverse,
    pad_velocity,
    subject_pair_flow,
)
from experiments.population.observation import appearance_prior, load_image
from experiments.population.train import load_population_checkpoint, observation_nll
from utils import pkload
from utils.cert_exact import certify_flow_exact
from utils.field import _warp, compose_flows
from utils.field_io import load_flow_npz, save_flow_npz_atomic


def velocity_energy(model, observed, interior, variance, mean, covariance, gain_min):
    """The unchanged data/prior energy in native voxel-unit velocity coordinates."""
    flow = canonical_flow(pad_velocity(interior, model.collar), model.integration_steps)
    template = _warp(model.atlas().expand(observed.shape[0], -1, -1, -1, -1), flow)
    data = -appearance_marginal(observed, template, variance, mean, covariance, gain_min).log_prob.sum()
    prior = -model.log_prior(haar_forward(interior, model.levels)).sum()
    return data, prior


def velocity_rms_by_scale(change: torch.Tensor, levels: int) -> list[float]:
    """Orthogonal scale contributions to whole-interior velocity RMS, coarse to fine."""
    bands = coefficient_bands(haar_forward(change.detach(), levels))
    groups = (bands[:1], *(bands[start : start + 7] for start in range(1, len(bands), 7)))
    return [float((sum(band.double().square().sum() for band in group) / change.numel()).sqrt()) for group in groups]


def fit_subject(model, observed, variance, mean, covariance, gain_min, *, steps: int, lr: float, log_every: int = 10):
    """Optimize full interior velocity at fixed globals, retaining the best visited MAP state."""
    if steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("MAP steps must be nonnegative and lr finite and positive")
    if log_every < 1:
        raise ValueError("MAP log_every must be positive")
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False)
    try:
        with torch.no_grad():
            initial = haar_inverse(model.posterior_mean(observed)).detach()
        interior = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([interior], lr=lr)
        best, best_loss, best_info, trace = None, float("inf"), {}, []
        previous_gradient_rms = previous_update_rms = None
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            data, prior = velocity_energy(model, observed, interior, variance, mean, covariance, gain_min)
            objective = (data + prior) / observed.numel()
            value = float(objective.detach())
            if not math.isfinite(value):
                raise FloatingPointError(f"nonfinite subject MAP objective at step {step}")
            if value < best_loss:
                best_loss, best = value, interior.detach().clone()
                best_info = {
                    "best_step": step,
                    "best_nll": float(data.detach()),
                    "best_prior_nll": float(prior.detach()),
                }
            if step % log_every == 0 or step == steps:
                trace.append(
                    {
                        "step": step,
                        "nll": float(data.detach()),
                        "prior_nll": float(prior.detach()),
                        "objective_per_voxel": value,
                        "velocity_change_rms_by_scale": velocity_rms_by_scale(interior - initial, model.levels),
                        "previous_gradient_rms": previous_gradient_rms,
                        "previous_update_rms": previous_update_rms,
                    }
                )
            if step == steps:
                break
            objective.backward()
            if interior.grad is None or not torch.isfinite(interior.grad).all():
                raise FloatingPointError(f"nonfinite subject MAP gradient at step {step}")
            measure_update = (step + 1) % log_every == 0 or step + 1 == steps
            before = interior.detach().clone() if measure_update else None
            previous_gradient_rms = float(interior.grad.double().square().mean().sqrt()) if measure_update else None
            optimizer.step()
            previous_update_rms = (
                float((interior.detach() - before).double().square().mean().sqrt()) if measure_update else None
            )
        with torch.no_grad():
            velocity = pad_velocity(best, model.collar)
            flow = canonical_flow(velocity, model.integration_steps)
            inverse = canonical_flow(-velocity, model.integration_steps)
            round_trip = compose_flows(flow, inverse)
        return (
            velocity,
            flow,
            {
                "trace": trace,
                "inference_mode": "encoder_only" if steps == 0 else "velocity_map",
                **best_info,
                "best_objective_per_voxel": best_loss,
                "best_velocity_change_rms_by_scale": velocity_rms_by_scale(best - initial, model.levels),
                "inverse_round_trip_rms": float(round_trip.square().mean().sqrt()),
            },
        )
    finally:
        for parameter, flag in zip(model.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)


@torch.no_grad()
def importance_diagnostic(model, observed, variance, mean, covariance, gain_min, samples: int) -> dict:
    """Finite-sample importance estimate; neither exact evidence nor an automatic selection verdict."""
    if samples < 1:
        raise ValueError("importance samples must be positive")
    weights = []
    for _ in range(samples):
        state = model.sample(observed)
        data = observation_nll(model, observed, state.coefficients, variance, mean, covariance, gain_min)
        weights.append(-data + state.log_prior - state.log_q)
    weights = torch.stack(weights)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("nonfinite importance weights")
    log_estimate = weights.logsumexp(dim=0) - math.log(samples)
    ess = 1 / weights.softmax(dim=0).square().sum(dim=0)
    return {
        "log_importance_estimate": log_estimate.cpu().tolist(),
        "effective_sample_size": ess.cpu().tolist(),
        "samples": samples,
        "scope": "finite-sample importance diagnostic; log-estimator bias and proposal failure remain possible; no model-selection verdict",
    }


def evaluate_model(args) -> None:
    if args.output.exists():
        raise FileExistsError(f"use a fresh output directory: {args.output}")
    if args.certify and not args.repair:
        raise ValueError("--certify requires --repair; both raw and repaired bytes will be checked")
    model, payload = load_population_checkpoint(args.checkpoint, args.device)
    model.eval()
    trained = argparse.Namespace(**payload["arguments"])
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(trained.tf32)
    torch.backends.cudnn.allow_tf32 = bool(trained.tf32)
    mean, covariance = appearance_prior(trained, args.device)
    variance = payload["log_noise_variance"].to(args.device).double().exp()
    args.output.mkdir(parents=True)
    configuration = {
        key: str(value)
        if isinstance(value, Path)
        else [str(path) for path in value]
        if isinstance(value, list)
        else value
        for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    if args.subjects:
        training = {Path(path).resolve() for path in trained.subjects}
        if training.intersection(path.resolve() for path in args.subjects):
            raise ValueError("heldout evidence diagnostic must not reuse training subject files")
        rows = []
        for path in args.subjects:
            observed = load_image(path, trained.image_scale, args.device)
            rows.append(
                {
                    "subject": str(path),
                    **importance_diagnostic(
                        model, observed, variance, mean, covariance, trained.gain_min, args.importance_samples
                    ),
                }
            )
        (args.output / "importance.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return
    all_rows = {"FF": [], "REP": []}
    for path in args.pairs:
        payload_pair = pkload(path)
        if not isinstance(payload_pair, (tuple, list)) or len(payload_pair) != 4:
            raise ValueError(f"{path}: expected an OASIS (moving, fixed, moving_seg, fixed_seg) pair")
        arrays = [np.asarray(array) for array in payload_pair]
        if any(array.ndim != 3 or array.shape != arrays[0].shape or not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"{path}: expected matching finite 3D pair arrays")
        if any(
            not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() > 35 for array in arrays[2:]
        ):
            raise ValueError(f"{path}: this evaluator expects OASIS dense labels 0..35")
        x, y = [
            torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None].to(args.device)
            / trained.image_scale
            for array in arrays[:2]
        ]
        x_seg, y_seg = [
            torch.from_numpy(np.ascontiguousarray(array.astype(np.int64)))[None, None].to(args.device)
            for array in arrays[2:]
        ]
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        moving_velocity, _, moving_info = fit_subject(
            model, x, variance, mean, covariance, trained.gain_min, steps=args.map_steps, lr=args.map_lr
        )
        fixed_velocity, _, fixed_info = fit_subject(
            model, y, variance, mean, covariance, trained.gain_min, steps=args.map_steps, lr=args.map_lr
        )
        with torch.no_grad():
            raw = subject_pair_flow(fixed_velocity, moving_velocity, model.integration_steps).float()
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - started
        case_dir = args.output / path.stem
        case_dir.mkdir()
        report = {
            "case": str(path),
            "moving": moving_info,
            "fixed": fixed_info,
            "certification": "NOT_RUN",
            "convention": "fixed_to_moving, zyx voxel displacement, align_corners=True",
            "timing": "two subject MAP fits plus pair composition; excludes IO and scoring",
            "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2**30
            if torch.device(args.device).type == "cuda"
            else None,
        }
        outputs = {"FF": (raw, seconds)}
        if args.repair:
            mask = torch.ones_like(raw[:, :1], dtype=torch.bool)
            mask[:, :, 1:-1, 1:-1, 1:-1] = False
            if torch.count_nonzero(raw.masked_select(mask.expand_as(raw))):
                raise ValueError("canonical pair has nonzero boundary; refusing a zero-boundary repair claim")
            repair_started = time.perf_counter()
            repaired, repair_info = repair(raw, fixed_mask=mask, fixed_values=torch.zeros_like(raw))
            report["repair"] = repair_info
            outputs["REP"] = (repaired, seconds + time.perf_counter() - repair_started)
        for mode, (flow, elapsed) in outputs.items():
            field_path = case_dir / f"{mode}.npz"
            save_flow_npz_atomic(field_path, flow)
            all_rows[mode].append(
                score(flow, path.stem, elapsed, x_seg, y_seg, REPAIR["tri_eps"] if mode == "REP" else 0.0, "canonical")
            )
            if args.certify:
                certificate = certify_flow_exact(load_flow_npz(field_path), eps=str(REPAIR["tri_eps"]))
                certificate["global_homeomorphism_hypotheses_met"] = (
                    certificate["certified"] and certificate["boundary_nonzero_count"] == 0
                )
                report[f"certificate_{mode}"] = certificate
        report["certification"] = "SEE_PER_FIELD_REPORTS" if args.certify else "NOT_RUN"
        (case_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"case": path.stem, "raw_dice": all_rows["FF"][-1]["dice_mean"]}), flush=True)
    for mode, rows in all_rows.items():
        if rows:
            folder = args.output / mode
            folder.mkdir()
            write_results(
                rows,
                str(folder),
                f"population_{model.arm}",
                str(args.checkpoint),
                "explicit paired OASIS files",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pairs", type=Path, nargs="+")
    inputs.add_argument(
        "--subjects", type=Path, nargs="+", help="heldout single-subject evidence diagnostics, without labels"
    )
    parser.add_argument("--map-steps", type=int, required=True, help="0 explicitly selects encoder-only inference")
    parser.add_argument("--map-lr", type=float, required=True, help="Adam step in native velocity voxel units")
    parser.add_argument("--importance-samples", type=int, default=1)
    parser.add_argument("--repair", type=int, choices=[0, 1], required=True)
    parser.add_argument("--certify", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    evaluate_model(parser.parse_args())


if __name__ == "__main__":
    main()
