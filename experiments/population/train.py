"""Train a nested A0/A_lin/A1 Gaussian reference on images, without anatomical labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from experiments.population.appearance import appearance_marginal
from experiments.population.model import PopulationAtlas
from experiments.population.observation import add_appearance_args, appearance_prior, load_image
from utils.field import _warp


def observation_nll(model, observed, coefficients, noise_variance, mean, covariance, gain_min):
    _, flow = model.materialize(coefficients)
    template = _warp(model.atlas().expand(observed.shape[0], -1, -1, -1, -1), flow)
    return -appearance_marginal(observed, template, noise_variance, mean, covariance, gain_min).log_prob


def global_penalty(model, log_noise_variance, *, atlas_precision, weight_precision, noise_shape, noise_rate):
    atlas = model.atlas().double()
    roughness = sum(atlas.diff(dim=axis).square().sum() for axis in (2, 3, 4))
    noise = log_noise_variance.double().exp()
    inverse_gamma = torch.distributions.InverseGamma(noise.new_tensor(noise_shape), noise.new_tensor(noise_rate))
    return atlas_precision * roughness + 0.5 * weight_precision * model.weight_penalty() - inverse_gamma.log_prob(noise)


def load_population_checkpoint(path: Path, device: str | torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PopulationAtlas(payload["model"]["raw_atlas"], **payload["model_options"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def _arguments(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else [str(p) for p in value] if key == "subjects" else value
        for key, value in vars(args).items()
    }


def _write_history(path: Path, history: list[dict]) -> None:
    if history:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def train_model(args) -> dict:
    arm = getattr(args, "arm", None)
    alias = getattr(args, "conditional", None)
    if arm is None:
        if alias is None:
            raise ValueError("choose --arm A0, A_lin or A1")
        arm = "A1" if alias else "A0"
    elif alias is not None and bool(alias) != (arm != "A0"):
        raise ValueError("conditional alias disagrees with arm")
    args.arm = arm
    if hasattr(args, "conditional"):
        delattr(args, "conditional")
    if args.epochs < 1 or args.warmup_epochs < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("positive epochs/lr and nonnegative warmup_epochs are required")
    if any(not math.isfinite(v) or v <= 0 for v in (args.weight_precision, args.noise_shape, args.noise_rate)):
        raise ValueError("weight_precision and inverse-Gamma parameters must be finite and positive")
    if not math.isfinite(args.atlas_precision) or args.atlas_precision < 0:
        raise ValueError("atlas_precision must be finite and nonnegative")
    paths = [Path(path).resolve() for path in args.subjects]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("supply a nonempty list of distinct training subjects")
    args.subjects = paths
    arguments = _arguments(args)
    device = torch.device(args.device)
    mean, covariance = appearance_prior(args, device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint_path = args.output / "last.pth"
    history, epoch_start = [], 0
    if args.resume:
        model, payload = load_population_checkpoint(checkpoint_path, device)
        ignored = {"epochs", "output", "device", "resume"}
        if {k: v for k, v in arguments.items() if k not in ignored} != {
            k: v for k, v in payload["arguments"].items() if k not in ignored
        }:
            raise ValueError("resume requires the same subjects, model, appearance and training recipe")
        history, epoch_start = payload["history"], payload["completed_epochs"]
        if torch.device(payload["arguments"]["device"]).type != device.type:
            raise ValueError("exact resume requires the same CPU/CUDA backend")
        if args.epochs < epoch_start:
            raise ValueError("requested total epochs precede the completed checkpoint")
        log_noise_variance = torch.nn.Parameter(payload["log_noise_variance"].to(device))
        model_options = payload["model_options"]
    else:
        if args.output.exists():
            raise FileExistsError(f"use a fresh output directory or --resume: {args.output}")
        template = load_image(paths[0], args.image_scale, "cpu").double()
        for path in paths[1:]:
            image = load_image(path, args.image_scale, "cpu")
            if image.shape != template.shape:
                raise ValueError("training subjects must share the processed voxel grid")
            template.add_(image)
        template = (template / len(paths)).float()
        model_options = {
            name: getattr(args, name)
            for name in (
                "levels",
                "collar",
                "width",
                "arm",
                "scale_min",
                "scale_max",
                "prior_std",
                "posterior_std",
                "integration_steps",
            )
        }
        model = PopulationAtlas(template, **model_options).to(device)
        log_noise_variance = torch.nn.Parameter(torch.tensor(2 * math.log(args.noise_std), device=device))
    optimizer = torch.optim.Adam([*model.parameters(), log_noise_variance], lr=args.lr)
    if args.resume:
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["shuffle_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"], device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(arguments, indent=2) + "\n", encoding="utf-8")
    _write_history(args.output / "metrics.csv", history)
    voxels = model.raw_atlas.numel()
    for epoch in range(epoch_start, args.epochs):
        for parameter in model.prior_parameters():
            parameter.requires_grad_(epoch >= args.warmup_epochs)
        model.train()
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = torch.zeros(4, dtype=torch.float64)
        compute_seconds = 0.0
        for index in torch.randperm(len(paths), generator=generator).tolist():
            observed = load_image(paths[index], args.image_scale, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            state = model.sample(observed)
            template = _warp(model.atlas(), state.flow)
            data = -appearance_marginal(
                observed, template, log_noise_variance.double().exp(), mean, covariance, args.gain_min
            ).log_prob.mean()
            kl = state.kl.mean()
            hyper = global_penalty(
                model,
                log_noise_variance,
                atlas_precision=args.atlas_precision,
                weight_precision=args.weight_precision,
                noise_shape=args.noise_shape,
                noise_rate=args.noise_rate,
            )
            loss = (data + kl + hyper / len(paths)) / voxels
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite objective at epoch {epoch}, subject {paths[index]}")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for group in optimizer.param_groups
                for p in group["params"]
            ):
                raise FloatingPointError(f"nonfinite gradient at epoch {epoch}, subject {paths[index]}")
            optimizer.step()
            model.project_atlas_()
            totals += torch.tensor(
                [
                    float(loss.detach()),
                    float(data.detach() / voxels),
                    float(kl.detach() / voxels),
                    float(hyper.detach() / (len(paths) * voxels)),
                ],
                dtype=torch.float64,
            )
            compute_seconds += time.perf_counter() - compute_started
        row = dict(
            zip(
                ("objective_per_voxel", "nll_per_voxel", "kl_per_voxel", "global_per_subject_voxel"),
                (totals / len(paths)).tolist(),
                strict=True,
            )
        )
        row.update(
            epoch=epoch + 1,
            updates=len(paths),
            seconds_with_io=time.perf_counter() - started,
            seconds_compute=compute_seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
            noise_std=float((0.5 * log_noise_variance.detach()).exp()),
        )
        history.append(row)
        payload = {
            "model": model.state_dict(),
            "model_options": model_options,
            "arguments": arguments,
            "log_noise_variance": log_noise_variance.detach(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": epoch + 1,
            "history": history,
            "shuffle_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        _write_history(args.output / "metrics.csv", history)
        print(json.dumps(row, allow_nan=False), flush=True)
    return {"checkpoint": str(checkpoint_path), "completed_epochs": len(history), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        type=Path,
        nargs="+",
        required=True,
        help="single-subject Train pickles, not paired evaluation data",
    )
    parser.add_argument("--output", type=Path, required=True)
    for name in ("epochs", "warmup-epochs", "levels", "collar", "width", "seed"):
        parser.add_argument(f"--{name}", type=int, required=True)
    for name in ("lr", "scale-min", "scale-max", "atlas-precision", "weight-precision", "noise-shape", "noise-rate"):
        parser.add_argument(f"--{name}", type=float, required=True)
    for name in ("prior-std", "posterior-std"):
        parser.add_argument(
            f"--{name}", type=float, nargs="+", required=True, help="one scale or coarse-to-fine stage scales"
        )
    arms = parser.add_mutually_exclusive_group(required=True)
    arms.add_argument("--arm", choices=["A0", "A_lin", "A1"])
    arms.add_argument("--conditional", type=int, choices=[0, 1], help=argparse.SUPPRESS)
    parser.add_argument("--integration-steps", type=int, default=7)
    parser.add_argument("--tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    add_appearance_args(parser)
    print(json.dumps(train_model(parser.parse_args()) | {"history": "metrics.csv and checkpoint"}))


if __name__ == "__main__":
    main()
