"""One preliminary dispatch: full-volume cost/fit checks, label-fit readout and descriptive residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.population.appearance import appearance_marginal
from experiments.population.evaluate import fit_subject, importance_diagnostic
from experiments.population.geometry import haar_forward
from experiments.population.observation import appearance_prior, load_image, residual_summary
from experiments.population.predictability import evaluate_predictability
from experiments.population.probe import run_probe
from experiments.population.train import load_population_checkpoint, train_model
from utils.field import _warp, legacy_sampling_displacement

ARMS = {"a0": "A0", "linear": "A_lin", "a1": "A1"}
PROBES = {"probe_ref": 0, "probe_train0": 1, "probe_train1": 2, "probe_train2": 3, "probe_val0": 4, "probe_val1": 5}
SPLIT = Path("datasets/splits/oasis_subject_disjoint_v2/split_manifest.json")
SETTINGS = {"PROFILE_WIDTH": 8, "PROFILE_LEVELS": 3, "PROFILE_EPOCHS": 2, "PROBE_STEPS": 40, "MAP_STEPS": 40}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_training_options(template, levels, width, epochs):
    """Explicit Gaussian-reference recipe shared by the profile and calibration dispatch."""
    gain = float(template.double().std(correction=0))
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Train-mean template has no finite contrast")
    scales = [1.0 / 2**stage * 2 ** (1.5 * depth) for stage, depth in enumerate((levels, *range(levels, 0, -1)))]
    return dict(
        epochs=epochs,
        warmup_epochs=1,
        levels=levels,
        collar=2,
        width=width,
        seed=0,
        lr=0.001,
        scale_min=min(scales) * 0.02,
        scale_max=max(scales) * 10,
        prior_std=scales,
        posterior_std=[scale / 4 for scale in scales],
        integration_steps=7,
        atlas_precision=0.01,
        weight_precision=1e-4,
        noise_shape=2.0,
        noise_rate=3 * (gain / 4) ** 2,
        image_scale=1.0,
        noise_std=gain / 4,
        gain_mean=gain,
        gain_std=gain / 2,
        offset_mean=float(template.double().mean()),
        offset_std=gain / 2,
        bias_std=gain / 10,
        gain_min=gain / 100,
        tf32=1,
    )


def prepare(args) -> dict:
    path = args.run / "config.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["data_root"] != str(args.data_root.resolve()) or config["parent_checkpoint"] != str(
            args.parent_checkpoint.resolve()
        ):
            raise ValueError("resume requires the same data root and parent checkpoint path")
        for name in SETTINGS:
            if name in os.environ and int(os.environ[name]) != config["settings"][name]:
                raise ValueError(f"resume setting changed: {name}; use a new RUN_ID")
        if _hash(SPLIT) != config["split_sha256"] or any(
            _hash(Path(p)) != digest for p, digest in config["input_sha256"].items()
        ):
            raise ValueError("frozen split or selected subject bytes changed")
        if _hash(Path(config["template"])) != config["template_sha256"]:
            raise ValueError("prepared Train-mean bytes changed")
        parent = Path(config["parent_checkpoint"])
        binding = args.run / "parent_binding.json"
        expected = config["parent_sha256"] or (json.loads(binding.read_text())["sha256"] if binding.exists() else None)
        if expected is not None and (not parent.is_file() or _hash(parent) != expected):
            raise ValueError("frozen parent checkpoint bytes changed; restore them or use a new RUN_ID")
        return config
    settings = {name: int(os.environ.get(name, default)) for name, default in SETTINGS.items()}
    if any(value < 1 for value in settings.values()):
        raise ValueError("profile sizes, epochs and steps must be positive")
    if settings["PROFILE_EPOCHS"] < 2:
        raise ValueError("profile needs a warm-up epoch and an active-prior epoch")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = [item["subject_id"] for item in split["training"]]
    selected = sorted(ids, key=lambda value: hashlib.sha256(f"population-profile-0:{value}".encode()).digest())[:6]
    subjects = [args.data_root.resolve() / f"p_{subject.rsplit('_P', 1)[1]}.pkl" for subject in selected]
    if len(subjects) != 6 or not all(path.is_file() for path in subjects):
        raise ValueError("six selected subjects of the frozen Train split must exist")
    template = load_image(subjects[0], 1.0, "cpu").double()
    for subject in subjects[1:4]:
        image = load_image(subject, 1.0, "cpu")
        if image.shape != template.shape:
            raise ValueError("selected Train subjects must share the processed grid")
        template.add_(image)
    template = (template / 4).float()
    levels = settings["PROFILE_LEVELS"]
    training = reference_training_options(template, levels, settings["PROFILE_WIDTH"], settings["PROFILE_EPOCHS"])
    haar_forward(torch.zeros(1, 3, *(n - 4 for n in template.shape[2:])), levels)
    config = dict(
        scope="PRELIMINARY_DIAGNOSTICS_NOT_MAIN_TRAINING_OR_MODEL_SELECTION",
        data_root=str(args.data_root.resolve()),
        parent_checkpoint=str(args.parent_checkpoint.resolve()),
        parent_sha256=_hash(args.parent_checkpoint) if args.parent_checkpoint.is_file() else None,
        split=str(SPLIT),
        split_sha256=_hash(SPLIT),
        subjects=[str(path) for path in subjects],
        subject_ids=selected,
        inner_train_indices=[0, 1, 2, 3],
        inner_heldout_indices=[4, 5],
        input_sha256={str(path): _hash(path) for path in subjects},
        settings=settings,
        training=training,
        map_lr=0.05,
        smoothing_sigmas=[4.0, 2.0, 0.0],
        probe_lr=0.15,
        probe_regularization=0.01,
        ridge_relative_grid=[1e-4, 1e-2, 1.0, 100.0],
        template=str(args.run.resolve() / "train_mean.npy"),
        shape=list(template.shape[2:]),
    )
    args.run.mkdir(parents=True, exist_ok=True)
    np.save(config["template"], template.numpy())
    config["template_sha256"] = _hash(Path(config["template"]))
    write_json(path, config)
    return config


def _attempt(run: Path, task: str) -> Path:
    path = run / task / "attempts" / f"{time.time_ns()}_{os.getpid()}"
    path.mkdir(parents=True)
    return path


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def profile_arm(args, config, output: Path) -> dict:
    arm = ARMS[args.task]
    options = argparse.Namespace(
        **config["training"],
        arm=arm,
        subjects=[Path(p) for p in config["subjects"][:4]],
        output=output / "fit",
        device=args.device,
        resume=False,
    )
    trained = train_model(options)
    model, payload = load_population_checkpoint(output / "fit/last.pth", args.device)
    model.eval()
    mean, covariance = appearance_prior(options, args.device)
    variance = payload["log_noise_variance"].to(args.device).double().exp()
    cases = []
    for index in config["inner_heldout_indices"]:
        image = load_image(Path(config["subjects"][index]), options.image_scale, args.device)
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        _sync(args.device)
        started = time.perf_counter()
        velocity, flow, information = fit_subject(
            model,
            image,
            variance,
            mean,
            covariance,
            options.gain_min,
            steps=config["settings"]["MAP_STEPS"],
            lr=config["map_lr"],
        )
        _sync(args.device)
        elapsed = time.perf_counter() - started
        map_peak = (
            torch.cuda.max_memory_allocated(args.device) / 2**30 if torch.device(args.device).type == "cuda" else None
        )
        np.save(output / f"map_{index}_velocity.npy", velocity.cpu().numpy())
        with torch.no_grad():
            marginal = appearance_marginal(
                image, _warp(model.atlas(), flow), variance, mean, covariance, options.gain_min
            )
            residual = residual_summary(marginal.residual)
        cases.append(
            dict(
                subject=config["subjects"][index],
                seconds=elapsed,
                residual=residual,
                **information,
                peak_allocated_gib=map_peak,
                memory_scope="MAP only, before importance sampling",
                importance=importance_diagnostic(model, image, variance, mean, covariance, options.gain_min, 2),
            )
        )
        del velocity, flow
    hardware = {"device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.device(args.device).type == "cuda":
        properties = torch.cuda.get_device_properties(args.device)
        hardware.update(name=properties.name, total_memory_bytes=properties.total_memory)
    return dict(
        task=args.task,
        arm=arm,
        hardware=hardware,
        epochs=trained["history"],
        cases=cases,
        train_updates=sum(row["updates"] for row in trained["history"]),
        scope="technical fit/cost only; short-run arm differences are not a scientific comparison",
        checkpoint=trained["checkpoint"],
        parameters_total=sum(parameter.numel() for parameter in model.parameters()),
        parameters_trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def profile_probe(args, config, output: Path) -> dict:
    subject = Path(config["subjects"][PROBES[args.task]])
    options = argparse.Namespace(
        moving=Path(config["subjects"][0]),
        fixed=subject,
        output=output / "fit",
        steps=config["settings"]["PROBE_STEPS"],
        lr=config["probe_lr"],
        regularization=config["probe_regularization"],
        collar=config["training"]["collar"],
        levels=config["training"]["levels"],
        integration_steps=config["training"]["integration_steps"],
        smoothing_sigmas=config["smoothing_sigmas"],
        device=args.device,
    )
    _sync(args.device)
    started = time.perf_counter()
    report = run_probe(options)
    _sync(args.device)
    return dict(
        task=args.task,
        seconds=time.perf_counter() - started,
        fit=str(options.output.resolve()),
        initial_dice=report["initial_dice"],
        dice=report["dice"],
        label_ids=report["label_ids"],
        foreground_mismatch_fraction=report["foreground_mismatch_fraction"],
        scope="supervised geometry diagnostic; quality is reported, not used as an A rejection gate",
    )


def axis_frequency_energy(residual: torch.Tensor) -> list[list[float]]:
    power = torch.fft.fftn(residual - residual.mean(), norm="ortho").abs().square()
    return [power.sum(dim=tuple(j for j in range(3) if j != axis)).cpu().tolist() for axis in range(3)]


@torch.no_grad()
def profile_residuals(args, config, output: Path) -> dict:
    from experiments.dispatch1.bridges import Parent

    checkpoint = Path(config["parent_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"K3 parent checkpoint missing: {checkpoint}")
    digest = _hash(checkpoint)
    binding = args.run / "parent_binding.json"
    expected = config["parent_sha256"] or (json.loads(binding.read_text())["sha256"] if binding.exists() else None)
    if expected is not None and digest != expected:
        raise ValueError("parent checkpoint bytes differ from the frozen input")
    if expected is None:
        write_json(binding, {"path": str(checkpoint), "sha256": digest})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parent = Parent.load(str(checkpoint), "CTCF-CascadeA-Mamba", False, 6, args.device)
    raw = torch.from_numpy(np.load(config["template"], allow_pickle=False)).to(args.device)
    centered = raw - raw.mean()
    template = centered / centered.square().mean().sqrt()
    options = argparse.Namespace(**config["training"])
    mean, covariance = appearance_prior(options, args.device)
    reports = []
    for index in config["inner_heldout_indices"]:
        observed = load_image(Path(config["subjects"][index]), options.image_scale, args.device)
        # Deliberately FP32/TF32; this is not a reproduction of historical AMP Dice.
        with torch.autocast(device_type=torch.device(args.device).type, enabled=False):
            _, legacy = parent.model(raw, observed, alpha_l1=1.0, alpha_l3=1.0)
        transformed = _warp(template, legacy_sampling_displacement(legacy.float()))
        for name, image in (("identity", template), ("parent", transformed)):
            marginal = appearance_marginal(observed, image, options.noise_std**2, mean, covariance, options.gain_min)
            residual = marginal.residual
            axis_energy = axis_frequency_energy(residual[0, 0])
            tile_rms = [
                float(tile.square().mean().sqrt())
                for slab in residual.chunk(2, 2)
                for block in slab.chunk(2, 3)
                for tile in block.chunk(2, 4)
            ]
            reports.append(
                dict(
                    subject=config["subjects"][index],
                    map=name,
                    nll_per_voxel=float(-marginal.log_prob[0] / observed.numel()),
                    residual=residual_summary(residual),
                    tile_rms=tile_rms,
                    axis_frequency_energy=axis_energy,
                )
            )
            np.save(output / f"residual_{index}_{name}.npy", residual[0, 0].float().cpu().numpy())
    return dict(
        task=args.task,
        parent_checkpoint=str(checkpoint),
        parent_sha256=digest,
        parent_load=parent.load_report,
        frequency_units="cycles per voxel, fftfreq on every axis; marginals of full orthonormal FFT energy",
        heldout_scope="disjoint from four profile-A images; frozen parent may have seen these subjects",
        cases=reports,
        scope="parent residual includes geometry, anatomy, interpolation and appearance; not pure noise or a model-selection verdict",
    )


def summarize(args, config) -> dict:
    tasks = (*ARMS, *PROBES, "k3")
    reports = {
        task: json.loads((args.run / task / "report.json").read_text(encoding="utf-8"))
        for task in tasks
        if (args.run / f"{task}.done").exists() and (args.run / task / "report.json").exists()
    }
    summary = {
        "scope": config["scope"],
        "available_tasks": list(reports),
        "missing_tasks": sorted(set(tasks) - reports.keys()),
        "tasks": reports,
    }
    if all(task in reports for task in PROBES):
        train = [Path(reports[task]["fit"]) for task in list(PROBES)[:4]]
        heldout = [Path(reports[task]["fit"]) for task in list(PROBES)[4:]]
        features = []
        collar, levels = config["training"]["collar"], config["training"]["levels"]
        for path in train[1:]:
            values = torch.from_numpy(np.load(path / "velocity.npy", allow_pickle=False))
            features.append(
                haar_forward(values[:, :, collar:-collar, collar:-collar, collar:-collar], levels).coarse.flatten()
            )
        features = torch.stack(features).double()
        scale = float((features - features.mean(0)).square().sum() / (len(features) - 1))
        if scale > 0:
            readout = evaluate_predictability(
                train, heldout, ridge_grid=[scale * factor for factor in config["ridge_relative_grid"]]
            )
            write_json(args.run / "k2_readout.json", readout)
            summary["k2"] = {"status": "DESCRIPTIVE_ONLY", "ridge_scale_train_only": scale, "file": "k2_readout.json"}
        else:
            summary["k2"] = {"status": "UNINFORMATIVE", "reason": "zero coarse variation in inner Train"}
    write_json(args.run / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("prepare", *ARMS, *PROBES, "k3", "summary"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.task == "prepare":
        prepare(args)
        print("[POPULATION PROFILE] prepared: fixed inner Train/heldout, diagnostic settings only", flush=True)
        return
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if args.task == "summary":
        config = prepare(args)
        summary = summarize(args, config)
        print(json.dumps({"available": summary["available_tasks"], "missing": summary["missing_tasks"]}), flush=True)
        if summary["missing_tasks"]:
            raise SystemExit(1)
        return
    output = _attempt(args.run, args.task)
    report = (
        profile_arm(args, config, output)
        if args.task in ARMS
        else profile_probe(args, config, output)
        if args.task in PROBES
        else profile_residuals(args, config, output)
    )
    report["attempt"] = str(output.resolve())
    write_json(args.run / args.task / "report.json", report)
    print(f"[POPULATION PROFILE] {args.task} done", flush=True)


if __name__ == "__main__":
    main()
