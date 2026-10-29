"""Expanded image-only A calibration and supervised geometry diagnostics for the train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.core.data_loaders import oasis_subject_paths, oasis_validation_pairs
from experiments.core.field_scoring import score, score_ff_rep
from experiments.core.inference_metrics import write_results
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.precision import configure_precision
from experiments.population.evaluate import fit_subject, predictive_subject, score_pair_field
from experiments.population.geometry import haar_forward, subject_pair_flow
from experiments.population.observation import appearance_prior, load_image
from experiments.population.predictability import _interior, _read_fits, evaluate_predictability, local_predictability
from experiments.population.probe import _load_labels, run_probe
from experiments.population.profile import SPLIT, reference_training_options, write_json
from experiments.population.train import load_population_checkpoint, train_model
from tools.analysis.run_artifacts import sha256_file
from utils.field import legacy_sampling_displacement

ARMS = {
    "lin64_s0": ("A_lin", 64, 8, 0, 100),
    "a1_64_s0": ("A1", 64, 8, 0, 100),
    "lin64_s1": ("A_lin", 64, 8, 1, 100),
    "a1_64_s1": ("A1", 64, 8, 1, 100),
    "lin278": ("A_lin", 278, 8, 0, 50),
    "a1_278": ("A1", 278, 8, 0, 50),
    "a1_w16": ("A1", 64, 16, 0, 100),
}


def save_array(path, array):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def prepare(run: Path, data: Path):
    config_path = run / "config.json"
    pairs = SPLIT.with_name("pair_manifest.json")
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if (
            config["data_root"] != str(data.resolve())
            or config["split_sha256"] != sha256_file(SPLIT)
            or config["pairs_sha256"] != sha256_file(pairs)
        ):
            raise ValueError("calibration split/data root changed")
        for path, digest in config["input_sha256"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"frozen subject bytes changed: {path}")
        if sha256_file(Path(config["template"])) != config["template_sha256"]:
            raise ValueError("frozen initial atlas changed")
        return config
    training, development = oasis_subject_paths(data, SPLIT)
    oasis_validation_pairs(development, pairs)
    if (len(training), len(development)) != (294, 100):
        raise ValueError("this calibration requires the frozen 294/100 split")
    ordered = sorted(
        training, key=lambda p: hashlib.sha256(f"population-profile-0:OASIS_ALL_P{Path(p).stem[2:]}".encode()).digest()
    )
    heldout = ordered[4:20]
    small = ordered[:4] + ordered[20:80]
    large = [path for path in ordered if path not in heldout]
    template = sum(load_image(Path(p), 1.0, "cpu").double() for p in small) / len(small)
    run.mkdir(parents=True, exist_ok=True)
    save_array(run / "initial_atlas.npy", template.float().numpy())
    (run / "split.json").write_bytes(SPLIT.read_bytes())
    (run / "pair_manifest.json").write_bytes(pairs.read_bytes())
    options = reference_training_options(template, 3, 8, 100)
    options.update(warmup_updates=320, warmup_epochs=5, snapshot_epochs=[50, 100], snapshot_updates=[3200, 6400])
    config = {
        "scope": "CALIBRATION_NOT_MAIN_A_OR_OFFICIAL_TEST",
        "data_root": str(data.resolve()),
        "split_sha256": sha256_file(SPLIT),
        "pairs_sha256": sha256_file(pairs),
        "training": ordered,
        "development": development,
        "small": small,
        "large": large,
        "heldout": heldout,
        "reference": small[0],
        "template": str((run / "initial_atlas.npy").resolve()),
        "template_sha256": sha256_file(run / "initial_atlas.npy"),
        "input_sha256": {p: sha256_file(Path(p)) for p in training + development},
        "training_options": options,
        "map_steps": 500,
        "map_lr": 0.05,
        "map_readouts": [0, 40, 100, 200, 500],
        "probe_steps": 40,
        "probe_sigmas": [4.0, 2.0, 0.0],
        "probe_lr": 0.15,
        "probe_regularization": 0.01,
        "local_epochs": 20,
        "predictive_block_size": 32,
        "predictive_gap": 4,
        "arms": {name: list(values) for name, values in ARMS.items()},
    }
    write_json(config_path, config)
    return config


def assess_checkpoint(checkpoint, output, config, device, *, predictive=False):
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "map_steps": config["map_steps"],
        "map_lr": config["map_lr"],
        "heldout": config["heldout"],
        "input_sha256": config["input_sha256"],
    }
    recipe = output / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != binding:
        raise ValueError("cached MAP belongs to a different checkpoint/recipe")
    write_json(recipe, binding)
    model, payload = load_population_checkpoint(checkpoint, device)
    model.eval()
    trained = argparse.Namespace(**payload["arguments"])
    torch.backends.cuda.matmul.allow_tf32 = bool(trained.tf32)
    torch.backends.cudnn.allow_tf32 = bool(trained.tf32)
    mean, covariance = appearance_prior(trained, device)
    variance = payload["log_noise_variance"].to(device).double().exp()
    fits = {}
    for path in config["heldout"]:
        directory = output / "subjects" / Path(path).stem
        directory.mkdir(parents=True, exist_ok=True)
        report_path, velocity_path = directory / "report.json", directory / "velocity.npy"
        if report_path.exists():
            info = json.loads(report_path.read_text())
            if sha256_file(velocity_path) != info["velocity_sha256"]:
                raise ValueError("cached velocity bytes changed")
        else:
            image = load_image(Path(path), trained.image_scale, device)
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()

            def snapshot(step, velocity, directory=directory):
                save_array(directory / f"step_{step}.npy", velocity.cpu().numpy())

            velocity, flow, info = fit_subject(
                model,
                image,
                variance,
                mean,
                covariance,
                trained.gain_min,
                steps=config["map_steps"],
                lr=config["map_lr"],
                readout_steps=config["map_readouts"],
                on_readout=snapshot,
            )
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            info["seconds"] = time.perf_counter() - started
            info["timing"] = "encoder/MAP wall time including readout transfers/writes; excludes initial image load"
            info["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 2**30 if torch.device(device).type == "cuda" else None
            )
            save_array(velocity_path, velocity.cpu().numpy())
            info["velocity_sha256"] = sha256_file(velocity_path)
            write_json(report_path, info)
            del image, velocity, flow
        fits[path] = info
        print(f"[A SUBJECT] {Path(path).stem} best={info['best_step']} seconds={info['seconds']:.1f}", flush=True)
    rows = {"FF": [], "REP": []}
    for i, moving in enumerate(config["heldout"]):
        fixed = config["heldout"][(i + 1) % len(config["heldout"])]
        case_id = f"{Path(moving).stem}__{Path(fixed).stem}"
        case = output / "pairs" / case_id
        if (case / "report.json").exists():
            pair = json.loads((case / "report.json").read_text())
            for mode, digest in pair["field_sha256"].items():
                if sha256_file(case / f"{mode}.npz") != digest:
                    raise ValueError("cached pair-field bytes changed")
        else:
            velocities = [
                torch.from_numpy(np.load(output / "subjects" / Path(p).stem / "velocity.npy")).to(device)
                for p in (moving, fixed)
            ]
            with torch.no_grad():
                flow = subject_pair_flow(velocities[1], velocities[0], trained.integration_steps).float()
            labels = [_load_labels(Path(p), device) for p in (moving, fixed)]
            scored, details = score_pair_field(
                flow,
                *labels,
                case_id,
                case,
                fits[moving]["seconds"] + fits[fixed]["seconds"],
                repair_output=True,
                certify_output=True,
            )
            pair = {
                "moving": moving,
                "fixed": fixed,
                "rows": scored,
                "field_sha256": {mode: sha256_file(case / f"{mode}.npz") for mode in scored},
                **details,
            }
            write_json(case / "report.json", pair)
            del velocities, flow, labels
        for mode in rows:
            rows[mode].append(pair["rows"][mode])
        readouts = case / "map_readouts.json"
        if not readouts.exists():
            values = {}
            labels = [_load_labels(Path(p), device) for p in (moving, fixed)]
            with torch.no_grad():
                for step in config["map_readouts"]:
                    velocities = [
                        torch.from_numpy(
                            np.load(output / "subjects" / Path(p).stem / f"step_{step}.npy", allow_pickle=False)
                        ).to(device)
                        for p in (moving, fixed)
                    ]
                    flow = subject_pair_flow(velocities[1], velocities[0], trained.integration_steps).float()
                    values[step] = score(flow, case_id, 0.0, *labels, 0.0, "canonical")
            write_json(
                readouts,
                {
                    "scope": "raw fixed-step MAP diagnostics; no repair/certification or Dice-based state selection",
                    "steps": values,
                },
            )
            del velocities, flow, labels
    for mode, values in rows.items():
        (output / mode).mkdir(exist_ok=True)
        write_results(
            values, str(output / mode), f"population_{model.arm}", str(checkpoint), "inner-heldout cyclic subject pairs"
        )
    if predictive:
        for path in config["heldout"]:
            target = output / "subjects" / Path(path).stem / "predictive.json"
            if not target.exists():
                image = load_image(Path(path), trained.image_scale, device)
                result = predictive_subject(
                    model,
                    image,
                    variance,
                    mean,
                    covariance,
                    trained.gain_min,
                    steps=config["map_steps"],
                    lr=config["map_lr"],
                    block_size=config["predictive_block_size"],
                    gap=config["predictive_gap"],
                )
                write_json(target, result)
    write_json(
        output / "report.json",
        {
            "checkpoint": str(checkpoint),
            "subjects": fits,
            "pair_count": len(rows["FF"]),
            "timing": "pair cost sums two cached subject fits; composition, scoring, repair and certification excluded",
            "scope": "exploratory; inner-heldout from frozen Train, not an independent test",
        },
    )


def fit_arm(run, config, name, device):
    arm, count, width, seed, epochs = ARMS[name]
    root = run / "arms" / name
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "training_path.json"
    fit = Path(json.loads(pointer.read_text())["path"]) if pointer.exists() else root / "fit"
    if fit.exists() and not (fit / "last.pth").exists():
        fit = root / f"fit_{time.time_ns()}"
    write_json(pointer, {"path": str(fit.resolve())})
    options = config["training_options"] | dict(
        arm=arm,
        width=width,
        seed=seed,
        epochs=epochs,
        subjects=[Path(p) for p in config["small" if count == 64 else "large"]],
        output=fit,
        initial_atlas=Path(config["template"]),
        device=device,
        resume=(fit / "last.pth").exists(),
    )
    train_model(argparse.Namespace(**options))
    assess_checkpoint(fit / "last.pth", root / "last", config, device, predictive=name == "lin64_s0")
    if count == 278:
        assess_checkpoint(fit / "update_6400.pth", root / "update_6400", config, device)
    write_json(
        root / "report.json",
        {
            "arm": name,
            "training_directory": str(fit),
            "completed_epochs": epochs,
            "evaluations": ["last", "update_6400"] if count == 278 else ["last"],
        },
    )


def label_directory(run, path):
    return Path(json.loads((run / "labels" / Path(path).stem / "report.json").read_text())["directory"])


def parent_reference(run, config, checkpoint, device):
    """Descriptive historical-parent reference on exactly the A inner-heldout pairs."""
    output = run / "parent"
    binding = {
        "checkpoint": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "heldout": config["heldout"],
        "precision": "tf32",
    }
    if (output / "binding.json").exists() and json.loads((output / "binding.json").read_text()) != binding:
        raise ValueError("parent reference checkpoint or subjects changed")
    write_json(output / "binding.json", binding)
    shape = tuple(load_image(Path(config["heldout"][0]), 1.0, "cpu").shape[2:])
    model = get_model_adapter("ctcf").build(
        config_key="CTCF-CascadeA-Mamba",
        time_steps=6,
        img_size=shape,
        dwin_size=volume_window_size(shape),
        l3_svf=False,
    )
    loaded = load_checkpoint_state(model, str(checkpoint), strict=True)
    model = model.to(device).eval()
    configure_precision("tf32", torch.device(device))
    rows = {"FF": [], "REP": []}
    with torch.no_grad():
        for index, moving in enumerate(config["heldout"]):
            fixed = config["heldout"][(index + 1) % len(config["heldout"])]
            x, y = [load_image(Path(p), 1.0, device) for p in (moving, fixed)]
            labels = [_load_labels(Path(p), device) for p in (moving, fixed)]
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            _, stored = model(x, y, alpha_l1=1.0, alpha_l3=1.0)
            flow = legacy_sampling_displacement(stored.float())
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            ff, rep = score_ff_rep(
                flow, f"{Path(moving).stem}__{Path(fixed).stem}", time.perf_counter() - started, *labels, "canonical"
            )
            rows["FF"].append(ff)
            rows["REP"].append(rep)
    for mode, values in rows.items():
        (output / mode).mkdir(exist_ok=True)
        write_results(values, str(output / mode), "historical_mamba", str(checkpoint), "A inner-heldout cyclic pairs")
    write_json(
        output / "report.json",
        {
            **binding,
            "load": loaded,
            "pairs": len(rows["FF"]),
            "convention": "canonical sampler-map displacement, materialized from legacy flow",
            "certification": "NOT_RUN; no zero-boundary or global-homeomorphism claim for parent",
            "scope": "descriptive in-sample parent, not a leakage contrast or equal-time MAP comparison",
        },
    )


def labels(run, config, shard, shards, device):
    for path in config["training"][shard::shards]:
        output = run / "labels" / Path(path).stem
        if (output / "report.json").exists():
            continue
        attempt = output / "attempts" / str(time.time_ns())
        report = run_probe(
            argparse.Namespace(
                moving=Path(config["reference"]),
                fixed=Path(path),
                output=attempt,
                steps=config["probe_steps"],
                lr=config["probe_lr"],
                regularization=config["probe_regularization"],
                collar=2,
                levels=3,
                integration_steps=7,
                smoothing_sigmas=config["probe_sigmas"],
                device=device,
            )
        )
        write_json(output / "report.json", report | {"directory": str(attempt.resolve())})
    (run / f"labels_{shards}_{shard}.done").touch()


def ridge_curve(run, config):
    reference = label_directory(run, config["reference"])
    train = [label_directory(run, p) for p in config["large"] if p != config["reference"]]
    heldout = [label_directory(run, p) for p in config["heldout"]]
    records = _read_fits(train)
    features = torch.stack([haar_forward(_interior(row, 2), 3).coarse.flatten() for row in records]).double()
    for count in (16, 32, 63, 127, len(train)):
        destination = run / "ridge" / f"n_{count}.json"
        if destination.exists():
            continue
        subset = features[:count]
        scale = float((subset - subset.mean(0)).square().sum() / (count - 1))
        result = evaluate_predictability(
            [reference, *train[:count]], heldout, ridge_grid=[scale * f for f in (1e-4, 1e-2, 1.0, 100.0)]
        )
        write_json(destination, result)


def summarize(run, config):
    arms = {}
    for name in ARMS:
        paths = list((run / "arms" / name).glob("*/REP/summary.json"))
        arms[name] = {p.parent.parent.name: json.loads(p.read_text()) for p in paths}
    result = {
        "scope": config["scope"],
        "arms": arms,
        "label_reports": len(list((run / "labels").glob("*/report.json"))),
        "ridge_reports": len(list((run / "ridge").glob("*.json"))),
        "local_reports": {p.parent.name: json.loads(p.read_text()) for p in (run / "local").glob("*/report.json")},
        "parent": json.loads((run / "parent/report.json").read_text())
        if (run / "parent/report.json").exists()
        else None,
        "automatic_scientific_verdict": "NOT_APPLICABLE",
    }
    write_json(run / "summary.json", result)
    if (
        any(
            not {"last", *(["update_6400"] if ARMS[name][1] == 278 else [])}.issubset(value)
            for name, value in arms.items()
        )
        or result["label_reports"] != 294
        or result["ridge_reports"] != 5
        or len(result["local_reports"]) != 3
        or result["parent"] is None
    ):
        raise RuntimeError("calibration report inventory is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("prepare", "fit", "labels", "ridge", "local", "parent", "summary"), required=True
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/data/mooncake/P/OASIS_L2R_2021_task03/All"))
    parser.add_argument("--arm", choices=(*ARMS, "A0", "A_lin", "A1"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-checkpoint", type=Path)
    args = parser.parse_args()
    if args.task == "prepare":
        prepare(args.run, args.data)
        return
    config = json.loads((args.run / "config.json").read_text())
    if args.task == "fit":
        fit_arm(args.run, config, args.arm, args.device)
    elif args.task == "labels":
        if not 0 <= args.shard < args.shards:
            raise ValueError("invalid label shard")
        labels(args.run, config, args.shard, args.shards, args.device)
    elif args.task == "ridge":
        ridge_curve(args.run, config)
    elif args.task == "parent":
        if args.parent_checkpoint is None:
            parser.error("parent requires --parent-checkpoint")
        parent_reference(args.run, config, args.parent_checkpoint, args.device)
    elif args.task == "local":
        train = [label_directory(args.run, p) for p in config["large"]]
        heldout = [label_directory(args.run, p) for p in config["heldout"]]
        options = {
            k: config["training_options"][k]
            for k in (
                "levels",
                "collar",
                "width",
                "scale_min",
                "scale_max",
                "prior_std",
                "posterior_std",
                "integration_steps",
            )
        }
        local_predictability(
            train,
            heldout,
            options | {"arm": args.arm},
            args.run / "local" / args.arm,
            epochs=config["local_epochs"],
            device=args.device,
        )
    else:
        summarize(args.run, config)


if __name__ == "__main__":
    main()
