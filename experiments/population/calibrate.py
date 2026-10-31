"""Expanded image-only A calibration and supervised geometry diagnostics for the train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.core.data_loaders import oasis_subject_paths, oasis_validation_pairs
from experiments.core.field_scoring import score, score_ff_rep
from experiments.core.inference_metrics import write_results
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.precision import configure_precision
from experiments.population.evaluate import (
    evaluate_model,
    fit_subject,
    pair_errors,
    predictive_subject,
    score_pair_field,
    velocity_rms_by_scale,
)
from experiments.population.geometry import canonical_flow, haar_forward, subject_pair_flow
from experiments.population.observation import appearance_prior, load_image
from experiments.population.predictability import _interior, _read_fits, evaluate_predictability, local_predictability
from experiments.population.probe import _load_labels, run_probe
from experiments.population.profile import SPLIT, reference_training_options, write_json
from experiments.population.train import load_population_checkpoint, train_model
from tools.analysis.run_artifacts import sha256_file
from utils import pkload
from utils.field import _warp, legacy_sampling_displacement
from utils.field_io import load_flow_npz

ARMS = {
    "lin64_s0": ("A_lin", 64, 8, 0, 100),
    "a1_64_s0": ("A1", 64, 8, 0, 100),
    "lin64_s1": ("A_lin", 64, 8, 1, 100),
    "a1_64_s1": ("A1", 64, 8, 1, 100),
    "lin278": ("A_lin", 278, 8, 0, 100),
    "a1_278": ("A1", 278, 8, 0, 100),
    "a1_w16": ("A1", 64, 16, 0, 100),
    "a0_64_s0": ("A0", 64, 8, 0, 100),
}
# 278 x 100 passes 19200 updates, the budget of the 64 x 300 extension; both are compared there.
LARGE_SNAPSHOT_UPDATES = [3200, 6400, 19200]
EXTENDED_EPOCHS = 300
EXTENDED_SNAPSHOT_EPOCHS = [200, 300]
TRAIN_PROBE_SUBJECTS = 8


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


def cyclic_pairs(subjects):
    return [(moving, subjects[(i + 1) % len(subjects)]) for i, moving in enumerate(subjects)]


def cohort_pairs(run, config, cohort):
    """Subjects and directed (moving, fixed) pairs of one evaluation cohort."""
    if cohort == "dev":
        return config["development"], oasis_validation_pairs(config["development"], run / "pair_manifest.json")
    subjects = config["heldout"] if cohort == "heldout" else config["small"][:TRAIN_PROBE_SUBJECTS]
    return subjects, cyclic_pairs(subjects)


def case_name(moving, fixed):
    return f"{Path(moving).stem}__{Path(fixed).stem}"


def load_velocity(output, subject, name, device):
    return torch.from_numpy(np.load(output / "subjects" / Path(subject).stem / name, allow_pickle=False)).to(device)


def fit_subjects(subjects, output, config, device, loaded, *, readouts):
    """Fit each subject once; a killed fit restarts, a finished one is checked against its stored velocity hash."""
    model, trained, mean, covariance, variance = loaded
    for path in subjects:
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
                readout_steps=config["map_readouts"] if readouts else (),
                on_readout=snapshot if readouts else None,
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
        print(f"[A SUBJECT] {Path(path).stem} best={info['best_step']} seconds={info['seconds']:.1f}", flush=True)


def score_pairs(pairs, output, config, device, loaded, *, repair_output, certify_output, readouts):
    """Score each pair once; only a pair whose step raised is recomputed, a negative verdict stays an outcome."""
    trained = loaded[1]
    for moving, fixed in pairs:
        case_id = case_name(moving, fixed)
        case = output / "pairs" / case_id
        pair = json.loads((case / "report.json").read_text()) if (case / "report.json").exists() else None
        if pair is not None and pair_errors(pair):
            pair = None
        if pair is not None:
            for mode, digest in pair["field_sha256"].items():
                if sha256_file(case / f"{mode}.npz") != digest:
                    raise ValueError("cached pair-field bytes changed")
        else:
            seconds = [
                json.loads((output / "subjects" / Path(p).stem / "report.json").read_text())["seconds"]
                for p in (moving, fixed)
            ]
            velocities = [load_velocity(output, p, "velocity.npy", device) for p in (moving, fixed)]
            with torch.no_grad():
                flow = subject_pair_flow(velocities[1], velocities[0], trained.integration_steps).float()
            labels = [_load_labels(Path(p), device) for p in (moving, fixed)]
            scored, details = score_pair_field(
                flow,
                *labels,
                case_id,
                case,
                sum(seconds),
                repair_output=repair_output,
                certify_output=certify_output,
            )
            write_json(
                case / "report.json",
                {
                    "moving": moving,
                    "fixed": fixed,
                    "rows": scored,
                    "field_sha256": {mode: sha256_file(case / f"{mode}.npz") for mode in scored},
                    **details,
                },
            )
            del velocities, flow, labels
        target = case / "map_readouts.json"
        if readouts and not target.exists():
            values = {}
            labels = [_load_labels(Path(p), device) for p in (moving, fixed)]
            with torch.no_grad():
                for step in config["map_readouts"]:
                    velocities = [load_velocity(output, p, f"step_{step}.npy", device) for p in (moving, fixed)]
                    flow = subject_pair_flow(velocities[1], velocities[0], trained.integration_steps).float()
                    values[step] = score(flow, case_id, 0.0, *labels, 0.0, "canonical")
            write_json(
                target,
                {
                    "scope": "raw fixed-step MAP diagnostics; no repair/certification or Dice-based state selection",
                    "steps": values,
                },
            )
            del velocities, flow, labels


def slice_panel(volume):
    """Three central orthogonal slices of one [D,H,W] array as one 8-bit row with a shared intensity window."""
    values = volume.detach().float().cpu().numpy()
    low, high = np.percentile(values, [1, 99])
    slices = [np.take(values, size // 2, axis=axis) for axis, size in enumerate(values.shape)]
    height = max(part.shape[0] for part in slices)
    padded = [np.pad(part, ((0, height - part.shape[0]), (0, 0)), constant_values=low) for part in slices]
    return (np.clip((np.hstack(padded) - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)


def write_preview(path, output, subject, device, loaded):
    """Atlas, one subject and the atlas warped onto it at MAP: a visual check of what the fit reached."""
    model, trained = loaded[:2]
    with torch.no_grad():
        image = load_image(Path(subject), trained.image_scale, device)
        velocity = load_velocity(output, subject, "velocity.npy", device)
        warped = _warp(model.atlas(), canonical_flow(velocity, trained.integration_steps))
        rows = [slice_panel(model.atlas()[0, 0]), slice_panel(image[0, 0]), slice_panel(warped[0, 0])]
    Image.fromarray(np.vstack(rows)).save(path)


def label_map_distances(output, pairs, label_root, device, loaded):
    """Distance of each pair field from the label-driven pair map, a reference geometry and not a true deformation."""
    model, trained = loaded[:2]
    collar, levels = model.collar, model.levels
    distances = {}
    for moving, fixed in pairs:
        reports = [label_root / Path(p).stem / "report.json" for p in (moving, fixed)]
        if not all(report.exists() for report in reports):
            continue
        fields = [
            torch.from_numpy(
                np.load(Path(json.loads(report.read_text())["directory"]) / "velocity.npy", allow_pickle=False)
            ).to(device)
            for report in reports
        ]
        with torch.no_grad():
            reference = subject_pair_flow(fields[1], fields[0], trained.integration_steps).float()
            field = load_flow_npz(output / "pairs" / case_name(moving, fixed) / "FF.npz").to(device)
            foreground = _load_labels(Path(fixed), device) > 0
            difference = field - reference
            distances[case_name(moving, fixed)] = {
                "foreground_distance": float(
                    torch.linalg.vector_norm(difference, dim=1, keepdim=True)[foreground].mean()
                ),
                "foreground_label_map_norm": float(
                    torch.linalg.vector_norm(reference, dim=1, keepdim=True)[foreground].mean()
                ),
                "rms_by_scale": velocity_rms_by_scale(
                    difference[:, :, collar:-collar, collar:-collar, collar:-collar], levels
                ),
            }
    return distances


def write_diagnostics(path, output, subjects, pairs, label_root, device, loaded):
    """What each prior term and the encoder contribute at the inferred states; nothing here selects a model."""
    model = loaded[0]
    collar, levels = model.collar, model.levels

    def interior(subject, name):
        return load_velocity(output, subject, name, device)[:, :, collar:-collar, collar:-collar, collar:-collar]

    rows, finals, starts = {}, [], []
    for subject in subjects:
        final = interior(subject, "velocity.npy")
        rows[Path(subject).stem] = {"map": model.prior_terms(haar_forward(final, levels))}
        finals.append(final.cpu())
        if (output / "subjects" / Path(subject).stem / "step_0.npy").exists():
            start = interior(subject, "step_0.npy")
            rows[Path(subject).stem]["encoder"] = model.prior_terms(haar_forward(start, levels))
            starts.append(start.cpu())

    def spread(states):
        if len(states) < 2:
            return None
        mean = torch.stack(states).mean(dim=0)
        return np.mean([velocity_rms_by_scale(state - mean, levels) for state in states], axis=0).tolist()

    write_json(
        path,
        {
            "scope": "descriptive; explained fractions are detail energy at one inferred state, not predictive evidence",
            "prior_scales_by_band": model._scale(model.prior_scale_logits).detach().cpu().tolist(),
            "noise_std": float(loaded[4].sqrt()),
            "across_subject_rms_by_scale": {"map": spread(finals), "encoder": spread(starts)},
            "subjects": rows,
            "label_map_distance_voxels": label_map_distances(output, pairs, label_root, device, loaded),
        },
    )


def assess_checkpoint(
    checkpoint,
    output,
    config,
    device,
    *,
    subjects=None,
    pairs=None,
    predictive=False,
    repair_output=True,
    certify_output=True,
    readouts=True,
    stage="all",
    shard=0,
    shards=1,
    limit_pairs=None,
    scope="inner-heldout cyclic subject pairs",
    label_root=Path("labels"),
):
    """Subject MAP, then pair scores, then the summary; shards split the first two across processes."""
    subjects = config["heldout"] if subjects is None else subjects
    pairs = cyclic_pairs(subjects) if pairs is None else pairs
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "map_steps": config["map_steps"],
        "map_lr": config["map_lr"],
        "heldout": subjects,
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
    loaded = (model, trained, mean, covariance, variance)
    if limit_pairs:
        # The whole chain on real full-size pairs before the long queue. Its report exists only if the raw chain
        # held: the runner holds evaluation without it, since that failure would repeat on every pair.
        (output / "pair_check.json").unlink(missing_ok=True)
        needed = sorted({p for pair in pairs[:limit_pairs] for p in pair}, key=subjects.index)
        fit_subjects(needed, output, config, device, loaded, readouts=readouts)
        score_pairs(
            pairs[:limit_pairs],
            output,
            config,
            device,
            loaded,
            repair_output=repair_output,
            certify_output=certify_output,
            readouts=readouts,
        )
        failures = {}
        for moving, fixed in pairs[:limit_pairs]:
            report = json.loads((output / "pairs" / case_name(moving, fixed) / "report.json").read_text())
            failures[case_name(moving, fixed)] = pair_errors(report)
        status = "RAW_ONLY" if any(failures.values()) else "PASS"
        write_json(output / "pair_check.json", {"pairs": failures, "status": status})
        if status != "PASS":
            raise RuntimeError(f"raw scores are written, but the pair chain raised: {failures}")
        return
    if stage in ("all", "fit"):
        fit_subjects(subjects[shard::shards], output, config, device, loaded, readouts=readouts)
    if stage in ("all", "score"):
        score_pairs(
            pairs[shard::shards],
            output,
            config,
            device,
            loaded,
            repair_output=repair_output,
            certify_output=certify_output,
            readouts=readouts,
        )
    if stage in ("fit", "score"):
        (output / f"{stage}_{shard}_{shards}.done").touch()
        return
    rows, failures, finished = {"FF": [], "REP": []}, {}, True
    for moving, fixed in pairs:
        case = output / "pairs" / case_name(moving, fixed)
        pair = json.loads((case / "report.json").read_text())
        if pair_errors(pair):
            failures[case.name] = pair_errors(pair)
        for mode, row in pair["rows"].items():
            rows[mode].append(row)
        finished = finished and (not readouts or (case / "map_readouts.json").exists())
    for mode, values in rows.items():
        if values:
            (output / mode).mkdir(exist_ok=True)
            write_results(values, str(output / mode), f"population_{model.arm}", str(checkpoint), scope)
    write_json(
        output / "report.json",
        {
            "checkpoint": str(checkpoint),
            "subjects": {
                p: json.loads((output / "subjects" / Path(p).stem / "report.json").read_text()) for p in subjects
            },
            "pair_count": len(rows["FF"]),
            "failed_pairs": failures,
            "timing": "pair cost sums two cached subject fits; composition, scoring, repair and certification excluded",
            "scope": f"exploratory; {scope}, not an independent test",
        },
    )
    if not (output / "atlas_preview.png").exists():
        write_preview(output / "atlas_preview.png", output, subjects[0], device, loaded)
    if readouts and not (output / "diagnostics.json").exists():
        write_diagnostics(output / "diagnostics.json", output, subjects, pairs, label_root, device, loaded)
    if predictive:
        for path in subjects:
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
    if readouts and finished:
        for path in output.glob("subjects/*/step_*.npy"):
            path.unlink()
    if failures:
        raise RuntimeError(f"pair steps raised; every completed score is written: {failures}")


def training_options(config, name, **overrides):
    arm, count, width, seed, epochs = ARMS[name]
    return (
        config["training_options"]
        | dict(
            arm=arm,
            width=width,
            seed=seed,
            epochs=epochs,
            subjects=[Path(p) for p in config["small" if count == 64 else "large"]],
            initial_atlas=Path(config["template"]),
        )
        | overrides
    )


def fit_arm(run, config, name, device):
    root = run / "arms" / name
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "training_path.json"
    fit = Path(json.loads(pointer.read_text())["path"]) if pointer.exists() else root / "fit"
    if fit.exists() and not (fit / "last.pth").exists():
        fit = root / f"fit_{time.time_ns()}"
    write_json(pointer, {"path": str(fit.resolve())})
    snapshots = {"snapshot_updates": LARGE_SNAPSHOT_UPDATES} if ARMS[name][1] == 278 else {}
    options = training_options(config, name, output=fit, device=device, resume=(fit / "last.pth").exists(), **snapshots)
    train_model(argparse.Namespace(**options))
    write_json(root / "report.json", {"arm": name, "training_directory": str(fit), "completed_epochs": ARMS[name][4]})


def arm_point(run, name, point):
    """Checkpoint and folder of an evaluation point: a snapshot of the fit, or ext/<snapshot> after the extension."""
    root = run / "arms" / name
    if point.startswith("ext/"):
        return root / f"{point}.pth", point.replace("/", "_")
    return Path(json.loads((root / "training_path.json").read_text())["path"]) / f"{point}.pth", point


def extend_arm(run, config, name, device):
    """Continue a finished arm in its own folder; the evaluated checkpoint keeps its bytes and its cached MAP."""
    target = run / "arms" / name / "ext"
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "last.pth").exists():
        temporary = target / "last.pth.tmp"
        shutil.copyfile(arm_point(run, name, "last")[0], temporary)
        temporary.replace(target / "last.pth")
    options = training_options(
        config,
        name,
        epochs=EXTENDED_EPOCHS,
        output=target,
        device=device,
        resume=True,
        snapshot_epochs=EXTENDED_SNAPSHOT_EPOCHS,
        snapshot_updates=[],
    )
    train_model(argparse.Namespace(**options))
    write_json(target / "report.json", {"arm": name, "completed_epochs": EXTENDED_EPOCHS})


def assess_arm(run, config, name, point, device, *, cohort, **options):
    checkpoint, label = arm_point(run, name, point)
    subjects, pairs = cohort_pairs(run, config, cohort)
    assess_checkpoint(
        checkpoint,
        run / "arms" / name / (label if cohort == "heldout" else f"{label}_{cohort}"),
        config,
        device,
        subjects=subjects,
        pairs=pairs,
        readouts=cohort == "heldout",
        label_root=run / "labels",
        scope={
            "heldout": "inner-heldout cyclic subject pairs",
            "train": "fit-subject cyclic pairs, the training side of a generalization gap",
            "dev": "frozen Dev pairs in both directions",
        }[cohort],
        **options,
    )


def oasis19_arm(run, config, name, point, device):
    """Descriptive placement on the long inspected OASIS-19; no setting is ever chosen from it."""
    checkpoint, label = arm_point(run, name, point)
    output = run / "arms" / name / f"{label}_oasis19"
    if (output / "REP" / "summary.json").exists():
        return
    if output.exists():
        output.rename(output.with_name(f"{output.name}.attempt_{time.time_ns()}"))
    evaluate_model(
        argparse.Namespace(
            checkpoint=checkpoint,
            output=output,
            pairs=sorted(Path(config["data_root"]).with_name("Test").glob("*.pkl")),
            subjects=None,
            map_steps=config["map_steps"],
            map_lr=config["map_lr"],
            importance_samples=1,
            repair=1,
            certify=1,
            seed=0,
            device=device,
        )
    )


def label_directory(run, path):
    return Path(json.loads((run / "labels" / Path(path).stem / "report.json").read_text())["directory"])


def reference_cases(run, config, cohort, device):
    """(case id, moving, fixed, moving labels, fixed labels) of one cohort, from subject or paired OASIS files."""
    if cohort == "oasis19":
        for path in sorted(Path(config["data_root"]).with_name("Test").glob("*.pkl")):
            x, y, x_seg, y_seg = pkload(path)
            images = [
                torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))[None, None].to(device) for v in (x, y)
            ]
            labels = [
                torch.from_numpy(np.ascontiguousarray(np.asarray(v).astype(np.int64)))[None, None].to(device)
                for v in (x_seg, y_seg)
            ]
            yield (path.stem, *images, *labels)
        return
    pairs = cyclic_pairs(config["heldout"]) if cohort == "heldout" else cohort_pairs(run, config, cohort)[1]
    for moving, fixed in pairs:
        images = [load_image(Path(p), 1.0, device) for p in (moving, fixed)]
        labels = [_load_labels(Path(p), device) for p in (moving, fixed)]
        yield (case_name(moving, fixed), *images, *labels)


def parent_reference(run, config, checkpoint, device, *, output=None, cohort="heldout"):
    """Descriptive reference of a Mamba checkpoint; its own training split decides whether a cohort is in-sample."""
    output = run / "parent" if output is None else output
    binding = {
        "checkpoint": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "precision": "tf32",
        **({"heldout": config["heldout"]} if cohort == "heldout" else {"cohort": cohort}),
    }
    if (output / "binding.json").exists() and json.loads((output / "binding.json").read_text()) != binding:
        raise ValueError("parent reference checkpoint or subjects changed")
    write_json(output / "binding.json", binding)
    model, loaded = None, None
    rows = {"FF": [], "REP": []}
    with torch.no_grad():
        for case_id, x, y, *labels in reference_cases(run, config, cohort, device):
            if model is None:
                shape = tuple(x.shape[2:])
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
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            _, stored = model(x, y, alpha_l1=1.0, alpha_l3=1.0)
            flow = legacy_sampling_displacement(stored.float())
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            ff, rep = score_ff_rep(flow, case_id, time.perf_counter() - started, *labels, "canonical")
            rows["FF"].append(ff)
            rows["REP"].append(rep)
    for mode, values in rows.items():
        (output / mode).mkdir(exist_ok=True)
        write_results(values, str(output / mode), "historical_mamba", str(checkpoint), f"reference on {cohort}")
    write_json(
        output / "report.json",
        {
            **binding,
            "load": loaded,
            "pairs": len(rows["FF"]),
            "convention": "canonical sampler-map displacement, materialized from legacy flow",
            "certification": "NOT_RUN; no zero-boundary or global-homeomorphism claim for parent",
            "scope": "descriptive reference, not a leakage contrast or equal-time MAP comparison",
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
            if name != "a0_64_s0"
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
        "--task",
        choices=(
            "prepare",
            "fit",
            "extend",
            "assess",
            "oasis19",
            "labels",
            "ridge",
            "local",
            "parent",
            "reference",
            "summary",
        ),
        required=True,
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/data/mooncake/P/OASIS_L2R_2021_task03/All"))
    parser.add_argument("--arm", choices=(*ARMS, "A0", "A_lin", "A1"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--point", default="last", help="fit snapshot name, or ext/<name> after the extension")
    parser.add_argument("--cohort", choices=("heldout", "train", "dev", "oasis19"), default="heldout")
    parser.add_argument("--stage", choices=("all", "fit", "score", "summary"), default="all")
    parser.add_argument("--repair", type=int, choices=(0, 1), default=1)
    parser.add_argument("--certify", type=int, choices=(0, 1), default=1)
    parser.add_argument("--predictive", type=int, choices=(0, 1), default=0)
    parser.add_argument("--limit-pairs", type=int, help="run the chain on the first pairs only, without a summary")
    parser.add_argument("--label", help="reference folder name under references/")
    args = parser.parse_args()
    if args.task == "prepare":
        prepare(args.run, args.data)
        return
    config = json.loads((args.run / "config.json").read_text())
    if args.task == "fit":
        fit_arm(args.run, config, args.arm, args.device)
    elif args.task == "extend":
        extend_arm(args.run, config, args.arm, args.device)
    elif args.task == "assess":
        assess_arm(
            args.run,
            config,
            args.arm,
            args.point,
            args.device,
            cohort=args.cohort,
            stage=args.stage,
            shard=args.shard,
            shards=args.shards,
            repair_output=bool(args.repair),
            certify_output=bool(args.certify),
            predictive=bool(args.predictive),
            limit_pairs=args.limit_pairs,
        )
    elif args.task == "oasis19":
        oasis19_arm(args.run, config, args.arm, args.point, args.device)
    elif args.task == "reference":
        if args.parent_checkpoint is None or args.label is None:
            parser.error("reference requires --parent-checkpoint and --label")
        parent_reference(
            args.run,
            config,
            args.parent_checkpoint,
            args.device,
            output=args.run / "references" / args.label / args.cohort,
            cohort=args.cohort,
        )
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
