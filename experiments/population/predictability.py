"""Read out coarse-to-fine predictability of label-fitted velocities on disjoint subjects."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.population.geometry import HaarPyramid, canonical_flow, coefficient_bands, haar_forward, haar_inverse
from experiments.population.model import PopulationAtlas
from experiments.population.observation import load_image
from experiments.population.probe import _load_labels
from utils.dice import dice_per_label
from utils.field import _warp


def ridge_weights(train_coarse: torch.Tensor, heldout_coarse: torch.Tensor, ridge: float) -> torch.Tensor:
    """Linear weights for min ||Y-XW||² + ridge*||W||²; no feature rescaling and an unpenalized intercept."""
    if train_coarse.ndim != 2 or heldout_coarse.ndim != 2 or train_coarse.shape[1] != heldout_coarse.shape[1]:
        raise ValueError("coarse features must be matching matrices with subjects along rows")
    if len(train_coarse) < 2 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("at least two training subjects and a finite positive ridge are required")
    train, heldout = train_coarse.double(), heldout_coarse.double()
    if not torch.isfinite(train).all() or not torch.isfinite(heldout).all():
        raise ValueError("coarse features must be finite")
    centred = train - train.mean(dim=0)
    gram = centred @ centred.T + ridge * torch.eye(len(train), device=train.device, dtype=torch.float64)
    weights = torch.linalg.solve(gram, centred @ (heldout - train.mean(dim=0)).T).T
    return weights - weights.mean(dim=1, keepdim=True) + 1 / len(train)


def replace_coarse(interior: torch.Tensor, coarse: torch.Tensor, levels: int) -> torch.Tensor:
    pyramid = haar_forward(interior, levels)
    if coarse.shape != pyramid.coarse.shape:
        raise ValueError("coarse tensor does not match the velocity pyramid")
    return haar_inverse(HaarPyramid(coarse.to(interior), pyramid.levels))


def select_ridge(features: torch.Tensor, target_gram: torch.Tensor, candidates: list[float], fine_count: int) -> dict:
    """Exact leave-one-subject-out coefficient MSE; only inner-Train arrays enter this choice."""
    count = len(features)
    if count < 3 or target_gram.shape != (count, count) or fine_count < 1:
        raise ValueError("ridge selection requires at least three regression-training subjects and a fine target Gram")
    if not torch.isfinite(target_gram).all():
        raise ValueError("fine target Gram must be finite")
    if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("ridge candidates must be finite and positive")
    scores = []
    for ridge in sorted(set(candidates), reverse=True):
        smoother = ridge_weights(features, features, ridge)
        denominator = 1 - smoother.diagonal()
        if bool((denominator <= 0).any()):
            scores.append({"ridge": ridge, "coefficient_mse": None})
            continue
        loo = smoother.clone()
        loo.fill_diagonal_(0)
        residual_weights = loo / denominator[:, None] - torch.eye(count, dtype=torch.float64, device=loo.device)
        error = (residual_weights * (residual_weights @ target_gram.double())).sum() / (count * fine_count)
        scores.append({"ridge": ridge, "coefficient_mse": max(0.0, float(error))})
    valid = [
        entry for entry in scores if entry["coefficient_mse"] is not None and math.isfinite(entry["coefficient_mse"])
    ]
    if not valid:
        raise ValueError("the supplied ridge grid has no numerically defined leave-one-out score")
    winner = min(valid, key=lambda entry: entry["coefficient_mse"])
    return {
        "method": "inner_train_leave_one_subject_out_coefficient_mse",
        "selected": winner["ridge"],
        "scores": scores,
        "tie_rule": "prefer larger ridge",
        "heldout_used": False,
    }


def _fine_gram(train: list[dict], collar: int, coarse_features: torch.Tensor) -> tuple[torch.Tensor, int]:
    arrays = [np.load(record["directory"] / "velocity.npy", allow_pickle=False, mmap_mode="r") for record in train]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("all fitted velocities must share the processed grid")
    gram = torch.zeros(len(train), len(train), dtype=torch.float64)
    # Work plane by plane: no stack of complete subject velocity volumes is retained.
    for z in range(collar, arrays[0].shape[2] - collar):
        plane = torch.from_numpy(
            np.stack([array[:, :, z, collar:-collar, collar:-collar].reshape(-1) for array in arrays])
        ).double()
        gram += plane @ plane.T
    interior_count = math.prod(size - 2 * collar for size in arrays[0].shape[2:]) * 3
    return gram - coarse_features.double() @ coarse_features.double().T, interior_count - coarse_features.shape[1]


def _read_fits(directories: list[Path]) -> list[dict]:
    records = []
    for directory in directories:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("kind") != "SUPERVISED_DIAGNOSTIC" or report.get("direction") != "fixed_to_moving":
            raise ValueError(f"{directory}: expected a label-fit diagnostic")
        arguments = report["arguments"]
        records.append(
            {
                "directory": directory,
                "subject": Path(arguments["fixed"]).resolve(),
                "template": Path(arguments["moving"]).resolve(),
                "arguments": arguments,
            }
        )
    return records


def validate_fits(train: list[dict], heldout: list[dict]) -> dict:
    if len(train) < 2 or not heldout:
        raise ValueError("need at least two training fits and one heldout fit")
    records = train + heldout
    subjects = [record["subject"] for record in records]
    if len(set(subjects)) != len(subjects):
        raise ValueError("training/heldout subjects must be distinct, without duplicate fits")
    template = records[0]["template"]
    if any(record["template"] != template for record in records) or template not in subjects[: len(train)]:
        raise ValueError("all fits must use one reference subject belonging to inner-Train")
    keys = ("steps", "lr", "regularization", "collar", "levels", "integration_steps", "smoothing_sigmas")
    settings = {key: records[0]["arguments"].get(key) for key in keys}
    if settings["levels"] < 1:
        raise ValueError("predictability requires at least one fine level")
    if any(any(record["arguments"].get(key) != value for key, value in settings.items()) for record in records):
        raise ValueError("all fits must share their label-fitting settings and Haar representation")
    return settings


def _interior(record: dict, collar: int) -> torch.Tensor:
    values = np.load(record["directory"] / "velocity.npy", allow_pickle=False, mmap_mode="r")
    if (
        values.dtype != np.float32
        or values.ndim != 5
        or values.shape[:2] != (1, 3)
        or min(values.shape[2:]) - 2 * collar < 2
    ):
        raise ValueError("stored velocity must have [1,3,D,H,W] shape and a nonempty interior")
    border = np.array(values, copy=True)
    border[:, :, collar:-collar, collar:-collar, collar:-collar] = 0
    if np.any(border != 0) or not np.isfinite(values).all():
        raise ValueError("stored velocity must be finite with the declared zero collar")
    return torch.from_numpy(np.array(values[:, :, collar:-collar, collar:-collar, collar:-collar], copy=True))


def _band_errors(predicted: torch.Tensor, reference: torch.Tensor, levels: int) -> dict:
    bands = coefficient_bands(haar_forward(predicted.double() - reference.double(), levels))
    fine = bands[1:]
    return {
        "coarse_mse": float(bands[0].square().mean()),
        "fine_mse": float(sum(band.square().sum() for band in fine) / sum(band.numel() for band in fine)),
        "fine_band_mse": [float(band.square().mean()) if band.numel() else None for band in fine],
    }


def evaluate_predictability(
    train_dirs: list[Path],
    heldout_dirs: list[Path],
    *,
    ridge: float | None = None,
    ridge_grid: list[float] | None = None,
    f_denominator_min: float | None = None,
) -> dict:
    """Predict fine bands by dual ridge, retaining each heldout subject's actual coarse band.

    Full-volume targets are streamed: linearity of Haar makes weighted training velocities
    followed by coarse replacement identical to independent ridge regressions of every fine coefficient.
    These label-driven maps and point substitutions are not an upper bound on an image-conditioned prior.
    """
    train, heldout = _read_fits(train_dirs), _read_fits(heldout_dirs)
    settings = validate_fits(train, heldout)
    if (ridge is None) == (ridge_grid is None):
        raise ValueError("supply either a fixed ridge or an inner-Train ridge grid")
    if f_denominator_min is not None and (not math.isfinite(f_denominator_min) or f_denominator_min <= 0):
        raise ValueError("f_denominator_min must be finite and positive, or omitted to suppress f")
    collar, levels = settings["collar"], settings["levels"]
    reference = next(record for record in train if record["subject"] == record["template"])
    train = [record for record in train if record["subject"] != record["template"]]
    if len(train) < 2:
        raise ValueError("need at least two regression-training subjects after excluding reference self-fit")
    reference_qc = {
        "subject": str(reference["subject"]),
        "directory": str(reference["directory"]),
        "velocity_rms": float(_interior(reference, collar).double().square().mean().sqrt()),
        "included_in_regression": False,
    }
    train_coarse = [haar_forward(_interior(record, collar).double(), levels).coarse for record in train]
    shape = train_coarse[0].shape
    if any(coarse.shape != shape for coarse in train_coarse):
        raise ValueError("all fitted velocities must share the processed grid")
    features = torch.stack([coarse.flatten() for coarse in train_coarse])
    if ridge_grid is not None:
        target_gram, fine_count = _fine_gram(train, collar, features)
        selection = select_ridge(features, target_gram, ridge_grid, fine_count)
        ridge = selection["selected"]
    else:
        selection = {"method": "fixed_before_heldout", "selected": ridge, "heldout_used": False}
    template = _load_labels(train[0]["template"], "cpu")
    unconditional_mean = torch.zeros_like(_interior(train[0], collar), dtype=torch.float64)
    for record in train:
        unconditional_mean += _interior(record, collar).double() / len(train)
    cases = []
    for record in heldout:
        actual = _interior(record, collar)
        coarse = haar_forward(actual, levels).coarse
        weights = ridge_weights(features, coarse.flatten()[None], ridge)[0]
        conditional = torch.zeros_like(unconditional_mean)
        for weight, source in zip(weights, train, strict=True):
            values = _interior(source, collar).double()
            if values.shape != actual.shape:
                raise ValueError("all fitted velocities must share the processed grid")
            conditional += values * weight
        unconditional = replace_coarse(unconditional_mean, coarse, levels).float()
        conditional = replace_coarse(conditional, coarse, levels).float()
        labels = _load_labels(record["subject"], "cpu")
        if labels.shape != template.shape or labels.shape[2:] != tuple(size + 2 * collar for size in actual.shape[2:]):
            raise ValueError("label maps and fitted velocities must share the processed grid")
        ids = sorted(set(template.unique().tolist()) | set(labels.unique().tolist()))
        ids = [label for label in ids if label > 0]
        if not ids:
            raise ValueError("a heldout diagnostic must contain foreground labels")
        readouts = {}
        for name, interior in (("unconditional", unconditional), ("conditional", conditional), ("full_fit", actual)):
            velocity = torch.nn.functional.pad(interior, (collar,) * 6)
            flow = canonical_flow(velocity, settings["integration_steps"])
            warped = _warp(template.float(), flow, mode="nearest").long()
            dice = dice_per_label(warped, labels, ids)
            readouts[name] = {
                "dice": dice.tolist(),
                "dice_mean": float(dice.mean()),
                **_band_errors(interior, actual, levels),
            }
        baseline = readouts["unconditional"]["dice_mean"]
        denominator = readouts["full_fit"]["dice_mean"] - baseline
        gain = readouts["conditional"]["dice_mean"] - baseline
        informative = f_denominator_min is not None and denominator > f_denominator_min
        cases.append(
            {
                "subject": str(record["subject"]),
                "label_ids": ids,
                "readouts": readouts,
                "dice_gain": gain,
                "f_denominator": denominator,
                "f": gain / denominator if informative else None,
                "f_status": "DESCRIPTIVE_ONLY" if informative else "UNINFORMATIVE_OR_NOT_REQUESTED",
            }
        )
    return {
        "kind": "SUPERVISED_DIAGNOSTIC",
        "certification": "NOT_RUN",
        "scope": "point substitutions in label-fitted geometry; no automatic hypothesis verdict or upper bound",
        "conditioning": "true coarsest Haar band; all detail bands predicted jointly",
        "relation_to_A1": "global linear cross-subject ridge and local nonlinear atlas-conditioned prior are different, non-nested function classes; neither result predicts the other",
        "dice_scope": "foreground union present in template/subject, not benchmark Dice35",
        "reference": str(train[0]["template"]),
        "train_subjects": [str(record["subject"]) for record in train],
        "reference_qc": reference_qc,
        "settings": settings,
        "ridge": ridge,
        "ridge_selection": selection,
        "f_denominator_min": f_denominator_min,
        "cases": cases,
    }


def local_predictability(train_dirs, heldout_dirs, options, output, *, epochs=20, lr=0.001, device="cpu"):
    """Fit the actual local hierarchical prior to supervised diagnostic velocities, with a fixed template."""
    train, heldout = _read_fits(train_dirs), _read_fits(heldout_dirs)
    settings = validate_fits(train, heldout)
    train = [row for row in train if row["subject"] != row["template"]]
    torch.manual_seed(0)
    model = PopulationAtlas(load_image(train[0]["template"], 1.0, device), **options).to(device)
    model.requires_grad_(False)
    for parameter in model.prior_parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(list(model.prior_parameters()), lr=lr)
    history = []
    for epoch in range(epochs):
        total = 0.0
        for i in torch.randperm(len(train)).tolist():
            coefficients = haar_forward(_interior(train[i], settings["collar"]).to(device), settings["levels"])
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prior(coefficients).mean() / model.raw_atlas.numel() + 1e-4 * model.weight_penalty() / (
                len(train) * model.raw_atlas.numel()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite local-prior loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.prior_parameters()), float("inf"), norm_type=float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "objective_per_voxel": total / len(train)})
        print(json.dumps(history[-1]), flush=True)
    cases = []
    with torch.no_grad():
        for record in heldout:
            actual = haar_forward(_interior(record, settings["collar"]).to(device), settings["levels"])
            values = coefficient_bands(actual)
            prefix, errors, offset = model._empty_bands(1), [], 0
            for stage in range(model.levels + 1):
                means, _ = model._prior_step(stage, prefix)
                selected = values[offset : offset + len(means)]
                if stage:
                    errors.append(
                        float(
                            sum((a.double() - b.double()).square().sum() for a, b in zip(selected, means, strict=True))
                            / sum(a.numel() for a in selected)
                        )
                    )
                prefix[offset : offset + len(means)] = selected
                offset += len(means)
            cases.append(
                {
                    "subject": str(record["subject"]),
                    "fine_nll_per_coefficient": float(
                        -model.log_prior(actual, include_coarse=False)[0] / sum(v.numel() for v in values[1:])
                    ),
                    "stage_mse": errors,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "options": options}, output / "prior.pth")
    report = {
        "arm": model.arm,
        "training_subjects": len(train),
        "history": history,
        "cases": cases,
        "scope": "supervised diagnostic with true previous scales; not image-trained A, not predicted-map Dice or a rejection gate",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, nargs="+", required=True, help="probe output folders, including the reference subject"
    )
    parser.add_argument(
        "--heldout", type=Path, nargs="+", required=True, help="probe output folders of distinct heldout subjects"
    )
    ridge_args = parser.add_mutually_exclusive_group(required=True)
    ridge_args.add_argument("--ridge", type=float, help="fixed before examining heldout results")
    ridge_args.add_argument(
        "--ridge-grid", type=float, nargs="+", help="select by inner-Train leave-one-subject-out coefficient MSE"
    )
    parser.add_argument(
        "--f-denominator-min", type=float, help="explicit positive minimum; omit to suppress all f ratios"
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON output file")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"use a fresh output file: {args.output}")
    report = evaluate_predictability(
        args.train, args.heldout, ridge=args.ridge, ridge_grid=args.ridge_grid, f_denominator_min=args.f_denominator_min
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps({"output": str(args.output), "heldout_subjects": len(report["cases"]), "verdict": "NOT_APPLICABLE"})
    )


if __name__ == "__main__":
    main()
