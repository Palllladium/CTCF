"""Matched IXI scratch training and controls for the source of the auxiliary contrast signal."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from scipy import ndimage

from datasets.IXI import read_ixi_manifest
from datasets.LPBA40 import read_lpba_manifest
from experiments.contrast_diagnostics import observe, read_csv
from experiments.contrasts import training_args
from experiments.core.data_loaders import ixi_paired_loaders
from experiments.core.distributed import restored_state_digest
from experiments.core.field_scoring import paired_dice_delta
from experiments.core.precision import configure_precision
from experiments.core.train_runtime import run_train
from experiments.train_CTCF import Runner
from tools.analysis.prepare_ixi import linear_matrix, processed_image
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import pkload, setup_device

KINDS = ("scratch_b0", "scratch_b1", "resampled_t1", "shuffled_t2", "b0", "b1", "normalised_t1")


def derangement(size, seed=20261114):
    if size < 2:
        raise ValueError("A shuffled contrast requires at least two subjects")
    order = np.random.default_rng(seed).permutation(size)
    assigned = np.empty(size, dtype=np.int64)
    assigned[order] = np.roll(order, -1)
    return assigned.tolist()


def resample_t1_control(t1, native_t2, geometry, processed_to_t1, t1_to_t2, *, normalisation_only=False):
    """Use only the native T2 grid: processed T1 -> native T2 grid -> processed grid."""
    for index, transform in enumerate((processed_to_t1, t1_to_t2)):
        matrix = linear_matrix(transform)
        if index == 1 and not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
            raise ValueError("The frozen raw T1-to-T2 transform must be rigid")
    fixed = processed_image(t1, geometry)
    pull = sitk.CompositeTransform([t1_to_t2, processed_to_t1])
    restored = t1
    if not normalisation_only:
        native = sitk.Resample(fixed, native_t2, pull.GetInverse(), sitk.sitkLinear, 0.0)
        restored = sitk.GetArrayFromImage(sitk.Resample(native, fixed, pull, sitk.sitkLinear, 0.0))
    coverage = sitk.Image(native_t2.GetSize(), sitk.sitkUInt8) + 1
    coverage.CopyInformation(native_t2)
    covered = sitk.GetArrayFromImage(sitk.Resample(coverage, fixed, pull, sitk.sitkNearestNeighbor, 0)) > 0
    mask = ndimage.binary_fill_holes(t1 > 0)
    if not np.isfinite(restored).all() or not (mask & covered).any():
        raise ValueError("Resampled T1 needs finite values and covered T1 support")
    lo, hi = np.percentile(restored[mask & covered], [1, 99])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Resampled T1 has no usable intensity range")
    result = np.where(mask, np.clip((restored - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    return result, {"percentiles": [float(lo), float(hi)], "coverage": float(covered[mask].mean())}


def historical_repair(folder, raw_rows):
    saved = json.loads((folder / "repair_summary.json").read_text(encoding="utf-8"))
    cases, expected = [row["case_id"] for row in saved["rows"]], [row["case_id"] for row in raw_rows]
    if (
        saved.get("certified") is not True
        or saved.get("fallbacks") != 0
        or saved.get("directions") != len(expected)
        or len(cases) != len(set(cases))
        or len(expected) != len(set(expected))
        or set(cases) != set(expected)
    ):
        raise ValueError(f"Historical repaired readout is incomplete, uncertified or uses fallback: {folder}")
    return saved["rows"]


def frozen_references(args, source_hash):
    source = json.loads((args.source / "inputs.json").read_text(encoding="utf-8"))
    if source["manifest_sha256"] != source_hash or source["parent_sha256"] != sha256_file(args.checkpoint):
        raise ValueError("Frozen controls use another parent or paired manifest")
    references = {}
    for arm in ("b0", "b1"):
        folder = args.source / f"{arm}_w1_s0"
        endpoint = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        expected = {"arm": arm, "seed": 0, "w_reg": 1, "updates": 1197, "manifest_sha256": source_hash}
        if any(endpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen controls must be the three-epoch w1 endpoints")
        historical_repair(folder, read_csv(folder / "epoch3/per_case.csv"))
        references[arm] = {
            "endpoint_sha256": sha256_file(folder / "report.json"),
            "FF_sha256": sha256_file(folder / "epoch3/per_case.csv"),
            "REP_summary_sha256": sha256_file(folder / "repair_summary.json"),
        }
    return references


def preflight(args):
    manifest = read_ixi_manifest(args.manifest, verify_bytes=True)
    source_hash = manifest["fingerprint"]
    references = frozen_references(args, source_hash)
    files = {str(args.manifest.resolve()): source_hash, str(args.checkpoint.resolve()): sha256_file(args.checkpoint)}
    files[str((args.source / "inputs.json").resolve())] = sha256_file(args.source / "inputs.json")
    for arm, path in (("b0", args.b0_checkpoint), ("b1", args.base_checkpoint)):
        endpoint = args.bm_source / f"{arm}_s1/report.json"
        saved = json.loads(endpoint.read_text(encoding="utf-8"))
        expected = {"arm": arm, "epochs": 10, "seed": 1, "w_reg": 1, "manifest_sha256": source_hash}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"BM {arm} is not the fixed ten-epoch seed1 endpoint")
        digest = sha256_file(path)
        if saved["checkpoint_sha256"] != digest:
            raise ValueError(f"BM {arm} checkpoint differs from its recorded SHA256")
        files[str(endpoint.resolve())], files[str(path.resolve())] = sha256_file(endpoint), digest
    external = read_lpba_manifest(args.external_manifest, verify_bytes=True)
    if (
        external.get("dataset") != "LPBA40"
        or len(external["subjects"]) != 40
        or len(external["pairs"]) != 120
        or external["shape"] != manifest["shape"]
        or external["label_ids"] != list(range(1, 57))
        or external["tensor_axes"] != "LIA"
    ):
        raise ValueError("LPBA must be the fixed forty-subject LIA evaluation dataset")
    files[str(args.external_manifest.resolve())] = external["fingerprint"]
    binding = {
        "files": files,
        "reference_controls": references,
        "external_subjects": 40,
        "external_directions": 120,
        "scope": "Input identities only; auxiliary raw-T2 preparation is a separate independent stage",
    }
    output = args.out / "preflight.json"
    if output.exists() and json.loads(output.read_text()) != binding:
        raise ValueError("Preflight input identities changed; use another RUN_ID")
    atomic_write_json(output, binding)


def prepare(args):
    manifest = read_ixi_manifest(args.manifest, verify_bytes=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    source_hash = manifest.pop("fingerprint")
    references = frozen_references(args, source_hash)
    raw_rows, assignments = [], derangement(len(manifest["training"]))
    for row in manifest["training"]:
        folder = Path(row["t2"]).parent
        qc_path = folder / "qc.json"
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        sources = [(p, digest) for p, digest in qc["binding"]["sources"].items() if p.lower().endswith("-t2.nii.gz")]
        if len(sources) != 1:
            raise ValueError(f"Cannot identify one raw T2 reference for {row['id']}")
        source, expected = sources[0]
        raw = args.raw_root / Path(source.replace("\\", "/")).name
        transforms = [folder / f"{name}.tfm" for name in ("processed_to_t1", "t1_to_t2")]
        if qc["id"] != row["id"] or qc["t1_sha256"] != row["t1_sha256"] or qc["t2_sha256"] != row["t2_sha256"]:
            raise ValueError(f"Prepared row and frozen QC disagree for {row['id']}")
        if sha256_file(raw) != expected or any(not p.is_file() for p in transforms):
            raise ValueError(f"Raw T2 reference or saved transforms missing/changed for {row['id']}")
        raw_rows.append((qc_path, qc, raw, transforms))
    binding = {
        "source_manifest_sha256": source_hash,
        "recipe": "ixi-target-source-controls-v1",
        "reference_controls": references,
        "shuffle_seed": 20261114,
        "shuffle": {
            row["id"]: manifest["training"][i]["id"] for row, i in zip(manifest["training"], assignments, strict=True)
        },
        "sources": {str(p.resolve()): sha256_file(p) for q, _, r, ts in raw_rows for p in (q, r, *ts)},
        "scope": "Training-only auxiliary targets; validation and T1 model inputs are unchanged. Resampling is not PSF matching.",
    }
    binding_path = args.out / "inputs.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Target-control sources changed; use a new output directory")
    atomic_write_json(binding_path, binding)
    altered = {kind: copy.deepcopy(manifest) for kind in ("resampled_t1", "shuffled_t2")}
    for index, (row, (qc_path, qc, raw, transforms)) in enumerate(zip(manifest["training"], raw_rows, strict=True)):
        destination = args.out / "targets/resampled_t1/Train" / row["id"] / "aux.npy"
        report_path = destination.with_suffix(".json")
        subject_binding = {
            "input_sha256": row["t1_sha256"],
            "sources": {str(p.resolve()): binding["sources"][str(p.resolve())] for p in (qc_path, raw, *transforms)},
        }
        previous = json.loads(report_path.read_text()) if report_path.exists() else None
        if previous and previous["binding"] != subject_binding:
            raise ValueError(f"Resampling sources changed for {row['id']}")
        if previous and destination.is_file():
            if sha256_file(destination) != previous["sha256"]:
                raise ValueError(f"Saved auxiliary target changed for {row['id']}")
        else:
            t1 = np.asarray(pkload(row["t1"])[0], dtype=np.float32)
            values, quality = resample_t1_control(
                t1,
                sitk.ReadImage(str(raw), sitk.sitkFloat32),
                qc["binding"]["geometry"],
                *[sitk.ReadTransform(str(p)) for p in transforms],
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(destination)
            atomic_write_json(
                report_path, {"binding": subject_binding, "sha256": sha256_file(destination), "quality": quality}
            )
        target = altered["resampled_t1"]["training"][index]
        target.update(t2=str(destination.resolve()), t2_sha256=sha256_file(destination))
        other = manifest["training"][assignments[index]]
        altered["shuffled_t2"]["training"][index].update(
            t2=other["t2"], t2_sha256=other["t2_sha256"], auxiliary_subject=other["id"]
        )
        print(f"[AUXILIARY TARGET] {index + 1}/{len(raw_rows)} {row['id']}", flush=True)
    for kind, document in altered.items():
        document.update(source_manifest_sha256=source_hash, target_control=kind)
        path = args.out / "targets" / kind / "manifest.json"
        atomic_write_json(path, document)
        read_ixi_manifest(path)
    atomic_write_json(
        args.out / "report.json",
        {
            "complete": True,
            "source_manifest_sha256": source_hash,
            "training_subjects": len(raw_rows),
            "validation_pairs": manifest["validation_pairs"],
        },
    )


def active_manifest(args):
    if args.kind.startswith("scratch_") or args.kind in ("b0", "b1"):
        return args.manifest
    if args.prepared is None:
        raise ValueError("Auxiliary controls require --prepared from the prepare stage")
    return args.prepared / "targets" / args.kind / "manifest.json"


def options_for(args):
    arm = "b0" if args.kind in ("scratch_b0", "b0") else "b1"
    options = training_args(args, arm)
    options.exp, options.ixi_pairs = args.exp, str(active_manifest(args))
    options.w_reg, options.seed, options.num_workers = 1, getattr(args, "seed", 0), args.workers
    options.ncc_mode = getattr(args, "ncc_mode", "legacy")
    if args.kind.startswith("scratch_"):
        options.init_ckpt, options.finetune, options.lr = "", 0, 1e-4
        options.max_epoch, options.schedule_max_epoch = 100, 100
    else:
        options.max_epoch = getattr(args, "epochs", 3)
    return options


class ReadoutRunner(Runner):
    def __init__(self, args, device, manifest, out):
        super().__init__(args, device)
        self.observation_manifest, self.observation_out = manifest, out

    def on_val_end(self, epoch, _dice, _jac):
        readouts = {5, 20, 50, 100} if not self.args.finetune else {1, 3, self.args.max_epoch}
        if self.args.finetune and getattr(self.args, "ncc_mode", "legacy") == "stable":
            readouts |= {self.args.max_epoch - 2, self.args.max_epoch - 1}
        if epoch + 1 in readouts:
            observe(self, self.observation_manifest, self.observation_out / f"epoch{epoch + 1}")


def train(args):
    options = options_for(args)
    source = read_ixi_manifest(args.manifest, require_t2=False)
    selected = read_ixi_manifest(active_manifest(args), require_t2=options.ixi_arm == "b1")
    if source["validation_pairs"] != selected["validation_pairs"] or source["validation"] != selected["validation"]:
        raise ValueError("Auxiliary controls changed validation images, labels or pairs")
    binding = {
        "kind": args.kind,
        "source_manifest_sha256": source["fingerprint"],
        "manifest_sha256": selected["fingerprint"],
        "parent_sha256": None if args.kind.startswith("scratch_") else sha256_file(args.checkpoint),
        "recipe": vars(options).copy(),
    }
    binding = json.loads(json.dumps(binding))
    path = args.out / "recipe.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and {k: v for k, v in previous.items() if k != "initial_model_sha256"} != binding:
        raise ValueError("Training output belongs to another recipe or inputs")
    checkpoint = Path("results") / args.exp / "ckpt/last.pth"
    if checkpoint.exists():
        options.init_ckpt, options.resume = "", str(checkpoint)
    device = setup_device(seed=options.seed, deterministic=False)
    configure_precision(args.precision, device)
    runner = ReadoutRunner(options, device, source, args.out)
    if checkpoint.exists():
        if not previous or "initial_model_sha256" not in previous:
            raise ValueError("Resume lacks the original model initialization fingerprint")
        binding["initial_model_sha256"] = previous["initial_model_sha256"]
    else:
        binding["initial_model_sha256"] = restored_state_digest(runner.model, runner.optimizer, device)["model"]
        if previous and previous.get("initial_model_sha256") != binding["initial_model_sha256"]:
            raise ValueError("Fresh initialization changed within the same output directory")
    atomic_write_json(path, binding)
    run_train(options, runner, build_loaders=ixi_paired_loaders)
    saved = torch.load(checkpoint, map_location="cpu")
    if int(saved["epoch"]) + 1 != options.max_epoch:
        raise ValueError("Training did not reach the fixed last checkpoint")
    atomic_write_json(
        args.out / "report.json",
        {
            **{k: v for k, v in binding.items() if k != "recipe"},
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "epochs": options.max_epoch,
            "updates": options.max_epoch * 399,
            "arguments": saved["arguments"],
            "endpoint": "last, without Dice checkpoint selection",
        },
    )


def cost(args):
    stride = int((args.out / "stride.txt").read_text().strip())
    if stride != 1:
        raise ValueError("Cost measurement requires native descriptors")
    profiles = {}
    for name, path in (
        ("ctcf", args.out / "ctcf_profile/report.json"),
        ("student", args.out / f"profile_student_{stride}/profile.json"),
        ("teacher", args.out / "profile_teacher/profile.json"),
    ):
        value = json.loads(path.read_text())
        seconds = value["update_seconds"]
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid measured {name} update time")
        if name != "ctcf" and not value.get("passed", False):
            raise ValueError(f"The {name} profile failed its technical checks")
        profiles[name] = {"file_sha256": sha256_file(path), "update_seconds": seconds}
    components = {
        "ctcf_training_hours": (2 * 39900 + 2 * 1197) * profiles["ctcf"]["update_seconds"] / 3600,
        "teacher_training_hours": 2 * 4000 * profiles["teacher"]["update_seconds"] / 3600,
        "student_training_hours": 5 * 4000 * profiles["student"]["update_seconds"] / 3600,
        "evaluation_repair_reserve_hours": 12.0,
    }
    estimate = sum(components.values())
    output = {
        "stride": stride,
        "profiles": profiles,
        "components_gpu_hours": components,
        "estimated_gpu_hours": estimate,
        "scope": "Measured update-time extrapolation plus an unmeasured fixed readout/repair reserve; descriptive only, with no GPU-hour gate or runtime deadline.",
        "student_scope": "Five student trainings; the smaller encoder is conservatively charged at the main student's measured update time. Half-resolution descriptors are an inference-only ablation of identical weights.",
    }
    atomic_write_json(args.out / "cost.json", output)


def report(args):
    comparisons, missing = {}, []
    evaluations = args.evaluations_root or args.out / "IXI"
    initials = {}
    for kind in KINDS[:2]:
        path = args.out / "registration" / kind / "report.json"
        if path.exists():
            initials[kind] = json.loads(path.read_text())["initial_model_sha256"]
        else:
            missing.append(str(path))
    same_initialization = len(initials) == 2 and len(set(initials.values())) == 1
    if len(initials) == 2 and not same_initialization:
        missing.append("Scratch B0/B1 did not share the same initial model")
    for mode in ("FF", "REP"):
        tables = {}
        for kind in KINDS[:4]:
            path = evaluations / kind / mode / "per_case.csv"
            if path.exists():
                tables[kind] = read_csv(path)
            else:
                missing.append(str(path))
        for arm in ("b0", "b1"):
            directory = args.source / f"{arm}_w1_s0"
            try:
                if mode == "FF":
                    tables[arm] = read_csv(directory / "epoch3/per_case.csv")
                else:
                    tables[arm] = historical_repair(directory, read_csv(directory / "epoch3/per_case.csv"))
            except (OSError, ValueError) as error:
                missing.append(str(error))
        contrasts = [("scratch_b1", "scratch_b0")]
        contrasts += [(arm, control) for arm in KINDS[:4] for control in ("b0", "b1")]
        for arm, control in contrasts:
            if arm in tables and control in tables:
                comparisons[f"{mode}_{arm}-{control}"] = paired_dice_delta(tables[arm], tables[control])
    atomic_write_json(
        args.out / "learning_summary.json",
        {
            "comparisons": comparisons,
            "missing": missing,
            "initial_model_sha256": initials,
            "same_scratch_initialization": same_initialization,
            "scope": "One initialization; descriptive paired validation. Source-control comparisons with direct B0/B1 are added from the frozen prior dispatch.",
        },
    )
    if missing:
        raise RuntimeError("Some final evaluations are incomplete; available comparisons were saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "prepare", "train", "cost", "report"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--source", type=Path, default=Path("results/contrasts/B_CHECK_20261110T211549Z"))
    parser.add_argument("--bm_source", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--b0_checkpoint", type=Path)
    parser.add_argument("--external_manifest", type=Path)
    parser.add_argument("--evaluations_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("/data/mooncake/P/IXI_raw"))
    parser.add_argument("--kind", choices=KINDS, default="scratch_b0")
    parser.add_argument("--exp")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ncc_mode", choices=("legacy", "stable"), default="legacy")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.stage not in ("report", "cost") and args.manifest is None:
        parser.error("--manifest is required")
    if args.stage == "prepare" and args.checkpoint is None:
        parser.error("prepare requires --checkpoint for the frozen reference binding")
    if args.stage == "train" and (not args.exp or (not args.kind.startswith("scratch_") and args.checkpoint is None)):
        parser.error("train requires --exp and auxiliary controls require --checkpoint")
    if args.stage == "preflight" and any(
        getattr(args, key) is None
        for key in ("checkpoint", "bm_source", "base_checkpoint", "b0_checkpoint", "external_manifest")
    ):
        parser.error("preflight requires the parent, BM source/endpoints and external manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    {"preflight": preflight, "prepare": prepare, "train": train, "cost": cost, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
