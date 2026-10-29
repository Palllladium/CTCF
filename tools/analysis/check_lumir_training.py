"""Exercise ramp/checkpoint/TERM/resume/export on the actual node before the LUMIR control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from datasets.LUMIR import LUMIR_SHAPE, lumir_disp_name, read_lumir_json
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.precision import configure_precision, precision_autocast
from experiments.export_lumir import _load_pair, evaluator_displacement
from tools.analysis.run_artifacts import atomic_write_json, sha256_file


def validate_export(export):
    manifest = json.loads((export / "manifest.json").read_text())
    _, pairs = read_lumir_json(Path(manifest["pairs_json"]))
    expected = {f"{lumir_disp_name(f, m)}.nii.gz" for f, m in pairs}
    if {row["file"] for row in manifest["files"]} != expected or len(manifest["files"]) != len(expected):
        raise ValueError("export inventory differs from the official pair list")
    archive = export / manifest["archive"]
    expected_members = {f"{archive.stem}/{name}" for name in expected}
    with zipfile.ZipFile(archive) as handle:
        if (
            handle.testzip() is not None
            or set(handle.namelist()) != expected_members
            or len(handle.namelist()) != len(expected)
        ):
            raise ValueError("submission ZIP is incomplete or corrupt")
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise ValueError("submission archive SHA mismatch")
    for row in manifest["files"]:
        path = export / archive.stem / row["file"]
        image = nib.load(path)
        if (
            image.shape != (*LUMIR_SHAPE, 3)
            or image.get_data_dtype() != np.dtype("float32")
            or not np.isfinite(image.get_fdata(dtype=np.float32)).all()
        ):
            raise ValueError(f"invalid exported displacement: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"displacement SHA mismatch: {path}")
    return manifest


def checkpoint_comparison(left, right):
    a, b = (torch.load(path, map_location="cpu", weights_only=False) for path in (left, right))
    for state in (a, b):
        if state["epoch"] != 3 or state["precision"] != "bf16" or state["training"]["global_batch_size"] != 4:
            raise ValueError("wrong restored epoch/precision/global batch")
        if state["training"]["world_size"] != 4 or len(state["training"]["rng"]) != 4:
            raise ValueError("per-rank RNG state missing")
        tensors = list(state["state_dict"].values()) + [
            v for row in state["optimizer"]["state"].values() for v in row.values() if isinstance(v, torch.Tensor)
        ]
        if any(not torch.isfinite(t).all() for t in tensors):
            raise FloatingPointError("nonfinite checkpoint state")
    steps = [{key: float(row["step"]) for key, row in state["optimizer"]["state"].items()} for state in (a, b)]
    if (
        not steps[0]
        or steps[0] != steps[1]
        or a["training"]["sampler_fingerprint"] != b["training"]["sampler_fingerprint"]
    ):
        raise ValueError("resume optimizer step counters or pair-stream fingerprint differ")
    numerator = sum(
        float((a["state_dict"][key].double() - b["state_dict"][key].double()).square().sum()) for key in a["state_dict"]
    )
    denominator = sum(float(v.double().square().sum()) for v in a["state_dict"].values())
    return {"model_relative_l2_difference": (numerator / max(denominator, 1e-30)) ** 0.5, "optimizer_steps": steps[0]}


def resume_agreement(continuous, repeated, resumed):
    errors = {}
    for name, other in (("continuous_repeat", repeated), ("resumed", resumed)):
        difference = continuous.astype(np.float64) - other
        errors[name] = {"rms": float(np.sqrt(np.mean(difference**2))), "maximum": float(np.max(np.abs(difference)))}
    limits = {"rms": 0.05, "maximum": 0.5}
    passes = {
        name: all(np.isfinite(values[key]) and values[key] <= limit for key, limit in limits.items())
        for name, values in errors.items()
    }
    status = "PASS"
    if not passes["continuous_repeat"]:
        status = "CONTINUOUS_BASELINE_UNSTABLE"
    elif not passes["resumed"]:
        status = "RESUME_MISMATCH"
    return {
        "status": status,
        "errors_voxels": errors,
        "limits_voxels": limits,
        "scope": "one continuous repeat measures descriptive numerical noise; limits are fixed, never enlarged from two runs",
    }


def exercise(run, gpus):
    if len(gpus.replace(",", " ").split()) != 4:
        raise ValueError("the production readiness check requires four GPUs")
    if os.name != "posix":
        raise RuntimeError("actual process-group interruption is exercised on the Linux training node")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus.replace(",", " ").split())
    active = None

    def stop(signum, _frame):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            active.wait(timeout=120)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run.mkdir(parents=True, exist_ok=True)
    base = run.parent.name
    # A failed readiness attempt is retained; a retry must exercise a fresh interruption.
    token = time.time_ns()
    experiments = [f"{base}_{token}_continuous", f"{base}_{token}_continuous_repeat", f"{base}_{token}_resumed"]
    env = os.environ | dict(
        PYBIN=sys.executable,
        GPUS=gpus,
        PROFILE="0",
        GLOBAL_BATCH="4",
        BATCH_SIZE="1",
        CONFIG="CTCF-CascadeA-Mamba",
        MODEL="--l3_svf 0",
        TRAIN="--lr 0.0001 --schedule_max_epoch 20",
        PRECISION="bf16",
        STAGES="ramp",
        EPOCHS="4",
        ITERS="10",
        PAIRS="lumir25",
        PRETESTS="0",
    )
    runner = ["bash", "tools/runners/train/lumir.sh"]
    exports = []
    for index, experiment in enumerate(experiments):
        final = Path("results") / experiment / "ckpt/last.pth"
        if not final.exists():
            with (run / f"{index}_initial.log").open("w") as log:
                proc = subprocess.Popen(
                    runner,
                    env=env | dict(EXP=experiment, RESUME="0"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active = proc
                try:
                    if index == 2:
                        deadline = time.monotonic() + 1200
                        source = Path("logs") / experiment / "logfile.log"
                        while proc.poll() is None and time.monotonic() < deadline:
                            if source.exists() and "[epoch 000]" in source.read_text(errors="replace"):
                                os.killpg(proc.pid, signal.SIGTERM)
                                break
                            time.sleep(0.2)
                        code = proc.wait(timeout=120)
                        if code not in (143, -signal.SIGTERM):
                            raise RuntimeError(f"intentional TERM did not interrupt training: {code}")
                        interrupted_epoch = torch.load(final, map_location="cpu", weights_only=False)["epoch"]
                        if not 0 <= interrupted_epoch < 3:
                            raise RuntimeError(
                                "TERM arrived after the full readiness training, not at a resumable intermediate epoch"
                            )
                        atomic_write_json(
                            run / "interrupted.json",
                            {
                                "epoch": interrupted_epoch,
                                "returncode": code,
                            },
                        )
                    elif proc.wait(timeout=1800) != 0:
                        raise RuntimeError("continuous readiness training failed")
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=120)
        if index == 2:
            if not (run / "interrupted.json").exists():
                raise RuntimeError("missing evidence of actual interruption")
            with (run / "resume.log").open("w") as log:
                proc = subprocess.Popen(
                    runner,
                    env=env | dict(EXP=experiment, RESUME="1"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active = proc
                try:
                    if proc.wait(timeout=1800) != 0:
                        raise RuntimeError("readiness continuation failed")
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=120)
        with (Path("logs") / experiment / "metrics.csv").open(newline="") as handle:
            epochs = list(csv.DictReader(handle))
        if [int(row["epoch"]) for row in epochs] != list(range(4)):
            raise ValueError("duplicated or missing resumed CSV epochs")
        alphas = [json.loads(row["loss_meters_json"])["alpha_l1"]["avg"] for row in epochs]
        if alphas != [0.0, 0.0, 0.5, 1.0]:
            raise ValueError("off/partial/full cascade stages were not exercised")
        export = max(
            (Path("results/lumir") / experiment / "attempts").glob("*/export_lumir25"), key=lambda p: p.stat().st_mtime
        )
        exports.append((export, validate_export(export)))
    comparison = checkpoint_comparison(
        *(Path("results") / e / "ckpt/last.pth" for e in (experiments[0], experiments[-1]))
    )
    baseline_comparison = checkpoint_comparison(*(Path("results") / e / "ckpt/last.pth" for e in experiments[:2]))
    name = exports[0][1]["files"][0]["file"]
    fields = [
        nib.load(path / manifest["archive"].removesuffix(".zip") / name).get_fdata(dtype=np.float32)
        for path, manifest in exports
    ]
    agreement = resume_agreement(*fields)
    atomic_write_json(run / "agreement.json", agreement)
    if agreement["status"] != "PASS":
        raise ValueError(f"LUMIR readiness: {agreement['status']}; see agreement.json")
    export, manifest = exports[-1]
    device = torch.device("cuda:0")
    model = get_model_adapter("ctcf").build(
        time_steps=6,
        config_key="CTCF-CascadeA-Mamba",
        img_size=LUMIR_SHAPE,
        dwin_size=volume_window_size(LUMIR_SHAPE),
        l3_svf=False,
    )
    load_checkpoint_state(model, str(Path("results") / experiments[-1] / "ckpt/last.pth"), strict=True)
    model = model.to(device).eval()
    configure_precision("bf16", device)
    entry = manifest["files"][0]
    val_root = Path(manifest["pairs_json"]).parent
    x, y = _load_pair(val_root, entry["fixed"], entry["moving"], device)
    with torch.no_grad(), precision_autocast("bf16", device):
        _, flow = model(x, y, alpha_l1=1.0, alpha_l3=1.0)
    reproduced = evaluator_displacement(flow)
    reproduction_error = float(np.max(np.abs(reproduced - fields[-1])))
    if reproduction_error > 1e-4:
        raise ValueError(f"same-checkpoint export reproduction differs by {reproduction_error} voxels")
    del model, flow, x, y
    torch.cuda.empty_cache()
    # Same 160 pair presentations and per-epoch stage progress, with fewer batch-8 optimizer updates.
    contrast = {"status": "ERROR", "scope": "short descriptive batch-size contrast, not long-run quality equivalence"}
    experiment = f"{base}_{token}_batch8"
    try:
        with (run / "batch8.log").open("w") as log:
            proc = subprocess.Popen(
                runner,
                env=env | dict(EXP=experiment, RESUME="0", GLOBAL_BATCH="8", ITERS="5"),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active = proc
            try:
                if proc.wait(timeout=1800) != 0:
                    raise RuntimeError("optional batch-8 ramp training failed")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=120)
        curves = {}
        for batch, exp in ((4, experiments[0]), (8, experiment)):
            with (Path("logs") / exp / "metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if [int(row["epoch"]) for row in rows] != list(range(4)) or [
                json.loads(row["loss_meters_json"])["alpha_l1"]["avg"] for row in rows
            ] != [0.0, 0.0, 0.5, 1.0]:
                raise ValueError("optional batch-8 ramp did not complete off/partial/full stages")
            curves[batch] = [
                {
                    "epoch": int(row["epoch"]),
                    "pairs_seen": (int(row["epoch"]) + 1) * 40,
                    "optimizer_updates": (int(row["epoch"]) + 1) * int(row["train_iterations"]),
                    "val_dice": float(row["val_dice"]),
                    "epoch_seconds": float(row["perf_epoch_time_sec"]),
                }
                for row in rows
            ]
        contrast.update(status="PASS", curves=curves, experiment=experiment)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        contrast["error"] = str(error)
        print(f"[LUMIR OPTIONAL BATCH8] {error}", flush=True)
    report = {
        "status": "PASS",
        "experiments": experiments,
        "agreement": agreement,
        "continuous_repeat_checkpoint": baseline_comparison,
        "reproduction_max": reproduction_error,
        "export_pairs": len(manifest["files"]),
        "batch_contrast": contrast,
        **comparison,
        "scope": "short operational test, not bitwise CUDA equivalence or long-run convergence",
    }
    atomic_write_json(run / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    arguments = parser.parse_args()
    exercise(arguments.run, arguments.gpus)
