"""Short real-chain check and fixed Train contrast interventions for the MIND-input pilot."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from datasets.LUMIR import LUMIR_INTENSITY_MAX, LUMIR_SHAPE, load_lumir_volume, read_lumir_json
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.path_profiles import resolve_lumir_paths
from experiments.core.precision import configure_precision, precision_autocast
from experiments.train_CTCF import Runner, parse_args
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import Grad3d, NCCVxm, RegisterModel, setup_device
from utils.field import legacy_sampling_displacement


def contrast_image(image, mode):
    if mode == "original":
        return image
    foreground = image > 0
    if mode == "invert":
        changed = 1 - image
    elif mode == "gain_bias":
        changed = 0.5 * image + 0.25
    else:
        changed = image.clamp_min(0).pow({"gamma_half": 0.5, "gamma_two": 2}[mode])
    return torch.where(foreground, changed, image)


def image(root, name, device):
    values = load_lumir_volume(root / name) / LUMIR_INTENSITY_MAX
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Nonfinite training image: {name}")
    return torch.from_numpy(values)[None, None].to(device)


def check(args):
    paths = resolve_lumir_paths(args)
    training, _ = read_lumir_json(paths["train_json"])
    options = parse_args([])
    vars(options).update(
        ds="LUMIR",
        paths=3,
        config="CTCF-CascadeA-Mamba",
        l3_svf=0,
        input_representation="mind6",
        precision="bf16",
        time_steps=6,
        max_epoch=1,
        finetune=1,
        w_reg=1,
        exp="MIND_CHECK",
    )
    runner = Runner(options, args.device)
    x, y = [image(paths["train_root"], name, args.device) for name in training[:2]]
    losses = []
    runner.model.train()
    torch.cuda.reset_peak_memory_stats(args.device)
    for _ in range(3):
        runner.optimizer.zero_grad(set_to_none=True)
        with precision_autocast("bf16", args.device):
            loss, _ = runner.train_step((x, y), 0)
        if not torch.isfinite(loss):
            raise FloatingPointError("MIND real-chain check has a nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), float("inf"), error_if_nonfinite=True)
        runner.optimizer.step()
        losses.append({"loss": float(loss.detach()), "gradient_norm": float(norm)})
    checkpoint = args.out / "check.pth"
    torch.save(
        {"state_dict": runner.model.state_dict(), "precision": "bf16", "input_representation": "mind6"}, checkpoint
    )
    peak = torch.cuda.max_memory_allocated(args.device) / 2**30
    del runner, x, y, loss
    gc.collect()
    torch.cuda.empty_cache()
    export = args.out / "export"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.export_lumir",
            "--ckpt",
            str(checkpoint),
            "--out",
            str(export),
            "--paths",
            "3",
            "--config",
            "CTCF-CascadeA-Mamba",
            "--l3_svf",
            "0",
            "--input_representation",
            "mind6",
            "--time_steps",
            "6",
            "--precision",
            "bf16",
            "--pairs",
            "lumir25",
            "--certified",
            "1",
            "--limit_pairs",
            "1",
            "--name",
            "mind_check",
        ],
        check=True,
    )
    report = json.loads((export / "manifest.json").read_text())
    if report["certified"]["global_homeomorphism_pairs"] != 1 or len(report["files"]) != 1:
        raise ValueError("MIND one-pair chain has no serialized global certificate")
    atomic_write_json(
        args.out / "check.json",
        {
            "updates": losses,
            "peak_allocated_gib": peak,
            "checkpoint_sha256": sha256_file(checkpoint),
            "export": report,
            "scope": "Three actual full-cascade updates and one serialized certified export; not a quality result",
        },
    )


@torch.no_grad()
def contrasts(args):
    paths = resolve_lumir_paths(args)
    training, _ = read_lumir_json(paths["train_json"])
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(20261112)).tolist()[:16]
    pairs = [(training[a], training[b]) for a, b in zip(order[::2], order[1::2], strict=True)]
    results = {
        "pairs": pairs,
        "models": {},
        "scope": "Train intensity interventions; no official anatomical accuracy claim",
    }
    warp, ncc = RegisterModel(LUMIR_SHAPE).to(args.device), NCCVxm(win=(9, 9, 9))
    for representation, checkpoint in (("raw", args.raw_checkpoint), ("mind6", args.checkpoint)):
        model = (
            get_model_adapter("ctcf")
            .build(
                config_key="CTCF-CascadeA-Mamba",
                l3_svf=False,
                input_representation=representation,
                time_steps=6,
                img_size=LUMIR_SHAPE,
                dwin_size=volume_window_size(LUMIR_SHAPE),
            )
            .to(args.device)
            .eval()
        )
        load = load_checkpoint_state(model, str(checkpoint), strict=True)
        rows = []
        for moving, fixed in pairs:
            x, y = [image(paths["train_root"], name, args.device) for name in (moving, fixed)]
            for same_subject in (False, True):
                target, reference = (x if same_subject else y), None
                for mode in ("original", "gamma_half", "gamma_two", "invert", "gain_bias"):
                    with precision_autocast("bf16", args.device):
                        _, field = model(contrast_image(x, mode), target)
                    displacement = legacy_sampling_displacement(field.float())
                    if reference is None:
                        reference = displacement
                    change = displacement - reference
                    rows.append(
                        {
                            "moving": moving,
                            "fixed": moving if same_subject else fixed,
                            "same_subject": same_subject,
                            "intensity": mode,
                            "field_change_rms": float(change.square().mean().sqrt()),
                            "field_change_max": float(change.abs().max()),
                            "identity_error_rms": float(displacement.square().mean().sqrt()) if same_subject else None,
                            "regularity": float(Grad3d(penalty="l2")(displacement)),
                            "NCC_original_images": float(ncc(warp((x, field.float())), target)),
                        }
                    )
            print(f"[LUMIR CONTRAST] {representation} {moving} {fixed}", flush=True)
        results["models"][representation] = {"checkpoint_sha256": sha256_file(checkpoint), "load": load, "rows": rows}
        del model
        torch.cuda.empty_cache()
    atomic_write_json(args.out / "contrast_sensitivity.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "contrasts"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--raw-checkpoint", type=Path)
    args = parser.parse_args()
    args.paths, args.device = 3, setup_device(seed=0)
    configure_precision("bf16", args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    {"check": check, "contrasts": contrasts}[args.stage](args)


if __name__ == "__main__":
    main()
