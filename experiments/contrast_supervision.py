"""A fixed anatomical-loss control using the existing paired IXI trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.IXI import IXIPairedDataset, IXIPairSampler, read_ixi_manifest
from experiments.contrast_diagnostics import observe
from experiments.contrast_learning import options_for
from experiments.contrasts import sync
from experiments.core.data_loaders import ixi_paired_loaders
from experiments.core.distributed import restored_state_digest
from experiments.core.precision import configure_precision, precision_autocast
from experiments.core.train_runtime import run_train
from experiments.train_CTCF import Runner, _gradient_ratio, paired_ixi_dice
from experiments.weight_interventions import freeze, read, save_weights, verify
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import setup_device


def options(args, coefficient):
    settings = SimpleNamespace(
        **(vars(args) | {"kind": "b1", "epochs": 10, "precision": "bf16", "ncc_mode": "stable", "workers": 8})
    )
    options = options_for(settings)
    options.w_dice = coefficient
    options.ixi_label_ids = str(args.out / "discovery_ids.txt")
    return options


def labelled_pairs(args):
    manifest = read_ixi_manifest(args.manifest)
    ids = read(args.out / "splits.json")["subjects"]["discovery"]
    dataset = IXIPairedDataset(manifest, "b1", label_ids=ids)
    index = {row["id"]: i for i, row in enumerate(dataset.rows)}
    for moving, fixed in read(args.out / "splits.json")["pairs"]["discovery"][:8]:
        batch = dataset[index[moving], index[fixed]]
        yield [value[None].to(args.device) for value in batch[:-1]] + [torch.tensor([batch[-1]], device=args.device)]


def calibrate(args):
    inputs = verify(args)
    path = args.out / "supervision/calibration.json"
    if path.is_file():
        saved = read(path)
        if (
            saved["splits_sha256"] != sha256_file(args.out / "splits.json")
            or saved["parent_sha256"] != sha256_file(args.checkpoint)
            or saved["initial_model_sha256"] != inputs["initial"]["b1_s0"]
        ):
            raise ValueError("Frozen Dice calibration belongs to other inputs")
        return
    setup_device(seed=0, deterministic=False)
    runner = Runner(options(args, 1), args.device)
    reference = inputs["initial"]["b1_s0"]
    initial = restored_state_digest(runner.model, runner.optimizer, args.device)["model"]
    if initial != reference:
        raise ValueError("S does not start from the same expanded parent as stable B1")
    runner.model.eval()
    ratios = []
    runner.args.w_dice = 0
    for batch in labelled_pairs(args):
        x, y, tx, ty, xs, ys, available = batch
        with precision_autocast("bf16", args.device):
            base, _, xy, yx = runner._pair_loss(x, y, None, None, 0, t2=(tx, ty))
        auxiliary = paired_ixi_dice(xy, yx, xs, ys, available, runner.reg_bilin)
        ratios.append(_gradient_ratio(base, auxiliary, (xy, yx)))
    if len(ratios) != 8 or not np.isfinite(ratios).all() or min(ratios) <= 0:
        raise RuntimeError("Dice calibration did not produce eight finite positive ratios")
    freeze(
        args.out / "supervision/calibration.json",
        {
            "coefficient": float(np.median(ratios)),
            "ratios": ratios,
            "initial_model_sha256": initial,
            "splits_sha256": sha256_file(args.out / "splits.json"),
            "parent_sha256": sha256_file(args.checkpoint),
            "scope": "One fixed coefficient: median norm(base gradient)/norm(unit Dice gradient), with respect to both full-resolution legacy flow tensors, on eight fixed discovery directions; no optimizer step or Dice-quality sweep",
        },
    )


def profile_loaders(options):
    train, validation = ixi_paired_loaders(options)
    dataset = train.dataset
    dataset.rows = [row for row in dataset.rows if row["id"] in dataset.label_ids]
    sampler = IXIPairSampler(len(dataset), options.seed, "supervised-memory-profile", samples=20)
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0), validation


class SupervisionRunner(Runner):
    def __init__(self, args, device, manifest, out):
        super().__init__(args, device)
        self.observation_manifest, self.observation_out = manifest, out
        self.discovery_manifest = dict(manifest)
        self.discovery_manifest["validation"] = manifest["training"]
        self.discovery_manifest["validation_pairs"] = (
            read(out.parent / "splits.json")["pairs"]["discovery"] if out.name == "supervision" else []
        )
        self.labelled_updates = 0

    def train_step(self, batch, epoch):
        self.labelled_updates += int(batch[6].sum())
        return super().train_step(batch, epoch)

    def on_val_end(self, epoch, _dice, _jac):
        if self.args.max_epoch == 10 and epoch + 1 in (1, 3, 10):
            observe(self, self.observation_manifest, self.observation_out / f"epoch{epoch + 1}")
            observe(self, self.discovery_manifest, self.observation_out / f"discovery_epoch{epoch + 1}")
            if epoch + 1 in (1, 3):
                path = self.observation_out / "weights" / f"epoch{epoch + 1}.pth"
                header = {
                    "arguments": vars(self.args),
                    "precision": self.args.precision,
                    "ncc_mode": self.args.ncc_mode,
                    "input_representation": self.model.input_representation,
                }
                save_weights(path, header, self.model.state_dict())
                atomic_write_json(
                    self.observation_out / f"epoch{epoch + 1}/checkpoint.json",
                    {
                        "epoch": epoch + 1,
                        "checkpoint": str(path),
                        "sha256": sha256_file(path),
                        "scope": "Diagnostic snapshot; fitting resumes only from native last.pth",
                    },
                )


def fit(args):
    inputs = verify(args)
    calibration = read(args.out / "supervision/calibration.json")
    if calibration["splits_sha256"] != sha256_file(args.out / "splits.json"):
        raise ValueError("Discovery partition changed after Dice calibration")
    config = options(args, calibration["coefficient"])
    directory = args.out / ("supervision/profile" if args.stage == "profile" else "supervision")
    binding = {
        "arguments": vars(config).copy(),
        "calibration_sha256": sha256_file(args.out / "supervision/calibration.json"),
        "inputs_sha256": sha256_file(args.out / "inputs.json"),
    }
    freeze(directory / "recipe.json", json.loads(json.dumps(binding)))
    if args.stage == "profile":
        config.max_epoch, config.max_train_iters, config.max_val_batches, config.save_ckpt = 1, 20, 1, 0
        config.num_workers = 0
    checkpoint = Path("results") / args.exp / "ckpt/last.pth"
    if checkpoint.exists() and args.stage != "profile":
        config.init_ckpt, config.resume = "", str(checkpoint)
    setup_device(seed=0, deterministic=False)
    manifest = read_ixi_manifest(args.manifest)
    # Readouts include both discovery and the old Val curve; neither chooses training length or checkpoint.
    runner = SupervisionRunner(config, args.device, manifest, directory)
    if not config.resume:
        initial = restored_state_digest(runner.model, runner.optimizer, args.device)["model"]
        if initial != inputs["initial"]["b1_s0"]:
            raise ValueError("S initialization differs from the matched unsupervised baseline")
    counter, measured = 0, {}
    tick = None
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)

    def timing(_optimizer, _args, _kwargs):
        nonlocal counter, tick
        counter += 1
        if counter == 5:
            tick = sync(args.device)
        if counter == 20:
            measured["update_seconds"] = (sync(args.device) - tick) / 15

    runner.optimizer.register_step_post_hook(timing)
    run_train(config, runner, build_loaders=profile_loaders if args.stage == "profile" else ixi_paired_loaders)
    if args.stage == "profile":
        if counter != 20 or runner.labelled_updates != 20:
            raise RuntimeError("The full-volume profile must exercise twenty labelled updates")
        measured.update(
            updates=counter,
            labelled_updates=runner.labelled_updates,
            peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2**30
            if args.device.type == "cuda"
            else None,
            scope="All pairs labelled for worst-case one-hot/warp memory; five warmup updates excluded; not a time cap",
        )
        atomic_write_json(directory / "report.json", measured)
        print(f"[SUPERVISION PROFILE] {measured}", flush=True)
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != 9:
        raise RuntimeError("S did not reach fixed last10")
    sampler = IXIPairSampler(397, 0, "pair-order-only", samples=399)
    ids = set(read(args.out / "splits.json")["subjects"]["discovery"])
    expected = 0
    for epoch in range(10):
        sampler.set_epoch(epoch)
        expected += sum(
            manifest["training"][a]["id"] in ids and manifest["training"][b]["id"] in ids for a, b in sampler
        )
    atomic_write_json(
        directory / "report.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "updates": 3990,
            "labelled_updates_full_recipe": expected,
            "labelled_updates_this_attempt": runner.labelled_updates,
            "initial_model_sha256": inputs["initial"]["b1_s0"],
            "endpoint": "last10; no validation selection or extension",
            "supervision": "Dice uses IXI30 foreground labels only when both subjects are discovery237; all 397 remain in T1/T2 training",
            "coefficient": config.w_dice,
            "trajectory": [read(directory / f"epoch{epoch}/checkpoint.json") for epoch in (1, 3)]
            + [{"epoch": 10, "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint)}],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "profile", "train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exp", required=True)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision("bf16" if args.device.type == "cuda" else "fp32", args.device)
    (calibrate if args.stage == "calibrate" else fit)(args)


if __name__ == "__main__":
    main()
