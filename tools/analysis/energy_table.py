"""Build (sharded) or merge the frozen parent's energies between every training subject and the bank."""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.core.bridges import Parent, energy, sub_bank
from experiments.core.path_profiles import get_dataset_paths
from utils import pkload, setup_device


def load_image(path) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(pkload(path)[0], dtype=np.float32))[None, None]


def build(args) -> None:
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("--shard must be in [0, --shards)")
    device = setup_device(args.gpu, seed=0, deterministic=False)
    files = sorted(glob.glob(os.path.join(get_dataset_paths(args.paths, "OASIS")["train_dir"], "*.pkl")))
    files = files[: args.limit_files or None]
    subjects = [Path(f).stem for f in files]
    bank = sub_bank(subjects) if args.bank == "sub" else subjects
    by_stem = dict(zip(subjects, files, strict=True))
    rows = subjects[args.shard :: args.shards]
    if len(subjects) < 2 or not rows:
        raise ValueError("energy table requires at least two training subjects and a nonempty shard")
    parent = Parent.load(args.ckpt, args.ctcf_config, bool(args.ctcf_l3_svf), args.time_steps, device)
    bank_images = {s: load_image(by_stem[s]) for s in bank}
    to_bank = np.zeros((len(rows), len(bank)), dtype=np.float32)
    from_bank = np.zeros((len(bank), len(rows)), dtype=np.float32)
    t0 = time.perf_counter()
    for i, a in enumerate(rows):
        image = load_image(by_stem[a]).to(device)
        for k, s in enumerate(bank):
            via = bank_images[s].to(device)
            to_bank[i, k] = energy(parent.flow(via, image), image)
            from_bank[k, i] = energy(parent.flow(image, via), via)
        print(f"[energy] {i + 1}/{len(rows)} {a} {time.perf_counter() - t0:.0f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        subjects=np.array(rows),
        bank=np.array(bank),
        to_bank=to_bank,
        from_bank=from_bank,
        ckpt=np.array(args.ckpt),
        config=np.array(args.ctcf_config),
        l3_svf=np.array(args.ctcf_l3_svf),
        time_steps=np.array(args.time_steps),
    )


def merge(out: str, parts: list[str]) -> None:
    if not parts:
        raise ValueError("give at least one table part to merge")
    loaded = []
    for path in parts:
        with np.load(path, allow_pickle=False) as part:
            loaded.append(dict(part))
    bank = [str(s) for s in loaded[0]["bank"]]
    if any([str(s) for s in d["bank"]] != bank for d in loaded):
        raise ValueError("table parts were built over different banks")
    metadata = ("ckpt", "config", "l3_svf", "time_steps")
    for key in metadata:
        values = [part.get(key) for part in loaded]
        if any(not np.array_equal(value, values[0]) for value in values[1:]):
            raise ValueError(f"table parts have different {key}")
    for part in loaded:
        shape = (len(part["subjects"]), len(bank))
        if part["to_bank"].shape != shape or part["from_bank"].shape != shape[::-1]:
            raise ValueError("table part has inconsistent energy dimensions")
        if not np.isfinite(part["to_bank"]).all() or not np.isfinite(part["from_bank"]).all():
            raise ValueError("table part contains non-finite energies")
    subjects = np.concatenate([d["subjects"] for d in loaded])
    if len(set(subjects)) != len(subjects) or len(set(bank)) != len(bank):
        raise ValueError("table parts contain duplicate subject rows or bank subjects")
    order = np.argsort(subjects)
    to_bank = np.concatenate([d["to_bank"] for d in loaded])[order]
    from_bank = np.concatenate([d["from_bank"] for d in loaded], axis=1)[:, order]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        subjects=subjects[order],
        bank=np.array(bank),
        to_bank=to_bank,
        from_bank=from_bank,
        ckpt=loaded[0]["ckpt"],
        config=loaded[0]["config"],
        **{key: loaded[0][key] for key in metadata[2:] if key in loaded[0]},
    )
    print(f"[energy] merged {len(parts)} parts: {len(subjects)} subjects x {len(bank)} bank -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--merge", nargs="+", metavar="PATH", help="OUT PART [PART ...]: merge table parts into OUT")
    p.add_argument("--ckpt")
    p.add_argument("--ctcf_config", default="CTCF-CascadeA-Mamba")
    p.add_argument("--ctcf_l3_svf", type=int, choices=[0, 1], default=0)
    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--bank", choices=["sub", "full"], default="sub")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit_files", type=int, default=0, help="smoke tests only")
    args = p.parse_args()
    if args.merge:
        merge(args.merge[0], args.merge[1:])
    else:
        if not args.ckpt or not args.out:
            p.error("--ckpt and --out are required when building a table")
        build(args)


if __name__ == "__main__":
    main()
