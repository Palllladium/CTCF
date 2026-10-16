"""Known-warp diagnostics: the generator's set statistics (--stats) or a CTCF checkpoint's endpoint error (--ckpt).

The endpoint error is split by the clean target's structure weight into a normal and a tangential part. It measures
synthetic correspondence, not anatomical Dice.

    python -m experiments.dispatch1.eval_known_warp --stats RUN/stats.json --paths 3 --gpu 0
    python -m experiments.dispatch1.eval_known_warp --ckpt CKPT --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0 --out RUN/epe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.core.path_profiles import get_dataset_paths
from experiments.dispatch1.bridges import Parent
from experiments.dispatch1.known_warp import KnownWarpSource, effective_points, estimate_stats, structure_weights
from utils import setup_device

HELD_OUT_SEED = 900_001  # disjoint from the training and calibration streams


def endpoint_errors(parent: Parent, files, device, pairs: int) -> dict:
    source = KnownWarpSource(files, device, seed=HELD_OUT_SEED, target="point")
    totals = {"epe": [], "normal": [], "tangential": [], "valid_fraction": []}
    for _ in range(pairs):
        pair = source.sample()
        directions = (
            (pair.moving, pair.fixed, pair.clean_moving, pair.fm),
            (pair.fixed, pair.moving, pair.clean_fixed, pair.mf),
        )
        for moving, fixed, clean_target, target in directions:
            error = effective_points(parent.flow(moving, fixed), target.anchors) - target.points
            weight = structure_weights(clean_target, target.points)
            normal2 = torch.einsum("ni,nij,nj->n", error, weight, error).clamp_min(0)
            total2 = (error**2).sum(-1)
            totals["epe"].append(float(total2.sqrt().mean()))
            totals["normal"].append(float(normal2.sqrt().mean()))
            totals["tangential"].append(float((total2 - normal2).clamp_min(0).sqrt().mean()))
        totals["valid_fraction"].append(pair.valid_fraction)
    return {key: sum(values) / len(values) for key, values in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stats", help="output JSON of the generator's set statistics")
    mode.add_argument("--ckpt", help="CTCF checkpoint to measure")
    p.add_argument("--out", help="output JSON of the endpoint error")
    p.add_argument("--ctcf_config", default="CTCF-CascadeA-Mamba")
    p.add_argument("--ctcf_l3_svf", type=int, choices=[0, 1], default=0)
    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--limit_files", type=int, default=0)
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    if args.ckpt and not args.out:
        p.error("--ckpt needs --out")
    device = setup_device(args.gpu, seed=args.seed, deterministic=False)
    files = sorted(Path(get_dataset_paths(args.paths, "OASIS")["train_dir"]).glob("*.pkl"))[: args.limit_files or None]
    if args.stats:
        result, out = estimate_stats(files, device, args.samples, args.seed), Path(args.stats)
    else:
        parent = Parent.load(args.ckpt, args.ctcf_config, bool(args.ctcf_l3_svf), args.time_steps, device)
        result = endpoint_errors(parent, files, device, args.pairs)
        result.update(pairs=args.pairs, seed=HELD_OUT_SEED, ckpt=args.ckpt, load=parent.load_report)
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"[known_warp] {result}", flush=True)


if __name__ == "__main__":
    main()
