"""The first updates of an H1 matcher: Dice against identity, the chosen displacement per level and the separate losses.

python -m experiments.dispatch1.diag_h1_steps --mode D3 --lambda_file RUN/h1_prep_D3/lambda.json --out RUN/D3.json --paths 3 --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.core.field_scoring import PROFILE, warp_labels
from experiments.core.path_profiles import get_dataset_paths
from experiments.dispatch1.matcher_training import TrainingPairs, training_files
from experiments.dispatch1.train_multimatch import losses
from models.MultiMatch.model import MODES, MultiMatch, level_mismatch, level_smoothness, selected_displacement
from utils import pkload, setup_device
from utils.dice import dice_per_label
from utils.field import logdet_std_from_flow

ZERO_OFFSET = 0.25  # voxels of the level grid below which a chosen displacement counts as the zero offset
CHANGED = 0.5  # voxels of the level grid a chosen displacement must move to count as a changed peak


def dice35(moving_seg: torch.Tensor, fixed_seg: torch.Tensor, flow: torch.Tensor) -> float:
    warped = warp_labels(moving_seg, flow, "canonical")
    return float(np.mean(dice_per_label(warped, fixed_seg.long(), labels=PROFILE.labels)))


def gradient_norm(module: torch.nn.Module) -> float:
    norms = [q.grad.norm() for q in module.parameters() if q.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0


@torch.no_grad()
def snapshot(model: MultiMatch, pairs, previous) -> tuple[list[dict], list[list[torch.Tensor]]]:
    """Per test pair: Dice, SDlogJ (log(detJ + 3)), and per level the chosen displacement, its change since the
    previous snapshot, the level mismatch and smoothness."""
    model.eval()
    rows, chosen = [], []
    for k, (name, moving, fixed, moving_seg, fixed_seg) in enumerate(pairs):
        with torch.autocast(device_type=moving.device.type, dtype=torch.float16, enabled=moving.device.type == "cuda"):
            flow, states = model(moving, fixed)
        flow = flow.float()
        levels, picks = {}, []
        for level, coords, logmass in states:
            state = [(level, coords.float(), logmass.float())]
            displacement = selected_displacement(coords.float(), logmass.float())
            norm = displacement.norm(dim=1)
            entry = {
                "mean_displacement": float(norm.mean()),
                "zero_offset_fraction": float((norm < ZERO_OFFSET).float().mean()),
                "mismatch": float(level_mismatch(state, fixed, moving)),
                "smoothness": float(level_smoothness(state)),
            }
            if previous is not None:
                moved = (displacement - previous[k][len(picks)]).norm(dim=1) > CHANGED
                entry["changed_since_previous"] = float(moved.float().mean())
            levels[f"l{level}"] = entry
            picks.append(displacement)
        chosen.append(picks)
        rows.append(
            {
                "pair": name,
                "dice35": dice35(moving_seg, fixed_seg, flow),
                "sdlogj": logdet_std_from_flow(flow),
                "mean_displacement": float(flow.norm(dim=1).mean()),
                "levels": levels,
                "diagnostics": dict(model.diagnostics),
            }
        )
    model.train()
    return rows, chosen


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--lambda_file", required=True, help="lambda.json of the dispatch-1 preparation of this mode")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--at", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10, 20, 50], help="updates to snapshot after")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-9)
    p.add_argument("--pairs", type=int, default=3, help="first test pairs to snapshot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", type=int, nargs=3)
    p.add_argument("--limit_files", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = setup_device(args.gpu, seed=args.seed, deterministic=False)
    lam = json.loads(Path(args.lambda_file).read_text(encoding="utf-8"))["lambda"]
    torch.manual_seed(args.seed)
    model = MultiMatch(mode=args.mode).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stream = TrainingPairs(training_files(args), args.seed, 0, 2 * args.steps + 100, args.crop)

    pairs = []
    for path in sorted(Path(get_dataset_paths(args.paths, "OASIS")["val_dir"]).glob("*.pkl"))[: args.pairs]:
        arrays = [torch.from_numpy(np.ascontiguousarray(a)).to(device)[None, None] for a in pkload(path)]
        pairs.append((path.stem.removeprefix("p_"), arrays[0].float(), arrays[1].float(), arrays[2], arrays[3]))
    identity = {
        name: dice35(ms, fs, torch.zeros(1, 3, *mov.shape[2:], device=device)) for name, mov, _, ms, fs in pairs
    }
    record = {"mode": args.mode, "lambda": lam, "lr": args.lr, "seed": args.seed, "identity_dice35": identity}
    record["snapshots"], record["updates"] = [], []

    previous = None
    if 0 in args.at:
        rows, previous = snapshot(model, pairs, previous)
        record["snapshots"].append({"update": 0, "pairs": rows})
    loader = DataLoader(stream, batch_size=1, num_workers=2, pin_memory=device.type == "cuda")
    for update, (moving, fixed) in enumerate(loader, start=1):
        if update > args.steps:
            break
        moving, fixed = moving.to(device, non_blocking=True), fixed.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            similarity, regularity = losses(model, moving, fixed)
            loss = similarity + lam * regularity
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        row = {"update": update, "similarity": float(similarity.detach()), "regularity": float(regularity.detach())}
        row.update(encoder=gradient_norm(model.encoder), decoder=gradient_norm(model.decoder))
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        row["amp_skipped"] = bool(scaler.is_enabled() and scaler.get_scale() < scale)
        record["updates"].append(row)
        if update in args.at:
            rows, previous = snapshot(model, pairs, previous)
            record["snapshots"].append({"update": update, "pairs": rows})
            gap = np.mean([r["dice35"] - identity[r["pair"]] for r in rows])
            zero = {
                level: round(float(np.mean([r["levels"][level]["zero_offset_fraction"] for r in rows])), 3)
                for level in rows[0]["levels"]
            }
            print(f"[h1 {args.mode}] update {update}: Dice - identity {gap:+.4f}, zero offsets {zero}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"[h1 {args.mode}] {out}", flush=True)


if __name__ == "__main__":
    main()
