"""Test pairs of a frozen CTCF parent registered directly, by a second pass and through one training subject.

Minimax takes the bank subject whose harder leg has the least energy and falls back to the direct field when no path
eases it; proxy and random bridges take two passes. Every field keeps the parent's legacy sampling. Labels score the
outputs and the descriptive FF oracle; they never select a bridge.

    python -m experiments.dispatch1.eval_bridge --ckpt CKPT --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0 --out RUN/h3 \
        --check_ff HISTORICAL_FF/per_case.csv --paths 3 --gpu 0

--check_only stops after checking that the direct field reproduces the historical FF table (mean within 1e-4, every
pair within 1e-3) and writes check.json with the measured cost of one pass and of scoring; a failed check exits 3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from experiments.core.field_scoring import PROFILE, score_ff_rep, warp_labels
from experiments.core.inference_metrics import write_results
from experiments.core.path_profiles import get_dataset_paths
from experiments.dispatch1.bridges import (
    PROXY_FIT_FILES,
    SUB_BANK_SIZE,
    Parent,
    compose_bridge,
    csf_threshold,
    energy,
    nearest_proxy,
    sub_bank,
    ventricle_proxy,
)
from utils import pkload, setup_device
from utils.dice import dice_per_label
from utils.field import compose_flows, legacy_sampling_displacement

VENTRICLES = (3, 4, 11, 12, 22, 23)  # lateral, inferior lateral, 3rd and 4th ventricles: descriptive checks only
CHECK_MEAN, CHECK_PAIR = 1e-4, 1e-3
EXIT_CHECK = 3
ARMS = ("direct", "recursion", "bridge_minimax", "bridge_minimax64", "bridge_proxy", "bridge_random", "oracle_ff")


def dice35(flow, x_seg, y_seg) -> float:
    warped = warp_labels(x_seg, flow, "legacy").long()
    return float(np.mean(dice_per_label(warped, y_seg.long(), labels=PROFILE.labels)))


def synced(device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def load_pair(path, device):
    x, y, x_seg, y_seg = pkload(path)

    def to(a, dtype):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)[None, None]

    return to(x, torch.float32), to(y, torch.float32), to(x_seg, torch.long), to(y_seg, torch.long)


def load_bank(files):
    """Bank images in FP32 on the host (moved to the device one at a time) and their label ventricle volumes."""
    images, label_volumes = [], []
    for path in files:
        data = pkload(path)
        img, seg = data if len(data) == 2 else (data[0], data[2])
        images.append(torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None])
        label_volumes.append(int(np.isin(seg, VENTRICLES).sum()))
    return images, label_volumes


def check_direct(dice_by_case: dict, csv_path: str, allow_subset: bool = False) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as h:
        reference = {r["case_id"]: float(r["dice_mean"]) for r in csv.DictReader(h)}
    ours = set(dice_by_case)
    if not ours or ours - set(reference) or (not allow_subset and ours != set(reference)):
        return {"passed": False, "reason": "case ids differ from the reference table", "reference": csv_path}
    gaps = np.array([dice_by_case[c] - reference[c] for c in sorted(ours)])
    mean_gap, pair_gap = float(abs(gaps.mean())), float(np.abs(gaps).max())
    passed = mean_gap <= CHECK_MEAN and pair_gap <= CHECK_PAIR
    return {"passed": passed, "mean_gap": mean_gap, "max_pair_gap": pair_gap, "reference": csv_path}


def write_table(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_check(args, parent: Parent, pairs, bank_size: int, device, out: Path) -> dict:
    """The direct field against the historical table. A failed check, or --check_only, writes the report: the check
    and the measured seconds of one pass, of FF+REP scoring and of one oracle Dice."""
    direct_ff, pass_seconds = {}, []
    for case_id, path in pairs:
        x, y, x_seg, y_seg = load_pair(path, device)
        t0 = synced(device)
        flow = parent.flow(x, y)
        pass_seconds.append(synced(device) - t0)
        direct_ff[case_id] = dice35(flow, x_seg, y_seg)
    check = check_direct(direct_ff, args.check_ff, args.limit_test > 0) if args.check_ff else {"passed": None}
    print(f"[bridge] direct-field check: {check}", flush=True)
    if check["passed"] is not False and not args.check_only:
        return check
    report = {
        "check": check,
        "seconds_per_pass": float(np.median(pass_seconds[1:] or pass_seconds)),
        "load": parent.load_report,
        "test_pairs": len(pairs),
        "bank_subjects": bank_size,
        "sub_bank_subjects": min(SUB_BANK_SIZE, bank_size),
    }
    if check["passed"] is not False:
        t0 = synced(device)
        score_ff_rep(flow, case_id, 0.0, x_seg, y_seg, "legacy")
        report["score_seconds"] = synced(device) - t0
        t0 = synced(device)
        dice35(compose_bridge(flow, flow), x_seg, y_seg)
        report["oracle_score_seconds"] = synced(device) - t0
    name = "check.json" if args.check_only else "arms.json"
    (out / name).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return check


def run_bridges(args, parent: Parent, pairs, bank_files, device, out: Path, check: dict) -> None:
    bank, label_volumes = load_bank(bank_files)
    stems = [Path(f).stem for f in bank_files]
    threshold = csf_threshold([img[0, 0].numpy() for img in bank[:PROXY_FIT_FILES]])
    proxy = {s: ventricle_proxy(img[0, 0].numpy(), threshold) for s, img in zip(stems, bank, strict=True)}
    in_sub = set(sub_bank(stems))
    sub_index = [i for i, s in enumerate(stems) if s in in_sub]
    rows = {arm: {"FF": [], "REP": []} for arm in ARMS}
    seconds = {arm: [] for arm in ARMS}
    bridged = {"bridge_minimax": 0, "bridge_minimax64": 0}
    bank_rows, selection_rows, gaps, leg_seconds = [], [], [], []
    for k, (case_id, path) in enumerate(pairs):
        x, y, x_seg, y_seg = load_pair(path, device)
        t0 = synced(device)
        direct = parent.flow(x, y)
        t_direct = synced(device) - t0
        e_direct = energy(direct, y)
        t0 = synced(device)
        second = parent.flow(parent.warp(x, direct), y)
        fields = {
            "direct": (direct, t_direct),
            "recursion": (compose_flows(legacy_sampling_displacement(second), direct), t_direct + synced(device) - t0),
        }

        path_scores, dices, t_bank, t_sub, t_oracle = [], [], 0.0, 0.0, 0.0
        for s, host_image in enumerate(bank):
            t0 = synced(device)
            via = host_image.to(device)
            f_fs, f_sm = parent.flow(via, y), parent.flow(x, via)
            e_fs, e_sm = energy(f_fs, y), energy(f_sm, via)
            elapsed = synced(device) - t0
            t_bank += elapsed
            t_sub += elapsed if stems[s] in in_sub else 0.0
            leg_seconds.append(elapsed / 2)
            path_scores.append(max(e_fs, e_sm))
            t0 = synced(device)
            dices.append(dice35(compose_bridge(f_fs, f_sm), x_seg, y_seg))
            t_oracle += synced(device) - t0
            bank_rows.append(
                {
                    "case_id": case_id,
                    "bank_subject": stems[s],
                    "in_sub_bank": int(stems[s] in in_sub),
                    "e_fs": e_fs,
                    "e_sm": e_sm,
                    "path_score": path_scores[-1],
                    "proxy": proxy[stems[s]],
                    "dice35_ff": dices[-1],
                }
            )
        pair_proxy = {"F": ventricle_proxy(y[0, 0].cpu().numpy(), threshold)}
        pair_proxy["M"] = ventricle_proxy(x[0, 0].cpu().numpy(), threshold)
        choices = {
            "bridge_minimax": int(np.argmin(path_scores)),
            "bridge_minimax64": min(sub_index, key=lambda i: path_scores[i]),
            "bridge_proxy": stems.index(nearest_proxy({**proxy, **pair_proxy}, "F", "M", stems)),
            "bridge_random": random.Random(args.seed * 1000 + k).randrange(len(bank)),
            "oracle_ff": int(np.argmax(dices)),
        }
        search_cost = {
            "bridge_minimax": t_direct + t_bank,
            "bridge_minimax64": t_direct + t_sub,
            "oracle_ff": t_bank + t_oracle,
        }
        for arm, s in choices.items():
            use = path_scores[s] < e_direct if arm in bridged else True
            if arm in bridged:
                bridged[arm] += int(use)
            t0 = synced(device)
            via = bank[s].to(device)
            field = compose_bridge(parent.flow(via, y), parent.flow(x, via)) if use else direct
            fields[arm] = (field, synced(device) - t0 + search_cost.get(arm, 0.0))
            selection_rows.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "subject": stems[s],
                    "bridged": int(use),
                    "path_score": path_scores[s],
                    "e_direct": e_direct,
                    "dice35_ff": dices[s],
                }
            )
        for arm, (field, cost) in fields.items():
            ff, rep = score_ff_rep(field, case_id, cost, x_seg, y_seg, "legacy")
            rows[arm]["FF"].append(ff)
            rows[arm]["REP"].append(rep)
            seconds[arm].append(cost)
        vent = [max(int(torch.isin(seg, torch.tensor(VENTRICLES, device=device)).sum()), 1) for seg in (x_seg, y_seg)]
        gaps.append((abs(math.log(vent[0] / vent[1])), e_direct))
        scores = " ".join(f"{arm} {rows[arm]['REP'][-1]['dice_mean']:.4f}" for arm in ARMS)
        print(f"[bridge] {k + 1}/{len(pairs)} {case_id} REP {scores}", flush=True)

    summary = {}
    test_dir = str(pairs[0][1].parent)
    for arm in ARMS:
        for scoring in ("FF", "REP"):
            folder = out / arm / scoring
            folder.mkdir(parents=True, exist_ok=True)
            write_results(rows[arm][scoring], str(folder), f"bridge_{arm}", args.ckpt, test_dir)
        summary[arm] = {
            "FF": float(np.mean([r["dice_mean"] for r in rows[arm]["FF"]])),
            "REP": float(np.mean([r["dice_mean"] for r in rows[arm]["REP"]])),
            "seconds_per_pair": float(np.mean(seconds[arm])),
            **({"bridged_pairs": bridged[arm]} if arm in bridged else {}),
        }
    write_table(out / "bridges.csv", bank_rows)
    write_table(out / "selection.csv", selection_rows)
    proxy_values = {"csf_threshold": threshold, "fit_files": PROXY_FIT_FILES, "values": proxy}
    (out / "proxy_values.json").write_text(json.dumps(proxy_values, indent=1), encoding="utf-8")
    diagnostics = {
        "spearman_e_direct_vs_label_ventricle_gap": float(stats.spearmanr(*zip(*gaps, strict=True))[0])
        if len(gaps) > 2
        else None,
        "spearman_proxy_vs_label_ventricle_volume": float(stats.spearmanr(list(proxy.values()), label_volumes)[0]),
        "seconds_per_pass": float(np.median(leg_seconds)),
        "bank": len(bank),
        "sub_bank": len(sub_index),
    }
    parent_info = {
        "ckpt": args.ckpt,
        "config": args.ctcf_config,
        "l3_svf": args.ctcf_l3_svf,
        "load": parent.load_report,
    }
    report = {"check": check, "parent": parent_info, "arms": summary, "diagnostics": diagnostics}
    (out / "arms.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"[bridge] done: {json.dumps({a: [v['FF'], v['REP']] for a, v in summary.items()})}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ctcf_config", default="CTCF-CascadeA-Mamba")
    p.add_argument("--ctcf_l3_svf", type=int, choices=[0, 1], default=0)
    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--out", required=True)
    p.add_argument("--check_ff", default="", help="historical per-case FF table of this parent")
    p.add_argument("--check_only", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="random bridges")
    p.add_argument("--limit_test", type=int, default=0)
    p.add_argument("--limit_bank", type=int, default=0)
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = setup_device(args.gpu, seed=args.seed, deterministic=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = get_dataset_paths(args.paths, "OASIS")
    test_files = sorted(Path(data["val_dir"]).glob("*.pkl"))[: args.limit_test or None]
    bank_files = sorted(Path(data["train_dir"]).glob("*.pkl"))[: args.limit_bank or None]
    if not test_files or len(bank_files) < 3:
        raise RuntimeError("bridge evaluation needs test pairs and at least three training subjects")
    parent = Parent.load(args.ckpt, args.ctcf_config, bool(args.ctcf_l3_svf), args.time_steps, device)
    print(f"[bridge] parent {args.ckpt}: {parent.load_report}", flush=True)
    pairs = [(f.stem.removeprefix("p_"), f) for f in test_files]
    check = run_check(args, parent, pairs, len(bank_files), device, out)
    if check["passed"] is False:
        sys.exit(EXIT_CHECK)
    if not args.check_only:
        run_bridges(args, parent, pairs, bank_files, device, out, check)


if __name__ == "__main__":
    main()
