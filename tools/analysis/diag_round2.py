"""Round-2 stage-0 checks for the learned registration similarity, before any registration training.

features   per feature net: collapse, localisation under a known deformation (against 9^3-patch NCC), and at
           P10 fields, per mode, the field gradient of the feature term; where both modes fit, the agreement of
           E(I o phi) with E(I) o phi on the feature levels' own grids (relative gap, cosine of the field
           gradients). Also the share of anchors with NCC-hard negatives by local intensity spread.
cost       time and peak memory of full training steps (AMP, scaled backward, Adam update) for NCC alone and
           each net and mode.
calibrate  lambda per net and mode: the feature term's network-gradient norm equals SHARE x the NCC term's at the
           seed-0 initial state on fixed training pairs (the state every stage-1 run starts from).
gate       technical pass/fail per net, the extraction mode by the pre-registered rule, and its lambda.
regime     where the P10 and J15 fields differ: by local intensity spread of the fixed image and by structure.
tto        diagnostic only: per-pair optimisation from P10 and J15 with NCC and NCC + lambda x feature term; with
           --anatomix also the anatomix similarity alone, weighted to NCC's field-gradient scale at P10.

    python -m tools.analysis.diag_round2 --parts features,cost,calibrate,gate --out results/round2/diag \
        --arm HARD=<ckpt> --arm RAND=<ckpt> --arm AE=<ckpt> --paths 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiments.core.cli_common import add_common_args
from experiments.core.cli_ctcf import add_ctcf_train_args
from experiments.train_CTCF import Runner
from experiments.train_features import (
    HARD_DEFAULTS,
    foreground_points,
    inside_volume,
    mine_ncc_hard,
    patch_ncc,
    patches,
    random_displacement,
    warp_index,
)
from tools.analysis.diag_round1 import Ctx, dice35, fold_percent, groups, summarise, warp_seg, write_csv
from utils import ctcf_schedule, pkload, setup_device
from utils.anatomy import anatomix_features, load_anatomix
from utils.feature_similarity import LearnedSimilarity, level_points, load_feature_net, sample_at, warped_level_maps
from utils.losses import Grad3d, local_ncc_map

# Fixed before the stage-0 runs.
SHARE = 1.0  # feature-term / NCC-term network-gradient norm at the initial state
CALIBRATION_PAIRS = 8
GATE_LOCALISATION_VOX = 3.0  # contrastive nets: median error of the embedding match under a known warp
GATE_AE_RECON_DROP = 0.5  # autoencoder: last logged reconstruction loss <= this fraction of the first
# A coarse filter, not evidence that features are informative: the autoencoder's LeakyReLU activations share a
# positive offset, so random voxel pairs sit near cosine 0.94 without any collapse.
GATE_COLLAPSE_COS = 0.99
GATE_TIME_RATIO = 2.5  # chosen mode's step time / NCC-only step time
GATE_MEMORY_FRACTION = 0.85  # chosen mode's peak memory / device memory
MODE_TIME_SAVING = 0.30  # warped_features only if this much faster than warped_image ...
MODE_GRAD_COS = 0.90  # ... with field gradients agreeing this well (median) ...
MODE_DISCREPANCY = 0.20  # ... and features within this relative gap of E(I o phi) (median over pairs)
RULES = {
    "SHARE": SHARE,
    "CALIBRATION_PAIRS": CALIBRATION_PAIRS,
    "GATE_LOCALISATION_VOX": GATE_LOCALISATION_VOX,
    "GATE_AE_RECON_DROP": GATE_AE_RECON_DROP,
    "GATE_COLLAPSE_COS": GATE_COLLAPSE_COS,
    "GATE_TIME_RATIO": GATE_TIME_RATIO,
    "GATE_MEMORY_FRACTION": GATE_MEMORY_FRACTION,
    "MODE_TIME_SAVING": MODE_TIME_SAVING,
    "MODE_GRAD_COS": MODE_GRAD_COS,
    "MODE_DISCREPANCY": MODE_DISCREPANCY,
}
SEARCH_RADIUS = 6
MODES = ("warped_image", "warped_features")
STRUCTURE_GROUPS = {
    "deep_gm": (7, 8, 9, 10, 14, 15, 16, 17, 26, 27, 28, 29, 30, 31, 32, 33),
    "cortex": (2, 21),
    "white_matter": (1, 20),
    "cerebellum": (5, 6, 24, 25),
    "ventricles": (3, 11, 12, 22),
    "brainstem": (13,),
    "small6": (4, 18, 19, 23, 34, 35),
}


class Stage0:
    """Test pairs (diag_round1.Ctx), the feature nets of --arm, and the stage-1 training recipe."""

    def __init__(self, args):
        self.args = args
        self.ctx = Ctx(args)
        self.out = self.ctx.out
        self.device = self.ctx.device
        self.arms = dict(item.split("=", 1) for item in args.arm)
        self.nets = {}

    def net(self, name):
        if name not in self.nets:
            self.nets[name] = load_feature_net(self.arms[name], self.device)
        return self.nets[name]

    def training_args(self, extra):
        """The stage-1 recipe through the training CLI (synthetic volume size under --smoke)."""
        p = argparse.ArgumentParser()
        add_common_args(p, include_synth=True)
        p.set_defaults(exp="R2_STAGE0")
        add_ctcf_train_args(p)
        recipe = ["--config", "CTCF-CascadeA-VM-Unified", "--l3_svf", "1", "--seed", "0", "--max_epoch", "100"]
        recipe += ["--w_ncc", "1.0", "--w_icon", "0.05", "--w_reg", "1.0", "--w_jac", "0.005", "--gpu", "0"]
        data = ["--ds", "OASIS", "--paths", str(self.args.paths)]
        if self.args.smoke:
            data = ["--ds", "SYNTH", "--synth_vol_size", *map(str, self.ctx.shape)]
        return p.parse_args(recipe + data + extra)

    def runner(self, extra):
        """A stage-1 trainer at the seed-0 initial state."""
        setup_device(gpu_id=self.args.gpu, seed=0)
        return Runner(self.training_args(extra), self.device)

    def learned_extra(self, name, mode, weight=1.0):
        flags = ["--anatomy_loss", "learned", "--feature_ckpt", self.arms[name], "--feature_mode", mode]
        return [*flags, "--w_anat", str(weight)]

    def training_batches(self, count):
        """Fixed (x, y, x_seg, y_seg) pairs of training images (seed 0), cropped like the test pairs."""
        files = sorted(Path(self.ctx.train_dir).glob("*.pkl"))
        rng = np.random.default_rng(0)
        batches = []
        for _ in range(count):
            i, j = rng.choice(len(files), size=2, replace=False)
            (xi, xs), (yi, ys) = (pkload(str(files[k]))[:2] for k in (i, j))
            batches.append((self.ctx.vol(xi), self.ctx.vol(yi), self.ctx.vol(xs, True), self.ctx.vol(ys, True)))
        return batches


def embed_at(maps, shape, points):
    """Unit-norm multi-level embedding at full-resolution index points [P, 3] -> [P, C]."""
    levels = [F.normalize(sample_at(fmap, level_points(points, shape, fmap.shape[2:])), dim=-1) for fmap in maps]
    return torch.cat(levels, dim=-1) / math.sqrt(len(levels))


def known_warp_anchors(image, count, generator):
    """A known smooth warp of `image` and anchors of the warped view with their true points in `image`."""
    d = random_displacement(tuple(image.shape[2:]), 6.0, generator, image.device)
    view = warp_index(image, d)
    anchors = foreground_points(view, count, generator)
    index = anchors.long()
    truth = anchors + d[0, :, index[:, 0], index[:, 1], index[:, 2]].T
    keep = inside_volume(truth, image.shape[2:])
    return view, anchors[keep], truth[keep]


def localisation(net, image, generator, chunk=32):
    """Error (voxels) of the best match within SEARCH_RADIUS of the truth under a known warp: embedding vs NCC."""
    view, anchors, truth = known_warp_anchors(image, 256, generator)
    r = torch.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1, device=image.device, dtype=torch.float32)
    offsets = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    shape = tuple(image.shape[2:])
    errors = {"embedding": [], "ncc": []}
    with torch.no_grad():
        maps_image, maps_view = net.feature_maps(image), net.feature_maps(view)
        anchor_e = embed_at(maps_view, shape, anchors)
        for start in range(0, len(anchors), chunk):
            stop = start + chunk
            cand = truth[start:stop].round()[:, None, :] + offsets
            cand_e = embed_at(maps_image, shape, cand.reshape(-1, 3)).reshape(*cand.shape[:2], -1)
            by_embedding = (cand_e * anchor_e[start:stop, None, :]).sum(dim=-1).argmax(dim=1)
            by_ncc = patch_ncc(patches(image, cand), patches(view, anchors[start:stop])[:, None, :]).argmax(dim=1)
            for key, pick in (("embedding", by_embedding), ("ncc", by_ncc)):
                hit = cand[torch.arange(len(cand), device=cand.device), pick]
                errors[key].append((hit - truth[start:stop]).norm(dim=-1))
    return {key: torch.cat(value) for key, value in errors.items()}


def mode_gradient(net, st, x, y, flow, mode):
    """Field gradient of the feature term (weight 1) in one mode at `flow`, or None if it runs out of memory."""
    flow = flow.detach().clone().requires_grad_(True)
    try:
        value = LearnedSimilarity(net, mode)(x, st(x, flow) if mode == "warped_image" else None, y, flow)
        (grad,) = torch.autograd.grad(value, flow)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return grad


def feature_gap(net, st, x, y, flow):
    """Mean over levels of the relative gap between E(I o phi) and E(I) o phi on each level's grid, over the fixed
    image's foreground."""
    gaps = []
    with torch.no_grad():
        maps = zip(net.feature_maps(st(x, flow)), warped_level_maps(net.feature_maps(x), flow), strict=True)
        for of_warped, warped_of in maps:
            mask = (F.interpolate((y > 0).float(), size=of_warped.shape[2:], mode="area") > 0.5).expand_as(of_warped)
            gap = (of_warped - warped_of)[mask].norm() / of_warped[mask].norm().clamp_min(1e-30)
            gaps.append(float(gap))
    return float(np.mean(gaps))


def _median(values):
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def part_features(s: Stage0):
    ctx = s.ctx
    generator = torch.Generator().manual_seed(0)
    flows = ctx.flows("P10") if "P10" in ctx.models() else None
    images = [y for _, _, y, _, _ in ctx.pairs[:4]]
    st = ctx.st(ctx.shape)
    rows = []
    for name in s.arms:
        net = s.net(name)
        stds, coss, errors = [], [], {"embedding": [], "ncc": []}
        for image in images:
            with torch.no_grad():
                fmap = net.feature_maps(image)[0]
            mask = F.interpolate((image > 0).float(), size=fmap.shape[2:], mode="area")[0, 0] > 0.5
            vectors = F.normalize(fmap[0][:, mask].T, dim=-1)
            pick = torch.randint(len(vectors), (2, 2000), generator=generator).to(vectors.device)
            stds.append(float(fmap[0][:, mask].std(dim=1).mean()))
            coss.append(float((vectors[pick[0]] * vectors[pick[1]]).sum(dim=-1).mean()))
            for key, value in localisation(net, image, generator).items():
                errors[key].append(value)
        errors = {key: torch.cat(value) for key, value in errors.items()}

        finite = dict.fromkeys(MODES, flows is not None)
        norms, coss_modes, gaps = [], [], []
        for (_, x, y, _, _), flow in list(zip(ctx.pairs, flows or [], strict=False))[:6]:
            grads = {mode: mode_gradient(net, st, x, y, flow, mode) for mode in MODES}
            for mode, grad in grads.items():
                finite[mode] = finite[mode] and grad is not None and bool(torch.isfinite(grad).all())
            if grads["warped_image"] is not None:
                norms.append(float(grads["warped_image"].norm()))
            if all(g is not None for g in grads.values()):
                g_image, g_features = grads["warped_image"], grads["warped_features"]
                coss_modes.append(float((g_image * g_features).sum() / (g_image.norm() * g_features.norm())))
                gaps.append(feature_gap(net, st, x, y, flow))
        rows.append(
            {
                "arm": name,
                "channel_std": float(np.mean(stds)),
                "random_pair_cos": float(np.mean(coss)),
                "local_median_vox": float(errors["embedding"].median()),
                "local_p90_vox": float(errors["embedding"].quantile(0.9)),
                "ncc_local_median_vox": float(errors["ncc"].median()),
                "ncc_local_p90_vox": float(errors["ncc"].quantile(0.9)),
                "finite_warped_image": finite["warped_image"],
                "finite_warped_features": finite["warped_features"],
                "mode_grad_cos_median": _median(coss_modes),
                "mode_feature_gap_median": _median(gaps),
                "grad_norm_median": _median(norms),
            }
        )
        print(f"[features] {rows[-1]}")
    write_csv(s.out / "stage0_features.csv", rows)

    settings = SimpleNamespace(**HARD_DEFAULTS)
    coverage_rows = []
    for image in images:
        view, anchors, truth = known_warp_anchors(image, 512, generator)
        _, valid, _ = mine_ncc_hard(image, view, anchors, truth, settings, generator)
        spread = patches(view, anchors).std(dim=-1)
        tercile = torch.bucketize(spread, spread.quantile(torch.tensor([1 / 3, 2 / 3], device=spread.device)))
        for t, label in enumerate(("flat", "medium", "textured")):
            members = tercile == t
            coverage_rows.append({"tercile": label, "with_hard": float(valid[members].any(dim=1).float().mean())})
    coverage = summarise(coverage_rows, ("tercile",), ("with_hard",))
    write_csv(s.out / "stage0_hard_coverage.csv", coverage)
    for entry in coverage:
        print(f"[features] anchors with NCC-hard negatives, {entry['tercile']:8s} spread: {entry['with_hard']:.3f}")


def _step_cost(runner, batch, repeats=5):
    """Median seconds and peak GiB of full late-epoch training steps as experiments.core.train_runtime runs them:
    AMP forward, scaled backward, Adam update (its state included in the peak)."""
    epoch = runner.args.max_epoch - 1
    scaler = torch.amp.GradScaler("cuda")
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(repeats + 2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runner.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss, _ = runner.train_step(batch, epoch)
        scaler.scale(loss).backward()
        scaler.step(runner.optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t0)
    return float(np.median(times)), torch.cuda.max_memory_allocated() / 2**30


def part_cost(s: Stage0):
    batch = s.training_batches(1)[0]
    total = torch.cuda.get_device_properties(s.device).total_memory / 2**30
    base_time, base_peak = _step_cost(s.runner([]), batch)
    rows = [{"arm": "NCC", "mode": "-", "seconds": base_time, "peak_gib": base_peak, "device_gib": total}]
    for name in s.arms:
        for mode in MODES:
            try:
                seconds, peak = _step_cost(s.runner(s.learned_extra(name, mode)), batch)
            except torch.cuda.OutOfMemoryError:
                print(f"[cost] {name} {mode}: out of memory")
                seconds, peak = float("inf"), float("inf")
            rows.append({"arm": name, "mode": mode, "seconds": seconds, "peak_gib": peak, "device_gib": total})
            torch.cuda.empty_cache()
    for row in rows:
        row["time_ratio"] = row["seconds"] / base_time
        print(
            f"[cost] {row['arm']:6s} {row['mode']:16s} {row['seconds']:.3f}s x{row['time_ratio']:.2f} "
            f"peak {row['peak_gib']:.1f} GiB of {total:.0f}"
        )
    write_csv(s.out / "stage0_cost.csv", rows)


def _parameter_grad_norm(value, params):
    grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))


def _gradient_ratios(runner, batches):
    """Per pair: ||grad NCC term|| / ||grad feature term (weight 1)|| over the network at its initial state."""
    model, args = runner.model, runner.args
    alpha_l1, alpha_l3, _ = ctcf_schedule(epoch=0, max_epoch=args.max_epoch)
    params = [p for p in model.parameters() if p.requires_grad]
    ratios = []
    for x, y, _, _ in batches:
        def_xy, flow_xy = model(x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
        def_yx, flow_yx = model(y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
        l_ncc = 0.5 * (runner.ctx.ncc(def_xy, y) + runner.ctx.ncc(def_yx, x)) * args.w_ncc
        l_feat, _ = runner._anatomy_loss(x, y, def_xy, def_yx, flow_xy, flow_yx, 0)
        ratios.append(_parameter_grad_norm(l_ncc, params) / max(_parameter_grad_norm(l_feat, params), 1e-30))
    return ratios


def part_calibrate(s: Stage0):
    batches = s.training_batches(CALIBRATION_PAIRS)
    result = {"share": SHARE, "pairs": CALIBRATION_PAIRS, "lambda": {}, "ratios": {}}
    for name in s.arms:
        for mode in MODES:
            key = f"{name}/{mode}"
            try:
                ratios = _gradient_ratios(s.runner(s.learned_extra(name, mode)), batches)
            except torch.cuda.OutOfMemoryError:
                print(f"[calibrate] {key}: out of memory")
                ratios = [float("nan")]
            result["ratios"][key] = ratios
            result["lambda"][key] = SHARE * float(np.median(ratios))
            print(f"[calibrate] {key}: lambda {result['lambda'][key]:.4g} (ratios {np.round(ratios, 3).tolist()})")
            torch.cuda.empty_cache()
    (s.out / "lambda.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def _read_csv(path):
    def parse(value):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return {"True": True, "False": False}.get(value, value)

    with open(path, newline="", encoding="utf-8") as handle:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def part_gate(s: Stage0):
    features = {row["arm"]: row for row in _read_csv(s.out / "stage0_features.csv")}
    cost = {(row["arm"], row["mode"]): row for row in _read_csv(s.out / "stage0_cost.csv")}
    lambdas = json.loads((s.out / "lambda.json").read_text(encoding="utf-8"))["lambda"]
    gate = {"rules": RULES, "arms": {}}
    for name, checkpoint in s.arms.items():
        f = features[name]
        kind = torch.load(checkpoint, map_location="cpu", weights_only=False)["kind"]
        reasons = []
        if f["random_pair_cos"] >= GATE_COLLAPSE_COS:
            reasons.append(f"collapsed features (random-pair cosine {f['random_pair_cos']:.3f})")
        if kind == "contrastive" and f["local_median_vox"] > GATE_LOCALISATION_VOX:
            reasons.append(f"localisation median {f['local_median_vox']:.2f} vox")
        if kind == "autoencoder":
            history = json.loads((Path(checkpoint).parent / "summary.json").read_text(encoding="utf-8"))["history"]
            if history[-1]["loss"] > GATE_AE_RECON_DROP * history[0]["loss"]:
                reasons.append("reconstruction loss did not halve")
        image_cost, feature_cost = cost[(name, "warped_image")], cost[(name, "warped_features")]
        faster = feature_cost["seconds"] <= (1.0 - MODE_TIME_SAVING) * image_cost["seconds"]
        agrees = f["mode_grad_cos_median"] >= MODE_GRAD_COS and f["mode_feature_gap_median"] <= MODE_DISCREPANCY
        mode = "warped_features" if faster and agrees and f["finite_warped_features"] else "warped_image"
        chosen = cost[(name, mode)]
        if not f[f"finite_{mode}"]:
            reasons.append(f"non-finite or missing {mode} field gradients at P10")
        if chosen["time_ratio"] > GATE_TIME_RATIO:
            reasons.append(f"{mode} step x{chosen['time_ratio']:.2f} the NCC step")
        if chosen["peak_gib"] > GATE_MEMORY_FRACTION * chosen["device_gib"]:
            reasons.append(f"{mode} peak {chosen['peak_gib']:.1f} GiB")
        weight = lambdas[f"{name}/{mode}"]
        if not math.isfinite(weight) or weight <= 0.0:
            reasons.append(f"calibrated lambda {weight}")
        gate["arms"][name] = {
            "kind": kind,
            "pass": not reasons,
            "reasons": reasons,
            "mode": mode,
            "lambda": weight,
            "checkpoint": checkpoint,
        }
        verdict = "PASS" if not reasons else "FAIL " + "; ".join(reasons)
        print(f"[gate] {name}: {verdict}; mode {mode}, lambda {weight:.4g}")
    (s.out / "stage0_gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")


def part_regime(s: Stage0):
    ctx = s.ctx
    if len(ctx.models(("P10", "J15"))) < 2:
        raise RuntimeError("regime needs the P10 and J15 checkpoints")
    rows = []
    for (case, _, y, _, y_seg), u_p10, u_j15 in zip(ctx.pairs, ctx.flows("P10"), ctx.flows("J15"), strict=True):
        delta = (u_j15 - u_p10).norm(dim=1)[0]
        mean = F.avg_pool3d(y, kernel_size=9, stride=1, padding=4)
        spread = (F.avg_pool3d(y * y, kernel_size=9, stride=1, padding=4) - mean * mean).clamp_min(0).sqrt()[0, 0]
        seg = y_seg[0, 0]
        brain = seg > 0
        tercile = torch.bucketize(spread, spread[brain].quantile(torch.tensor([1 / 3, 2 / 3], device=seg.device)))
        regions = {f"spread_{label}": brain & (tercile == t) for t, label in enumerate(("flat", "medium", "textured"))}
        for group, labels in STRUCTURE_GROUPS.items():
            regions[f"group_{group}"] = torch.isin(seg, torch.tensor(labels, device=seg.device))
        total = float(delta[brain].sum())
        for region, mask in regions.items():
            rows.append(
                {
                    "case": case,
                    "region": region,
                    "voxel_share": float(mask.sum()) / float(brain.sum()),
                    "delta_share": float(delta[mask].sum()) / total,
                    "delta_mean": float(delta[mask].mean()),
                }
            )
    write_csv(s.out / "regime.csv", rows)
    for entry in summarise(rows, ("region",), ("voxel_share", "delta_share", "delta_mean")):
        print(
            f"[regime] {entry['region']:20s} voxels {entry['voxel_share']:.3f}, share of |u_J15 - u_P10| "
            f"{entry['delta_share']:.3f}, mean {entry['delta_mean']:.3f} vox"
        )


def _optimise(ctx, flow0, x, objective, steps, lr, reg_weight=1.0):
    """Adam on a dense residual from flow0 with Grad3d reg, as round-1 d2; flows at 0, steps/3 and steps."""
    grad3d, st = Grad3d(penalty="l2"), ctx.st(ctx.shape)
    residual = torch.zeros_like(flow0, requires_grad=True)
    opt = torch.optim.Adam([residual], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    snaps = {}
    for step in range(steps + 1):
        flow = flow0 + residual
        value = objective(st(x, flow), flow)
        if step in (0, steps // 3, steps):
            snaps[step] = (flow.detach(), float(value.detach()))
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        (value + reg_weight * grad3d(flow)).backward()
        opt.step()
        sched.step()
    return snaps


class AnatomixMaps(torch.nn.Module):
    """anatomix (utils.anatomy) as a one-level, full-resolution feature net for LearnedSimilarity."""

    def __init__(self, device):
        super().__init__()
        self.net = load_anatomix(device)

    def feature_maps(self, image):
        return [anatomix_features(self.net, image)]


def standalone_weight(pairs, flows, st, sim):
    """The weight that gives a similarity term NCC's force in the per-pair optimisation: the median over pairs of
    ||field gradient of the NCC loss|| / ||field gradient of the term|| at the given fields."""
    ratios = []
    for (_, x, y, _, _), flow in zip(pairs, flows, strict=True):
        flow = flow.detach().clone().requires_grad_(True)
        warped = st(x, flow)
        (g_ncc,) = torch.autograd.grad(-local_ncc_map(warped, y).mean(), flow, retain_graph=True)
        (g_term,) = torch.autograd.grad(sim(x, warped, y, flow), flow)
        ratios.append(float(g_ncc.norm() / g_term.norm().clamp_min(1e-30)))
    return float(np.median(ratios)), ratios


def part_tto(s: Stage0):
    ctx = s.ctx
    gate = json.loads((s.out / "stage0_gate.json").read_text(encoding="utf-8"))["arms"]
    starts = {name: ctx.flows(name) for name in ("P10", "J15") if name in ctx.models()}
    variants = [("ncc", None, 0.0, True)]  # name, similarity term, its weight, with the NCC loss
    for name, entry in gate.items():
        if entry["pass"]:
            variants.append((f"ncc+{name}", LearnedSimilarity(s.net(name), entry["mode"]), entry["lambda"], True))
    weights = {name: entry["lambda"] for name, entry in gate.items() if entry["pass"]}
    if s.args.anatomix:
        sim = LearnedSimilarity(AnatomixMaps(s.device), "warped_image")
        reference = starts["P10"] if "P10" in starts else next(iter(starts.values()))
        weight, ratios = standalone_weight(ctx.pairs, reference, ctx.st(ctx.shape), sim)
        variants.append(("anatomix", sim, weight, False))
        weights.update(anatomix=weight, anatomix_ratios=ratios)
        print(f"[tto] anatomix alone: weight {weight:.4g} (NCC's field-gradient scale at P10)")
    (s.out / "tto_weights.json").write_text(json.dumps(weights, indent=1), encoding="utf-8")
    rows = []
    for start, flows in starts.items():
        for variant, sim, weight, with_ncc in variants:
            t0 = time.perf_counter()
            for (case, x, y, x_seg, y_seg), flow0 in zip(ctx.pairs, flows, strict=True):

                def objective(warped, flow, x=x, y=y, sim=sim, weight=weight, with_ncc=with_ncc):
                    value = weight * sim(x, warped, y, flow) if sim is not None else 0.0
                    return value - local_ncc_map(warped, y).mean() if with_ncc else value

                for step, (flow, value) in _optimise(ctx, flow0, x, objective, s.args.tto_steps, 0.02).items():
                    rows.append(
                        {
                            "start": start,
                            "variant": variant,
                            "case": case,
                            "step": step,
                            "objective": value,
                            "fold_pct": fold_percent(flow),
                            **groups(dice35(warp_seg(ctx, x_seg, flow), y_seg)),
                        }
                    )
            print(f"[tto] {start} {variant}: {time.perf_counter() - t0:.0f}s")
    write_csv(s.out / "tto.csv", rows)
    summary = summarise(rows, ("start", "variant", "step"), ("dice35", "dice29", "dice6", "fold_pct"))
    write_csv(s.out / "tto_summary.csv", summary)
    for entry in summary:
        print(
            f"[tto] {entry['start']} {entry['variant']:12s} step {entry['step']:3d} dice35 {entry['dice35']:.4f} "
            f"dice6 {entry['dice6']:.4f} folds {entry['fold_pct']:.3f}%"
        )


PARTS = {
    "features": part_features,
    "cost": part_cost,
    "calibrate": part_calibrate,
    "gate": part_gate,
    "regime": part_regime,
    "tto": part_tto,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", required=True, help=f"comma-separated subset of {','.join(PARTS)}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", action="append", default=[], help="NAME=feature checkpoint, repeatable")
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N test pairs (0 = all 19)")
    parser.add_argument("--tto_steps", type=int, default=150)
    parser.add_argument("--anatomix", action="store_true", help="tto: add the anatomix similarity alone")
    parser.add_argument("--crop", type=int, nargs=3, default=None, help="centre crop D H W, multiples of 32 (smoke)")
    parser.add_argument("--smoke", action="store_true", help="random-init registration models (local smoke runs)")
    args = parser.parse_args()
    args.models, args.extra = (), []
    s = Stage0(args)
    failed = []
    for part in args.parts.split(","):
        t0 = time.perf_counter()
        print(f"[{part}] start")
        try:
            PARTS[part](s)
        except Exception:
            failed.append(part)
            (s.out / f"error_{part}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            traceback.print_exc()
        torch.cuda.empty_cache()
        print(f"[{part}] done in {time.perf_counter() - t0:.0f}s{' (FAILED)' if part in failed else ''}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
