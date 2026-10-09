"""Round-1 diagnostics after pilots P1-P3: one entry point, parts selected with --parts.

d0   Learn2Reg-2021 protocol check. Dice of every available checkpoint at full resolution, and with the field
     halved: scored on half-resolution labels, and upsampled back to full resolution; both pair directions.
d1   Local NCC of the warped moving image inside each fixed structure, with that structure's Dice.
d2b  Label-free regions: connected components of hard GMM tissue classes against the true structures.
d2   Per-pair optimisation of the P10 field under different objectives (NCC, weaker regulariser, MSE, NCC
     averaged per true structure, per label-free component, NCC + labelled Dice), and NCC from the J15 field.
d3   Gradients of each loss term w.r.t. the field and the network at the P10 weights: norms, cosines with the
     labelled-Dice direction, response to scaled and perturbed fields; writes the anatomix dose calibration.
d4   Transitivity: A->B composed with B->C against A->C on consecutive test subjects.
m2   Certified clip from the repaired field toward the feed-forward one, and the clip's gradient conditioning
     on real crops, from the identity (pilot P3) and from the repaired field (small steps).

    python -m tools.analysis.diag_round1 --parts d0,d1,d2b,d4,m2 --out results/round1/diag --paths 3

Every part writes CSV files into --out; a failing part leaves error_<part>.txt and the others still run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from experiments.core.path_profiles import get_dataset_paths
from utils import pkload, soft_dice_loss
from utils.anatomy import anatomix_features, fit_tissue_gmm, load_anatomix, tissue_posteriors
from utils.field import (
    certified_local_clip,
    compose_flows,
    differentiable_local_clip,
    digital_project,
    jacobian_det,
    trilinear_cert_bound,
    trilinear_project,
)
from utils.field_io import load_flow_npz
from utils.losses import Grad3d, NCCVxm, local_ncc_map, region_mean
from utils.spatial import SpatialTransformer

LABELS = tuple(range(1, 36))
SMALL = (4, 18, 19, 23, 34, 35)
SMALL_IDX = [LABELS.index(label) for label in SMALL]
REGULAR_IDX = [i for i, label in enumerate(LABELS) if label not in SMALL]
CONFIG = "CTCF-CascadeA-VM-Unified"
CKPTS = {
    "P10": "results/P10_LONGRUN_VXM_UNIFIED_SVF_OASIS/ckpt/best.pth",
    "P1_CONTROL": "results/PILOT_P1_CONTROL/ckpt/last.pth",
    "P1_GMM": "results/PILOT_P1_GMM/ckpt/last.pth",
    "P2_CONTROL": "results/PILOT_P2_CONTROL/ckpt/last.pth",
    "P2_LATENT": "results/PILOT_P2_LATENT/ckpt/last.pth",
    "P2_FOUNDATION": "results/PILOT_P2_FOUNDATION/ckpt/last.pth",
    "J15": "results/P16_W1_VXM_OASIS_LBL_DIG_J15/ckpt/best.pth",
    "NODIG": "results/P16_W1_VXM_OASIS_LBL_NODIG/ckpt/best.pth",
    "P3_CONTROL": "results/PILOT_P3_CONTROL/ckpt/last.pth",
}
CLIP_EPS, CERT_EPS = 0.0011, 0.001  # clip work margin and published certificate margin, as in the repair chain
GMM_FIT_FILES = 40  # the trainer's tissue mixture: first 40 sorted training images
MIN_COMPONENT = 8  # label-free components smaller than this many voxels are ignored as noise


class Ctx:
    """Test pairs on the device, cached transformers, models and the tissue mixture."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        paths = get_dataset_paths(args.paths, "OASIS")
        self.train_dir = paths["train_dir"]
        files = sorted(glob.glob(os.path.join(paths["val_dir"], "*.pkl")))
        self.pairs = [self._pair(path) for path in files[: args.limit or None]]
        self.shape = tuple(self.pairs[0][1].shape[2:])
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.extra = dict(item.split("=", 1) for item in args.extra)
        self._st, self._gmm, self._regions, self._flows = {}, None, {}, {}

    def vol(self, array, seg=False):
        t = torch.as_tensor(np.ascontiguousarray(array))[None, None]
        if self.args.crop:
            starts = [(n - c) // 2 for n, c in zip(t.shape[2:], self.args.crop, strict=True)]
            t = t[(slice(None), slice(None), *(slice(s, s + c) for s, c in zip(starts, self.args.crop, strict=True)))]
        return t.to(self.device).long() if seg else t.to(self.device).float()

    def _pair(self, path):
        x, y, x_seg, y_seg = pkload(path)
        return Path(path).stem[2:], self.vol(x), self.vol(y), self.vol(x_seg, True), self.vol(y_seg, True)

    def st(self, shape, mode="bilinear"):
        key = (tuple(shape), mode)
        if key not in self._st:
            self._st[key] = SpatialTransformer(shape, mode=mode).to(self.device)
        return self._st[key]

    def models(self, names=None):
        """Name -> checkpoint path of the models present (all random-init stand-ins under --smoke)."""
        if self.args.smoke:
            return {name: None for name in (names or ("P10", "J15"))}
        found = {name: path for name, path in {**CKPTS, **self.extra}.items() if os.path.isfile(path)}
        return {name: found[name] for name in (names or found) if name in found}

    def load(self, ckpt):
        adapter = get_model_adapter("ctcf")
        size = self.shape if self.args.crop else None
        model = adapter.build(time_steps=12, config_key=CONFIG, l3_svf=True, synth_img_size=size)
        if ckpt is not None:
            load_checkpoint_state(model, ckpt, strict=True)
        return adapter, model.to(self.device).eval()

    def flows(self, name):
        """Feed-forward flows of one model over the test pairs, cached on the CPU."""
        if name not in self._flows:
            adapter, model = self.load(self.models([name])[name])
            self._flows[name] = [predict(adapter, model, x, y).cpu() for _, x, y, _, _ in self.pairs]
            del model
            torch.cuda.empty_cache()
        return [flow.to(self.device) for flow in self._flows[name]]

    def gmm(self):
        if self._gmm is None:
            fit_files = sorted(glob.glob(os.path.join(self.train_dir, "*.pkl")))[:GMM_FIT_FILES]
            self._gmm = fit_tissue_gmm(pkload(path)[0] for path in fit_files)
        return self._gmm

    def regions(self, key, image):
        if key not in self._regions:
            self._regions[key] = tissue_regions(self, image)
        return self._regions[key]


def predict(adapter, model, moving, fixed):
    with torch.no_grad():
        return adapter.forward(model, moving, fixed)


def warp_seg(ctx, seg, flow):
    return ctx.st(seg.shape[2:], "nearest")(seg.float(), flow).round().long()


def dice35(warped, fixed):
    """Per-label Dice over the 35 OASIS structures; NaN where the label is absent from the fixed image."""
    out = np.full(len(LABELS), np.nan)
    for i, label in enumerate(LABELS):
        truth = fixed == label
        n_truth = int(truth.sum())
        if n_truth:
            pred = warped == label
            out[i] = 2.0 * int((pred & truth).sum()) / (int(pred.sum()) + n_truth)
    return out


def nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def groups(dice):
    return {
        "dice35": nanmean(dice),
        "dice29": nanmean(dice[REGULAR_IDX]),
        "dice6": nanmean(dice[SMALL_IDX]),
        "labels_scored": int(np.isfinite(dice).sum()),
    }


def fold_percent(flow):
    with torch.no_grad():
        return float((jacobian_det(flow) <= 0).float().mean().item() * 100.0)


def one_hot(seg):
    return F.one_hot(seg[:, 0], 36).permute(0, 4, 1, 2, 3).float()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows, keys, values):
    """Mean of `values` per combination of `keys`, in first-seen order."""
    buckets = {}
    for row in rows:
        buckets.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, members in buckets.items():
        entry = dict(zip(keys, key, strict=True))
        entry["n"] = len(members)
        for value in values:
            entry[value] = nanmean([m.get(value, float("nan")) for m in members])
        out.append(entry)
    return out


def tissue_regions(ctx, image):
    """Connected components (6-connected) of each hard GMM tissue class; ids from 1, 0 = background/noise.
    scipy is imported here so that an environment without it loses only the parts that need components."""
    from scipy import ndimage

    hard = tissue_posteriors(image, ctx.gmm())[:, 1:].argmax(dim=1)[0] + 1
    hard[image[0, 0] <= 0] = 0
    hard = hard.cpu().numpy()
    regions = np.zeros(hard.shape, dtype=np.int64)
    structure = ndimage.generate_binary_structure(3, 1)
    offset = 0
    for tissue in (1, 2, 3):
        components, n = ndimage.label(hard == tissue, structure=structure)
        keep = np.bincount(components.ravel(), minlength=n + 1) >= MIN_COMPONENT
        keep[0] = False
        ids = np.zeros(n + 1, dtype=np.int64)
        ids[keep] = offset + np.arange(1, int(keep.sum()) + 1)
        regions += ids[components]
        offset += int(keep.sum())
    return torch.from_numpy(regions)[None, None].to(ctx.device)


def components_or_none(ctx, key, image):
    """Label-free regions of `image` (cached under `key` unless it is None), or None without scipy."""
    try:
        return tissue_regions(ctx, image) if key is None else ctx.regions(key, image)
    except ImportError as error:
        print(f"scipy unavailable, component-based terms are skipped: {error}")
        return None


def _half_grid_scale(full, half, like):
    """Full-resolution voxels per half-resolution voxel on the end-point aligned half grid, per axis."""
    scale = [(n - 1) / (m - 1) for n, m in zip(full, half, strict=True)]
    return torch.tensor(scale, device=like.device, dtype=like.dtype).view(1, 3, 1, 1, 1)


def halve_index_field(d):
    """Index displacement to half resolution on the end-point aligned grid that scipy zoom and voxelmorph's
    ResizeTransform use (half voxel i at full position i * (N - 1) / (M - 1)), in half-resolution voxels."""
    full = tuple(d.shape[2:])
    half = tuple(n // 2 for n in full)
    return F.interpolate(d, size=half, mode="trilinear", align_corners=True) / _half_grid_scale(full, half, d)


def double_index_field(d_half, full):
    """Inverse of halve_index_field: back to full resolution, in full-resolution voxels."""
    scale = _half_grid_scale(full, tuple(d_half.shape[2:]), d_half)
    return F.interpolate(d_half * scale, size=tuple(full), mode="trilinear", align_corners=True)


def halve_labels(seg):
    """Labels resampled to half resolution by nearest neighbour on the same grid (scipy zoom, order 0)."""
    axes = [torch.linspace(-1.0, 1.0, n // 2, device=seg.device) for n in seg.shape[2:]]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)[None][..., [2, 1, 0]]
    return F.grid_sample(seg.float(), grid, mode="nearest", align_corners=True).round().long()


#  d0


def index_displacement(flow):
    """Our SpatialTransformer samples index (x + u) * N / (N - 1) - 0.5 on each axis (grid_sample with
    align_corners=False after a (N - 1) normalisation), so a zero field is not the identity. Learn2Reg's
    evaluator samples x + d with scipy map_coordinates. Returns that d, which samples the very same points."""
    d = torch.empty_like(flow)
    for axis, n in enumerate(flow.shape[2:]):
        view = [1, 1, 1]
        view[axis] = n
        coord = torch.arange(n, device=flow.device, dtype=flow.dtype).view(view)
        d[:, axis] = (coord + flow[:, axis]) * (n / (n - 1)) - 0.5 - coord
    return d


def warp_labels_index(seg, d):
    """map_coordinates(seg, identity + d, order=0) with zeros outside, on the GPU: nearest at index coordinates."""
    coords = []
    for axis, n in enumerate(seg.shape[2:]):
        view = [1, 1, 1]
        view[axis] = n
        coord = torch.arange(n, device=d.device, dtype=d.dtype).view(view)
        coords.append(2.0 * (coord + d[0, axis]) / (n - 1) - 1.0)
    grid = torch.stack(coords[::-1], dim=-1)[None]  # grid_sample wants (x, y, z) = (W, H, D) last
    return F.grid_sample(seg.to(d.dtype), grid, mode="nearest", align_corners=True).round().long()


def protocol_rows(ctx, model, case, direction, flow, moving_seg, fixed_seg, canonical=False):
    """Dice in our pipeline and in Learn2Reg's map_coordinates convention: at full resolution, with the field
    halved, and on halved labels. Learn2Reg 2021 took task-3 fields at half resolution in half-resolution
    voxels (a participant's code submits voxelmorph's half-size field), and im2grid's authors say it scored
    on x2 downsampled images: index_half_labels is that reading, index_half_field the alternative of
    upsampling the field. The official 2021 evaluator itself is not at hand.
    A CTCF flow is converted to that convention; a canonical external field (generate_external_fields:
    moving(i + flow(i))) already is in it and is scored as stored, so its "ours" row is its own warp."""
    d = flow if canonical else index_displacement(flow)
    d_half = halve_index_field(d)  # what a participant submitted: 3 x 80 x 96 x 112, half-resolution voxels
    scored = {
        "ours": dice35(warp_labels_index(moving_seg, d) if canonical else warp_seg(ctx, moving_seg, flow), fixed_seg),
        "index_full": dice35(warp_labels_index(moving_seg, d), fixed_seg),
        "index_half_field": dice35(
            warp_labels_index(moving_seg, double_index_field(d_half, flow.shape[2:])), fixed_seg
        ),
        "index_half_labels": dice35(warp_labels_index(halve_labels(moving_seg), d_half), halve_labels(fixed_seg)),
    }
    return [
        {"model": model, "case": case, "direction": direction, "mode": mode, **groups(dice)}
        for mode, dice in scored.items()
    ]


def part_d0(ctx):
    rows = []
    for name, ckpt in ctx.models(ctx.args.models or None).items():
        adapter, model = ctx.load(ckpt)
        for case, x, y, x_seg, y_seg in ctx.pairs:
            rows += protocol_rows(ctx, name, case, "fwd", predict(adapter, model, x, y), x_seg, y_seg)
            rows += protocol_rows(ctx, name, case, "rev", predict(adapter, model, y, x), y_seg, x_seg)
        del model
        torch.cuda.empty_cache()
    if not ctx.args.smoke and not ctx.args.models:
        for flow_dir in sorted(glob.glob("results/external/*/flows")):
            name = "EXT_" + Path(flow_dir).parent.name
            for case, _, _, x_seg, y_seg in ctx.pairs:
                path = Path(flow_dir) / f"flow_{case}.npz"
                if path.is_file():
                    flow = load_flow_npz(path).to(ctx.device)
                    rows += protocol_rows(ctx, name, case, "fwd", flow, x_seg, y_seg, canonical=True)
    write_csv(ctx.out / "d0_protocol.csv", rows)
    summary = summarise(rows, ("model", "direction", "mode"), ("dice35", "dice29", "dice6", "labels_scored"))
    write_csv(ctx.out / "d0_summary.csv", summary)
    for entry in summary:
        print(
            f"[d0] {entry['model']:14s} {entry['direction']} {entry['mode']:11s} dice35 {entry['dice35']:.4f} "
            f"dice29 {entry['dice29']:.4f} dice6 {entry['dice6']:.4f} labels {entry['labels_scored']:.1f}"
        )


#  d1


def part_d1(ctx):
    rows = []
    for name in ctx.models(("P10", "P2_CONTROL", "J15", "NODIG", "P3_CONTROL")):
        for (case, x, y, x_seg, y_seg), flow in zip(ctx.pairs, ctx.flows(name), strict=True):
            with torch.no_grad():
                warped = ctx.st(ctx.shape)(x, flow)
                cc = local_ncc_map(warped, y)[0, 0]
                residual = (warped - y).abs()[0, 0]
            dice = dice35(warp_seg(ctx, x_seg, flow), y_seg)
            seg = y_seg[0, 0]
            masks = {"volume": torch.ones_like(seg, dtype=torch.bool), "brain": seg > 0}
            masks.update({str(label): seg == label for label in LABELS})
            for region, mask in masks.items():
                n = int(mask.sum())
                rows.append(
                    {
                        "model": name,
                        "case": case,
                        "region": region,
                        "voxels": n,
                        "mean_cc": float(cc[mask].mean()) if n else float("nan"),
                        "mean_abs_residual": float(residual[mask].mean()) if n else float("nan"),
                        "dice": float(dice[LABELS.index(int(region))]) if region.isdigit() else float("nan"),
                    }
                )
    write_csv(ctx.out / "d1_local_ncc.csv", rows)
    summary = summarise(rows, ("model", "region"), ("mean_cc", "mean_abs_residual", "dice"))
    write_csv(ctx.out / "d1_summary.csv", summary)
    for entry in summary:
        if entry["region"] in ("volume", "brain"):
            print(f"[d1] {entry['model']:12s} {entry['region']:6s} mean cc {entry['mean_cc']:.4f}")


#  d2b


def fixed_id(case):
    """A pair file p_<A>_<B>.pkl stores x = subject B (moving) and y = subject A (fixed), Learn2Reg's order."""
    return case.split("_")[0]


def subjects(ctx):
    """Subject id -> (image, labels) from the chain of consecutive test pairs."""
    out = {}
    for case, x, y, x_seg, y_seg in ctx.pairs:
        fixed, moving = case.split("_")
        out.setdefault(moving, (x, x_seg))
        out.setdefault(fixed, (y, y_seg))
    return dict(sorted(out.items()))


def part_d2b(ctx):
    rows, sizes = [], []
    for sid, (image, seg) in subjects(ctx).items():
        regions = ctx.regions(sid, image).reshape(-1)
        labels = seg.reshape(-1)
        n_regions = int(regions.max()) + 1
        region_size = torch.bincount(regions, minlength=n_regions).double()
        sizes.append(
            {
                "subject": sid,
                "components": n_regions - 1,
                **{f"components_ge_{t}": int((region_size[1:] >= t).sum()) for t in (64, 512, 4096)},
            }
        )
        for label in LABELS:
            inside = labels == label
            n_label = int(inside.sum())
            if not n_label:
                continue
            overlap = torch.bincount(regions[inside], minlength=n_regions).double()
            overlap[0] = 0.0
            iou = overlap / (region_size + n_label - overlap).clamp_min(1.0)
            best = int(iou.argmax())
            rows.append(
                {
                    "subject": sid,
                    "label": label,
                    "label_voxels": n_label,
                    "best_iou": float(iou[best]),
                    "best_component_voxels": int(region_size[best]),
                    "purity": float(overlap[best] / region_size[best].clamp_min(1.0)),
                    "coverage": float(overlap[best] / n_label),
                }
            )
    write_csv(ctx.out / "d2b_regions.csv", rows)
    write_csv(ctx.out / "d2b_components.csv", sizes)
    summary = summarise(rows, ("label",), ("best_iou", "purity", "coverage", "label_voxels"))
    write_csv(ctx.out / "d2b_summary.csv", summary)
    for entry in summary:
        print(
            f"[d2b] label {entry['label']:2d} best IoU {entry['best_iou']:.3f} purity {entry['purity']:.3f} "
            f"coverage {entry['coverage']:.3f}"
        )


#  d2

D2_VARIANTS = (  # name, starting model, objective, regulariser weight (None: matched to NCC's start balance)
    ("ncc", "P10", "ncc", 1.0),
    ("ncc_reg0.25", "P10", "ncc", 0.25),
    ("mse", "P10", "mse", None),
    ("ncc_structures", "P10", "ncc_structures", 1.0),
    ("ncc_components", "P10", "ncc_components", 1.0),
    ("ncc_plus_dice", "P10", "ncc_dice", 1.0),
    ("ncc_from_J15", "J15", "ncc", 1.0),
)


def objective(kind, ctx, warped, y, flow, x_oh, y_oh, regions):
    """Similarity part of a d2 objective. `regions` is the fixed image's true labels for ncc_structures and
    its label-free components for ncc_components; one-hots are only built for ncc_dice."""
    if kind == "ncc":
        return -local_ncc_map(warped, y).mean()
    if kind == "mse":
        return (warped - y).pow(2).mean()
    if kind in ("ncc_structures", "ncc_components"):
        return -region_mean(local_ncc_map(warped, y), regions)
    return -local_ncc_map(warped, y).mean() + soft_dice_loss(ctx.st(ctx.shape)(x_oh, flow), y_oh)


def optimise(ctx, flow0, x, y, x_seg, y_seg, kind, reg_weight, regions):
    """Adam on a dense residual from flow0; returns {step: (flow, objective)} at 0, steps/3 and steps.
    Adam rescales each voxel's step, so re-weighting voxels acts here mainly through the local balance
    against the regulariser; in network training the same weights instead turn the shared gradient."""
    steps, grad3d, st = ctx.args.tto_steps, Grad3d(penalty="l2"), ctx.st(ctx.shape)
    x_oh, y_oh = (one_hot(x_seg), one_hot(y_seg)) if kind == "ncc_dice" else (None, None)
    if reg_weight is None:  # match the similarity/regulariser gradient balance NCC has at the start
        norms = []
        for k in ("mse", "ncc"):
            f = flow0.clone().requires_grad_(True)
            (g,) = torch.autograd.grad(objective(k, ctx, st(x, f), y, f, x_oh, y_oh, regions), f)
            norms.append(float(g.norm()))
        reg_weight = norms[0] / max(norms[1], 1e-30)
    residual = torch.zeros_like(flow0, requires_grad=True)
    opt = torch.optim.Adam([residual], lr=ctx.args.tto_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    snaps = {}
    for step in range(steps + 1):
        flow = flow0 + residual
        value = objective(kind, ctx, st(x, flow), y, flow, x_oh, y_oh, regions)
        if step in (0, steps // 3, steps):
            snaps[step] = (flow.detach(), float(value))
        if step == steps:
            break
        loss = value + reg_weight * grad3d(flow)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return snaps, reg_weight


def part_d2(ctx):
    rows = []
    starts = {name: ctx.flows(name) for name in {v[1] for v in D2_VARIANTS} if name in ctx.models()}
    for name, start, kind, reg_weight in D2_VARIANTS:
        if start not in starts:
            continue
        t0 = time.perf_counter()
        for i, (case, x, y, x_seg, y_seg) in enumerate(ctx.pairs):
            regions = {"ncc_structures": y_seg, "ncc_components": None}.get(kind)
            if kind == "ncc_components":
                regions = components_or_none(ctx, fixed_id(case), y)
                if regions is None:
                    break
            snaps, used_reg = optimise(ctx, starts[start][i], x, y, x_seg, y_seg, kind, reg_weight, regions)
            for step, (flow, value) in snaps.items():
                rows.append(
                    {
                        "variant": name,
                        "case": case,
                        "step": step,
                        "objective": value,
                        "reg_weight": used_reg,
                        "fold_pct": fold_percent(flow),
                        **groups(dice35(warp_seg(ctx, x_seg, flow), y_seg)),
                    }
                )
        print(f"[d2] {name}: {time.perf_counter() - t0:.0f}s")
    write_csv(ctx.out / "d2_tto.csv", rows)
    summary = summarise(rows, ("variant", "step"), ("dice35", "dice29", "dice6", "fold_pct", "objective"))
    write_csv(ctx.out / "d2_summary.csv", summary)
    for entry in summary:
        print(
            f"[d2] {entry['variant']:15s} step {entry['step']:3d} dice35 {entry['dice35']:.4f} "
            f"dice29 {entry['dice29']:.4f} dice6 {entry['dice6']:.4f} folds {entry['fold_pct']:.3f}%"
        )


#  d3


def loss_terms(ctx, x, y, x_seg, y_seg, flow, features, regions):
    """Every candidate training term for the direction x -> y, as the trainer computes it."""
    st = ctx.st(ctx.shape)
    warped = st(x, flow)
    gmm = ctx.gmm()
    terms = {
        "ncc": NCCVxm(win=[9, 9, 9])(warped, y),
        "reg": Grad3d(penalty="l2")(flow),
        "mse": (warped - y).pow(2).mean(),
        "ncc_structures": -region_mean(local_ncc_map(warped, y), y_seg),
        "gmm": soft_dice_loss(st(tissue_posteriors(x, gmm), flow), tissue_posteriors(y, gmm)),
        "dice": soft_dice_loss(st(one_hot(x_seg), flow), one_hot(y_seg)),
    }
    if regions is not None:
        terms["ncc_components"] = -region_mean(local_ncc_map(warped, y), regions)
    if features is not None:
        fx, fy = features
        cos = (F.normalize(st(fx, flow), dim=1) * fy).sum(dim=1)
        terms["anatomix"] = (1.0 - cos)[y[:, 0] > 0].mean()
    return terms


def cosine(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def smooth_noise(shape, amplitude, device, generator):
    noise = torch.randn((1, 3, *shape), device=device, generator=generator)
    kernel = torch.exp(-(torch.arange(-12, 13, device=device, dtype=noise.dtype) ** 2) / (2 * 4.0**2))
    kernel = kernel / kernel.sum()
    for axis in range(3):
        view = [1, 1, 1, 1, 1]
        view[2 + axis] = kernel.numel()
        pad = [0, 0, 0]
        pad[axis] = 12
        noise = F.conv3d(noise, kernel.view(view).repeat(3, 1, 1, 1, 1), padding=tuple(pad), groups=3)
    return noise * (amplitude / noise.abs().max().clamp_min(1e-12))


def anatomix_net(ctx):
    try:
        return load_anatomix(ctx.device)
    except Exception as error:  # not installed, or its weights cannot be fetched: the term is skipped
        print(f"[d3] anatomix unavailable, its term is skipped: {error}")
        return None


def part_d3(ctx):
    models = ctx.models(("P10",))
    if not models:
        raise RuntimeError("d3 needs the P10 checkpoint")
    _, model = ctx.load(models["P10"])
    net = anatomix_net(ctx)
    params = [p for p in model.parameters() if p.requires_grad]
    train_files = sorted(glob.glob(os.path.join(ctx.train_dir, "*.pkl")))
    rng = np.random.default_rng(0)
    grad_rows = []
    for k in range(ctx.args.train_pairs):
        i, j = rng.choice(len(train_files), size=2, replace=False)
        (x, x_seg), (y, y_seg) = [
            (ctx.vol(d[0]), ctx.vol(d[2] if len(d) > 2 else d[1], True))
            for d in (pkload(train_files[i]), pkload(train_files[j]))
        ]
        features = None
        if net is not None:
            with torch.no_grad():
                features = (anatomix_features(net, x), anatomix_features(net, y))
        regions = components_or_none(ctx, None, y)
        small = torch.zeros_like(y_seg, dtype=torch.bool)
        for label in SMALL:
            small |= y_seg == label
        near_small = (F.max_pool3d(small.float(), 7, 1, 3) > 0).expand(-1, 3, -1, -1, -1)
        _, flow = model(x, y, alpha_l1=1.0, alpha_l3=1.0)
        terms = loss_terms(ctx, x, y, x_seg, y_seg, flow, features, regions)
        grads = {}
        for name, value in terms.items():
            g = torch.autograd.grad(value, [flow, *params], retain_graph=True, allow_unused=True)
            theta = torch.cat(
                [(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(g[1:], params, strict=True)]
            )
            grads[name] = (g[0].detach(), theta.detach(), float(value))
        for name, (g_flow, g_theta, value) in grads.items():
            grad_rows.append(
                {
                    "pair": k,
                    "term": name,
                    "value": value,
                    "norm_flow": float(g_flow.norm()),
                    "norm_theta": float(g_theta.norm()),
                    "ratio_theta_to_ncc": float(g_theta.norm() / grads["ncc"][1].norm().clamp_min(1e-30)),
                    "cos_flow_dice": cosine(g_flow, grads["dice"][0]),
                    "cos_theta_dice": cosine(g_theta, grads["dice"][1]),
                    "cos_flow_dice_near_small": cosine(g_flow, grads["dice"][0], near_small),
                    "cos_flow_ncc": cosine(g_flow, grads["ncc"][0]),
                    "cos_theta_ncc": cosine(g_theta, grads["ncc"][1]),
                }
            )
        del flow, terms, grads
        torch.cuda.empty_cache()
    write_csv(ctx.out / "d3_gradients.csv", grad_rows)
    summary = summarise(
        grad_rows,
        ("term",),
        ("ratio_theta_to_ncc", "cos_theta_dice", "cos_flow_dice", "cos_flow_dice_near_small", "cos_theta_ncc", "value"),
    )
    write_csv(ctx.out / "d3_summary.csv", summary)
    for entry in summary:
        print(
            f"[d3] {entry['term']:15s} |g|/|g_ncc| {entry['ratio_theta_to_ncc']:.3e} cos(theta,dice) "
            f"{entry['cos_theta_dice']:+.3f} cos(field,dice) {entry['cos_flow_dice']:+.3f} near small "
            f"{entry['cos_flow_dice_near_small']:+.3f}"
        )

    ratios = {}
    for term in {row["term"] for row in grad_rows}:
        ratios[term] = float(np.median([r["ratio_theta_to_ncc"] for r in grad_rows if r["term"] == term]))
    calibration = {"median_ratio_theta_to_ncc": ratios, "pairs": ctx.args.train_pairs}
    if "anatomix" in ratios:
        calibration["w_anat"] = {str(share): share / ratios["anatomix"] for share in (0.5, 2.0)}
    (ctx.out / "d3_calibration.json").write_text(json.dumps(calibration, indent=1), encoding="utf-8")
    print(f"[d3] calibration {json.dumps(calibration)}")

    sens_rows = []
    generator = torch.Generator(device=ctx.device).manual_seed(0)
    for (case, x, y, x_seg, y_seg), flow in list(zip(ctx.pairs, ctx.flows("P10"), strict=True))[:4]:
        features = None
        if net is not None:
            with torch.no_grad():
                features = (anatomix_features(net, x), anatomix_features(net, y))
        regions = components_or_none(ctx, fixed_id(case), y)
        variants = [("scale", s, flow * s) for s in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)]
        variants += [("noise", a, flow + smooth_noise(ctx.shape, a, ctx.device, generator)) for a in (1.0, 2.0)]
        for kind, magnitude, field in variants:
            with torch.no_grad():
                terms = loss_terms(ctx, x, y, x_seg, y_seg, field, features, regions)
            sens_rows.append(
                {
                    "case": case,
                    "perturbation": kind,
                    "magnitude": magnitude,
                    **{name: float(v) for name, v in terms.items()},
                    **groups(dice35(warp_seg(ctx, x_seg, field), y_seg)),
                }
            )
    write_csv(ctx.out / "d3_sensitivity.csv", sens_rows)
    del model


#  d4


def part_d4(ctx):
    rows = []
    data = subjects(ctx)
    ids = list(data)
    for name, ckpt in ctx.models(("P10", "J15")).items():
        adapter, model = ctx.load(ckpt)
        for a, b, c in zip(ids, ids[1:], ids[2:], strict=False):
            (img_a, seg_a), (img_b, _), (img_c, seg_c) = data[a], data[b], data[c]
            u_ab, u_bc, u_ac = (
                predict(adapter, model, m, f) for m, f in ((img_a, img_b), (img_b, img_c), (img_a, img_c))
            )
            composed = compose_flows(u_bc, u_ab)
            error = (composed - u_ac).norm(dim=1)[0]
            direct = dice35(warp_seg(ctx, seg_a, u_ac), seg_c)
            via_b = dice35(warp_seg(ctx, seg_a, composed), seg_c)
            seg = seg_c[0, 0]
            rows.append(
                {
                    "model": name,
                    "triplet": f"{a}_{b}_{c}",
                    "region": "brain",
                    "mean_error": float(error[seg > 0].mean()),
                    "dice_direct": nanmean(direct),
                    "dice_composed": nanmean(via_b),
                }
            )
            for i, label in enumerate(LABELS):
                mask = seg == label
                if mask.any():
                    rows.append(
                        {
                            "model": name,
                            "triplet": f"{a}_{b}_{c}",
                            "region": str(label),
                            "mean_error": float(error[mask].mean()),
                            "dice_direct": float(direct[i]),
                            "dice_composed": float(via_b[i]),
                        }
                    )
        del model
        torch.cuda.empty_cache()
    write_csv(ctx.out / "d4_transitivity.csv", rows)
    summary = summarise(rows, ("model", "region"), ("mean_error", "dice_direct", "dice_composed"))
    write_csv(ctx.out / "d4_summary.csv", summary)
    for entry in summary:
        if entry["region"] == "brain":
            print(
                f"[d4] {entry['model']:4s} error {entry['mean_error']:.3f} vox, dice direct "
                f"{entry['dice_direct']:.4f} composed {entry['dice_composed']:.4f}"
            )


#  m2


def repaired(flow):
    """The repair chain, but to the clip's work margin: the clip needs its current field at >= CLIP_EPS."""
    flow, _, _ = digital_project(flow.float(), eps=0.0)
    flow, report = trilinear_project(flow, eps=CLIP_EPS)
    return flow, bool(report.certified)


def part_m2(ctx):
    clip_rows, grad_rows = [], []
    generator = torch.Generator(device=ctx.device).manual_seed(0)
    size, sweeps = min(64, *[(n // 2) * 2 for n in ctx.shape]), 2
    for name in ctx.models(("P3_CONTROL", "P10")):
        cases = list(zip(ctx.pairs, ctx.flows(name), strict=True))[: ctx.args.m2_cases]
        for (case, _, _, x_seg, y_seg), phi in cases:
            psi, certified = repaired(phi)
            if not certified:  # the clip's precondition fails: record the case, skip clip and gradients
                clip_rows.append({"model": name, "case": case, "state": "rep_not_certified"})
                continue
            states = {"ff": phi, "rep": psi}
            states.update(
                {f"clip_from_rep_s{s}": certified_local_clip(psi, phi, eps=CLIP_EPS, sweeps=s) for s in (1, 4)}
            )
            for state, flow in states.items():
                clip_rows.append(
                    {
                        "model": name,
                        "case": case,
                        "state": state,
                        "tri_cert_bound": trilinear_cert_bound(flow, 0, CERT_EPS),
                        "fold_pct": fold_percent(flow),
                        **groups(dice35(warp_seg(ctx, x_seg, flow), y_seg)),
                    }
                )
            for _ in range(2):
                origin = [
                    2 * int(torch.randint((n - size) // 2 + 1, (1,), generator=generator, device=ctx.device))
                    for n in ctx.shape
                ]
                block = (slice(None), slice(None), *(slice(o, o + size) for o in origin))
                weights = torch.randn((1, 3, size, size, size), device=ctx.device, generator=generator)
                for mode, current in (("from_identity", None), ("from_repaired", psi[block].detach())):
                    proposal = phi[block].detach().clone().requires_grad_(True)
                    out = differentiable_local_clip(proposal, CLIP_EPS, sweeps, flow_current=current)
                    (g,) = torch.autograd.grad((out * weights).mean(), proposal)
                    reference = weights.abs().mean() / weights.numel()
                    grad_rows.append(
                        {
                            "model": name,
                            "case": case,
                            "origin": "_".join(map(str, origin)),
                            "mode": mode,
                            "norm_ratio": float(g.norm() / (weights / weights.numel()).norm()),
                            "max_ratio": float(g.abs().max() / reference),
                            "nonfinite": int((~torch.isfinite(g)).sum()),
                            "moved_fraction": float((out - proposal).abs().gt(1e-6).float().mean()),
                        }
                    )
    write_csv(ctx.out / "m2_clip.csv", clip_rows)
    write_csv(ctx.out / "m2_gradients.csv", grad_rows)
    for entry in summarise(clip_rows, ("model", "state"), ("dice35", "dice6", "tri_cert_bound", "fold_pct")):
        print(
            f"[m2] {entry['model']:10s} {entry['state']:18s} dice35 {entry['dice35']:.4f} dice6 "
            f"{entry['dice6']:.4f} bound {entry['tri_cert_bound']:+.5f} folds {entry['fold_pct']:.3f}%"
        )
    for entry in summarise(grad_rows, ("model", "mode"), ("norm_ratio", "max_ratio", "nonfinite", "moved_fraction")):
        print(
            f"[m2] {entry['model']:10s} {entry['mode']:13s} |g| x{entry['norm_ratio']:.2f} max x{entry['max_ratio']:.2e} "
            f"nonfinite {entry['nonfinite']:.0f} moved {entry['moved_fraction']:.3f}"
        )


PARTS = {"d0": part_d0, "d1": part_d1, "d2b": part_d2b, "d2": part_d2, "d3": part_d3, "d4": part_d4, "m2": part_m2}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", required=True, help=f"comma-separated subset of {','.join(PARTS)}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--paths", type=int, default=3, help="path profile of experiments.core.path_profiles")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N test pairs (0 = all 19)")
    parser.add_argument("--models", default="", help="d0: comma-separated model names (default: all present)")
    parser.add_argument("--extra", action="append", default=[], help="d0: extra NAME=checkpoint, repeatable")
    parser.add_argument("--train_pairs", type=int, default=8, help="d3: random training pairs")
    parser.add_argument("--tto_steps", type=int, default=150)
    parser.add_argument("--tto_lr", type=float, default=0.02)
    parser.add_argument("--m2_cases", type=int, default=6)
    parser.add_argument("--crop", type=int, nargs=3, default=None, help="centre crop D H W (local smoke runs)")
    parser.add_argument("--smoke", action="store_true", help="random-init models instead of checkpoints")
    args = parser.parse_args()
    args.models = tuple(name for name in args.models.split(",") if name)
    ctx = Ctx(args)
    failed = []
    for part in args.parts.split(","):
        t0 = time.perf_counter()
        print(f"[{part}] start, {len(ctx.pairs)} test pairs")
        try:
            PARTS[part](ctx)
        except Exception:
            failed.append(part)
            (ctx.out / f"error_{part}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            traceback.print_exc()
        torch.cuda.empty_cache()
        print(f"[{part}] done in {time.perf_counter() - t0:.0f}s{' (FAILED)' if part in failed else ''}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
