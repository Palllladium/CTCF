"""Rank-one Conv3d interventions and labelled field diagnostics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

HEADS = {"l2": "level2.model.reg_head.0", "l3": "level3.out"}
CORTEX = (2, 22)
DEEP_GRAY = (7, 8, 9, 10, 27, 28, 29, 30)


def project_weight(weight, direction, strength):
    if weight.ndim != 5 or direction.shape != (weight.shape[1],):
        raise ValueError("The direction must index Conv3d input channels")
    direction = direction.to(weight)
    norm = direction.norm()
    if not torch.isfinite(norm) or norm <= 0 or not np.isfinite(strength):
        raise ValueError("Projection needs a finite nonzero direction and finite strength")
    direction = direction / norm
    component = torch.einsum("oczyx,c->ozyx", weight, direction)
    return weight - strength * component[:, None] * direction[None, :, None, None, None]


@torch.no_grad()
def projection_report(module, features, direction):
    """Check the padded head algebra in FP32; quantify the noncommuting BF16 rounding separately."""
    x, w, bias = features.float(), module.weight.float(), module.bias.float() if module.bias is not None else None
    v = direction.to(x)
    v = v / v.norm()
    config = dict(stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
    rows = []
    for strength in (0.5, 1):
        erased = x - strength * (x * v[None, :, None, None, None]).sum(1, keepdim=True) * v[None, :, None, None, None]
        edited = project_weight(w, v, strength)
        a, b = F.conv3d(erased, w, bias, **config), F.conv3d(x, edited, bias, **config)
        scale = max(1, float(a.abs().max()))
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6 * scale)
        row = {"strength": strength, "fp32_max_error": float((a - b).abs().max())}
        if features.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b, base = [
                    F.conv3d(values, kernel, bias, **config).float()
                    for values, kernel in ((erased, w), (x, edited), (x, w))
                ]
            discrepancy, effect = (a - b).norm(), (b - base).norm()
            row.update(
                bf16_max_error=float((a - b).abs().max()),
                bf16_rounding_over_compiled_effect=float(discrepancy / effect) if effect > 0 else None,
            )
        rows.append(row)
    return {
        "rows": rows,
        "scope": "Actual head, discovery feature crop and zero padding. Algebraic equivalence tested in FP32; BF16 is not bitwise equivalent. All scientific readouts use the compiled weights, not an activation hook.",
    }


def error_counts(predicted, fixed):
    result = {}
    for name, labels in (("cortex", CORTEX), ("cortex_2", (2,)), ("cortex_22", (22,)), ("deep_gray", DEEP_GRAY)):
        p, t = [torch.isin(value, value.new_tensor(labels)) for value in (predicted, fixed)]
        result.update(
            {
                f"{name}_{key}": int(value.sum())
                for key, value in (("tp", p & t), ("fp", p & ~t), ("fn", ~p & t), ("predicted", p), ("fixed", t))
            }
        )
    return result


def protected_weight(moving_labels, fixed_labels, first, second, width=4):
    """Conservative nearest-label protection along the entire straight field segment.

    Distance to a cortex voxel's Voronoi cube is bounded below by distance to its
    centre minus sqrt(3)/2. The actual sampler zero-pads; clamped positions here
    conservatively overprotect exterior points. Clamping is 1-Lipschitz. End-point
    label unions alone do not protect intermediate nearest-neighbour samples.
    """
    moving = torch.isin(moving_labels, moving_labels.new_tensor(CORTEX))[0, 0].cpu().numpy()
    fixed = torch.isin(fixed_labels, fixed_labels.new_tensor(CORTEX))[0, 0].cpu().numpy()
    shape = first.shape[2:]
    grid = np.indices(shape, dtype=np.float32)
    start = grid + first.detach().cpu().numpy()[0]
    start = np.clip(start, 0, np.asarray(shape)[:, None, None, None] - 1)
    centre = np.rint(start).astype(np.intp)
    if moving.any():
        distance = ndimage.distance_transform_edt(~moving)
        lower = distance[tuple(centre)] - np.linalg.norm(start - centre, axis=0) - np.sqrt(3) / 2
        length = (second - first).float().square().sum(1).sqrt()[0].cpu().numpy()
        protected = fixed | (lower <= length + 1e-4)
    else:
        protected = fixed
    distance = ndimage.distance_transform_edt(~protected) if protected.any() else np.full(shape, width)
    fraction = np.clip(distance / width, 0, 1)
    result = 3 * fraction**2 - 2 * fraction**3
    return torch.from_numpy(result.astype(np.float32))[None, None].to(first.device)


def matched_features(features, error, correct, image, fixed_labels, *, seed, count=4096):
    """Match error/control points within a pair, intensity bin and fixed-cortex distance bin."""
    shape = features.shape[2:]
    cortex = torch.isin(fixed_labels, fixed_labels.new_tensor(CORTEX))[0, 0].cpu().numpy()
    distance = ndimage.distance_transform_edt(~cortex) if cortex.any() else np.full(cortex.shape, 32)
    distance = torch.from_numpy(distance.astype(np.float32))[None, None].to(features.device)
    # Coarse head cells: any error versus all correct, with the same adaptive pooling footprints.
    error = F.adaptive_max_pool3d(error.float(), shape)[0, 0] > 0
    correct = F.adaptive_max_pool3d((~correct).float(), shape)[0, 0] == 0
    intensity = F.adaptive_avg_pool3d(image[:, :1].float(), shape)[0, 0]
    distance = F.adaptive_avg_pool3d(distance, shape)[0, 0]
    ibin = (intensity.clamp(0, 1) * 8).long().clamp_max(7)
    dbin = torch.bucketize(distance.contiguous(), distance.new_tensor((2, 4, 8, 16)))
    context = (ibin * 5 + dbin).flatten()
    error, correct = error.flatten(), correct.flatten()
    generator = torch.Generator(device=features.device).manual_seed(seed)
    first, second = [], []
    for group in range(40):
        a = torch.nonzero(error & (context == group)).flatten()
        b = torch.nonzero(correct & (context == group)).flatten()
        n = min(len(a), len(b), max(1, count // 40))
        if n:
            first.append(a[torch.randperm(len(a), generator=generator, device=a.device)[:n]])
            second.append(b[torch.randperm(len(b), generator=generator, device=b.device)[:n]])
    if not first:
        return None
    values = features.detach().float()[0].flatten(1).T
    return values[torch.cat(first)].double().cpu(), values[torch.cat(second)].double().cpu()


def feature_moments(first, second):
    if first.shape != second.shape or not len(first):
        raise ValueError("Feature contrasts require nonempty matched samples")
    return {
        "count": len(first),
        "means": torch.stack((first.mean(0), second.mean(0))),
        "second_moments": torch.stack((first.T @ first, second.T @ second)) / len(first),
    }


def fit_direction(moments, ridge=1e-3):
    """Equal pair weighting; fixed trace-scaled ridge, fitted on discovery only."""
    if not moments:
        return None
    means = torch.stack([m["means"] for m in moments]).mean(0)
    # Within-pair covariance avoids treating subject intensity shifts as an error direction.
    covariances = [m["second_moments"] - m["means"][:, :, None] * m["means"][:, None, :] for m in moments]
    covariance = torch.stack(covariances).mean((0, 1))
    scale = covariance.trace() / len(covariance)
    if not torch.isfinite(scale) or scale <= 0:
        return None
    direction = torch.linalg.solve(covariance + ridge * scale * torch.eye(len(covariance)), means[0] - means[1])
    norm = direction.norm()
    if not torch.isfinite(norm) or norm <= 1e-12:
        return None
    return (direction / norm).float()


def group_mean(rows, *, regions=None):
    if not rows:
        raise ValueError("Cannot aggregate an empty readout")
    pairs = defaultdict(list)
    for row in rows:
        pairs[tuple(sorted(row["case_id"].split("__")))].append(row)
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    mean = {k: float(np.mean([np.mean([r[k] for r in group]) for group in pairs.values()])) for k in keys}
    regions = {"cortex": CORTEX, "deep_gray": DEEP_GRAY} if regions is None else regions
    for name, labels in regions.items():
        if all(f"dice_lbl_{label}" in mean for label in labels):
            mean[name] = float(np.mean([mean[f"dice_lbl_{label}"] for label in labels]))
    return {"directions": len(rows), "unordered_pairs": len(pairs), "means": mean}


def joint_criterion(candidate, first, second):
    a, b, c = [value["means"] for value in (first, second, candidate)]
    gain = b["deep_gray"] - a["deep_gray"]
    gaps = {
        "cortex_vs_b0": c["cortex"] - a["cortex"],
        "deep_gain": gain,
        "deep_gain_retained": (c["deep_gray"] - a["deep_gray"]) / gain if gain > 0 else None,
        "macro_vs_b1": c["dice_mean"] - b["dice_mean"],
        "macro_vs_b0": c["dice_mean"] - a["dice_mean"],
    }
    passed = (
        gain > 0
        and gaps["cortex_vs_b0"] >= -0.001
        and gaps["deep_gain_retained"] >= 0.8
        and gaps["macro_vs_b1"] >= -0.001
    )
    return {**gaps, "passed": bool(passed)}
