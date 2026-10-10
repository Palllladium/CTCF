from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.core.path_profiles import get_dataset_paths
from utils import pkload, setup_device
from utils.anatomy import augment_intensity
from utils.feature_similarity import FEATURE_NETS, index_grid, level_points, sample_at, save_feature_net
from utils.losses import NCCVxm

# NCC window of the registration loss (9^3): hard negatives are scored with the same local NCC.
PATCH_RADIUS = 4
# Anchors and their true correspondences stay this far inside the volume so 9^3 patches are complete.
BORDER = 5
# Variance floor of the registration loss's local NCC (utils.losses.local_ncc_map).
NCC_EPS = NCCVxm().eps
# Hard-negative mining, pre-registered for round 2 (PILOTS.md); also used by tools/analysis/diag_round2.
# NCC here is the loss's squared local correlation; a 0.25 margin gives hard negatives to ~35% of anchors on T1.
HARD_DEFAULTS = {
    "candidates": 64,
    "r_min": 4.0,
    "r_max": 12.0,
    "ncc_tau": 0.5,
    "ncc_margin": 0.25,
    "hard_per_anchor": 4,
}


def random_displacement(
    shape: tuple[int, int, int],
    max_disp: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Smooth random displacement [1,3,D,H,W] in voxels: low-resolution noise, trilinear upsampling."""
    low = tuple(max(2, n // 16) for n in shape)
    noise = torch.randn((1, 3, *low), generator=generator).to(device)
    d = F.interpolate(noise, size=shape, mode="trilinear", align_corners=True)
    scale = max_disp * (0.5 + 0.5 * torch.rand((), generator=generator).item())
    return d * (scale / d.abs().amax().clamp_min(1e-6))


def bias_field(image: torch.Tensor, strength: float, generator: torch.Generator) -> torch.Tensor:
    """Smooth multiplicative intensity non-uniformity (+-strength) on the foreground."""
    low = tuple(max(2, n // 32) for n in image.shape[2:])
    field = F.interpolate(
        torch.randn((1, 1, *low), generator=generator).to(image.device),
        size=image.shape[2:],
        mode="trilinear",
        align_corners=True,
    )
    field = 1.0 + strength * field / field.abs().amax().clamp_min(1e-6)
    return torch.where(image > 0, image * field, image)


def augment(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return augment_intensity(bias_field(image, 0.15, generator), generator)


def warp_index(image: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """image(x + d(x)) with bilinear sampling at index coordinates: voxel x of the result shows point x + d(x)."""
    shape = image.shape[2:]
    axes = [torch.arange(n, device=d.device, dtype=d.dtype) for n in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)[None] + d.permute(0, 2, 3, 4, 1)
    return F.grid_sample(image, index_grid(grid, shape), mode="bilinear", align_corners=True)


def _patch_offsets(device: torch.device) -> torch.Tensor:
    r = torch.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)


def patches(image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Mean-centred 9^3 intensity patches of a [1,1,D,H,W] image at index points [..., 3] -> [..., 729]."""
    values = sample_at(image, points[..., None, :] + _patch_offsets(points.device))[..., 0]
    return values - values.mean(dim=-1, keepdim=True)


def patch_ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The registration loss's local NCC of two mean-centred 9^3 patches (broadcast over leading dims): squared
    correlation with each window variance floored at NCC_EPS, the value utils.losses.local_ncc_map gives a voxel."""
    cross = (a * b).sum(dim=-1)
    var_a = (a * a).sum(dim=-1).clamp_min(NCC_EPS)
    var_b = (b * b).sum(dim=-1).clamp_min(NCC_EPS)
    return cross * cross / (var_a * var_b)


def foreground_points(image: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """`count` random foreground voxels at least BORDER voxels from every face, as float (z, y, x)."""
    inside = image[0, 0] > 0
    for axis, n in enumerate(inside.shape):
        index = [slice(None)] * 3
        index[axis] = slice(0, BORDER)
        inside[tuple(index)] = False
        index[axis] = slice(n - BORDER, n)
        inside[tuple(index)] = False
    candidates = inside.nonzero()
    pick = torch.randint(len(candidates), (count,), generator=generator).to(candidates.device)
    return candidates[pick].float()


def inside_volume(points: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    upper = torch.tensor([n - 1 - BORDER for n in shape], device=points.device, dtype=points.dtype)
    return ((points >= BORDER) & (points <= upper)).all(dim=-1)


def mine_ncc_hard(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    anchors_b: torch.Tensor,
    truth_a: torch.Tensor,
    settings: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Wrong correspondences that the registration loss's local NCC rates nearly as high as the true one.

    For each anchor of view_b, candidates of view_a lie r_min..r_max voxels from the true point; a candidate is
    hard when its 9^3 NCC (patch_ncc) with the anchor patch reaches max(ncc_tau, NCC of the true point -
    ncc_margin). Returns up to `hard_per_anchor` points [K, h, 3], their validity [K, h] and coverage statistics.
    """
    k, s = len(anchors_b), settings.candidates
    direction = F.normalize(torch.randn((k, s, 3), generator=generator), dim=-1).to(anchors_b.device)
    radius = settings.r_min + (settings.r_max - settings.r_min) * torch.rand((k, s, 1), generator=generator)
    candidates = truth_a[:, None, :] + direction * radius.to(anchors_b.device)
    anchor_patches = patches(view_b, anchors_b)
    ncc = patch_ncc(patches(view_a, candidates), anchor_patches[:, None, :])
    ncc_true = patch_ncc(patches(view_a, truth_a), anchor_patches)
    threshold = torch.maximum(torch.full_like(ncc_true, settings.ncc_tau), ncc_true - settings.ncc_margin)
    hard = (ncc >= threshold[:, None]) & inside_volume(candidates, view_a.shape[2:])
    score = torch.where(hard, ncc, torch.full_like(ncc, -2.0))
    top, index = score.topk(settings.hard_per_anchor, dim=1)
    points = torch.gather(candidates, 1, index[..., None].expand(-1, -1, 3))
    stats = {
        "coverage": float(hard.any(dim=1).float().mean()),
        "hard_per_anchor": float(hard.sum(dim=1).float().mean()),
        "ncc_true": float(ncc_true.mean()),
    }
    return points, top > -2.0, stats


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    random_negatives: torch.Tensor,
    hard_negatives: torch.Tensor | None,
    hard_valid: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE over unit vectors: each anchor against its positive, the other anchors' positives, shared random
    negatives and, if given, its own hard negatives (invalid ones masked out)."""
    k = len(anchor)
    logits = [
        (anchor * positive).sum(dim=-1, keepdim=True),
        (anchor @ positive.T).masked_fill(torch.eye(k, dtype=torch.bool, device=anchor.device), float("-inf")),
        anchor @ random_negatives.T,
    ]
    if hard_negatives is not None:
        hard = torch.einsum("kc,khc->kh", anchor, hard_negatives)
        logits.append(hard.masked_fill(~hard_valid, float("-inf")))
    target = torch.zeros(k, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(torch.cat(logits, dim=1) / temperature, target)


def contrastive_step(net, image, args, generator):
    shape = tuple(image.shape[2:])
    d = random_displacement(shape, args.max_disp, generator, image.device)
    view_a = augment(image, generator)
    view_b = warp_index(augment(image, generator), d)
    feats_a, feats_b = net(view_a), net(view_b)

    def embed(feats, points):
        return F.normalize(sample_at(feats, level_points(points, shape, feats.shape[2:])), dim=-1)

    anchors = foreground_points(view_b, args.anchors, generator)
    index = anchors.long()
    truth = anchors + d[0, :, index[:, 0], index[:, 1], index[:, 2]].T
    keep = inside_volume(truth, shape)
    anchors, truth = anchors[keep], truth[keep]
    anchor_f, positive_f = embed(feats_b, anchors), embed(feats_a, truth)
    random_f = embed(feats_a, foreground_points(view_a, args.random_negatives, generator))

    hard_f, hard_valid, stats = None, None, {}
    if args.negatives == "ncc_hard":
        hard_points, hard_valid, stats = mine_ncc_hard(view_a, view_b, anchors, truth, args, generator)
        hard_f = embed(feats_a, hard_points)
    loss = info_nce(anchor_f, positive_f, random_f, hard_f, hard_valid, args.temperature)
    stats["positive_cos"] = float((anchor_f * positive_f).sum(dim=-1).mean())
    return loss, stats


def autoencoder_step(net, image, args, generator):
    del args, generator
    return F.mse_loss(net(image), F.avg_pool3d(image, kernel_size=2)), {}


def load_images(args) -> list[torch.Tensor]:
    train_dir = get_dataset_paths(args.paths, "OASIS")["train_dir"]
    files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))[: args.limit_files or None]
    images = []
    for path in files:
        image = torch.as_tensor(np.ascontiguousarray(pkload(path)[0]))[None, None]
        if args.crop:
            starts = [(n - c) // 2 for n, c in zip(image.shape[2:], args.crop, strict=True)]
            image = image[(..., *(slice(s, s + c) for s, c in zip(starts, args.crop, strict=True)))]
        images.append(image.half())
    return images


def parse_args():
    p = argparse.ArgumentParser(description="Train a frozen feature net for the round-2 learned similarity.")
    p.add_argument("--kind", choices=sorted(FEATURE_NETS), required=True)
    p.add_argument("--negatives", choices=["random", "ncc_hard"], default="random", help="contrastive only")
    p.add_argument("--out", required=True, help="output directory (checkpoint, summary.json)")
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_disp", type=float, default=6.0, help="largest known displacement, voxels")
    p.add_argument("--anchors", type=int, default=512)
    p.add_argument("--random_negatives", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    h = HARD_DEFAULTS
    p.add_argument("--candidates", type=int, default=h["candidates"], help="ncc_hard: candidates per anchor")
    p.add_argument("--r_min", type=float, default=h["r_min"], help="ncc_hard: nearest candidate, voxels from truth")
    p.add_argument("--r_max", type=float, default=h["r_max"])
    p.add_argument("--ncc_tau", type=float, default=h["ncc_tau"], help="ncc_hard: absolute NCC floor of a hard pair")
    p.add_argument("--ncc_margin", type=float, default=h["ncc_margin"], help="ncc_hard: NCC deficit to the truth")
    p.add_argument("--hard_per_anchor", type=int, default=h["hard_per_anchor"])
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--limit_files", type=int, default=0, help="first N training images (0 = all; smoke runs)")
    p.add_argument("--crop", type=int, nargs=3, default=None, help="centre crop D H W (smoke runs)")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=args.gpu, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    images = load_images(args)
    config: dict = {}
    net = FEATURE_NETS[args.kind](**config).to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    step = contrastive_step if args.kind == "contrastive" else autoencoder_step
    print(
        f">>> {args.kind} ({args.negatives}) on {len(images)} images, {sum(p.numel() for p in net.parameters())} "
        f"parameters, {args.iters} iterations"
    )

    history, window, t0 = [], [], time.perf_counter()
    for it in range(1, args.iters + 1):
        lr = 0.5 * args.lr * (1.0 + math.cos(math.pi * (it - 1) / args.iters))
        for group in optimizer.param_groups:
            group["lr"] = lr
        image = images[int(torch.randint(len(images), (1,), generator=generator))].to(device).float()
        loss, stats = step(net, image, args, generator)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {it}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        window.append({"loss": float(loss), **stats})
        if it % args.log_every == 0 or it == args.iters:
            entry = {"iter": it, "lr": lr, "seconds": time.perf_counter() - t0}
            entry.update({key: float(np.mean([w[key] for w in window])) for key in window[0]})
            history.append(entry)
            window = []
            print(
                "[feat] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items())
            )

    meta = {"args": vars(args), "history": history, "seconds": time.perf_counter() - t0}
    save_feature_net(os.path.join(args.out, "feature_net.pth"), net.eval(), args.kind, config, meta)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    print(f">>> saved {os.path.join(args.out, 'feature_net.pth')} after {meta['seconds']:.0f}s")


if __name__ == "__main__":
    main()
