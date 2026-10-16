"""Corrected adaptation of CST (Bigalke et al., MICCAI 2023) with the affine-equivariance term of CoRe (Kats et al.,
arXiv 2603.23694) for OASIS: the pseudo-label stream, training targets under affine augmentation, the equivariance
InfoNCE and the conversion to the canonical voxel flow. Each correction against the published code names the code it
replaces. Offsets are normalized grid_sample offsets in (x, y, z) channel order,
sampled with align_corners=False, as in models.CST.core.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CST.core import AdamReg, coupled_convex

TRE_SCALE = 1.5  # the CST/CoRe training loss multiplies the voxel error by 1.5; the contrastive schedule assumes it
TRE_CROP = 8  # interior of the half-resolution field used by the loss and the sample difficulty
TRUNK_DEPTH = 10  # equivariance features: through the 4th convolution (grid 1/4, 128 channels), as the CoRe code
POINT_MARGIN = 5  # equivariance points stay this many feature cells inside the volume, as the CoRe code


def identity_grid(shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, 4, device=like.device, dtype=like.dtype).unsqueeze(0)
    return F.affine_grid(eye, (1, 1, *shape), align_corners=False)


def warp(volume: torch.Tensor, offsets: torch.Tensor, mode: str = "bilinear", pad: float = 0.0) -> torch.Tensor:
    """volume(x + offsets(x)); `pad` is the value assumed outside the volume."""
    grid = identity_grid(tuple(volume.shape[2:]), offsets) + offsets.permute(0, 2, 3, 4, 1)
    return F.grid_sample(volume - pad, grid.to(volume.dtype), mode=mode, align_corners=False) + pad


def warp_affine(volume: torch.Tensor, matrix: torch.Tensor, pad: float = 0.0) -> torch.Tensor:
    """volume(A x) for a 4x4 affine A in normalized (x, y, z) coordinates."""
    grid = F.affine_grid(matrix[:3].unsqueeze(0).to(volume), (1, 1, *volume.shape[2:]), align_corners=False)
    return F.grid_sample(volume - pad, grid, align_corners=False) + pad


def compose(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Total offset of warping by `first` and then registering the result with `second`: second + first(id + second).
    The CST code sums the two the other way round (registration_pipeline.py:80)."""
    return second + warp(first, second)


def to_canonical(offsets: torch.Tensor) -> torch.Tensor:
    """Normalized (x, y, z) offsets -> voxel-unit pull flow in (z, y, x), the contract of utils.field._warp: an
    align_corners=False sample at grid + offset reads index i + offset * n / 2 along an axis of size n."""
    n = offsets.shape[2:]
    scale = offsets.new_tensor([n[0] / 2, n[1] / 2, n[2] / 2]).view(1, 3, 1, 1, 1)
    return offsets.flip(1) * scale


def random_projection(c_in: int, c_out: int, seed: int):
    """A fixed 1x1x1 convolution with nn.Conv3d's default uniform initialisation. The CST code draws a new one on every
    call; seeding it makes the pseudo labels and the evaluation reproducible."""
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.rand((c_out, c_in, 1, 1, 1), generator=generator) * 2 - 1) / math.sqrt(c_in)
    return lambda x: F.conv3d(x, weight.to(device=x.device, dtype=x.dtype))


@torch.no_grad()
def solver_offsets(
    net: nn.Sequential,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    num_warps: int = 2,
    ice: bool = True,
    pad: float = 0.0,
) -> torch.Tensor:
    """The solver part of the CST pseudo-label stream (registration_pipeline.update_fields): coupled convex
    optimisation with inverse consistency, then a second warp of the moving image (`pad`: its background value)."""
    net.eval()
    shape = tuple(fixed.shape[2:])
    with torch.autocast(device_type=fixed.device.type, dtype=torch.float16, enabled=fixed.is_cuda):
        feat_fix = net(fixed)
        offsets = coupled_convex(feat_fix, net(moving), use_ice=ice, img_shape=shape).float()
        for _ in range(num_warps - 1):
            warped = warp(moving, offsets, mode="nearest", pad=pad)
            second = coupled_convex(feat_fix, net(warped), use_ice=ice, img_shape=shape).float()
            offsets = compose(offsets, second)
    return offsets


@torch.no_grad()
def refine(
    net: nn.Sequential,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    offsets: torch.Tensor,
    *,
    reg_fac: float,
    proj_seed: int,
) -> torch.Tensor:
    """The Adam instance optimisation of the stream: feature MSE on random projections of early features (half
    resolution, 64 -> 32 channels) plus diffusion regularisation weighted by `reg_fac`."""
    net.eval()
    with torch.autocast(device_type=fixed.device.type, dtype=torch.float16, enabled=fixed.is_cuda):
        project = random_projection(64, 32, proj_seed)
        early = net[:6]
        refined = AdamReg(5 * project(early(fixed)), 5 * project(early(moving)), offsets, reg_fac=reg_fac)
    return refined.float()


def teacher_fields(
    net: nn.Sequential, fixed: torch.Tensor, moving: torch.Tensor, *, reg_fac: float, proj_seed: int, pad: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solver and refined offsets of one pair in one pass; the CST code runs the stream twice to get both."""
    offsets = solver_offsets(net, fixed, moving, pad=pad)
    return offsets, refine(net, fixed, moving, offsets, reg_fac=reg_fac, proj_seed=proj_seed)


def half(offsets: torch.Tensor) -> torch.Tensor:
    return F.interpolate(offsets, scale_factor=0.5, mode="trilinear")


def voxel_error(a: torch.Tensor, b: torch.Tensor, full_shape: tuple[int, ...]) -> torch.Tensor:
    """Per-voxel Euclidean distance (full-resolution voxels, times TRE_SCALE) between two half-resolution offset
    fields over the interior, as the CST/CoRe training loss and sample difficulty."""
    scale = a.new_tensor([full_shape[2] / 2, full_shape[1] / 2, full_shape[0] / 2]).view(1, 3, 1, 1, 1)
    inner = (slice(None), slice(None), *(slice(TRE_CROP, -TRE_CROP),) * 3)
    # vector_norm has a zero subgradient at a zero residual, where sum(square).sqrt() gives NaN
    return torch.linalg.vector_norm((a - b)[inner] * scale, dim=1) * TRE_SCALE


def sampling_weights(difficulty: torch.Tensor) -> torch.Tensor:
    """CST's weighted pair sampling: rank by difficulty, sigmoid weights from 5 (easiest) to -5 (hardest)."""
    weights = torch.zeros(len(difficulty))
    weights[torch.argsort(difficulty)] = torch.sigmoid(torch.linspace(5, -5, len(difficulty)))
    return weights


def random_affine_pair(strength: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A1 = I + 1.5 s N, A2 = (I + s N) A1, as data_utils.augment_affine_nl."""
    a1 = torch.eye(4) + torch.randn((4, 4), generator=generator) * strength * 1.5
    a1[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    b = torch.eye(4) + torch.randn((4, 4), generator=generator) * strength
    b[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    return a1, b @ a1


def adjust_target(offsets: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """Pseudo offsets u of the pair (fixed, moving) -> offsets of (fixed o A1, moving o A2):
    u'(x) = A2^-1 (A1 x + u(A1 x)) - x. The CST code (data_utils.augment_affine_nl) samples u at
    expm(log A1 - log A2) x, which is exact only for commuting matrices and drops the linear part of A2^-1 on u."""
    shape = tuple(offsets.shape[2:])
    point = F.affine_grid(a1[:3].unsqueeze(0).to(offsets), (1, 1, *shape), align_corners=False)
    point = point + F.grid_sample(offsets, point, align_corners=False).permute(0, 2, 3, 4, 1)
    inverse = torch.linalg.inv(a2.double()).to(offsets)
    mapped = point @ inverse[:3, :3].T + inverse[:3, 3]
    return (mapped - identity_grid(shape, offsets)).permute(0, 4, 1, 2, 3)


def augment_pair(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    target_half: torch.Tensor,
    strength: float,
    generator: torch.Generator,
    pads: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CST's affine input augmentation: images warped by A1 and A2 (padded with their background values), the target
    smoothed through the quarter grid as in the CST code and adjusted exactly (adjust_target)."""
    a1, a2 = random_affine_pair(strength, generator)
    shape = tuple(fixed.shape[2:])
    smooth = F.interpolate(F.interpolate(target_half, scale_factor=0.5, mode="trilinear"), size=shape, mode="trilinear")
    target = half(adjust_target(smooth, a1, a2))
    return warp_affine(fixed, a1, pads[0]), warp_affine(moving, a2, pads[1]), target


def equivariance_points(mask: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """`count` distinct feature-grid positions inside the mask and at least POINT_MARGIN cells from the border. The
    CoRe code divides already-downsampled indices once more (train.py:518-525) and so draws from a small corner with
    repeats."""
    inner = torch.zeros_like(mask)
    m = POINT_MARGIN
    inner[m:-m, m:-m, m:-m] = True
    candidates = torch.nonzero(mask & inner)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} feature positions inside the mask, {count} requested")
    pick = torch.randperm(len(candidates), generator=generator)[:count]
    return candidates[pick.to(candidates.device)]


def equivariance_nce(anchor: torch.Tensor, other: torch.Tensor, temperature: float) -> torch.Tensor:
    """Eq. (3) of CoRe: anchor j against its counterpart and against every other sampled vector of both views. The
    CoRe code takes negatives from the second view only."""
    a, b = F.normalize(anchor.float(), dim=-1), F.normalize(other.float(), dim=-1)
    same = (a @ a.T / temperature).masked_fill(torch.eye(len(a), dtype=torch.bool, device=a.device), float("-inf"))
    logits = torch.cat([a @ b.T / temperature, same], dim=1)
    return F.cross_entropy(logits, torch.arange(len(a), device=a.device))


def equivariance_loss(
    net: nn.Sequential,
    image: torch.Tensor,
    brain: torch.Tensor,
    affine: torch.Tensor,
    count: int,
    temperature: float,
    generator: torch.Generator,
    pad: float,
) -> torch.Tensor:
    """InfoNCE between the transformed features of an image, G_A = T(G(I)), and the features of the transformed
    image, G_B = G(T(I)), at positions inside the transformed brain mask. Image and features are warped with the same
    align_corners=False affine; the CoRe code mixes conventions there."""
    trunk = net[:TRUNK_DEPTH]
    features = trunk(image).float()
    grid = F.affine_grid(affine[:3].unsqueeze(0).to(features), (1, 1, *features.shape[2:]), align_corners=False)
    view_a = F.grid_sample(features, grid, align_corners=False)
    view_b = trunk(warp_affine(image, affine, pad))
    mask = warp_affine(brain.float(), affine)
    mask = F.interpolate(mask, size=features.shape[2:], mode="area")[0, 0] > 0.5
    points = equivariance_points(mask, count, generator)

    def at(view):
        return view[0, :, points[:, 0], points[:, 1], points[:, 2]].T

    return equivariance_nce(at(view_a), at(view_b), temperature)


def make_pairs(n: int, per_subject: int, seed: int) -> list[tuple[int, int]]:
    """(fixed, moving) index pairs: every subject `per_subject` times as fixed and as moving, no self or repeated
    pairs. All pairs of 394 OASIS images would need a 744 GiB pseudo-label buffer."""
    generator = torch.Generator().manual_seed(seed)
    taken: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for _ in range(per_subject):
        while True:
            perm = torch.randperm(n, generator=generator).tolist()
            batch = [(i, perm[i]) for i in range(n)]
            if all(i != j and (i, j) not in taken for i, j in batch):
                break
        taken.update(batch)
        pairs.extend(batch)
    return pairs
