"""Synthetic registration targets from F = I o b and M = I o a, with the known correspondence T_FM = a^{-1} o b.

Each warp is a small affine map followed by invertible radial maps; coordinates are voxel indices (z, y, x). All
targets share the anchors and their clean-view brain masks. The source draws from its own generator, so synthetic
steps do not change the sequence of real training pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from utils import pkload
from utils.field import _warp

ANCHORS = 2048
MARGIN = 8
OVERSAMPLE = 4
SET_RADIUS = 4  # set candidates: integer offsets with |delta| <= 4
AFFINE_SCALE = 0.03
SHIFT = 2.0
RADIAL_COUNT = (1, 3)
SIGMA = (8.0, 14.0)
ALPHA = (-0.6, 1.5)
GAMMA = (0.8, 1.25)
BIAS = 0.1
NOISE = 0.01
TAU_QUANTILE = 0.95
TARGETS = ("point", "set", "structure", "deadzone")


def ball_offsets(radius: int = SET_RADIUS) -> torch.Tensor:
    """Integer offsets with Euclidean norm <= radius, the zero offset first."""
    r = torch.arange(-radius, radius + 1, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    grid = grid[(grid**2).sum(-1) <= radius**2]
    return torch.cat([torch.zeros(1, 3), grid[(grid != 0).any(-1)]])


def patch_offsets(size: int, stride: int) -> torch.Tensor:
    half = (size - 1) // 2
    r = torch.arange(-half, half + 1, stride, dtype=torch.float32)
    return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)


@dataclass
class Affine:
    matrix: torch.Tensor  # (3, 3)
    center: torch.Tensor  # (3,)
    shift: torch.Tensor  # (3,)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.center + (p - self.center) @ self.matrix.T + self.shift

    def inverse(self, p: torch.Tensor) -> torch.Tensor:
        return self.center + (p - self.center - self.shift) @ torch.linalg.inv(self.matrix).T


@dataclass
class Radial:
    center: torch.Tensor  # (3,)
    sigma: float
    alpha: float

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        d = p - self.center
        r2 = (d**2).sum(-1, keepdim=True)
        return self.center + d * (1.0 + self.alpha * torch.exp(-r2 / (2 * self.sigma**2)))

    def inverse(self, p: torch.Tensor, iters: int = 40) -> torch.Tensor:
        # rho * (1 + alpha * exp(-rho^2 / (2 sigma^2))) is strictly increasing for -0.9 < alpha < 2.
        d = p - self.center
        r = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        rho = r.clone()
        s2 = self.sigma**2
        for _ in range(iters):
            e = torch.exp(-(rho**2) / (2 * s2))
            f = rho * (1.0 + self.alpha * e) - r
            df = 1.0 + self.alpha * e * (1.0 - rho**2 / s2)
            rho = (rho - f / df).clamp_min(0.0)
        scale = torch.where(r > 0, rho / r.clamp_min(1e-12), torch.ones_like(r))
        return self.center + d * scale


@dataclass
class Warp:
    affine: Affine
    radials: list[Radial]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        p = self.affine.forward(p)
        for radial in self.radials:
            p = radial.forward(p)
        return p

    def inverse(self, p: torch.Tensor) -> torch.Tensor:
        for radial in reversed(self.radials):
            p = radial.inverse(p)
        return self.affine.inverse(p)


@dataclass
class AnchorTargets:
    anchors: torch.Tensor  # (n, 3) long, voxel indices on the fixed grid of this direction
    points: torch.Tensor  # (n, 3) T at the anchors, voxel coordinates of the moving view
    members: torch.Tensor | None = None  # (n, B, 3) set candidates
    member_mask: torch.Tensor | None = None  # (n, B) admissible candidates; the exact point is always admissible
    normal_weight: torch.Tensor | None = None  # (n, 3, 3) structure-tensor weight


@dataclass
class KnownWarpPair:
    fixed: torch.Tensor  # (1, 1, D, H, W) F after the intensity change
    moving: torch.Tensor  # M after the intensity change
    clean_fixed: torch.Tensor
    clean_moving: torch.Tensor
    fm: AnchorTargets  # anchors on F's grid, targets in M
    mf: AnchorTargets  # anchors on M's grid, targets in F
    valid_fraction: float  # observable anchors among brain candidates, both directions


def _grid_points(shape, device) -> torch.Tensor:
    axes = [torch.arange(n, device=device, dtype=torch.float32) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def sample_points(image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Trilinear values of image (1, C, D, H, W) at voxel coordinates points (..., 3); zero outside."""
    shape = image.shape[2:]
    scale = torch.tensor([2.0 / (n - 1) for n in shape], device=points.device)
    norm = points * scale - 1.0
    grid = norm.reshape(1, -1, 1, 1, 3)[..., [2, 1, 0]]
    out = F.grid_sample(image, grid, mode="bilinear", align_corners=True)
    return out.reshape(image.shape[1], *points.shape[:-1]).movedim(0, -1)


def descriptors(image: torch.Tensor, centers: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-norm patches of image at centers (..., 3); a flat patch is the zero descriptor."""
    values = sample_points(image, centers[..., None, :] + offsets.to(centers.device))[..., 0]
    values = values - values.mean(-1, keepdim=True)
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, values / norm.clamp_min(1e-6), torch.zeros_like(values))


def _gaussian_blur(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    channels = volume.shape[1]
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = k.numel()
        kernel = k.view(1, 1, *shape).repeat(channels, 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = radius
        volume = F.conv3d(F.pad(volume, pad, mode="replicate"), kernel, groups=channels)
    return volume


def structure_weights(image: torch.Tensor, points: torch.Tensor, grad_sigma=1.0, window_sigma=2.0) -> torch.Tensor:
    """Anisotropic weight N at points: the structure tensor of image, its eigenvalues divided by the 99th percentile of
    the largest eigenvalue over these points and clipped to [0, 1]. A strong edge gives about the projector on its normal,
    a strong isotropic texture about the identity, a flat region about zero."""
    smooth = _gaussian_blur(image, grad_sigma)
    gz, gy, gx = torch.gradient(smooth[0, 0])
    g = (gz, gy, gx)
    products = torch.stack([g[i] * g[j] for i in range(3) for j in range(i, 3)])[None]
    tensor = _gaussian_blur(products, window_sigma)
    values = sample_points(tensor, points)  # (n, 6)
    j = torch.zeros(points.shape[0], 3, 3, device=points.device)
    idx = 0
    for a in range(3):
        for b in range(a, 3):
            j[:, a, b] = j[:, b, a] = values[:, idx]
            idx += 1
    evals, evecs = torch.linalg.eigh(j)
    reference = torch.quantile(evals[:, -1], 0.99).clamp_min(1e-12)
    evals = (evals / reference).clamp(0.0, 1.0)
    return evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)


class KnownWarpSource:
    """Draws known-warp pairs from real training images on the device, from its own seeded generator."""

    def __init__(
        self, files, device, seed: int, anchors: int = ANCHORS, target: str = "point", stats: dict | None = None
    ):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        if target == "set" and not stats:
            raise ValueError("the set target needs thresholds from `python -m experiments.eval_known_warp --stats`")
        self.files = sorted(files)
        if not self.files or anchors < 1:
            raise ValueError("known-warp sampling needs training images and a positive anchor count")
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.anchors = anchors
        self.target = target
        self.stats = stats or {}
        self.ball = ball_offsets().to(self.device)
        self.small = patch_offsets(3, 1).to(self.device)
        self.large = patch_offsets(7, 2).to(self.device)

    def _uniform(self, low, high, size=()):
        return low + (high - low) * torch.rand(size, generator=self.generator, device=self.device)

    def _image(self) -> torch.Tensor:
        index = int(torch.randint(len(self.files), (1,), generator=self.generator, device=self.device))
        img = pkload(self.files[index])[0]
        return torch.from_numpy(np.ascontiguousarray(img)).to(self.device, torch.float32)[None, None]

    def _view(self, image: torch.Tensor, warp: Warp) -> torch.Tensor:
        """The clean view image o warp, one trilinear interpolation of the source."""
        grid = _grid_points(image.shape[2:], self.device)
        return _warp(image, (warp.forward(grid) - grid).permute(3, 0, 1, 2)[None])

    def _interior_brain(self, clean: torch.Tensor, count: int) -> torch.Tensor:
        """Random brain voxels of a clean view at least MARGIN voxels inside the volume."""
        shape = torch.tensor(clean.shape[2:], device=self.device)
        candidates = torch.nonzero(clean[0, 0] > 0)
        candidates = candidates[((candidates >= MARGIN) & (candidates <= shape - 1 - MARGIN)).all(-1)]
        if not len(candidates):
            raise ValueError(f"no brain voxels have the required {MARGIN}-voxel interior support")
        pick = torch.randint(len(candidates), (count,), generator=self.generator, device=self.device)
        return candidates[pick]

    def _warp(self, image: torch.Tensor) -> Warp:
        shape = torch.tensor(image.shape[2:], device=self.device, dtype=torch.float32)
        center = (shape - 1) / 2
        matrix = torch.eye(3, device=self.device) + AFFINE_SCALE * (2 * self._uniform(0, 1, (3, 3)) - 1)
        affine = Affine(matrix, center, SHIFT * (2 * self._uniform(0, 1, (3,)) - 1))
        brain = torch.nonzero(image[0, 0] > 0).float()
        box = ((brain >= shape / 4) & (brain < 3 * shape / 4)).all(-1)
        central = brain[box]
        dark = central[image[0, 0][tuple(central.long().T)] < 0.5 * image[0, 0][image[0, 0] > 0].median()]
        count = int(
            torch.randint(RADIAL_COUNT[0], RADIAL_COUNT[1] + 1, (1,), generator=self.generator, device=self.device)
        )
        radials = []
        for k in range(count):
            pool = dark if (k % 2 == 0 and len(dark)) else brain
            centre = pool[int(torch.randint(len(pool), (1,), generator=self.generator, device=self.device))]
            radials.append(Radial(centre, float(self._uniform(*SIGMA)), float(self._uniform(*ALPHA))))
        return Warp(affine, radials)

    def _appearance(self, clean: torch.Tensor) -> torch.Tensor:
        """Independent intensity change of one view: gamma, smooth multiplicative bias, noise inside the brain only."""
        brain = clean > 0
        out = clean.clamp_min(0) ** float(self._uniform(*GAMMA))
        low = torch.randn((1, 1, 4, 4, 4), generator=self.generator, device=self.device)
        bias = torch.exp(BIAS * F.interpolate(low, size=clean.shape[2:], mode="trilinear", align_corners=True))
        noise = NOISE * torch.randn(clean.shape, generator=self.generator, device=self.device)
        return torch.where(brain, out * bias + noise, torch.zeros_like(clean))

    def _anchors(self, clean_fixed, clean_moving, source_of, target_of) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Up to ``anchors`` observable anchors on the fixed grid, their targets, and valid/candidate counts."""
        shape = torch.tensor(clean_fixed.shape[2:], device=self.device, dtype=torch.float32)
        candidates = self._interior_brain(clean_fixed, OVERSAMPLE * self.anchors)
        points = candidates.float()
        source, target = source_of(points), target_of(points)

        def inside(p):
            return ((p >= MARGIN) & (p <= shape - 1 - MARGIN)).all(-1)

        valid = inside(source) & inside(target) & (sample_points(clean_moving, target)[..., 0] > 0)
        kept = torch.nonzero(valid)[:, 0][: self.anchors]
        return candidates[kept], target[kept], int(valid.sum()), len(candidates)

    def _targets(self, anchors, points, clean_target) -> AnchorTargets:
        targets = AnchorTargets(anchors, points)
        if self.target == "set":
            targets.members, targets.member_mask = self.set_members(clean_target, points)
        elif self.target == "structure":
            targets.normal_weight = structure_weights(clean_target, points)
        return targets

    def _set_distances(self, clean_target, points):
        members = points[:, None, :] + self.ball[None]
        ref_small = descriptors(clean_target, points, self.small)
        ref_large = descriptors(clean_target, points, self.large)
        d_small = ((descriptors(clean_target, members, self.small) - ref_small[:, None]) ** 2).sum(-1)
        d_large = ((descriptors(clean_target, members, self.large) - ref_large[:, None]) ** 2).sum(-1)
        return d_small, d_large

    def set_members(self, clean_target, points, tau=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidates y* + delta whose clean-view patches match y* at both scales."""
        tau_small, tau_large = tau or (self.stats["tau_small"], self.stats["tau_large"])
        d_small, d_large = self._set_distances(clean_target, points)
        mask = (d_small <= tau_small) & (d_large <= tau_large)
        mask[:, 0] = True
        return points[:, None, :] + self.ball[None], mask

    @torch.no_grad()
    def sample(self) -> KnownWarpPair:
        """One pair and its targets, in float32 whatever the caller's autocast: the geometry must stay exact."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            image = self._image()
            a, b = self._warp(image), self._warp(image)
            clean_fixed, clean_moving = self._view(image, b), self._view(image, a)
            fm = self._anchors(clean_fixed, clean_moving, b.forward, lambda p: a.inverse(b.forward(p)))
            mf = self._anchors(clean_moving, clean_fixed, a.forward, lambda p: b.inverse(a.forward(p)))
            if not len(fm[0]) or not len(mf[0]):
                raise ValueError("synthetic warp leaves no observable anchors in one direction")
            return KnownWarpPair(
                fixed=self._appearance(clean_fixed),
                moving=self._appearance(clean_moving),
                clean_fixed=clean_fixed,
                clean_moving=clean_moving,
                fm=self._targets(fm[0], fm[1], clean_moving),
                mf=self._targets(mf[0], mf[1], clean_fixed),
                valid_fraction=(fm[2] + mf[2]) / max(fm[3] + mf[3], 1),
            )


def effective_points(flow: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Psi at integer anchors of a stored CTCF flow: s * (x + u(x)) - 1/2, the point the legacy sampler reads."""
    shape = flow.shape[2:]
    u = flow[0, :, anchors[:, 0], anchors[:, 1], anchors[:, 2]].T.float()
    scale = torch.tensor([n / (n - 1) for n in shape], device=flow.device)
    return (anchors.float() + u) * scale - 0.5


def anchor_loss(flow: torch.Tensor, targets: AnchorTargets, kind: str, radius: float = 0.0) -> torch.Tensor:
    """Mean anchor loss of one direction: point, set (distance to the nearest admissible candidate), structure
    (e^T N e) or dead zone (squared excess of |e| over ``radius``), e = Psi(x) - T(x)."""
    with torch.autocast(device_type=flow.device.type, enabled=False):
        psi = effective_points(flow, targets.anchors)
        error = psi - targets.points
        if kind == "point":
            return (error**2).sum(-1).mean()
        if kind == "set":
            d2 = ((psi[:, None, :] - targets.members) ** 2).sum(-1).masked_fill(~targets.member_mask, float("inf"))
            return d2.min(dim=1).values.mean()
        if kind == "structure":
            return torch.einsum("ni,nij,nj->n", error, targets.normal_weight, error).mean()
        if kind == "deadzone":
            return F.relu(torch.linalg.vector_norm(error, dim=-1) - radius).pow(2).mean()
    raise ValueError(f"unknown target {kind!r}")


@torch.no_grad()
def estimate_stats(files, device, samples: int, seed: int, points_per_view: int = 256) -> dict:
    """Set thresholds: the TAU_QUANTILE of descriptor distances of the same position between two independent intensity
    changes of a clean view. With them, the dead-zone radius (mean widest admissible offset of a set) and descriptive
    shares of the sets. Draws from its own generator; training streams are untouched."""
    source = KnownWarpSource(files, device, seed=seed, target="point")
    same = {"small": [], "large": []}
    distances = []
    for _ in range(samples):
        image = source._image()
        clean = source._view(image, source._warp(image))
        points = source._interior_brain(clean, points_per_view).float()
        first, second = source._appearance(clean), source._appearance(clean)
        for name, offsets in (("small", source.small), ("large", source.large)):
            d = ((descriptors(first, points, offsets) - descriptors(second, points, offsets)) ** 2).sum(-1)
            same[name].append(d.cpu())
        distances.append(tuple(d.cpu() for d in source._set_distances(clean, points)))
    tau = (
        float(torch.quantile(torch.cat(same["small"]), TAU_QUANTILE)),
        float(torch.quantile(torch.cat(same["large"]), TAU_QUANTILE)),
    )
    widths, sizes = [], []
    reach = torch.linalg.vector_norm(source.ball.cpu(), dim=-1)
    for d_small, d_large in distances:
        mask = (d_small <= tau[0]) & (d_large <= tau[1])
        mask[:, 0] = True
        widths.append(reach[None].expand_as(mask).masked_fill(~mask, 0.0).max(dim=1).values)
        sizes.append(mask.sum(1).float())
    widths, sizes = torch.cat(widths), torch.cat(sizes)
    return {
        "tau_small": tau[0],
        "tau_large": tau[1],
        "tau_quantile": TAU_QUANTILE,
        "deadzone_radius": float(widths.mean()),
        "set_size_median": float(sizes.median()),
        "fraction_singleton": float((sizes == 1).float().mean()),
        "fraction_full_ball": float((sizes == len(source.ball)).float().mean()),
        "ball_size": len(source.ball),
        "samples": samples,
        "points_per_view": points_per_view,
        "seed": seed,
    }
