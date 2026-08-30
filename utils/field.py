"""Displacement-field geometry: warping, composition, Jacobians, folds and certified repair.

Every field in this module is a Phi displacement in **voxel units**, channel-first
``(B, 3, D, H, W)``, on a unit-spaced grid, with channel ``c`` paired to spatial axis ``c``.
That is the CTCF convention; it is not the Psi source-index displacement the deployment warp
samples, and it is not a normalized-grid flow. Converting between the two is
``tools.analysis.search.transaction.phi_to_psi_displacement`` and its inverse.

Fold and determinant results here come from several *different* schemes that do not agree:
the central-difference ``jacobian_det``, the digital corner/star determinants, and the
trilinear cell bounds each answer a different question. A count from one is not a count from
another, so a number taken out of this module must travel with the name of the predicate that
produced it. ``trilinear_cert_bound`` and ``certified_local_clip`` decide *sufficient*
predicates: failing one means "not certified by this predicate", never "this map folds".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _crop_spatial(t: torch.Tensor, n: int) -> torch.Tensor:
    """Crop n voxels from each spatial side for [B,C,D,H,W] tensors."""
    if n <= 0:
        return t
    if min(t.shape[-3:]) <= 2 * n:
        return t
    return t[..., n:-n, n:-n, n:-n]


def _warp(tensor: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp tensor by dense voxel-unit flow on a unit-spaced grid."""
    _, _, d, h, w = tensor.shape
    device = tensor.device
    zz = torch.arange(end=d, device=device)
    yy = torch.arange(end=h, device=device)
    xx = torch.arange(end=w, device=device)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=0).float().unsqueeze(0)
    new_locs = grid + flow
    new_locs[:, 0] = 2.0 * (new_locs[:, 0] / (d - 1) - 0.5)
    new_locs[:, 1] = 2.0 * (new_locs[:, 1] / (h - 1) - 0.5)
    new_locs[:, 2] = 2.0 * (new_locs[:, 2] / (w - 1) - 0.5)
    grid_sample_grid = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    # Cascade-composition convention: align_corners=True here, deliberately unlike
    # SpatialTransformer's align_corners=False. Trained checkpoints depend on both — do not unify.
    return F.grid_sample(tensor, grid_sample_grid, mode=mode, align_corners=True)


def compose_flows(flow_ab: torch.Tensor, flow_bc: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Compose flows A->B and B->C into A->C in voxel units."""
    return flow_ab + _warp(flow_bc, flow_ab, mode=mode)


def integrate_svf(vel: torch.Tensor, st, steps: int = 7) -> torch.Tensor:
    """Integrate a stationary velocity field into a displacement via scaling-and-squaring."""
    disp = vel * (1.0 / (2**steps))
    for _ in range(steps):
        disp = disp + st(disp, disp)
    return disp


def jacobian_det(flow: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant map for a 3D displacement field [B,3,D,H,W]."""
    dz = flow[:, :, 2:, :, :] - flow[:, :, :-2, :, :]
    dy = flow[:, :, :, 2:, :] - flow[:, :, :, :-2, :]
    dx = flow[:, :, :, :, 2:] - flow[:, :, :, :, :-2]
    dz = F.pad(dz, pad=(0, 0, 0, 0, 1, 1)) * 0.5
    dy = F.pad(dy, pad=(0, 0, 1, 1, 0, 0)) * 0.5
    dx = F.pad(dx, pad=(1, 1, 0, 0, 0, 0)) * 0.5

    fz_z, fz_y, fz_x = dz[:, 0], dy[:, 0], dx[:, 0]
    fy_z, fy_y, fy_x = dz[:, 1], dy[:, 1], dx[:, 1]
    fx_z, fx_y, fx_x = dz[:, 2], dy[:, 2], dx[:, 2]

    j00 = 1.0 + fz_z
    j01 = fz_y
    j02 = fz_x
    j10 = fy_z
    j11 = 1.0 + fy_y
    j12 = fy_x
    j20 = fx_z
    j21 = fx_y
    j22 = 1.0 + fx_x

    det = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
    return det.unsqueeze(1)


def _neg_jac_penalty_from_det(
    det: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 1,
    eps: float = 0.0,
) -> torch.Tensor:
    pen = torch.relu(-_crop_spatial(det, crop) + eps)

    if mask is None:
        return pen.mean()

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    m = (mask > 0).to(pen.dtype)
    m = _crop_spatial(m, crop)
    denom = torch.clamp(m.sum(), min=1.0)
    return (pen * m).sum() / denom


def neg_jacobian_penalty(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 1,
    eps: float = 0.0,
) -> torch.Tensor:
    """Mean penalty over non-positive Jacobian determinant voxels.
    `eps` > 0 widens the band to detJ < eps; 0.0 is the form every trained checkpoint depends on.
    """
    return _neg_jac_penalty_from_det(jacobian_det(flow), mask, crop, eps)


_AXIS_MODES = (
    ("+", "+", "+"), ("+", "+", "-"), ("+", "-", "+"), ("+", "-", "-"),
    ("-", "+", "+"), ("-", "+", "-"), ("-", "-", "+"), ("-", "-", "-"),
)  # fmt: skip


def _one_sided_diff(t: torch.Tensor, axis: int, mode: str) -> torch.Tensor:
    """Forward ('+') or backward ('-') difference of [3,D,H,W] along a spatial axis, edge-clamped."""
    dim = axis + 1
    n = t.shape[dim]
    idx = torch.arange(n, device=t.device)
    if mode == "+":
        return t.index_select(dim, torch.clamp(idx + 1, max=n - 1)) - t
    return t - t.index_select(dim, torch.clamp(idx - 1, min=0))


_STAR_OFFSETS = (
    ((-1, -1, 0), (-1, 0, -1), (0, -1, -1)),
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
)  # J1*, J2* — the two tetrahedralisations of the cell; face-diagonal, not axis-aligned


def _shift_diff(t: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    """t[p + offset] - t[p] for [3,D,H,W], edge-clamped like the one-sided differences."""
    shifted = t
    for axis, off in enumerate(offset):
        if off == 0:
            continue
        dim = axis + 1
        n = shifted.shape[dim]
        idx = torch.clamp(torch.arange(n, device=t.device) + off, 0, n - 1)
        shifted = shifted.index_select(dim, idx)
    return shifted - t


def _det3(gx: torch.Tensor, gy: torch.Tensor, gz: torch.Tensor) -> torch.Tensor:
    """Determinant of the 3x3 matrix built from three difference vectors of [3,D,H,W]."""
    return (
        gx[0] * (gy[1] * gz[2] - gy[2] * gz[1])
        - gx[1] * (gy[0] * gz[2] - gy[2] * gz[0])
        + gx[2] * (gy[0] * gz[1] - gy[1] * gz[0])
    )


def digital_fold_percent(
    flow: torch.Tensor, corners_only: bool = False, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Percent of voxels failing the digital diffeomorphism criterion of Liu et al., IJCV 2024
    (doi:10.1007/s11263-024-02047-1): all ten determinants positive — the 8 one-sided combinations
    plus J1*/J2*. `corners_only=True` tests only the 8. `mask` restricts the count to the brain
    interior; `mask=None` is the whole-volume count and stays byte-identical to the frozen form.
    """
    disp = flow[0]
    d, h, w = disp.shape[1:]
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=flow.device),
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    trans = disp + torch.stack([zz, yy, xx], dim=0).to(disp.dtype)

    all_pos = None
    for mx, my, mz in _AXIS_MODES:
        det = _det3(
            _one_sided_diff(trans, 0, mx),
            _one_sided_diff(trans, 1, my),
            _one_sided_diff(trans, 2, mz),
        )[1:-1, 1:-1, 1:-1]
        pos = det > 0.0
        all_pos = pos if all_pos is None else (all_pos & pos)

    if not corners_only:
        for ox, oy, oz in _STAR_OFFSETS:
            det = _det3(
                _shift_diff(trans, ox),
                _shift_diff(trans, oy),
                _shift_diff(trans, oz),
            )[1:-1, 1:-1, 1:-1]
            all_pos = all_pos & (det > 0.0)

    fold = (~all_pos).to(flow.dtype)
    if mask is None:
        return fold.mean() * 100.0
    return _interior_masked_mean(fold, mask) * 100.0


def _digital_determinants(flow: torch.Tensor, corners_only: bool = False) -> list[torch.Tensor]:
    """The ten (or eight, `corners_only`) differentiable determinant maps of the digital criterion,
    each cropped to the interior [1:-1,1:-1,1:-1]. Raw values for a hinge/barrier to act on; kept
    separate from the frozen `digital_fold_percent`, whose sign count it reproduces.
    """
    disp = flow[0]
    d, h, w = disp.shape[1:]
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=flow.device),
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    trans = disp + torch.stack([zz, yy, xx], dim=0).to(disp.dtype)

    dets = [
        _det3(
            _one_sided_diff(trans, 0, mx),
            _one_sided_diff(trans, 1, my),
            _one_sided_diff(trans, 2, mz),
        )[1:-1, 1:-1, 1:-1]
        for mx, my, mz in _AXIS_MODES
    ]
    if not corners_only:
        for ox, oy, oz in _STAR_OFFSETS:
            dets.append(_det3(_shift_diff(trans, ox), _shift_diff(trans, oy), _shift_diff(trans, oz))[1:-1, 1:-1, 1:-1])
    return dets


def _interior_masked_mean(pen_map: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of an interior [D-2,H-2,W-2] map, over the brain if `mask` is given.
    The mask (any of [.,.,D,H,W] / [.,D,H,W] / [D,H,W]) is cropped to the same interior.
    """
    if mask is None:
        return pen_map.mean()
    m = mask
    while m.dim() > 3:
        m = m[0]
    m = (m[1:-1, 1:-1, 1:-1] > 0).to(pen_map.dtype)
    return (pen_map * m).sum() / torch.clamp(m.sum(), min=1.0)


def digital_fold_penalty(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """Differentiable hinge sum(relu(eps - det_k)) over the ten digital determinants, restricted to
    the brain interior when `mask` is given. eps>0 widens the band to det < eps (a soft margin);
    since J1*/J2* have identity scale 2 vs 1 for the corners, a shared eps>0 margins the corners tighter.
    """
    pen_map = None
    for det in _digital_determinants(flow):
        h = torch.relu(eps - det)
        pen_map = h if pen_map is None else pen_map + h
    return _interior_masked_mean(pen_map, mask)


def digital_penalty_and_folds(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Hinge penalty (honours `mask`) and the strict fold percentage (always whole-interior, matching
    `digital_fold_percent`) in one pass over the ten determinants.
    """
    pen_map = None
    all_pos = None
    for det in _digital_determinants(flow):
        h = torch.relu(eps - det)
        pen_map = h if pen_map is None else pen_map + h
        pos = det > 0.0
        all_pos = pos if all_pos is None else (all_pos & pos)
    pen = _interior_masked_mean(pen_map, mask)
    with torch.no_grad():
        folds = float((~all_pos).to(flow.dtype).mean().item() * 100.0)
    return pen, folds


# Identity-map value of each determinant (1 for the corner tetrahedra, 2 for the face-diagonal
# J1*/J2*); normalising by it lets one threshold `t` mean the same on all ten.
_DET_IDENTITY = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0)


def _shifted_relaxed_barrier(det: torch.Tensor, scale: float, t: float, u_min: float = 0.1) -> torch.Tensor:
    """One-sided C1 log-barrier on one determinant, normalised by its identity `scale`. With
    u = det/(scale*t): 0 for u >= 1; -log(u)+(u-1) for u_min <= u < 1 (-> +inf as det -> 0); a C1-linear
    extension for u < u_min, keeping a still-folded start (det <= 0) finite rather than NaN. `t` is the
    fraction of the identity determinant at which it engages.
    """
    u = det / (scale * t)
    safe = torch.clamp(u, min=u_min)
    val = -torch.log(safe) + (safe - 1.0)
    v0 = -math.log(u_min) + (u_min - 1.0)
    s0 = -1.0 / u_min + 1.0
    lin = v0 + s0 * (u - u_min)
    return torch.where(u >= 1.0, torch.zeros_like(u), torch.where(u >= u_min, val, lin))


def digital_barrier_penalty(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    t: float = 0.1,
) -> torch.Tensor:
    """Sum of the one-sided relaxed log-barriers over the ten determinants, restricted to the brain
    interior when `mask` is given. `t` sets the engagement threshold; finite on a folded start.
    """
    pen_map = None
    for det, scale in zip(_digital_determinants(flow), _DET_IDENTITY, strict=True):
        b = _shifted_relaxed_barrier(det, scale, t)
        pen_map = b if pen_map is None else pen_map + b
    return _interior_masked_mean(pen_map, mask)


def digital_barrier_and_folds(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    t: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Relaxed digital log-barrier penalty and the strict whole-interior fold percentage, one pass."""
    pen_map = None
    all_pos = None
    for det, scale in zip(_digital_determinants(flow), _DET_IDENTITY, strict=True):
        b = _shifted_relaxed_barrier(det, scale, t)
        pen_map = b if pen_map is None else pen_map + b
        pos = det > 0.0
        all_pos = pos if all_pos is None else (all_pos & pos)
    pen = _interior_masked_mean(pen_map, mask)
    with torch.no_grad():
        folds = float((~all_pos).to(flow.dtype).mean().item() * 100.0)
    return pen, folds


def _vertex_constraint(
    flow: torch.Tensor,
    fixed_mask: torch.Tensor | None,
    fixed_values: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Validate a vertex constraint and return a broadcastable boolean mask plus fixed values."""
    if fixed_mask is None:
        if fixed_values is not None:
            raise ValueError("fixed_values requires fixed_mask")
        return None
    mask = fixed_mask.to(device=flow.device, dtype=torch.bool)
    if mask.dim() == 3:
        mask = mask[None, None]
    elif mask.dim() == 4:
        mask = mask[:, None]
    if mask.dim() != 5 or mask.shape[0] not in (1, flow.shape[0]) or mask.shape[1] not in (1, flow.shape[1]):
        raise ValueError(f"fixed_mask must broadcast to {tuple(flow.shape)}, got {tuple(mask.shape)}")
    if tuple(mask.shape[-3:]) != tuple(flow.shape[-3:]):
        raise ValueError(f"fixed_mask spatial shape {tuple(mask.shape[-3:])} != flow {tuple(flow.shape[-3:])}")
    values = flow.detach().clone() if fixed_values is None else fixed_values.to(device=flow.device, dtype=flow.dtype)
    if values.shape != flow.shape:
        try:
            values = torch.broadcast_to(values, flow.shape)
        except RuntimeError as exc:
            raise ValueError(f"fixed_values must broadcast to {tuple(flow.shape)}, got {tuple(values.shape)}") from exc
    return mask, values


def _apply_vertex_constraint(
    candidate: torch.Tensor,
    constraint: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    return candidate if constraint is None else torch.where(constraint[0], constraint[1], candidate)


def digital_project(
    flow: torch.Tensor,
    eps: float = 0.0,
    damp: float = 0.6,
    max_iters: int = 80,
    fixed_mask: torch.Tensor | None = None,
    fixed_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, int]:
    """Project a displacement field onto the digital-diffeomorphic set by feathered local relaxation:
    at voxels whose digital determinants fail (det <= eps), blend the displacement toward its local
    (mean-smoothed) value under a blurred weight, re-checking all ten every pass until none fail or
    `max_iters` is reached. Smoothing removes the fold; blending toward the local-smooth field (not
    identity) with a feathered weight avoids the boundary discontinuity that makes a hard contraction
    spawn new folds and run away. Returns (projected flow, residual fold %, passes applied). A zero
    residual certifies every one of the ten determinants is positive; a non-zero residual is returned
    honestly (max_iters reached without a valid field), never as a false certificate.
    """
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [1,3,D,H,W], got {tuple(flow.shape)}.")

    def _smooth(t: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(F.pad(t, (1, 1, 1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)

    constraint = _vertex_constraint(flow, fixed_mask, fixed_values)
    out = _apply_vertex_constraint(flow.detach().clone(), constraint)
    d, h, w = out.shape[2:]
    applied = 0
    with torch.no_grad():
        for _ in range(max_iters):
            fail_interior = None
            for det in _digital_determinants(out):
                bad = det <= eps
                fail_interior = bad if fail_interior is None else (fail_interior | bad)
            if not bool(fail_interior.any()):
                break
            fail = torch.zeros((1, 1, d, h, w), device=out.device, dtype=out.dtype)
            fail[0, 0, 1:-1, 1:-1, 1:-1] = fail_interior.to(out.dtype)
            feather = _smooth(_smooth(fail)).clamp(0.0, 1.0)  # blurred mask: no hard boundary
            candidate = out * (1.0 - damp * feather) + _smooth(out) * (damp * feather)
            out = _apply_vertex_constraint(candidate, constraint)
            applied += 1
        residual = float(digital_fold_percent(out).item())
    return out, residual, applied


def digital_min_det(flow: torch.Tensor) -> float:
    """Smallest of the ten Liu-et-al. digital determinants over all interior voxels — the formal
    certificate quantity. ``min_det >= eps`` proves the field is digitally diffeomorphic with a
    margin of ``eps``; a fold *count* of zero only says min_det > 0, it never exposes the margin."""
    with torch.no_grad():
        return min(float(det.min().item()) for det in _digital_determinants(flow))


def _trilinear_corner_targets(flow: torch.Tensor) -> dict[tuple[int, int, int], torch.Tensor]:
    """Target coordinates phi = index + disp at the eight corners of every unit cell, each
    [3, D-1, H-1, W-1] in (z,y,x) order. This is what grid_sample trilinearly interpolates."""
    disp = flow[0]
    d, h, w = disp.shape[1:]
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=flow.device, dtype=flow.dtype),
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    trans = disp + torch.stack([zz, yy, xx], dim=0)
    return {
        (i, j, k): trans[:, i : i + d - 1, j : j + h - 1, k : k + w - 1] for i in (0, 1) for j in (0, 1) for k in (0, 1)
    }


def _trilinear_det_at(p: dict[tuple[int, int, int], torch.Tensor], a: float, b: float, c: float) -> torch.Tensor:
    """det of the trilinear Jacobian at local coords (a,b,c) in [0,1]^3, for all cells at once.
    Each Jacobian column is the partial of the trilinear map, bilinear in the other two coordinates."""
    na, nb, nc = 1.0 - a, 1.0 - b, 1.0 - c
    col_a = (
        (p[1, 0, 0] - p[0, 0, 0]) * nb * nc
        + (p[1, 1, 0] - p[0, 1, 0]) * b * nc
        + (p[1, 0, 1] - p[0, 0, 1]) * nb * c
        + (p[1, 1, 1] - p[0, 1, 1]) * b * c
    )
    col_b = (
        (p[0, 1, 0] - p[0, 0, 0]) * na * nc
        + (p[1, 1, 0] - p[1, 0, 0]) * a * nc
        + (p[0, 1, 1] - p[0, 0, 1]) * na * c
        + (p[1, 1, 1] - p[1, 0, 1]) * a * c
    )
    col_c = (
        (p[0, 0, 1] - p[0, 0, 0]) * na * nb
        + (p[1, 0, 1] - p[1, 0, 0]) * a * nb
        + (p[0, 1, 1] - p[0, 1, 0]) * na * b
        + (p[1, 1, 1] - p[1, 1, 0]) * a * b
    )
    return (
        col_a[0] * (col_b[1] * col_c[2] - col_b[2] * col_c[1])
        - col_a[1] * (col_b[0] * col_c[2] - col_b[2] * col_c[0])
        + col_a[2] * (col_b[0] * col_c[1] - col_b[1] * col_c[0])
    )


def _trilinear_cell_min_det(flow: torch.Tensor, samples: int = 5) -> torch.Tensor:
    """Per-cell minimum of det J of the trilinear deformation over an SxSxS interior lattice, shape
    [D-1,H-1,W-1]. Sound for DETECTION: a cell whose value is < 0 provably folds (a real sample is
    negative); the sampling only ever under-counts folds, never invents one."""
    flow = flow.detach().float()
    p = _trilinear_corner_targets(flow)
    ts = torch.linspace(0.0, 1.0, samples).tolist()
    cell_min: torch.Tensor | None = None
    for a in ts:
        for b in ts:
            for c in ts:
                det = _trilinear_det_at(p, a, b, c)
                cell_min = det if cell_min is None else torch.minimum(cell_min, det)
    return cell_min


def trilinear_min_det(flow: torch.Tensor, samples: int = 5) -> float:
    """Tight, sound DETECTION of trilinear folding: minimum of det J of the actual trilinear
    (grid_sample) deformation over an SxSxS lattice inside every unit cell. A negative value PROVES
    the applied warp folds; the ten digital determinants only test the cell corners and miss an
    interior dip, so digital-10 positivity does not imply this is >= 0."""
    with torch.no_grad():
        return float(_trilinear_cell_min_det(flow, samples).min().item())


def trilinear_fold_percent(flow: torch.Tensor, samples: int = 5) -> float:
    """Percent of cells that PROVABLY fold under the trilinear warp (some interior sample has det < 0)
    — the interpolation-consistent analogue of digital_fold_percent, and a sound LOWER bound on the
    true trilinear fold fraction (only cells with an actually-negative sample are counted). This is the
    audit headline: how much of a field folds trilinearly even when digital_fold_percent reads zero."""
    with torch.no_grad():
        cell_min = _trilinear_cell_min_det(flow, samples)
        return float((cell_min < 0).to(cell_min.dtype).mean().item() * 100.0)


def _values_to_bernstein_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """3x3 map from a degree-2 polynomial's values at {0, 1/2, 1} to its Bernstein coefficients."""
    nodes = [0.0, 0.5, 1.0]
    vander = [[1.0, t, t * t] for t in nodes]  # power basis at the nodes
    val_to_pow = torch.linalg.inv(torch.tensor(vander, dtype=torch.float64))
    pow_to_bern = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    return (pow_to_bern @ val_to_pow).to(device=device, dtype=dtype)


def _trilinear_bernstein_coeffs(flow: torch.Tensor) -> torch.Tensor:
    """The 27 Bernstein coefficients of det J per cell, [3,3,3,D-1,H-1,W-1], at the input's dtype and
    grad. det J is degree <=2 in each of (a,b,c); these coefficients bound it on the whole cell (convex-
    hull property), so the per-cell minimum is a rigorous lower bound and a hinge on them is a
    differentiable trilinear-fold penalty. Grad-transparent (no detach), so callers pick precision:
    the certificate runs this in float64/no_grad, the training penalty in float32 with grad."""
    p = _trilinear_corner_targets(flow)
    mat = _values_to_bernstein_matrix(flow.device, flow.dtype)
    nodes = (0.0, 0.5, 1.0)
    vals = torch.stack([_trilinear_det_at(p, a, b, c) for a in nodes for b in nodes for c in nodes]).reshape(
        3, 3, 3, *p[0, 0, 0].shape[1:]
    )
    bern = torch.einsum("pa,abcijk->pbcijk", mat, vals)
    bern = torch.einsum("pb,abcijk->apcijk", mat, bern)
    return torch.einsum("pc,abcijk->abpijk", mat, bern)


def _trilinear_subdiv_matrices(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """de Casteljau split-at-midpoint maps for a degree-2 Bernstein polynomial: L takes its 3 coefficients
    to those of its restriction to [0,1/2], R to [1/2,1]. Applied per axis they subdivide a cell into 8."""
    left = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.5, 0.25]], device=device, dtype=dtype)
    right = torch.tensor([[0.25, 0.5, 0.25], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    return left, right


def _trilinear_subdivide_min(coeffs: torch.Tensor, depth: int, mats=None) -> torch.Tensor:
    """Tightest sound lower bound on det J over each cell after `depth` Bernstein (de Casteljau) subdivision
    levels: recursively split every cell into 8 sub-cells and take the min sub-coefficient. Monotone
    (>= the un-subdivided amin) and still sound (<= the true min det — subdivision only raises the bound).
    `coeffs` is [3,3,3,N]; returns [N]."""
    if depth <= 0:
        return coeffs.amin(dim=(0, 1, 2))
    if mats is None:
        mats = _trilinear_subdiv_matrices(coeffs.device, coeffs.dtype)
    left, right = mats
    best: torch.Tensor | None = None
    for m0 in (left, right):
        c0 = torch.einsum("xi,ijkn->xjkn", m0, coeffs)
        for m1 in (left, right):
            c1 = torch.einsum("yj,xjkn->xykn", m1, c0)
            for m2 in (left, right):
                child = torch.einsum("zk,xykn->xyzn", m2, c1)
                m = _trilinear_subdivide_min(child, depth - 1, mats)
                best = m if best is None else torch.minimum(best, m)
    return best


def _trilinear_cell_cert_bound(flow: torch.Tensor, subdiv_depth: int = 0, eps: float = 0.0) -> torch.Tensor:
    """Per-cell sound Bernstein lower bound on det J, [D-1,H-1,W-1], computed in FLOAT64 so the
    calculation avoids float32 rounding but is still an ordinary, non-directed float64 screen. A cell with
    value comfortably above eps passes the mathematical sufficient condition numerically; publication-grade
    machine verification of stored bytes is performed by ``utils.cert_exact``.

    `subdiv_depth` > 0 refines only the cells whose coarse bound is < eps by that many de Casteljau
    subdivision levels, tightening the (conservative) first-level bound and certifying cells it falsely
    flags in exact arithmetic. The float64 implementation is an operational screen, not the final
    machine-sound verdict."""
    with torch.no_grad():
        coeffs = _trilinear_bernstein_coeffs(flow.detach().double())
        bound = coeffs.amin(dim=(0, 1, 2))
        if subdiv_depth > 0:
            suspect = bound < eps
            if bool(suspect.any()):
                bound = bound.clone()
                bound[suspect] = _trilinear_subdivide_min(coeffs[:, :, :, suspect], subdiv_depth)
        return bound


def trilinear_cert_bound(flow: torch.Tensor, subdiv_depth: int = 0, eps: float = 0.0) -> float:
    """Global float64 Bernstein lower-bound estimate over all cells.

    It is the fast operational screen for the sufficient orientation-preservation predicate. Because the
    arithmetic is not outward-rounded, use ``utils.cert_exact`` on the final float32 bytes for a machine-sound
    verdict. ``subdiv_depth`` tightens the bound on sub-eps cells (see `_trilinear_cell_cert_bound`).
    """
    with torch.no_grad():
        return float(_trilinear_cell_cert_bound(flow, subdiv_depth, eps).min().item())


def displacement_grad_norm_max(flow: torch.Tensor) -> float:
    """SOUND upper bound on the max operator norm of the displacement Jacobian d u of the piecewise-TRILINEAR
    interpolant (phi = id + u). Uses per-edge FORWARD differences u[i+1]-u[i] — the EXACT edge slopes of the
    interpolant. Central differences are NOT a bound: they average adjacent edges and cancel (an alternating
    field u_i=a(-1)^i has zero central difference but edge slope 2a). Within a cell the column d u/d a is a
    convex combination of the four parallel a-edges, so a max over incident edges bounds it; the per-cell
    Frobenius of the three column bounds >= the operator norm. value < 1 => u is a contraction => phi is
    GLOBALLY injective and bi-Lipschitz onto its image. Conservative (Frobenius >= spectral,
    edge-max >= in-cell value) but SOUND. Fields with
    value >= 1 need the weaker boundary route (Ball 1981 / Kroemer 2020); `boundary_max_disp` is its input."""
    with torch.no_grad():
        u = flow
        na = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]).pow(2).sum(1).sqrt()  # a-edge slope norms [1,D-1,H,W]
        nb = (u[:, :, :, 1:, :] - u[:, :, :, :-1, :]).pow(2).sum(1).sqrt()  # [1,D,H-1,W]
        nc = (u[:, :, :, :, 1:] - u[:, :, :, :, :-1]).pow(2).sum(1).sqrt()  # [1,D,H,W-1]
        ca = F.max_pool3d(na.unsqueeze(1), kernel_size=(1, 2, 2), stride=1)  # max over the cell's 4 a-edges
        cb = F.max_pool3d(nb.unsqueeze(1), kernel_size=(2, 1, 2), stride=1)
        cc = F.max_pool3d(nc.unsqueeze(1), kernel_size=(2, 2, 1), stride=1)
        return float(torch.sqrt(ca * ca + cb * cb + cc * cc).max().item())


def boundary_max_disp(flow: torch.Tensor) -> float:
    """Max displacement magnitude ||u|| on the six boundary faces.

    This is a diagnostic, not a proof: only exact zero establishes an identity trace, and an unspecified
    "small" value does not exclude collisions between distinct faces.
    """
    with torch.no_grad():
        mag = flow.pow(2).sum(dim=1).sqrt()[0]  # [D,H,W]
        faces = (mag[0], mag[-1], mag[:, 0], mag[:, -1], mag[:, :, 0], mag[:, :, -1])
        return float(max(f.max().item() for f in faces))


def boundary_vertex_mask(flow: torch.Tensor) -> torch.Tensor:
    """Boolean ``[B,1,D,H,W]`` mask of the six outer vertex faces."""
    if flow.dim() != 5 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [B,3,D,H,W], got {tuple(flow.shape)}.")
    mask = torch.zeros((flow.shape[0], 1, *flow.shape[-3:]), device=flow.device, dtype=torch.bool)
    mask[:, :, (0, -1), :, :] = True
    mask[:, :, :, (0, -1), :] = True
    mask[:, :, :, :, (0, -1)] = True
    return mask


def enforce_identity_boundary(flow: torch.Tensor) -> torch.Tensor:
    """Set all boundary displacement components to exact ``+0.0``."""
    return torch.where(boundary_vertex_mask(flow), torch.zeros_like(flow), flow)


def boundary_nonzero_count(flow: torch.Tensor) -> int:
    """Number of non-zero displacement components on the unique boundary-vertex set."""
    mask = boundary_vertex_mask(flow).expand_as(flow)
    return int(torch.count_nonzero(flow.masked_select(mask)).item())


def _face_tangential_lip(face: torch.Tensor) -> torch.Tensor:
    """SOUND bound on the tangential Lipschitz constant over one boundary face, `face` is [3,A,B]. Per-edge
    FORWARD differences are the exact in-plane edge slopes (central differences cancel and are NOT a bound);
    within a face-cell each column is a convex combination of its two parallel edges (max-pool bounds it), and
    the Frobenius of the two column bounds >= the spectral norm of the 3x2 in-plane Jacobian. Conservative but
    sound."""
    na = (face[:, 1:, :] - face[:, :-1, :]).pow(2).sum(0).sqrt()  # A-edge slope norms [A-1,B]
    nb = (face[:, :, 1:] - face[:, :, :-1]).pow(2).sum(0).sqrt()  # [A,B-1]
    ca = F.max_pool2d(na[None, None], kernel_size=(1, 2), stride=1)[0, 0]  # max over the 2 parallel A-edges
    cb = F.max_pool2d(nb[None, None], kernel_size=(2, 1), stride=1)[0, 0]
    return torch.sqrt(ca * ca + cb * cb).max()


def boundary_tangential_lip(flow: torch.Tensor) -> float:
    """Max tangential Lipschitz constant of the displacement over the six boundary faces (a SOUND bound from
    forward-difference edge slopes). value < 1 SOUNDLY certifies phi maps each face injectively (u contracts along
    each convex face, so ||phi(p)-phi(q)|| >= (1 - lip)||p-q|| > 0). It does not exclude collisions between
    different faces; ``boundary_max_disp`` without a quantitative separation argument cannot close that gap.
    This can fire more often than the global contraction test ``displacement_grad_norm_max`` because the
    boundary may be smoother than the interior."""
    with torch.no_grad():
        u = flow[0]  # [3,D,H,W]
        faces = (
            u[:, 0, :, :],
            u[:, -1, :, :],  # z faces, tangential (H, W)
            u[:, :, 0, :],
            u[:, :, -1, :],  # y faces, tangential (D, W)
            u[:, :, :, 0],
            u[:, :, :, -1],  # x faces, tangential (D, H)
        )
        return float(torch.stack([_face_tangential_lip(f) for f in faces]).max().item())


def _collar_ramp(n: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1-D smoothstep ramp: 0 at the two ends (the boundary faces), 1 at depth >= width inward."""
    idx = torch.arange(n, device=device, dtype=dtype)
    dist = torch.minimum(idx, (n - 1) - idx)
    t = (dist / float(max(width, 1))).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def identity_collar(flow: torch.Tensor, width: int = 4) -> torch.Tensor:
    """Force phi = id on the domain boundary by tapering the displacement to zero over a `width`-voxel collar.
    Returns flow * m, where m(x) = ra(d) rb(h) rc(w) is a product of per-axis smoothstep ramps that is 0 on the
    union of the six faces and 1 at depth >= width from every face. phi|boundary = id is a strong input to a
    global-invertibility argument. A classical piecewise-trilinear HOMEOMORPHISM claim still requires all
    theorem hypotheses to be checked explicitly. It is not a classical diffeomorphism:
    the trilinear gradient jumps across cell faces. The collar can change accuracy and can introduce locally
    uncertified cells; both effects must be measured, and all later repair operations must preserve the boundary."""
    if flow.dim() != 5 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [B,3,D,H,W], got {tuple(flow.shape)}.")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    _, _, d, h, w = flow.shape
    if min(d, h, w) < 2:
        raise ValueError(f"every spatial dimension must contain at least two vertices, got {(d, h, w)}")
    ra = _collar_ramp(d, width, flow.device, flow.dtype).view(1, 1, d, 1, 1)
    rb = _collar_ramp(h, width, flow.device, flow.dtype).view(1, 1, 1, h, 1)
    rc = _collar_ramp(w, width, flow.device, flow.dtype).view(1, 1, 1, 1, w)
    return enforce_identity_boundary(flow * (ra * rb * rc))


def certified_local_clip(
    flow_current: torch.Tensor,
    flow_proposal: torch.Tensor,
    eps: float = 0.0,
    sweeps: int = 1,
) -> torch.Tensor:
    """8-parity-color certified LOCAL clip (Gate B). Moves each grid vertex from `flow_current` (which MUST be
    certified: every cell's Bernstein coeff >= eps) toward `flow_proposal` by the largest per-vertex fraction
    alpha in [0,1] that keeps every incident cell's 27 Bernstein coeffs >= eps. Vertices are swept in 8 parity
    colors so no cell has two simultaneously-moved corners => each Bernstein coeff is AFFINE in the moved
    vertex's step (rank-one Jacobian update; matrix-determinant lemma), giving a closed-form alpha. Sound and
    feasibility-preserving in float64 working arithmetic (alpha=0 is always feasible). The LOCAL analogue of
    the failed global line-search: only constrained vertices shrink, safe ones keep alpha=1. The returned
    float32 materialisation must be checked again, normally using a work margin above the published margin;
    no accuracy dominance over ``trilinear_project`` is asserted."""
    if flow_current.shape != flow_proposal.shape:
        raise ValueError("current and proposal must share shape [1,3,D,H,W]")
    with torch.no_grad():
        cur = flow_current.detach().double()
        prop = flow_proposal.detach().double()
        _, _, d, h, w = cur.shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(d, device=cur.device),
            torch.arange(h, device=cur.device),
            torch.arange(w, device=cur.device),
            indexing="ij",
        )
        for _ in range(max(1, sweeps)):
            for color in range(8):
                ci, cj, ck = (color >> 2) & 1, (color >> 1) & 1, color & 1
                cmask = (zz % 2 == ci) & (yy % 2 == cj) & (xx % 2 == ck)  # this color's vertices [D,H,W]
                field1 = torch.where(cmask[None, None], prop, cur)  # this color's vertices fully at proposal
                b0 = _trilinear_bernstein_coeffs(cur)  # [3,3,3,D-1,H-1,W-1], all >= eps (cur is certified)
                s = _trilinear_bernstein_coeffs(field1) - b0  # per-cell affine slope of each coeff in alpha
                ratio = torch.where(s < 0, (b0 - eps) / (-s).clamp_min(1e-30), torch.full_like(s, float("inf")))
                alpha_cell = ratio.amin(dim=(0, 1, 2)).clamp(0.0, 1.0)  # max safe alpha per cell [D-1,H-1,W-1]
                # each color vertex is the color-c corner of its <=8 incident cells; alpha_v = min over them
                # (min-pool = -maxpool(-x); pad missing boundary cells with +inf so they never constrain)
                padded = F.pad(alpha_cell[None, None], (1, 1, 1, 1, 1, 1), value=float("inf"))
                alpha_v = (-F.max_pool3d(-padded, kernel_size=2, stride=1))[0, 0]  # [D,H,W]
                alpha_v = torch.where(cmask, alpha_v.clamp(0.0, 1.0), torch.zeros_like(alpha_v))
                cur = cur + alpha_v[None, None] * (prop - cur)
        return cur.float()


def _tri_pen_map(flow: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    """Per-cell trilinear-fold hinge map [D-1,H-1,W-1] for one (sub)volume. 'bernstein': hinge over the 27
    sound Bernstein coefficients of det J; 'sampled': hinge on det J at a 3^3 interior lattice (a proxy that
    can miss the true minimum between samples, hence not a certificate). Both retain ~27 full-res det
    sub-graphs for backward — the caller tiles + checkpoints this to bound peak memory."""
    if mode == "bernstein":
        return torch.relu(eps - _trilinear_bernstein_coeffs(flow)).sum(dim=(0, 1, 2))
    if mode == "sampled":
        p = _trilinear_corner_targets(flow)
        ts = torch.linspace(0.0, 1.0, 3, device=flow.device, dtype=flow.dtype).tolist()
        pen_map: torch.Tensor | None = None
        for a in ts:
            for b in ts:
                for c in ts:
                    h = torch.relu(eps - _trilinear_det_at(p, a, b, c))
                    pen_map = h if pen_map is None else pen_map + h
        return pen_map
    raise ValueError(f"unknown trilinear penalty mode {mode!r} (bernstein|sampled)")


def trilinear_fold_penalty(
    flow: torch.Tensor,
    mode: str = "bernstein",
    eps: float = 0.0,
    mask: torch.Tensor | None = None,
    tiles: int | None = None,
    reduce: str = "mean",
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Differentiable penalty that trains the DEPLOYED trilinear warp to be fold-free. 'bernstein': hinge
    sum_k relu(eps - coeff_k) over the 27 per-cell Bernstein coefficients (the sound criterion). 'sampled':
    hinge on det J at a 3^3 interior lattice per cell (a cheaper proxy). `reduce` sets the cell average:
    'mean' over ALL cells (the folds, being sparse, are smeared into a tiny gradient) or 'active' over only
    the violating cells (hinge > 0), which concentrates the gradient where the field actually folds — a
    CVaR-style reduction for the sparse-violation regime. `mask` (any [..,D,H,W]) restricts to the brain via
    each cell's lower corner.

    Memory: each mode retains ~27 full-res det sub-graphs for backward, which OOMs an 80 GB card at full
    resolution. When training (grad on), the cells are split into `tiles` slabs along D and each slab's map
    is gradient-checkpointed, so only one slab's graph is ever live — bounded peak, numerically identical to
    the untiled result. Inference (no_grad) runs a single tile with no checkpoint (unchanged legacy path)."""
    ckpt = torch.is_grad_enabled() and flow.requires_grad
    n_tiles = (8 if ckpt else 1) if tiles is None else max(1, tiles)
    n_cells_d = flow.shape[2] - 1
    edges = torch.linspace(0, n_cells_d, n_tiles + 1).round().long().tolist()

    m = None
    if mask is not None:
        m = mask
        while m.dim() > 3:
            m = m[0]
        m = m[:-1, :-1, :-1] > 0  # cell grid [D-1,H-1,W-1], lower-corner membership

    total = flow.new_zeros(())
    count = 0.0
    active = 0.0
    for t in range(n_tiles):
        c0, c1 = edges[t], edges[t + 1]
        if c1 <= c0:
            continue
        sub = flow[:, :, c0 : c1 + 1, :, :]  # +1 voxel to close the slab's top cells
        pm = (
            checkpoint(_tri_pen_map, sub, mode, float(eps), use_reentrant=False)
            if ckpt
            else _tri_pen_map(sub, mode, float(eps))
        )
        if m is None:
            total = total + pm.sum()
            count += pm.numel()
            active += float((pm > 0).sum().item())
        else:
            mt = m[c0:c1].to(pm.dtype)
            total = total + (pm * mt).sum()
            count += float(mt.sum().item())
            active += float(((pm > 0) & (mt > 0)).sum().item())
    denom = active if reduce == "active" else count
    loss = total / max(denom, 1.0)
    if return_stats:
        # active = violating cells; count = evaluated cells. active_frac drives the 'active'-reduce
        # amplification (1/active_frac vs mean), so it is the runaway diagnostic for the sparse regime.
        stats = {"active": active, "count": count, "active_frac": active / max(count, 1.0)}
        return loss, stats
    return loss


@dataclass(frozen=True)
class TrilinearProjectionReport:
    """Operational (float64) result of the heuristic Bernstein repair.

    This report is deliberately separate from the machine-sound verifier in ``utils.cert_exact``.  A failed
    sufficient predicate is called *uncertified*, not folded; ``sampled_negative_cell_percent`` is the separate
    witnessed-fold diagnostic.
    """

    certified: bool
    cert_bound: float
    n_uncertified_cells: int
    sampled_negative_cell_percent: float
    iterations: int
    status: str


def trilinear_project(
    flow: torch.Tensor,
    eps: float = 0.0,
    damp: float = 0.6,
    max_iters: int = 80,
    subdiv_depth: int = 0,
    fixed_mask: torch.Tensor | None = None,
    fixed_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TrilinearProjectionReport]:
    """Repair a displacement field onto the TRILINEAR fold-free (orientation-preserving) set. Each pass: flag every cell whose
    sound Bernstein bound of the actual grid_sample warp is < eps, expand the flagged cells to the eight
    voxels each touches, and blend those voxels' displacement toward the local (mean-smoothed) field under
    a feathered weight — the same boundary-safe relaxation as `digital_project`, but gated on the
    TRILINEAR certificate, not the digital determinants. Repeat until no cell fails (global
    tri_cert_bound >= eps) or `max_iters`. Returns the repaired flow and a structured report.
    A report with ``certified=True`` and bound >= eps establishes the Bernstein sufficient predicate in the
    repair's working arithmetic.  The returned structured report cannot confuse a zero sampled-fold count with passing the
    sufficient Bernstein predicate.  ``fixed_mask`` vertices are restored after every update; this is required
    by the identity-boundary collar.  The heuristic is not guaranteed to converge and reports failure closed."""
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [1,3,D,H,W], got {tuple(flow.shape)}.")

    def _smooth(t: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(F.pad(t, (1, 1, 1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)

    constraint = _vertex_constraint(flow, fixed_mask, fixed_values)
    out = _apply_vertex_constraint(flow.detach().clone().float(), constraint)
    applied = 0
    with torch.no_grad():
        for _ in range(max_iters):
            bounds = _trilinear_cell_cert_bound(out, subdiv_depth, eps)
            cell_safe = torch.isfinite(bounds) & (bounds >= eps)
            cell_bad = (~cell_safe).to(out.dtype)  # [D-1,H-1,W-1]; NaN/Inf fail closed
            if not bool(cell_bad.any()):
                break
            # A voxel is touched if any of the (up to 8) cells incident to it is flagged: a 2^3 max over
            # the cell grid padded by one, mapping [D-1,H-1,W-1] cells back to the [D,H,W] voxel grid.
            vox = F.max_pool3d(F.pad(cell_bad[None, None], (1, 1, 1, 1, 1, 1)), kernel_size=2, stride=1)
            feather = _smooth(_smooth(vox)).clamp(0.0, 1.0)  # blurred mask: no hard boundary
            candidate = out * (1.0 - damp * feather) + _smooth(out) * (damp * feather)
            out = _apply_vertex_constraint(candidate, constraint)
            applied += 1
        final_bounds = _trilinear_cell_cert_bound(out, subdiv_depth, eps)
        final_safe = torch.isfinite(final_bounds) & (final_bounds >= eps)
        n_uncertified = int((~final_safe).sum().item())
        cert_bound = float(final_bounds.min().item())
        sampled_negative = trilinear_fold_percent(out)
    report = TrilinearProjectionReport(
        certified=n_uncertified == 0,
        cert_bound=cert_bound,
        n_uncertified_cells=n_uncertified,
        sampled_negative_cell_percent=sampled_negative,
        iterations=applied,
        status="certified" if n_uncertified == 0 else "max_iters_uncertified",
    )
    return out, report


def perturb_flow(flow: torch.Tensor, mode: str = "none", scale: float = 0.02) -> torch.Tensor:
    """Emulate a deployment step that perturbs a displacement field AFTER it was certified, to test how
    much certificate margin survives it. 'fp16' = round-trip through float16 storage (the common case);
    'noise' = additive uniform +-scale voxels (an independent-per-voxel stress, harsher than fp16 since
    it perturbs neighbour differences). The determinant depends on gradients, so a knife-edge certificate
    (margin ~0) dies while a margin >= the induced Jacobian perturbation survives."""
    if mode == "none":
        return flow
    if mode == "fp16":
        return flow.half().float()
    if mode == "noise":
        return flow + (torch.rand_like(flow) * 2.0 - 1.0) * scale
    raise ValueError(f"unknown perturb mode {mode!r} (none|fp16|noise)")


def certified_max_step(candidate_fn, eps: float = 0.0, max_bisect: int = 12) -> tuple[float, torch.Tensor]:
    """Largest step t in [0,1] whose candidate field is trilinear-certified (tri_cert_bound >= eps), by
    bisection. ``candidate_fn(t)`` builds the field for step t, and ``candidate_fn(0)`` must be feasible
    (the pre-step flow). Returns (t, certified_flow). This is the heart of certified iterative refinement:
    an L3 (or TTO) update d is clipped to the largest topologically-safe fraction of itself, so every
    iterate stays fold-free (orientation-preserving) on the DEPLOYED warp — no post-hoc repair, no folds introduced. The
    caller chooses the space by how it builds candidate_fn (velocity: integrate t*d then compose;
    displacement: t*d added / t*integrated-d composed)."""
    with torch.no_grad():
        full = candidate_fn(1.0)
        if trilinear_cert_bound(full) >= eps:
            return 1.0, full
        lo, hi = 0.0, 1.0  # lo feasible (t=0 keeps the pre-step flow), hi infeasible
        best_t, best_flow = 0.0, candidate_fn(0.0)
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            cand = candidate_fn(mid)
            if trilinear_cert_bound(cand) >= eps:
                lo, best_t, best_flow = mid, mid, cand
            else:
                hi = mid
        return best_t, best_flow


def erode_mask(mask: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Binary erosion of a mask by `iters` voxels (3x3x3 min over 26-neighbours); outside the volume
    counts as background, so the border erodes inward. `iters<=0` is a no-op.
    """
    if iters <= 0:
        return mask
    orig_dim = mask.dim()
    m = (mask > 0).float()
    while m.dim() < 5:
        m = m.unsqueeze(0)
    for _ in range(iters):
        m = -F.max_pool3d(-m, kernel_size=3, stride=1, padding=1)
    while m.dim() > orig_dim:
        m = m.squeeze(0)
    return m


def jacobian_penalty_and_folds(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 1,
    eps: float = 0.0,
    strict: bool = True,
) -> tuple[torch.Tensor, float]:
    """Fold penalty on the central-difference detJ, plus a fold percentage, from one pass.
    `strict` counts the percentage by `digital_fold_percent` instead of that same detJ.
    """
    det = jacobian_det(flow)
    pen = _neg_jac_penalty_from_det(det, mask, crop, eps)
    with torch.no_grad():
        if strict:
            folds = float(digital_fold_percent(flow).item())
        else:
            folds = float((_crop_spatial(det, crop) <= 0.0).to(det.dtype).mean().item() * 100.0)
    return pen, folds


def jacobian_nonpositive_percent(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 0,
) -> float:
    """Non-positive Jacobian ratio (detJ <= 0) in percent."""
    det = _crop_spatial(jacobian_det(flow.float()), crop)
    neg = (det <= 0.0).float()

    if mask is None:
        return float(neg.mean().item() * 100.0)

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    m = (mask > 0).to(neg.dtype)
    m = _crop_spatial(m, crop)
    denom = float(torch.clamp(m.sum(), min=1.0).item())
    num = float((neg * m).sum().item())
    return num / denom * 100.0


def logdet_std_from_flow(flow: torch.Tensor, eps: float = 1e-9) -> float:
    """std(log(detJ + 3)) — matches the UTSRMorph reporting convention."""
    det = torch.clamp(jacobian_det(flow.float()) + 3.0, min=eps, max=1e9)
    return float(torch.std(torch.log(det)).item())


def digital_jacobian_metrics(flow: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Digital Jacobian metrics (%|J|<=0 and %NDV) from displacement and brain mask."""
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [1,3,D,H,W], got {tuple(flow.shape)}.")
    if mask is None:
        raise ValueError("digital_jacobian_metrics requires x_seg mask.")

    if mask.dim() == 5:
        mask_np = mask.detach().cpu().numpy()[0, 0]
    elif mask.dim() == 4:
        mask_np = mask.detach().cpu().numpy()[0]
    else:
        raise ValueError(f"Expected mask shape [1,1,D,H,W] or [1,D,H,W], got {tuple(mask.shape)}.")

    disp = flow.detach().float().cpu().numpy()[0]
    d, h, w = disp.shape[1:]
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    trans = disp + np.stack([zz, yy, xx], axis=0).astype(np.float32)

    def _det_from_axis_modes(mx: str, my: str, mz: str) -> np.ndarray:
        def fd(arr: np.ndarray, axis: int, mode: str) -> np.ndarray:
            n = arr.shape[axis]
            idx = np.arange(n)
            if mode == "+":
                return np.take(arr, np.clip(idx + 1, 0, n - 1), axis=axis) - arr
            if mode == "-":
                return arr - np.take(arr, np.clip(idx - 1, 0, n - 1), axis=axis)
            return 0.5 * (
                np.take(arr, np.clip(idx + 1, 0, n - 1), axis=axis)
                - np.take(arr, np.clip(idx - 1, 0, n - 1), axis=axis)
            )

        dx0, dx1, dx2 = fd(trans[0], 0, mx), fd(trans[1], 0, mx), fd(trans[2], 0, mx)
        dy0, dy1, dy2 = fd(trans[0], 1, my), fd(trans[1], 1, my), fd(trans[2], 1, my)
        dz0, dz1, dz2 = fd(trans[0], 2, mz), fd(trans[1], 2, mz), fd(trans[2], 2, mz)
        det = dx0 * (dy1 * dz2 - dy2 * dz1) - dx1 * (dy0 * dz2 - dy2 * dz0) + dx2 * (dy0 * dz1 - dy1 * dz0)
        return det[1:-1, 1:-1, 1:-1]

    def _corr3d_nearest(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kz, ky, kx = kernel.shape
        pz, py, px = kz // 2, ky // 2, kx // 2
        pad = np.pad(arr, ((pz, pz), (py, py), (px, px)), mode="edge")
        out = np.zeros_like(arr, dtype=np.float32)
        d0, h0, w0 = arr.shape
        for iz in range(kz):
            for iy in range(ky):
                for ix in range(kx):
                    w = float(kernel[iz, iy, ix])
                    if w == 0.0:
                        continue
                    out += w * pad[iz : iz + d0, iy : iy + h0, ix : ix + w0]
        return out

    def _det_from_kernels(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
        gradx = np.stack([_corr3d_nearest(trans[c], kx) for c in range(3)], axis=0)
        grady = np.stack([_corr3d_nearest(trans[c], ky) for c in range(3)], axis=0)
        gradz = np.stack([_corr3d_nearest(trans[c], kz) for c in range(3)], axis=0)
        det = (
            gradx[0] * (grady[1] * gradz[2] - grady[2] * gradz[1])
            - gradx[1] * (grady[0] * gradz[2] - grady[2] * gradz[0])
            + gradx[2] * (grady[0] * gradz[1] - grady[1] * gradz[0])
        )
        return det[1:-1, 1:-1, 1:-1]

    det_pm = []
    for mx, my, mz in (
        ("+", "+", "+"),
        ("+", "+", "-"),
        ("+", "-", "+"),
        ("+", "-", "-"),
        ("-", "+", "+"),
        ("-", "+", "-"),
        ("-", "-", "+"),
        ("-", "-", "-"),
    ):
        det_pm.append(_det_from_axis_modes(mx, my, mz))

    k1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    jstar1 = _det_from_kernels(k1.reshape(3, 3, 1), k1.reshape(3, 1, 3), k1.reshape(1, 3, 3))
    jstar2 = _det_from_kernels(k2.reshape(3, 3, 1), k2.reshape(1, 3, 3), k2.reshape(3, 1, 3))

    # All ten determinants, per Definition 8 of Liu et al., IJCV 2024 — J1*/J2* are required,
    # the 8 corner tetrahedra do not fill the cell.
    all_pos = np.ones_like(det_pm[0], dtype=np.bool_)
    for det in (*det_pm, jstar1, jstar2):
        all_pos &= det > 0.0
    j_leq0_percent = float((~all_pos).sum() / all_pos.size * 100.0)

    brain = (mask_np[1:-1, 1:-1, 1:-1] > 0).astype(np.float32)
    denom = float(brain.sum())
    if denom <= 0.0:
        return j_leq0_percent, 0.0

    ndv = 0.0
    for det in (*det_pm, jstar1, jstar2):
        ndv += float((-0.5 * np.minimum(det, 0.0) * brain / 6.0).sum())
    ndv_percent = ndv / denom * 100.0
    return j_leq0_percent, float(ndv_percent)
