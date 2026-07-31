from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


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


def digital_project(
    flow: torch.Tensor,
    eps: float = 0.0,
    damp: float = 0.6,
    max_iters: int = 80,
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

    out = flow.detach().clone()
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
            out = out * (1.0 - damp * feather) + _smooth(out) * (damp * feather)
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


def _trilinear_cell_cert_bound(flow: torch.Tensor) -> torch.Tensor:
    """Per-cell sound Bernstein lower bound on det J of the trilinear deformation, shape [D-1,H-1,W-1].
    det J is degree <=2 in each of (a,b,c); its 27 Bernstein coefficients bound it on the whole cell
    (convex-hull property), so the per-cell minimum coefficient is a rigorous lower bound over the
    continuum — a cell with value >= eps is CERTIFIED fold-free, no sampling gap."""
    flow = flow.detach().float()
    p = _trilinear_corner_targets(flow)
    mat = _values_to_bernstein_matrix(flow.device, flow.dtype)
    nodes = (0.0, 0.5, 1.0)
    vals = torch.stack([_trilinear_det_at(p, a, b, c) for a in nodes for b in nodes for c in nodes]).reshape(
        3, 3, 3, *p[0, 0, 0].shape[1:]
    )
    bern = torch.einsum("pa,abcijk->pbcijk", mat, vals)
    bern = torch.einsum("pb,abcijk->apcijk", mat, bern)
    bern = torch.einsum("pc,abcijk->abpijk", mat, bern)
    return bern.amin(dim=(0, 1, 2))


def trilinear_cert_bound(flow: torch.Tensor) -> float:
    """Global sound Bernstein lower bound over every cell (the min of `_trilinear_cell_cert_bound`).
    > 0 CERTIFIES the materialized trilinear warp is everywhere orientation-preserving — the
    interpolation-consistent certificate the corner-only digital criterion cannot give."""
    with torch.no_grad():
        return float(_trilinear_cell_cert_bound(flow).min().item())


def trilinear_project(
    flow: torch.Tensor,
    eps: float = 0.0,
    damp: float = 0.6,
    max_iters: int = 80,
) -> tuple[torch.Tensor, float, int]:
    """Repair a displacement field onto the TRILINEAR-diffeomorphic set. Each pass: flag every cell whose
    sound Bernstein bound of the actual grid_sample warp is < eps, expand the flagged cells to the eight
    voxels each touches, and blend those voxels' displacement toward the local (mean-smoothed) field under
    a feathered weight — the same boundary-safe relaxation as `digital_project`, but gated on the
    TRILINEAR certificate, not the digital determinants. Repeat until no cell fails (global
    tri_cert_bound >= eps) or `max_iters`. Returns (repaired flow, residual trilinear fold %, passes).
    A zero residual with the returned bound >= eps certifies the DEPLOYED warp is orientation-preserving
    with margin eps; a non-zero residual is returned honestly, never as a false certificate."""
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [1,3,D,H,W], got {tuple(flow.shape)}.")

    def _smooth(t: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(F.pad(t, (1, 1, 1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)

    out = flow.detach().clone().float()
    applied = 0
    with torch.no_grad():
        for _ in range(max_iters):
            cell_bad = (_trilinear_cell_cert_bound(out) < eps).to(out.dtype)  # [D-1,H-1,W-1]
            if not bool(cell_bad.any()):
                break
            # A voxel is touched if any of the (up to 8) cells incident to it is flagged: a 2^3 max over
            # the cell grid padded by one, mapping [D-1,H-1,W-1] cells back to the [D,H,W] voxel grid.
            vox = F.max_pool3d(F.pad(cell_bad[None, None], (1, 1, 1, 1, 1, 1)), kernel_size=2, stride=1)
            feather = _smooth(_smooth(vox)).clamp(0.0, 1.0)  # blurred mask: no hard boundary
            out = out * (1.0 - damp * feather) + _smooth(out) * (damp * feather)
            applied += 1
        residual = trilinear_fold_percent(out)
    return out, residual, applied


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
