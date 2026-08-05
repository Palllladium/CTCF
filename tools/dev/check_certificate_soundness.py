"""Gate A — adversarial soundness suite for the trilinear fold-free certificate, the injectivity
bounds, the 8-color certified clip (Gate B core), and the warp/compose geometry contract.

Every check here is a SOUNDNESS invariant, not an accuracy target: the certificate must never claim a
folding field is fold-free, the injectivity bounds must never under-report the interpolant's slope
(the central-difference bug that motivated commit 4eb35d7), and the clip must return a certified field.
The geometry tests LOCK the current align_corners convention (regression guard) — they assert the
present behaviour, they do not change it (align_corners is settled; do not unify — see field.py:33).

Runs standalone (like its sibling check_quality.py): `python tools/dev/check_certificate_soundness.py`.
Also pytest-discoverable if a runner is ever added (functions are named test_*).
"""
import sys
from fractions import Fraction as Fr
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: tools/dev/ -> ../../

from utils.field import (  # noqa: E402
    _trilinear_bernstein_coeffs,
    _warp,
    boundary_max_disp,
    boundary_tangential_lip,
    certified_local_clip,
    compose_flows,
    digital_min_det,
    displacement_grad_norm_max,
    trilinear_cert_bound,
    trilinear_min_det,
    trilinear_project,
)

TOL = 1e-6


def _smooth_field(shape, scale, seed, passes=3):
    """A smooth random displacement field [1,3,D,H,W] (blurred noise) at a given amplitude. Fewer blur
    `passes` leave more local curvature — enough to open the digital/trilinear gap (Group 2)."""
    torch.manual_seed(seed)
    d, h, w = shape
    u = torch.randn(1, 3, d + 6, h + 6, w + 6, dtype=torch.float64)
    for _ in range(passes):
        u = F.avg_pool3d(F.pad(u, (1,) * 6, mode="replicate"), 3, 1)
    return u[:, :, :d, :h, :w] * scale


# ---------------------------------------------------------------------------------------------------
# Group 1 — the certificate is a SOUND lower bound (never optimistic) and subdivision only tightens it.
# ---------------------------------------------------------------------------------------------------
def test_cert_bound_never_exceeds_dense_sampled_min():
    """The Bernstein bound must be <= the densely-sampled actual trilinear min det for EVERY field:
    if it ever exceeded the truth the certificate would be unsound (could certify a folding field)."""
    for seed in range(12):
        flow = _smooth_field((16, 15, 14), scale=1.5, seed=seed)
        dense_min = trilinear_min_det(flow, samples=9)  # sound detection min over a fine lattice
        for depth in (0, 1, 2):
            bound = trilinear_cert_bound(flow, subdiv_depth=depth, eps=0.0)
            assert bound <= dense_min + 1e-4, (
                f"UNSOUND: cert_bound({depth})={bound:.5f} > dense min det {dense_min:.5f} (seed {seed})"
            )


def _exact_bernstein_cell(disp):
    """The 27 Bernstein coeffs of det J on cell (0,0,0) computed in EXACT rational arithmetic. Corner
    targets index+disp are exact Fractions; det J at nodes {0,1/2,1}^3 is an exact rational function of
    them; the degree-2 value->Bernstein map is b0=v0, b1=(-v0+4v1-v2)/2, b2=v2 (rational). So the whole
    array is exact -- the reference the float64 pipeline is checked against."""
    p = {(i, j, k): [Fr(i if d == 0 else j if d == 1 else k) + Fr(float(disp[d, i, j, k])) for d in range(3)]
         for i in (0, 1) for j in (0, 1) for k in (0, 1)}

    def det_at(a, b, c):
        na, nb, nc = 1 - a, 1 - b, 1 - c
        ca = [(p[1, 0, 0][d] - p[0, 0, 0][d]) * nb * nc + (p[1, 1, 0][d] - p[0, 1, 0][d]) * b * nc
              + (p[1, 0, 1][d] - p[0, 0, 1][d]) * nb * c + (p[1, 1, 1][d] - p[0, 1, 1][d]) * b * c for d in range(3)]
        cb = [(p[0, 1, 0][d] - p[0, 0, 0][d]) * na * nc + (p[1, 1, 0][d] - p[1, 0, 0][d]) * a * nc
              + (p[0, 1, 1][d] - p[0, 0, 1][d]) * na * c + (p[1, 1, 1][d] - p[1, 0, 1][d]) * a * c for d in range(3)]
        cc = [(p[0, 0, 1][d] - p[0, 0, 0][d]) * na * nb + (p[1, 0, 1][d] - p[1, 0, 0][d]) * a * nb
              + (p[0, 1, 1][d] - p[0, 1, 0][d]) * na * b + (p[1, 1, 1][d] - p[1, 1, 0][d]) * a * b for d in range(3)]
        return (ca[0] * (cb[1] * cc[2] - cb[2] * cc[1]) - ca[1] * (cb[0] * cc[2] - cb[2] * cc[0])
                + ca[2] * (cb[0] * cc[1] - cb[1] * cc[0]))

    nd = (Fr(0), Fr(1, 2), Fr(1))
    v = {(ia, ib, ic): det_at(nd[ia], nd[ib], nd[ic]) for ia in range(3) for ib in range(3) for ic in range(3)}

    def v2b(x0, x1, x2):
        return [x0, (-x0 + 4 * x1 - x2) / 2, x2]

    for axis in range(3):  # apply the 1D value->Bernstein map along each axis in turn
        nv = {}
        for i in range(3):
            for j in range(3):
                idx = lambda t: (t, i, j) if axis == 0 else (i, t, j) if axis == 1 else (i, j, t)  # noqa: E731
                col = v2b(v[idx(0)], v[idx(1)], v[idx(2)])
                for t in range(3):
                    nv[idx(t)] = col[t]
        v = nv
    return v


def test_bernstein_coeffs_match_exact_rational():
    """Machine-soundness at the operating margin: the float64 Bernstein coeffs must equal the EXACT rational
    values to far better than eps=1e-3. Measured error ~3.5e-15 (~1e11x below eps) => interval arithmetic is
    unnecessary at our margin; it would only matter for eps near 1e-14. Closes the 'fp64 != machine-sound' hole."""
    flow = torch.randn(1, 3, 3, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    exact = _exact_bernstein_cell(flow[0])
    f64 = _trilinear_bernstein_coeffs(flow)[:, :, :, 0, 0, 0]
    err = max(abs(float(exact[a, b, c]) - float(f64[a, b, c])) for a in range(3) for b in range(3) for c in range(3))
    assert err < 1e-9, f"float64 Bernstein error {err:.2e} not << eps=1e-3 (machine-soundness in question)"


def test_subdivision_is_monotone_and_sound():
    """Deeper de Casteljau subdivision raises the bound (monotone) but never above the true min det."""
    for seed in range(8):
        flow = _smooth_field((14, 13, 12), scale=2.0, seed=seed)
        dense_min = trilinear_min_det(flow, samples=11)
        b0 = trilinear_cert_bound(flow, subdiv_depth=0)
        b2 = trilinear_cert_bound(flow, subdiv_depth=2)
        assert b0 <= b2 + TOL, f"subdivision must not lower the bound: {b0:.5f} !<= {b2:.5f}"
        assert b2 <= dense_min + 1e-4, f"subdivided bound {b2:.5f} exceeds true min {dense_min:.5f}"


# ---------------------------------------------------------------------------------------------------
# Group 2 — the interpolation-consistency gap: the certificate catches folds digital-10 declares absent.
# ---------------------------------------------------------------------------------------------------
def test_digital_positive_but_trilinear_folds_exists():
    """The certificate's reason to exist: a field the corner-only digital-10 criterion calls fold-free
    (digital_min_det >= 0) whose DEPLOYED trilinear warp actually folds (cert_bound < 0). Found by a
    seeded search — deterministic, and it documents the gap is real, not hypothetical."""
    found = None
    for scale in (1.0, 1.5, 2.0):
        for seed in range(40):
            # 1 blur pass: smooth enough that the corners (digital) stay positive, curved enough that
            # the degree-2 interior det dips negative -- the field family that exhibits the gap.
            flow = _smooth_field((8, 8, 8), scale=scale, seed=seed, passes=1)
            if digital_min_det(flow) >= 0.0 and trilinear_cert_bound(flow) < 0.0:
                found = (seed, digital_min_det(flow), trilinear_cert_bound(flow), trilinear_min_det(flow, 9))
                break
        if found is not None:
            break
    assert found is not None, "expected at least one digital-passes / trilinear-folds field in the sweep"
    _, dig, cert, tri = found
    assert dig >= 0.0 and cert < 0.0 and tri < 0.0  # digital blind, certificate + sampled both catch it


# ---------------------------------------------------------------------------------------------------
# Group 3 — injectivity bounds use FORWARD differences (the central-difference bug, commit 4eb35d7).
# ---------------------------------------------------------------------------------------------------
def test_checkerboard_defeats_central_difference():
    """u_i = a(-1)^i along one axis has zero CENTRAL difference everywhere but true edge slope 2a. A
    central-difference bound returns ~0 (unsound); the forward-difference bound must report >= 2a."""
    a = 0.3
    d, h, w = 12, 10, 11
    idx = torch.arange(d).view(1, 1, d, 1, 1)
    u = torch.zeros(1, 3, d, h, w, dtype=torch.float64)
    u[:, 0] = a * ((-1.0) ** idx)  # alternate the z-displacement along z
    got = displacement_grad_norm_max(u)
    assert got >= 2 * a - TOL, f"forward-diff must see the 2a={2*a} edge slope, got {got:.5f} (central-diff bug)"


def test_ramp_and_identity_grad_norm():
    """A linear ramp u_z = s*z has edge slope exactly s; the identity field has slope 0."""
    d, h, w = 10, 9, 8
    s = 0.4
    z = torch.arange(d, dtype=torch.float64).view(1, 1, d, 1, 1)
    ramp = torch.zeros(1, 3, d, h, w, dtype=torch.float64)
    ramp[:, 0] = s * z
    assert abs(displacement_grad_norm_max(ramp) - s) < 1e-4, "ramp edge slope must equal s"
    assert displacement_grad_norm_max(torch.zeros(1, 3, d, h, w, dtype=torch.float64)) == 0.0


def test_boundary_bounds_read_the_faces():
    """boundary_max_disp reports the largest ||u|| on the six faces; an identity-boundary field reads ~0."""
    d, h, w = 10, 9, 8
    u = _smooth_field((d, h, w), scale=0.5, seed=1)
    interior_only = u.clone()
    interior_only[:, :, 0] = interior_only[:, :, -1] = 0.0
    interior_only[:, :, :, 0] = interior_only[:, :, :, -1] = 0.0
    interior_only[:, :, :, :, 0] = interior_only[:, :, :, :, -1] = 0.0
    assert boundary_max_disp(interior_only) < 1e-9, "identity-boundary field must have ~0 boundary displacement"
    assert boundary_max_disp(u) > 0.0
    assert boundary_tangential_lip(u) >= 0.0  # a sound non-negative Lipschitz bound


# ---------------------------------------------------------------------------------------------------
# Group 4 — certified_local_clip (Gate B core): certified by construction, better repair than heuristic.
# ---------------------------------------------------------------------------------------------------
def _folding_target(seed=0):
    d, h, w = 20, 18, 19
    tgt = _smooth_field((d, h, w), scale=3.0, seed=seed)
    tgt[0, :, d // 2, h // 2, w // 2] += 2.5  # a sharp interior bump -> a local fold
    return tgt


def test_clip_output_is_certified():
    """Driving identity -> a folding target by the 8-color clip yields a field whose certificate holds."""
    eps = 1e-3
    tgt = _folding_target()
    assert trilinear_min_det(tgt) < 0.0, "target must actually fold for the test to be meaningful"
    clip = certified_local_clip(torch.zeros_like(tgt), tgt, eps=eps, sweeps=4)
    assert trilinear_cert_bound(clip.double(), eps=eps) >= eps - TOL, "clip output must be certified fold-free"


def test_clip_preserves_at_least_as_much_as_heuristic_repair():
    """The clip keeps the proposal wherever safe, so it should preserve >= the feathered heuristic repair."""
    eps = 1e-3
    tgt = _folding_target(seed=2)
    clip = certified_local_clip(torch.zeros_like(tgt), tgt, eps=eps, sweeps=4).double()
    repair, _, _ = trilinear_project(tgt, eps=eps, max_iters=80)
    ref = tgt.pow(2).sum().sqrt()
    kept_clip = float(clip.pow(2).sum().sqrt() / ref)
    kept_rep = float(repair.double().pow(2).sum().sqrt() / ref)
    assert kept_clip >= kept_rep - TOL, f"clip kept {kept_clip:.4f} < repair {kept_rep:.4f}"


def test_clip_leaves_already_certified_field_untouched():
    """A gentle proposal that is already certified must be returned essentially unchanged (alpha=1)."""
    eps = 1e-3
    gentle = _smooth_field((16, 15, 14), scale=0.3, seed=5)
    assert trilinear_cert_bound(gentle, eps=eps) >= eps, "precondition: gentle field is already certified"
    clip = certified_local_clip(torch.zeros_like(gentle), gentle, eps=eps, sweeps=2).double()
    assert (clip - gentle).abs().max() < 1e-6, "an already-certified proposal must pass through unchanged"


# ---------------------------------------------------------------------------------------------------
# Group 5 — geometry contract: LOCK the current warp/compose behaviour (align_corners=True, do not drift).
# ---------------------------------------------------------------------------------------------------
def test_identity_flow_is_a_no_op():
    img = torch.randn(1, 1, 8, 9, 10)
    out = _warp(img, torch.zeros(1, 3, 8, 9, 10))
    assert (out - img).abs().max() < 1e-5, "warping by the zero flow must return the image unchanged"


def test_integer_translation_shifts_content():
    """A constant +1-voxel flow along z pulls content from z+1 (grid_sample samples index+flow)."""
    d, h, w = 8, 6, 6
    img = torch.arange(d, dtype=torch.float32).view(1, 1, d, 1, 1).expand(1, 1, d, h, w).contiguous()
    flow = torch.zeros(1, 3, d, h, w)
    flow[:, 0] = 1.0
    out = _warp(img, flow)
    assert (out[0, 0, : d - 1] - img[0, 0, 1:]).abs().max() < 1e-4, "interior must shift by exactly one voxel"


def test_compose_with_zero_is_identity_on_flows():
    """Composition identities: compose(f, 0) == f and compose(0, g) == g (voxel-unit A->C = A->B + warp)."""
    f = _smooth_field((8, 7, 6), scale=0.5, seed=3).float()
    g = _smooth_field((8, 7, 6), scale=0.5, seed=4).float()
    zero = torch.zeros_like(f)
    assert (compose_flows(f, zero) - f).abs().max() < 1e-5
    assert (compose_flows(zero, g) - g).abs().max() < 1e-5


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
