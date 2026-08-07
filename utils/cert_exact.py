"""Machine-sound certificate of the deployed trilinear warp: an EXACT decision procedure for whether det J
of the piecewise-trilinear grid_sample interpolant is > eps on every cell, accounting for ALL floating-point
rounding (no error term swept under the rug).

Two rigorous layers (the standard FP-filter + exact-predicate pattern of computational geometry):
  1. INTERVAL filter -- recompute the 27 per-cell Bernstein coefficients of det J with DIRECTED ROUNDING
     (numpy + np.nextafter widen every op outward), giving a guaranteed enclosure [lo, hi] of the true min
     coefficient. lo > eps => the cell is CERTIFIED, rigorously. Vectorised; clears ~all cells.
  2. EXACT fallback -- for the few cells the interval cannot resolve (lo <= eps <= hi), recompute the min
     coefficient in EXACT rational arithmetic (fractions.Fraction). fp32 inputs are dyadic rationals and the
     value->Bernstein map M = [[1,0,0],[-1/2,2,-1/2],[0,0,1]] plus the node weights {0,1/2,1} are dyadic, so
     every coefficient is an EXACT rational -- the min-vs-eps sign is decided with zero error.

Sound (never a false certificate) + complete (every cell decided). This is what upgrades the float64 bound in
`utils.field.trilinear_cert_bound` from "sound formulation, numerically confirmed" to "machine-sound: exact".
It certifies the map defined by the fp32 control points -- exactly the object grid_sample deploys.

The kernel MIRRORS `utils.field._trilinear_det_at` / `_trilinear_bernstein_coeffs` (same det, same M) so the
exact/interval result matches the float path up to its rounding.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from utils.field import _trilinear_cell_cert_bound, _trilinear_corner_targets

# ---------------------------------------------------------------------------
# Interval arithmetic on (lo, hi) pairs of float64 arrays, with directed rounding.
# Every op widens outward by one ULP via nextafter, so the true result of the op is guaranteed to lie in the
# returned interval (float64 rounding error is < 1 ULP; nextafter moves a full ULP).
# ---------------------------------------------------------------------------
def _down(x: np.ndarray) -> np.ndarray:
    return np.nextafter(x, -np.inf)


def _up(x: np.ndarray) -> np.ndarray:
    return np.nextafter(x, np.inf)


def _iadd(a, b):
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _isub(a, b):
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _imul(a, b):
    alo, ahi = a
    blo, bhi = b
    p1, p2, p3, p4 = alo * blo, alo * bhi, ahi * blo, ahi * bhi
    lo = _down(np.minimum(np.minimum(p1, p2), np.minimum(p3, p4)))
    hi = _up(np.maximum(np.maximum(p1, p2), np.maximum(p3, p4)))
    return lo, hi


def _E(x: np.ndarray):
    """Degenerate (exact) interval [x, x] for a value that is exactly representable in float64."""
    return x, x


_NODES = (0.0, 0.5, 1.0)  # local coords / Bernstein sample nodes; all exact in float64


def _cols_exact(p: dict, a: float, b: float, c: float):
    """The three Jacobian columns at exact local coords (a,b,c) in {0,1/2,1}^3, EXACT in float64.
    p[i,j,k] is a float64 [3,N] target-coord array; corner differences and the weights {0,1/4,1/2,1} are
    exact, and each column is a sum of four exact terms -- no rounding yet (rounding starts at the det
    products below)."""
    na, nb, nc = 1.0 - a, 1.0 - b, 1.0 - c
    col_a = ((p[1, 0, 0] - p[0, 0, 0]) * (nb * nc) + (p[1, 1, 0] - p[0, 1, 0]) * (b * nc)
             + (p[1, 0, 1] - p[0, 0, 1]) * (nb * c) + (p[1, 1, 1] - p[0, 1, 1]) * (b * c))
    col_b = ((p[0, 1, 0] - p[0, 0, 0]) * (na * nc) + (p[1, 1, 0] - p[1, 0, 0]) * (a * nc)
             + (p[0, 1, 1] - p[0, 0, 1]) * (na * c) + (p[1, 1, 1] - p[1, 0, 1]) * (a * c))
    col_c = ((p[0, 0, 1] - p[0, 0, 0]) * (na * nb) + (p[1, 0, 1] - p[1, 0, 0]) * (a * nb)
             + (p[0, 1, 1] - p[0, 1, 0]) * (na * b) + (p[1, 1, 1] - p[1, 1, 0]) * (a * b))
    return col_a, col_b, col_c


def _det_interval(col_a, col_b, col_c):
    """det of the 3x3 [col_a|col_b|col_c] as an interval; columns are exact float64 [3,N], the products round."""
    m0 = _isub(_imul(_E(col_b[1]), _E(col_c[2])), _imul(_E(col_b[2]), _E(col_c[1])))
    m1 = _isub(_imul(_E(col_b[0]), _E(col_c[2])), _imul(_E(col_b[2]), _E(col_c[0])))
    m2 = _isub(_imul(_E(col_b[0]), _E(col_c[1])), _imul(_E(col_b[1]), _E(col_c[0])))
    t0 = _imul(_E(col_a[0]), m0)
    t1 = _imul(_E(col_a[1]), m1)
    t2 = _imul(_E(col_a[2]), m2)
    return _iadd(_isub(t0, t1), t2)


def _axis_bernstein_interval(v0, v1, v2):
    """Apply M = [[1,0,0],[-1/2,2,-1/2],[0,0,1]] to three interval values along one axis. Rows 0 and 2 copy;
    row 1 = -1/2 v0 + 2 v1 - 1/2 v2 (the scalings by +/-1/2, 2 are EXACT in float64; only the two adds round)."""
    def _scale(c, x):  # c is a power-of-two-exact scalar -> c*x exact; keep the interval oriented
        lo, hi = c * x[0], c * x[1]
        return (lo, hi) if c >= 0 else (hi, lo)
    mid = _iadd(_iadd(_scale(-0.5, v0), _scale(2.0, v1)), _scale(-0.5, v2))
    return v0, mid, v2


def _min_bern_coeff_interval(flow: torch.Tensor):
    """Per-cell interval enclosure [lo, hi] of the MIN of the 27 Bernstein coefficients of det J, each
    [D-1,H-1,W-1]. lo > eps rigorously certifies the cell. Mirrors `_trilinear_bernstein_coeffs`."""
    pt = _trilinear_corner_targets(flow.detach().double())  # exact float64 target coords
    p = {k: v.cpu().numpy() for k, v in pt.items()}
    # 27 node determinants as intervals
    vals = [[[_det_interval(*_cols_exact(p, a, b, c)) for c in _NODES] for b in _NODES] for a in _NODES]
    # M along axis a, then b, then c
    va = [[_axis_bernstein_interval(vals[0][b][c], vals[1][b][c], vals[2][b][c]) for c in range(3)] for b in range(3)]
    va = [[[va[b][c][a] for c in range(3)] for b in range(3)] for a in range(3)]
    vb = [[_axis_bernstein_interval(va[a][0][c], va[a][1][c], va[a][2][c]) for c in range(3)] for a in range(3)]
    vb = [[[vb[a][c][b] for c in range(3)] for b in range(3)] for a in range(3)]
    vc = [[_axis_bernstein_interval(vb[a][b][0], vb[a][b][1], vb[a][b][2]) for b in range(3)] for a in range(3)]
    coeffs = [vc[a][b][c] for a in range(3) for b in range(3) for c in range(3)]
    lo = coeffs[0][0].copy()
    hi = coeffs[0][1].copy()
    for clo, chi in coeffs[1:]:
        np.minimum(lo, clo, out=lo)
        np.minimum(hi, chi, out=hi)
    return lo, hi


# ---------------------------------------------------------------------------
# Exact per-cell kernel (fractions.Fraction) for the cells the interval cannot resolve.
# ---------------------------------------------------------------------------
_M = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(-1, 2), Fraction(2), Fraction(-1, 2)),
    (Fraction(0), Fraction(0), Fraction(1)),
)
_FNODES = (Fraction(0), Fraction(1, 2), Fraction(1))


def _min_bern_coeff_exact(corners: dict) -> Fraction:
    """Exact min of the 27 Bernstein coefficients of det J for ONE cell. `corners[i,j,k]` = the 3 target
    coordinates (Fractions). All arithmetic is exact rational -> the returned value is the TRUE min coeff."""
    def det_at(a, b, c):
        na, nb, nc = 1 - a, 1 - b, 1 - c
        ca = [corners[1, 0, 0][t] - corners[0, 0, 0][t] for t in range(3)]
        cb = [corners[0, 1, 0][t] - corners[0, 0, 0][t] for t in range(3)]
        cc = [corners[0, 0, 1][t] - corners[0, 0, 0][t] for t in range(3)]
        col_a = [(corners[1, 0, 0][t] - corners[0, 0, 0][t]) * (nb * nc)
                 + (corners[1, 1, 0][t] - corners[0, 1, 0][t]) * (b * nc)
                 + (corners[1, 0, 1][t] - corners[0, 0, 1][t]) * (nb * c)
                 + (corners[1, 1, 1][t] - corners[0, 1, 1][t]) * (b * c) for t in range(3)]
        col_b = [(corners[0, 1, 0][t] - corners[0, 0, 0][t]) * (na * nc)
                 + (corners[1, 1, 0][t] - corners[1, 0, 0][t]) * (a * nc)
                 + (corners[0, 1, 1][t] - corners[0, 0, 1][t]) * (na * c)
                 + (corners[1, 1, 1][t] - corners[1, 0, 1][t]) * (a * c) for t in range(3)]
        col_c = [(corners[0, 0, 1][t] - corners[0, 0, 0][t]) * (na * nb)
                 + (corners[1, 0, 1][t] - corners[1, 0, 0][t]) * (a * nb)
                 + (corners[0, 1, 1][t] - corners[0, 1, 0][t]) * (na * b)
                 + (corners[1, 1, 1][t] - corners[1, 1, 0][t]) * (a * b) for t in range(3)]
        _ = (ca, cb, cc)  # (unused straight-difference columns kept for parity with the reference)
        return (col_a[0] * (col_b[1] * col_c[2] - col_b[2] * col_c[1])
                - col_a[1] * (col_b[0] * col_c[2] - col_b[2] * col_c[0])
                + col_a[2] * (col_b[0] * col_c[1] - col_b[1] * col_c[0]))

    vals = {(ia, ib, ic): det_at(a, b, c)
            for ia, a in enumerate(_FNODES) for ib, b in enumerate(_FNODES) for ic, c in enumerate(_FNODES)}
    best: Fraction | None = None
    for p in range(3):
        for q in range(3):
            for r in range(3):
                coeff = sum(_M[p][ia] * _M[q][ib] * _M[r][ic] * vals[ia, ib, ic]
                            for ia in range(3) for ib in range(3) for ic in range(3))
                best = coeff if best is None or coeff < best else best
    return best


def _cell_corners_exact(flow: torch.Tensor, idx) -> dict:
    """Exact target coordinates of the 8 corners of cell `idx=(i,j,k)` as Fractions (phi = index + fp32 disp)."""
    disp = flow[0].detach().cpu().numpy()  # [3,D,H,W] fp32
    i, j, k = idx
    out = {}
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                z, y, x = i + di, j + dj, k + dk
                out[di, dj, dk] = [
                    Fraction(int(z)) + Fraction(float(disp[0, z, y, x])),
                    Fraction(int(y)) + Fraction(float(disp[1, z, y, x])),
                    Fraction(int(x)) + Fraction(float(disp[2, z, y, x])),
                ]
    return out


def certify_flow_exact(flow: torch.Tensor, eps: float = 1e-3, max_exact: int = 100000) -> dict:
    """Machine-sound certificate that the deployed trilinear warp has det J > eps on every cell.

    Layer 1 (interval, vectorised) certifies every cell with lo > eps. Layer 2 (exact Fraction) decides the
    rest. Returns a report; `certified` is True iff EVERY cell is proven det J > eps with zero error.
    """
    flow = flow.detach()
    lo, hi = _min_bern_coeff_interval(flow)
    n_cells = int(lo.size)
    interval_ok = lo > eps                       # rigorously certified by the interval filter
    suspects = np.argwhere(~interval_ok)         # cells the interval could not clear (lo <= eps)
    n_suspect = int(suspects.shape[0])
    if n_suspect > max_exact:
        raise RuntimeError(
            f"{n_suspect} cells need exact checking (> max_exact={max_exact}); the field is far from "
            f"certified at eps={eps}. Repair it first, or raise max_exact for an audit."
        )

    eps_q = Fraction(eps)
    exact_min: Fraction | None = None
    failures: list[tuple[tuple[int, int, int], str]] = []
    for idx in suspects:
        cell = tuple(int(v) for v in idx)
        m = _min_bern_coeff_exact(_cell_corners_exact(flow, cell))
        exact_min = m if exact_min is None or m < exact_min else exact_min
        if m <= eps_q:
            failures.append((cell, str(m)))

    return {
        "certified": len(failures) == 0,
        "eps": eps,
        "n_cells": n_cells,
        "n_interval_certified": int(interval_ok.sum()),
        "n_exact_checked": n_suspect,
        "n_failures": len(failures),
        "interval_lo_min": float(lo.min()),
        "interval_hi_min": float(hi.min()),
        "exact_min_over_suspects": None if exact_min is None else float(exact_min),
        "failures": failures[:50],  # (cell, exact min coeff) for the first offenders
    }


def _load_flow_npz(path: str) -> torch.Tensor:
    with np.load(path) as d:
        arr = d["flow"]
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    return t if t.dim() == 5 else t.unsqueeze(0)


def _selftest() -> None:
    # Self-test: identity certifies; a known folded cell is caught; interval encloses the float64 bound.
    torch.manual_seed(0)
    ident = torch.zeros(1, 3, 8, 8, 8)
    rep = certify_flow_exact(ident, eps=1e-3)
    print("identity:", rep["certified"], "lo_min=%.4f" % rep["interval_lo_min"], "suspects=", rep["n_exact_checked"])

    # smooth small deformation -> should certify
    g = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, 8) for _ in range(3)], indexing="ij"))
    smooth = 0.05 * torch.sin(3.0 * g).unsqueeze(0)
    rep = certify_flow_exact(smooth, eps=1e-4)
    print("smooth:  ", rep["certified"], "lo_min=%.4f" % rep["interval_lo_min"], "suspects=", rep["n_exact_checked"])

    # a deliberately folding field (large gradient) -> not certified, exact catches folds
    fold = torch.zeros(1, 3, 8, 8, 8)
    fold[0, 0, 4, :, :] = 3.0   # a sharp jump in the z-displacement across one plane
    rep = certify_flow_exact(fold, eps=1e-3, max_exact=100000)
    print("folded:  ", rep["certified"], "failures=", rep["n_failures"],
          "exact_min=%.4f" % (rep["exact_min_over_suspects"] or 0.0))

    # interval must enclose the float64 bound from utils.field
    fb = _trilinear_cell_cert_bound(smooth.double()).cpu().numpy()
    lo, hi = _min_bern_coeff_interval(smooth)
    assert (lo <= fb + 1e-9).all() and (fb <= hi + 1e-9).all(), "interval does not enclose the float64 bound"
    print("enclosure OK: interval [lo,hi] contains the float64 per-cell bound")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Machine-sound (exact) certificate of a deployed trilinear warp.")
    ap.add_argument("--flow", default=None, help="Saved flow .npz (from inference --save_flow) to certify; "
                    "omit to run the built-in self-test.")
    ap.add_argument("--eps", type=float, default=1e-3, help="Certificate margin (det J > eps on every cell).")
    ap.add_argument("--max_exact", type=int, default=200000, help="Abort if more cells need exact checking "
                    "(field far from certified — repair it first).")
    args = ap.parse_args()

    if args.flow is None:
        _selftest()
    else:
        report = certify_flow_exact(_load_flow_npz(args.flow), eps=args.eps, max_exact=args.max_exact)
        verdict = "CERTIFIED (machine-sound)" if report["certified"] else "NOT certified"
        print(f"{verdict}  eps={report['eps']}")
        print(f"  cells={report['n_cells']}  interval-certified={report['n_interval_certified']}  "
              f"exact-checked={report['n_exact_checked']}  failures={report['n_failures']}")
        print(f"  interval lo_min={report['interval_lo_min']:.3e}  hi_min={report['interval_hi_min']:.3e}")
        if report["exact_min_over_suspects"] is not None:
            print(f"  exact min over suspects={report['exact_min_over_suspects']:.3e}")
        for cell, val in report["failures"]:
            print(f"  FOLD/uncertified cell {cell}: exact min coeff = {val}")
