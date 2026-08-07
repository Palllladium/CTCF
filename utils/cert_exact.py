"""Machine-sound verification of the adopted Bernstein certificate.

The object being verified is the mathematical piecewise-trilinear map

    phi(i, j, k) = (i, j, k) + u[i, j, k],

where ``u`` is the exact array of IEEE-754 binary32 values stored in a flow file.  For every cell we express
``det D phi`` in the tensor-product Bernstein basis of degree (2, 2, 2).  If all 27 coefficients are at least
an exact rational ``eps > 0``, the convex-hull property proves ``det D phi >= eps`` throughout that cell.

This module decides that *sufficient Bernstein predicate*.  A negative result means "not certified by this
predicate"; it does not by itself prove that the map folds.  It also does not prove global injectivity or
formally verify the arithmetic inside a particular ``grid_sample`` CUDA kernel.

The verifier has two layers:

1. A tiled float64 interval filter.  Every arithmetic operation, including edge differences, identity-column
   additions, determinants and the value-to-Bernstein transform, is widened outwards with ``nextafter``.
2. Exact ``Fraction`` arithmetic for cells whose interval straddles the threshold.

Only finite binary32 inputs are accepted.  Reports bind the verdict to the exact array bytes with SHA-256.
The interval proof assumes ordinary IEEE-754 binary64 round-to-nearest arithmetic, gradual underflow and no
``fast-math`` reassociation; the executable checks the observable gradual-underflow prerequisites fail closed.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

Interval: TypeAlias = tuple[np.ndarray, np.ndarray]

_NODES = (0.0, 0.5, 1.0)
_FNODES = (Fraction(0), Fraction(1, 2), Fraction(1))
_M = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(-1, 2), Fraction(2), Fraction(-1, 2)),
    (Fraction(0), Fraction(0), Fraction(1)),
)
_PREDICATE = "bernstein-222-depth0-v1"


def _down(x: np.ndarray) -> np.ndarray:
    return np.nextafter(x, -np.inf)


def _up(x: np.ndarray) -> np.ndarray:
    return np.nextafter(x, np.inf)


def _exact_interval(x: np.ndarray | float) -> Interval:
    """A degenerate interval, used only for exactly representable binary32 values and small dyadic constants."""
    a = np.asarray(x, dtype=np.float64)
    return a, a


def _iadd(a: Interval, b: Interval) -> Interval:
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _isub(a: Interval, b: Interval) -> Interval:
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _imul(a: Interval, b: Interval) -> Interval:
    alo, ahi = a
    blo, bhi = b
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        products = (alo * blo, alo * bhi, ahi * blo, ahi * bhi)
    lo = _down(np.minimum(np.minimum(products[0], products[1]), np.minimum(products[2], products[3])))
    hi = _up(np.maximum(np.maximum(products[0], products[1]), np.maximum(products[2], products[3])))
    return lo, hi


def _iscale(a: Interval, scalar: float) -> Interval:
    if scalar == 0.0:
        return _exact_interval(0.0)
    if scalar == 1.0:
        return a
    # Scalar multiplication needs only the two oriented endpoints, while still enclosing underflow/overflow.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if scalar > 0:
            return _down(a[0] * scalar), _up(a[1] * scalar)
        return _down(a[1] * scalar), _up(a[0] * scalar)


def _isum(terms: list[Interval]) -> Interval:
    if not terms:
        return _exact_interval(0.0)
    out = terms[0]
    for term in terms[1:]:
        out = _iadd(out, term)
    return out


def _det_interval(col_a: list[Interval], col_b: list[Interval], col_c: list[Interval]) -> Interval:
    m0 = _isub(_imul(col_b[1], col_c[2]), _imul(col_b[2], col_c[1]))
    m1 = _isub(_imul(col_b[0], col_c[2]), _imul(col_b[2], col_c[0]))
    m2 = _isub(_imul(col_b[0], col_c[1]), _imul(col_b[1], col_c[0]))
    return _iadd(_isub(_imul(col_a[0], m0), _imul(col_a[1], m1)), _imul(col_a[2], m2))


def _axis_bernstein_interval(v0: Interval, v1: Interval, v2: Interval) -> tuple[Interval, Interval, Interval]:
    # b1 = -1/2 v0 + 2 v1 - 1/2 v2.  Every operation remains interval-valued.
    mid = _iadd(_iadd(_iscale(v0, -0.5), _iscale(v1, 2.0)), _iscale(v2, -0.5))
    return v0, mid, v2


def _edge_interval(corners: dict[tuple[int, int, int], list[Interval]], axis: int, i: int, j: int) -> list[Interval]:
    if axis == 0:
        left, right = corners[0, i, j], corners[1, i, j]
    elif axis == 1:
        left, right = corners[i, 0, j], corners[i, 1, j]
    else:
        left, right = corners[i, j, 0], corners[i, j, 1]
    return [_isub(right[c], left[c]) for c in range(3)]


def _bilinear_column_interval(
    edges: tuple[list[Interval], list[Interval], list[Interval], list[Interval]],
    axis: int,
    s: float,
    t: float,
) -> list[Interval]:
    weights = ((1.0 - s) * (1.0 - t), s * (1.0 - t), (1.0 - s) * t, s * t)
    out: list[Interval] = []
    for component in range(3):
        terms = [_iscale(edge[component], weight) for edge, weight in zip(edges, weights, strict=True) if weight != 0.0]
        value = _isum(terms)
        if component == axis:
            value = _iadd(value, _exact_interval(1.0))
        out.append(value)
    return out


def _bernstein_intervals_tile(flow_tile: np.ndarray) -> list[Interval]:
    """Return all 27 coefficient intervals for a binary32 vertex tile ``[3,D+1,H+1,W+1]``."""
    if flow_tile.dtype != np.float32:
        raise TypeError(f"interval kernel requires float32 controls, got {flow_tile.dtype}")
    _, d1, h1, w1 = flow_tile.shape
    d, h, w = d1 - 1, h1 - 1, w1 - 1
    corners: dict[tuple[int, int, int], list[Interval]] = {}
    # binary32 -> binary64 is exact, so these are legitimate degenerate source intervals.
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                arr = flow_tile[:, i : i + d, j : j + h, k : k + w].astype(np.float64, copy=False)
                corners[i, j, k] = [_exact_interval(arr[c]) for c in range(3)]

    edges = {
        axis: tuple(_edge_interval(corners, axis, i, j) for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)))
        for axis in range(3)
    }
    col_a = [[_bilinear_column_interval(edges[0], 0, b, c) for c in _NODES] for b in _NODES]
    col_b = [[_bilinear_column_interval(edges[1], 1, a, c) for c in _NODES] for a in _NODES]
    col_c = [[_bilinear_column_interval(edges[2], 2, a, b) for b in _NODES] for a in _NODES]

    vals: list[list[list[Interval]]] = []
    for ia in range(3):
        plane: list[list[Interval]] = []
        for ib in range(3):
            row: list[Interval] = []
            for ic in range(3):
                row.append(_det_interval(col_a[ib][ic], col_b[ia][ic], col_c[ia][ib]))
            plane.append(row)
        vals.append(plane)

    va = [[_axis_bernstein_interval(vals[0][b][c], vals[1][b][c], vals[2][b][c]) for c in range(3)] for b in range(3)]
    va = [[[va[b][c][a] for c in range(3)] for b in range(3)] for a in range(3)]
    vb = [[_axis_bernstein_interval(va[a][0][c], va[a][1][c], va[a][2][c]) for c in range(3)] for a in range(3)]
    vb = [[[vb[a][c][b] for c in range(3)] for b in range(3)] for a in range(3)]
    vc = [[_axis_bernstein_interval(vb[a][b][0], vb[a][b][1], vb[a][b][2]) for b in range(3)] for a in range(3)]
    return [vc[a][b][c] for a in range(3) for b in range(3) for c in range(3)]


def _min_interval(coeffs: list[Interval]) -> Interval:
    lo = coeffs[0][0].copy()
    hi = coeffs[0][1].copy()
    for clo, chi in coeffs[1:]:
        np.minimum(lo, clo, out=lo)
        np.minimum(hi, chi, out=hi)
    return lo, hi


def _fraction_from_binary32(value: np.float32) -> Fraction:
    return Fraction.from_float(float(value))


def _cell_corners_exact(flow: np.ndarray, idx: tuple[int, int, int]) -> dict[tuple[int, int, int], list[Fraction]]:
    i, j, k = idx
    out: dict[tuple[int, int, int], list[Fraction]] = {}
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                z, y, x = i + di, j + dj, k + dk
                out[di, dj, dk] = [
                    Fraction(z) + _fraction_from_binary32(flow[0, z, y, x]),
                    Fraction(y) + _fraction_from_binary32(flow[1, z, y, x]),
                    Fraction(x) + _fraction_from_binary32(flow[2, z, y, x]),
                ]
    return out


def _min_bern_coeff_exact(corners: dict[tuple[int, int, int], list[Fraction]]) -> Fraction:
    """Exact minimum of the 27 base Bernstein coefficients for one cell."""

    def det_at(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
        na, nb, nc = 1 - a, 1 - b, 1 - c
        col_a = [
            (corners[1, 0, 0][t] - corners[0, 0, 0][t]) * (nb * nc)
            + (corners[1, 1, 0][t] - corners[0, 1, 0][t]) * (b * nc)
            + (corners[1, 0, 1][t] - corners[0, 0, 1][t]) * (nb * c)
            + (corners[1, 1, 1][t] - corners[0, 1, 1][t]) * (b * c)
            for t in range(3)
        ]
        col_b = [
            (corners[0, 1, 0][t] - corners[0, 0, 0][t]) * (na * nc)
            + (corners[1, 1, 0][t] - corners[1, 0, 0][t]) * (a * nc)
            + (corners[0, 1, 1][t] - corners[0, 0, 1][t]) * (na * c)
            + (corners[1, 1, 1][t] - corners[1, 0, 1][t]) * (a * c)
            for t in range(3)
        ]
        col_c = [
            (corners[0, 0, 1][t] - corners[0, 0, 0][t]) * (na * nb)
            + (corners[1, 0, 1][t] - corners[1, 0, 0][t]) * (a * nb)
            + (corners[0, 1, 1][t] - corners[0, 1, 0][t]) * (na * b)
            + (corners[1, 1, 1][t] - corners[1, 1, 0][t]) * (a * b)
            for t in range(3)
        ]
        return (
            col_a[0] * (col_b[1] * col_c[2] - col_b[2] * col_c[1])
            - col_a[1] * (col_b[0] * col_c[2] - col_b[2] * col_c[0])
            + col_a[2] * (col_b[0] * col_c[1] - col_b[1] * col_c[0])
        )

    vals = {
        (ia, ib, ic): det_at(a, b, c)
        for ia, a in enumerate(_FNODES)
        for ib, b in enumerate(_FNODES)
        for ic, c in enumerate(_FNODES)
    }
    best: Fraction | None = None
    for p in range(3):
        for q in range(3):
            for r in range(3):
                coeff = sum(
                    _M[p][ia] * _M[q][ib] * _M[r][ic] * vals[ia, ib, ic]
                    for ia in range(3)
                    for ib in range(3)
                    for ic in range(3)
                )
                best = coeff if best is None or coeff < best else best
    assert best is not None
    return best


def _as_exact_eps(eps: str | float | Fraction) -> Fraction:
    if isinstance(eps, Fraction):
        value = eps
    elif isinstance(eps, str):
        value = Fraction(eps)
    else:
        # Programmatic float input is interpreted as its human-readable decimal, not as an accidental binary64
        # threshold.  Callers needing the exact binary64 value can pass Fraction.from_float explicitly.
        value = Fraction(str(eps))
    if value <= 0:
        raise ValueError(f"eps must be a positive rational, got {eps!r}")
    return value


def _float_enclosure(value: Fraction) -> tuple[float, float]:
    centre = float(value)
    if not np.isfinite(centre):
        raise ValueError("eps is outside the finite float64 range")
    represented = Fraction.from_float(centre)
    if represented < value:
        return centre, float(np.nextafter(centre, np.inf))
    if represented > value:
        return float(np.nextafter(centre, -np.inf)), centre
    return centre, centre


def _validate_fp_environment() -> None:
    """Reject runtimes that visibly flush binary64 subnormals, invalidating one-ULP outward rounding."""
    smallest_normal = np.float64(np.finfo(np.float64).tiny)
    if smallest_normal * np.float64(0.5) == 0.0 or np.nextafter(np.float64(0.0), np.float64(1.0)) == 0.0:
        raise RuntimeError("binary64 gradual underflow is disabled; interval soundness prerequisites are absent")


def _boundary_nonzero_count(arr: np.ndarray) -> int:
    """Count non-zero components on the six faces of a validated ``[3,D,H,W]`` array."""
    mask = np.zeros(arr.shape[-3:], dtype=bool)
    mask[(0, -1), :, :] = True
    mask[:, (0, -1), :] = True
    mask[:, :, (0, -1)] = True
    return int(np.count_nonzero(arr[:, mask]))


def _validate_flow(flow: torch.Tensor) -> np.ndarray:
    if flow.dim() == 4:
        flow = flow.unsqueeze(0)
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"expected [1,3,D,H,W], got {tuple(flow.shape)}")
    if flow.dtype != torch.float32:
        raise TypeError(f"machine verifier accepts the stored float32 object only, got {flow.dtype}")
    if min(flow.shape[-3:]) < 2:
        raise ValueError(f"every spatial dimension must contain at least two vertices, got {tuple(flow.shape[-3:])}")
    arr = flow.detach().cpu().contiguous().numpy()
    if not np.isfinite(arr).all():
        raise ValueError("flow contains NaN or Inf")
    return arr[0]


def _normalise_tile_shape(tile_shape: tuple[int, int, int], cell_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(tile_shape) != 3 or any(int(v) <= 0 for v in tile_shape):
        raise ValueError(f"tile_shape must contain three positive integers, got {tile_shape}")
    return tuple(min(int(tile_shape[i]), cell_shape[i]) for i in range(3))


def certify_flow_exact(
    flow: torch.Tensor,
    eps: str | float | Fraction = "0.001",
    max_exact: int = 100_000,
    tile_shape: tuple[int, int, int] = (8, 64, 64),
) -> dict:
    """Decide the exact base-Bernstein sufficient predicate for one stored binary32 field.

    Status is one of ``CERTIFIED``, ``NOT_CERTIFIED_BY_PREDICATE`` or ``INCONCLUSIVE_RESOURCE_LIMIT``.
    ``certified`` is true only for the first status.  Resource exhaustion is never reported as predicate
    failure, and predicate failure is never called a witnessed fold.
    """
    _validate_fp_environment()
    arr = _validate_flow(flow)
    eps_q = _as_exact_eps(eps)
    eps_lo, eps_hi = _float_enclosure(eps_q)
    cell_shape = tuple(int(v - 1) for v in arr.shape[-3:])
    td, th, tw = _normalise_tile_shape(tile_shape, cell_shape)

    interval_pass = 0
    interval_fail = 0
    exact_checked = 0
    exact_fail = 0
    ambiguous_total = 0
    exact_min: Fraction | None = None
    interval_lo_min = float("inf")
    interval_hi_min = float("inf")
    failures: list[dict[str, object]] = []
    budget = max(0, int(max_exact))
    exact_targets: list[tuple[int, int, int]] = []

    for z0 in range(0, cell_shape[0], td):
        z1 = min(z0 + td, cell_shape[0])
        for y0 in range(0, cell_shape[1], th):
            y1 = min(y0 + th, cell_shape[1])
            for x0 in range(0, cell_shape[2], tw):
                x1 = min(x0 + tw, cell_shape[2])
                tile = arr[:, z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
                lo, hi = _min_interval(_bernstein_intervals_tile(tile))
                finite = np.isfinite(lo) & np.isfinite(hi)
                interval_lo_min = min(interval_lo_min, float(np.nanmin(lo)) if np.isfinite(lo).any() else -float("inf"))
                interval_hi_min = min(interval_hi_min, float(np.nanmin(hi)) if np.isfinite(hi).any() else -float("inf"))

                passed = finite & (lo >= eps_hi)
                failed = finite & (hi < eps_lo)
                ambiguous = ~(passed | failed)
                interval_pass += int(passed.sum())
                interval_fail += int(failed.sum())
                ambiguous_total += int(ambiguous.sum())

                for local in np.argwhere(failed):
                    cell = (z0 + int(local[0]), y0 + int(local[1]), x0 + int(local[2]))
                    if len(failures) < 50:
                        failures.append({"cell": cell, "kind": "interval", "upper": float(hi[tuple(local)])})

                for local in np.argwhere(ambiguous):
                    if len(exact_targets) < budget:
                        exact_targets.append((z0 + int(local[0]), y0 + int(local[1]), x0 + int(local[2])))

    # One rigorous interval failure already decides the existential predicate negatively.  Avoid spending
    # minutes on exact ambiguities that cannot change that verdict.
    if interval_fail == 0:
        for cell in exact_targets:
            value = _min_bern_coeff_exact(_cell_corners_exact(arr, cell))
            exact_checked += 1
            exact_min = value if exact_min is None or value < exact_min else exact_min
            if value < eps_q:
                exact_fail += 1
                if len(failures) < 50:
                    failures.append({"cell": cell, "kind": "exact", "value": str(value)})

    n_unresolved = ambiguous_total - exact_checked
    n_failures = interval_fail + exact_fail
    if n_failures:
        status = "NOT_CERTIFIED_BY_PREDICATE"
    elif n_unresolved:
        status = "INCONCLUSIVE_RESOURCE_LIMIT"
    else:
        status = "CERTIFIED"

    raw = arr[np.newaxis, ...].tobytes(order="C")
    return {
        "status": status,
        "certified": status == "CERTIFIED",
        "complete": n_unresolved == 0,
        "predicate": _PREDICATE,
        "implication": "det_D_phi_greater_equal_eps_on_every_cell",
        "epsilon_decimal": str(eps_q.numerator / eps_q.denominator),
        "epsilon_fraction": f"{eps_q.numerator}/{eps_q.denominator}",
        "comparator": ">=",
        "shape": list(arr[np.newaxis, ...].shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "boundary_nonzero_count": _boundary_nonzero_count(arr),
        "arithmetic_assumptions": "IEEE-754 binary64 round-to-nearest, gradual underflow, no fast-math",
        "tile_shape_cells": [td, th, tw],
        "n_cells": int(np.prod(cell_shape)),
        "n_interval_certified": interval_pass,
        "n_interval_failed": interval_fail,
        "n_exact_checked": exact_checked,
        "n_unresolved": n_unresolved,
        "n_failures": n_failures,
        "interval_lo_min": interval_lo_min,
        "interval_hi_min": interval_hi_min,
        "exact_min_over_ambiguous": None if exact_min is None else float(exact_min),
        "failures": failures,
    }


def _load_flow_npz(path: str | Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        if "flow" not in data:
            raise KeyError(f"{path}: expected an array named 'flow'")
        arr = np.asarray(data["flow"])
    if arr.dtype != np.float32:
        raise TypeError(f"{path}: expected stored float32 flow, got {arr.dtype}; refusing an implicit cast")
    if arr.ndim == 4:
        arr = arr[None]
    return torch.from_numpy(np.ascontiguousarray(arr))


def _resolve_paths(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.npz")))
        elif any(ch in spec for ch in "*?["):
            paths.extend(Path(v) for v in sorted(glob.glob(spec)))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(spec)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise FileNotFoundError("no flow .npz files matched")
    return unique


def _selftest() -> None:
    identity = torch.zeros(1, 3, 5, 6, 7, dtype=torch.float32)
    report = certify_flow_exact(identity, eps="0.001", tile_shape=(2, 3, 3))
    assert report["status"] == "CERTIFIED"

    folded = identity.clone()
    folded[0, 0, 2] = 3.0
    report = certify_flow_exact(folded, eps="0.001", tile_shape=(2, 3, 3))
    assert report["status"] == "NOT_CERTIFIED_BY_PREDICATE"
    print("cert_exact self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow", nargs="+", help="one or more .npz files, directories, or quoted glob patterns")
    parser.add_argument("--eps", default="0.001", help="exact decimal/rational threshold, e.g. 0.001 or 1/1000")
    parser.add_argument("--max-exact", type=int, default=100_000, help="maximum ambiguous cells checked exactly")
    parser.add_argument("--tile", type=int, nargs=3, metavar=("D", "H", "W"), default=(8, 64, 64))
    parser.add_argument("--report", help="optional JSON output containing every per-file report")
    parser.add_argument(
        "--require-zero-boundary",
        action="store_true",
        help="also require every saved boundary displacement component to equal zero",
    )
    args = parser.parse_args()

    if not args.flow:
        _selftest()
        return 0

    reports = []
    for path in _resolve_paths(args.flow):
        try:
            report = certify_flow_exact(
                _load_flow_npz(path), eps=args.eps, max_exact=args.max_exact, tile_shape=tuple(args.tile)
            )
            if args.require_zero_boundary and report["boundary_nonzero_count"] != 0:
                report["bernstein_status"] = report["status"]
                report["status"] = "BOUNDARY_CONSTRAINT_FAILED"
                report["certified"] = False
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            report = {"status": "INVALID_INPUT", "certified": False, "complete": False, "error": str(exc)}
        report = {"file": str(path), **report}
        reports.append(report)
        print(
            f"{report['status']:<35} {path}"
            + (f"  cells={report['n_cells']} exact={report['n_exact_checked']}" if "n_cells" in report else "")
        )

    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {output}")

    if all(r["status"] == "CERTIFIED" for r in reports):
        return 0
    if any(r["status"] in {"INCONCLUSIVE_RESOURCE_LIMIT", "INVALID_INPUT"} for r in reports):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
