"""Exact boundary condition that turns the cellwise certificate of one stored field into a global statement.

For a stored binary32 displacement array ``d`` of shape ``[1,3,D,H,W]`` let ``F(x) = x + I(d)(x)`` be its
piecewise-trilinear interpolant on the node lattice, the object that ``utils.cert_exact`` certifies cell by cell.
Suppose every boundary node ``v`` satisfies ``d_j(v) = q_j(v_j)`` (component ``j`` depends only on coordinate ``j``)
and ``a_j(k) = k + q_j(k)`` increases strictly in ``k``.  Then on the boundary ``F`` equals the product of the
piecewise-linear interpolants of the ``a_j``, an orientation-preserving homeomorphism of space, and a positive
cellwise certificate makes ``F`` a homeomorphism of the box onto ``prod_j [a_j(0), a_j(N_j - 1)]`` (Proposition
"separable monotone boundary" of the certificate note).  A zero boundary is the special case ``q_j = 0``.

Both conditions are sufficient, not necessary, and are decided exactly: separability by equality of the stored
binary32 values, monotonicity in rational arithmetic.  A failed condition proves no fold.  Nothing here concerns
the floating-point arithmetic inside a grid_sample kernel.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from itertools import pairwise

import numpy as np
import torch

from utils.cert_exact import _validate_flow

CHECK = "separable-monotone-boundary-v1"


def _boundary_nodes(plane: np.ndarray, end: bool) -> np.ndarray:
    """Values of one lattice plane that lie on the box boundary: the whole plane at an end, else its border."""
    if end:
        return plane.reshape(-1)
    return np.concatenate((plane[0], plane[-1], plane[1:-1, 0], plane[1:-1, -1]))


def boundary_report(flow: torch.Tensor) -> dict:
    """Decide separability and strict monotonicity of the boundary values of one stored float32 field."""
    arr = _validate_flow(flow)
    shape = arr.shape[1:]
    violations: list[dict[str, object]] = []
    profiles: list[list[np.float32]] = []
    for axis, size in enumerate(shape):
        planes = np.moveaxis(arr[axis], axis, 0)
        profile = []
        for k in range(size):
            nodes = _boundary_nodes(planes[k], k in (0, size - 1))
            low, high = nodes.min(), nodes.max()
            if low != high and len(violations) < 20:
                violations.append(
                    {"kind": "separability", "component": axis, "index": k, "min": float(low), "max": float(high)}
                )
            profile.append(low)
        profiles.append(profile)
    separable = not violations

    monotone = image_box = min_increment = None
    if separable:
        image_box, min_increment = [], []
        for axis, profile in enumerate(profiles):
            values = [Fraction(k) + Fraction.from_float(float(q)) for k, q in enumerate(profile)]
            steps = [right - left for left, right in pairwise(values)]
            for k, step in enumerate(steps):
                if step <= 0 and len(violations) < 20:
                    violations.append({"kind": "monotonicity", "component": axis, "index": k, "increment": str(step)})
            image_box.append([str(values[0]), str(values[-1])])
            min_increment.append(str(min(steps)))
        monotone = not violations

    status = "PASSED" if separable and monotone else "NOT_SEPARABLE" if not separable else "NOT_MONOTONE"
    return {
        "check": CHECK,
        "status": status,
        "passed": status == "PASSED",
        "separable": separable,
        "monotone": monotone,
        "image_box": image_box,
        "min_increment": min_increment,
        "violations": violations,
        "shape": list(arr[np.newaxis, ...].shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(arr[np.newaxis, ...].tobytes(order="C")).hexdigest(),
    }


def global_verdict(cell: dict, boundary: dict) -> dict:
    """Combine a ``utils.cert_exact`` report with ``boundary_report`` of the same stored bytes.

    ``GLOBAL_HOMEOMORPHISM`` needs a ``CERTIFIED`` cell predicate with positive epsilon and a passed boundary check.
    A resource-limited cell predicate stays inconclusive; every other outcome only fails to establish the claim.
    """
    if any(cell.get(key) != boundary.get(key) for key in ("sha256", "shape", "dtype")):
        raise ValueError("cell and boundary reports refer to different stored arrays")
    if not boundary["passed"]:
        status, reason = "NOT_ESTABLISHED", f"boundary check {boundary['status']}"
    elif cell["status"] == "INCONCLUSIVE_RESOURCE_LIMIT":
        status, reason = "INCONCLUSIVE_RESOURCE_LIMIT", "cell predicate undecided within its exact-arithmetic budget"
    elif cell["status"] == "CERTIFIED" and Fraction(cell["epsilon_fraction"]) > 0:
        status, reason = "GLOBAL_HOMEOMORPHISM", "positive cell certificate and separable monotone boundary"
    else:
        status, reason = "NOT_ESTABLISHED", f"cell predicate {cell['status']}"
    return {
        "status": status,
        "global_homeomorphism": status == "GLOBAL_HOMEOMORPHISM",
        "reason": reason,
        "image_box": boundary["image_box"] if status == "GLOBAL_HOMEOMORPHISM" else None,
        "cell_status": cell["status"],
        "boundary_status": boundary["status"],
        "sha256": boundary["sha256"],
        "scope": "sufficient conditions for the stored array's interpolant; NOT_ESTABLISHED is not a fold; "
        "grid_sample kernel arithmetic is not covered",
    }
