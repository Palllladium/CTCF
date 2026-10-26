import json
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch

from tools.analysis import materialize_psi
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import certify_flow_exact
from utils.field import legacy_sampling_displacement
from utils.field_io import load_flow_npz, save_flow_npz_atomic

SHAPE = (6, 7, 8)


def _smooth_interior_flow(amplitude=0.2):
    z, y, x = torch.meshgrid(*(torch.linspace(0, torch.pi, n) for n in SHAPE), indexing="ij")
    bump = torch.sin(z) * torch.sin(y) * torch.sin(x)
    return (amplitude * torch.stack((bump, -bump, 0.5 * bump))[None]).float()


def test_zero_boundary_is_the_special_case_with_the_identity_box():
    flow = torch.zeros(1, 3, *SHAPE)
    boundary = boundary_report(flow)
    assert boundary["passed"] and boundary["separable"] and boundary["monotone"]
    assert boundary["image_box"] == [["0", str(n - 1)] for n in SHAPE]
    verdict = global_verdict(certify_flow_exact(flow, eps="0.001"), boundary)
    assert verdict["status"] == "GLOBAL_HOMEOMORPHISM"


def test_materialized_sampler_map_has_a_separable_monotone_boundary():
    psi = legacy_sampling_displacement(_smooth_interior_flow())
    boundary = boundary_report(psi)
    assert boundary["passed"]
    for (low, high), n in zip(boundary["image_box"], SHAPE, strict=True):
        assert float(Fraction(low)) == pytest.approx(-0.5, abs=1e-6)
        assert float(Fraction(high)) == pytest.approx(n - 0.5, abs=1e-6)
    cell = certify_flow_exact(psi, eps="0.001")
    assert global_verdict(cell, boundary)["status"] == "GLOBAL_HOMEOMORPHISM"


def test_a_tangential_boundary_change_breaks_separability_without_claiming_a_fold():
    flow = torch.zeros(1, 3, *SHAPE)
    flow[0, 1, 0, 2, 3] = 0.25
    boundary = boundary_report(flow)
    assert boundary["status"] == "NOT_SEPARABLE"
    assert boundary["violations"][0] == {"kind": "separability", "component": 1, "index": 2, "min": 0.0, "max": 0.25}
    verdict = global_verdict(certify_flow_exact(flow, eps="0.001"), boundary)
    assert verdict["status"] == "NOT_ESTABLISHED" and verdict["cell_status"] == "CERTIFIED"


def test_monotonicity_is_decided_in_exact_arithmetic():
    flow = torch.zeros(1, 3, *SHAPE)
    step = np.float32(-1 + 2.0**-24)
    flow[0, 2, :, :, 3] = float(step)
    assert np.float32(3) + step == np.float32(2)  # float32 rounding would call this step zero
    assert boundary_report(flow)["status"] == "PASSED"
    assert boundary_report(flow)["min_increment"][2] == "1/16777216"
    for value in (-1.0, -1.5):
        flow[0, 2, :, :, 3] = value
        report = boundary_report(flow)
        assert report["status"] == "NOT_MONOTONE"
        assert report["violations"][0]["component"] == 2 and report["violations"][0]["index"] == 2


def test_non_finite_input_is_rejected():
    flow = torch.zeros(1, 3, *SHAPE)
    flow[0, 0, 2, 3, 4] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        boundary_report(flow)


def test_a_passed_boundary_never_overrides_a_failed_or_unresolved_cell_predicate():
    flow = torch.zeros(1, 3, *SHAPE)
    flow[0, 0, 2, 2:-2, 2:-2] = 3.0
    boundary = boundary_report(flow)
    cell = certify_flow_exact(flow, eps="0.001")
    assert boundary["passed"] and cell["status"] == "NOT_CERTIFIED_BY_PREDICATE"
    assert global_verdict(cell, boundary)["status"] == "NOT_ESTABLISHED"
    identity = torch.zeros(1, 3, *SHAPE)
    unresolved = {**certify_flow_exact(identity, eps="0.001"), "status": "INCONCLUSIVE_RESOURCE_LIMIT"}
    verdict = global_verdict(unresolved, boundary_report(identity))
    assert verdict["status"] == "INCONCLUSIVE_RESOURCE_LIMIT" and not verdict["global_homeomorphism"]


def test_reports_of_different_bytes_are_not_combined():
    first, second = torch.zeros(1, 3, *SHAPE), torch.zeros(1, 3, *SHAPE)
    second[0, 0, 2, 3, 4] = 0.1
    with pytest.raises(ValueError, match="different stored arrays"):
        global_verdict(certify_flow_exact(first, eps="0.001"), boundary_report(second))


def test_materialize_psi_certifies_the_bytes_it_saved(tmp_path, monkeypatch):
    save_flow_npz_atomic(tmp_path / "phi" / "case.npz", _smooth_interior_flow())
    report_path = tmp_path / "report.json"
    argv = ["materialize_psi", "--flows", str(tmp_path / "phi"), "--out", str(tmp_path / "psi"), "--report"]
    monkeypatch.setattr(sys, "argv", [*argv, str(report_path)])
    assert materialize_psi.main() == 0
    (report,) = json.loads(report_path.read_text(encoding="utf-8"))
    stored = load_flow_npz(tmp_path / "psi" / "case.npz")
    assert report["status"] == "GLOBAL_HOMEOMORPHISM"
    assert report["sha256"] == boundary_report(stored)["sha256"] == report["cell"]["sha256"]
