import json
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from experiments.repair_diagnostics import (
    cell_probe,
    certify_candidate,
    finish,
    load_repair_result,
    normal_band,
    project,
    repair_one,
    spatial_report,
)
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import certify_flow_exact
from utils.field import boundary_vertex_mask, identity_collar


def test_normal_band_fixes_a_collar_induced_fold_and_preserves_the_anchor():
    torch.set_num_threads(2)
    raw = torch.zeros(1, 3, 37, 49, 37)
    raw[:, 1] = -11
    broken = identity_collar(raw, 4)
    corrected = normal_band(raw, 12)
    assert float(corrected[0, 1, 18, 12, 18]) + 12 == 1
    assert not corrected.masked_select(boundary_vertex_mask(corrected).expand_as(corrected)).any()
    bad = spatial_report(broken, reference=raw)
    assert bad["introduced_bad_cells_float"] > 0
    assert bad["introduced_probe_after"]["negative_witness"]
    assert not bad["introduced_probe_before"]["below_epsilon_witness"]
    cell = certify_flow_exact(corrected, eps="0.001")
    assert global_verdict(cell, boundary_report(corrected))["global_homeomorphism"]


def test_exact_probe_distinguishes_a_fold_from_a_positive_low_margin():
    for slope, negative in ((-0.5, True), (0.0008, False)):
        field = torch.zeros(1, 3, 4, 4, 4)
        field[:, 1] = (slope - 1) * torch.arange(4)[None, :, None]
        result = cell_probe(field, [0, 0, 0])
        assert result["negative_witness"] is negative
        assert result["below_epsilon_witness"]


def test_exact_subdivision_can_prove_a_positive_cell_rejected_by_depth_zero():
    z, y, x = torch.meshgrid(*[torch.arange(2) for _ in range(3)], indexing="ij")
    field = torch.stack(((y - 0.5) * z - 0.25 * x - z, torch.zeros_like(y), 0.25 * z + (y - 0.5) * x - x))[None].float()
    result = cell_probe(field, [0, 0, 0])
    assert not result["negative_witness"] and not result["below_epsilon_witness"]
    assert Fraction(result["subdivision_bounds"]["0"]) < 0
    assert Fraction(result["subdivision_bounds"]["1"]) == Fraction(1, 16)
    assert "diagnostic only" in result["subdivision_scope"]
    assert not certify_flow_exact(field, eps="0.001")["certified"]


def test_full_cell_localisation_is_not_truncated_to_fifty_examples():
    raw = torch.zeros(1, 3, 19, 21, 23)
    raw[:, 1] = -7
    report = spatial_report(identity_collar(raw, 4), labels=torch.zeros(1, 1, 19, 21, 23))
    assert report["failed_cells_float"] > 50
    assert sum(report["distance_histogram"]) == report["failed_cells_float"]
    assert sum(report["nearest_face_counts"]) == report["failed_cells_float"]
    assert report["bad_cells_touching_fixed_labels"] == 0


def test_both_projection_stages_keep_the_exact_zero_boundary():
    raw = identity_collar(torch.randn(1, 3, 9, 11, 13) * 0.2, 2)
    repaired, after_digital, _ = project(raw, digital_iters=2, tri_iters=2)
    mask = boundary_vertex_mask(raw).expand_as(raw)
    assert not repaired.masked_select(mask).any()
    assert not after_digital.masked_select(mask).any()


@pytest.mark.parametrize("width", (0, 20))
def test_invalid_normal_band_is_rejected(width):
    with pytest.raises(ValueError, match="boundary layers"):
        normal_band(torch.zeros(1, 3, 13, 15, 17), width)


def test_cached_result_is_bound_to_the_selected_saved_bytes(tmp_path):
    field = torch.zeros(1, 3, 5, 5, 5)
    labels = [torch.ones(1, 1, 5, 5, 5, dtype=torch.long)] * 2
    candidate = certify_candidate(field, tmp_path / "standard80", {"dataset": "IXI", "case_id": "a__b"}, labels)
    assert "time_sec" not in candidate["metrics"]
    record = {"selected": "standard80", "source_sha256": "original", "global_REP": candidate["global_REP"]}
    (tmp_path / "report.json").write_text(json.dumps(record))
    assert load_repair_result(tmp_path, "original") == record
    with pytest.raises(ValueError, match="changed after repair"):
        load_repair_result(tmp_path, "another source")
    (tmp_path / "standard80/REP.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Selected repaired bytes"):
        load_repair_result(tmp_path, "original")


def test_intermediate_readout_does_not_claim_an_unperformed_exact_certificate(tmp_path):
    field = torch.zeros(1, 3, 5, 5, 5)
    labels = [torch.ones(1, 1, 5, 5, 5, dtype=torch.long)] * 2
    result = certify_candidate(field, tmp_path, {"dataset": "IXI", "case_id": "a__b"}, labels, verify=False)
    assert not result["exact_verification_performed"]
    assert "global_REP" not in result and "certificate_REP" not in result
    assert not (tmp_path / "REP.npz").exists()


def test_bounded_search_keeps_a_certified_control_and_checks_its_saved_endpoint(tmp_path):
    field = torch.zeros(1, 3, 37, 37, 37)
    labels = [torch.ones(1, 1, 37, 37, 37, dtype=torch.long)] * 2
    case = {"dataset": "IXI", "case_id": "a__b", "old_tri_iters": 0, "source_sha256": "source"}
    result = repair_one(field, field, case, labels, tmp_path)
    assert result["global_REP"]["global_homeomorphism"] and not result["fallback"]
    assert result["selected"] == "continue_160"
    assert result["stages"]["fresh_tri80"]["failed_cells_float"] == 0
    assert all(not item["report"]["exact_verification_performed"] for item in result["candidates"][1:])
    assert load_repair_result(tmp_path, "source") == result
    assert result["candidates"][0]["report"]["change_max_voxels"] == 0


@pytest.mark.parametrize("components,shift", [((1, 2), -11), ((0, 1, 2), -8)])
def test_normal_band_can_fold_where_multiple_boundary_bands_intersect(components, shift):
    raw = torch.zeros(1, 3, 48, 48, 48)
    raw[:, components] = shift
    report = spatial_report(normal_band(raw, 16), reference=raw)
    assert report["failed_cells_float"] > 0
    assert report["bad_near_multiple_faces"] == report["failed_cells_float"]
    assert report["exact_probe"]["negative_witness"]
    assert not report["introduced_probe_before"]["below_epsilon_witness"]


@pytest.mark.parametrize("diagnostic_pass", ("width4", "both640"))
def test_representatives_do_not_change_the_endpoint_selection_rule(tmp_path, monkeypatch, diagnostic_pass):
    from experiments import repair_diagnostics as checks

    monkeypatch.setattr(checks, "spatial_report", lambda *_, **__: {"failed_cells_float": 1})
    monkeypatch.setattr(checks, "project", lambda field, **_: (field, field, {}))
    monkeypatch.setattr(
        checks, "trilinear_project", lambda field, **_: (field, SimpleNamespace(iterations=80, certified=False))
    )
    monkeypatch.setattr(checks, "_trilinear_cell_cert_bound", lambda *_, **__: torch.ones(1))

    def candidate(_field, directory, _case, _labels, *_, verify=True):
        result = {"metrics": {"dice_mean": 0.5}, "exact_verification_performed": verify}
        if verify:
            result["global_REP"] = {"global_homeomorphism": directory.name in (diagnostic_pass, "normal16")}
        return result

    monkeypatch.setattr(checks, "certify_candidate", candidate)
    field = torch.zeros(1, 3, 37, 37, 37)
    for representative in (False, True):
        case = {"case_id": "a__b", "representative": representative}
        result = repair_one(field, field, case, [field[:, :1]] * 2, tmp_path / str(representative))
        assert result["selected"] == "normal16" and not result["fallback"]


def test_finish_writes_partial_results_before_reporting_unresolved_fields(tmp_path):
    from tools.analysis.run_artifacts import sha256_file

    source, output = tmp_path / "source", tmp_path / "out"
    output.mkdir()
    cases = []
    for index, directory in enumerate(("b0/pairs/a__b", "b0/pairs/b__a", "consensus/direct/pairs/c__d")):
        path = source / directory
        path.mkdir(parents=True)
        (path / "REP.npz").write_bytes(b"original source bytes")
        (path / "report.json").write_text(json.dumps({"global_REP": {"global_homeomorphism": index == 0}}))
        (path / "metrics.json").write_text(json.dumps({"REP": {"dice_mean": 0.79}}))
        cases.append(
            {
                "directory": directory,
                "old_pass": index == 0,
                "ff_dice": 0.8,
                "old_rep_dice": 0.79,
                "report_sha256": sha256_file(path / "report.json"),
                "metrics_sha256": sha256_file(path / "metrics.json"),
                "field_sha256": {"REP": sha256_file(path / "REP.npz")},
            }
        )
    (output / "inputs.json").write_text(json.dumps({"cases": cases}))
    failure = output / "repair/b0/b__a"
    failure.mkdir(parents=True)
    (failure / "error.json").write_text(json.dumps({"error": "repair failed"}))
    with pytest.raises(RuntimeError, match=r"partial repair_summary\.json was saved"):
        finish(SimpleNamespace(source=source, out=output))
    result = json.loads((output / "repair_summary.json").read_text())
    assert result["fields"] == 3 and result["resolved_fields"] == result["global_certified"] == 1
    assert len(result["unresolved"]) == 2 and not result["complete"]
    assert not result["functional_repair_complete_without_fallback"]
    assert result["groups"]["b0"]["directions"] == 1 and result["groups"]["b0"]["expected_directions"] == 2
    assert result["groups"]["consensus/direct"]["new_REP"] is None
