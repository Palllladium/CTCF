from __future__ import annotations

import csv
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments import eval_matching as evaluation
from experiments.core.inference_metrics import MetricProfile, metric_profile_for
from models.Matching import FeatureEncoder
from tools.analysis.run_artifacts import sha256_file
from utils.dice import IXI_VOI_LABELS
from utils.field_io import save_flow_npz_atomic


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_zero_solver_steps_preserve_parent_fine_details_exactly():
    torch.manual_seed(8)
    encoder = FeatureEncoder(1, width=2, channels=4, stride=2)
    moving, fixed = torch.rand(1, 1, 16, 16, 16), torch.rand(1, 1, 16, 16, 16)
    parent = 0.2 * torch.randn(1, 3, 16, 16, 16)
    config = dict(steps=0, step_size=0.25, reg_weight=0.01, outside_weight=1.0, max_update=0.5)
    output, diagnostics = evaluation.refine_field(encoder, moving, fixed, parent, config)
    assert torch.equal(output, parent)
    assert np.isfinite(diagnostics["energy_before"])


def test_half_descriptors_do_not_restrict_the_correction_to_a_coarse_field(monkeypatch):
    encoder = FeatureEncoder(1, width=2, channels=4, stride=2)
    moving = torch.rand(1, 1, 16, 16, 16) + 0.1
    parent = torch.zeros(1, 3, 16, 16, 16)

    def solver(fixed, _moving, field, **_options):
        assert fixed.shape[2:] == (8, 8, 8)
        assert field.shape[2:] == (16, 16, 16)
        result = field.clone()
        result[:, 0, 7, 7, 7] = 0.1
        return result

    monkeypatch.setattr(evaluation, "refine", solver)
    output, _ = evaluation.refine_field(
        encoder, moving, moving, parent, {"steps": 3, "reg_weight": 0.01, "outside_weight": 1.0}
    )
    assert torch.count_nonzero(output) == 1
    assert output[0, 0, 7, 7, 7] == pytest.approx(0.1)


def test_inference_freezes_overlap_in_warped_moving_coordinates(monkeypatch):
    moving, fixed = torch.ones(1, 1, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)
    moving[..., :3] = 0
    parent = torch.zeros(1, 3, 8, 8, 8)
    parent[:, 2] = 2
    encoder = nn.Identity()
    config = dict(steps=3, step_size=0.25, reg_weight=0.01, outside_weight=1.0, max_update=0.5)
    observed = []

    def solver(_fixed, _moving, start, *, mask, **_options):
        observed.append(mask)
        return start

    monkeypatch.setattr(evaluation, "refine", solver)
    evaluation.refine_field(encoder, moving, fixed, parent, config)
    # Pulling moving(x+2) moves its three-voxel zero strip to a one-voxel strip.
    expected = torch.ones_like(fixed)
    expected[..., :1] = 0
    assert torch.equal(observed[0], expected) and not observed[0].requires_grad
    with pytest.raises(ValueError, match="overlap support is empty"):
        evaluation.overlap_support(torch.zeros_like(moving), fixed, parent)


def test_ixi_evaluation_does_not_require_or_open_t2(tmp_path):
    rows = []
    for index in range(4):
        path = tmp_path / f"subject{index}.pkl"
        with path.open("wb") as stream:
            pickle.dump((np.ones((8, 8, 8), dtype=np.float32), np.full((8, 8, 8), 3, dtype=np.int16)), stream)
        rows.append(
            {
                "id": f"s{index}",
                "ixi_id": f"IXI{index:03}",
                "split": "Train" if index < 2 else "Val",
                "t1": path.name,
                "t2": f"absent{index}.npy",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "reviewed": True,
                "training": rows[:2],
                "validation": rows[2:],
                "expected_counts": {"training": 2, "validation": 2},
                "shape": [8, 8, 8],
                "validation_pairs": [["s2", "s3"], ["s3", "s2"]],
            }
        )
    )
    items, _profile, _signature, count = evaluation.cases(SimpleNamespace(dataset="IXI", manifest=manifest, limit=0))
    values = list(items)
    assert count == len(values) == 2
    assert values[0][0].shape == (1, 8, 8, 8)
    assert torch.unique(values[0][2]).tolist() == [2]


@pytest.fixture
def toy_evaluation(tmp_path, monkeypatch):
    base, metric = tmp_path / "base.pth", tmp_path / "metric.pth"
    base.write_bytes(b"base")
    metric.write_bytes(b"metric")
    image = torch.ones(1, 8, 8, 8)
    labels = torch.ones(1, 8, 8, 8, dtype=torch.long)
    samples = [(image, image, labels, labels, name) for name in ("a__b", "b__a")]
    monkeypatch.setattr(evaluation, "cases", lambda _args: (iter(samples), SimpleNamespace(labels=(1,)), "manifest", 2))
    encoder = nn.Identity()
    encoder.stride = 1
    monkeypatch.setattr(
        evaluation,
        "load_encoder",
        lambda _path, _device: (
            encoder,
            {"kind": "student", "base_sha256": sha256_file(base), "step": 1, "recipe": {"updates": 1}},
        ),
    )
    monkeypatch.setattr(
        evaluation, "refine_config", lambda _saved: {"steps": 3, "reg_weight": 0.01, "outside_weight": 1.0}
    )
    monkeypatch.setattr(evaluation, "load_base", lambda *_args: object())
    calls = []

    def forward(_model, moving, _fixed):
        calls.append("forward")
        return torch.zeros(moving.shape[0], 3, *moving.shape[2:])

    monkeypatch.setattr(evaluation, "base_flow", forward)
    monkeypatch.setattr(evaluation, "refine_field", lambda _net, _x, _y, flow, _config: (flow, {}))
    monkeypatch.setattr(evaluation, "dice_per_label", lambda *_args, **_kwargs: np.array([0.8]))
    monkeypatch.setattr(evaluation, "warp_labels", lambda labels, *_args: labels)
    monkeypatch.setattr(
        evaluation,
        "score",
        lambda _flow, case, elapsed, *_args, **_kwargs: {"case_id": case, "time_sec": elapsed, "dice_mean": 0.8},
    )
    monkeypatch.setattr(evaluation, "match_regularity", lambda flow, _target: (flow, {"matched": True}))
    args = SimpleNamespace(
        out=tmp_path / "output",
        base_checkpoint=base,
        metric_checkpoint=metric,
        endpoint=None,
        dataset="IXI",
        limit=0,
        certify=0,
        controls=1,
        device=torch.device("cpu"),
    )
    args.out.mkdir()
    return args, calls


def test_control_failure_keeps_raw_scores_and_resume_retries_only_failed_case(toy_evaluation, monkeypatch):
    args, calls = toy_evaluation
    attempts = []

    def control(_moving, _fixed, parent, _seconds, _config):
        attempts.append("control")
        if len(attempts) == 1:
            raise RuntimeError("control failed")
        return parent, {"seconds": 0.0, "matched_time": False}

    monkeypatch.setattr(evaluation, "timed_ncc", control)
    with pytest.raises(RuntimeError, match="Primary scores saved"):
        evaluation.evaluate(args)
    with (args.out / "FF/per_case.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert json.loads((args.out / "report.json").read_text())["control_failures"] == ["a__b"]
    evaluation.evaluate(args)
    assert len(calls) == 3
    assert len(attempts) == 3
    assert not json.loads((args.out / "report.json").read_text())["control_failures"]
    # A different checkpoint must not reuse the successful per-case cache.
    args.base_checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs or policy changed"):
        evaluation.evaluate(args)


def test_endpoint_rejects_wrong_weights_before_model_loading(toy_evaluation):
    args, calls = toy_evaluation
    args.endpoint = args.out / "endpoint.json"
    args.endpoint.write_text(json.dumps({"checkpoint_sha256": "wrong", "source_manifest_sha256": "manifest"}))
    with pytest.raises(ValueError, match="completed training endpoint"):
        evaluation.evaluate(args)
    assert not calls


def test_identity_endpoint_certifies_native_saved_fields_without_recovery(tmp_path):
    flow = torch.zeros(1, 3, 8, 8, 8)
    labels = torch.ones(1, 1, 8, 8, 8, dtype=torch.long)
    common = metric_profile_for("IXI")
    profile = MetricProfile((1,), common.jac_metrics, common.jac_log)
    rows, report = evaluation.certified_scores(flow, "a__b", 0.1, labels, labels, tmp_path, profile, "IXI")
    loaded, persisted = evaluation.load_saved_scores(tmp_path)
    assert loaded == rows and persisted == report
    assert report["global_homeomorphism"] and not report["errors"]
    assert rows["FF"]["dice_mean"] == rows["REP"]["dice_mean"] == pytest.approx(1)
    assert rows["REP"]["identity_used"] == 1 and rows["REP"]["recovery_used"] == 0


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_accepted_recovery_recertifies_chosen_bytes_and_keeps_raw_on_failure(tmp_path, monkeypatch, recovery_fails):
    raw = torch.ones(1, 3, 4, 4, 4)
    labels = torch.zeros(1, 1, 4, 4, 4)
    observed = []

    def scorer(flow, case, seconds, _xs, _ys, directory, **options):
        assert torch.equal(flow, raw)
        repaired = options["repair_fn"](flow)[0] if "repair_fn" in options else flow * 0.5
        observed.append(repaired.clone())
        save_flow_npz_atomic(directory / "FF.npz", flow)
        save_flow_npz_atomic(directory / "REP.npz", repaired)
        return {"FF": {"dice_mean": 0.7}, "REP": {"dice_mean": 0.71, "time_sec": seconds}}, {
            "errors": [],
            "global_homeomorphism": len(observed) > 1,
            "repair": {"tri_proj_iters": 53},
        }

    def recovery(flow, old, case, actual_labels, directory):
        assert torch.equal(flow, raw) and torch.equal(old, raw * 0.5)
        assert case["old_tri_iters"] == 53 and case["dataset"] == "IXI"
        assert actual_labels == (labels, labels)
        if recovery_fails:
            raise RuntimeError("recovery refused")
        save_flow_npz_atomic(directory / "fallback_0" / "REP.npz", torch.zeros_like(raw))
        return {"selected": "fallback_0", "fallback": True}

    monkeypatch.setattr(evaluation, "score_saved_field", scorer)
    monkeypatch.setattr(evaluation, "repair_one", recovery)
    rows, report = evaluation.certified_scores(raw, "a__b", 0.2, labels, labels, tmp_path, object(), "IXI")
    assert rows["FF"]["dice_mean"] == 0.7
    assert rows["REP"]["recovery_used"] == 1
    assert (tmp_path / "recovery/standard_report.json").exists()
    if recovery_fails:
        assert len(observed) == 1 and report["errors"]
        assert not report["global_homeomorphism"]
    else:
        assert len(observed) == 2 and not torch.count_nonzero(observed[-1])
        assert rows["REP"]["identity_used"] == rows["REP"]["recovery_fallback"] == 1
        assert report["global_homeomorphism"]


def write_readout(root, dataset, arm, value, *, certified=True):
    folder = root / dataset / arm
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(
        json.dumps({"binding": {"expected_cases": 2}, "complete_certified_endpoint": certified})
    )
    for mode in ("FF", "REP") if dataset == "IXI" else ("FF",):
        destination = folder / mode
        destination.mkdir()
        rows = [
            {"case_id": case, "dice_mean": value, **{f"dice_lbl_{label}": value for label in IXI_VOI_LABELS}}
            for case in ("a__b", "b__a")
        ]
        with (destination / "per_case.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def test_summary_uses_repaired_endpoints_and_reports_factor_interaction(tmp_path):
    values = {"base": 0.7, "t1_value": 0.71, "t1_solver": 0.72, "paired_value": 0.73, "paired_solver": 0.76}
    for arm, value in values.items():
        write_readout(tmp_path, "IXI", arm, value)
    args = SimpleNamespace(out=tmp_path, datasets=["IXI"], required_arms=list(values))
    evaluation.report(args)
    saved = json.loads((tmp_path / "matching_summary.json").read_text())
    comparisons = saved["comparisons"]
    assert comparisons["IXI_REP_paired_solver-base"]["resource_signal"]
    assert comparisons["IXI_REP_interaction"]["mean"] == pytest.approx(0.02)
    assert comparisons["IXI_REP_paired_solver-base"]["region_deltas"]["cortex"]["mean"] == pytest.approx(0.06)
    assert not comparisons["IXI_FF_paired_solver-base"]["resource_signal"]
    endpoint = tmp_path / "IXI/base/report.json"
    endpoint.write_text(json.dumps({"binding": {"expected_cases": 2}, "complete_certified_endpoint": False}))
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluation.report(args)
    assert json.loads((tmp_path / "matching_summary.json").read_text())["errors"]


def test_external_graph_summary_has_no_independent_pair_interval(tmp_path):
    for arm, value in (("base", 0.7), ("paired_solver", 0.71)):
        write_readout(tmp_path, "LPBA", arm, value)
    args = SimpleNamespace(out=tmp_path, datasets=["LPBA"], required_arms=["base", "paired_solver"])
    evaluation.report(args)
    delta = json.loads((tmp_path / "matching_summary.json").read_text())["comparisons"]["LPBA_FF_paired_solver-base"]
    assert "ci95" not in delta
    assert "share subjects" in delta["scope"]


def test_empty_summary_fails_and_writes_reason(tmp_path):
    args = SimpleNamespace(out=tmp_path, datasets=["IXI"], required_arms=["base", "paired_solver"])
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluation.report(args)
    assert "No complete comparisons" in json.loads((tmp_path / "matching_summary.json").read_text())["errors"]


def test_report_cli_does_not_initialize_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "report", lambda _args: None)
    monkeypatch.setattr(
        evaluation, "configure_precision", lambda *_args: pytest.fail("CPU report touched CUDA precision")
    )
    monkeypatch.setattr("sys.argv", ["eval_matching", "report", "--out", str(tmp_path)])
    evaluation.main()
