import copy
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from experiments import contrast_probes
from experiments.contrast_probes import edited_state, match_regularity, train_pairs, weight_spectra
from experiments.core.cli_ctcf import ixi_similarity_weights
from experiments.core.helpers import select_lr_policy
from experiments.train_CTCF import Runner, parse_args
from models.CTCF import model as cascade
from tools.analysis.tests.test_contrasts import TinyL2
from utils import Grad3d, NCCVxm, RegisterModel


@pytest.fixture(autouse=True)
def small_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_fixed_probe_pairs_use_only_distinct_training_subjects():
    manifest = {"training": [{"id": str(i)} for i in range(397)], "validation": [{"id": "heldout"}]}
    first = train_pairs(manifest)
    assert first == train_pairs(manifest) and len(first) == 30
    assert len({subject for pair in first for subject in pair}) == 60
    with pytest.raises(ValueError, match="distinct"):
        train_pairs({"training": manifest["training"][:3]})


def test_weight_edits_preserve_buffers_and_isolate_each_level():
    first = {f"level{i}.weight": torch.eye(3) for i in (1, 2, 3)} | {"st.grid": torch.arange(3)}
    second = {key: value + 1 if "weight" in key else value for key, value in first.items()}
    names = {key for key in first if "weight" in key}
    hybrid = edited_state(first, second, levels=("level2",), parameter_names=names)
    assert torch.equal(hybrid["level1.weight"], first["level1.weight"])
    assert torch.equal(hybrid["level2.weight"], second["level2.weight"])
    mix = edited_state(first, second, fraction=1.5, parameter_names=names)
    torch.testing.assert_close(mix["level1.weight"], first["level1.weight"] + 1.5)
    assert torch.equal(mix["st.grid"], first["st.grid"])
    rows = weight_spectra(first, second, names)
    assert all(row["rank90"] == 1 and row["top_energy_fraction"] == pytest.approx(1) for row in rows)
    with pytest.raises(ValueError, match="buffer"):
        edited_state(first, second | {"st.grid": torch.zeros(3)}, fraction=0.5, parameter_names=names)


def test_geometry_matching_reaches_the_target_without_using_images_or_labels():
    field = torch.rand(1, 3, 12, 13, 14)
    from experiments.dispatch1.eval_consensus import smooth

    target = float(Grad3d(penalty="l2")(smooth(field, 0.37)))
    output, report = match_regularity(field, target)
    assert report["matched"] and report["relative_R_gap"] < 0.001
    assert float(Grad3d(penalty="l2")(output)) == pytest.approx(report["regularity"])
    _, impossible = match_regularity(field, 2 * float(Grad3d(penalty="l2")(field)))
    assert not impossible["matched"] and impossible["sigma"] == 0


def test_explicit_contrast_weights_validate_and_keep_old_defaults():
    assert ixi_similarity_weights(SimpleNamespace(ixi_arm="b0")) == (1, 0)
    assert ixi_similarity_weights(SimpleNamespace(ixi_arm="b1")) == (0.5, 0.5)
    with pytest.raises(ValueError, match="true T2"):
        ixi_similarity_weights(SimpleNamespace(ixi_arm="b0", ixi_t2_weight=0.5))
    for a, b in ((-1, 0), (0, 0), (float("nan"), 1)):
        with pytest.raises(ValueError, match="weights"):
            ixi_similarity_weights(SimpleNamespace(ixi_arm="b1", ixi_t1_weight=a, ixi_t2_weight=b))


def test_t2_intervention_scores_the_original_observations_not_the_corrupted_inputs(monkeypatch):
    runner = Runner.__new__(Runner)
    runner.args = SimpleNamespace(ixi_arm="b2")
    runner.ctx = SimpleNamespace(ncc=NCCVxm(win=(3, 3, 3)), reg_model=RegisterModel((7, 8, 9)))
    runner._forward_flow = lambda x, y: (x[:, 1:2] - y[:, 1:2]).expand(-1, 3, -1, -1, -1) * 0.01
    monkeypatch.setattr(contrast_probes, "score", lambda *_, **__: {})
    x, y = torch.rand(1, 2, 7, 8, 9), torch.rand(1, 2, 7, 8, 9)
    tx, ty = torch.rand(1, 1, 7, 8, 9), torch.rand(1, 1, 7, 8, 9)
    zero_x, zero_y = x.clone(), y.clone()
    zero_x[:, 1:] = zero_y[:, 1:] = 0
    row, _ = contrast_probes.measure(runner, (x, y, tx, ty, [tx, ty], "a__b"), (zero_x, zero_y))
    field = runner._forward_flow(zero_x, zero_y)
    expected = 0.5 * (
        runner.ctx.ncc(runner.ctx.reg_model((tx, field)), ty) + runner.ctx.ncc(runner.ctx.reg_model((ty, field)), tx)
    )
    assert row["d2"] == pytest.approx(float(expected)) and row["d2"] < -0.01


def test_summary_separates_fixed_dose_effects_interaction_and_missing_rep(tmp_path):
    source, out = tmp_path / "source", tmp_path / "new"
    for name, dice in (
        ("b0", 0.7),
        ("b1", 0.72),
        ("half_t1", 0.69),
        ("full_t1_t2", 0.74),
        ("b0_s1", 0.71),
        ("b1_s1", 0.73),
    ):
        directory = source / f"{name}_w1_s0" if name in ("b0", "b1") else out / name
        for epoch in (3, 10) if name.endswith("s1") else (3,):
            path = directory / f"epoch{epoch}/per_case.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("case_id", "dice_mean", "dice_lbl_1"))
                writer.writeheader()
                writer.writerows(
                    {"case_id": f"{a}{i}__{b}{i}", "dice_mean": dice, "dice_lbl_1": dice}
                    for i in range(29)
                    for a, b in (("a", "b"), ("b", "a"))
                )
    contrast_probes.summary(SimpleNamespace(source=source, out=out))
    report = json.loads((out / "mechanism_summary.json").read_text())
    assert report["T2_at_half_T1"]["mean"] == pytest.approx(0.03)
    assert report["T2_at_full_T1"]["mean"] == pytest.approx(0.04)
    assert report["T1_T2_interaction"]["mean"] == pytest.approx(0.01)
    assert report["T2_at_half_T1"]["metric_deltas"]["dice_lbl_1"] == pytest.approx(0.03)
    assert len(report["factor_REP"]["missing"]) == 4


def test_smoothing_keeps_search_status_in_trace_and_writes_real_numeric_tables(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "out"
    cases = [f"{a}{i}__{b}{i}" for i in range(29) for a, b in (("a", "b"), ("b", "a"))]
    for case in cases:
        path = source / "b1_w1_s0/epoch3/pairs" / case / "FF.npz"
        path.parent.mkdir(parents=True)
        path.touch()
    out.mkdir()
    (out / "probe_inputs.json").write_text(
        json.dumps({"fields": {arm: dict.fromkeys(cases, "sha") for arm in ("b0", "b1", "parent")}})
    )
    manifest = {"validation": [{"id": f"{a}{i}", "t1": "unused"} for i in range(29) for a in ("a", "b")]}
    monkeypatch.setattr(contrast_probes, "read_paired_manifest", lambda *_, **__: manifest)
    monkeypatch.setattr(contrast_probes, "sha256_file", lambda *_: "sha")
    monkeypatch.setattr(contrast_probes, "load_flow_npz", lambda *_: torch.zeros(1, 3, 3, 3, 3))
    monkeypatch.setattr(
        contrast_probes, "pkload", lambda *_: (torch.zeros(3, 3, 3).numpy(), torch.zeros(3, 3, 3).numpy())
    )
    monkeypatch.setattr(
        contrast_probes,
        "match_regularity",
        lambda flow, target: (
            flow,
            {
                "regularity": target,
                "matched": True,
                "relative_R_gap": 0,
                "search_status": "BRACKET_REFINED",
                "evaluations": [{"sigma": 0, "R": target}],
            },
        ),
    )
    monkeypatch.setattr(contrast_probes, "score", lambda _field, case, *_, **__: {"case_id": case, "dice_mean": 0.5})
    contrast_probes.smoothing(
        SimpleNamespace(source=source, out=out, original=tmp_path / "original", manifest="manifest", device="cpu")
    )
    with (out / "smoothing/b0/per_case.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 58 and "search_status" not in rows[0] and rows[0]["matched"] == "True"
    traces = json.loads((out / "smoothing/search.json").read_text())
    assert len(traces) == 116 and all(row["search_status"] == "BRACKET_REFINED" for row in traces)


def small_runner(monkeypatch, **changes):
    monkeypatch.setattr(cascade, "_build_level2", lambda config, _: TinyL2(config))
    args = parse_args(
        [
            "--ds",
            "SYNTH",
            "--config",
            "CTCF-CascadeA-Mamba",
            "--image_channels",
            "2",
            "--synth_vol_size",
            "32",
            "32",
            "32",
            "--l1_base_ch",
            "2",
            "--l3_base_ch",
            "2",
            "--l3_svf",
            "0",
            "--finetune",
            "1",
            "--ixi_arm",
            "b1",
            "--lr",
            "0.00002",
        ]
    )
    vars(args).update(w_reg=1, **changes)
    runner = Runner(args, torch.device("cpu"))
    runner.is_synth = False
    return runner


def test_actual_loss_uses_both_coefficients_and_preserves_three_epoch_prefix(monkeypatch):
    torch.manual_seed(4)
    first = small_runner(monkeypatch, max_epoch=3)
    initial = copy.deepcopy(first.model.state_dict())
    x, y, tx, ty = [torch.rand(1, channels, 32, 32, 32) for channels in (2, 2, 1, 1)]
    endpoints = []
    for horizon in (3, 10):
        runner = small_runner(monkeypatch, max_epoch=horizon)
        runner.model.load_state_dict(initial)
        for epoch in range(3):
            assert select_lr_policy(runner, runner.optimizer, epoch, horizon, 2e-5) == 2e-5
            runner.optimizer.zero_grad()
            loss, *_ = runner._pair_loss(x, y, None, None, epoch, (tx, ty))
            loss.backward()
            runner.optimizer.step()
        endpoints.append(runner.model.state_dict())
    for key in endpoints[0]:
        torch.testing.assert_close(endpoints[0][key], endpoints[1][key], rtol=0, atol=0)
    runner.args.w_reg = runner.args.w_icon = runner.args.w_jac = 0
    for a, b in ((0.5, 0), (1, 0.5), (0.5, 0.5), (1, 0)):
        runner.args.ixi_t1_weight, runner.args.ixi_t2_weight = a, b
        loss, logs, *_ = runner._pair_loss(x, y, None, None, 0, (tx, ty))
        assert float(loss.detach()) == pytest.approx(a * logs["ncc_t1"] + b * logs["ncc_t2"], abs=1e-6)
