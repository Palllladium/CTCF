import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from datasets.IXI import IXIPairSampler
from experiments import contrast_diagnostics as checks
from utils.runtime import adjust_lr_ctcf_schedule


def test_anchor_compares_matching_early_epochs_not_the_late_plateau():
    old = [{"epoch": 0, "val_dice": 0.7789}, {"epoch": 2, "val_dice": 0.7788}, {"epoch": 99, "val_dice": 0.7801}]
    new = [{"epoch": 0, "val_dice": 0.7788}, {"epoch": 2, "val_dice": 0.7789}]
    assert checks.anchor_comparison(old, new)["passed"]
    new[-1]["val_dice"] = 0.7800
    assert not checks.anchor_comparison(old, new)["passed"]


def test_short_horizon_preserves_lr_and_pair_stream():
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=2e-5)
    for epoch in range(3):
        assert (
            adjust_lr_ctcf_schedule(optimizer, epoch, 3, 2e-5)
            == adjust_lr_ctcf_schedule(optimizer, epoch, 100, 2e-5)
            == 2e-5
        )
        short = IXIPairSampler(397, 0, "short", samples=399)
        full = IXIPairSampler(397, 0, "historical", samples=399)
        short.set_epoch(epoch)
        full.set_epoch(epoch)
        assert list(short) == list(full)


def test_audit_allows_new_diagnostics_but_rejects_changed_historical_training(monkeypatch):
    def output(command, **_):
        return (
            "experiments/train_CTCF.py\nutils/losses.py\n"
            if command[1] == "ls-tree"
            else "experiments/contrast_diagnostics.py\nutils/losses.py\n"
        )

    monkeypatch.setattr(checks.subprocess, "check_output", output)
    assert checks.frozen_training_diff() == ["utils/losses.py"]


@pytest.mark.parametrize("fail", [False, True])
def test_observation_restores_rng_and_model_mode_even_on_failure(tmp_path, monkeypatch, fail):
    class Dataset:
        def __init__(self, *_, **__):
            self.pairs = [(0, 1), (1, 0)]
            self.rows = [{"id": "a"}, {"id": "b"}]

        def __len__(self):
            return 2

        def __getitem__(self, _):
            return [torch.zeros(2, 3, 3, 3), torch.zeros(2, 3, 3, 3), torch.ones(1, 3, 3, 3), torch.ones(1, 3, 3, 3)]

    model = nn.Dropout()
    model.train()
    runner = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        args=SimpleNamespace(ixi_arm="b0", init_ckpt="parent", exp="unit"),
        _forward_flow=lambda *_: torch.rand(1, 3, 3, 3, 3),
    )

    def scoring(*_, **__):
        assert not model.training
        torch.rand(19)
        if fail:
            raise ValueError("readout failed")
        return {"dice_mean": 0.5}

    monkeypatch.setattr(checks, "IXIPairedDataset", Dataset)
    monkeypatch.setattr(checks, "score", scoring)
    monkeypatch.setattr(checks, "write_results", lambda *_, **__: None)
    torch.manual_seed(17)
    state = torch.random.get_rng_state().clone()
    if fail:
        with pytest.raises(ValueError, match="readout failed"):
            checks.observe(runner, {}, tmp_path)
    else:
        checks.observe(runner, {}, tmp_path)
    assert model.training
    assert torch.equal(state, torch.random.get_rng_state())


def test_optimizer_hook_observes_parameters_after_the_update(monkeypatch, tmp_path):
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    runner = checks.DiagnosticRunner.__new__(checks.DiagnosticRunner)
    runner.args = SimpleNamespace(ixi_arm="b0")
    runner.completed_updates = 19
    runner.observation_manifest = {}
    runner.observation_out = tmp_path
    values = []
    monkeypatch.setattr(checks, "observe", lambda *_: values.append(float(parameter.detach())))
    optimizer.register_step_post_hook(runner.observe_step)
    parameter.grad = torch.tensor(2.0)
    optimizer.step()
    assert values == pytest.approx([0.8])
    assert runner.completed_updates == 20


def test_readout_metadata_identifies_the_live_model_instead_of_the_initial_checkpoint(tmp_path, monkeypatch):
    class Dataset:
        def __init__(self, *_, **__):
            self.pairs = [(0, 1), (1, 0)]
            self.rows = [{"id": "a"}, {"id": "b"}]

        def __len__(self):
            return 2

        def __getitem__(self, _):
            return [torch.zeros(2, 3, 3, 3)] * 2 + [torch.ones(1, 3, 3, 3)] * 2

    runner = SimpleNamespace(
        model=nn.Identity(),
        device=torch.device("cpu"),
        args=SimpleNamespace(ixi_arm="b0", init_ckpt="parent.pth", exp="new_run"),
        _forward_flow=lambda *_: torch.zeros(1, 3, 3, 3, 3),
    )
    monkeypatch.setattr(checks, "IXIPairedDataset", Dataset)
    monkeypatch.setattr(checks, "score", lambda *_a, **_k: {"case_id": _a[1], "dice_mean": 0.5})
    checks.observe(runner, {}, tmp_path / "epoch3")
    report = json.loads((tmp_path / "epoch3/summary.json").read_text())
    assert report["ckpt_path"] == "live:new_run/epoch3"


def test_anchor_does_not_authorize_history_when_training_diff_is_nonempty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checks, "frozen_training_diff", lambda: ["utils/losses.py"])
    for name in ("new", "old_b0"):
        path = Path("logs") / name / "metrics.csv"
        path.parent.mkdir(parents=True)
        path.write_text("epoch,val_dice\n0,0.7789\n1,0.7788\n2,0.7788\n")
    args = SimpleNamespace(exp="new", source=Path("old"), out=tmp_path)
    checks.anchor(args)
    result = json.loads((tmp_path / "anchor.json").read_text())
    assert result["passed"] and not result["reuse_old_curves"] and result["add_b1_w4"]


def test_anchor_records_losses_and_initial_directions_without_new_gates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checks, "frozen_training_diff", lambda: [])
    for name, reg in (("old_b0", 0.1), ("new", 0.2)):
        path = Path("logs") / name / "metrics.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "val_dice", "loss_meters_json"))
            writer.writeheader()
            for epoch in (0, 2):
                writer.writerow(
                    {"epoch": epoch, "val_dice": 0.7788, "loss_meters_json": json.dumps({"reg": {"avg": reg}})}
                )
    for directory, delta in (
        (Path("old/parent/FF"), 0),
        (tmp_path / "b0_w1_s0/epoch0", 0.0001),
        (tmp_path / "b0_w4_s0/epoch0", 0),
    ):
        directory.mkdir(parents=True)
        (directory / "per_case.csv").write_text(f"case_id,dice_mean\na__b,{0.8 + delta}\nb__a,0.7\n")
    checks.anchor(SimpleNamespace(exp="new", source=Path("old"), out=tmp_path))
    result = json.loads((tmp_path / "anchor.json").read_text())
    assert result["reuse_old_curves"] and not result["add_b1_w4"]
    assert result["first_epoch_losses"]["fresh"]["reg"] == 0.2
    initial = result["epoch0_vs_historic_parent"]["b0_w1_s0"]
    assert len(initial["dice_delta_by_direction"]) == 2 and not initial["missing_directions"]
    assert initial["max_absolute_dice_delta"] == pytest.approx(0.0001)


def test_repaired_contrasts_expose_fallbacks_without_changing_the_raw_repeat_rule(tmp_path):
    for arm, gain, fallbacks in (("b0", 0, 0), ("b1", 0.006, 1), ("b2", 0, 0)):
        path = tmp_path / f"{arm}_w1_s0"
        (path / "epoch3").mkdir(parents=True)
        rows = [
            {"case_id": f"{a}{i}__{b}{i}", "dice_mean": 0.5 + gain}
            for i in range(29)
            for a, b in (("a", "b"), ("b", "a"))
        ]
        with (path / "epoch3/per_case.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("case_id", "dice_mean"))
            writer.writeheader()
            writer.writerows(rows)
        (path / "repair_summary.json").write_text(json.dumps({"rows": rows, "fallbacks": fallbacks, "certified": True}))
    checks.decide(SimpleNamespace(out=tmp_path))
    result = json.loads((tmp_path / "decision.json").read_text())
    assert result["repeat_arm"] == "b1"
    assert not result["REP_comparisons"]["b1"]["both_arms_fallback_free"]
    assert result["REP_comparisons"]["b2"]["both_arms_fallback_free"]
