from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from experiments import dispatch1

H3 = {
    "direct": 0.80,
    "recursion": 0.81,
    "bridge_minimax": 0.81,
    "bridge_minimax64": 0.801,
    "bridge_proxy": 0.802,
    "bridge_random": 0.79,
    "oracle_ff": 0.85,
}
REP = {"h2_B": 0.800, "h2_P": 0.803, "h2_A": 0.807, "d_minimax_bridge": 0.81, "d_minimax_direct": 0.80}
REP |= {"d_minimax_random": 0.80, "im2grid_l1": 0.84, "im2grid_l4": 0.83, "h1_D3": 0.80, "h1_E3": 0.79}
CONTROL = {"FF": {"dice_mean": 0.80, "sdlogj": 0.1}, "REP": {"dice_mean": 0.82, "sdlogj": 0.1}}
WEIGHTS = {"known_warp_gamma": 0.25, "distill_beta": 0.5, "w_mse": 30.0}


class _Handle:
    def __init__(self, code, polls=1):
        self.code, self.polls = code, polls

    def poll(self):
        self.polls -= 1
        return self.code if self.polls <= 0 else None


def _outputs(step_sec, h3):
    """What each preparation task writes, by file name."""
    step = {"step_sec": step_sec}
    gamma = {f"gamma_{t}.json": WEIGHTS for t in ("point", "set", "structure")}
    matcher = {"lambda.json": {"lambda": 2.0, "step_sec": step_sec}}
    check = {"seconds_per_pass": 1e-4, "score_seconds": 0.01, "oracle_score_seconds": 1e-5}
    return {
        "pre_parent": {
            "parent.json": {"name": "mamba_nosvf", **dispatch1.PARENTS["mamba_nosvf"], "load": {}},
            "control.json": CONTROL,
        },
        "pre_check": {"check.json": {**check, "test_pairs": 19, "bank_subjects": 394, "sub_bank_subjects": 64}},
        "h2_prep": {"h2_B.json": step, "h2_P.json": step, "h2_A.json": step, **gamma},
        "d_prep": {"d_random.json": step},
        "mse_prep": {"mse.json": step, "weight.json": WEIGHTS},
        "im2grid_prep": matcher,
        "h1_prep_D3": matcher,
        "h1_prep_E3": matcher,
        "h3": {"arms.json": {"arms": {arm: {"REP": value} for arm, value in h3.items()}}},
    }


def _fake(ctx, step_sec=1e-4, check_code=0, h3=None, launched=None):
    outputs = _outputs(step_sec, h3 or H3)

    def write(task, name, data):
        path = Path(ctx.path(task, name))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def launch(task, gpu, commands, log):
        if launched is not None:
            launched[task.name] = (gpu, " ".join(commands))
        if task.name in outputs:
            for name, data in outputs[task.name].items():
                write(task.name, name, data)
        elif task.name.endswith("_cal") or task.name.startswith("d_cal"):
            write(task.name, "weight.json", WEIGHTS)
        elif not task.name.startswith("pre_"):
            dice = REP.get(task.name, 0.8)
            write(task.name, "REP/summary.json", {"metrics": {"dice_mean": {"mean": dice}}})
            write(task.name, "FF/summary.json", {"metrics": {"dice_mean": {"mean": dice}, "sdlogj": {"mean": 0.1}}})
        return _Handle(check_code if task.name == "pre_check" else 0)

    return launch


def _dispatch(tmp_path, gpus=("0", "1"), branch_cap=72.0, total_cap=300.0, **kw):
    ctx = dispatch1.Ctx(tmp_path / "run", "STAMP")
    ctx.run.mkdir(parents=True)
    launched = {}
    d = dispatch1.Dispatcher(
        dispatch1.tasks(), ctx, list(gpus), branch_cap, total_cap, _fake(ctx, launched=launched, **kw)
    )
    for _ in range(1000):
        if not d.step():
            break
    return ctx, d, launched


def _passed(d):
    return {name for name, state in d.states.items() if state.status == "PASS"}


def test_dry_run_follows_the_minimax_policy_and_the_signal_of_a(tmp_path):
    ctx, d, launched = _dispatch(tmp_path)
    status = {n: s.status for n, s in d.states.items()}
    assert all(s in ("PASS", "SKIP") for s in status.values()), status
    for name in ("d_table", "d_minimax_bridge", "d_minimax_recursion", "h2_DZ", "h2_A_s1", "h2_P_s1", "h2_B_s1"):
        assert status[name] == "PASS", name
    for name in ("h2_ST", "mse_train", "im2grid_l4", "h1_D3"):
        assert status[name] == "PASS", name
    for name in ("d_proxy_bridge", "d_cal_proxy", "r_cal", "r_recursion"):
        assert status[name] == "SKIP", name
    bridge = launched["d_minimax_bridge"][1]
    assert "--energy_table" in bridge and "--distill_beta 0.5" in bridge and "--init_ckpt" in bridge
    assert "--known_warp_gamma 0.25" in launched["h2_A"][1] and "--known_warp_seed 1" in launched["h2_A_s1"][1]
    assert "--bank full" in launched["d_table_0"][1]  # the sub-bank stayed below T_D
    assert "--w_mse 30.0" in launched["mse_train"][1] and "--init_ckpt" not in launched["mse_train"][1]
    assert "--tto_tri_project_eps 0.001" in launched["h2_B"][1]
    assert launched["d_table"][0] == "" and launched["pre_tests"][0] == ""  # merging and the unit tests take no GPU
    verdicts = {v["comparison"]: v["status"] for v in dispatch1.verdicts(ctx, _passed(d))}
    assert verdicts["H3 T_D minimax"] == "PASS" and verdicts["H3 sub-bank"] == "NEGATIVE"
    assert verdicts["H2 A signal"] == "PASS" and verdicts["H2 P signal"] == "NEGATIVE"
    assert verdicts["D minimax"] == "PASS" and verdicts["D minimax selection"] == "PASS"
    assert verdicts["D proxy"] == "MISSING" and verdicts["R"] == "MISSING"
    assert verdicts["im2grid"] == "PASS" and verdicts["H1"] == "PASS"
    assert verdicts["VXM-MSE recipe"] == "DESCRIPTIVE"


def test_preflight_runs_every_branch_path_before_its_arms(tmp_path):
    _, _, launched = _dispatch(tmp_path)
    h2 = launched["h2_prep"][1]
    for target in ("point", "set", "structure", "deadzone"):
        assert f"--known_warp_target {target}" in h2, target
    assert "experiments.inference" in h2 and "experiments.eval_known_warp" in h2
    d = launched["d_prep"][1]
    assert all(f"--distill_teacher {kind}" in d for kind in ("random_bridge", "direct", "recursion"))
    for task in ("im2grid_prep", "h1_prep_D3", "h1_prep_E3"):
        assert (
            "--calibrate" in launched[task][1]
            and "--eval_ckpt" in launched[task][1]
            and "--limit_test 1" in launched[task][1]
        )
    assert "experiments.inference" in launched["mse_prep"][1] and "pytest" in launched["pre_tests"][1]


def test_dry_run_takes_the_proxy_policy_when_only_the_proxy_bridge_passed(tmp_path):
    _, d, launched = _dispatch(tmp_path, h3={**H3, "bridge_minimax": 0.802, "bridge_proxy": 0.81, "recursion": 0.80})
    status = {n: s.status for n, s in d.states.items()}
    assert status["d_table_0"] == "SKIP" and status["d_cal_minimax"] == "SKIP"
    assert status["d_proxy_bridge"] == status["d_proxy_random"] == "PASS"
    assert status["d_proxy_recursion"] == "SKIP" and status["r_cal"] == "SKIP"
    proxy = launched["d_proxy_bridge"][1]
    assert "--proxy_values" in proxy and "--energy_table" not in proxy and "--distill_pairs all" in proxy


def test_a_failed_parent_check_blocks_the_parent_and_spares_the_rest(tmp_path):
    _, d, _ = _dispatch(tmp_path, check_code=3)
    status = {n: s.status for n, s in d.states.items()}
    assert status["pre_check"] == "ERROR"
    for name in ("h3", "h2_prep", "h2_B", "h2_A", "d_minimax_bridge", "r_recursion"):
        assert status[name] == "BLOCKED", name
    for name in ("mse_train", "im2grid_l1", "h1_E3", "h2_stats"):
        assert status[name] == "PASS", name


def test_budget_skips_arms_whose_projection_exceeds_the_branch_cap(tmp_path):
    _, d, _ = _dispatch(tmp_path, step_sec=1.0, branch_cap=5.0)
    status = {n: s.status for n, s in d.states.items()}
    assert status["h2_B"] == status["h2_A"] == "SKIP-BUDGET"
    assert status["h2_DZ_cal"] == "SKIP" and status["h2_A_s1"] == "SKIP" and status["h2_prep"] == "PASS"


def test_budget_counts_the_reservations_of_running_tasks(tmp_path):
    arms = [dispatch1.Task(n, "H2", 1, (), lambda c: ["true"], hours=lambda c: 13.0) for n in ("b", "p", "a")]
    d = dispatch1.Dispatcher(arms, dispatch1.Ctx(tmp_path, "S"), ["0", "1", "2"], 30.0, 300.0, lambda *a: _Handle(0, 3))
    d.step()
    assert [d.states[n].status for n in ("b", "p", "a")] == ["RUNNING", "RUNNING", "SKIP-BUDGET"]
    while d.step():
        pass
    assert d.states["b"].status == d.states["p"].status == "PASS"


def test_running_cost_grows_beyond_the_initial_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch1.time, "time", lambda: 8200.0)
    d = dispatch1.Dispatcher(
        [dispatch1.Task("long", "H2", 1, (), lambda c: [])], dispatch1.Ctx(tmp_path, "S"), ["3"], 72.0, 300.0, None
    )
    d.states["long"] = dispatch1.State(status="RUNNING", start=1000.0, hours_reserved=1.0, hours_used=0.5)
    assert d.spent("H2") == pytest.approx(2.5)


def test_resume_at_a_new_head_reruns_unfinished_tasks_under_new_names(tmp_path):
    names = []
    arm = dispatch1.Task("h2_B", "H2", 1, (), lambda c: names.append(c.exp("h2_B")) or ["true"])
    done = dispatch1.Task("h2_P", "H2", 1, (), lambda c: names.append(c.exp("h2_P")) or ["true"])
    states = {
        "h2_B": dispatch1.State(status="RUNNING", start=1000.0, attempt=1, head="old"),
        "h2_P": dispatch1.State(status="PASS", attempt=1, head="old", hours_used=2.0),
    }
    (tmp_path / "h2_B").mkdir()
    (tmp_path / "h2_B" / "partial.csv").write_text("old result")
    ctx = dispatch1.Ctx(tmp_path, "S")
    d = dispatch1.Dispatcher([arm, done], ctx, ["2", "3"], 72.0, 300.0, lambda *a: _Handle(0), states, 4600.0, "new")
    assert d.states["h2_B"].status == "PENDING" and d.states["h2_B"].hours_used == pytest.approx(1.0)
    while d.step():
        pass
    assert names == ["D1_S_h2_B_r2"] and d.states["h2_B"].attempt == 2
    assert (d.states["h2_B"].head, d.states["h2_P"].head) == ("new", "old")
    assert not (tmp_path / "h2_B" / "partial.csv").exists()
    assert [p.read_text() for p in (tmp_path / "attempts").glob("h2_B_1_*/partial.csv")] == ["old result"]


def test_dispatch_refuses_modified_tracked_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch1.subprocess, "run", lambda *a, **kw: type("Result", (), {"returncode": 1})())
    with pytest.raises(RuntimeError, match="tracked files differ"):
        dispatch1.run_dispatch(tmp_path, False)


def test_p_only_signal_keeps_its_seed_comparison(tmp_path, monkeypatch):
    monkeypatch.setitem(REP, "h2_A", 0.801)
    monkeypatch.setitem(REP, "h2_P", 0.810)
    ctx, d, _ = _dispatch(tmp_path)
    assert d.states["h2_A_s1"].status == "SKIP" and d.states["h2_P_s1"].status == "PASS"
    rows = {row["comparison"]: row["status"] for row in dispatch1.verdicts(ctx, _passed(d))}
    assert rows["H2 P seed 1"] == "DESCRIPTIVE" and rows["H2 A seed 1"] == "MISSING"


def test_verdicts_need_passed_tasks_and_finite_results(tmp_path):
    ctx, _, _ = _dispatch(tmp_path)
    assert all(row["status"] == "MISSING" for row in dispatch1.verdicts(ctx, completed=set()))
    (tmp_path / "bad" / "REP").mkdir(parents=True)
    (tmp_path / "bad" / "REP" / "summary.json").write_text('{"metrics": {"dice_mean": {"mean": NaN}}}')
    with pytest.raises(ValueError, match="non-finite"):
        dispatch1.Ctx(tmp_path, "S").rep("bad")
    with pytest.raises(ValueError, match="distinct GPU"):
        dispatch1.Dispatcher([], ctx, ["2", "2"], 72.0, 300.0, None)


def test_finish_packs_results_without_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx, d, _ = _dispatch(tmp_path)
    (ctx.run / "h2_B" / "model.pth").write_text("weights")
    assert dispatch1.finish(ctx, d) == 0
    with zipfile.ZipFile(Path("results/exports/DISPATCH1_STAMP.zip")) as z:
        names = z.namelist()
    assert "DISPATCH1_STAMP/verdicts.json" in names and "DISPATCH1_STAMP/h2_B/REP/summary.json" in names
    assert not any(n.endswith(".pth") for n in names)


def test_prepare_checks_the_p1_control_before_choosing_a_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dispatch1, "P1", {**dispatch1.P1, "ckpt": "p1.pth"})
    with pytest.raises(FileNotFoundError):
        dispatch1.prepare("out")
    Path("p1.pth").write_text("")
    monkeypatch.setattr(dispatch1, "P1_SUMMARY", "p1_{}.json")
    summary = {"n_cases": 19, "metrics": {"dice_mean": {"mean": 0.8}, "sdlogj": {"mean": 0.1}}}
    for mode in ("FF", "REP"):
        Path(f"p1_{mode}.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="no usable parent"):  # neither parent's files are here
        dispatch1.prepare("out")
    assert json.loads(Path("out/control.json").read_text())["REP"]["dice_mean"] == 0.8
    summary["n_cases"] = 18
    Path("p1_REP.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="19 test pairs"):
        dispatch1.prepare("out")
