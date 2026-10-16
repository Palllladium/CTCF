"""Dispatch 1: the pre-registered arms of H3, H2, D, R, im2grid, VXM-MSE and H1, scheduled over the given GPUs.

A task starts when its dependencies passed, its pre-registered condition on their results holds and its projected
GPU-hours fit the branch and dispatch caps after the reservations of running tasks; higher priority first, one task per
GPU. The preflight comes first: the unit tests, the parent's decision and its check against the historical FF table,
then for every branch a preparation task that profiles it, calibrates its weights and runs each of its code paths once
on real data (short training, short evaluation), so a broken path fails early and blocks only its branch. Statuses:
PASS, ERROR, SKIP (condition false or a dependency skipped), SKIP-BUDGET, BLOCKED (a dependency failed). A resume, also
after a fix at a new HEAD, reruns ERROR and BLOCKED tasks under new experiment names and keeps the rest; status.json
records the HEAD of every attempt. The end writes verdicts.json, dice_groups.txt and a ZIP without weights.

    GPUS="0 1 2 3 4 5 6 7" nohup bash tools/runners/train/dispatch1.sh > dispatch1.out 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from experiments.core.field_scoring import repair_flags

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("PY", sys.executable)
NODE = "--paths 3 --gpu 0"  # the node's path profile (CTCF_DATA_DIR) and the one GPU a task sees
PARENTS = {
    "mamba_nosvf": {
        "ckpt": "results/P10_LONGRUN_MAMBA_NOSVF_OASIS/ckpt/best.pth",
        "config": "CTCF-CascadeA-Mamba",
        "l3_svf": 0,
        "historical_ff": "results/SEDM/inference/P10_LONGRUN_MAMBA_NOSVF_OASIS/per_case.csv",
    },
    "p1_control": {
        "ckpt": "results/PILOT_P1_CONTROL/ckpt/last.pth",
        "config": "CTCF-CascadeA-VM-Unified",
        "l3_svf": 1,
        "historical_ff": "results/pilots/P1/PILOT_P1_CONTROL__FF/per_case.csv",
    },
}
P1 = PARENTS["p1_control"]
P1_SUMMARY = "results/pilots/P1/PILOT_P1_CONTROL__{}/summary.json"
RECIPE = "--ds OASIS --use_tb 0 --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0 --w_jac 0.005"  # the P1 and P10 losses
EPOCHS, UPDATES = 100, 39_400  # 100 epochs of the 394 training pairs: the P1 control's updates
PROFILE_UPDATES, SHORT_UPDATES = 60, 10
TABLE_SHARDS = 4
TESTS = "tools/analysis/tests/dispatch1"
IM2GRID = "experiments.dispatch1.train_im2grid"
SIGNAL, SMALL_SIGNAL = 0.005, 0.003  # pre-registered REP Dice35 differences
OVERHEAD, EVAL_HOURS, ELIGIBLE_COST = 1.15, 0.4, 1.2  # approximate cost margins of the projections
POLL_SECONDS = 30
READ_PROFILE = (
    "import csv, json, sys; r = list(csv.DictReader(open(sys.argv[1])))[-1]; json.dump({'step_sec': "
    "float(r['perf_iter_time_ms']) / 1000, 'peak_gib': r['perf_peak_gpu_mem_gib']}, open(sys.argv[2], 'w'))"
)


def finite(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite result: {value}")
    return value


class Ctx:
    """The run folder, the attempt of the task being prepared, and the results that conditions read."""

    def __init__(self, run: Path, stamp: str):
        self.run, self.stamp, self.attempt = run, stamp, 1

    def path(self, *parts) -> str:
        return self.run.joinpath(*parts).as_posix()

    def exp(self, name: str) -> str:
        return f"D1_{self.stamp}_{name}" + (f"_r{self.attempt}" if self.attempt > 1 else "")

    def json(self, *parts) -> dict:
        return json.loads(self.run.joinpath(*parts).read_text(encoding="utf-8"))

    @property
    def parent(self) -> dict:
        return self.json("pre_parent", "parent.json")

    def metric(self, task: str, mode: str, key: str = "dice_mean") -> float:
        return finite(self.json(task, mode, "summary.json")["metrics"][key]["mean"])

    def rep(self, task: str) -> float:
        return self.metric(task, "REP")

    def h3(self, arm: str) -> float:
        return finite(self.json("h3", "arms.json")["arms"][arm]["REP"])

    def step_sec(self, task: str, name: str) -> float:
        seconds = finite(self.json(task, f"{name}.json")["step_sec"])
        if seconds <= 0:
            raise ValueError(f"non-positive step time in {task}/{name}.json")
        return seconds


def gain(ctx: Ctx, arm: str) -> float:
    return ctx.h3(arm) - ctx.h3("direct")


def d_policy(ctx: Ctx) -> str | None:
    """The minimax bridge teaches when it passed T_D; the proxy bridge teaches all pairs only when it alone passed."""
    if gain(ctx, "bridge_minimax") >= SIGNAL:
        return "minimax"
    if gain(ctx, "bridge_proxy") >= SIGNAL:
        return "proxy"
    return None


def t_r(ctx: Ctx) -> bool:
    return gain(ctx, "recursion") >= SIGNAL


def h3_note(ctx: Ctx) -> str:
    arms = ("bridge_minimax", "bridge_minimax64", "bridge_proxy", "recursion")
    return f"policy {d_policy(ctx)}, T_R {t_r(ctx)} ({', '.join(f'{a} {gain(ctx, a):+.4f}' for a in arms)})"


def when_policy(policy: str, recursion: bool = False):
    def condition(ctx: Ctx):
        return d_policy(ctx) == policy and (t_r(ctx) or not recursion), h3_note(ctx)

    return condition


def when_r_only(ctx: Ctx):
    return t_r(ctx) and d_policy(ctx) is None, h3_note(ctx)


def a_signal(ctx: Ctx) -> bool:
    a = ctx.rep("h2_A")
    return a - ctx.rep("h2_B") >= SIGNAL or a - ctx.rep("h2_P") >= SMALL_SIGNAL


def p_signal(ctx: Ctx) -> bool:
    return ctx.rep("h2_P") - ctx.rep("h2_B") >= SIGNAL


def h2_note(ctx: Ctx) -> str:
    a, b, p = (ctx.rep(f"h2_{k}") for k in "ABP")
    return f"A-B {a - b:+.4f}, A-P {a - p:+.4f}, P-B {p - b:+.4f}"


def when_a(ctx: Ctx):
    return a_signal(ctx), h2_note(ctx)


def when_a_or_p(ctx: Ctx):
    return a_signal(ctx) or p_signal(ctx), h2_note(ctx)


def py(args: str) -> str:
    return f"{shlex.quote(PY)} -m {args}"


def model_flags(parent: dict, ckpt: str = "") -> str:
    return f"--ckpt {ckpt or parent['ckpt']} --ctcf_config {parent['config']} --ctcf_l3_svf {parent['l3_svf']}"


def train(exp: str, extra: str, parent: dict, seed=0, init=True, updates=0, save=1) -> str:
    """experiments.train_CTCF with the parent's losses, fine-tuned from the parent with every level on or from scratch;
    ``updates`` makes it a one-epoch run of that many steps with every level on."""
    args = f"--config {parent['config']} --l3_svf {parent['l3_svf']} --seed {seed} {RECIPE} {extra}"
    if init or updates:
        args += " --finetune 1" + (f" --init_ckpt {parent['ckpt']}" if init else "")
    length = f"--max_epoch 1 --max_train_iters {updates} --max_val_batches 1" if updates else f"--max_epoch {EPOCHS}"
    return py(f"experiments.train_CTCF {args} {length} --save_ckpt {save} --exp {exp} {NODE}")


def calibrate(ctx: Ctx, out: str, extra: str, parent: dict) -> str:
    """The auxiliary weight at the starting weights into the run's file ``out`` (train_CTCF --aux_calibrate)."""
    name = out.removesuffix(".json").replace("/", "_")
    return train(ctx.exp(name), f"{extra} --aux_calibrate {ctx.path(out)}", parent, save=0)


def profile(ctx: Ctx, task: str, name: str, extra: str, parent: dict, init=True, save=0) -> list[str]:
    """PROFILE_UPDATES steps; their mean step time into <task>/<name>.json."""
    exp = ctx.exp(f"PROFILE_{name}")
    read = shlex.join([PY, "-c", READ_PROFILE, f"logs/{exp}/metrics.csv", ctx.path(task, f"{name}.json")])
    return [train(exp, extra, parent, init=init, updates=PROFILE_UPDATES, save=save), read]


def short(ctx: Ctx, name: str, extra: str) -> str:
    """SHORT_UPDATES steps that only exercise a code path."""
    return train(ctx.exp(f"SHORT_{name}"), extra, ctx.parent, updates=SHORT_UPDATES, save=0)


def inference(parent: dict, ckpt: str, out: str) -> list[str]:
    """FF and REP of a checkpoint by experiments.inference, the legacy sampler of the historical tables."""
    base = f"experiments.inference --model ctcf --strict_ckpt 1 --ds OASIS --tto_mode none {NODE}"
    base += f" {model_flags(parent, ckpt)}"
    return [py(f"{base} --out_dir {out}/FF"), py(f"{base} {repair_flags()} --out_dir {out}/REP")]


def keep_artifacts(ctx: Ctx, name: str, exp: str) -> str:
    """Copy the run's logs and compact results into the task folder however the task ends."""
    out = ctx.path(name)
    copy = (
        f"mkdir -p {out}/logs; if [ -d logs/{exp} ]; then cp -r logs/{exp} {out}/logs/; fi; "
        f'for f in results/{exp}/*.csv results/{exp}/*.json; do if [ -f "$f" ]; then cp "$f" {out}/; fi; done'
    )
    return f"trap {shlex.quote(copy)} EXIT"


def known_warp(ctx: Ctx, target: str, seed=0, weight_file: str = "", gamma=None) -> str:
    out = f"--known_warp_every 4 --known_warp_target {target} --known_warp_seed {seed}"
    out += f" --known_warp_stats {ctx.path('h2_stats', 'stats.json')}"
    gamma = ctx.json(weight_file)["known_warp_gamma"] if weight_file else gamma
    return out + (f" --known_warp_gamma {gamma}" if gamma is not None else "")


def distill(ctx: Ctx, kind: str, pairs: str, weight_file: str = "", beta=None) -> str:
    out = f"--distill_teacher {kind} --distill_pairs {pairs} --distill_seed 0"
    if pairs == "eligible" or kind == "bridge_minimax":
        out += f" --energy_table {ctx.path('d_table', 'table.npz')}"
    if kind == "bridge_proxy":
        out += f" --proxy_values {ctx.path('h3', 'proxy_values.json')}"
    beta = ctx.json(weight_file)["distill_beta"] if weight_file else beta
    return out + (f" --distill_beta {beta}" if beta is not None else "")


def partial_distill(kind: str, pairs: str, weight_file: str):
    return lambda ctx: distill(ctx, kind, pairs, weight_file=weight_file)


def ctcf_arm(ctx: Ctx, name: str, extra: str = "", seed=0, parent=None, init=True, epe=False) -> list[str]:
    """Train, then FF and REP of last.pth, and the endpoint error on known-warp pairs when ``epe``."""
    parent = parent or ctx.parent
    exp = ctx.exp(name)
    ckpt = f"results/{exp}/ckpt/last.pth"
    commands = [keep_artifacts(ctx, name, exp), train(exp, extra, parent, seed, init)]
    commands += inference(parent, ckpt, ctx.path(name))
    if epe:
        out = ctx.path(name, "epe.json")
        commands.append(py(f"experiments.dispatch1.eval_known_warp {model_flags(parent, ckpt)} --out {out} {NODE}"))
    return commands


def h2_prep(ctx: Ctx) -> list[str]:
    """Profiles of B, P and A; gamma of P, A and ST; short runs of the structure and dead-zone targets; FF/REP of the
    B profile's checkpoint; the parent's endpoint error, the reference of the arms'."""
    parent, ckpt = ctx.parent, f"results/{ctx.exp('PROFILE_h2_B')}/ckpt/last.pth"
    commands = [*profile(ctx, "h2_prep", "h2_B", "", parent, save=1)]
    for name, target in (("h2_P", "point"), ("h2_A", "set")):
        commands += profile(ctx, "h2_prep", name, known_warp(ctx, target, gamma=1), parent)
    for target in ("point", "set", "structure"):
        commands.append(calibrate(ctx, f"h2_prep/gamma_{target}.json", known_warp(ctx, target), parent))
    commands += [short(ctx, f"h2_{target}", known_warp(ctx, target, gamma=1)) for target in ("structure", "deadzone")]
    commands += inference(parent, ckpt, ctx.path("h2_prep", "inference"))
    epe = f"experiments.dispatch1.eval_known_warp {model_flags(parent)} --out {ctx.path('h2_prep', 'epe_parent.json')}"
    return [*commands, py(f"{epe} {NODE}")]


def d_prep(ctx: Ctx) -> list[str]:
    """Profile of the random-bridge teacher on all pairs; short runs of the direct and second-pass teachers."""
    commands = profile(ctx, "d_prep", "d_random", distill(ctx, "random_bridge", "all", beta=1), ctx.parent)
    return [*commands, *(short(ctx, f"d_{k}", distill(ctx, k, "all", beta=1)) for k in ("direct", "recursion"))]


def mse_prep(ctx: Ctx) -> list[str]:
    """The MSE weight at the P1 control; a profile from scratch; FF/REP of its checkpoint."""
    ckpt = f"results/{ctx.exp('PROFILE_mse')}/ckpt/last.pth"
    commands = [calibrate(ctx, "mse_prep/weight.json", "--sim mse", P1)]
    commands += profile(ctx, "mse_prep", "mse", "--sim mse", P1, init=False, save=1)
    return [*commands, *inference(P1, ckpt, ctx.path("mse_prep", "inference"))]


def matcher_prep(ctx: Ctx, task: str, script: str) -> list[str]:
    """lambda* and the step time of its warm-up; two updates with lambda* and the evaluation of one test pair."""
    lam, exp = ctx.path(task, "lambda.json"), ctx.exp(f"SHORT_{task}")
    evaluation = f"--eval_ckpt results/{exp}/ckpt/last.pth --limit_test 1 --out {ctx.path(task, 'short')}"
    return [
        py(f"{script} --calibrate {lam} {NODE}"),
        py(f"{script} --lambda_file {lam} --exp {exp} --iters 2 {NODE}"),
        py(f"{script} {evaluation} {NODE}"),
    ]


def matcher_arm(ctx: Ctx, name: str, script: str, prep: str, scale: str = "1.0") -> list[str]:
    exp, lam = ctx.exp(name), ctx.path(prep, "lambda.json")
    training = f"--lambda_file {lam} --lambda_scale {scale} --exp {exp} --iters {UPDATES} --seed 0"
    evaluation = f"--eval_ckpt results/{exp}/ckpt/last.pth --out {ctx.path(name)}"
    return [keep_artifacts(ctx, name, exp), py(f"{script} {training} {NODE}"), py(f"{script} {evaluation} {NODE}")]


def energy_shard(ctx: Ctx, shard: int) -> list[str]:
    rows = f"--bank {table_bank(ctx)} --shard {shard} --shards {TABLE_SHARDS}"
    out = ctx.path(f"d_table_{shard}", "part.npz")
    return [py(f"experiments.dispatch1.energy_table {model_flags(ctx.parent)} {rows} --out {out} {NODE}")]


def energy_merge(ctx: Ctx) -> list[str]:
    parts = " ".join(ctx.path(f"d_table_{k}", "part.npz") for k in range(TABLE_SHARDS))
    return [py(f"experiments.dispatch1.energy_table --merge {ctx.path('d_table', 'table.npz')} {parts}")]


def bridges(ctx: Ctx, task: str, check_only=False) -> list[str]:
    args = f"{model_flags(ctx.parent)} --check_ff {ctx.parent['historical_ff']} --out {ctx.path(task)}"
    return [py(f"experiments.dispatch1.eval_bridge {args}{' --check_only' if check_only else ''} {NODE}")]


def train_hours(task: str, name: str, factor: float = 1.0):
    return lambda ctx: UPDATES * ctx.step_sec(task, name) * factor * OVERHEAD / 3600 + EVAL_HOURS


def h3_hours(ctx: Ctx) -> float:
    check = ctx.json("pre_check", "check.json")
    pairs, bank = check["test_pairs"], check["bank_subjects"]
    passes = pairs * (2 + 2 * bank + 2 * 5)  # direct and second pass, the bank, the chosen bridges
    scoring = pairs * (7 * check["score_seconds"] + bank * check["oracle_score_seconds"])
    return (passes * check["seconds_per_pass"] * 1.3 + scoring) / 3600 + 0.2


def table_bank(ctx: Ctx) -> str:
    return "sub" if gain(ctx, "bridge_minimax64") >= SIGNAL else "full"


def table_hours(ctx: Ctx) -> float:
    check = ctx.json("pre_check", "check.json")
    bank = check["sub_bank_subjects"] if table_bank(ctx) == "sub" else check["bank_subjects"]
    return check["bank_subjects"] * bank * 2 / TABLE_SHARDS * check["seconds_per_pass"] * OVERHEAD / 3600 + 0.1


@dataclass
class Task:
    name: str
    branch: str
    priority: int
    deps: tuple[str, ...]
    commands: Callable[[Ctx], list[str]]
    hours: Callable[[Ctx], float] = lambda ctx: 0.3
    condition: Callable[[Ctx], tuple[bool, str]] | None = None
    gpu: bool = True


def tasks() -> list[Task]:
    """The dispatch; branch caps apply to H2, H3_MSE, D, R, IM and H1, and the preflight counts toward the total."""
    t: list[Task] = []

    def add(name, branch, priority, deps, commands, hours=None, condition=None, gpu=True):
        t.append(Task(name, branch, priority, tuple(deps), commands, hours or (lambda c: 0.3), condition, gpu))

    def arm(name, branch, priority, deps, extra, hours, condition=None, **options):
        add(name, branch, priority, deps, lambda c: ctcf_arm(c, name, extra(c), **options), hours, condition)

    def cal(name, branch, priority, deps, extra, condition=None):
        out = f"{name}/weight.json"
        add(name, branch, priority, deps, lambda c: [calibrate(c, out, extra(c), c.parent)], condition=condition)

    add("pre_fetch", "pre", 0, [], lambda c: [py("models.Im2grid.fetch")], gpu=False)
    add("pre_tests", "pre", 0, ["pre_fetch"], lambda c: [py(f"pytest -q -p no:cacheprovider {TESTS}")], gpu=False)
    add(
        "pre_parent",
        "pre",
        0,
        ["pre_tests"],
        lambda c: [py(f"experiments.dispatch1.run --prepare {c.path('pre_parent')}")],
    )
    add("pre_check", "pre", 0, ["pre_parent"], lambda c: bridges(c, "pre_check", check_only=True))

    add(
        "h2_stats",
        "H2",
        1,
        ["pre_tests"],
        lambda c: [py(f"experiments.dispatch1.eval_known_warp --stats {c.path('h2_stats', 'stats.json')} {NODE}")],
    )
    add("h2_prep", "H2", 2, ["pre_check", "h2_stats"], h2_prep)
    add("d_prep", "D", 2, ["pre_check"], d_prep)
    add("mse_prep", "H3_MSE", 2, ["pre_parent"], mse_prep)
    add("im2grid_prep", "IM", 2, ["pre_parent"], lambda c: matcher_prep(c, "im2grid_prep", IM2GRID))
    for mode in ("D3", "E3"):
        script, prep = f"experiments.dispatch1.train_multimatch --mode {mode}", f"h1_prep_{mode}"
        add(prep, "H1", 2, ["pre_tests"], lambda c, s=script, p=prep: matcher_prep(c, p, s))
        add(
            f"h1_{mode}",
            "H1",
            9,
            [prep],
            lambda c, s=script, p=prep, m=mode: matcher_arm(c, f"h1_{m}", s, p),
            train_hours(prep, "lambda"),
        )
    for name, scale in (("im2grid_l1", "1.0"), ("im2grid_l4", "0.25")):
        add(
            name,
            "IM",
            4,
            ["im2grid_prep"],
            lambda c, n=name, s=scale: matcher_arm(c, n, IM2GRID, "im2grid_prep", s),
            train_hours("im2grid_prep", "lambda"),
        )
    arm(
        "mse_train",
        "H3_MSE",
        8,
        ["mse_prep"],
        lambda c: f"--sim mse --w_mse {c.json('mse_prep', 'weight.json')['w_mse']}",
        train_hours("mse_prep", "mse"),
        parent=P1,
        init=False,
    )
    add("h3", "H3_MSE", 4, ["pre_check"], lambda c: bridges(c, "h3"), h3_hours)

    for name, target, priority, profiled in (
        ("h2_P", "point", 3, "h2_P"),
        ("h2_A", "set", 3, "h2_A"),
        ("h2_ST", "structure", 8, "h2_A"),
    ):
        arm(
            name,
            "H2",
            priority,
            ["h2_prep"],
            lambda c, t=target: known_warp(c, t, weight_file=f"h2_prep/gamma_{t}.json"),
            train_hours("h2_prep", profiled),
            epe=True,
        )
    arm("h2_B", "H2", 3, ["h2_prep"], lambda c: "", train_hours("h2_prep", "h2_B"))
    after_h2 = ["h2_A", "h2_P", "h2_B"]
    cal("h2_DZ_cal", "H2", 6, after_h2, lambda c: known_warp(c, "deadzone"), when_a)
    arm(
        "h2_DZ",
        "H2",
        6,
        ["h2_DZ_cal"],
        lambda c: known_warp(c, "deadzone", weight_file="h2_DZ_cal/weight.json"),
        train_hours("h2_prep", "h2_A"),
        epe=True,
    )
    for name, target, profiled, condition in (
        ("h2_A_s1", "set", "h2_A", when_a),
        ("h2_P_s1", "point", "h2_P", when_a_or_p),
    ):
        arm(
            name,
            "H2",
            7,
            after_h2,
            lambda c, t=target: known_warp(c, t, seed=1, weight_file=f"h2_prep/gamma_{t}.json"),
            train_hours("h2_prep", profiled),
            condition,
            seed=1,
            epe=True,
        )
    arm("h2_B_s1", "H2", 7, after_h2, lambda c: "", train_hours("h2_prep", "h2_B"), when_a_or_p, seed=1)

    shards = [f"d_table_{k}" for k in range(TABLE_SHARDS)]
    for k, shard in enumerate(shards):
        add(shard, "D", 4, ["h3", "d_prep"], lambda c, k=k: energy_shard(c, k), table_hours, when_policy("minimax"))
    add("d_table", "D", 4, shards, energy_merge, condition=when_policy("minimax"), gpu=False)
    for policy, pairs, teacher, deps in (
        ("minimax", "eligible", "bridge_minimax", ["d_table"]),
        ("proxy", "all", "bridge_proxy", ["h3", "d_prep"]),
    ):
        cal(f"d_cal_{policy}", "D", 4, deps, lambda c, k=teacher, p=pairs: distill(c, k, p), when_policy(policy))
        hours = train_hours("d_prep", "d_random", ELIGIBLE_COST if pairs == "eligible" else 1.0)
        for suffix, kind, recursion in (
            ("bridge", teacher, False),
            ("direct", "direct", False),
            ("random", "random_bridge", False),
            ("recursion", "recursion", True),
        ):
            extra = partial_distill(kind, pairs, f"d_cal_{policy}/weight.json")
            arm(
                f"d_{policy}_{suffix}",
                "D",
                6 if recursion else 5,
                [f"d_cal_{policy}"],
                extra,
                hours,
                when_policy(policy, recursion),
            )
    cal("r_cal", "R", 5, ["h3", "d_prep"], lambda c: distill(c, "recursion", "all"), when_r_only)
    for name, kind in (("r_recursion", "recursion"), ("r_direct", "direct")):
        arm(
            name,
            "R",
            5,
            ["r_cal"],
            partial_distill(kind, "all", "r_cal/weight.json"),
            train_hours("d_prep", "d_random"),
            when_r_only,
        )
    return t


def verdicts(ctx: Ctx, completed: set[str]) -> list[dict]:
    """The pre-registered comparisons on the tasks that passed: MISSING without an input, DESCRIPTIVE without a rule."""

    def rep(task: str) -> float:
        if task not in completed:
            raise ValueError(f"{task} did not pass")
        return ctx.rep(task)

    def diff(a: str, b: str):
        return lambda: rep(a) - rep(b)

    def h3(a: str, b: str = "direct"):
        def value() -> float:
            if "h3" not in completed:
                raise ValueError("h3 did not pass")
            return ctx.h3(a) - ctx.h3(b)

        return value

    def control(mode: str, key: str) -> float:
        return finite(ctx.json("pre_parent", "control.json")[mode][key])

    rules = [  # name, [(label, value, threshold or None)], combine
        ("H3 T_D minimax", [("bridge_minimax-direct", h3("bridge_minimax"), SIGNAL)], all),
        ("H3 T_D proxy", [("bridge_proxy-direct", h3("bridge_proxy"), SIGNAL)], all),
        ("H3 sub-bank", [("bridge_minimax64-direct", h3("bridge_minimax64"), SIGNAL)], all),
        ("H3 T_R", [("recursion-direct", h3("recursion"), SIGNAL)], all),
        ("H3 proxy vs recursion", [("bridge_proxy-recursion", h3("bridge_proxy", "recursion"), None)], all),
        ("H2 A signal", [("A-B", diff("h2_A", "h2_B"), SIGNAL), ("A-P", diff("h2_A", "h2_P"), SMALL_SIGNAL)], any),
        ("H2 P signal", [("P-B", diff("h2_P", "h2_B"), SIGNAL)], all),
        ("H2 ST", [("ST-B", diff("h2_ST", "h2_B"), None), ("ST-A", diff("h2_ST", "h2_A"), None)], all),
        ("H2 DZ", [("DZ-A", diff("h2_DZ", "h2_A"), None), ("DZ-B", diff("h2_DZ", "h2_B"), None)], all),
        ("H2 A seed 1", [("A_s1-B_s1", diff("h2_A_s1", "h2_B_s1"), None)], all),
        ("H2 P seed 1", [("P_s1-B_s1", diff("h2_P_s1", "h2_B_s1"), None)], all),
        (
            "R",
            [("R-B", diff("r_recursion", "h2_B"), SIGNAL), ("R-R0", diff("r_recursion", "r_direct"), SMALL_SIGNAL)],
            all,
        ),
        ("H1", [("D3-E3", diff("h1_D3", "h1_E3"), SIGNAL)], all),
        ("VXM-MSE recipe", [("MSE-P1 control", lambda: rep("mse_train") - control("REP", "dice_mean"), None)], all),
    ]
    for p in ("minimax", "proxy"):
        bridge, direct = f"d_{p}_bridge", f"d_{p}_direct"
        rules += [
            (
                f"D {p}",
                [
                    ("D_bridge-B", diff(bridge, "h2_B"), SIGNAL),
                    ("D_bridge-D_direct", diff(bridge, direct), SMALL_SIGNAL),
                ],
                all,
            ),
            (f"D {p} selection", [("D_bridge-D_random", diff(bridge, f"d_{p}_random"), SMALL_SIGNAL)], all),
            (f"D {p} recursion", [("D_recursion-D_bridge", diff(f"d_{p}_recursion", bridge), None)], all),
        ]
    results = []
    for name, parts, combine in rules:
        try:
            values = [(label, value(), threshold) for label, value, threshold in parts]
        except (OSError, KeyError, ValueError, TypeError) as error:
            results.append({"comparison": name, "status": "MISSING", "note": repr(error)})
            continue
        ruled = [v >= th for _, v, th in values if th is not None]
        status = "DESCRIPTIVE" if not ruled else "PASS" if combine(ruled) else "NEGATIVE"
        thresholds = {label: th for label, _, th in values if th is not None}
        results.append(
            {
                "comparison": name,
                "status": status,
                "values": {label: round(v, 5) for label, v, _ in values},
                "thresholds": thresholds,
            }
        )
    return [*results, im2grid_verdict(ctx, completed, control)]


def im2grid_verdict(ctx: Ctx, completed: set[str], control) -> dict:
    """Signal: the best lambda within 1.25x the control's FF SDlogJ reaches control REP + 0.015. Failure: both lambdas
    within that smoothness stay below control REP + 0.005. Otherwise undecided."""
    arms = ("im2grid_l1", "im2grid_l4")
    try:
        if not set(arms) <= completed:
            raise ValueError("an im2grid arm did not pass")
        reference, bound = control("REP", "dice_mean"), 1.25 * control("FF", "sdlogj")
        values = {a: {"REP": ctx.rep(a), "sdlogj_FF": ctx.metric(a, "FF", "sdlogj")} for a in arms}
    except (OSError, KeyError, ValueError, TypeError) as error:
        return {"comparison": "im2grid", "status": "MISSING", "note": repr(error)}
    smooth = [v["REP"] for v in values.values() if v["sdlogj_FF"] <= bound]
    if smooth and max(smooth) >= reference + 0.015:
        status = "PASS"
    elif len(smooth) == len(arms) and max(smooth) < reference + 0.005:
        status = "NEGATIVE"
    else:
        status = "UNDECIDED"
    return {
        "comparison": "im2grid",
        "status": status,
        "control_rep": reference,
        "sdlogj_bound": bound,
        "values": values,
    }


@dataclass
class State:
    status: str = "PENDING"
    gpu: str = ""
    start: float = 0.0
    attempt: int = 0
    head: str = ""
    hours_reserved: float = 0.0
    hours_used: float = 0.0
    note: str = ""


TERMINAL = ("PASS", "SKIP", "SKIP-BUDGET")


@dataclass
class Dispatcher:
    tasks: list[Task]
    ctx: Ctx
    gpus: list[str]
    branch_cap: float
    total_cap: float
    launch: Callable[[Task, str, list[str], Path], object]
    states: dict[str, State] = field(default_factory=dict)
    saved_at: float = 0.0
    head: str = ""
    running: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.gpus or len(set(self.gpus)) != len(self.gpus):
            raise ValueError("GPUS must list distinct GPU ids")
        if any(not math.isfinite(cap) or cap < 0 for cap in (self.branch_cap, self.total_cap)):
            raise ValueError("budget caps must be finite and non-negative")
        uses_gpu = {t.name: t.gpu for t in self.tasks}
        for task in self.tasks:
            self.states.setdefault(task.name, State())
        for name, state in self.states.items():
            if state.status in (*TERMINAL, "PENDING"):
                continue
            if state.status == "RUNNING":  # interrupted: its hours up to the last saved status
                state.hours_used += max(self.saved_at - state.start, 0.0) / 3600 * uses_gpu.get(name, True)
            state.status, state.gpu, state.hours_reserved, state.note = "PENDING", "", 0.0, "resumed"

    def log(self, line: str) -> None:
        print(f"[dispatch1] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}", flush=True)
        with open(self.ctx.path("status.txt"), "a", encoding="utf-8") as h:
            h.write(line + "\n")

    def save(self) -> None:
        temporary = Path(self.ctx.path("status.json.tmp"))
        temporary.write_text(json.dumps({n: vars(s) for n, s in self.states.items()}, indent=1), encoding="utf-8")
        temporary.replace(self.ctx.path("status.json"))

    def spent(self, branch: str | None) -> float:
        """GPU-hours used plus, for running tasks, the larger of their reservation and their elapsed time."""
        now, total = time.time(), 0.0
        for task in self.tasks:
            if branch is None or task.branch == branch:
                state = self.states[task.name]
                elapsed = (now - state.start) / 3600 * task.gpu if state.status == "RUNNING" else 0.0
                total += state.hours_used + max(state.hours_reserved, elapsed)
        return total

    def _finish(self, task: Task, code: int) -> None:
        state = self.states[task.name]
        state.hours_used += (time.time() - state.start) / 3600 * task.gpu
        state.hours_reserved = 0.0
        state.status = "PASS" if code == 0 else "ERROR"
        state.note = f"exit code {code}" if code else ""
        self.running.pop(task.name)
        self.log(f"{state.status} {task.name} ({state.hours_used:.2f} GPU-h{', ' + state.note if code else ''})")

    def _settle(self, task: Task, state: State, status: str, note: str) -> None:
        state.status, state.note = status, note
        self.log(f"{status} {task.name} ({note})")

    def _ready(self, task: Task, state: State) -> bool:
        """Settles a task its dependencies or condition decide; True when it may start."""
        deps = [self.states[d].status for d in task.deps]
        if any(s in ("ERROR", "BLOCKED") for s in deps):
            self._settle(task, state, "BLOCKED", "a dependency failed")
            return False
        if any(s in ("SKIP", "SKIP-BUDGET") for s in deps):
            self._settle(task, state, "SKIP", "a dependency was skipped")
            return False
        if any(s != "PASS" for s in deps):
            return False
        if task.condition is None:
            return True
        try:
            ok, state.note = task.condition(self.ctx)
        except Exception as error:  # a result the condition reads is missing or malformed
            self._settle(task, state, "ERROR", f"condition failed: {error!r}")
            return False
        if not ok:
            self._settle(task, state, "SKIP", state.note)
        return ok

    def _start(self, task: Task, state: State, gpu: str) -> bool:
        try:
            hours = finite(task.hours(self.ctx)) * task.gpu
            self.ctx.attempt = state.attempt + 1
            commands = task.commands(self.ctx)
        except Exception as error:  # a result the task reads is missing or malformed
            self._settle(task, state, "ERROR", f"could not prepare: {error!r}")
            return False
        branch_left = math.inf if task.branch == "pre" else self.branch_cap - self.spent(task.branch)
        total_left = self.total_cap - self.spent(None)
        if hours > min(branch_left, total_left):
            note = f"projected {hours:.1f} GPU-h; left {branch_left:.1f} in {task.branch}, {total_left:.1f} in all"
            self._settle(task, state, "SKIP-BUDGET", note)
            return False
        folder = self.ctx.run / task.name
        try:
            if state.attempt and folder.exists():  # keep the failed attempt's outputs aside
                (self.ctx.run / "attempts").mkdir(exist_ok=True)
                folder.rename(self.ctx.run / "attempts" / f"{task.name}_{state.attempt}_{time.time_ns()}")
            folder.mkdir(parents=True)
            self.running[task.name] = self.launch(task, gpu, commands, folder / "log.txt")
        except OSError as error:
            self._settle(task, state, "ERROR", f"could not prepare the output or launch: {error}")
            return False
        state.status, state.gpu, state.start, state.hours_reserved = "RUNNING", gpu, time.time(), hours
        state.attempt, state.head = state.attempt + 1, self.head
        note = f" ({state.note})" if state.note and state.note != "resumed" else ""
        self.log(f"START {task.name} on GPU {gpu or '-'}, attempt {state.attempt}, {hours:.2f} GPU-h{note}")
        return True

    def step(self) -> bool:
        """One scheduling pass; False when nothing runs and nothing started."""
        by_name = {t.name: t for t in self.tasks}
        for name, handle in list(self.running.items()):
            code = handle.poll()
            if code is not None:
                self._finish(by_name[name], code)
        started = False
        for task in sorted(self.tasks, key=lambda t: t.priority):
            state = self.states[task.name]
            if state.status != "PENDING" or not self._ready(task, state):
                continue
            free = [g for g in self.gpus if g not in {self.states[n].gpu for n in self.running}]
            if task.gpu and not free:
                continue
            started |= self._start(task, state, free[0] if task.gpu else "")
        self.save()
        return bool(self.running) or started

    def stop(self) -> None:
        """Stop every running task's process group, so a later resume cannot run it twice."""
        for handle in self.running.values():
            if handle.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(handle.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for name, handle in list(self.running.items()):
            try:
                handle.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(handle.pid, signal.SIGKILL)
                handle.wait()
            self._finish(next(t for t in self.tasks if t.name == name), handle.returncode or 130)
        self.save()


def popen_launch(task: Task, gpu: str, commands: list[str], log: Path):
    script = "set -euo pipefail\n" + "\n".join(commands) + "\n"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with open(log, "a", encoding="utf-8") as h:
        h.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n{script}")
        h.flush()
        return subprocess.Popen(
            ["bash", "-c", script], stdout=h, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )


def arm_tables(ctx: Ctx) -> list[str]:
    """REP tables of the arms, not of the preflight's short evaluations."""
    tables = []
    for path in sorted(ctx.run.glob("*/**/REP/per_case.csv")):
        top = path.relative_to(ctx.run).parts[0]
        if not top.startswith(("pre_", "attempts")) and not top.endswith(("_prep", "_cal")) and "_prep_" not in top:
            tables.append(path.as_posix())
    return tables


def finish(ctx: Ctx, dispatcher: Dispatcher) -> int:
    """Verdicts, Dice groups of the arms and the ZIP; 1 when a task ended in ERROR or BLOCKED."""
    completed = {name for name, state in dispatcher.states.items() if state.status == "PASS"}
    results = verdicts(ctx, completed)
    Path(ctx.path("verdicts.json")).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for row in results:
        dispatcher.log(f"VERDICT {row['comparison']}: {row['status']} {row.get('values', row.get('note', ''))}")
    if tables := arm_tables(ctx):
        groups = subprocess.run([PY, "-m", "tools.analysis.dice_groups", *tables], capture_output=True, text=True)
        Path(ctx.path("dice_groups.txt")).write_text(groups.stdout + groups.stderr, encoding="utf-8")
    archive = Path("results/exports") / f"DISPATCH1_{ctx.stamp}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ctx.run.rglob("*")):
            if path.is_file() and path.name != ".lock" and path.suffix not in (".pth", ".pt"):
                z.write(path, Path(f"DISPATCH1_{ctx.stamp}") / path.relative_to(ctx.run))
    failed = [n for n, s in dispatcher.states.items() if s.status in ("ERROR", "BLOCKED")]
    dispatcher.log(f"done: {archive}; spent {dispatcher.spent(None):.1f} GPU-h; failed: {failed or 'none'}")
    return int(bool(failed))


def prepare(out: str) -> None:
    """parent.json: Mamba NoSVF when its checkpoint and historical FF table are on this node, mamba_ssm imports and the
    strict load passes, otherwise the P1 control. control.json: the P1 control's FF and REP over the 19 test pairs, the
    reference of the MSE and im2grid arms; its checkpoint must be present for the MSE weight."""
    from experiments.dispatch1.bridges import Parent

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(P1["ckpt"]).is_file():
        raise FileNotFoundError(P1["ckpt"])
    control = {}
    for mode in ("FF", "REP"):
        summary = json.loads(Path(P1_SUMMARY.format(mode)).read_text(encoding="utf-8"))
        if summary["n_cases"] != 19:
            raise ValueError(f"the P1 {mode} summary must cover the 19 test pairs")
        control[mode] = {key: finite(summary["metrics"][key]["mean"]) for key in ("dice_mean", "sdlogj")}
    (out_dir / "control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
    notes = []
    for name, parent in PARENTS.items():
        missing = [p for p in (parent["ckpt"], parent["historical_ff"]) if not Path(p).is_file()]
        if missing:
            notes.append(f"{name}: {missing} not on this node")
            continue
        try:
            if name == "mamba_nosvf":
                import mamba_ssm  # noqa: F401
            report = Parent.load(parent["ckpt"], parent["config"], bool(parent["l3_svf"]), 12, "cuda:0").load_report
        except Exception as error:  # the import or the strict load failed: the next parent
            notes.append(f"{name}: {error!r}")
            continue
        choice = {"name": name, **parent, "load": report, "notes": notes}
        (out_dir / "parent.json").write_text(json.dumps(choice, indent=1, default=str), encoding="utf-8")
        print(f"[dispatch1] parent {name}; {notes or 'no notes'}", flush=True)
        return
    raise RuntimeError(f"no usable parent: {notes}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def run_dispatch(run: Path, resume: bool) -> int:
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"]).returncode:
        raise RuntimeError("tracked files differ from HEAD: commit them so that HEAD describes the code that runs")
    head = git("rev-parse", "HEAD")
    branch_cap, total_cap = float(os.environ.get("BRANCH_CAP_H", "72")), float(os.environ.get("TOTAL_CAP_H", "300"))
    gpus = os.environ.get("GPUS", "0").split()
    with open(run / "run_info.txt", "a", encoding="utf-8") as h:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.write(f"{now} HEAD {head} GPUS={' '.join(gpus)} PY={PY} caps={branch_cap}/{total_cap} resume={resume}\n")
    states, saved_at, status = {}, 0.0, run / "status.json"
    if status.exists():
        states = {k: State(**v) for k, v in json.loads(status.read_text(encoding="utf-8")).items()}
        saved_at = status.stat().st_mtime
    ctx = Ctx(run, run.name)
    dispatcher = Dispatcher(tasks(), ctx, gpus, branch_cap, total_cap, popen_launch, states, saved_at, head)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    interrupted = False
    try:
        while dispatcher.step():
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        interrupted = True
        dispatcher.log("interrupted; stopping the running tasks")
    finally:
        if dispatcher.running:
            dispatcher.stop()
    code = finish(ctx, dispatcher)
    return 130 if interrupted else code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepare", metavar="DIR", help="decide and strictly load the parent, check the P1 control")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return 0
    os.chdir(ROOT)
    resume = os.environ.get("RESUME", "")
    run = Path(resume) if resume else Path("results/dispatch1") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if resume and not (run / "status.json").is_file():
        raise ValueError(f"RESUME must name a dispatch with a saved status: {run}")
    run.mkdir(parents=True, exist_ok=bool(resume))
    import fcntl

    with open(run / ".lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"this dispatch is already running: {run}") from error
        return run_dispatch(run, bool(resume))


if __name__ == "__main__":
    sys.exit(main())
