from __future__ import annotations

import argparse
import csv
import importlib
import re
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPILE_DIRS = ["models", "experiments", "utils", "datasets", "tools/analysis", "tools/dev"]
IMPORT_TARGETS = [
    "tools.analysis.aggregate_results",
    "tools.analysis.compute_stats",
    "tools.analysis.model_complexity",
]
OPTIONAL_DEPS = {"mamba_ssm", "selective_scan_cuda", "causal_conv1d"}

MANIFEST_REL = "tools/analysis/manifests/experiments.csv"
MANIFEST_COLUMNS = {"exp_name", "ds", "group", "family", "backbone", "svf", "params_m", "note"}
MANIFEST_DATASETS = {"OASIS", "IXI"}
MANIFEST_GROUPS = {"cascade", "L2-only", "cross-dataset"}
MANIFEST_SVF = {"", "ON", "OFF"}

STATUS_TAG = {"ok": "[ OK ]", "warn": "[WARN]", "skip": "[SKIP]", "fail": "[FAIL]"}


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, label: str, detail: str = "") -> None:
        self.rows.append((status, label, detail))
        print(f"  {STATUS_TAG[status]}  {label}")
        if detail and status in {"warn", "skip", "fail"}:
            for line in detail.strip().splitlines():
                print(f"           {line}")

    def from_proc(self, proc: subprocess.CompletedProcess | None, label: str) -> None:
        if proc is None:
            self.add("skip", label, "command not found")
            return
        if proc.returncode == 0:
            self.add("ok", label)
            return
        self.add("fail", label, _tail((proc.stdout or "") + (proc.stderr or "")))

    def count(self, status: str) -> int:
        return sum(row_status == status for row_status, _, _ in self.rows)


def _tail(text: str, max_lines: int = 12) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def _ensure_repo_on_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _pinned_ruff_version() -> str | None:
    cfg = REPO_ROOT / ".pre-commit-config.yaml"
    if not cfg.exists():
        return None
    text = cfg.read_text(encoding="utf-8")
    match = re.search(r"ruff-pre-commit.*?\n\s*rev:\s*v?([0-9][0-9.]*)", text, re.DOTALL)
    return match.group(1) if match else None


def _resolve_ruff() -> list[str] | None:
    version = _pinned_ruff_version()
    candidates: list[list[str]] = []
    if version:
        candidates.append(["uvx", f"ruff@{version}"])
    candidates.extend((["ruff"], [sys.executable, "-m", "ruff"], ["uvx", "ruff"]))

    for base in candidates:
        proc = _run([*base, "--version"])
        if proc is not None and proc.returncode == 0:
            return base
    return None


def check_ruff(results: Results) -> None:
    base = _resolve_ruff()
    if base is None:
        results.add("skip", "ruff", "not found; install ruff or provide uvx")
        return

    version = _run([*base, "--version"])
    label_suffix = f" ({version.stdout.strip()})" if version and version.returncode == 0 else ""
    results.from_proc(_run([*base, "format", "--check", "."]), f"ruff format --check{label_suffix}")
    results.from_proc(_run([*base, "check", "."]), f"ruff check{label_suffix}")


def check_git_diff(results: Results) -> None:
    results.from_proc(_run(["git", "diff", "--check"]), "git diff --check")


def check_compile(results: Results) -> None:
    label = "compileall (" + ", ".join(COMPILE_DIRS) + ")"
    results.from_proc(_run([sys.executable, "-m", "compileall", "-q", *COMPILE_DIRS]), label)


def check_imports(results: Results) -> None:
    _ensure_repo_on_path()
    for module in IMPORT_TARGETS:
        try:
            importlib.import_module(module)
            results.add("ok", f"import {module}")
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing in OPTIONAL_DEPS:
                results.add("skip", f"import {module}", f"missing optional dep: {exc.name}")
            else:
                results.add("fail", f"import {module}", f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            results.add("fail", f"import {module}", f"{type(exc).__name__}: {exc}")


def check_pyproject_paths(results: Results) -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        results.add("fail", "pyproject per-file-ignores", "pyproject.toml is missing")
        return

    block = _toml_block(pyproject.read_text(encoding="utf-8"), "[tool.ruff.lint.per-file-ignores]")
    patterns = re.findall(r'^"([^"]+)"\s*=', block, flags=re.MULTILINE)
    if not patterns:
        results.add("warn", "pyproject per-file-ignores", "no entries parsed")
        return

    missing = []
    for pattern in patterns:
        glob_pattern = pattern if "/" in pattern else f"**/{pattern}"
        if next(REPO_ROOT.glob(glob_pattern), None) is None:
            missing.append(pattern)

    if missing:
        results.add("fail", "pyproject per-file-ignores", "matches nothing: " + ", ".join(missing))
    else:
        results.add("ok", f"pyproject per-file-ignores ({len(patterns)} patterns)")


def _toml_block(text: str, header: str) -> str:
    lines = text.splitlines()
    inside = False
    block: list[str] = []
    for line in lines:
        if line.strip() == header:
            inside = True
            continue
        if inside and line.lstrip().startswith("["):
            break
        if inside:
            block.append(line)
    return "\n".join(block)


def check_manifest(results: Results) -> None:
    path = REPO_ROOT / MANIFEST_REL
    if not path.exists():
        results.add("fail", "experiment manifest", f"missing: {MANIFEST_REL}")
        return

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = sorted(MANIFEST_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            results.add("fail", "experiment manifest", "missing columns: " + ", ".join(missing_columns))
            return
        rows = list(reader)

    seen: set[str] = set()
    problems: list[str] = []
    for line_no, row in enumerate(rows, start=2):
        name = row.get("exp_name", "").strip()
        if not name:
            problems.append(f"line {line_no}: empty exp_name")
            continue
        if name in seen:
            problems.append(f"line {line_no}: duplicate exp_name {name}")
        seen.add(name)

        if row["ds"] not in MANIFEST_DATASETS:
            problems.append(f"{name}: bad ds {row['ds']!r}")
        if row["group"] not in MANIFEST_GROUPS:
            problems.append(f"{name}: bad group {row['group']!r}")
        if row["svf"] not in MANIFEST_SVF:
            problems.append(f"{name}: bad svf {row['svf']!r}")
        try:
            float(row["params_m"])
        except ValueError:
            problems.append(f"{name}: params_m is not a float ({row['params_m']!r})")

    if problems:
        results.add("fail", "experiment manifest", "\n".join(problems[:12]))
    else:
        results.add("ok", f"experiment manifest ({len(rows)} rows)")


def check_public_hygiene(results: Results) -> None:
    proc = _run(["git", "ls-files", "tools/archive"])
    if proc is None:
        results.add("skip", "tracked tools/archive", "git not found")
        return
    tracked = proc.stdout.strip()
    if tracked:
        results.add("warn", "tracked tools/archive", tracked)
    else:
        results.add("ok", "no tracked tools/archive files")


def smoke_vxm(results: Results) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        results.add("fail", "VxM wrapper forward", "torch absent; activate the project conda env")
        return

    _ensure_repo_on_path()
    try:
        from models.VoxelMorph.wrapper import VxmDenseHalf

        vol = (32, 32, 32)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = VxmDenseHalf(
                vol_size=vol,
                enc_nf=[16, 32, 32, 32],
                dec_nf=[32, 32, 32, 32, 32, 16, 16],
                int_steps=1,
            ).eval()
            moving = torch.rand(1, 1, *vol)
            fixed = torch.rand(1, 1, *vol)
            warped, flow = net(moving, fixed)
            warped_i, flow_i = net(moving, fixed, init_flow=torch.zeros(1, 3, *vol))

        shapes = (tuple(warped.shape), tuple(flow.shape), tuple(warped_i.shape), tuple(flow_i.shape))
        expected = ((1, 1, *vol), (1, 3, *vol), (1, 1, *vol), (1, 3, *vol))
        if shapes == expected:
            results.add("ok", "VxM wrapper forward (init_flow None + provided)")
        else:
            results.add("fail", "VxM wrapper forward", f"unexpected shapes: {shapes}")
    except Exception as exc:
        results.add("fail", "VxM wrapper forward", f"{type(exc).__name__}: {exc}")


def smoke_cli(results: Results) -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        results.add("fail", "CTCF override CLI parity", "torch absent; activate the project conda env")
        return

    _ensure_repo_on_path()
    try:
        from experiments import inference, train_CTCF
        from experiments.core.cli_ctcf import CTCF_OVERRIDE_KEYS, ctcf_overrides_from_args

        argv = sys.argv
        try:
            sys.argv = ["train_CTCF"]
            train_args = train_CTCF.parse_args()
            sys.argv = ["inference", "--model", "ctcf", "--ckpt", "x"]
            infer_args = inference.parse_args()
        finally:
            sys.argv = argv

        train_keys = set(ctcf_overrides_from_args(train_args))
        infer_keys = set(ctcf_overrides_from_args(infer_args, prefix="ctcf_"))
        expected = set(CTCF_OVERRIDE_KEYS)
        if train_keys == infer_keys == expected:
            results.add("ok", f"CTCF override CLI parity train/infer ({len(expected)} flags)")
        else:
            detail = f"train={sorted(train_keys)} infer={sorted(infer_keys)} expected={sorted(expected)}"
            results.add("fail", "CTCF override CLI parity", detail)
    except Exception as exc:
        results.add("fail", "CTCF override CLI parity", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local CTCF quality gate.")
    parser.add_argument("--full", action="store_true", help="also run CPU model/CLI smoke tests")
    args = parser.parse_args()

    _ensure_repo_on_path()
    results = Results()

    print("== quick ==")
    check_ruff(results)
    check_git_diff(results)
    check_compile(results)
    check_imports(results)
    check_pyproject_paths(results)
    check_manifest(results)
    check_public_hygiene(results)

    if args.full:
        print("== full ==")
        smoke_vxm(results)
        smoke_cli(results)

    print()
    summary = "  ".join(f"{STATUS_TAG[key]} {results.count(key)}" for key in STATUS_TAG)
    print(f"summary: {summary}")

    failures = results.count("fail")
    if failures:
        print(f"\nFAILED: {failures} check(s) need attention.")
        return 1

    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
