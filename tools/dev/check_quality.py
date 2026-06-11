from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tracked source dirs that must always byte-compile (skips gitignored tools/paper, tools/archive).
COMPILE_DIRS = ["models", "experiments", "utils", "datasets", "tools/analysis", "tools/dev"]

# Modules that must import cleanly. Heavy-dep failures downgrade to SKIP (see HEAVY_DEPS).
IMPORT_TARGETS = [
    "tools.analysis.aggregate_results",
    "tools.analysis.compute_stats",
    "tools.analysis.model_complexity",
]

# An ImportError naming one of these is an environment gap, not a code regression.
HEAVY_DEPS = {"torch", "mamba_ssm", "selective_scan_cuda", "causal_conv1d", "scipy", "pandas", "numpy"}

# Tokens that must not survive a rename anywhere in tracked sources.
STALE_TOKENS = ["aggregate_sedm_results"]

STATUS_TAG = {"ok": "[ OK ]", "warn": "[WARN]", "skip": "[SKIP]", "fail": "[FAIL]"}


class Results:
    """Accumulator for (status, label, detail) check rows."""

    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, label: str, detail: str = "") -> None:
        self.rows.append((status, label, detail))
        line = f"  {STATUS_TAG[status]}  {label}"
        print(line)
        if detail and status in {"warn", "skip", "fail"}:
            for chunk in detail.strip().splitlines():
                print(f"           {chunk}")

    def from_proc(self, proc: subprocess.CompletedProcess, label: str) -> None:
        if proc.returncode == 0:
            self.add("ok", label)
            return
        tail = (proc.stdout or "") + (proc.stderr or "")
        self.add("fail", label, _tail(tail))

    @property
    def failed(self) -> int:
        return sum(1 for status, _, _ in self.rows if status == "fail")


def _tail(text: str, max_lines: int = 12) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def _resolve_ruff() -> list[str] | None:
    for base in (["ruff"], [sys.executable, "-m", "ruff"], ["uvx", "ruff"]):
        proc = _run([*base, "--version"])
        if proc is not None and proc.returncode == 0:
            return base
    return None


def check_ruff(results: Results) -> None:
    base = _resolve_ruff()
    if base is None:
        results.add("skip", "ruff", "not found (tried ruff, python -m ruff, uvx ruff)")
        return
    results.from_proc(_run([*base, "format", "--check", "."]), "ruff format --check")
    results.from_proc(_run([*base, "check", "."]), "ruff check")


def check_compile(results: Results) -> None:
    proc = _run([sys.executable, "-m", "compileall", "-q", *COMPILE_DIRS])
    results.from_proc(proc, "compileall (" + ", ".join(COMPILE_DIRS) + ")")


def check_imports(results: Results) -> None:
    for mod in IMPORT_TARGETS:
        try:
            importlib.import_module(mod)
            results.add("ok", f"import {mod}")
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing in HEAVY_DEPS:
                results.add("skip", f"import {mod}", f"missing optional dep: {exc.name}")
            else:
                results.add("fail", f"import {mod}", f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            results.add("fail", f"import {mod}", f"{type(exc).__name__}: {exc}")


def check_pyproject_paths(results: Results) -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = _extract_block(text, "[tool.ruff.lint.per-file-ignores]")
    keys = re.findall(r'^"([^"]+)"\s*=', block, flags=re.MULTILINE)
    if not keys:
        results.add("warn", "pyproject per-file-ignores", "no entries parsed")
        return
    for key in keys:
        pattern = key if "/" in key else f"**/{key}"
        if next(REPO_ROOT.glob(pattern), None) is not None:
            results.add("ok", f"per-file-ignore resolves: {key}")
        else:
            results.add("fail", f"per-file-ignore matches nothing: {key}")


def _extract_block(text: str, header: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == header:
            inside = True
            continue
        if inside and line.lstrip().startswith("["):
            break
        if inside:
            out.append(line)
    return "\n".join(out)


def check_stale_tokens(results: Results) -> None:
    for token in STALE_TOKENS:
        proc = _run(["git", "grep", "-l", token])
        if proc is None:
            results.add("skip", f"stale token '{token}'", "git unavailable")
        elif proc.returncode == 0 and proc.stdout.strip():
            results.add("fail", f"stale token '{token}' present", proc.stdout.strip().replace("\n", ", "))
        else:
            results.add("ok", f"no stale token '{token}'")


def smoke_vxm(results: Results) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        results.add("skip", "VxM wrapper forward", "torch absent")
        return

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
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
            mov = torch.rand(1, 1, *vol)
            fix = torch.rand(1, 1, *vol)
            warped, flow = net(mov, fix)
            warped_i, flow_i = net(mov, fix, init_flow=torch.zeros(1, 3, *vol))

        shapes = (tuple(warped.shape), tuple(flow.shape), tuple(warped_i.shape), tuple(flow_i.shape))
        expect = ((1, 1, *vol), (1, 3, *vol), (1, 1, *vol), (1, 3, *vol))
        if shapes == expect:
            results.add("ok", "VxM wrapper forward (init_flow None + provided)")
        else:
            results.add("fail", "VxM wrapper forward", f"unexpected shapes: {shapes}")
    except Exception as exc:
        results.add("fail", "VxM wrapper forward", f"{type(exc).__name__}: {exc}")


def probe_mamba(results: Results) -> None:
    try:
        importlib.import_module("mamba_ssm")
        results.add("ok", "mamba_ssm importable (Mamba/VMamba smoke possible)")
    except Exception as exc:
        results.add("skip", "Mamba/VMamba smoke", f"mamba_ssm unavailable: {type(exc).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CTCF repo quality gate (ruff + compile + import + smoke).")
    parser.add_argument("--full", action="store_true", help="Also run CPU model smokes (needs torch).")
    args = parser.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    results = Results()

    print("== quick ==")
    check_ruff(results)
    check_compile(results)
    check_imports(results)
    check_pyproject_paths(results)
    check_stale_tokens(results)

    if args.full:
        print("== full (model smoke) ==")
        smoke_vxm(results)
        probe_mamba(results)

    print()
    counts = {key: 0 for key in STATUS_TAG}
    for status, _, _ in results.rows:
        counts[status] += 1
    summary = "  ".join(f"{STATUS_TAG[key]} {counts[key]}" for key in STATUS_TAG)
    print(f"summary: {summary}")

    if results.failed:
        print(f"\nFAILED: {results.failed} check(s) need attention.")
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
