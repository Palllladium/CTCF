"""
Model complexity report for the paper: params, FLOPs, peak VRAM, inference throughput.
Reports one line per model/configuration at the OASIS/IXI input shape (160, 192, 224).

Usage:
  python tools/model_complexity.py --gpu 0
  python tools/model_complexity.py --gpu 0 --out results/model_complexity.csv --warmup 3 --iters 10

Notes:
  - FLOPs use thop if installed, otherwise fvcore if installed, otherwise reported as "n/a".
    For backbones with custom CUDA kernels (Mamba's selective_scan), the reported FLOPs
    UNDER-COUNT the work done by those kernels (the counter only sees standard torch ops).
  - Peak VRAM is measured via torch.cuda.max_memory_allocated during a no-grad forward pass.
  - Throughput averages `iters` forward passes after `warmup` warmup passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.core.model_adapters import get_model_adapter

MODELS = [
    {"role": "mdpi", "label": "CTCF-CascadeA", "model": "ctcf", "config": "CTCF-CascadeA"},
    {"role": "mdpi", "label": "TransMorph-3-LVL", "model": "tm-dca", "config": "TransMorph-3-LVL"},
    {"role": "mdpi", "label": "UTSRMorph-Large", "model": "utsrmorph", "config": "UTSRMorph-Large"},
    {"role": "mdpi", "label": "UTSRMorph-IXI-Large", "model": "utsrmorph", "config": "UTSRMorph-IXI-Large"},
    {"role": "standalone", "label": "LKU-4", "model": "lkunet", "config": "LKU-4"},
    {"role": "standalone", "label": "LKU-8", "model": "lkunet", "config": "LKU-8"},
    {
        "role": "standalone",
        "label": "EfficientMorph_2x3_2_hires",
        "model": "efficientmorph",
        "config": "EfficientMorph_2x3_2_hires",
    },
    {"role": "standalone", "label": "MambaMorph", "model": "mambamorph", "config": "MambaMorph"},
    {"role": "standalone", "label": "VMambaMorph", "model": "vmambamorph", "config": "VMambaMorph"},
    {"role": "sedm_l2_only", "label": "CTCF-VM-solo", "model": "ctcf", "config": "CTCF-VM-solo"},
    {"role": "sedm_l2_only", "label": "CTCF-LKU8-solo", "model": "ctcf", "config": "CTCF-LKU8-solo"},
    {"role": "sedm_l2_only", "label": "CTCF-LKU32-solo", "model": "ctcf", "config": "CTCF-LKU32-solo"},
    {"role": "sedm_l2_only", "label": "CTCF-Mamba-solo", "model": "ctcf", "config": "CTCF-Mamba-solo"},
    {"role": "sedm_l2_only", "label": "CTCF-VMamba-solo", "model": "ctcf", "config": "CTCF-VMamba-solo"},
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-VM-SVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-VM",
        "l3_svf": True,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-VM-NoSVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-VM",
        "l3_svf": False,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-LKU8-SVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-LKU8",
        "l3_svf": True,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-LKU8-NoSVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-LKU8",
        "l3_svf": False,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-LKU32-SVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-LKU32",
        "l3_svf": True,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-LKU32-NoSVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-LKU32",
        "l3_svf": False,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-Mamba-SVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-Mamba",
        "l3_svf": True,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-Mamba-NoSVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-Mamba",
        "l3_svf": False,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-VMamba-SVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-VMamba",
        "l3_svf": True,
    },
    {
        "role": "sedm_cascade",
        "label": "CTCF-CascadeA-VMamba-NoSVF",
        "model": "ctcf",
        "config": "CTCF-CascadeA-VMamba",
        "l3_svf": False,
    },
]

INPUT_SHAPE = (1, 1, 160, 192, 224)
# Models whose forward returns (warped, flow) directly via forward(mov, fix).
# Used by both try_flops and the inference path of the adapter.
_PAIR_OUTPUT_MODELS = {"lkunet", "efficientmorph", "mambamorph", "vmambamorph"}


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def svf_label(value) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "config"


class _FlopWrapped(torch.nn.Module):
    """Adapter-agnostic wrapper that exposes a forward(mov, fix) -> flow signature
    so FLOPs counters (thop / fvcore) see a single, well-defined forward path."""

    def __init__(self, inner: torch.nn.Module, name: str):
        super().__init__()
        self.inner = inner
        self.name = name

    def forward(self, xx, yy):
        if self.name == "ctcf":
            _, flow = self.inner(xx, yy, alpha_l1=1.0)
            return flow
        if self.name == "tm-dca":
            return self.inner((F.avg_pool3d(xx, 2), F.avg_pool3d(yy, 2)))
        if self.name == "utsrmorph":
            _, flow = self.inner(torch.cat((xx, yy), dim=1))
            return flow
        if self.name in _PAIR_OUTPUT_MODELS:
            _, flow = self.inner(xx, yy)
            return flow
        raise ValueError(f"Unknown model_name: {self.name}")


def try_flops(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, model_name: str) -> str:
    """Return GFLOPs as a string ('xx.xx') or 'n/a' if no counter is available / fails.

    For Mamba-based models, the counter under-counts the selective_scan_fn CUDA kernel
    (it is invisible to thop/fvcore hooks). Treat reported numbers as a lower bound.
    """
    wrapped = _FlopWrapped(model, model_name)

    try:
        from thop import profile  # type: ignore

        macs, _ = profile(wrapped, inputs=(x, y), verbose=False)
        return f"{2.0 * macs / 1e9:.2f}"
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        f = FlopCountAnalysis(wrapped, (x, y)).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        return f"{f.total() / 1e9:.2f}"
    except Exception:
        return "n/a"


def measure_peak_vram(adapter, model, x, y, device) -> float:
    """Peak VRAM in GB over a single no-grad forward (per-adapter AMP setting)."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    with torch.no_grad():
        _ = adapter.forward(model, x, y)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def measure_throughput(adapter, model, x, y, device, warmup: int, iters: int) -> tuple[float, float]:
    """Return (mean_sec_per_pair, throughput_pairs_per_sec)."""
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = adapter.forward(model, x, y)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = adapter.forward(model, x, y)
        torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / iters
    return dt, 1.0 / dt if dt > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", type=str, default="results/model_complexity.csv")
    ap.add_argument("--skip-flops", action="store_true", help="Skip FLOPs counting (thop can be slow on large models).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: complexity measurements require a GPU.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    header = [
        "role",
        "label",
        "model",
        "config",
        "l3_svf",
        "params_M",
        "params_trainable_M",
        "gflops",
        "peak_vram_GB",
        "sec_per_pair",
        "pairs_per_sec",
    ]
    rows = []

    for spec in MODELS:
        role = spec["role"]
        label = spec["label"]
        model_name = spec["model"]
        config_key = spec["config"]
        l3_svf = spec.get("l3_svf")

        print(f"\n=== {role} | {label} | {model_name} | {config_key} ===")
        adapter = get_model_adapter(model_name)
        build_kwargs: dict = {"config_key": config_key}
        if model_name in ("ctcf", "tm-dca"):
            build_kwargs["time_steps"] = 12
        if model_name == "ctcf" and l3_svf is not None:
            build_kwargs["l3_svf"] = bool(l3_svf)

        torch.cuda.empty_cache()
        try:
            model = adapter.build(**build_kwargs).to(device).eval()
        except Exception as exc:
            print(f"  BUILD FAILED: {type(exc).__name__}: {exc}")
            rows.append(
                [
                    role,
                    label,
                    model_name,
                    config_key,
                    svf_label(l3_svf),
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                ]
            )
            continue

        total, trainable = count_params(model)
        print(f"  params total    : {total / 1e6:.2f} M")
        print(f"  params trainable: {trainable / 1e6:.2f} M")

        x = y = None
        try:
            x = torch.randn(*INPUT_SHAPE, device=device)
            y = torch.randn(*INPUT_SHAPE, device=device)

            gflops = "n/a" if args.skip_flops else try_flops(model, x, y, model_name)
            print(f"  gflops          : {gflops}")

            peak_gb = measure_peak_vram(adapter, model, x, y, device)
            print(f"  peak VRAM       : {peak_gb:.2f} GB")

            sec_per_pair, pairs_per_sec = measure_throughput(adapter, model, x, y, device, args.warmup, args.iters)
            print(f"  sec/pair        : {sec_per_pair:.4f}")
            print(f"  pairs/sec       : {pairs_per_sec:.2f}")
        except Exception as exc:
            print(f"  MEASURE FAILED: {type(exc).__name__}: {exc}")
            rows.append(
                [
                    role,
                    label,
                    model_name,
                    config_key,
                    svf_label(l3_svf),
                    f"{total / 1e6:.2f}",
                    f"{trainable / 1e6:.2f}",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                ]
            )
            del model
            if x is not None:
                del x
            if y is not None:
                del y
            torch.cuda.empty_cache()
            continue

        rows.append(
            [
                role,
                label,
                model_name,
                config_key,
                svf_label(l3_svf),
                f"{total / 1e6:.2f}",
                f"{trainable / 1e6:.2f}",
                gflops,
                f"{peak_gb:.2f}",
                f"{sec_per_pair:.4f}",
                f"{pairs_per_sec:.2f}",
            ]
        )

        del model, x, y
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n[SAVED] {out_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([dict(zip(header, r, strict=False)) for r in rows], f, indent=2)
    print(f"[SAVED] {json_path}")


if __name__ == "__main__":
    main()
