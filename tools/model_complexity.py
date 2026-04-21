"""
Model complexity report for the paper: params, FLOPs, peak VRAM, inference throughput.

Reports one line per (model, config_key) pair at the OASIS/IXI input shape (160,192,224).

Usage:
  python tools/model_complexity.py --gpu 0
  python tools/model_complexity.py --gpu 0 --out results/model_complexity.csv --warmup 3 --iters 10

Notes:
  - FLOPs use thop if installed, otherwise fvcore if installed, otherwise reported as "n/a".
  - Peak VRAM is measured via torch.cuda.max_memory_allocated during a no-grad forward pass.
  - Throughput averages `iters` forward passes after `warmup` warmup passes.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from experiments.core.model_adapters import get_model_adapter


MODELS = [
    ("ctcf",      "CTCF-CascadeA"),
    ("tm-dca",    "TransMorph-3-LVL"),
    ("utsrmorph", "UTSRMorph-Large"),
    ("utsrmorph", "UTSRMorph-IXI-Large"),
]

INPUT_SHAPE = (1, 1, 160, 192, 224)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_flops(model, x, y, model_name: str) -> str:
    """Return GFLOPs as string, or 'n/a' if no counter is available / fails."""
    try:
        from thop import profile  # type: ignore

        def _wrap_forward(m, xx, yy):
            if model_name == "ctcf":
                _, flow = m(xx, yy, return_all=False, alpha_l1=1.0)
                return flow
            if model_name == "tm-dca":
                import torch.nn.functional as F
                return m((F.avg_pool3d(xx, 2), F.avg_pool3d(yy, 2)))
            if model_name == "utsrmorph":
                _, flow = m(torch.cat((xx, yy), dim=1))
                return flow
            raise ValueError(model_name)

        class Wrapped(torch.nn.Module):
            def __init__(self, inner, name):
                super().__init__()
                self.inner = inner
                self.name = name
            def forward(self, xx, yy):
                return _wrap_forward(self.inner, xx, yy)

        w = Wrapped(model, model_name)
        macs, _ = profile(w, inputs=(x, y), verbose=False)
        gflops = 2.0 * macs / 1e9  # MACs -> FLOPs
        return f"{gflops:.2f}"
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        class Wrapped(torch.nn.Module):
            def __init__(self, inner, name):
                super().__init__()
                self.inner = inner
                self.name = name
            def forward(self, xx, yy):
                if self.name == "ctcf":
                    _, flow = self.inner(xx, yy, return_all=False, alpha_l1=1.0)
                    return flow
                if self.name == "tm-dca":
                    import torch.nn.functional as F
                    return self.inner((F.avg_pool3d(xx, 2), F.avg_pool3d(yy, 2)))
                _, flow = self.inner(torch.cat((xx, yy), dim=1))
                return flow

        w = Wrapped(model, model_name)
        f = FlopCountAnalysis(w, (x, y)).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        return f"{f.total() / 1e9:.2f}"
    except Exception:
        return "n/a"


def measure_peak_vram(adapter, model, x, y, device) -> float:
    """Peak VRAM in GB over a single no-grad forward (AMP fp16)."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    with torch.no_grad():
        _ = adapter.forward(model, x, y)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


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
    ap.add_argument("--skip-flops", action="store_true", help="Skip FLOPs counting (thop can be slow).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable — complexity measurements require a GPU.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    header = ["model", "config", "params_M", "params_trainable_M", "gflops", "peak_vram_GB", "sec_per_pair", "pairs_per_sec"]
    rows = []

    for model_name, config_key in MODELS:
        print(f"\n=== {model_name} | {config_key} ===")
        adapter = get_model_adapter(model_name)
        build_kwargs = {"config_key": config_key}
        if model_name in ("ctcf", "tm-dca"):
            build_kwargs["time_steps"] = 12

        # Free up memory between runs
        torch.cuda.empty_cache()
        model = adapter.build(**build_kwargs).to(device).eval()
        total, trainable = count_params(model)
        print(f"  params total    : {total/1e6:.2f} M")
        print(f"  params trainable: {trainable/1e6:.2f} M")

        x = torch.randn(*INPUT_SHAPE, device=device)
        y = torch.randn(*INPUT_SHAPE, device=device)

        gflops = "n/a" if args.skip_flops else try_flops(model, x, y, model_name)
        print(f"  gflops          : {gflops}")

        peak_gb = measure_peak_vram(adapter, model, x, y, device)
        print(f"  peak VRAM       : {peak_gb:.2f} GB")

        sec_per_pair, pairs_per_sec = measure_throughput(adapter, model, x, y, device, args.warmup, args.iters)
        print(f"  sec/pair        : {sec_per_pair:.4f}")
        print(f"  pairs/sec       : {pairs_per_sec:.2f}")

        rows.append([
            model_name, config_key,
            f"{total/1e6:.2f}", f"{trainable/1e6:.2f}",
            gflops, f"{peak_gb:.2f}",
            f"{sec_per_pair:.4f}", f"{pairs_per_sec:.2f}",
        ])

        del model, x, y
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n[SAVED] {out_path}")

    # Also emit JSON for reuse
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([dict(zip(header, r)) for r in rows], f, indent=2)
    print(f"[SAVED] {json_path}")


if __name__ == "__main__":
    main()
