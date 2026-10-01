"""Shared diagnostic workload and exception vocabulary; standard library only."""

from __future__ import annotations

SOURCE_RUN = "S5_DEVELOPMENT_20260907T202438Z_ffd3090f6129"
SOURCE_HEAD = "ffd3090f6129a48960d849ac345d2fc981dec063"
JOBS = ("F0", "F2V", "F2P", "coverage")
VARIANTS = ("F0", "F2V", "F2S", "F2P", "F4P", "F24P", "A2P", "A24P")
SEEDS = (0, 1, 2)
PROBE_SCALES = (65536.0, 32768.0, 8192.0, 1024.0, 1.0)
TRAINING_SUBJECTS = 294
FP32_EPOCHS = 2
COVERAGE_UPDATES = 8
BENCHMARK_REPEATS = 3
WORKER_SCHEMA = "ctcf-stage5-precision-diagnostic-v1"


def fp32_updates():
    return FP32_EPOCHS * TRAINING_SUBJECTS // 2


def coverage_cases():
    return {(seed, variant) for seed in SEEDS for variant in VARIANTS}


def workload_contract():
    return {
        "fp32_epochs": FP32_EPOCHS,
        "training_subjects": TRAINING_SUBJECTS,
        "fp32_updates": fp32_updates(),
        "coverage_updates": COVERAGE_UPDATES,
        "coverage_cases": len(coverage_cases()),
        "benchmark_repeats": BENCHMARK_REPEATS,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "probe_scales": list(PROBE_SCALES),
    }


def error_status(exc):
    # Avoid importing torch into the stdlib-only finalizer/runner bootstrap.
    # This also recognizes a torch OOM without a message and its subclasses.
    typed_oom = any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__)
    message = str(exc).lower()
    allocation_failure = any(
        marker in message for marker in ("out of memory", "cudnn_status_alloc_failed", "cublas_status_alloc_failed")
    )
    if typed_oom or allocation_failure:
        return "OOM"
    if isinstance(exc, FloatingPointError):
        return "MATH_ERROR"
    return "ERROR"
