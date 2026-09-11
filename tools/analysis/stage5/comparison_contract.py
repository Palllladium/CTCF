"""One workload for the bounded precision comparison; no production selection."""

SCHEMA = "ctcf-stage5-precision-comparison-worker-v1"
MODES = ("fp32_strict", "tf32", "bf16")
PAIRED_MODES = (("fp32_strict", 0), ("fp32_strict", 1), ("tf32", 0), ("bf16", 0))
TRAJECTORY_VARIANTS = ("F0", "F2V", "F2P")
VARIANTS = ("F0", "F2V", "F2S", "F2P", "F4P", "F24P", "A2P", "A24P")
SEEDS = (0, 1, 2)
EPOCHS = 2
COVERAGE_UPDATES = 8
BENCHMARK_REPEATS = 3
TRAINING_SUBJECTS = 294


def workload_contract():
    return {
        "schema": "ctcf-stage5-precision-comparison-workload-v1",
        "epochs": EPOCHS,
        "coverage_updates": COVERAGE_UPDATES,
        "benchmark_repeats": BENCHMARK_REPEATS,
        "training_subjects": TRAINING_SUBJECTS,
        "modes": list(MODES),
        "paired_modes": [list(value) for value in PAIRED_MODES],
        "trajectory_variants": list(TRAJECTORY_VARIANTS),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "labels_accessed": False,
        "production_training": False,
        "automatic_mode_selection": False,
        "scope": "Numerical and execution observations, not controller quality or convergence validation",
    }


def expected_updates(stage, subject_count=TRAINING_SUBJECTS):
    if subject_count != TRAINING_SUBJECTS or subject_count % 2:
        raise ValueError("Comparison requires the frozen training subject count")
    if stage in ("trajectory", "throughput"):
        return EPOCHS * (subject_count // 2)
    if stage == "coverage":
        return len(MODES) * COVERAGE_UPDATES
    if stage == "paired":
        return len(PAIRED_MODES)
    raise ValueError(f"Unknown comparison stage: {stage}")
