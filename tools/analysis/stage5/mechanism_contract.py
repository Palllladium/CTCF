"""Frozen inputs and bounded workload for the post-replay mechanism audit."""

from __future__ import annotations

import argparse

from tools.analysis.stage5.precision_contract import SOURCE_RUN, TRAINING_SUBJECTS

PRECISION_RUN = "S5_PRECISIONDIAG_20260909T140904Z_3575584_cdede42ac60c"
PRECISION_HEAD = "cdede42ac60c1ee1aa93d176683d9e2174819bbb"
PRECISION_SUMS_SHA256 = "0a1ed5f40c864d4499ddcbeeeff192aacd606316d94b1f8ae122a3ca787aba70"
JOBS = ("F0", "F2V", "F2P")
F2P_ATTEMPTS = 2
F2P_UPDATES = TRAINING_SUBJECTS // 2
SCHEMA = "ctcf-stage5-mechanism-diagnostic-v1"
FIELD_MODES = (
    ("fp32_tf32_off", False, False),
    ("fp32_tf32_on", False, True),
    ("fp16_tf32_off", True, False),
    ("fp16_tf32_on", True, True),
)
BIAS_MODES = (
    ("fp16_scale32768", False, 32768.0, False),
    ("fp16_scale1", False, 1.0, False),
    ("fp16_scale65536", False, 65536.0, False),
    ("fp32_strict", True, 1.0, True),
)
BIAS_TF32_OFF_MODE = ("fp16_scale32768_tf32_off", False, 32768.0, True)
BIAS_COMPARISONS = (
    ("fp16_scale1", "fp16_scale32768"),
    ("fp16_scale65536", "fp16_scale32768"),
    ("fp16_scale32768", "fp32_strict"),
    ("fp16_scale32768_tf32_off", "fp32_strict"),
    ("fp16_scale32768_tf32_off", "fp16_scale32768"),
)


def bias_modes(backend_flags):
    modes = list(BIAS_MODES)
    if backend_flags["cudnn_allow_tf32"] or backend_flags["matmul_allow_tf32"]:
        modes.append(BIAS_TF32_OFF_MODE)
    return modes


def workload_contract():
    return {
        "source_precision_run": PRECISION_RUN,
        "source_precision_head": PRECISION_HEAD,
        "source_precision_sums_sha256": PRECISION_SUMS_SHA256,
        "jobs": list(JOBS),
        "saved_failure_jobs": ["F0", "F2V"],
        "f2p_max_attempts": F2P_ATTEMPTS,
        "f2p_epochs_per_attempt": 1,
        "f2p_updates_per_attempt": F2P_UPDATES,
        "f2p_seed": 0,
        "f2p_stop_after_first_captured_failure": True,
        "field_modes": [list(mode) for mode in FIELD_MODES],
        "bias_modes": [list(mode) for mode in BIAS_MODES],
        "bias_optional_tf32_off_mode": list(BIAS_TF32_OFF_MODE),
        "bias_comparisons": [list(pair) for pair in BIAS_COMPARISONS],
        "production_training": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-values", required=True, action="store_true")
    parser.parse_args()
    print("\n".join((SOURCE_RUN, PRECISION_RUN, *JOBS)))


if __name__ == "__main__":
    main()
