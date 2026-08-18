from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

from tools.analysis.run_artifacts import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed CTCF checkpoint compatibility check.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ctcf-config", required=True)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--ctcf-l3-svf", type=int, choices=[0, 1], default=None)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report: dict[str, object] = {
        "schema": "ctcf-checkpoint-preflight-v1",
        "started_at_utc": started_at,
        "checkpoint": str(checkpoint),
        "ctcf_config": args.ctcf_config,
        "time_steps": args.time_steps,
        "ctcf_l3_svf": None if args.ctcf_l3_svf is None else bool(args.ctcf_l3_svf),
        "expected_sha256": args.expected_sha256.lower() or None,
        "host": platform.node(),
        "python": platform.python_version(),
        "status": "FAILED",
    }

    try:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"Checkpoint is empty: {checkpoint}")

        actual_sha256 = sha256_file(checkpoint)
        report["bytes"] = checkpoint.stat().st_size
        report["sha256"] = actual_sha256
        if args.expected_sha256 and actual_sha256.lower() != args.expected_sha256.lower():
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch: expected {args.expected_sha256.lower()}, got {actual_sha256}"
            )

        import torch

        from experiments.core.inference_runtime import load_checkpoint_state
        from experiments.core.model_adapters import get_model_adapter

        report["torch"] = torch.__version__
        adapter = get_model_adapter("ctcf")
        model = adapter.build(
            time_steps=args.time_steps,
            config_key=args.ctcf_config,
            l3_svf=None if args.ctcf_l3_svf is None else bool(args.ctcf_l3_svf),
        )
        report["load"] = load_checkpoint_state(model, str(checkpoint), strict=True)
        report["status"] = "PASS"
        return_code = 0
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        return_code = 1
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        atomic_write_json(output, report)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
