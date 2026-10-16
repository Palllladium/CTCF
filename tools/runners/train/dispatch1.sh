#!/usr/bin/env bash
# Dispatch 1 (experiments/dispatch1/run.py): every pre-registered arm, scheduled over GPUS within the budget caps.
#   GPUS (default "0 1 2 3"), BRANCH_CAP_H (72), TOTAL_CAP_H (300), RESUME=results/dispatch1/<stamp> to continue a run.
#
#   GPUS="0 1 2 3 4 5 6 7" nohup bash tools/runners/train/dispatch1.sh > dispatch1.out 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export PY="${PY:-/data/mooncake/P/envs/ctcf/bin/python}"
export CTCF_DATA_DIR="${CTCF_DATA_DIR:-/data/mooncake/P}"
export GPUS="${GPUS:-0 1 2 3}"
"$PY" -m pytest --version > /dev/null || { echo "[dispatch1] pytest is missing: $PY -m pip install pytest"; exit 1; }
exec "$PY" -m experiments.dispatch1.run
