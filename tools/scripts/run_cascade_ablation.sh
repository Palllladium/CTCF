#!/usr/bin/env bash
# Cascade decomposition ablation (Round 4).
# Proves the independent contribution of each cascade level.
#
# Configs:
#   ABL4_01_L2_ONLY     = L2 only (no L1, no L3) — baseline backbone
#   ABL4_02_L1_L2       = L1 + L2 (coarse init, no refiner)
#   ABL4_03_L2_L3       = L2 + L3 (no coarse init, refiner only)
#   ABL4_04_L1_L2_L3    = Full CTCF (L1 + L2 + L3) — reference
#
# Usage:
#   bash tools/scripts/run_cascade_ablation.sh [--work-dir ~/CTCF] [--data-dir /data]
#                                               [--paths-profile 3] [--env-name oasis-ctcf]
#                                               [--skip-to N] [--gpu 0] [--auto-pack 0|1]
set -euo pipefail

WORK_DIR="${CTCF_WORK_DIR:-}"
ENV_NAME="${CTCF_ENV_NAME:-${CONDA_DEFAULT_ENV:-oasis-ctcf}}"
PATHS_PROFILE=3
DATA_DIR=""
SKIP_TO=0
GPU=0
NUM_WORKERS=8
AUTO_PACK=0
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --skip-to) SKIP_TO="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --paths-profile) PATHS_PROFILE="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --auto-pack) AUTO_PACK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$WORK_DIR" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

if [ -z "$DATA_DIR" ]; then
  if [ -f "$WORK_DIR/.env_data" ]; then
    source "$WORK_DIR/.env_data"
    DATA_DIR="${DATA_DIR:-/data}"
  else
    DATA_DIR="/data"
  fi
fi

cd "$WORK_DIR"
export CTCF_DATA_DIR="$DATA_DIR"

# Common flags: 100 epochs, OASIS, optimal config (L1=32, L3=64, NCC, TS6)
COMMON="--ds OASIS --max_epoch 100 --w_reg 1.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 32 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6 --save_ckpt 0 --use_checkpoint 0 --use_tb 0"

EXPERIMENTS=(
  "ABL4_01_L2_ONLY|$COMMON --disable_l1 1 --disable_l3 1"
  "ABL4_02_L1_L2|$COMMON --disable_l3 1"
  "ABL4_03_L2_L3|$COMMON --disable_l1 1"
  "ABL4_04_L1_L2_L3|$COMMON"
)

echo "============================================"
echo "  CTCF Cascade Decomposition Ablation (R4)"
echo "  Data:        $DATA_DIR"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Env:         $ENV_NAME"
echo "  GPU:         $GPU"
echo "  Auto-pack:   $AUTO_PACK"
echo "============================================"
echo ""

RESULTS_FILE="$WORK_DIR/ablation_4_results.txt"
echo "=== Cascade Decomposition Ablation ===" > "$RESULTS_FILE"
echo "Started: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for i in "${!EXPERIMENTS[@]}"; do
  if [ "$i" -lt "$SKIP_TO" ]; then
    echo "[SKIP] Experiment $i"
    continue
  fi

  IFS='|' read -r EXP_NAME EXTRA_ARGS <<< "${EXPERIMENTS[$i]}"
  EXP_NAME=$(echo "$EXP_NAME" | xargs)
  EXTRA_ARGS=$(echo "$EXTRA_ARGS" | xargs)

  echo ""
  echo "============================================"
  echo "  [$i/${#EXPERIMENTS[@]}] $EXP_NAME"
  echo "  Started: $(date)"
  echo "============================================"

  START_TIME=$(date +%s)
  mkdir -p "logs/${EXP_NAME}"

  conda run -n "$ENV_NAME" --no-capture-output \
    python -m experiments.train_CTCF \
      --$PATHS_PROFILE \
      --exp "$EXP_NAME" \
      --gpu "$GPU" \
      --num_workers "$NUM_WORKERS" \
      $EXTRA_ARGS || {
    echo "WARNING: $EXP_NAME failed with exit code $?"
    echo "[$i] $EXP_NAME: FAILED" >> "$RESULTS_FILE"
    continue
  }

  END_TIME=$(date +%s)
  ELAPSED=$(( END_TIME - START_TIME ))
  echo "[$i] $EXP_NAME: completed in ${ELAPSED}s" >> "$RESULTS_FILE"

  LOGFILE="logs/${EXP_NAME}/logfile.log"
  if [ -f "$LOGFILE" ]; then
    LAST_DICE=$(grep -oP 'val_dice=\K[0-9.]+' "$LOGFILE" | tail -1 || echo "N/A")
    BEST_DICE=$(grep -oP 'val_dice=\K[0-9.]+' "$LOGFILE" | sort -rn | head -1 || echo "N/A")
    echo "  Last Dice: $LAST_DICE, Best Dice: $BEST_DICE" >> "$RESULTS_FILE"
    echo "  >> Last Dice: $LAST_DICE, Best Dice: $BEST_DICE"
  fi
done

echo ""
echo "============================================"
echo "  Cascade Decomposition Ablation Complete!"
echo "============================================"
cat "$RESULTS_FILE"

if [ "$AUTO_PACK" = "1" ]; then
  echo ""
  echo "Packing logs into archive..."
  ARCHIVE_NAME="ctcf_abl4_$(date +%Y%m%d).tar.gz"
  if bash "$WORK_DIR/tools/scripts/package_results.sh" --work-dir "$WORK_DIR" --archive-name "$ARCHIVE_NAME" --exp-glob "ABL4_*"; then
    echo "Archive ready: $WORK_DIR/$ARCHIVE_NAME"
  else
    echo "WARNING: auto packaging failed. You can retry manually:"
    echo "  bash tools/scripts/package_results.sh --work-dir $WORK_DIR --exp-glob 'ABL4_*'"
  fi
fi
