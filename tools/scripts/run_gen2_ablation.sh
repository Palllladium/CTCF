#!/usr/bin/env bash
# GEN2 Enhancement Ablation (Round 5).
# All runs use half-res L1 + full-res L3 as baseline upgrade.
# Tests remaining enhancements on top of that.
#
# Reference: ABL3_01 (Dice=0.8162, 100ep, quarter-res L1, half-res L3)
#
# Usage:
#   bash tools/scripts/run_gen2_ablation.sh [--work-dir ~/CTCF] [--data-dir /data]
#                                            [--paths-profile 3] [--env-name oasis-ctcf]
#                                            [--skip-to N] [--gpu 0] [--auto-pack 0|1]
set -euo pipefail

WORK_DIR="${CTCF_WORK_DIR:-}"
ENV_NAME="${CTCF_ENV_NAME:-${CONDA_DEFAULT_ENV:-oasis-ctcf}}"
PATHS_PROFILE=3
DATA_DIR=""
SKIP_TO=0
GPU=0
NUM_WORKERS=8
AUTO_PACK=0

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

# Base config: optimal from R3 + half-res L1 + full-res L3 in ALL runs
BASE="--ds OASIS --max_epoch 100 --schedule_max_epoch 500 --w_reg 1.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 32 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6 --save_ckpt 0 --use_checkpoint 1 --use_tb 0 --l1_half_res 1 --l3_full_res 1"

EXPERIMENTS=(
  # Individual enhancements (each on top of base = half-res L1 + full-res L3)
  "GEN2_01_ITER_L3_N2|$BASE --l3_iters 2"
  "GEN2_02_ITER_L3_N3|$BASE --l3_iters 3"
  "GEN2_03_LEARNED_UP|$BASE --learned_upsample 1"
  "GEN2_04_L2_L3_SKIP|$BASE --l2_l3_skip 1"
  "GEN2_05_L1_L2_SKIP|$BASE --l1_l2_skip 1"

  # L3 zone: all L3 enhancements together
  "GEN2_06_L3_ZONE|$BASE --l3_iters 2 --learned_upsample 1 --l2_l3_skip 1"

  # Full cocktail: everything
  "GEN2_07_COCKTAIL|$BASE --l3_iters 2 --learned_upsample 1 --l2_l3_skip 1 --l1_l2_skip 1"
)

echo "============================================"
echo "  CTCF GEN2 Enhancement Ablation (R5)"
echo "  Base: half-res L1 + full-res L3"
echo "  Ref:  ABL3_01 (Dice=0.8162, 100ep)"
echo "  Data:        $DATA_DIR"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Env:         $ENV_NAME"
echo "  GPU:         $GPU"
echo "============================================"
echo ""

RESULTS_FILE="$WORK_DIR/ablation_5_results.txt"
echo "=== GEN2 Enhancement Ablation ===" > "$RESULTS_FILE"
echo "Base: half-res L1 + full-res L3 (all runs)" >> "$RESULTS_FILE"
echo "Ref: ABL3_01 (Dice=0.8162)" >> "$RESULTS_FILE"
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
echo "  GEN2 Enhancement Ablation Complete!"
echo "============================================"
cat "$RESULTS_FILE"

if [ "$AUTO_PACK" = "1" ]; then
  echo ""
  echo "Packing logs into archive..."
  ARCHIVE_NAME="ctcf_gen2_$(date +%Y%m%d).tar.gz"
  if bash "$WORK_DIR/tools/scripts/package_results.sh" --work-dir "$WORK_DIR" --archive-name "$ARCHIVE_NAME" --exp-glob "GEN2_*"; then
    echo "Archive ready: $WORK_DIR/$ARCHIVE_NAME"
  else
    echo "WARNING: auto packaging failed."
  fi
fi
