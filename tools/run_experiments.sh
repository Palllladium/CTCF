#!/usr/bin/env bash
# Experiment runner for CTCF ablation study on a remote GPU machine.
# Usage:
#   bash tools/run_experiments.sh [--data-dir /data] [--skip-to N]
#                                 [--save-ckpt 0|1] [--use-tb 0|1]
#                                 [--tmux-session NAME]
set -euo pipefail

WORK_DIR="$HOME/CTCF"
ENV_NAME="oasis-ctcf"
PATHS_PROFILE=3
DATA_DIR=""
SKIP_TO=0
SAVE_CKPT=0
USE_TB=0
USE_CHECKPOINT=0
GPU=0
NUM_WORKERS=8
TIME_STEPS=8
TMUX_SESSION=""
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --skip-to) SKIP_TO="$2"; shift 2 ;;
    --save-ckpt) SAVE_CKPT="$2"; shift 2 ;;
    --use-tb) USE_TB="$2"; shift 2 ;;
    --use-checkpoint) USE_CHECKPOINT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --time-steps) TIME_STEPS="$2"; shift 2 ;;
    --tmux-session) TMUX_SESSION="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$SAVE_CKPT" != "0" && "$SAVE_CKPT" != "1" ]]; then
  echo "--save-ckpt must be 0 or 1"
  exit 1
fi
if [[ "$USE_TB" != "0" && "$USE_TB" != "1" ]]; then
  echo "--use-tb must be 0 or 1"
  exit 1
fi
if [[ "$USE_CHECKPOINT" != "0" && "$USE_CHECKPOINT" != "1" ]]; then
  echo "--use-checkpoint must be 0 or 1"
  exit 1
fi

if [ -n "$TMUX_SESSION" ] && [ -z "${TMUX:-}" ] && [ "${RUN_EXPERIMENTS_IN_TMUX:-0}" != "1" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed, cannot start detached session '$TMUX_SESSION'"
    exit 1
  fi
  CMD="cd \"$WORK_DIR\" && RUN_EXPERIMENTS_IN_TMUX=1 bash tools/run_experiments.sh"
  for a in "${ORIG_ARGS[@]}"; do
    CMD+=" $(printf "%q" "$a")"
  done
  tmux new-session -d -s "$TMUX_SESSION" "$CMD"
  echo "Started tmux session: $TMUX_SESSION"
  echo "Attach with: tmux attach -t $TMUX_SESSION"
  exit 0
fi

if [ -z "$DATA_DIR" ] && [ -f "$WORK_DIR/.env_data" ]; then
  source "$WORK_DIR/.env_data"
  DATA_DIR="${DATA_DIR:-}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/data"
  echo "No --data-dir specified, using default: $DATA_DIR"
fi

cd "$WORK_DIR"
export CTCF_DATA_DIR="$DATA_DIR"

# Format: EXP_NAME|EXTRA_ARGS
EXPERIMENTS=(
  "ABL_01_BASELINE|--max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005"
  "ABL_02_WREG05_L1START|--max_epoch 100 --w_reg 0.5 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_from_start 1"
  "ABL_03_ICONL2|--max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.1 --w_jac 0.005 --icon_mode l2"
  "ABL_04_L3_NCC_CH32|--max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l3_base_ch 32 --l3_error_mode ncc"
  "ABL_05_PREALIGN_L1START|--max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_from_start 1 --prealign_encoder 1"
)

echo "============================================"
echo "  CTCF Ablation Study Runner"
echo "  Data:           $DATA_DIR"
echo "  Experiments:    ${#EXPERIMENTS[@]}"
echo "  Skip to:        $SKIP_TO"
echo "  save_ckpt:      $SAVE_CKPT"
echo "  use_tb:         $USE_TB"
echo "  use_checkpoint: $USE_CHECKPOINT"
echo "============================================"
echo ""

RESULTS_FILE="$WORK_DIR/ablation_results.txt"
echo "=== Ablation Results ===" > "$RESULTS_FILE"
echo "Started: $(date)" >> "$RESULTS_FILE"
echo "Config: save_ckpt=$SAVE_CKPT use_tb=$USE_TB use_checkpoint=$USE_CHECKPOINT" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for i in "${!EXPERIMENTS[@]}"; do
  if [ "$i" -lt "$SKIP_TO" ]; then
    echo "[SKIP] Experiment $i (--skip-to $SKIP_TO)"
    continue
  fi

  IFS='|' read -r EXP_NAME EXTRA_ARGS <<< "${EXPERIMENTS[$i]}"
  EXP_NAME=$(echo "$EXP_NAME" | xargs)
  EXTRA_ARGS=$(echo "$EXTRA_ARGS" | xargs)

  echo ""
  echo "============================================"
  echo "  [$i/${#EXPERIMENTS[@]}] $EXP_NAME"
  echo "  Args: $EXTRA_ARGS"
  echo "  Started: $(date)"
  echo "============================================"

  START_TIME=$(date +%s)
  mkdir -p "logs/${EXP_NAME}"

  conda run -n "$ENV_NAME" --no-capture-output \
    python -m experiments.train_CTCF \
      --ds OASIS \
      --$PATHS_PROFILE \
      --exp "$EXP_NAME" \
      --time_steps "$TIME_STEPS" \
      --save_ckpt "$SAVE_CKPT" \
      --use_checkpoint "$USE_CHECKPOINT" \
      --use_tb "$USE_TB" \
      --gpu "$GPU" \
      --num_workers "$NUM_WORKERS" \
      $EXTRA_ARGS || {
      echo "WARNING: Experiment $EXP_NAME failed with exit code $?"
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
echo "  All experiments finished!"
echo "  Results summary: $RESULTS_FILE"
echo "============================================"
cat "$RESULTS_FILE"
