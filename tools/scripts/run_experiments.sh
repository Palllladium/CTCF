#!/usr/bin/env bash
# Experiment runner for CTCF ablation study on a remote GPU machine.
# Usage:
#   bash tools/scripts/run_experiments.sh [--work-dir ~/CTCF] [--data-dir /data] [--skip-to N]
#                                         [--save-ckpt 0|1] [--use-tb 0|1]
#                                         [--schedule-epochs 500] [--paths-profile 1|2|3]
#                                         [--env-name NAME] [--tmux-session NAME] [--auto-pack 0|1]
set -euo pipefail

WORK_DIR="${CTCF_WORK_DIR:-}"
ENV_NAME="${CTCF_ENV_NAME:-${CONDA_DEFAULT_ENV:-oasis-ctcf}}"
PATHS_PROFILE=3
DATA_DIR=""
SKIP_TO=0
SAVE_CKPT=0
USE_TB=0
USE_CHECKPOINT=0
GPU=0
NUM_WORKERS=8
TIME_STEPS=8
SCHEDULE_EPOCHS=500
TMUX_SESSION=""
AUTO_PACK=1
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --skip-to) SKIP_TO="$2"; shift 2 ;;
    --save-ckpt) SAVE_CKPT="$2"; shift 2 ;;
    --use-tb) USE_TB="$2"; shift 2 ;;
    --use-checkpoint) USE_CHECKPOINT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --time-steps) TIME_STEPS="$2"; shift 2 ;;
    --schedule-epochs) SCHEDULE_EPOCHS="$2"; shift 2 ;;
    --paths-profile) PATHS_PROFILE="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --tmux-session) TMUX_SESSION="$2"; shift 2 ;;
    --auto-pack) AUTO_PACK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$WORK_DIR" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
if [ ! -d "$WORK_DIR" ]; then
  echo "Work dir not found: $WORK_DIR"
  exit 1
fi
if [ ! -f "$WORK_DIR/experiments/train_CTCF.py" ]; then
  echo "Not a CTCF repo root (missing experiments/train_CTCF.py): $WORK_DIR"
  exit 1
fi

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
if [[ "$AUTO_PACK" != "0" && "$AUTO_PACK" != "1" ]]; then
  echo "--auto-pack must be 0 or 1"
  exit 1
fi
if [[ "$PATHS_PROFILE" != "1" && "$PATHS_PROFILE" != "2" && "$PATHS_PROFILE" != "3" ]]; then
  echo "--paths-profile must be 1, 2, or 3"
  exit 1
fi

if [ -n "$TMUX_SESSION" ] && [ -z "${TMUX:-}" ] && [ "${RUN_EXPERIMENTS_IN_TMUX:-0}" != "1" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed, cannot start detached session '$TMUX_SESSION'"
    exit 1
  fi
  CMD="cd \"$WORK_DIR\" && RUN_EXPERIMENTS_IN_TMUX=1 bash tools/scripts/run_experiments.sh"
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

if ! conda run -n "$ENV_NAME" --no-capture-output python -V >/dev/null 2>&1; then
  echo "Conda environment '$ENV_NAME' not found or not runnable."
  echo "Pass valid env with --env-name <name>."
  echo "Available envs:"
  conda env list || true
  exit 1
fi

# Format: EXP_NAME|EXTRA_ARGS
# Round 3: L1 strengthening + TS4 + IXI baseline (final round before longruns)
# Base: L3_CH64 + NCC + TS6 (winner of Round 2)
EXPERIMENTS=(
  "ABL3_01_L1CH32_L3CH64_TS6|--ds OASIS --max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 32 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6"
  "ABL3_02_L1CH64_L3CH64_TS6|--ds OASIS --max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 64 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6"
  "ABL3_03_L1CH64_L3CH64_TS4|--ds OASIS --max_epoch 100 --w_reg 1.0 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 64 --l3_base_ch 64 --l3_error_mode ncc --time_steps 4"
  "ABL3_04_L1CH64_L3CH64_TS6_WREG05|--ds OASIS --max_epoch 100 --w_reg 0.5 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 64 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6"
  "ABL3_05_CTCF_IXI_BEST|--ds IXI --max_epoch 100 --w_cyc 0.0 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 64 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6"
)

echo "============================================"
echo "  CTCF Ablation Study Runner"
echo "  Data:           $DATA_DIR"
echo "  Experiments:    ${#EXPERIMENTS[@]}"
echo "  Skip to:        $SKIP_TO"
echo "  save_ckpt:      $SAVE_CKPT"
echo "  use_tb:         $USE_TB"
echo "  use_checkpoint: $USE_CHECKPOINT"
echo "  schedule_epoch: $SCHEDULE_EPOCHS"
echo "  paths_profile:  $PATHS_PROFILE"
echo "  env_name:       $ENV_NAME"
echo "  auto_pack:      $AUTO_PACK"
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
      --schedule_max_epoch "$SCHEDULE_EPOCHS" \
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

if [ "$AUTO_PACK" = "1" ]; then
  echo ""
  echo "Packing logs into archive..."
  ARCHIVE_NAME="ctcf_abl_$(date +%Y%m%d).tar.gz"
  if bash "$WORK_DIR/tools/scripts/package_results.sh" --work-dir "$WORK_DIR" --archive-name "$ARCHIVE_NAME" --exp-glob "ABL3_*"; then
    echo "Archive ready: $WORK_DIR/$ARCHIVE_NAME"
  else
    echo "WARNING: auto packaging failed. You can retry manually:"
    echo "  bash tools/scripts/package_results.sh --work-dir $WORK_DIR --exp-glob 'ABL3_*'"
  fi
fi
