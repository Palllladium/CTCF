#!/usr/bin/env bash
# Unified ablation runner for CTCF.
# Reproduces all ablation experiments from the paper (Rounds 1-5).
#
# Usage:
#   bash tools/run_ablation.sh <round> [options]
#
# Rounds:
#   R1   — Loss and strategy variants (5 experiments)
#   R2   — Level 3 and integration tuning (5 experiments)
#   R3   — Level 1 capacity (5 experiments, incl. IXI baseline)
#   R4   — Cascade decomposition (3 experiments)
#   R5   — Resolution scaling (7 experiments)
#   R6   — GEN2.5 capacity ablation (8 experiments)
#   all  — Run R1 through R6 sequentially
#
# Options:
#   --work-dir DIR        Project root (default: auto-detect)
#   --data-dir DIR        Dataset directory (default: $CTCF_DATA_DIR or /data)
#   --paths-profile N     Path profile 1/2/3 (default: 3)
#   --env-name NAME       Conda env (default: $CONDA_DEFAULT_ENV or oasis-ctcf)
#   --gpu N               GPU index (default: 0)
#   --skip-to N           Skip first N experiments in the round
#   --num-workers N       DataLoader workers (default: 8)
#
# Examples:
#   bash tools/run_ablation.sh R4 --data-dir /data --gpu 0
#   bash tools/run_ablation.sh all --paths-profile 3
set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────
ROUND="${1:-}"
if [ -z "$ROUND" ]; then
  echo "Usage: bash tools/run_ablation.sh <R1|R2|R3|R4|R5|R6|all> [options]"
  exit 1
fi
shift

WORK_DIR="${CTCF_WORK_DIR:-}"
ENV_NAME="${CTCF_ENV_NAME:-${CONDA_DEFAULT_ENV:-oasis-ctcf}}"
PATHS_PROFILE=3
DATA_DIR=""
SKIP_TO=0
GPU=0
NUM_WORKERS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)       WORK_DIR="$2";       shift 2 ;;
    --data-dir)       DATA_DIR="$2";       shift 2 ;;
    --paths-profile)  PATHS_PROFILE="$2";  shift 2 ;;
    --env-name)       ENV_NAME="$2";       shift 2 ;;
    --gpu)            GPU="$2";            shift 2 ;;
    --skip-to)        SKIP_TO="$2";        shift 2 ;;
    --num-workers)    NUM_WORKERS="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$WORK_DIR" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [ ! -f "$WORK_DIR/experiments/train_CTCF.py" ]; then
  echo "Not a CTCF repo root: $WORK_DIR"
  exit 1
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${CTCF_DATA_DIR:-/data}"
fi

cd "$WORK_DIR"
export CTCF_DATA_DIR="$DATA_DIR"

# ── Experiment definitions ───────────────────────────────────────────
# Format: "EXP_NAME|flags"
# All rounds use 100 epochs on OASIS with schedule_max_epoch=500.

COMMON="--ds OASIS --max_epoch 100 --schedule_max_epoch 500 --w_reg 1.0 --w_icon 0.05 --w_jac 0.005 --save_ckpt 0 --use_checkpoint 0 --use_tb 0"

define_experiments() {
  local round="$1"
  case "$round" in

    R1)
      # Round 1: Loss and strategy variants.
      # Base: default CTCF (L3=32, absdiff, TS=8).
      local R1_BASE="$COMMON --time_steps 8"
      EXPERIMENTS=(
        "ABL_01_BASELINE|$R1_BASE"
        "ABL_02_WREG05_L1START|$R1_BASE --w_reg 0.5 --l1_from_start 1"
        "ABL_03_ICONL2|$R1_BASE --icon_mode l2"
        "ABL_04_L3_NCC_CH32|$R1_BASE --l3_error_mode ncc"
        "ABL_05_PREALIGN_L1START|$R1_BASE --prealign_encoder 1 --l1_from_start 1"
      )
      ROUND_NAME="R1: Loss and Strategy Variants"
      ;;

    R2)
      # Round 2: Level 3 and integration tuning.
      # Base: R1 winner (default + L1 ICON).
      local R2_BASE="$COMMON --time_steps 8"
      EXPERIMENTS=(
        "ABL2_01_L3_NCC_CH64|$R2_BASE --l3_error_mode ncc --l3_base_ch 64"
        "ABL2_02_L3_NCC_CH64_TS6|$R2_BASE --l3_error_mode ncc --l3_base_ch 64 --time_steps 6"
        "ABL2_03_L3_NCC_CH64_TS6_DROP01|$R2_BASE --l3_error_mode ncc --l3_base_ch 64 --time_steps 6 --drop_path_rate 0.1"
        "ABL2_04_DROP01_QKVT|$R2_BASE --l3_error_mode ncc --l3_base_ch 64 --time_steps 6 --drop_path_rate 0.1 --qkv_bias_learnable 1"
        "ABL2_05_FULL_COMBO|$R2_BASE --l3_error_mode ncc --l3_base_ch 64 --time_steps 6 --drop_path_rate 0.1 --qkv_bias_learnable 1 --l1_from_start 1"
      )
      ROUND_NAME="R2: Level 3 and Integration Tuning"
      ;;

    R3)
      # Round 3: Level 1 capacity.
      # Base: R2 winner (L3=64, NCC, TS6).
      local R3_BASE="$COMMON --l3_base_ch 64 --l3_error_mode ncc --time_steps 6"
      EXPERIMENTS=(
        "ABL3_01_L1CH32_L3CH64_TS6|$R3_BASE --l1_base_ch 32"
        "ABL3_02_L1CH64_L3CH64_TS6|$R3_BASE --l1_base_ch 64"
        "ABL3_03_L1CH64_L3CH64_TS4|$R3_BASE --l1_base_ch 64 --time_steps 4"
        "ABL3_04_L1CH64_L3CH64_TS6_WREG05|$R3_BASE --l1_base_ch 64 --w_reg 0.5"
        "ABL3_05_CTCF_IXI_BEST|--ds IXI --max_epoch 100 --w_icon 0.05 --w_jac 0.005 --l1_base_ch 64 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6 --save_ckpt 0 --use_checkpoint 0 --use_tb 0"
      )
      ROUND_NAME="R3: Level 1 Capacity"
      ;;

    R4)
      # Round 4: Cascade decomposition.
      # Isolates contribution of each cascade level.
      local R4_BASE="$COMMON --l1_base_ch 32 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6"
      EXPERIMENTS=(
        "ABL4_01_L2_ONLY|$R4_BASE --disable_l1 1 --disable_l3 1"
        "ABL4_02_L1_L2|$R4_BASE --disable_l3 1"
        "ABL4_03_L2_L3|$R4_BASE --disable_l1 1"
      )
      ROUND_NAME="R4: Cascade Decomposition"
      ;;

    R5)
      # Round 5: Resolution scaling.
      # Half-res L1 + full-res L3, various enhancements.
      local R5_BASE="$COMMON --l1_base_ch 32 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6 --use_checkpoint 1 --l1_half_res 1 --l3_full_res 1"
      EXPERIMENTS=(
        "GEN2_01_ITER_L3_N2|$R5_BASE --l3_iters 2"
        "GEN2_02_ITER_L3_N3|$R5_BASE --l3_iters 3"
        "GEN2_03_LEARNED_UP|$R5_BASE --learned_upsample 1"
        "GEN2_04_L2_L3_SKIP|$R5_BASE --l2_l3_skip 1"
        "GEN2_05_L1_L2_SKIP|$R5_BASE --l1_l2_skip 1"
        "GEN2_06_L3_ZONE|$R5_BASE --l3_iters 2 --learned_upsample 1 --l2_l3_skip 1"
        "GEN2_07_COCKTAIL|$R5_BASE --l3_iters 2 --learned_upsample 1 --l2_l3_skip 1 --l1_l2_skip 1"
      )
      ROUND_NAME="R5: Resolution Scaling"
      ;;

    R6)
      # Round 6: GEN2.5 Capacity ablation.
      # Tests the effect of increasing L1/L3 parameter capacity.
      # All runs use half-res L1 + full-res L3 as base (proven in R5).
      # Reference: ABL3_01 (Dice=0.8162, 100ep, default L1=32ch, L3=64ch)
      local R6_BASE="$COMMON --l1_base_ch 32 --l3_base_ch 64 --l3_error_mode ncc --time_steps 6 --use_checkpoint 1 --l1_half_res 1 --l3_full_res 1"
      EXPERIMENTS=(
        "CAP_01_L3_CH128|$R6_BASE --l3_base_ch 128"
        "CAP_02_L3_CH128_CAB|$R6_BASE --l3_base_ch 128 --l3_cab 1"
        "CAP_03_L3_CH128_CTX|$R6_BASE --l3_base_ch 128 --l3_context 2"
        "CAP_04_L3_CH128_GATE|$R6_BASE --l3_base_ch 128 --l3_gate 1"
        "CAP_05_L1_CH64|$R6_BASE --l1_base_ch 64"
        "CAP_06_L1_CH64_CAB|$R6_BASE --l1_base_ch 64 --l1_cab 1"
        "CAP_07_REBALANCE|$R6_BASE --l1_base_ch 64 --l3_base_ch 128"
        "CAP_08_COCKTAIL|$R6_BASE --l1_base_ch 64 --l1_cab 1 --l3_base_ch 128 --l3_cab 1 --l3_gate 1"
      )
      ROUND_NAME="R6: GEN2.5 Capacity"
      ;;

    *)
      echo "Unknown round: $round (expected R1-R6 or all)"
      exit 1
      ;;
  esac
}

# ── Execution loop ───────────────────────────────────────────────────

run_round() {
  local round="$1"
  define_experiments "$round"

  echo ""
  echo "============================================"
  echo "  $ROUND_NAME"
  echo "  Data:        $DATA_DIR"
  echo "  Experiments: ${#EXPERIMENTS[@]}"
  echo "  Env:         $ENV_NAME"
  echo "  GPU:         $GPU"
  echo "============================================"

  for i in "${!EXPERIMENTS[@]}"; do
    if [ "$i" -lt "$SKIP_TO" ]; then
      echo "[SKIP] Experiment $i"
      continue
    fi

    IFS='|' read -r EXP_NAME EXTRA_ARGS <<< "${EXPERIMENTS[$i]}"
    EXP_NAME=$(echo "$EXP_NAME" | xargs)
    EXTRA_ARGS=$(echo "$EXTRA_ARGS" | xargs)

    echo ""
    echo "--------------------------------------------"
    echo "  [$((i+1))/${#EXPERIMENTS[@]}] $EXP_NAME"
    echo "  Started: $(date)"
    echo "--------------------------------------------"

    START_TIME=$(date +%s)
    mkdir -p "logs/${EXP_NAME}"

    conda run -n "$ENV_NAME" --no-capture-output \
      python -m experiments.train_CTCF \
        --$PATHS_PROFILE \
        --exp "$EXP_NAME" \
        --gpu "$GPU" \
        --num_workers "$NUM_WORKERS" \
        $EXTRA_ARGS || {
      echo "WARNING: $EXP_NAME failed (exit $?)"
      continue
    }

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))

    LOGFILE="logs/${EXP_NAME}/logfile.log"
    if [ -f "$LOGFILE" ]; then
      BEST_DICE=$(grep -oP 'val_dice=\K[0-9.]+' "$LOGFILE" | sort -rn | head -1 || echo "N/A")
      echo "  Done in ${ELAPSED}s — Best Dice: $BEST_DICE"
    else
      echo "  Done in ${ELAPSED}s"
    fi
  done

  echo ""
  echo "  $ROUND_NAME — complete."
}

# ── Main ─────────────────────────────────────────────────────────────

if [ "$ROUND" = "all" ]; then
  for r in R1 R2 R3 R4 R5 R6; do
    SKIP_TO=0  # reset skip-to for each round
    run_round "$r"
  done
else
  ROUND=$(echo "$ROUND" | tr '[:lower:]' '[:upper:]')
  run_round "$ROUND"
fi

echo ""
echo "============================================"
echo "  All done!"
echo "============================================"
