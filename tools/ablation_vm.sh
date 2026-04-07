#!/usr/bin/env bash
# Phase 1 ablation: VoxelMorph as dev backbone for CTCF cascade.
#
# Runs all experiments from ablation_plan_v2.md Phase 1 + full-res VM baseline.
# Each experiment logs to logs/<EXP_NAME>/logfile.log (--quiet suppresses console spam).
#
# Usage:
#   bash tools/ablation_vm.sh [options]
#
# Options:
#   --paths-profile N   Path profile 1/2/3 (default: 1)
#   --gpu N             GPU index (default: 0)
#   --max-epoch N       Training epochs (default: 100)
#   --skip-to N         Skip first N experiments (default: 0)
#   --ds DATASET        OASIS or IXI (default: OASIS)
#   --dry-run           Print commands without executing
#   --no-quiet          Show full training logs in console
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
PATHS_PROFILE=1
GPU=0
MAX_EPOCH=100
SKIP_TO=0
DS=OASIS
DRY_RUN=0
QUIET=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paths-profile)  PATHS_PROFILE="$2"; shift 2 ;;
    --gpu)            GPU="$2";           shift 2 ;;
    --max-epoch)      MAX_EPOCH="$2";     shift 2 ;;
    --skip-to)        SKIP_TO="$2";       shift 2 ;;
    --ds)             DS="$2";            shift 2 ;;
    --dry-run)        DRY_RUN=1;          shift ;;
    --no-quiet)       QUIET=0;            shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"

# ── Auto-detect w_reg ────────────────────────────────────────────────
if [ "$DS" = "IXI" ]; then
  W_REG=4.0
else
  W_REG=1.0
fi

# ── Shared flags ─────────────────────────────────────────────────────
COMMON_CTCF="--ds $DS --$PATHS_PROFILE --gpu $GPU --max_epoch $MAX_EPOCH --quiet $QUIET --use_tb 0 --save_ckpt 1 --w_reg $W_REG --w_icon 0.05 --w_jac 0.005"
COMMON_VM="--ds $DS --$PATHS_PROFILE --gpu $GPU --max_epoch $MAX_EPOCH --quiet $QUIET --use_tb 0 --save_ckpt 1 --w_reg $W_REG"
CASCADE="--config CTCF-CascadeA-VM"
SOLO="--config CTCF-VM-solo"

# ── Experiment table ─────────────────────────────────────────────────
# Format: "EXP_NAME|SCRIPT|FLAGS"
#   SCRIPT: ctcf = train_CTCF.py, vm = train_VoxelMorph.py

EXPERIMENTS=(
  # --- Baselines ---
  "P1_00_VM_FULLRES|vm|--config VxmDense --exp P1_00_VM_FULLRES $COMMON_VM"
  "P1_01_VM_SOLO|ctcf|$SOLO --exp P1_01_VM_SOLO $COMMON_CTCF"
  "P1_02_VM_CASCADE|ctcf|$CASCADE --exp P1_02_VM_CASCADE $COMMON_CTCF"

  # --- L3 iterations ---
  "P1_03_L3_ITER2|ctcf|$CASCADE --exp P1_03_L3_ITER2 $COMMON_CTCF --l3_iters 2"
  "P1_04_L3_ITER2_UNSHARED|ctcf|$CASCADE --exp P1_04_L3_ITER2_UNSHARED $COMMON_CTCF --l3_iters 2 --l3_unshared 1"

  # --- L3 enhancements (isolated) ---
  "P1_05_L3_GATE|ctcf|$CASCADE --exp P1_05_L3_GATE $COMMON_CTCF --l3_gate 1"
  "P1_06_L3_CAB|ctcf|$CASCADE --exp P1_06_L3_CAB $COMMON_CTCF --l3_cab 1"
  "P1_07_L3_CTX2|ctcf|$CASCADE --exp P1_07_L3_CTX2 $COMMON_CTCF --l3_context 2"
  "P1_08_L3_CH128|ctcf|$CASCADE --exp P1_08_L3_CH128 $COMMON_CTCF --l3_base_ch 128"

  # --- L1 enhancements ---
  "P1_09_PREALIGN|ctcf|$CASCADE --exp P1_09_PREALIGN $COMMON_CTCF --prealign_encoder 1"
  "P1_10_L1_CAB|ctcf|$CASCADE --exp P1_10_L1_CAB $COMMON_CTCF --l1_cab 1"

  # --- Combos ---
  "P1_11_ITER2_GATE|ctcf|$CASCADE --exp P1_11_ITER2_GATE $COMMON_CTCF --l3_iters 2 --l3_gate 1"
  "P1_12_BEST_COMBO|ctcf|$CASCADE --exp P1_12_BEST_COMBO $COMMON_CTCF --l3_iters 2 --l3_gate 1 --l3_cab 1 --l3_base_ch 128"

  # --- Cascade decomposition (VM version of R4) ---
  "P1_13_L2_ONLY|ctcf|$CASCADE --exp P1_13_L2_ONLY $COMMON_CTCF --disable_l1 1 --disable_l3 1"
  "P1_14_L1_L2|ctcf|$CASCADE --exp P1_14_L1_L2 $COMMON_CTCF --disable_l3 1"
  "P1_15_L2_L3|ctcf|$CASCADE --exp P1_15_L2_L3 $COMMON_CTCF --disable_l1 1"
)

TOTAL=${#EXPERIMENTS[@]}

# ── Header ───────────────────────────────────────────────────────────
echo "============================================"
echo "  Phase 1: VoxelMorph Ablation"
echo "  Dataset:     $DS"
echo "  Epochs:      $MAX_EPOCH"
echo "  Experiments: $TOTAL"
echo "  GPU:         $GPU"
echo "  Quiet:       $QUIET"
echo "  Logs dir:    logs/"
echo "============================================"

PASSED=0
FAILED=0
SKIPPED=0

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r EXP_NAME SCRIPT FLAGS <<< "${EXPERIMENTS[$i]}"
  EXP_NAME=$(echo "$EXP_NAME" | xargs)
  SCRIPT=$(echo "$SCRIPT" | xargs)
  FLAGS=$(echo "$FLAGS" | xargs)

  IDX=$((i + 1))

  if [ "$i" -lt "$SKIP_TO" ]; then
    echo "[$IDX/$TOTAL] SKIP $EXP_NAME"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Resolve python module
  if [ "$SCRIPT" = "vm" ]; then
    MODULE="experiments.train_VoxelMorph"
  else
    MODULE="experiments.train_CTCF"
  fi

  echo ""
  echo "--------------------------------------------"
  echo "[$IDX/$TOTAL] $EXP_NAME"
  echo "  module: $MODULE"
  echo "  started: $(date '+%H:%M:%S')"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [DRY-RUN] python -m $MODULE $FLAGS"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  START_SEC=$(date +%s)

  if python -m "$MODULE" $FLAGS; then
    END_SEC=$(date +%s)
    ELAPSED=$(( END_SEC - START_SEC ))

    # Extract best dice from log
    LOGFILE="logs/${EXP_NAME}/logfile.log"
    if [ -f "$LOGFILE" ]; then
      BEST_LINE=$(grep "best=" "$LOGFILE" | tail -1 || true)
      if [ -n "$BEST_LINE" ]; then
        echo "  $BEST_LINE"
      fi
    fi
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))
    echo "  finished in ${MINS}m${SECS}s"
    PASSED=$((PASSED + 1))
  else
    echo "  FAILED (exit $?)"
    FAILED=$((FAILED + 1))
  fi
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Phase 1 complete"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "============================================"

# Print results table from logs
echo ""
echo "Results summary:"
printf "%-30s %8s %8s\n" "Experiment" "Dice" "J<=0%"
printf "%-30s %8s %8s\n" "------------------------------" "--------" "--------"

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r EXP_NAME _ _ <<< "${EXPERIMENTS[$i]}"
  EXP_NAME=$(echo "$EXP_NAME" | xargs)
  LOGFILE="logs/${EXP_NAME}/logfile.log"
  if [ -f "$LOGFILE" ]; then
    BEST_DSC=$(grep -oP 'best=\K[0-9.]+' "$LOGFILE" | tail -1 || echo "—")
    BEST_JAC=$(grep -oP 'j<=0%=\K[0-9.]+' "$LOGFILE" | tail -1 || echo "—")
    printf "%-30s %8s %8s\n" "$EXP_NAME" "$BEST_DSC" "$BEST_JAC"
  fi
done
