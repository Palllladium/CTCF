#!/usr/bin/env bash
# Cross-dataset zero-shot inference runner (native + cross on OASIS & IXI).
#
# For each (model, ckpt_dataset) pair we run inference twice:
#   - on the native dataset (sanity check, should reproduce published numbers)
#   - on the other dataset with zero fine-tuning (generalization measurement)
#
# Outputs:
#   results/infer/<DS>/<model>/<ckpt-stem>/{per_case.csv, summary.json, summary.csv}
#   results/infer/cross_dataset_summary.csv  (aggregated final table)
#
# Usage:
#   bash tools/cross_dataset_eval.sh --paths-profile 3 --gpu 0
#   bash tools/cross_dataset_eval.sh --paths-profile 2 --gpu 0 --dry-run
#
# Checkpoint resolution:
#   1) Each experiment is associated with a *folder* under $RESULTS_ROOT.
#   2) The script then picks best.pth -> best.pth.tar -> last.pth -> last.pth.tar.
#   3) Either the folder name (EXP_*) or the full checkpoint path (CKPT_*) may
#      be overridden via env vars; CKPT_* wins when both are set.
#
# Defaults are set to the advisor's layout:
#   RESULTS_ROOT=/home/roman/P/CTCF/results
#   EXP_CTCF_OASIS=CTCF_UPD_OASIS_E500
#   EXP_CTCF_IXI=CTCF_IXI_TUNED
#   EXP_TMDCA_OASIS=TM_DCA_unsup_OASIS
#   EXP_TMDCA_IXI=TM_DCA_IXI
#   EXP_UTSR_OASIS=UTSRMorph_OASIS
#   EXP_UTSR_IXI=UTSR_IXI_WREG4_E500
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
PATHS_PROFILE=1
GPU=0
DRY_RUN=0
SKIP_EXISTING=0
RESULTS_ROOT_DEFAULT="/home/roman/P/CTCF/results"
PREFER_DEFAULT="best"   # best | last

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paths-profile)   PATHS_PROFILE="$2";         shift 2 ;;
    --gpu)             GPU="$2";                   shift 2 ;;
    --results-root)    RESULTS_ROOT_DEFAULT="$2";  shift 2 ;;
    --prefer)          PREFER_DEFAULT="$2";        shift 2 ;;
    --dry-run)         DRY_RUN=1;                  shift ;;
    --skip-existing)   SKIP_EXISTING=1;            shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-$RESULTS_ROOT_DEFAULT}"
PREFER="${PREFER:-$PREFER_DEFAULT}"

# Experiment folder defaults match advisor's layout on /home/roman/P/CTCF/results/
EXP_CTCF_OASIS="${EXP_CTCF_OASIS:-CTCF_UPD_OASIS_E500}"
EXP_CTCF_IXI="${EXP_CTCF_IXI:-CTCF_IXI_TUNED}"
EXP_TMDCA_OASIS="${EXP_TMDCA_OASIS:-TM_DCA_unsup_OASIS}"
EXP_TMDCA_IXI="${EXP_TMDCA_IXI:-TM_DCA_IXI}"
EXP_UTSR_OASIS="${EXP_UTSR_OASIS:-UTSRMorph_OASIS}"
EXP_UTSR_IXI="${EXP_UTSR_IXI:-UTSR_IXI_WREG4_E500}"

# ── Checkpoint resolver ──────────────────────────────────────────────
# Picks the first existing candidate inside an experiment folder.
# Search order depends on $PREFER: either best-first or last-first.
# Looks in the folder root AND in ./ckpt/ (the latter matches train_runtime.py layout).
resolve_ckpt() {
  local folder="$1"
  local -a candidates
  if [ "$PREFER" = "last" ]; then
    candidates=(last.pth last.pth.tar best.pth best.pth.tar)
  else
    candidates=(best.pth best.pth.tar last.pth last.pth.tar)
  fi
  for sub in "" "ckpt/"; do
    for name in "${candidates[@]}"; do
      local p="$folder/$sub$name"
      if [ -f "$p" ]; then
        echo "$p"
        return 0
      fi
    done
  done
  echo ""   # not found
  return 1
}

# Either user supplied an explicit CKPT_* path, or resolve from the folder.
CKPT_CTCF_OASIS="${CKPT_CTCF_OASIS:-$(resolve_ckpt "$RESULTS_ROOT/$EXP_CTCF_OASIS" || true)}"
CKPT_CTCF_IXI="${CKPT_CTCF_IXI:-$(resolve_ckpt "$RESULTS_ROOT/$EXP_CTCF_IXI" || true)}"
CKPT_TMDCA_OASIS="${CKPT_TMDCA_OASIS:-$(resolve_ckpt "$RESULTS_ROOT/$EXP_TMDCA_OASIS" || true)}"
CKPT_TMDCA_IXI="${CKPT_TMDCA_IXI:-$(resolve_ckpt "$RESULTS_ROOT/$EXP_TMDCA_IXI" || true)}"
CKPT_UTSR_OASIS="${CKPT_UTSR_OASIS:-$(resolve_ckpt "$RESULTS_ROOT/$EXP_UTSR_OASIS" || true)}"
CKPT_UTSR_IXI="${CKPT_UTSR_IXI:-$(resolve_ckpt "$RESULTS_ROOT/$EXP_UTSR_IXI" || true)}"

# ── Experiment matrix ────────────────────────────────────────────────
# Fields: model | ckpt_path | ckpt_ds | eval_ds | config_key | extra_flags
EXPERIMENTS=(
  # CTCF: one config covers both datasets
  "ctcf|$CKPT_CTCF_OASIS|OASIS|OASIS|CTCF-CascadeA|"
  "ctcf|$CKPT_CTCF_OASIS|OASIS|IXI|CTCF-CascadeA|--use_test"
  "ctcf|$CKPT_CTCF_IXI|IXI|IXI|CTCF-CascadeA|--use_test"
  "ctcf|$CKPT_CTCF_IXI|IXI|OASIS|CTCF-CascadeA|"

  # TM-DCA: single config
  "tm-dca|$CKPT_TMDCA_OASIS|OASIS|OASIS|TransMorph-3-LVL|"
  "tm-dca|$CKPT_TMDCA_OASIS|OASIS|IXI|TransMorph-3-LVL|--use_test"
  "tm-dca|$CKPT_TMDCA_IXI|IXI|IXI|TransMorph-3-LVL|--use_test"
  "tm-dca|$CKPT_TMDCA_IXI|IXI|OASIS|TransMorph-3-LVL|"

  # UTSRMorph: config MUST match the checkpoint architecture (Large vs IXI-Large differ in embed dim)
  "utsrmorph|$CKPT_UTSR_OASIS|OASIS|OASIS|UTSRMorph-Large|"
  "utsrmorph|$CKPT_UTSR_OASIS|OASIS|IXI|UTSRMorph-Large|--use_test"
  "utsrmorph|$CKPT_UTSR_IXI|IXI|IXI|UTSRMorph-IXI-Large|--use_test"
  "utsrmorph|$CKPT_UTSR_IXI|IXI|OASIS|UTSRMorph-IXI-Large|"
)

# ── Header ───────────────────────────────────────────────────────────
TOTAL=${#EXPERIMENTS[@]}
echo "=========================================================="
echo "  Cross-dataset zero-shot evaluation"
echo "  paths profile : $PATHS_PROFILE"
echo "  gpu           : $GPU"
echo "  results root  : $RESULTS_ROOT"
echo "  prefer ckpt   : $PREFER"
echo "  experiments   : $TOTAL"
echo "  dry-run       : $DRY_RUN"
echo "=========================================================="

# Validate that all required ckpt files exist (unless dry-run)
echo ""
echo "Checking checkpoints:"
MISSING=0
declare -A SEEN
for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ CKPT_PATH _ _ _ _ <<< "$spec"
  if [[ -z "${SEEN[$CKPT_PATH]:-}" ]]; then
    SEEN[$CKPT_PATH]=1
    if [ -n "$CKPT_PATH" ] && [ -f "$CKPT_PATH" ]; then
      echo "  [OK]   $CKPT_PATH"
    else
      echo "  [MISS] ${CKPT_PATH:-<not resolved>}"
      MISSING=$((MISSING + 1))
    fi
  fi
done
if [ "$MISSING" -gt 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "ERROR: $MISSING checkpoint(s) missing."
  echo "       Either set RESULTS_ROOT / EXP_* env vars to point at experiment folders,"
  echo "       or override CKPT_* env vars with full paths. Abort."
  exit 1
fi

# ── Run loop ─────────────────────────────────────────────────────────
PASSED=0; FAILED=0; SKIPPED=0

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r MODEL CKPT_PATH CKPT_DS EVAL_DS CONFIG_KEY EXTRA <<< "${EXPERIMENTS[$i]}"
  IDX=$((i + 1))

  # Tag encodes which ckpt dataset and which eval dataset.
  # Use the experiment folder name (or its parent if ckpt lives in ckpt/ subdir),
  # so OASIS/IXI runs of the same model don't collide on "best.pth".
  CKPT_PARENT="$(dirname "$CKPT_PATH")"
  if [ "$(basename "$CKPT_PARENT")" = "ckpt" ]; then
    CKPT_TAG="$(basename "$(dirname "$CKPT_PARENT")")"
  else
    CKPT_TAG="$(basename "$CKPT_PARENT")"
  fi
  OUT_DIR="results/infer/${EVAL_DS}/${MODEL}/${CKPT_TAG}"
  TAG="[${CKPT_DS}->${EVAL_DS}]"

  echo ""
  echo "----------------------------------------------------------"
  echo "[$IDX/$TOTAL] $MODEL $TAG"
  echo "  ckpt   : $CKPT_PATH"
  echo "  config : $CONFIG_KEY"
  echo "  eval   : $EVAL_DS (profile $PATHS_PROFILE)"
  echo "  out    : $OUT_DIR"
  echo "  start  : $(date '+%H:%M:%S')"

  # Choose the right model-specific config flag
  case "$MODEL" in
    ctcf)      CFG_FLAG="--ctcf_config $CONFIG_KEY" ;;
    tm-dca)    CFG_FLAG="--tm_config $CONFIG_KEY"   ;;
    utsrmorph) CFG_FLAG="--utsr_config $CONFIG_KEY" ;;
    *) echo "Unknown model: $MODEL"; exit 1 ;;
  esac

  CMD="python -m experiments.inference \
    --model $MODEL \
    --ckpt $CKPT_PATH \
    --ds $EVAL_DS \
    --$PATHS_PROFILE \
    --gpu $GPU \
    --strict_ckpt 0 \
    --hd95 \
    --print_every 5 \
    $CFG_FLAG \
    $EXTRA"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [DRY-RUN] $CMD"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "$OUT_DIR/summary.json" ]; then
    echo "  [SKIP] summary.json already exists"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  START_SEC=$(date +%s)
  if $CMD; then
    END_SEC=$(date +%s); ELAPSED=$((END_SEC - START_SEC))
    MINS=$((ELAPSED / 60)); SECS=$((ELAPSED % 60))
    echo "  finished in ${MINS}m${SECS}s"
    PASSED=$((PASSED + 1))
  else
    echo "  FAILED (exit $?)"
    FAILED=$((FAILED + 1))
  fi
done

# ── Aggregate ────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  Done. Passed: $PASSED, Failed: $FAILED, Skipped: $SKIPPED"
echo "=========================================================="

if [ "$DRY_RUN" -eq 0 ] && [ "$PASSED" -gt 0 ]; then
  echo ""
  echo "Aggregating results..."
  python tools/aggregate_cross_eval.py \
    --out results/infer/cross_dataset_summary.csv \
    --manifest <(for spec in "${EXPERIMENTS[@]}"; do echo "$spec"; done)
fi
