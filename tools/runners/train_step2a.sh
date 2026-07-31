#!/usr/bin/env bash
# Phase 14 — Step 2a: weak-supervision + digital-penalty CALIBRATION (not a novelty claim).
#
# First time CTCF is trained in the 2026 regime — labels (soft-Dice) + the digital topology penalty
# from Steps 0/1 — to find WHERE it lands on Dice and topology. Reconnaissance for the decision gate,
# not a mechanism paper. Base = VxM Unified SVF, OASIS, 100ep from scratch (resuming would restart the
# L1/L3 warm-up schedule and wreck the converged model).
#
#   A0  unsup baseline           = existing ckpt P10_LONGRUN_VXM_UNIFIED_SVF_OASIS (no training)
#   A1  + labels (w_dice=1.0)    = feed-forward Dice under weak supervision
#   A2  + labels + digital jac   = Dice AND feed-forward folds with the train-time topology penalty
#
# Each is then scored feed-forward and with the test-time barrier (Step 1) — the two-tier picture and
# the gate: is high labelled Dice compatible with near-zero digital topology?
#
#   BLOCK=TRAIN bash tools/runners/train_step2a.sh   # ~8 h/run on VxM (2 runs)
#   BLOCK=EVAL  bash tools/runners/train_step2a.sh   # after training
#   bash tools/runners/train_step2a.sh               # both
#
# DECISION GATE (fixed in advance): feed-forward+labels Dice >=0.88 competitive / <0.86 pivot;
# post-barrier Dice within ~0.005 of feed-forward at digital folds <=0.01% => topology does not cap
# Dice (green for digital-by-construction); barrier dragging Dice back to ~0.835 => topology caps Dice.
set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
BLOCK="${BLOCK:-ALL}"
OUT_ROOT="${OUT_ROOT:-results/step2a_eval}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

VXM="--config CTCF-CascadeA-VM-Unified --l3_svf 1"
OASIS="--ds OASIS ${PROFILE} --gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1 \
        --w_ncc 1.0 --w_icon 0.05 --w_reg 1.0"

A0_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_OASIS"   # existing unsup baseline
A1_EXP="P14_2A_VXM_DICE_OASIS"
A2_EXP="P14_2A_VXM_DICE_DIGITAL_OASIS"

# ---- TRAIN ----
run_ctcf() { local exp="$1"; shift; echo "> train ${exp}"; "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp}"; }

if want TRAIN; then
  echo "########## Step-2a TRAIN (VxM Unified SVF, OASIS, ${MAX_EPOCH}ep) ##########"
  # A1: weak supervision. w_jac stays the inert central default (0.005) — labels only.
  # shellcheck disable=SC2086
  run_ctcf "$A1_EXP" $VXM $OASIS --w_jac 0.005 --w_dice 1.0
  # A2: weak supervision + train-time digital penalty (the Step-0/1 hinge, warm-up ramped).
  # shellcheck disable=SC2086
  run_ctcf "$A2_EXP" $VXM $OASIS --w_jac 5.0 --jac_mode digital --w_dice 1.0
fi

# ---- EVAL: feed-forward + test-time barrier, digital metrics ----
ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  # infer <tag> <exp> <tto-flags...>
  local tag="$1" exp="$2"; shift 2
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt"; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference --model ctcf --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1 \
    --ds OASIS "$PROFILE" --ckpt "$ckpt" --strict_ckpt 0 --gpu "$GPU" --hd95 --print_every 5 \
    --out_dir "$out" "$@"
}

if want EVAL; then
  echo "########## Step-2a EVAL (feed-forward + test-time barrier) ##########"
  BARRIER="--tto_mode svf --tto_steps 400 --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 --tto_trace 25 100 400"
  for pair in "A0:$A0_EXP" "A1:$A1_EXP" "A2:$A2_EXP"; do
    tag="${pair%%:*}"; exp="${pair#*:}"
    infer "${tag}_feedfwd" "$exp" --tto_mode none
    # shellcheck disable=SC2086
    infer "${tag}_barrier" "$exp" $BARRIER
  done

  echo
  echo "=================== STEP-2a GATE TABLE ==================="
  printf "%-16s %8s %9s %9s\n" "run" "dice" "j<=0%" "brain%"
  for d in "$OUT_ROOT"/*/; do
    [[ -f "$d/summary.csv" ]] || continue
    get() { awk -F, -v k="$1" '$1==k{printf "%.4f",$2}' "$d/summary.csv"; }
    printf "%-16s %8s %9s %9s\n" "$(basename "$d")" \
      "$(get dice_mean)" "$(get j_leq0_percent)" "$(get j_leq0_brain_percent)"
  done
fi

echo
echo "Ckpts: results/P14_2A_*/ckpt/best.pth ; eval CSV: $OUT_ROOT/ (send these back)."
