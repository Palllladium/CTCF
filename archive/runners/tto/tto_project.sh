#!/usr/bin/env bash
# Phase 15 — Dice-preserving topology at test time (EVAL only, no training).
#
# The Step-2a gate showed the GLOBAL barrier (NCC+barrier, 400 steps) kills folds but costs ~0.034
# Dice, because label-free NCC drags the label-trained field back toward its own optimum. This runner
# tests two mechanisms that fix folds WITHOUT touching the Dice the labels bought:
#
#   PROJ    hard digital projection on the feed-forward field  (no TTO): contract the displacement
#           toward identity only at folded voxels until every one of the ten Liu-et-al. determinants
#           is positive. Certified zero folds; touches ~fold% of voxels, so Dice is preserved by
#           construction. Simplest candidate — likely the answer.
#   BARRIER proximal-anchored barrier TTO: barrier + anchor ||flow-flow0||^2, NCC AND diffusion OFF
#           (--tto_w_ncc 0 --tto_w_reg 0), so only the folded voxels move. Anchor sweep; optionally
#           followed by PROJ to certify the residual to exact zero. The soft alternative.
#
# Runs on archived A0/A1/A2 @100ep checkpoints — restore them to results/<EXP>/ckpt/best.pth first.
#   A0 = P10_LONGRUN_VXM_UNIFIED_SVF_OASIS   (unsup baseline)
#   A1 = P14_2A_VXM_DICE_OASIS               (+labels)
#   A2 = P14_2A_VXM_DICE_DIGITAL_OASIS       (+labels +digital penalty)   <- primary
#
#   BLOCK=PROJ    bash tools/runners/tto_project.sh   # fast: feed-forward + projection
#   BLOCK=BARRIER bash tools/runners/tto_project.sh   # proximal-barrier TTO sweep (~2 h)
#   bash tools/runners/tto_project.sh                 # both
#
# READING: per_case.csv gives mean+/-std (feed the folder to tools/compute_stats.py for paired tests).
# GATE (fixed): a mechanism passes if dice stays within ~0.005 of A2 feed-forward (0.8950) at
# j<=0% (whole) <= 0.01 and proj_folds_end == 0 (the certificate). Then topology does NOT cap Dice.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
BLOCK="${BLOCK:-ALL}"
OUT_ROOT="${OUT_ROOT:-results/tto_project}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

A0_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_OASIS"
A1_EXP="P14_2A_VXM_DICE_OASIS"
A2_EXP="P14_2A_VXM_DICE_DIGITAL_OASIS"

BASE="--model ctcf --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1 --ds OASIS ${PROFILE} \
      --strict_ckpt 0 --gpu ${GPU} --print_every 5"
# barrier (Step-1 setting); PROX adds the anchor and turns NCC + diffusion off (repair only).
BARRIER="--tto_mode svf --tto_steps 400 --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1"
PROX="${BARRIER} --tto_w_ncc 0 --tto_w_reg 0"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  # infer <tag> <exp> <extra-flags...>
  local tag="$1" exp="$2"; shift 2
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt (restore from archive)"; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

if want PROJ; then
  echo "########## PROJ: feed-forward + hard digital projection ##########"
  for pair in "A0:$A0_EXP" "A1:$A1_EXP" "A2:$A2_EXP"; do
    tag="${pair%%:*}"; exp="${pair#*:}"
    infer "${tag}_feedfwd"      "$exp" --tto_mode none
    infer "${tag}_feedfwd_proj" "$exp" --tto_mode none --tto_project 1
  done
fi

if want BARRIER; then
  echo "########## BARRIER: proximal-anchored barrier TTO ##########"
  # A2 (primary): anchor sweep, then best + projection, then the old global barrier for contrast.
  for aw in 0.5 2 8; do
    # shellcheck disable=SC2086
    infer "A2_prox_a${aw}" "$A2_EXP" $PROX --tto_anchor_w "$aw"
  done
  # shellcheck disable=SC2086
  infer "A2_prox_a2_proj" "$A2_EXP" $PROX --tto_anchor_w 2 --tto_project 1
  # shellcheck disable=SC2086
  infer "A2_global_barrier" "$A2_EXP" $BARRIER   # w_ncc=1, anchor=0: the -0.034 Dice contrast
  # A0/A1 context at the mid anchor.
  # shellcheck disable=SC2086
  infer "A1_prox_a2_proj" "$A1_EXP" $PROX --tto_anchor_w 2 --tto_project 1
  # shellcheck disable=SC2086
  infer "A0_prox_a2_proj" "$A0_EXP" $PROX --tto_anchor_w 2 --tto_project 1
fi

echo
echo "=================== TTO-PROJECT GATE TABLE ==================="
printf "%-20s %8s %9s %9s %10s\n" "run" "dice" "j<=0%" "brain%" "proj_fold%"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.4f" "$v"; }
  printf "%-20s %8s %9s %9s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt j_leq0_percent)" "$(fmt j_leq0_brain_percent)" "$(fmt proj_folds_end)"
done

echo
echo "Eval CSV: $OUT_ROOT/ (send back). per_case.csv in each folder carries the distribution for stats."
