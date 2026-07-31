#!/usr/bin/env bash
# Phase 15b — feathered digital projection, certified zero (EVAL only, no training).
#
# Supersedes the projection passes of tto_project.sh (those used the broken contract-to-identity
# digital_project, which ran away and collapsed Dice to ~0.60). digital_project is now the feathered
# local relaxation (verified on multi-fold synthetics: exact zero at ~7x less distortion). This runner
# measures it on the REAL A2 field across the axes that actually carry uncertainty:
#
#   phase-1 anchor   extend the proximal barrier sweep to a16/a32 (recover the last Dice)
#   projection-only  feed-forward -> feathered projection (is the barrier phase even needed?)
#   two-phase        proximal barrier (a16/a32) -> feathered projection (certify residual to exact 0)
#   damp             feathered blend-strength robustness on the real field
#
# GATE: two-phase reaches proj_folds_end == 0 (certificate) at dice within ~0.005 of A2 feed-forward
# (0.8950). Then: competitive Dice + certified-zero digital topology.
#
#   BLOCK=P1   bash tools/runners/tto_project2.sh   # phase-1 anchor only (fast, no projection)
#   BLOCK=PROJ bash tools/runners/tto_project2.sh   # projection-only + two-phase
#   bash tools/runners/tto_project2.sh              # all
#
# Restore archived A0/A1/A2 @100ep to results/<EXP>/ckpt/best.pth first.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
BLOCK="${BLOCK:-ALL}"
OUT_ROOT="${OUT_ROOT:-results/tto_project2}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

A0_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_OASIS"
A1_EXP="P14_2A_VXM_DICE_OASIS"
A2_EXP="P14_2A_VXM_DICE_DIGITAL_OASIS"

BASE="--model ctcf --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1 --ds OASIS ${PROFILE} \
      --strict_ckpt 0 --gpu ${GPU} --print_every 5"
BARRIER="--tto_mode svf --tto_steps 400 --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1"
PROX="${BARRIER} --tto_w_ncc 0 --tto_w_reg 0"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt (restore from archive)"; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

if want P1; then
  echo "########## phase-1: extended anchor (proximal barrier, no projection) ##########"
  for aw in 8 16 32; do
    # shellcheck disable=SC2086
    infer "A2_prox_a${aw}" "$A2_EXP" $PROX --tto_anchor_w "$aw"
  done
fi

if want PROJ; then
  echo "########## projection-only (feed-forward -> feathered) ##########"
  for dm in 0.4 0.6; do
    infer "A2_ff_proj_d${dm}" "$A2_EXP" --tto_mode none --tto_project 1 --tto_project_damp "$dm"
  done
  infer "A0_ff_proj_d0.6" "$A0_EXP" --tto_mode none --tto_project 1 --tto_project_damp 0.6

  echo "########## two-phase (proximal barrier -> feathered) ##########"
  for aw in 16 32; do
    # shellcheck disable=SC2086
    infer "A2_prox_a${aw}_proj" "$A2_EXP" $PROX --tto_anchor_w "$aw" --tto_project 1 --tto_project_damp 0.6
  done
  # shellcheck disable=SC2086
  infer "A1_prox_a16_proj" "$A1_EXP" $PROX --tto_anchor_w 16 --tto_project 1 --tto_project_damp 0.6
fi

echo
echo "=================== TTO-PROJECT2 GATE TABLE ==================="
printf "%-20s %8s %9s %9s %10s %8s\n" "run" "dice" "j<=0%" "brain%" "proj_fold%" "proj_it"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.4f" "$v"; }
  printf "%-20s %8s %9s %9s %10s %8s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt j_leq0_percent)" "$(fmt j_leq0_brain_percent)" \
    "$(fmt proj_folds_end)" "$(fmt proj_iters)"
done

echo
echo "Eval CSV: $OUT_ROOT/ (send back). per_case.csv carries the distribution for paired stats."
