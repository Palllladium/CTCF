#!/usr/bin/env bash
# Phase 15c — robust + universal digital certificate (EVAL only, on the archived 100ep checkpoints).
#
# tto_project2 settled the mechanism (two-phase barrier a32 -> feathered projection = 0.8931 Dice at
# certified EXACT zero, and that zero holds under all schemes: digital-10 = 8-corner = central = 0).
# This runner pushes the two remaining cheap axes that feed the Paper-3 framing:
#
#   EPS   robust certificate: project to det >= eps (not the det>0 knife-edge) so the guarantee
#         survives a resampling/scheme perturbation. Sweep eps -> read the Dice cost of a margin.
#   UNIV  training-agnostic: the same two-phase on A0 (unsup, no digital training, 0.39% folds) and A1
#         (labels, no digital penalty) — does the projection certify zero regardless of how the field
#         was trained? Evidence toward a *universal post-hoc certifier*, not just "CTCF is clean".
#
# Gate table reports all three fold schemes (digital-10 / 8-corner / central) so scheme-domination is
# visible per row. Restore A0/A1/A2 @100ep to results/<EXP>/ckpt/best.pth first.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
BLOCK="${BLOCK:-ALL}"
OUT_ROOT="${OUT_ROOT:-results/tto_certify}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

A0_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_OASIS"
A1_EXP="P14_2A_VXM_DICE_OASIS"
A2_EXP="P14_2A_VXM_DICE_DIGITAL_OASIS"

BASE="--model ctcf --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1 --ds OASIS ${PROFILE} \
      --strict_ckpt 0 --gpu ${GPU} --print_every 5"
# two-phase = proximal barrier (a32, Dice-preserving) then feathered projection.
PROX="--tto_mode svf --tto_steps 400 --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 \
      --tto_w_ncc 0 --tto_w_reg 0 --tto_anchor_w 32"

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

if want EPS; then
  echo "########## robust certificate: two-phase + projection margin eps ##########"
  for eps in 0 0.02 0.05; do
    # shellcheck disable=SC2086
    infer "A2_a32_proj_e${eps}" "$A2_EXP" $PROX --tto_project 1 --tto_project_eps "$eps"
  done
fi

if want UNIV; then
  echo "########## training-agnostic: same two-phase (+eps 0.02) across regimes ##########"
  for pair in "A0:$A0_EXP" "A1:$A1_EXP" "A2:$A2_EXP"; do
    tag="${pair%%:*}"; exp="${pair#*:}"
    # shellcheck disable=SC2086
    infer "${tag}_a32_proj_e02" "$exp" $PROX --tto_project 1 --tto_project_eps 0.02
  done
fi

echo
echo "================== TTO-CERTIFY GATE TABLE (all schemes) =================="
printf "%-22s %8s %10s %9s %9s %10s\n" "run" "dice" "digital10" "8corner" "central" "proj_fold%"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.4f" "$v"; }
  printf "%-22s %8s %10s %9s %9s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt j_leq0_percent)" "$(fmt j_leq0_corners_percent)" \
    "$(fmt j_leq0_central_percent)" "$(fmt proj_folds_end)"
done

echo
echo "Eval CSV: $OUT_ROOT/ (send back). Runs on the --2 box (100ep checkpoints); independent of Wave 1."
