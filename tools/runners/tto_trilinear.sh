#!/usr/bin/env bash
# Phase 15e — the interpolation-consistency GATE (EVAL only, --2 box, archived 100ep checkpoints).
#
# tto_probe added cert_min_det and showed our "certified zero" lives entirely inside the DIGITAL
# (corner/tetrahedral) criterion. But grid_sample warps with TRILINEAR interpolation, and a synthetic
# cell proves digital-10 positivity (margin +0.168) can coexist with a trilinear fold (detJ -0.46).
# So the open question that decides the whole Paper-3 branch is EMPIRICAL, not theoretical:
#
#   Do OUR real fields — and above all the PROJECTED ones — fold under the trilinear warp we apply?
#
# Every row now carries two new columns computed on the FINAL field:
#   tri_min_det    sampled min of det J of the actual trilinear deformation (5^3 per cell).
#                  < 0 PROVES the applied warp folds. This is the gate number.
#   tri_cert_bound sound Bernstein lower bound over every cell. > 0 CERTIFIES trilinear diffeomorphism.
# Compare against cert_min_det (digital) and digital10 in the same row: a positive digital certificate
# with a negative tri_min_det is the "zero is not zero" gap, in-house, on our own fields.
#
# GATE DECISION (read the table):
#   projected rows tri_min_det >= 0 everywhere  -> digital certificate is empirically trilinear-safe
#       for these smooth SVF fields; topology stays as an HONEST guarantee, growth axis is Dice.
#   any projected row tri_min_det < 0 (esp. e0)  -> feathered projection injects trilinear folds while
#       zeroing the digital count; the "audit + trilinear-aware repair" branch is earned in-house.
#
# Checkpoints already live at results/<EXP>/ckpt (names from phase10_inference.sh); nothing to restore.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_trilinear}"
FORCE="${FORCE:-0}"                            # 1 = recompute even if summary.csv exists (new columns)

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

A0_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_OASIS"     # unsup baseline (0.39% digital folds)
A2_EXP="P14_2A_VXM_DICE_DIGITAL_OASIS"         # champion field (labels + digital penalty)
HF_EXP="P10_LONGRUN_MAMBA_NOSVF_OASIS"         # high fold density (2.2% digital, no SVF)
MB_EXP="P10_LONGRUN_MAMBA_SVF_OASIS"
LK_EXP="P10_LONGRUN_LKU8_SVF_OASIS"
IXI_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_IXI"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
BASE="--model ctcf --ds OASIS ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
# two-phase = proximal barrier (a32, Dice-preserving) then feathered projection.
PROX="--tto_mode svf --tto_steps 400 --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 \
      --tto_w_ncc 0 --tto_w_reg 0 --tto_anchor_w 32"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" && "$FORCE" != "1" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt"; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

echo "########## GATE: trilinear validity of feed-forward vs projected fields ##########"
# --- champion (low-fold SVF): the pivotal before/after. Does projection fold it trilinearly? ---
# shellcheck disable=SC2086
infer A2_feedfwd      "$A2_EXP" $VM --tto_mode none
# shellcheck disable=SC2086
infer A2_projonly_e0  "$A2_EXP" $VM --tto_mode none --tto_project 1 --tto_project_eps 0
# shellcheck disable=SC2086
infer A2_projonly_e05 "$A2_EXP" $VM --tto_mode none --tto_project 1 --tto_project_eps 0.05
# shellcheck disable=SC2086
infer A2_twophase_e0  "$A2_EXP" $VM $PROX --tto_project 1 --tto_project_eps 0

# --- unsupervised baseline (different training regime) ---
# shellcheck disable=SC2086
infer A0_feedfwd      "$A0_EXP" $VM --tto_mode none
# shellcheck disable=SC2086
infer A0_projonly_e0  "$A0_EXP" $VM --tto_mode none --tto_project 1 --tto_project_eps 0

# --- high fold density (Mamba NoSVF ~2.2%): the worst case for the projector ---
# shellcheck disable=SC2086
infer HF_feedfwd      "$HF_EXP" --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0 --tto_mode none
# shellcheck disable=SC2086
infer HF_projonly_e0  "$HF_EXP" --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0 --tto_mode none --tto_project 1 --tto_project_eps 0
# shellcheck disable=SC2086
infer HF_projonly_e05 "$HF_EXP" --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0 --tto_mode none --tto_project 1 --tto_project_eps 0.05

# --- architecture generality (SSM / large-kernel CNN) ---
# shellcheck disable=SC2086
infer MAMBA_projonly_e0 "$MB_EXP" --ctcf_config CTCF-CascadeA-Mamba --tto_mode none --tto_project 1 --tto_project_eps 0
# shellcheck disable=SC2086
infer LKU8_projonly_e0  "$LK_EXP" --ctcf_config CTCF-CascadeA-LKU8  --tto_mode none --tto_project 1 --tto_project_eps 0

# --- dataset generality (IXI test) ---
# shellcheck disable=SC2086
infer IXI_projonly_e0 "$IXI_EXP" $VM --ds IXI --use_test --tto_mode none --tto_project 1 --tto_project_eps 0

echo
echo "===================== TRILINEAR GATE TABLE ====================="
printf "%-20s %8s %9s %11s %11s %10s %10s\n" \
  "run" "dice" "digital10" "tri_min" "tri_bound" "tri_fold%" "case_fold%"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  pct() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.2f" "$(awk -v x="$v" 'BEGIN{print x*100}')"; }
  printf "%-20s %8s %9s %11s %11s %10s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt j_leq0_percent)" "$(fmt tri_min_det)" \
    "$(fmt tri_cert_bound)" "$(fmt tri_fold_pct)" "$(pct tri_case_folds)"
done
echo
echo "AUDIT: tri_fold% = mean %% of cells that PROVABLY fold trilinearly (digital10 hides these);"
echo "       case_fold% = %% of the N cases with >=1 trilinear fold. tri_min<0 => that field folds."
echo "Re-run with FORCE=1 to recompute existing rows with the new tri_fold columns."
echo "Eval CSV: $OUT_ROOT/ (send back). Runs on the --2 box; independent of Wave 1 on the H-box."
