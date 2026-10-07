#!/usr/bin/env bash
# Phase 17c — TTO cross-domain Dice lever + certificate (EVAL only). CORRECTED direction 2026-08-02.
#
# The first probe ran OASIS->IXI @57 steps and TTO HURT — but phase12b's +0.0417 was the OTHER direction,
# IXI->OASIS @800 steps (OASIS->IXI is the WEAK direction: the field "arrives already broken" on IXI). So
# this reproduces the strong direction and confirms the weak one. TTO uses NCC only (no labels — using test
# labels would leak the fixed-image segmentation the task must predict); domain-invariance the leak-free way
# is MIND, a future objective, not test labels.
#
#   feedfwd      no adaptation (cross-domain baseline Dice)
#   tto          fixed STEPS (the raw gain, phase12b used 800; no early guard)
#   tto_guard    STEPS + adaptive stop (fold_k 1.25) = the clean operating point (+0.0228 in phase12b)
#   tto_g_repair the guarded adapted field, then repair -> certified cross-domain warp
# EPS = certificate margin (fp32 operating point, default 0.001). STEPS default 800.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_domain}"
FORCE="${FORCE:-0}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
_CALLNO=0
EPS="${EPS:-0.001}"
STEPS="${STEPS:-800}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
TTO="--tto_mode svf --tto_steps ${STEPS} --tto_lr 0.01 --tto_w_ncc 1"
GUARD="${TTO} --tto_stop both --tto_fold_k 1.25"
REPAIR="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local mine=$(( _CALLNO % NSHARD )); _CALLNO=$(( _CALLNO + 1 ))
  [[ "$NSHARD" -gt 1 && "$mine" != "$SHARD" ]] && return 0
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" && "$FORCE" != "1" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt" >&2; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $COMMON $VM --ckpt "$ckpt" --out_dir "$out" "$@"
}

# One direction = one ckpt + its --ds flags (passed as "$@"): feedfwd, raw TTO, guarded TTO, guarded+repair.
run_dir() {
  local name="$1" exp="$2"; shift 2
  # shellcheck disable=SC2086
  infer "${name}_feedfwd"      "$exp" "$@" --tto_mode none
  # shellcheck disable=SC2086
  infer "${name}_tto"          "$exp" "$@" $TTO
  # shellcheck disable=SC2086
  infer "${name}_tto_guard"    "$exp" "$@" $GUARD
  # shellcheck disable=SC2086
  infer "${name}_tto_g_repair" "$exp" "$@" $GUARD $REPAIR
}

echo "########## TTO cross-domain (eps=${EPS}, steps=${STEPS}) — IXI<->OASIS ##########"
run_dir IXI2OASIS P10_LONGRUN_VXM_UNIFIED_SVF_IXI --ds OASIS               # STRONG direction (+0.0417)
run_dir OASIS2IXI P16_W1_VXM_OASIS_LBL_DIG_J15 --ds IXI --use_test         # weak contrast

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done — run one plain pass for the table]"; exit 0; fi

echo
echo "===================== TTO CROSS-DOMAIN TABLE ====================="
printf "%-22s %9s %11s %13s %10s\n" "run" "dice" "tri_fold%" "tri_cert_bnd" "tto_steps"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-22s %9s %11s %13s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)" "$(fmt tto_steps)"
done
echo
echo "READ: IXI2OASIS_tto - IXI2OASIS_feedfwd = the adaptation gain (expect ~+0.04, phase12b's strong dir)."
echo "  tto_guard = clean operating point; tto_g_repair tri_cert_bound>=eps => certified cross-domain warp."
echo "  OASIS2IXI stays weak (field arrives broken) — the contrast, not the headline."
