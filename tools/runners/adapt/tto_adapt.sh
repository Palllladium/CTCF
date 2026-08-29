#!/usr/bin/env bash
# Phase 17d — TTO adaptation, the full picture (EVAL only). Closes every open TTO question at once:
#   (1) reproduce the +0.0417 cross-domain gain in the RIGHT direction (IXI->OASIS @800, NCC);
#   (2) does MIND (intensity-invariant) beat NCC cross-domain, where NCC drifts to the intensity match?
#   (3) OASIS->IXI (the weak direction) — NCC vs MIND;
#   (4) in-domain weak-sup TTO ceiling (UNTESTED — Phase 12 was unsup@0.835, not weak-sup@0.906);
#   (5) does the repair certify the ADAPTED field (dataset-agnostic guarantee)?
#
# Per cross-domain scenario: feedfwd + {ncc,mind} x {tto@800 raw, guard, guard+repair}. In-domain: NCC only
# (intensities match, MIND ~ NCC). MIND is the LEAK-FREE domain-invariant objective — test labels would leak
# the fixed-image segmentation the task must predict, so we never optimise on them.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_adapt}"
FORCE="${FORCE:-0}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
_CALLNO=0
EPS="${EPS:-0.001}"
STEPS="${STEPS:-800}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
RAW="--tto_mode svf --tto_steps ${STEPS} --tto_lr 0.01"
GUARD_ADD="--tto_stop both --tto_fold_k 1.25"
REPAIR="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
NCC="--tto_w_ncc 1 --tto_w_mind 0"
MIND="--tto_w_ncc 0 --tto_w_mind 1"
IXI_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_IXI"      # IXI-trained
OAS_EXP="P16_W1_VXM_OASIS_LBL_DIG_J15"         # OASIS-trained (J15)

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

# Cross-domain scenario: feedfwd + NCC/MIND x {raw tto, guard, guard+repair}. ds passed as "$@".
cross() {
  local name="$1" exp="$2"; shift 2
  # shellcheck disable=SC2086
  infer "${name}_feedfwd" "$exp" "$@" --tto_mode none
  for o in NCC MIND; do
    local obj="${!o}"
    # shellcheck disable=SC2086
    infer "${name}_${o}_tto"   "$exp" "$@" $RAW $obj
    # shellcheck disable=SC2086
    infer "${name}_${o}_guard" "$exp" "$@" $RAW $obj $GUARD_ADD
    # shellcheck disable=SC2086
    infer "${name}_${o}_grep"  "$exp" "$@" $RAW $obj $GUARD_ADD $REPAIR
  done
}

echo "########## TTO adaptation (eps=${EPS}, steps=${STEPS}) ##########"
cross IXI2OASIS "$IXI_EXP" --ds OASIS                    # STRONG cross-domain (reproduce +0.04)
cross OASIS2IXI "$OAS_EXP" --ds IXI --use_test           # weak cross-domain

# In-domain weak-sup ceiling (J15 on OASIS): does TTO help at all above the 0.90 feed-forward?
# shellcheck disable=SC2086
infer INDOM_feedfwd   "$OAS_EXP" --ds OASIS --tto_mode none
# shellcheck disable=SC2086
infer INDOM_NCC_tto   "$OAS_EXP" --ds OASIS $RAW $NCC
# shellcheck disable=SC2086
infer INDOM_NCC_guard "$OAS_EXP" --ds OASIS $RAW $NCC $GUARD_ADD

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done — run one plain pass for the table]"; exit 0; fi

echo
echo "===================== TTO ADAPTATION TABLE ====================="
printf "%-24s %9s %11s %13s %10s\n" "run" "dice" "tri_fold%" "tri_cert_bnd" "tto_steps"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-24s %9s %11s %13s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)" "$(fmt tto_steps)"
done
echo
echo "READ: IXI2OASIS_{NCC,MIND}_tto - IXI2OASIS_feedfwd = adaptation gain (NCC should ~reproduce +0.04;"
echo "  does MIND beat it?). _grep tri_cert_bound>=eps => certified adapted warp. INDOM_* = does TTO lift"
echo "  the in-domain 0.90 (untested weak-sup regime)? OASIS2IXI = weak-direction contrast."
