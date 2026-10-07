#!/usr/bin/env bash
# Phase 17f — GLOBAL-INJECTIVITY probe (EVAL only, --2). Gathers every quantity the "fold-free -> genuine
# diffeomorphism" upgrade needs, on real deployed fields, for both datasets, feed-forward AND post-repair:
#   tri_cert_bound     interior: > 0 => no folds (LOCAL orientation preserving)         [have it]
#   disp_grad_norm     global contraction: < 1 ALONE => globally injective              [route A, strong]
#   boundary_tan_lip   per-face tangential Lipschitz: < 1 => each face injective        [route B, Kroemer 2020]
#   boundary_max_disp  small => no cross-face collision, completing route B
# Decision: if (interior cert > 0) AND (boundary_tan_lip < 1) AND (boundary_max_disp small) => we may write
# "certified diffeomorphism", not just "fold-free". Route A (disp_grad_norm<1) is known dead on these fields
# (~8); this run tests whether the weaker Route B fires.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/injectivity}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
OAS_EXP="${OAS_EXP:-P16_W1_VXM_OASIS_LBL_DIG_J15}"
IXI_EXP="${IXI_EXP:-P10_LONGRUN_VXM_UNIFIED_SVF_IXI}"

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

echo "########## GLOBAL-INJECTIVITY probe (eps=${EPS}) ##########"
infer OAS_FEEDFWD "$OAS_EXP" --ds OASIS --tto_mode none
# shellcheck disable=SC2086
infer OAS_REPAIR  "$OAS_EXP" --ds OASIS --tto_mode none $CHAIN
infer IXI_FEEDFWD "$IXI_EXP" --ds IXI --use_test --tto_mode none
# shellcheck disable=SC2086
infer IXI_REPAIR  "$IXI_EXP" --ds IXI --use_test --tto_mode none $CHAIN

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "===================== INJECTIVITY TABLE ====================="
printf "%-12s %9s %13s %11s %13s %12s\n" \
  "run" "dice" "tri_cert_bnd" "dispGrad" "bnd_tan_lip" "bnd_maxDisp"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-12s %9s %13s %11s %13s %12s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_cert_bound)" "$(fmt disp_grad_norm)" \
    "$(fmt boundary_tan_lip)" "$(fmt boundary_max_disp)"
done
echo
echo "READ (per REPAIR row): tri_cert_bnd >= eps (interior fold-free) is DONE. If bnd_tan_lip < 1 as well,"
echo "  each boundary face is injective; with a small bnd_maxDisp that closes GLOBAL injectivity (Kroemer"
echo "  2020) => we upgrade the claim to 'certified diffeomorphism'. If bnd_tan_lip >= 1, report fold-free"
echo "  only and note global injectivity is open (a real, honest boundary)."
