#!/usr/bin/env bash
# Identity-boundary evaluation.  The operational repair targets WORK_EPS; the independent exact verifier
# certifies the saved float32 bytes at CLAIM_EPS.  Any failed case or inconclusive verifier exits non-zero.
set -euo pipefail

GPU="${GPU:-0}"
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/collar}"
FORCE="${FORCE:-0}"
CLAIM_EPS="${CLAIM_EPS:-0.001}"
WORK_EPS="${WORK_EPS:-0.0011}"
WIDTHS="${WIDTHS:-4 8}"
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5 --tto_mode none"
# Digital is retained as the established preconditioner; both repairs now receive the fixed boundary mask.
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${WORK_EPS}"
collar_flags() { echo "--tto_collar 1 --tto_collar_width $1 --save_flow"; }
OAS_EXP="${OAS_EXP:-P16_W1_VXM_OASIS_LBL_DIG_J15}"
IXI_EXP="${IXI_EXP:-P10_LONGRUN_VXM_UNIFIED_SVF_IXI}"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local mine=$(( _CALLNO % NSHARD )); _CALLNO=$(( _CALLNO + 1 ))
  [[ "$NSHARD" -gt 1 && "$mine" != "$SHARD" ]] && return 0
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" && "$FORCE" != "1" ]]; then echo "[SKIP] $tag"; return 0; fi
  [[ -f "$ckpt" ]] || { echo "[FAIL] $tag: no checkpoint at $ckpt" >&2; return 1; }
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "$PYBIN" -m experiments.inference $COMMON $VM --ckpt "$ckpt" --out_dir "$out" "$@"
}

verify_saved() {
  local tag="$1" out="$OUT_ROOT/$1"
  [[ -d "$out/flows" ]] || { echo "[FAIL] $tag: saved flow directory is missing" >&2; return 1; }
  [[ -f "$out/per_case.csv" ]] || { echo "[FAIL] $tag: per_case.csv is missing" >&2; return 1; }
  # The verifier must see exactly one field for every reported case: neither a missing field nor a stale extra
  # is allowed to turn a subset into an apparent all-case pass.
  if ! diff -u \
      <(awk -F, 'NR==1 {for(i=1;i<=NF;i++) if($i=="case_id") c=i; if(!c) exit 2; next}
                   {print "flow_" $c ".npz"}' "$out/per_case.csv" | LC_ALL=C sort) \
      <(find "$out/flows" -maxdepth 1 -type f -name 'flow_*.npz' -printf '%f\n' | LC_ALL=C sort); then
    echo "[FAIL] $tag: per_case.csv and saved flow set differ (remove stale output or rerun cleanly)" >&2
    return 1
  fi
  "$PYBIN" -m utils.cert_exact --flow "$out/flows" --eps "$CLAIM_EPS" \
    --require-zero-boundary --report "$out/exact_certificate.json"
  # Per-case, not averaged: every final field must retain a bitwise-zero boundary and pass the operational gate.
  awk -F, '
    NR==1 { for(i=1;i<=NF;i++){ if($i=="identity_boundary_exact") b=i; if($i=="bernstein_pass_float64") c=i };
            if(!b || !c) exit 2; next }
    ($b != 1 || $c != 1) { exit 1 }
  ' "$out/per_case.csv" || { echo "[FAIL] $tag: at least one per-case invariant failed" >&2; return 1; }
}

echo "########## identity-boundary audit (claim=${CLAIM_EPS}, work=${WORK_EPS}, widths={${WIDTHS}}) ##########"
infer OAS_REPAIR "$OAS_EXP" --ds OASIS $CHAIN
infer IXI_REPAIR "$IXI_EXP" --ds IXI --use_test $CHAIN
for width in $WIDTHS; do
  # shellcheck disable=SC2046
  infer "OAS_COLLAR_w${width}" "$OAS_EXP" --ds OASIS $(collar_flags "$width") $CHAIN
  # shellcheck disable=SC2046
  infer "IXI_COLLAR_w${width}" "$IXI_EXP" --ds IXI --use_test $(collar_flags "$width") $CHAIN
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done; run verification after all shards]"; exit 0; fi

for width in $WIDTHS; do
  verify_saved "OAS_COLLAR_w${width}"
  verify_saved "IXI_COLLAR_w${width}"
done

echo
echo "===================== COLLAR TABLE ====================="
printf "%-16s %9s %13s %14s %14s\n" "run" "dice_mean" "cert_min" "boundary_max" "all_exact_bnd"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v key="$1" -v col="$2" '$1==key{printf "%s",$col}' "$d/summary.csv"; }
  printf "%-16s %9.5f %13.6g %14.6g %14.0f\n" "$(basename "$d")" \
    "$(get dice_mean 2)" "$(get tri_cert_bound 6)" "$(get boundary_max_disp 7)" \
    "$(get identity_boundary_exact 6)"
done
echo
echo "PASS means: every saved float32 field passed the exact Bernstein predicate at ${CLAIM_EPS}, and every"
echo "boundary displacement component was bitwise zero. Accuracy cost is empirical and remains in the table."
