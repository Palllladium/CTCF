#!/usr/bin/env bash
# Gate B decisive experiment (EVAL only, --2). The make-or-break for the certificate-carrying SEARCH leg:
# can a per-vertex CERTIFIED local update keep the network's accuracy, or does forcing the deployed warp
# fold-free cost as much as the heuristic repair? Three modes on the SAME feed-forward field, both datasets:
#   FEEDFWD  raw network flow (no repair)        -> accuracy ceiling; may fold trilinearly (tri_cert < 0)
#   REPAIR   trilinear_project (feathered smooth) -> the current certified number (~0.898 OASIS)
#   CLIP     certified_local_clip(identity->flow) -> 8-color certified update; the Gate B candidate
# All three report tri_cert_bound: REPAIR and CLIP must read >= eps (certified); FEEDFWD is the honest
# folding baseline. GATE: CLIP certified Dice >= 0.900 AND cost <= ~0.002 vs FEEDFWD, beating REPAIR.
# PASS => build the certificate-carrying search engine (Stage 3). FAIL (~0.898) => the local update needs
# work before features/GRU; the universal audit "floor" (Stage 1) still stands regardless.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/gateb}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
SWEEPS_LIST="${SWEEPS_LIST:-2 4 8}"   # clip is Gauss-Seidel: more sweeps recover more of the proposal
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
REPAIR="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
clip_flags() { echo "--tto_clip_from_identity 1 --tto_tri_project_eps ${EPS} --tto_clip_sweeps $1"; }
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

echo "########## Gate B decisive experiment (eps=${EPS}, sweeps={${SWEEPS_LIST}}) ##########"
infer OAS_FEEDFWD "$OAS_EXP" --ds OASIS --tto_mode none
# shellcheck disable=SC2086
infer OAS_REPAIR  "$OAS_EXP" --ds OASIS --tto_mode none $REPAIR
infer IXI_FEEDFWD "$IXI_EXP" --ds IXI --use_test --tto_mode none
# shellcheck disable=SC2086
infer IXI_REPAIR  "$IXI_EXP" --ds IXI --use_test --tto_mode none $REPAIR
for sw in $SWEEPS_LIST; do
  # shellcheck disable=SC2086
  infer "OAS_CLIP_s${sw}" "$OAS_EXP" --ds OASIS --tto_mode none $(clip_flags "$sw")
  # shellcheck disable=SC2086
  infer "IXI_CLIP_s${sw}" "$IXI_EXP" --ds IXI --use_test --tto_mode none $(clip_flags "$sw")
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "===================== GATE B TABLE ====================="
printf "%-13s %9s %13s %11s %11s\n" "run" "dice" "tri_cert_bnd" "tri_fold%" "tri_min_det"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-13s %9s %13s %11s %11s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_cert_bound)" "$(fmt tri_fold_pct)" "$(fmt tri_min_det)"
done
echo
echo "READ: compare *_CLIP dice vs *_REPAIR dice (both must have tri_cert_bnd >= ${EPS} = certified)."
echo "  GATE PASS if CLIP dice >= 0.900 and CLIP dice - FEEDFWD dice >= REPAIR dice - FEEDFWD dice (clip"
echo "  keeps more accuracy for the same certificate). Then Stage 3 (search engine) is greenlit. If CLIP"
echo "  ~= REPAIR ~0.898, the local per-vertex update needs work before features/GRU — Stage 1 floor holds."
