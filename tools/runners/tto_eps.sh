#!/usr/bin/env bash
# Phase 17a — epsilon-survival: what certificate margin survives a deployment step? (EVAL only)
#
# chain_e0 certifies at tri_cert_bound = +8e-6 — mathematically fold-free but a KNIFE-EDGE that a storage
# round-trip or resampling could push negative. The determinant depends on displacement GRADIENTS, so a
# margin >= the perturbation's induced Jacobian change survives, a smaller one dies. We repair to a range
# of margins eps and then perturb (fp16 storage round-trip; additive noise), recomputing every metric on
# the PERTURBED field. The smallest eps whose perturbed tri_fold% = 0 (tri_cert_bound >= 0) is the real
# operating margin — replacing the guessed 0.02, and it feeds the L3 line-search eps.
#
# Read J15 (the operating field): the smallest eps column with tri_fold% = 0 under BOTH fp16 and noise.
# Parallel: same SHARD/NSHARD as the other runners (one process per card).
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_eps}"
FORCE="${FORCE:-0}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
_CALLNO=0
NOISE="${NOISE:-0.02}"                          # noise amplitude in voxels (~fp16 scale for our displacements)

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
BASE="--model ctcf --ds OASIS ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
# chained repair: cheap digital bulk removal, then trilinear repair to margin eps.
CHAIN="--tto_mode none --tto_project 1 --tto_project_eps 0 --tto_tri_project 1"
CKPTS="J15:P16_W1_VXM_OASIS_LBL_DIG_J15 NODIG:P16_W1_VXM_OASIS_LBL_NODIG"
EPS_LIST="${EPS_LIST:-0 0.005 0.01 0.02 0.05 0.1}"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local mine=$(( _CALLNO % NSHARD )); _CALLNO=$(( _CALLNO + 1 ))
  [[ "$NSHARD" -gt 1 && "$mine" != "$SHARD" ]] && return 0
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" && "$FORCE" != "1" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt"; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

echo "########## EPS-SURVIVAL: repair margin x deployment perturbation ##########"
for item in $CKPTS; do
  name="${item%%:*}"; exp="${item#*:}"
  for eps in $EPS_LIST; do
    e="${eps/./p}"                              # 0.005 -> 0p005 for a clean tag
    # shellcheck disable=SC2086
    infer "${name}_e${e}_clean" "$exp" $VM $CHAIN --tto_tri_project_eps "$eps" --tto_perturb none
    # shellcheck disable=SC2086
    infer "${name}_e${e}_fp16"  "$exp" $VM $CHAIN --tto_tri_project_eps "$eps" --tto_perturb fp16
    # shellcheck disable=SC2086
    infer "${name}_e${e}_noise" "$exp" $VM $CHAIN --tto_tri_project_eps "$eps" --tto_perturb noise --tto_perturb_scale "$NOISE"
  done
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done — run one plain pass for the table]"; exit 0; fi

echo
echo "===================== EPS-SURVIVAL TABLE ====================="
printf "%-20s %8s %11s %13s %10s\n" "run" "dice" "tri_fold%" "tri_cert_bnd" "tri_min"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-20s %8s %11s %13s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)" "$(fmt tri_min_det)"
done
echo
echo "READ: for each ckpt, the smallest eps whose *_fp16 AND *_noise rows both show tri_fold% = 0 is the"
echo "  real operating margin. *_clean is the un-perturbed reference (should already be 0). Dice at that"
echo "  eps = the honest certified operating Dice. This eps then feeds the L3 certified line-search."
