#!/usr/bin/env bash
# Phase 17b — L3 certified line-search vs post-hoc repair (EVAL only). Stage A: inference-only.
#
# Instead of "L3 predicts residual -> we repair the folds it made", clip each L3 update to the largest
# trilinear-diffeomorphic fraction of itself (certified_max_step). Two spaces (vel: integrate t*delta per
# probe; disp: integrate once, scale) x compared against the chain repair. The questions:
#   (1) does line-search alone CERTIFY? (only if the L1oL2 input into L3 is itself feasible)
#   (2) does it keep MORE Dice than post-hoc repair at an equal certificate? (no undone update)
#   (3) global-t line-search vs the (local-t-like) repair — which wins Dice?
# ls_*_rep = line-search THEN chain repair: a belt-and-suspenders that always certifies; its repair-iters
# show how much work line-search already did (few iters => line-search carried it).
#
# EPS is the operating margin (set from the eps-survival run; default 0.02). J15 is the operating field.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_l3ls}"
FORCE="${FORCE:-0}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
_CALLNO=0
EPS="${EPS:-0.02}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
BASE="--model ctcf --ds OASIS ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5 --tto_mode none"
REPAIR="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
CKPTS="J15:P16_W1_VXM_OASIS_LBL_DIG_J15 NODIG:P16_W1_VXM_OASIS_LBL_NODIG"

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

echo "########## L3 LINE-SEARCH vs REPAIR (eps=${EPS}) ##########"
for item in $CKPTS; do
  name="${item%%:*}"; exp="${item#*:}"
  # shellcheck disable=SC2086
  infer "${name}_feedfwd"    "$exp" $VM
  # shellcheck disable=SC2086
  infer "${name}_repair"     "$exp" $VM $REPAIR
  # shellcheck disable=SC2086
  infer "${name}_ls_vel"     "$exp" $VM --ctcf_l3_ls_space vel  --ctcf_l3_ls_eps "$EPS"
  # shellcheck disable=SC2086
  infer "${name}_ls_disp"    "$exp" $VM --ctcf_l3_ls_space disp --ctcf_l3_ls_eps "$EPS"
  # shellcheck disable=SC2086
  infer "${name}_ls_vel_rep" "$exp" $VM --ctcf_l3_ls_space vel  --ctcf_l3_ls_eps "$EPS" $REPAIR
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done — run one plain pass for the table]"; exit 0; fi

echo
echo "===================== L3 LINE-SEARCH TABLE ====================="
printf "%-20s %8s %11s %13s %11s %10s\n" "run" "dice" "tri_fold%" "tri_cert_bnd" "tri_min" "rep_iters"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-20s %8s %11s %13s %11s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)" "$(fmt tri_min_det)" "$(fmt tri_proj_iters)"
done
echo
echo "READ per ckpt: (1) ls_vel/ls_disp tri_cert_bound >= eps AND tri_fold%=0 -> line-search alone"
echo "  certifies (L1oL2 input was feasible). (2) Dice(ls_*) vs Dice(repair): line-search should keep"
echo "  more. (3) ls_vel_rep rep_iters: if small, line-search did the work; its Dice = the guaranteed"
echo "  operating point. Best certified Dice across rows = the L3-mechanism operating point."
