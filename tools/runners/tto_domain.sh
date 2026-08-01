#!/usr/bin/env bash
# Phase 17c — TTO as cross-domain Dice lever + certificate (EVAL only, --2/--3, archived checkpoints).
#
# TTO's real Dice value is NOT in-domain (near ceiling) but CROSS-DOMAIN: an OASIS-trained model degrades
# on unseen IXI; per-pair test-time optimisation adapts it. This probes whether TTO lifts IXI Dice for our
# OASIS operating checkpoints, and whether the trilinear repair still certifies the ADAPTED field. This is
# where "beats everyone" lives — feed-forward competitors don't adapt. (Only OASIS/IXI are local; more
# unseen datasets are the rigor phase.)
#
#   feedfwd     OASIS-trained, no adaptation (the cross-domain baseline Dice)
#   tto         SVF test-time optimisation on IXI (NCC objective, adaptive stop)
#   tto_repair  the adapted field, then chained repair -> certified cross-domain warp
# EPS = certificate margin (from the fp32 eps-survival run; default 0.001). Sweep TTO steps via STEPS.
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
STEPS="${STEPS:-200}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
BASE="--model ctcf --ds IXI --use_test ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
TTO="--tto_mode svf --tto_steps ${STEPS} --tto_lr 0.01 --tto_w_ncc 1 --tto_stop both --tto_fold_k 1.25"
REPAIR="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
# OASIS-trained operating checkpoints, evaluated cross-domain on IXI test.
CKPTS="J15:P16_W1_VXM_OASIS_LBL_DIG_J15 NODIG:P16_W1_VXM_OASIS_LBL_NODIG"

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
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

echo "########## TTO cross-domain OASIS->IXI (eps=${EPS}, steps=${STEPS}) ##########"
for item in $CKPTS; do
  name="${item%%:*}"; exp="${item#*:}"
  # shellcheck disable=SC2086
  infer "${name}_feedfwd"    "$exp" $VM --tto_mode none
  # shellcheck disable=SC2086
  infer "${name}_tto"        "$exp" $VM $TTO
  # shellcheck disable=SC2086
  infer "${name}_tto_repair" "$exp" $VM $TTO $REPAIR
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done — run one plain pass for the table]"; exit 0; fi

echo
echo "===================== TTO CROSS-DOMAIN TABLE (IXI test) ====================="
printf "%-20s %8s %11s %13s %10s\n" "run" "dice" "tri_fold%" "tri_cert_bnd" "tto_steps"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-20s %8s %11s %13s %10s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)" "$(fmt tto_steps)"
done
echo
echo "READ: dice(tto) - dice(feedfwd) = the cross-domain adaptation gain (the 'beats everyone' axis)."
echo "  dice(tto_repair) vs dice(tto) = repair cost on the adapted field; tri_cert_bound>=eps => the"
echo "  cross-domain warp is certified too. Sweep STEPS to find where the Dice gain saturates."
