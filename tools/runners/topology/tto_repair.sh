#!/usr/bin/env bash
# Trilinear-aware repair of the deployed warp (EVAL only).
#
# The gate proved every digital-certified field folds under the trilinear warp grid_sample applies.
# trilinear_project heuristically contracts cells whose sufficient Bernstein bound is below eps. A successful
# operational report still has to be checked on the saved float32 bytes by utils.cert_exact before it becomes
# the machine-checked result used in a claim. This runner asks:
#
#   Does the repair satisfy the sufficient predicate (tri_cert_bound >= eps) at a Dice cost
#   we can defend (target <= ~0.003 vs feed-forward)?
#
# Per checkpoint, four fields side by side:
#   feedfwd        raw field (baseline Dice, its trilinear fold%)
#   dproj_e0       digital projection (current mechanism — certifies the WRONG criterion, still folds)
#   tri_e0/e02     trilinear repair of the RAW field (fixes thousands of cells -> costly)
#   chain_e0/e02   digital project (cheap bulk, thousands->~7 trilinear folds) THEN trilinear repair the
#                  sparse residual -> the same certificate at (predicted) ~half the Dice cost
# Checkpoints: A2 100ep champion + the P16 500ep operating candidates (J5/J15 win post-repair; NODIG for
# contrast). All VxM Unified SVF, at default results/<EXP>/ckpt paths.
#
# Parallel (one process per card; SHARD is a logical 0..NSHARD-1 index, card is CUDA_VISIBLE_DEVICES):
#   cards="2 3 4 5 6 7"; i=0
#   for dev in $cards; do
#     CUDA_VISIBLE_DEVICES=$dev SHARD=$i NSHARD=6 GPU=0 PROFILE=--3 \
#       nohup bash tools/runners/topology/tto_repair.sh > rep_s$i.log 2>&1 &
#     i=$((i+1))
#   done; wait
#   bash tools/runners/topology/tto_repair.sh  # NSHARD=1: all SKIP, prints the table
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_repair}"
FORCE="${FORCE:-0}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
_CALLNO=0

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
BASE="--model ctcf --ds OASIS ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
# NAME:EXP — all VxM Unified SVF. A2 = 100ep labels+digital champion; P16 = 500ep weak-sup w_jac sweep.
CKPTS="A2:P14_2A_VXM_DICE_DIGITAL_OASIS NODIG:P16_W1_VXM_OASIS_LBL_NODIG \
       J5:P16_W1_VXM_OASIS_LBL_DIG_J5 J15:P16_W1_VXM_OASIS_LBL_DIG_J15"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local mine=$(( _CALLNO % NSHARD )); _CALLNO=$(( _CALLNO + 1 ))
  [[ "$NSHARD" -gt 1 && "$mine" != "$SHARD" ]] && return 0
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" && "$FORCE" != "1" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[FAIL] $tag — no ckpt at $ckpt" >&2; return 1; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

echo "########## MAKE-OR-BREAK: trilinear repair vs digital projection ##########"
for item in $CKPTS; do
  name="${item%%:*}"; exp="${item#*:}"
  # shellcheck disable=SC2086
  infer "${name}_feedfwd" "$exp" $VM --tto_mode none
  # shellcheck disable=SC2086
  infer "${name}_dproj_e0" "$exp" $VM --tto_mode none --tto_project 1 --tto_project_eps 0
  # shellcheck disable=SC2086
  infer "${name}_tri_e0"  "$exp" $VM --tto_mode none --tto_tri_project 1 --tto_tri_project_eps 0
  # shellcheck disable=SC2086
  infer "${name}_tri_e02" "$exp" $VM --tto_mode none --tto_tri_project 1 --tto_tri_project_eps 0.02
  # chained: cheap digital bulk removal, then trilinear repair certifies the sparse residual
  # shellcheck disable=SC2086
  infer "${name}_chain_e0"  "$exp" $VM --tto_mode none --tto_project 1 --tto_project_eps 0 \
        --tto_tri_project 1 --tto_tri_project_eps 0
  # shellcheck disable=SC2086
  infer "${name}_chain_e02" "$exp" $VM --tto_mode none --tto_project 1 --tto_project_eps 0 \
        --tto_tri_project 1 --tto_tri_project_eps 0.02
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done — run one plain pass for the table]"; exit 0; fi

echo
echo "===================== TRILINEAR REPAIR TABLE ====================="
printf "%-18s %8s %11s %13s %10s %12s\n" "run" "dice" "tri_fold%" "tri_cert_min" "tri_iters" "uncert_cells"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  get_col() { awk -F, -v k="$1" -v c="$2" '$1==k{printf "%s",$c}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  cert_min="$(get_col tri_cert_bound 6)"; [[ -n "$cert_min" ]] || cert_min="$(get tri_cert_bound)"
  uncert_max="$(get_col tri_proj_uncertified_cells 7)"
  [[ -n "$uncert_max" ]] || uncert_max="$(get tri_proj_uncertified_cells)"
  printf "%-18s %8s %11s %13s %10s %12s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$cert_min" \
    "$(fmt tri_proj_iters)" "${uncert_max:--}"
done
echo
echo "READ: an operational pass requires tri_cert_min >= eps and uncert_cells = 0 for every case."
echo "  Save the fields and run utils.cert_exact on their float32 bytes before making a machine-sound claim."
echo "  Dice delta vs the same *_feedfwd row is the empirical cost of repair."
echo "Eval CSV: $OUT_ROOT/ (send back)."
