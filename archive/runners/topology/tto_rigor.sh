#!/usr/bin/env bash
# Phase 17e — RIGOR batch (EVAL only, --2). Measures the two new certificate upgrades on REAL fields, which
# synthetic tests cannot settle:
#   (1) Bernstein SUBDIVISION (--tto_tri_subdiv_depth): does refining the conservative per-cell bound on the
#       operating J15 field cut needless repair contractions (tri_proj_iters) and its Dice cost, and tighten
#       the reported tri_cert_bound? SD0 (coarse) vs SD2 vs SD3 on the SAME chained repair.
#   (2) GLOBAL injectivity (Ball/Kroemer): the new columns disp_grad_norm (<1 alone certifies global
#       injectivity) and boundary_max_disp (small => phi=id on the boundary => global injectivity with the
#       interior fold-free cert). Reported feed-forward AND post-repair — does the repaired warp qualify?
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/tto_rigor}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
OAS_EXP="${OAS_EXP:-P16_W1_VXM_OASIS_LBL_DIG_J15}"   # the operating (J15) field

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
  "${PYBIN}" -m experiments.inference $COMMON $VM --ckpt "$ckpt" --out_dir "$out" --ds OASIS "$@"
}

echo "########## RIGOR batch (eps=${EPS}, ckpt=${OAS_EXP}) ##########"
infer FEEDFWD    "$OAS_EXP" --tto_mode none
# shellcheck disable=SC2086
infer REPAIR_SD0 "$OAS_EXP" --tto_mode none $CHAIN --tto_tri_subdiv_depth 0
# shellcheck disable=SC2086
infer REPAIR_SD2 "$OAS_EXP" --tto_mode none $CHAIN --tto_tri_subdiv_depth 2
# shellcheck disable=SC2086
infer REPAIR_SD3 "$OAS_EXP" --tto_mode none $CHAIN --tto_tri_subdiv_depth 3

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "======================== RIGOR TABLE ========================"
printf "%-12s %9s %11s %13s %11s %9s %11s\n" \
  "run" "dice" "tri_fold%" "tri_cert_bnd" "tri_iters" "dispGrad" "bndMaxDisp"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-12s %9s %11s %13s %11s %9s %11s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)" \
    "$(fmt tri_proj_iters)" "$(fmt disp_grad_norm)" "$(fmt boundary_max_disp)"
done
echo
echo "READ: SD0->SD2->SD3 = does subdivision shrink tri_iters / recover Dice / raise tri_cert_bound?"
echo "  (if flat, the coarse bound was already tight on real fields — a real, publishable negative result)."
echo "  disp_grad_norm < 1 => the field is GLOBALLY injective (a genuine diffeomorphism, not just fold-free);"
echo "  else read boundary_max_disp: ~0 => id on the boundary => global injectivity via the interior cert."
