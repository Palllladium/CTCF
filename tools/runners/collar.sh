#!/usr/bin/env bash
# Identity-collar HOMEOMORPHISM upgrade (EVAL only). Closes the global-injectivity boundary route: taper the
# displacement to zero on a boundary shell so phi|boundary = id (trivially injective), then re-certify the
# interior with the chain repair. Interior fold-free cert + identity boundary => phi is GLOBALLY injective, a
# piecewise-trilinear HOMEOMORPHISM (Ball 1981 / Kroemer 2020) -- NOT a diffeomorphism (trilinear gradient
# jumps across cells). This is Path 1 (chosen over the fiddly edge/corner non-collision hand-proof).
#
# Per dataset: REPAIR (chain, no collar = the current certified baseline) vs COLLAR@width (chain + collar).
# WIN: COLLAR reaches boundary_max_disp ~ 0 (phi|boundary = id) AND tri_cert_bound >= eps (interior certified)
# at a Dice cost ~ 0 vs REPAIR (the collar only zeroes the FOV border = background). Then we may write
# "globally injective piecewise-trilinear homeomorphism". Width sweep shows the ramp is not fold-limited.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/collar}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
WIDTHS="${WIDTHS:-4 8}"          # collar taper width in voxels (sweep to show the ramp is not fold-limited)
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5 --tto_mode none"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"
collar_flags() { echo "--tto_collar 1 --tto_collar_width $1"; }
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

echo "########## Identity-collar homeomorphism upgrade (eps=${EPS}, widths={${WIDTHS}}) ##########"
# baselines: chain repair, no collar (boundary displacement stays as the network left it)
infer OAS_REPAIR "$OAS_EXP" --ds OASIS $CHAIN
infer IXI_REPAIR "$IXI_EXP" --ds IXI --use_test $CHAIN
for w in $WIDTHS; do
  # shellcheck disable=SC2086
  infer "OAS_COLLAR_w${w}" "$OAS_EXP" --ds OASIS $(collar_flags "$w") $CHAIN
  # shellcheck disable=SC2086
  infer "IXI_COLLAR_w${w}" "$IXI_EXP" --ds IXI --use_test $(collar_flags "$w") $CHAIN
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "===================== COLLAR TABLE ====================="
printf "%-16s %9s %13s %13s %12s\n" "run" "dice" "tri_cert_bnd" "bnd_tan_lip" "bnd_maxDisp"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-16s %9s %13s %13s %12s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_cert_bound)" "$(fmt boundary_tan_lip)" "$(fmt boundary_max_disp)"
done
echo
echo "READ: a COLLAR row with bnd_maxDisp ~ 0 (phi|boundary=id) AND tri_cert_bnd >= ${EPS} (interior certified)"
echo "  AND dice ~ its REPAIR row = GLOBAL injectivity closed => 'globally injective piecewise-trilinear"
echo "  homeomorphism' (NOT diffeomorphism). If dice drops or tri_cert < eps, widen the collar (gentler ramp)."
