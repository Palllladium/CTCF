#!/usr/bin/env bash
# Stage-1 audit of OUR model zoo (EVAL only). The "floor": across methods and datasets, the surrogate
# fold-schemes (central-diff, digital-10) UNDER-report folds of the deployed trilinear warp that the sound
# Bernstein certificate catches; a certificate-gated repair removes them at small Dice cost. Every inference
# run ALREADY emits the full panel per case (j_leq0_central_percent, j_leq0_percent, tri_fold_pct,
# tri_cert_bound) + the repair path, so this is just inference feed-forward vs certified-repair over the zoo
# -- NO separate audit tool for our own models (field_audit.py is for FOREIGN saved fields). OASIS + IXI.
#
# ⚠️ CONFIRM the results/<EXP>/ckpt paths below against what is actually on this machine (names are from
# memory/log_layout; the config KEYS are verified against models/CTCF/configs.py). A wrong name -> [MISS].
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/stage1}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
DS="${DS:-OASIS IXI}"          # restrict with DS="OASIS" if IXI ckpts are elsewhere
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5 --tto_mode none"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"

# tag | ctcf_config | l3_svf | OASIS_exp | IXI_exp     (verified configs; CONFIRM the exp/ckpt names)
ENTRIES=(
  "MAMBA_SVF|CTCF-CascadeA-Mamba|1|P10_LONGRUN_MAMBA_SVF_OASIS|P10_LONGRUN_MAMBA_SVF_IXI"
  "VXM_SVF|CTCF-CascadeA-VM-Unified|1|P10_LONGRUN_VXM_UNIFIED_SVF_OASIS|P10_LONGRUN_VXM_UNIFIED_SVF_IXI"
  "LKU8_SVF|CTCF-CascadeA-LKU8|1|P10_LONGRUN_LKU8_SVF_OASIS|P10_LONGRUN_LKU8_SVF_IXI"
)

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
  "${PYBIN}" -m experiments.inference $COMMON --ckpt "$ckpt" --out_dir "$out" "$@"
}

echo "########## Stage-1 zoo audit (eps=${EPS}, ds={${DS}}) ##########"
for row in "${ENTRIES[@]}"; do
  IFS='|' read -r tag cfg svf oas ixi <<< "$row"
  MODEL="--ctcf_config $cfg --ctcf_l3_svf $svf"
  for ds in $DS; do
    if [[ "$ds" == "OASIS" ]]; then exp="$oas"; dsflags="--ds OASIS"; else exp="$ixi"; dsflags="--ds IXI --use_test"; fi
    # shellcheck disable=SC2086
    infer "${tag}_${ds}_FF"  "$exp" $MODEL $dsflags
    # shellcheck disable=SC2086
    infer "${tag}_${ds}_REP" "$exp" $MODEL $dsflags $CHAIN
  done
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "===== STAGE-1 DISCREPANCY + REPAIR TABLE ====="
printf "%-22s %8s %10s %10s %11s %12s\n" "run" "dice" "central%" "digital10%" "tri_fold%" "tri_cert_bnd"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-22s %8s %10s %10s %11s %12s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt j_leq0_central_percent)" "$(fmt j_leq0_percent)" \
    "$(fmt tri_fold_pct)" "$(fmt tri_cert_bound)"
done
echo
echo "READ: on each _FF row, central% and digital10% read ~0 while tri_cert_bnd < 0 and tri_fold% > 0 = the"
echo "  surrogate schemes MISS folds the certificate catches (the Stage-1 headline). The paired _REP row is"
echo "  certified (tri_cert_bnd >= ${EPS}) at a Dice cost = REP-FF. Repeat the story across every method/ds row."
