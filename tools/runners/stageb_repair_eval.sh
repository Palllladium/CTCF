#!/usr/bin/env bash
# Stage-B post-repair eval (EVAL only, --2). Closes the open Wave-2 question: does training the DEPLOYED
# trilinear criterion (P17 Bernstein penalty) make the field REPAIR more cheaply / to a higher post-repair
# Dice than training the digital criterion (P16 digital penalty)? Feed-forward Dice already slightly favours
# digital (BERN_J* 0.9028-0.9038 < DIG_J* 0.9029-0.9048); the decision must be made POST-REPAIR, on the
# certified field we actually deploy. For each checkpoint: feed-forward Dice vs certified-repair Dice + cost.
# Same VM-Unified anchor + repair chain as injectivity.sh, so numbers are directly comparable.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/stageb}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 0 --gpu ${GPU} --print_every 5 --ds OASIS --tto_mode none"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"

# P16 = digital penalty (Wave-1); P17 = trilinear Bernstein penalty (Wave-2). NODIG = no topology penalty.
EXPS="P16_W1_VXM_OASIS_LBL_NODIG P16_W1_VXM_OASIS_LBL_DIG_J1 P16_W1_VXM_OASIS_LBL_DIG_J5 \
P16_W1_VXM_OASIS_LBL_DIG_J15 P17_W2_VXM_OASIS_TRI_BERN_J1 P17_W2_VXM_OASIS_TRI_BERN_J5 \
P17_W2_VXM_OASIS_TRI_BERN_J15"

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

echo "########## Stage-B post-repair eval (eps=${EPS}) ##########"
for e in $EXPS; do
  infer "${e}__FF"  "$e"                # feed-forward (may fold trilinearly)
  # shellcheck disable=SC2086
  infer "${e}__REP" "$e" $CHAIN         # certified trilinear repair
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "===================== STAGE-B TABLE (OASIS) ====================="
printf "%-38s %9s %13s %11s\n" "run" "dice" "tri_cert_bnd" "tri_fold%"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  printf "%-38s %9s %13s %11s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt tri_cert_bound)" "$(fmt tri_fold_pct)"
done
echo
echo "READ: per checkpoint compare __FF vs __REP dice (repair cost = REP-FF). The winner is the training"
echo "  penalty whose REPAIRED field has the highest certified Dice (cert_bnd >= ${EPS}). If a P17 Bernstein"
echo "  row repairs to a HIGHER certified Dice than the best P16 digital row, targeting the deployed criterion"
echo "  pays off post-repair (reverses the feed-forward ordering); else the digital penalty stays the pick."
