#!/usr/bin/env bash
# Loss-ablation EVAL (inference only). Scores the 8 loss_ablation.sh checkpoints by the metric of record =
# POST-repair certified Dice (the guarantee lives at inference, so the winner is the loss config whose REPAIRED
# field has the highest certified Dice, NOT the highest feed-forward Dice). For each checkpoint: feed-forward
# vs certified trilinear-repair, Dice + cert + fold%. Same VM-Unified anchor + chain repair as stageb_repair_eval
# so numbers are directly comparable. OASIS only (the ablation trains OASIS).
#
# The table is descriptive: conclusions must be computed from the observed rows, not declared in advance.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---3}"
PYBIN="${PYBIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/loss_ablation}"
FORCE="${FORCE:-0}"
EPS="${EPS:-0.001}"
_CALLNO=0
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
SMOKE="${SMOKE:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

VM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"
COMMON="--model ctcf ${PROFILE} --strict_ckpt 1 --deterministic --time_steps 6 --gpu ${GPU} --print_every 5 --ds OASIS --tto_mode none"
CHAIN="--tto_project 1 --tto_project_eps 0 --tto_tri_project 1 --tto_tri_project_eps ${EPS}"

EXPS="P18_ABL_VXM_OASIS_FULL P18_ABL_VXM_OASIS_NOICON P18_ABL_VXM_OASIS_NOJAC \
P18_ABL_VXM_OASIS_NOICON_NOJAC P18_ABL_VXM_OASIS_NOREG P18_ABL_VXM_OASIS_TRI_MEAN \
P18_ABL_VXM_OASIS_TRI_ACTIVE P18_ABL_VXM_OASIS_ICON_L2 \
P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.005 P18_ABL_VXM_OASIS_TRI_ACTIVE_W0.05"

ck() { echo "results/$1/ckpt/best.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local mine=$(( _CALLNO % NSHARD )); _CALLNO=$(( _CALLNO + 1 ))
  [[ "$NSHARD" -gt 1 && "$mine" != "$SHARD" ]] && return 0
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" && "$FORCE" != "1" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[FAIL] $tag — no ckpt at $ckpt" >&2; return 1; fi
  echo; echo "=== eval $tag ==="
  printf '[COMMAND] %q ' "${PYBIN}" -m experiments.inference $COMMON $VM --ckpt "$ckpt" --out_dir "$out" "$@"
  printf '\n'
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $COMMON $VM --ckpt "$ckpt" --out_dir "$out" "$@"
}

echo "########## Loss-ablation eval (eps=${EPS}) ##########"
for base_exp in $EXPS; do
  e="$base_exp"
  [[ "$SMOKE" == "1" ]] && e="${e}_SMOKE"
  infer "${e}__FF"  "$e"                # feed-forward (may fold trilinearly)
  # shellcheck disable=SC2086
  infer "${e}__REP" "$e" $CHAIN         # certified trilinear repair
done

if [[ "$NSHARD" -ne 1 ]]; then echo "[shard $SHARD/$NSHARD done]"; exit 0; fi

echo
echo "===================== LOSS-ABLATION TABLE (OASIS, post-repair = metric of record) ====================="
printf "%-38s %9s %13s %11s\n" "run" "dice" "tri_cert_min" "tri_fold%"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  get_col() { awk -F, -v k="$1" -v c="$2" '$1==k{printf "%s",$c}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.5f" "$v"; }
  cert_min="$(get_col tri_cert_bound 6)"; [[ -n "$cert_min" ]] || cert_min="$(get tri_cert_bound)"
  printf "%-38s %9s %13s %11s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$cert_min" "$(fmt tri_fold_pct)"
done
echo
echo "READ: among rows with cert_bnd >= ${EPS}, rank configurations by the observed __REP dice."
echo "This runner does not print a pre-declared scientific interpretation; derive it from the table."
