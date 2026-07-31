#!/usr/bin/env bash
# Phase 13 — digital-topology control, Step 0: the objective function.
#
# The advisor's question was "think about the objective function". The w_jac penalty we ship acts on
# the central-difference detJ, which stays positive — hence contributes no gradient — on exactly the
# SVF fields we train (min central detJ > 0.1 while the digital criterion still counts 0.5 % folds).
# So topology in TTO is currently only *guarded* (rolled back), never *driven down*.
#
# This block replaces that penalty with a differentiable hinge on the ten Liu-et-al. (IJCV 2024)
# determinants — the same ten the strict fold metric counts — via --tto_jac_mode digital. The
# question it answers, cheaply and before any training or barrier work (Step 1):
#     does a digital-aware objective drive the cascade's residual folds toward zero, and at what
#     Dice cost, starting from the cascade's own SVF field?
# If the hinge cannot repair the field to ~0 folds here, the log-barrier guarantee cannot either.
#
# Inference only, no training. Blocks are independent and resumable (a run with summary.csv is skipped).
#   BLOCK=G1 bash tools/runners/tto_digital.sh     # one block
#   bash tools/runners/tto_digital.sh              # all, in order
set -euo pipefail

PROFILE="${PROFILE:---2}"
GPU="${GPU:-0}"
PYBIN="${PYBIN:-python}"
STEPS="${STEPS:-400}"
TRACE="${TRACE:-5 10 25 50 100 200 400}"
OUT_ROOT="${OUT_ROOT:-results/tto_digital}"
BLOCK="${BLOCK:-ALL}"
HD95="${HD95:---hd95}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

BASE="--model ctcf --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"

# Mamba P10 runs store ckpt/best.pth; older flat layouts store best.pth[.tar].
ck() {
  local folder="results/$1" sub name p
  for sub in "ckpt/" ""; do
    for name in best.pth best.pth.tar last.pth last.pth.tar; do
      p="$folder/$sub$name"
      [[ -f "$p" ]] && { echo "$p"; return 0; }
    done
  done
  echo "$folder/ckpt/best.pth"
}

run() {
  # run <tag> <ds> <ckpt> <arch-flags...> -- <tto-flags...>
  # Arch flags land after $BASE; argparse takes the last occurrence, so a block overrides the
  # backbone just by naming --ctcf_config again (e.g. the VxM block).
  local tag="$1" ds="$2" ckpt="$3"; shift 3
  local arch=() tto=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do arch+=("$1"); shift; done
  shift || true  # drop the "--"
  tto=("$@")

  local out="$OUT_ROOT/$tag"
  if [[ -f "$out/summary.csv" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no checkpoint at $ckpt"; return 0; fi

  echo; echo "=== $tag ==="
  local extra=()
  [[ "$ds" == "IXI" ]] && extra+=(--use_test)
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference \
    $BASE --ds "$ds" "$PROFILE" --ckpt "$ckpt" --strict_ckpt 0 --gpu "$GPU" $HD95 \
    --print_every 5 --out_dir "$out" "${arch[@]}" "${extra[@]}" "${tto[@]}"
}

want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

OASIS_CK=$(ck P10_LONGRUN_MAMBA_SVF_OASIS)
IXI_CK=$(ck P10_LONGRUN_MAMBA_SVF_IXI)
VXM_OASIS_CK=$(ck P10_LONGRUN_VXM_UNIFIED_SVF_OASIS)
VXM_IXI_CK=$(ck P10_LONGRUN_VXM_UNIFIED_SVF_IXI)
VXM="--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1"

# G1 — feasibility + the central-vs-digital contrast + a w_jac ladder.                    (~2 h)
# 'none' is the untouched checkpoint. 'central' is svf-TTO with the shipped penalty: it must leave
# folds essentially where svf-TTO alone leaves them (the penalty is inert), and it is the control that
# proves the digital arm's fold reduction comes from the objective, not from the extra optimisation.
# The digital ladder traces folds vs Dice as the hinge strengthens. w_jac is swept wide because the
# right scale is unknown: the penalty is a mean over voxels that is ~0 except on the folding fraction.
if want G1; then
  echo "########## G1 — digital objective vs inert central, w_jac ladder ##########"
  run "G1_OASIS__none"           OASIS "$OASIS_CK" -- --tto_mode none
  run "G1_OASIS__central_wj0p5"  OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode central --tto_w_jac 0.5 --tto_trace $TRACE
  for WJ in 0.05 0.5 5.0; do
    run "G1_OASIS__digital_wj${WJ/./p}" OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" \
        --tto_jac_mode digital --tto_w_jac "$WJ" --tto_trace $TRACE
  done

  run "G1_IXI__none"            IXI "$IXI_CK" -- --tto_mode none
  for WJ in 0.5 5.0; do
    run "G1_IXI__digital_wj${WJ/./p}" IXI "$IXI_CK" -- --tto_mode svf --tto_steps "$STEPS" \
        --tto_jac_mode digital --tto_w_jac "$WJ" --tto_trace $TRACE
  done
fi

# G2 — where should topology be controlled: whole volume or brain ROI?                    (~30 min)
# The penalty defaults to the whole interior (matches the reported fold%%, and is stricter). This runs
# the same operating point restricted to the brain mask — the eventual claim scope, consistent with how
# NDV is already measured. If brain-masked reaches ~0 brain folds at a lower Dice cost, that is the
# operating point to carry into Step 1.
if want G2; then
  echo "########## G2 — brain-ROI penalty vs whole-volume ##########"
  run "G2_OASIS__digital_wj0p5_mask" OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode digital --tto_w_jac 0.5 --tto_topo_mask 1 --tto_trace $TRACE
fi

# G3 — is the repair backbone-agnostic, and is the lighter backbone the better base?      (~1 h)
# VxM Unified is 9.24M vs Mamba's 11.91M, ~23 %% faster (709 vs 918 ms/iter), and starts with FEWER
# folds (digital-10 0.391 %% vs 0.510 %%) at ~0.0035 lower OASIS Dice / higher IXI Dice. Under TTO the
# backbone axis already collapses (phase12b), so if the digital hinge repairs VxM's field just as it
# repairs Mamba's, the lighter backbone is the cleaner base for the topology story. Same operating
# point as G1's digital w_jac=0.5, one probe per dataset.
if want G3; then
  echo "########## G3 — digital repair on the lighter VxM base ##########"
  # shellcheck disable=SC2086
  run "G3_VXM_OASIS__none"          OASIS "$VXM_OASIS_CK" $VXM -- --tto_mode none
  # shellcheck disable=SC2086
  run "G3_VXM_OASIS__digital_wj0p5" OASIS "$VXM_OASIS_CK" $VXM -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode digital --tto_w_jac 0.5 --tto_trace $TRACE
  # shellcheck disable=SC2086
  run "G3_VXM_IXI__none"            IXI   "$VXM_IXI_CK" $VXM -- --tto_mode none
  # shellcheck disable=SC2086
  run "G3_VXM_IXI__digital_wj0p5"   IXI   "$VXM_IXI_CK" $VXM -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode digital --tto_w_jac 0.5 --tto_trace $TRACE
fi

echo
echo "=================== RESULTS ==================="
printf "%-34s %8s %8s %8s %8s\n" "run" "dice" "j<=0%" "sdlogj" "steps"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%.4f",$2}' "$d/summary.csv"; }
  printf "%-34s %8s %8s %8s %8s\n" "$(basename "$d")" \
    "$(get dice_mean)" "$(get j_leq0_percent)" "$(get sdlogj)" "$(get tto_steps)"
done
echo
echo "Send back all of $OUT_ROOT/ (CSV/JSON only, a few MB)."
