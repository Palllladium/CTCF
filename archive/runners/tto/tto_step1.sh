#!/usr/bin/env bash
# Phase 14 — digital-topology control, Step 1: barrier, weight/eps/steps sweeps, IXI fix, masks.
#
# Step 0 (results/tto_digital/) proved the digital hinge repairs the cascade's folds free in Dice on
# OASIS, but not to zero (0.064 % at wj5), and that it BACKFIRES on IXI where feed-forward is already
# optimal (TTO adds folds, weak hinge can't keep up). Step 1 collects every remaining piece in one pass:
#   - how far a strong hinge / an eps margin / more steps drive folds toward zero, and the Dice cost;
#   - the relaxed log-barrier, which HOLDS the field admissible (level-3 guarantee) instead of nudging;
#   - the IXI fix: central control + matched w_reg=4.0 + early stop + guard;
#   - where residual folds live (brain vs eroded-brain fold%, now auto-emitted every run);
#   - the same on the VxM base (base decision).
#
# Every run now also emits j_leq0_brain_percent and j_leq0_brain_erode2_percent (folds inside the mask,
# and inside the mask minus a 2-voxel shell) — the claim scope and a boundary-vs-interior probe.
#
# Inference only. Blocks independent + resumable (a run with summary.csv is skipped).
#   BLOCK=H3 bash tools/runners/tto_step1.sh      # one block
#   bash tools/runners/tto_step1.sh               # all, in order
#
# COST: an OASIS run is ~17 min (19 pairs), an IXI run ~100 min (115 pairs). OASIS sweeps are cheap;
# IXI blocks (part of H4/H6/H7) are the expensive ones — run them selectively / overnight.
# Suggested order: H0 H3 H1 (core OASIS, ~4 h) -> H5 H2 -> H6 -> H4 H7 (IXI).
set -euo pipefail

PROFILE="${PROFILE:---2}"
GPU="${GPU:-0}"
PYBIN="${PYBIN:-python}"
STEPS="${STEPS:-400}"
TRACE="${TRACE:-5 10 25 50 100 200 400}"
TRACE800="${TRACE800:-5 10 25 50 100 200 400 800}"
OUT_ROOT="${OUT_ROOT:-results/tto_step1}"
BLOCK="${BLOCK:-ALL}"
HD95="${HD95:---hd95}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

BASE="--model ctcf --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"

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
  local tag="$1" ds="$2" ckpt="$3"; shift 3
  local arch=() tto=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do arch+=("$1"); shift; done
  shift || true
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

# H0 — dense hinge w_jac ladder, OASIS Mamba. Where is the knee folds->0, and where does Dice break?
# Re-runs 0.5/5.0 from Step 0 on purpose: this pass also emits the new brain/eroded fold% columns.  (~2 h)
if want H0; then
  echo "########## H0 — hinge w_jac ladder (OASIS Mamba) ##########"
  for WJ in 0.5 1 2 5 10 20 50; do
    run "H0_OASIS__digital_wj${WJ/./p}" OASIS "$OASIS_CK" -- \
        --tto_mode svf --tto_steps "$STEPS" --tto_jac_mode digital --tto_w_jac "$WJ" --tto_trace $TRACE
  done
fi

# H1 — eps margin: a soft one-sided barrier via the existing flag. Penalise det<eps, not just det<0.
# If eps>0 reaches ~0 folds, the true barrier may be unnecessary.                                 (~1.7 h)
if want H1; then
  echo "########## H1 — eps overcorrection margin (OASIS Mamba) ##########"
  for WJ in 5 20; do
    for EPS in 0.01 0.05 0.1; do
      run "H1_OASIS__digital_wj${WJ}_eps${EPS/./p}" OASIS "$OASIS_CK" -- \
          --tto_mode svf --tto_steps "$STEPS" --tto_jac_mode digital --tto_w_jac "$WJ" \
          --tto_jac_eps "$EPS" --tto_trace $TRACE
    done
  done
fi

# H2 — does time alone reach zero? OASIS Mamba, 800 steps (hinge fell monotonically to 400 at wj5).  (~1 h)
if want H2; then
  echo "########## H2 — 800 steps (OASIS Mamba) ##########"
  for WJ in 5 20; do
    run "H2_OASIS__digital_wj${WJ}_s800" OASIS "$OASIS_CK" -- \
        --tto_mode svf --tto_steps 800 --tto_jac_mode digital --tto_w_jac "$WJ" --tto_trace $TRACE800
  done
fi

# H3 — the relaxed log-barrier: holds the field admissible. mu (=w_jac) x engagement threshold t.
# Does it reach/keep near-zero folds at a lower weight than the hinge, without a Dice cost?         (~1.7 h)
if want H3; then
  echo "########## H3 — relaxed log-barrier (OASIS Mamba) ##########"
  for MU in 0.1 0.5 2; do
    for T in 0.05 0.1; do
      run "H3_OASIS__barrier_mu${MU/./p}_t${T/./p}" OASIS "$OASIS_CK" -- \
          --tto_mode svf --tto_steps "$STEPS" --tto_jac_mode barrier --tto_w_jac "$MU" \
          --tto_barrier_t "$T" --tto_trace $TRACE
    done
  done
fi

# H4 — the IXI fix. Step 0: digital hinge INCREASED IXI folds because svf-TTO adds them and IXI has no
# Dice headroom. Isolate the causes: a central control, matched w_reg=4.0 (TTO ran at 1.0, IXI trained
# at 4.0), an early stop at the Dice peak (~step 50), and the barrier.               (3 full + short, ~5 h)
if want H4; then
  echo "########## H4 — IXI fix (Mamba) ##########"
  run "H4_IXI__central_wj0p5"        IXI "$IXI_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode central --tto_w_jac 0.5 --tto_trace $TRACE
  run "H4_IXI__digital_wj0p5_wreg4"  IXI "$IXI_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode digital --tto_w_jac 0.5 --tto_w_reg 4.0 --tto_trace $TRACE
  run "H4_IXI__digital_wj5_s50"      IXI "$IXI_CK" -- --tto_mode svf --tto_steps 50 \
      --tto_jac_mode digital --tto_w_jac 5.0 --tto_trace 5 10 25 50
  run "H4_IXI__barrier_mu0p5_t0p1"   IXI "$IXI_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 --tto_trace $TRACE
fi

# H5 — masks. brain/eroded fold% are already emitted everywhere; this penalises INSIDE the eroded brain
# to see whether concentrating the objective there clears interior folds at a lower whole-volume cost. (~0.6 h)
if want H5; then
  echo "########## H5 — penalise inside the eroded brain (OASIS Mamba) ##########"
  run "H5_OASIS__digital_wj5_maskerode2" OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode digital --tto_w_jac 5.0 --tto_topo_mask 1 --tto_topo_erode 2 --tto_trace $TRACE
  run "H5_OASIS__barrier_mu0p5_maskerode2" OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 --tto_topo_mask 1 --tto_topo_erode 2 \
      --tto_trace $TRACE
fi

# H6 — the VxM base under the strong controls (hinge + barrier). Base decision vs Mamba. (~0.6 h OASIS + ~1.7 h IXI)
if want H6; then
  echo "########## H6 — VxM base (hinge + barrier) ##########"
  # shellcheck disable=SC2086
  run "H6_VXM_OASIS__digital_wj5"      OASIS "$VXM_OASIS_CK" $VXM -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode digital --tto_w_jac 5.0 --tto_trace $TRACE
  # shellcheck disable=SC2086
  run "H6_VXM_OASIS__barrier_mu0p5"    OASIS "$VXM_OASIS_CK" $VXM -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 --tto_trace $TRACE
  # shellcheck disable=SC2086
  run "H6_VXM_IXI__barrier_mu0p5"      IXI   "$VXM_IXI_CK"   $VXM -- --tto_mode svf --tto_steps "$STEPS" \
      --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 --tto_trace $TRACE
fi

# H7 — guard + active penalty. The rollback guard should stop IXI's fold growth early; on OASIS the
# penalty already prevents growth, so the guard should rarely fire. (OASIS ~0.3 h, IXI stops early)
if want H7; then
  echo "########## H7 — topology guard + digital penalty ##########"
  run "H7_OASIS__digital_wj5_guard" OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps 800 \
      --tto_jac_mode digital --tto_w_jac 5.0 --tto_stop topology --tto_fold_k 1.25 --tto_fold_check_every 5 \
      --tto_trace $TRACE
  run "H7_IXI__digital_wj5_guard"   IXI   "$IXI_CK" -- --tto_mode svf --tto_steps 800 \
      --tto_jac_mode digital --tto_w_jac 5.0 --tto_stop topology --tto_fold_k 1.25 --tto_fold_check_every 5 \
      --tto_trace $TRACE
fi

echo
echo "=================== RESULTS ==================="
printf "%-38s %8s %8s %9s %9s %8s\n" "run" "dice" "j<=0%" "brain%" "brainE2%" "steps"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%.4f",$2}' "$d/summary.csv"; }
  printf "%-38s %8s %8s %9s %9s %8s\n" "$(basename "$d")" \
    "$(get dice_mean)" "$(get j_leq0_percent)" "$(get j_leq0_brain_percent)" \
    "$(get j_leq0_brain_erode2_percent)" "$(get tto_steps)"
done
echo
echo "Send back all of $OUT_ROOT/ (CSV/JSON only, a few MB)."
