#!/usr/bin/env bash
# Phase 12-A — test-time optimisation (TTO) of the deformation field.
#
# The cascade emits a flow in one forward pass; TTO then keeps the WEIGHTS FROZEN and optimises
# the field itself for that one image pair. No training, no gradient ever touches the network:
# this is inference-side refinement.
#
# Blocks are ordered so the decisive answers land first, and are independent and resumable
# (a run whose summary.csv already exists is skipped), so you can stop after any of them.
#
#   BLOCK=A bash tools/runners/tto_ladder.sh      # one block
#   bash tools/runners/tto_ladder.sh              # all blocks, in order
#
# Everything is inference-only: ~20 h total on the RTX PRO 6000, vs 69 h for a single 500ep train.
set -euo pipefail

PROFILE="${PROFILE:---2}"          # advisor's machine
GPU="${GPU:-0}"
PYBIN="${PYBIN:-python}"
STEPS="${STEPS:-400}"
TRACE="${TRACE:-25 50 100 200 400}"
OUT_ROOT="${OUT_ROOT:-results/tto}"
BLOCK="${BLOCK:-ALL}"
HD95="${HD95:---hd95}"             # keep per_case.csv schema-compatible with results/SEDM/inference

# Required for the experiments.* package imports
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

MAMBA="--model ctcf --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"

# Checkpoints live where our own training code writes them: results/<EXP_NAME>/ckpt/best.pth
ck() { echo "results/$1/ckpt/best.pth"; }

run() {
  # run <tag> <ds> <ckpt> <arch-flags...> -- <tto-flags...>
  local tag="$1" ds="$2" ckpt="$3"; shift 3
  local arch=() tto=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do arch+=("$1"); shift; done
  shift || true
  tto=("$@")

  local out="$OUT_ROOT/$tag"
  if [[ -f "$out/summary.csv" ]]; then echo "[SKIP] $tag (already done)"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no checkpoint at $ckpt"; return 0; fi

  echo; echo "=== $tag ==="
  local extra=()
  [[ "$ds" == "IXI" ]] && extra+=(--use_test)   # IXI: the 115 held-out test pairs, not the val split
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference \
    $MAMBA --ds "$ds" "$PROFILE" --ckpt "$ckpt" --strict_ckpt 0 --gpu "$GPU" $HD95 \
    --print_every 5 --out_dir "$out" \
    "${arch[@]}" "${extra[@]}" "${tto[@]}"
}

want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

OASIS_CK=$(ck P10_LONGRUN_MAMBA_SVF_OASIS)  # Paper-2 headline: 500ep, Dice 0.8314, 0.00% folds
IXI_CK=$(ck P10_LONGRUN_MAMBA_SVF_IXI)      # same recipe trained on IXI: Dice 0.7628

# ---------------------------------------------------------------------------------------------
# BLOCK A — Does TTO work, and which parameterisation wins?                     (~4 h)
# The first run is a CONTROL: TTO off, must reproduce 0.8314 exactly. If it does not, STOP.
# `kan` / `randkan` are KAN-IDIR's own networks (MIT) plugged in as our residual, which makes
# "their network started from our cascade" vs "their network started from identity" a direct
# head-to-head on exactly the contribution we claim.
# ---------------------------------------------------------------------------------------------
if want A; then
  echo "########## BLOCK A — parameterisation sweep (headline checkpoint) ##########"
  run "A_HEAD__none"    OASIS "$OASIS_CK" -- --tto_mode none
  run "A_HEAD__svf"     OASIS "$OASIS_CK" -- --tto_mode svf     --tto_steps "$STEPS" --tto_trace $TRACE
  run "A_HEAD__disp"    OASIS "$OASIS_CK" -- --tto_mode disp    --tto_steps "$STEPS" --tto_trace $TRACE
  run "A_HEAD__inr"     OASIS "$OASIS_CK" -- --tto_mode inr     --tto_steps "$STEPS" --tto_trace $TRACE
  run "A_HEAD__kan"     OASIS "$OASIS_CK" -- --tto_mode kan     --tto_steps "$STEPS" --tto_trace $TRACE
  run "A_HEAD__randkan" OASIS "$OASIS_CK" -- --tto_mode randkan --tto_steps "$STEPS" --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
# BLOCK B — Learning rate and schedule.                                         (~2 h)
# TTO has one critical hyperparameter. Without this we cannot claim the result is anything other
# than a well- or badly-tuned optimiser.
# ---------------------------------------------------------------------------------------------
if want B; then
  echo "########## BLOCK B — TTO learning rate / schedule ##########"
  run "B_HEAD__svf_lr003"  OASIS "$OASIS_CK" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_lr 0.003 --tto_trace $TRACE
  run "B_HEAD__svf_lr030"  OASIS "$OASIS_CK" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_lr 0.03 --tto_trace $TRACE
  run "B_HEAD__svf_1cycle" OASIS "$OASIS_CK" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_lr_schedule onecycle --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
# BLOCK C — THE LADDER. The decisive experiment.                                (~3 h)
# Four Phase-11 checkpoints sit at four Dice levels (0.8257 / 0.8288 / 0.8296 / 0.8305).
# If TTO lifts them all to the SAME place, the architectural gains bought only what TTO supplies
# anyway -> bury L3CH64 and skip the 69 h STACK@500ep training run entirely.
# If the ladder survives, capacity is complementary to TTO and we pay for it.
# (Their no-TTO baselines already exist in results/SEDM/inference/ — no reruns needed.)
# ---------------------------------------------------------------------------------------------
if want C; then
  echo "########## BLOCK C — does TTO collapse the Phase-11 ladder? ##########"
  run "C_CTRL_NONE__svf" OASIS "$(ck P11_MAMBA_SVF_CTRL_NONE_OASIS)" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  run "C_HADAMARD__svf"  OASIS "$(ck P11_MAMBA_SVF_HADAMARD_OASIS)" \
      --ctcf_l3_corr_mode hadamard \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  run "C_L3CH64__svf"    OASIS "$(ck P11_MAMBA_SVF_L3CH64_OASIS)" \
      --ctcf_l3_base_ch 64 \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  run "C_STACK__svf"     OASIS "$(ck P11_MAMBA_SVF_STACK_OASIS)" \
      --ctcf_l3_corr_mode hadamard --ctcf_l3_base_ch 64 \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
# BLOCK D — DOMAIN ADAPTATION. Can one checkpoint + TTO replace a per-dataset one?   (~6 h)
# Run the OASIS-trained model on IXI and vice versa, with and without TTO. The weights never see
# the new dataset; only the field is re-optimised, per pair, unsupervised. If TTO recovers most of
# the cross-dataset drop, the claim is "one checkpoint + TTO ~= a dataset-specific checkpoint" —
# a far cheaper story than a foundation model, and uniGradICON manages only 0.791 on OASIS
# (below plain TransMorph), so the bar to clear is low.
# ---------------------------------------------------------------------------------------------
if want D; then
  echo "########## BLOCK D — cross-dataset adaptation without touching the weights ##########"
  run "D_OASISck_on_IXI__none" IXI   "$OASIS_CK" -- --tto_mode none
  run "D_OASISck_on_IXI__svf"  IXI   "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  run "D_IXIck_on_OASIS__none" OASIS "$IXI_CK"   -- --tto_mode none
  run "D_IXIck_on_OASIS__svf"  OASIS "$IXI_CK"   -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
# BLOCK E — The topology claim, tested head-on.                                 (~1.5 h)
# Switch the Jacobian penalty off entirely. A displacement field then has nothing holding it
# together and should fold badly; an SVF is a diffeomorphism by construction and should not.
# This is the experiment the paper's central claim rests on.
# ---------------------------------------------------------------------------------------------
if want E; then
  echo "########## BLOCK E — SVF vs disp with the Jacobian penalty OFF ##########"
  run "E_HEAD__svf_nojac"  OASIS "$OASIS_CK" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_w_jac 0 --tto_trace $TRACE
  run "E_HEAD__disp_nojac" OASIS "$OASIS_CK" \
      -- --tto_mode disp --tto_steps "$STEPS" --tto_w_jac 0 --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
# BLOCK F — Two tricks borrowed from KAN-IDIR: do they earn their keep?         (~1.5 h)
#   eps margin: penalise detJ < 0.1 rather than detJ < 0, pushing voxels back before they fold.
#   brain mask: optimise inside the brain only, ignoring the ~60% of the volume that is background.
#               Note this rescales the similarity term against the regulariser.
# ---------------------------------------------------------------------------------------------
if want F; then
  echo "########## BLOCK F — epsilon margin and brain-masked loss ##########"
  run "F_HEAD__svf_eps01" OASIS "$OASIS_CK" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_jac_eps 0.1 --tto_trace $TRACE
  run "F_HEAD__svf_mask"  OASIS "$OASIS_CK" \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_mask 1 --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
# BLOCK G — Native IXI, for the paper's second dataset.                         (~5 h)
# The most expensive block (115 test pairs). Run it last.
# ---------------------------------------------------------------------------------------------
if want G; then
  echo "########## BLOCK G — native IXI (115 test pairs) ##########"
  run "G_IXI__none" IXI "$IXI_CK" -- --tto_mode none
  run "G_IXI__svf"  IXI "$IXI_CK" -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
fi

# ---------------------------------------------------------------------------------------------
echo
echo "=================== TTO RESULTS ==================="
printf "%-32s %10s %10s %10s\n" "run" "dice" "sdlogj" "j<=0 %"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  dice=$(awk -F, '$1=="dice_mean"{printf "%.4f",$2}' "$d/summary.csv")
  sdl=$(awk -F, '$1=="sdlogj"{printf "%.4f",$2}' "$d/summary.csv")
  jl=$(awk -F, '$1=="j_leq0_percent"{printf "%.4f",$2}' "$d/summary.csv")
  printf "%-32s %10s %10s %10s\n" "$(basename "$d")" "${dice:--}" "${sdl:--}" "${jl:--}"
done
echo
echo "Send back the whole results/tto/ directory (CSV/JSON only, a few MB)."
