#!/usr/bin/env bash
# Phase 12-B — TTO round 2. Inference only, no training.
#
# Round 1 (results/tto/) scored OASIS on sdlogj alone, because the OASIS inference profile never
# emitted a fold count. It does now, so every OASIS conclusion from round 1 has to be re-measured on
# the topology axis. Blocks are independent and resumable (a run with a summary.csv is skipped).
#
#   BLOCK=A2 bash tools/runners/tto_phase2.sh     # one block
#   bash tools/runners/tto_phase2.sh              # all, in order
set -euo pipefail

PROFILE="${PROFILE:---2}"
GPU="${GPU:-0}"
PYBIN="${PYBIN:-python}"
STEPS="${STEPS:-400}"
TRACE="${TRACE:-5 10 25 50 100 200 400}"
OUT_ROOT="${OUT_ROOT:-results/tto2}"
BLOCK="${BLOCK:-ALL}"
HD95="${HD95:---hd95}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Arch flags passed to run() land after these, and argparse takes the last occurrence, so a block can
# override the config or the SVF switch just by naming them again.
BASE="--model ctcf --ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"

ck() { echo "results/$1/ckpt/best.pth"; }

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
    --print_every 5 --out_dir "$out" \
    "${arch[@]}" "${extra[@]}" "${tto[@]}"
}

# Architecture flags per Phase-11 checkpoint; they must match how each was trained.
arch_of() {
  case "$1" in
    HADAMARD) echo "--ctcf_l3_corr_mode hadamard" ;;
    L3CH64)   echo "--ctcf_l3_base_ch 64" ;;
    STACK)    echo "--ctcf_l3_corr_mode hadamard --ctcf_l3_base_ch 64" ;;
    *)        echo "" ;;
  esac
}

want() { [[ "$BLOCK" == "ALL" || "$BLOCK" == "$1" ]]; }

OASIS_CK=$(ck P10_LONGRUN_MAMBA_SVF_OASIS)
IXI_CK=$(ck P10_LONGRUN_MAMBA_SVF_IXI)
LADDER="CTRL_NONE HADAMARD L3CH64 STACK"

# F0 — what do the untouched checkpoints actually score on folds?                       (~30 min)
# No OASIS number in results/ has a fold count. Everything downstream needs this baseline.
if want F0; then
  echo "########## F0 — baselines, strict fold metric ##########"
  run "F0_OASIS_500ep" OASIS "$OASIS_CK" -- --tto_mode none
  run "F0_IXI_500ep"   IXI   "$IXI_CK"   -- --tto_mode none
  for CFG in $LADDER; do
    # shellcheck disable=SC2046
    run "F0_$CFG" OASIS "$(ck "P11_MAMBA_SVF_${CFG}_OASIS")" $(arch_of "$CFG") -- --tto_mode none
  done
fi

# F1 — the whole Paper-2 headline table, re-scored under the strict fold rule.          (~40 min)
# Paper 2 reports "0.00 % folds" on OASIS. That is the lenient central-difference count; the OASIS and
# IXI fold columns are not the same quantity. Nobody can decide what the paper should report without
# seeing the strict numbers for every headline row, and they cost minutes. No TTO, no model changes.
if want F1; then
  echo "########## F1 — Paper-2 headline configs, strict fold metric ##########"
  # $2 is deliberately unquoted: it carries several flags that must word-split.
  # shellcheck disable=SC2086
  f1() { run "F1_$1" "$3" "$(ck "$1")" $2 -- --tto_mode none; }
  f1 P10_LONGRUN_MAMBA_SVF_OASIS     "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"      OASIS
  f1 P10_LONGRUN_MAMBA_SVF_IXI       "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 1"      IXI
  f1 P10_LONGRUN_MAMBA_NOSVF_OASIS   "--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0"      OASIS
  f1 P10_LONGRUN_LKU8_SVF_OASIS      "--ctcf_config CTCF-CascadeA-LKU8 --ctcf_l3_svf 1"       OASIS
  f1 P10_LONGRUN_LKU8_SVF_IXI        "--ctcf_config CTCF-CascadeA-LKU8 --ctcf_l3_svf 1"       IXI
  f1 P10_LONGRUN_VXM_UNIFIED_SVF_OASIS "--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1" OASIS
  f1 P10_LONGRUN_VXM_UNIFIED_SVF_IXI   "--ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1" IXI
fi

# A2 — the Dice/topology trade-off curve. Round 1's Pareto was drawn on sdlogj and is void. (~6 h)
# A single-pair local probe says the two parameterisations are not on one curve at all: for the same
# pair, disp bought +0.0070 Dice at 6.4x the folds, svf bought +0.0061 at 1.00x. This block is the
# 19-pair version of that comparison, and it decides the paper's central claim.
if want A2; then
  echo "########## A2 — Dice vs folds, svf against disp ##########"
  for M in svf disp; do
    for LR in 0.003 0.01 0.03; do
      run "A2_OASIS__${M}_lr${LR/./p}" OASIS "$OASIS_CK" \
          -- --tto_mode "$M" --tto_steps "$STEPS" --tto_lr "$LR" --tto_trace $TRACE
    done
    # Round 1 never ran disp on IXI, so the claim rests on one dataset. One lr is enough to see
    # whether the fold cost separates the two parameterisations here as well.
    run "A2_IXI__${M}" IXI "$IXI_CK" -- --tto_mode "$M" --tto_steps "$STEPS" --tto_trace $TRACE
  done
fi

# I — is svf's residual fold count just discretisation error?                            (~4 h)
# svf left OASIS folds untouched (f0 0.367 %) but took IXI's from 0.036 % to 0.224 % — consistent with
# scaling-and-squaring contributing a roughly constant absolute floor rather than scaling with f0.
# exp(v) is a diffeomorphism in the continuum; 2^7 Euler steps on a grid are not. If the floor is
# discretisation, more integration steps must lower it, and the cost is one warp per step.
if want I; then
  echo "########## I — svf integration steps vs residual folds ##########"
  for N in 5 7 9 11; do
    run "I_IXI__int$N"   IXI   "$IXI_CK" \
        -- --tto_mode svf --tto_steps "$STEPS" --tto_svf_int_steps "$N" --tto_trace $TRACE
    run "I_OASIS__int$N" OASIS "$OASIS_CK" \
        -- --tto_mode svf --tto_steps "$STEPS" --tto_svf_int_steps "$N" --tto_trace $TRACE
  done
fi

# C2 — does the ladder still collapse when folds are the axis?                           (~3 h)
# Round 1: post-TTO Dice spread 0.0005, 0/6 pairs significant. If the four also land on different
# fold counts, the configs were never equivalent and the collapse was an artefact of the metric.
if want C2; then
  echo "########## C2 — Phase-11 ladder under TTO, strict folds ##########"
  for CFG in $LADDER; do
    # shellcheck disable=SC2046
    run "C2_$CFG" OASIS "$(ck "P11_MAMBA_SVF_${CFG}_OASIS")" $(arch_of "$CFG") \
        -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  done
fi

# E2 — is the parameterisation the only thing holding the field together?                (~3 h)
# Round 1's Block E switched off w_jac, which is a no-op: relu(-detJ) is already exactly 0 on these
# fields, so the penalty contributes no gradient. Diffusion (w_reg) does the work. Switch that off.
if want E2; then
  echo "########## E2 — svf vs disp with all regularisation off ##########"
  run "E2_svf_noreg"  OASIS "$OASIS_CK" \
      -- --tto_mode svf  --tto_steps "$STEPS" --tto_w_reg 0 --tto_w_jac 0 --tto_trace $TRACE
  run "E2_disp_noreg" OASIS "$OASIS_CK" \
      -- --tto_mode disp --tto_steps "$STEPS" --tto_w_reg 0 --tto_w_jac 0 --tto_trace $TRACE
  run "E2_svf_nojac"  OASIS "$OASIS_CK" \
      -- --tto_mode svf  --tto_steps "$STEPS" --tto_w_jac 0 --tto_trace $TRACE
  run "E2_disp_nojac" OASIS "$OASIS_CK" \
      -- --tto_mode disp --tto_steps "$STEPS" --tto_w_jac 0 --tto_trace $TRACE
fi

# S — adaptive stopping. Calibrates fold_k, which is currently a guess.                  (~5 h)
# A fixed step count cannot serve every dataset, and the fold budget is the only stopping signal that
# tracks the damage: the loss keeps falling on IXI for all 400 steps while Dice decays.
# fold_k=1.25 killed TTO outright in a local probe (guard fired at the first check), so the sweep
# spans an order of magnitude. Choose the winner on the VAL split, then report on test.
if want S; then
  echo "########## S — topology-budget stopping, fold_k sweep ##########"
  for K in 1.25 2.0 4.0 8.0; do
    for DS_CK in "OASIS:$OASIS_CK" "IXI:$IXI_CK"; do
      run "S_${DS_CK%%:*}__k${K/./p}" "${DS_CK%%:*}" "${DS_CK#*:}" \
          -- --tto_mode svf --tto_steps 800 --tto_stop topology \
             --tto_fold_k "$K" --tto_fold_check_every 5 --tto_trace $TRACE
    done
  done
  # The plateau guard on its own — expected to fail, and worth showing that it does.
  run "S_IXI__plateau"   IXI   "$IXI_CK"   -- --tto_mode svf --tto_steps 800 --tto_stop plateau
  run "S_OASIS__plateau" OASIS "$OASIS_CK" -- --tto_mode svf --tto_steps 800 --tto_stop plateau
fi

# D2 — domain adaptation, re-measured with folds.                                        (~6 h)
# Round 1: the IXI checkpoint gains +0.0356 on OASIS, the OASIS checkpoint only +0.0066 on IXI. What
# that cost in topology is unknown on the OASIS side.
if want D2; then
  echo "########## D2 — cross-dataset TTO, strict folds ##########"
  run "D2_IXIck_on_OASIS__none" OASIS "$IXI_CK"   -- --tto_mode none
  run "D2_IXIck_on_OASIS__svf"  OASIS "$IXI_CK"   -- --tto_mode svf --tto_steps 800 --tto_trace $TRACE 800
  run "D2_OASISck_on_IXI__none" IXI   "$OASIS_CK" -- --tto_mode none
  run "D2_OASISck_on_IXI__svf"  IXI   "$OASIS_CK" -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  # The guard applied to the case it was built for. Adapting to an unseen domain is the one setting
  # where nobody can tune the step count, because there are no labels to tune it against.
  run "D2_IXIck_on_OASIS__guard" OASIS "$IXI_CK" \
      -- --tto_mode svf --tto_steps 800 --tto_stop topology --tto_fold_k 2.0 --tto_fold_check_every 5
  run "D2_OASISck_on_IXI__guard" IXI  "$OASIS_CK" \
      -- --tto_mode svf --tto_steps 800 --tto_stop topology --tto_fold_k 2.0 --tto_fold_check_every 5
fi

# T — does the backbone matter at all under TTO?                                         (~3 h)
# The Phase-11 ladder collapse was within the Mamba family (a capacity axis). This is the same test
# across the whole backbone axis: the original Paper-1 Swin-DCA cascade is 296M — 25x the Mamba —
# and was the entire subject of Paper 1. TTO adds a fresh SVF velocity on top of flow0 regardless of
# how flow0 was produced, so it applies unchanged. Config is the l3_svf=False Swin-DCA preset.
# If 296M+TTO lands where 11.9M+TTO does (~0.835), the backbone is not a lever under TTO — which
# settles both "is there architectural headroom" and Paper 2's efficiency claim in one run.
# The control MUST reproduce the checkpoint's known feed-forward Dice; if it does not, the flags are
# wrong and the SVF number is meaningless — stop.
if want T; then
  echo "########## T — Swin-DCA (296M) under TTO vs Mamba (11.9M) ##########"
  SWIN="--ctcf_config CTCF-CascadeA --ctcf_l3_svf 0"
  # shellcheck disable=SC2086
  run "T_SWIN_OASIS__none" OASIS "$(ck CTCF_UPD_OASIS_E500)" $SWIN -- --tto_mode none
  # shellcheck disable=SC2086
  run "T_SWIN_OASIS__svf"  OASIS "$(ck CTCF_UPD_OASIS_E500)" $SWIN \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  # shellcheck disable=SC2086
  run "T_SWIN_IXI__none"   IXI   "$(ck CTCF_IXI_TUNED)" $SWIN -- --tto_mode none
  # shellcheck disable=SC2086
  run "T_SWIN_IXI__svf"    IXI   "$(ck CTCF_IXI_TUNED)" $SWIN \
      -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
fi

# R — does a smoother source field adapt better?                                         (~4 h)
# The asymmetry above tracks w_reg: IXI trains at 4.0, OASIS at 1.0. If smoothness is what travels,
# a reference checkpoint for domain adaptation must be trained smooth, not merely on more data.
# The Phase-11 M3 grid gives four OASIS checkpoints spanning sdlogj 0.078-0.095 at equal Dice. They
# vary per-level reg rather than global w_reg — a blunt instrument, but free, and it sets the
# direction before anyone spends 69 h on a global-w_reg longrun.
if want R; then
  echo "########## R — source-field smoothness vs adaptability ##########"
  for REG in REG_FLAT REG_MID REG_STRONG REG_VSTRONG; do
    CKPT=$(ck "P11_MAMBA_SVF_${REG}_OASIS")
    run "R_${REG}_on_IXI__none" IXI "$CKPT" -- --tto_mode none
    run "R_${REG}_on_IXI__svf"  IXI "$CKPT" -- --tto_mode svf --tto_steps "$STEPS" --tto_trace $TRACE
  done
fi

echo
echo "=================== RESULTS ==================="
printf "%-28s %8s %8s %8s %8s\n" "run" "dice" "j<=0%" "sdlogj" "steps"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%.4f",$2}' "$d/summary.csv"; }
  printf "%-28s %8s %8s %8s %8s\n" "$(basename "$d")" \
    "$(get dice_mean)" "$(get j_leq0_percent)" "$(get sdlogj)" "$(get tto_steps)"
done
echo
echo "Send back all of $OUT_ROOT/ (CSV/JSON only, a few MB)."
