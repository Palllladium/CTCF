#!/usr/bin/env bash
# Phase 15d — topology-mechanism probe (EVAL only, --2 box, archived 100ep checkpoints).
#
# Back-planned from the TMI-complete claim we are trying to earn the right to write:
#   "A post-hoc projection maps ANY displacement field onto the digital-diffeomorphic set with a
#    CERTIFIED MARGIN (min over the ten Liu determinants, all voxels >= eps > 0), at a Dice cost we
#    CHARACTERISE as a function of input fold density, ACROSS architectures and datasets, WITHOUT any
#    test-time optimisation (TTO instead buys a separate, quantified Dice/domain-adaptation benefit)."
# Every clause of that sentence is one block; each runs on a checkpoint that ALREADY exists at the
# standard results/<EXP>/ckpt path (names lifted from phase10_inference.sh) — nothing to restore:
#   DECOMP  isolate each phase on A2 -> is TTO needed for the GUARANTEE? (proj-only vs two-phase)
#   UNIV    projection-only across training regimes + resolve the A1 eps=0.02/80it residual
#   STRESS  Dice cost vs input fold density (Mamba NoSVF ~2.2%) -> the high-fold end of the envelope
#   CROSS   the same projection on Mamba (SSM) and LKU8 (CNN) -> architecture-agnostic certifier
#   DSET    the same projection on VxM Unified / IXI (test) -> dataset-agnostic certifier
#   DOMAIN  OASIS-trained Mamba -> IXI (out-of-domain, folds more) -> guarantee under distribution shift
# BLOCK=CORE (default) = DECOMP+UNIV on A0/A1/A2 now; BLOCK=ALL = every block; or name one block.
#
# NEW column cert_min_det is the MARGIN the fold count hides: min(det) over the ten maps and every
# voxel of the FINAL field. >=eps proves diffeomorphic with headroom; <0 means the field still folds.
# A0/A1/A2 @100ep were restored to results/<EXP>/ckpt for tto_certify and are reused here as-is.
set -e

GPU="${GPU:-0}"
PROFILE="${PROFILE:---2}"
PYBIN="${PYBIN:-python}"
BLOCK="${BLOCK:-CORE}"                 # CORE = DECOMP+UNIV (run now) | DECOMP | UNIV | STRESS | CROSS | DSET | ALL
OUT_ROOT="${OUT_ROOT:-results/tto_probe}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
want() {
  case "$BLOCK" in
    ALL)  return 0 ;;
    CORE) [[ "$1" == DECOMP || "$1" == UNIV ]] ;;
    *)    [[ "$BLOCK" == "$1" ]] ;;
  esac
}

A0_EXP="P10_LONGRUN_VXM_UNIFIED_SVF_OASIS"     # unsup (no labels, no digital penalty)
A1_EXP="P14_2A_VXM_DICE_OASIS"                 # labels, no digital penalty
A2_EXP="P14_2A_VXM_DICE_DIGITAL_OASIS"         # labels + digital penalty (champion field)

BASE="--model ctcf --ctcf_config CTCF-CascadeA-VM-Unified --ctcf_l3_svf 1 --ds OASIS ${PROFILE} \
      --strict_ckpt 0 --gpu ${GPU} --print_every 5"
# two-phase = proximal barrier (a32, Dice-preserving) then feathered projection.
PROX="--tto_mode svf --tto_steps 400 --tto_jac_mode barrier --tto_w_jac 0.5 --tto_barrier_t 0.1 \
      --tto_w_ncc 0 --tto_w_reg 0 --tto_anchor_w 32"

ck() { local p="results/$1/ckpt/best.pth"; [[ -f "$p" ]] && echo "$p" || echo "results/$1/ckpt/last.pth"; }
infer() {
  local tag="$1" exp="$2"; shift 2
  local out="$OUT_ROOT/$tag" ckpt; ckpt="$(ck "$exp")"
  if [[ -f "$out/summary.csv" ]]; then echo "[SKIP] $tag"; return 0; fi
  if [[ ! -f "$ckpt" ]]; then echo "[MISS] $tag — no ckpt at $ckpt (restore from archive)"; return 0; fi
  echo; echo "=== eval $tag ==="
  # shellcheck disable=SC2086
  "${PYBIN}" -m experiments.inference $BASE --ckpt "$ckpt" --out_dir "$out" "$@"
}

# ---- DECOMP: isolate each phase on A2. Is projection-alone Dice-neutral AND certifying? Then TTO is
#      not needed for the topology guarantee (it stays only as the domain-adaptation Dice lever). ----
if want DECOMP; then
  echo "########## DECOMP: per-phase isolation on A2 (labels+digital) ##########"
  # shellcheck disable=SC2086
  infer A2_feedfwd      "$A2_EXP" --tto_mode none
  # shellcheck disable=SC2086
  infer A2_projonly_e0  "$A2_EXP" --tto_mode none --tto_project 1 --tto_project_eps 0
  # shellcheck disable=SC2086
  infer A2_projonly_e05 "$A2_EXP" --tto_mode none --tto_project 1 --tto_project_eps 0.05
  # shellcheck disable=SC2086
  infer A2_barrier      "$A2_EXP" $PROX
  # shellcheck disable=SC2086
  infer A2_twophase_e0  "$A2_EXP" $PROX --tto_project 1 --tto_project_eps 0
fi

# ---- UNIV: projection-alone across training regimes (does it universalise without the barrier?) and
#      resolve the one open wrinkle — the A1 residual at eps=0.02/80it — with eps=0 and 200 iters. ----
if want UNIV; then
  echo "########## UNIV: projection-only across regimes + A1 residual resolution ##########"
  # shellcheck disable=SC2086
  infer A0_projonly_e0  "$A0_EXP" --tto_mode none --tto_project 1 --tto_project_eps 0
  # shellcheck disable=SC2086
  infer A1_projonly_e0  "$A1_EXP" --tto_mode none --tto_project 1 --tto_project_eps 0
  # shellcheck disable=SC2086
  infer A1_2ph_e0_it200 "$A1_EXP" $PROX --tto_project 1 --tto_project_eps 0 --tto_project_iters 200
fi

# ---- STRESS: Dice cost vs INPUT fold density — the high-fold end of the operating envelope. Mamba
#      NoSVF folds ~2.2% (strict digital-10) vs the ~0.4-0.5% of every SVF field above. ----
HIGHFOLD_EXP="${HIGHFOLD_EXP:-P10_LONGRUN_MAMBA_NOSVF_OASIS}"
[[ -n "${HIGHFOLD_EXTRA+x}" ]] || HIGHFOLD_EXTRA="--ctcf_config CTCF-CascadeA-Mamba --ctcf_l3_svf 0"
if want STRESS; then
  echo "########## STRESS: Dice cost vs fold density on ${HIGHFOLD_EXP} ##########"
  # shellcheck disable=SC2086
  infer HF_feedfwd      "$HIGHFOLD_EXP" $HIGHFOLD_EXTRA --tto_mode none
  # shellcheck disable=SC2086
  infer HF_projonly_e0  "$HIGHFOLD_EXP" $HIGHFOLD_EXTRA --tto_mode none --tto_project 1 --tto_project_eps 0
  # shellcheck disable=SC2086
  infer HF_projonly_e05 "$HIGHFOLD_EXP" $HIGHFOLD_EXTRA --tto_mode none --tto_project 1 --tto_project_eps 0.05
fi

# ---- CROSS: architecture-agnostic certifier — one projector over three L2 families (VxM=UNet above,
#      Mamba=SSM, LKU8=large-kernel CNN). ARCH_LIST items are NAME:EXP:CTCF_CONFIG. ----
ARCH_LIST="${ARCH_LIST:-MAMBA:P10_LONGRUN_MAMBA_SVF_OASIS:CTCF-CascadeA-Mamba LKU8:P10_LONGRUN_LKU8_SVF_OASIS:CTCF-CascadeA-LKU8}"
if want CROSS; then
  echo "########## CROSS: architecture-agnostic certifier ##########"
  for item in $ARCH_LIST; do
    name="${item%%:*}"; rest="${item#*:}"; exp="${rest%%:*}"; cfg="${rest#*:}"
    # shellcheck disable=SC2086
    infer "${name}_feedfwd"     "$exp" --ctcf_config "$cfg" --tto_mode none
    # shellcheck disable=SC2086
    infer "${name}_projonly_e0" "$exp" --ctcf_config "$cfg" --tto_mode none --tto_project 1 --tto_project_eps 0
  done
fi

# ---- DSET: dataset-agnostic certifier — same VxM Unified architecture, trained + eval on IXI. ----
IXI_EXP="${IXI_EXP:-P10_LONGRUN_VXM_UNIFIED_SVF_IXI}"
if want DSET; then
  echo "########## DSET: dataset-agnostic certifier on IXI (test) ##########"
  # shellcheck disable=SC2086
  infer IXI_feedfwd     "$IXI_EXP" --ds IXI --use_test --tto_mode none
  # shellcheck disable=SC2086
  infer IXI_projonly_e0 "$IXI_EXP" --ds IXI --use_test --tto_mode none --tto_project 1 --tto_project_eps 0
fi

# ---- DOMAIN: out-of-domain certifier — OASIS-trained Mamba evaluated on IXI (a domain-shifted field
#      that folds more). Does the guarantee survive distribution shift? Feeds the TTO x topology axis. --
DOMAIN_EXP="${DOMAIN_EXP:-P10_LONGRUN_MAMBA_SVF_OASIS}"
if want DOMAIN; then
  echo "########## DOMAIN: OASIS-trained Mamba -> IXI (out-of-domain) ##########"
  # shellcheck disable=SC2086
  infer DOM_o2i_feedfwd     "$DOMAIN_EXP" --ctcf_config CTCF-CascadeA-Mamba --ds IXI --use_test --tto_mode none
  # shellcheck disable=SC2086
  infer DOM_o2i_projonly_e0 "$DOMAIN_EXP" --ctcf_config CTCF-CascadeA-Mamba --ds IXI --use_test --tto_mode none --tto_project 1 --tto_project_eps 0
fi

echo
echo "===================== TTO-PROBE GATE TABLE ====================="
printf "%-22s %8s %10s %12s %10s %9s\n" "run" "dice" "digital10" "cert_min_det" "proj_fold%" "proj_iter"
for d in "$OUT_ROOT"/*/; do
  [[ -f "$d/summary.csv" ]] || continue
  get() { awk -F, -v k="$1" '$1==k{printf "%s",$2}' "$d/summary.csv"; }
  fmt() { local v; v="$(get "$1")"; [[ -z "$v" ]] && echo "-" || printf "%.4f" "$v"; }
  printf "%-22s %8s %10s %12s %10s %9s\n" "$(basename "$d")" \
    "$(fmt dice_mean)" "$(fmt j_leq0_percent)" "$(fmt cert_min_det)" \
    "$(fmt proj_folds_end)" "$(fmt proj_iters)"
done
echo
echo "Read: DECOMP A2_projonly_e0 vs A2_twophase_e0 -> is TTO droppable for the guarantee?"
echo "      cert_min_det>=eps on every *_projonly/*_2ph row -> the margin holds (fold count hides it)."
echo "      A1_2ph_e0_it200 digital10 -> does the A1 blemish vanish at eps=0/200it? (budget vs failure)"
echo "      cert_min_det trend across HF/CROSS/DSET/DOMAIN feedfwd rows -> input-fold-density envelope."
echo "Eval CSV: $OUT_ROOT/ (send back). Runs on the --2 box; independent of Wave 1 on the H-box."
