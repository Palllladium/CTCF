#!/usr/bin/env bash
# Phase 11 — batch 2: finish the SACB baseline + M3 cascade-aware reg grid.
#
# Context (batch1 results, 100ep OASIS, anchor = 0.8274 = P7 Mamba SVF K=1):
#   Track M correspondence works and the cheapest form suffices:
#     HADAMARD 0.8288 (+0.0014, ~0 params) == CORR27 0.8288 (27ch buys nothing).
#     L3CH64   0.8296 (+0.0022, brute capacity). M1 multi-head dead (MH4 = anchor).
#   Track V on OUR protocol collapses the claimed +4-6 gap:
#     CorrMLP OASIS 0.8275 ~= our 0.8274 (but folds 0.78%, SDlogJ 0.103 vs our 0.00/0.077).
#     CorrMLP IXI 0.7443 < our ~0.763.  SACB CRASHED @ep0 (fp16/kmeans bug, now fixed).
#
# This batch closes the two items the screening still needs:
#   V. SACB-Net OASIS + IXI  — re-run after the AMP fix in experiments/train_SACB.py
#      (train_step now forces autocast(enabled=False); kmeans_gpu needs fp32).
#   D. M3 cascade-aware per-level diffusion reg grid on the Mamba SVF base (never run yet).
#      FLAT (1,1,1) is the sanity baseline: per-level decomposition with neutral weights
#      should reproduce the uniform-reg anchor (~0.8274), proving the split itself is neutral;
#      STRONG/VSTRONG/MID then test whether re-weighting L1 up / L3 down actually helps.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase11_batch2.sh
# Override (env vars):
#   GPU=0 MAX_EPOCH=100 PATHS_PROFILE=--2 bash tools/runners/phase11_batch2.sh
# Staged skips:
#   SKIP_V_SACB_OASIS=1 SKIP_V_SACB_IXI=1                       (Track V SACB baselines)
#   SKIP_D_FLAT=1 SKIP_D_STRONG=1 SKIP_D_VSTRONG=1 SKIP_D_MID=1 (M3 reg grid)
#
# Deps (pip-install into the advisor's `ctcf` conda env):
#   SACB needs `einops kmeans_gpu timm monai pystrum scipy`.

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_V_SACB_OASIS="${SKIP_V_SACB_OASIS:-0}"
SKIP_V_SACB_IXI="${SKIP_V_SACB_IXI:-0}"
SKIP_D_FLAT="${SKIP_D_FLAT:-0}"
SKIP_D_STRONG="${SKIP_D_STRONG:-0}"
SKIP_D_VSTRONG="${SKIP_D_VSTRONG:-0}"
SKIP_D_MID="${SKIP_D_MID:-0}"

COMMON="--gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"

# CTCF M3 base (Mamba SVF cascade, OASIS, loss identical to batch1)
CTCF_BASE="${COMMON} --w_ncc 1.0 --w_icon 0.05 --w_jac 0.005"
CTCF_OASIS="--ds OASIS ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 1.0"
MAMBA_SVF="--config CTCF-CascadeA-Mamba --l3_svf 1"

run_ctcf() {
    local exp_name="$1"; shift
    echo "> ${exp_name}"
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}"
}

run_sacb() {
    local exp_name="$1"; shift
    echo "> ${exp_name} (SACB-Net baseline)"
    "${PYBIN}" -m experiments.train_SACB "$@" --exp "${exp_name}"
}


# Track V — SACB-Net baseline on OUR split (loss = NCC9 + diffusion, w=[1, 0.3], SACB native).
#   Voxel-units flow -> runs through our Runner/val directly. AMP fix applied (fp32 train_step).
#   Confirms its IXI 0.769 vs our 0.7635, and OASIS vs our 0.8274.
if [ "${SKIP_V_SACB_OASIS}" != "1" ]; then
    run_sacb "P11_SACB_OASIS" \
        --ds OASIS ${PATHS_PROFILE} ${COMMON} \
        --w_ncc 1.0 --w_reg 0.3
fi

if [ "${SKIP_V_SACB_IXI}" != "1" ]; then
    run_sacb "P11_SACB_IXI" \
        --ds IXI ${PATHS_PROFILE} ${COMMON} \
        --w_ncc 1.0 --w_reg 0.3
fi


# Block D — M3 cascade-aware per-level reg grid (Mamba SVF, OASIS, 100ep).
#   --w_reg_l1/l2/l3 replace the uniform --w_reg with per-level diffusion weights on
#   {phi_l1, phi_l2_residual, delta_l3}. Compare each vs anchor 0.8274.
#     FLAT    (1, 1, 1)   -> sanity: should reproduce the uniform-reg anchor.
#     STRONG  (3, 1, 0.3) -> heavier coarse L1, lighter L3 residual.
#     VSTRONG (5, 1, 0.1) -> push the same direction harder.
#     MID     (2, 1, 0.5) -> moderate version.
if [ "${SKIP_D_FLAT}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_REG_FLAT_OASIS" \
        ${MAMBA_SVF} ${CTCF_OASIS} \
        --w_reg_l1 1.0 --w_reg_l2 1.0 --w_reg_l3 1.0
fi

if [ "${SKIP_D_STRONG}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_REG_STRONG_OASIS" \
        ${MAMBA_SVF} ${CTCF_OASIS} \
        --w_reg_l1 3.0 --w_reg_l2 1.0 --w_reg_l3 0.3
fi

if [ "${SKIP_D_VSTRONG}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_REG_VSTRONG_OASIS" \
        ${MAMBA_SVF} ${CTCF_OASIS} \
        --w_reg_l1 5.0 --w_reg_l2 1.0 --w_reg_l3 0.1
fi

if [ "${SKIP_D_MID}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_REG_MID_OASIS" \
        ${MAMBA_SVF} ${CTCF_OASIS} \
        --w_reg_l1 2.0 --w_reg_l2 1.0 --w_reg_l3 0.5
fi


echo ""
echo "Phase 11 batch 2 complete (or skipped per env flags)."
echo ""
echo "Next:"
echo "  1. Confirm SACB val Dice > 0.7 (sanity) on OASIS+IXI; compare vs our 0.8274 / 0.7635."
echo "  2. Rank the M3 grid vs anchor 0.8274; FLAT should ~= anchor (decomposition is neutral)."
echo "  3. If any M3 weighting beats the anchor, fold it into the locked config alongside"
echo "     Hadamard correspondence before the (single) 500ep longrun."
