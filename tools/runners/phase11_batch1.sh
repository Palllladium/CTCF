#!/usr/bin/env bash
# Phase 11 — batch 1: explicit-correspondence mechanism in CTCF + competitor baselines.
#
# Replaces the deprioritised batch0 tail (M1 multi-head was empirically dead: K=2 +0.0011,
# K=4 ~0 vs the 100ep anchor 0.8274 = logs/SEDM/P7_CASCADES/P7_CASC_MAMBA_SVF_OASIS).
#
# Two tracks (see memory/competitor_landscape_2026.md, memory/phase11_status.md):
#   M. Correspondence cost-volume injected into L3 (FlowRefiner3D), on the Mamba SVF base:
#        --l3_corr_mode hadamard  (EOIR [F_m+F_f, F_m-F_f], ~0 params)   <- cheapest, run first
#        --l3_corr_mode corr      (CorrMLP-style local 3D correlation, 27ch, ~0 params)
#        --l3_corr_mode corr_feat (correlation on a shallow feature embedding, +1 conv)
#   V. Competitor baselines on OUR split to verify the +4-6 unsupervised gap:
#        CorrMLP (GPL, our harness, loss = ours)   -> experiments.train_CorrMLP
#        SACB-Net (our harness, voxel-units flow)  -> experiments.train_SACB
#
# Anchor for all CTCF 100ep OASIS runs: 0.8274 (P7 Mamba SVF K=1). NOT the 500ep 0.8314.
#
# Usage:
#   conda activate ctcf
#   bash tools/runners/phase11_batch1.sh
# Override (env vars):
#   GPU=0 MAX_EPOCH=100 PATHS_PROFILE=--2 bash tools/runners/phase11_batch1.sh
# Staged skips:
#   SKIP_M_HADAMARD=1 SKIP_M_CORR=1 SKIP_M_CORRFEAT=1   (Track M correspondence runs)
#   SKIP_V_CORRMLP_OASIS=1 SKIP_V_CORRMLP_IXI=1         (Track V CorrMLP baselines)
#   SKIP_V_SACB_OASIS=1 SKIP_V_SACB_IXI=1               (Track V SACB baselines)
#
# Deps (pip-install into the advisor's `ctcf` conda env): CorrMLP needs `einops`;
#   SACB needs `einops kmeans_gpu timm monai pystrum scipy`.

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_M_HADAMARD="${SKIP_M_HADAMARD:-0}"
SKIP_M_CORR="${SKIP_M_CORR:-0}"
SKIP_M_CORRFEAT="${SKIP_M_CORRFEAT:-1}"          # off by default: run only if hadamard/corr show life
SKIP_V_CORRMLP_OASIS="${SKIP_V_CORRMLP_OASIS:-0}"
SKIP_V_CORRMLP_IXI="${SKIP_V_CORRMLP_IXI:-0}"
SKIP_V_SACB_OASIS="${SKIP_V_SACB_OASIS:-0}"
SKIP_V_SACB_IXI="${SKIP_V_SACB_IXI:-0}"

COMMON="--gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"

# CTCF Track M base (Mamba SVF cascade, OASIS, loss identical to batch0)
CTCF_BASE="${COMMON} --w_ncc 1.0 --w_icon 0.05 --w_jac 0.005"
CTCF_OASIS="--ds OASIS ${PATHS_PROFILE} ${CTCF_BASE} --w_reg 1.0"
MAMBA_SVF="--config CTCF-CascadeA-Mamba --l3_svf 1"

run_ctcf() {
    local exp_name="$1"; shift
    echo "> ${exp_name}"
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}"
}

run_corrmlp() {
    local exp_name="$1"; shift
    echo "> ${exp_name} (CorrMLP baseline)"
    "${PYBIN}" -m experiments.train_CorrMLP "$@" --exp "${exp_name}"
}

run_sacb() {
    local exp_name="$1"; shift
    echo "> ${exp_name} (SACB-Net baseline)"
    "${PYBIN}" -m experiments.train_SACB "$@" --exp "${exp_name}"
}


# Track M — correspondence cost-volume in L3 (Mamba SVF, OASIS, 100ep)
#   Compare each vs anchor 0.8274. Hadamard is the headline cheap test.
if [ "${SKIP_M_HADAMARD}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_HADAMARD_OASIS" \
        ${MAMBA_SVF} --l3_corr_mode hadamard \
        ${CTCF_OASIS}
fi

if [ "${SKIP_M_CORR}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_CORR27_OASIS" \
        ${MAMBA_SVF} --l3_corr_mode corr \
        ${CTCF_OASIS}
fi

if [ "${SKIP_M_CORRFEAT}" != "1" ]; then
    run_ctcf "P11_MAMBA_SVF_CORRFEAT_OASIS" \
        ${MAMBA_SVF} --l3_corr_mode corr_feat \
        ${CTCF_OASIS}
fi


# Track V — CorrMLP baseline on OUR split (loss = NCC9 + diffusion, w=[1,1], Adam 1e-4).
#   Verifies the ~0.871 OASIS / ~0.769 IXI unsupervised claim on our data.
#   GPL upstream (models/CorrMLP/networks.py) -> standalone baseline only, never merged into CTCF.
if [ "${SKIP_V_CORRMLP_OASIS}" != "1" ]; then
    run_corrmlp "P11_CORRMLP_OASIS" \
        --ds OASIS ${PATHS_PROFILE} ${COMMON} \
        --w_ncc 1.0 --w_reg 1.0
fi

if [ "${SKIP_V_CORRMLP_IXI}" != "1" ]; then
    run_corrmlp "P11_CORRMLP_IXI" \
        --ds IXI ${PATHS_PROFILE} ${COMMON} \
        --w_ncc 1.0 --w_reg 1.0
fi


# Track V — SACB-Net baseline on OUR split (loss = NCC9 + diffusion, w=[1, 0.3], SACB native).
#   Voxel-units flow -> runs through our Runner/val directly. Needs CUDA + SACB deps
#   (einops kmeans_gpu timm monai pystrum scipy). Confirms IXI 0.769 vs our 0.7635.
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


echo ""
echo "Phase 11 batch 1 complete (or skipped per env flags)."
echo ""
echo "Next:"
echo "  1. Inference + aggregate the P11_* ckpts (reuse phase10_inference pattern)."
echo "  2. Rank Track-M corr modes vs anchor 0.8274; pick the winner for a 500ep longrun."
echo "  3. Compare CorrMLP/SACB baselines vs CTCF to confirm the unsupervised gap."
