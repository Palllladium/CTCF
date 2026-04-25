#!/usr/bin/env bash
# Phase 6 — solo baseline runs on OASIS (100 epochs each).
#
# Backbones (4 total):
#   - LKU-Net          (Jia et al., WBIR 2022)        — pure CNN, no extra deps
#   - EfficientMorph   (Bin Aziz et al., WACV 2025)   — axial-plane transformer, no extra deps
#   - MambaMorph       (Guo et al., 2024)             — single-direction Mamba SSM
#   - VMambaMorph      (Wang et al., 2024)            — 4-direction Mamba cross-scan
#
# Native loss per backbone (no supervised Dice anywhere; we are unsupervised):
#   - LKU-Net:        MSE + LKU axis-scaled smoothloss (lambda=0.01)
#   - EfficientMorph: NCC_gauss + Grad3d L2, weights all 1.0
#   - MambaMorph:     NCC + Grad3d L2 on velocity (preint_flow)
#   - VMambaMorph:    NCC + Grad3d L2 on velocity (preint_flow)
#
# Training infrastructure (Adam+amsgrad, lr=1e-4, polynomial LR decay, AMP fp16,
# single-direction forward) is uniform across backbones.
#
# Pre-flight on advisor's machine:
#   conda activate ctcf
#   bash tools/install_mamba.sh   # installs mamba_ssm + causal-conv1d + einops
#                                  # (one-time; ~10-15 min compile)
#
# Run all four backbones sequentially:
#   bash tools/phase6_solo_oasis.sh
#
# Override defaults:
#   MAX_EPOCH=200 GPU=0 PATHS_PROFILE=--2 bash tools/phase6_solo_oasis.sh
#
# Skip individual backbones (e.g. if mamba_ssm not yet installed):
#   SKIP_MAMBA=1 SKIP_VMAMBA=1 bash tools/phase6_solo_oasis.sh
#
# Outputs:
#   logs/<EXP_NAME>/logfile.log
#   experiments/<EXP_NAME>/ckpt/{last,best}.pth

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"
SKIP_LKU="${SKIP_LKU:-0}"
SKIP_EFFM="${SKIP_EFFM:-0}"
SKIP_MAMBA="${SKIP_MAMBA:-0}"
SKIP_VMAMBA="${SKIP_VMAMBA:-0}"

COMMON="--ds OASIS ${PATHS_PROFILE} --gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"

run() {
    local exp_name="$1"; shift
    echo "═══════════════════════════════════════════════════════════════════"
    echo "▶ ${exp_name}"
    echo "═══════════════════════════════════════════════════════════════════"
    "${PYBIN}" -m "$@" --exp "${exp_name}" ${COMMON}
}

# ── LKU-Net ────────────────────────────────────────────────────────────────
if [ "${SKIP_LKU}" != "1" ]; then
    run "P6_LKU8_OASIS" experiments.train_LKUNet --config LKU-8 --sim mse --w_reg 0.01
    run "P6_LKU4_OASIS" experiments.train_LKUNet --config LKU-4 --sim mse --w_reg 0.01
fi

# ── EfficientMorph ─────────────────────────────────────────────────────────
if [ "${SKIP_EFFM}" != "1" ]; then
    run "P6_EFFM_2x3_2_HIRES_OASIS" experiments.train_EfficientMorph --config EfficientMorph_2x3_2_hires
    run "P6_EFFM_2x3_2_OASIS"       experiments.train_EfficientMorph --config EfficientMorph_2x3_2
fi

# ── MambaMorph ─────────────────────────────────────────────────────────────
if [ "${SKIP_MAMBA}" != "1" ]; then
    run "P6_MAMBA_DIFFEO_OASIS"     experiments.train_MambaMorph --config MambaMorph --diffeo 1
fi

# ── VMambaMorph ────────────────────────────────────────────────────────────
if [ "${SKIP_VMAMBA}" != "1" ]; then
    run "P6_VMAMBA_OASIS"           experiments.train_VMambaMorph --config VMambaMorph
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "✓ Phase 6 solo runs complete."
echo "  Results in: logs/P6_*/logfile.log and experiments/P6_*/ckpt/best.pth"
echo "═══════════════════════════════════════════════════════════════════"
