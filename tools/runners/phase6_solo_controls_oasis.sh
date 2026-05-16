#!/usr/bin/env bash
# Phase 6 + Phase 7 — solo baselines and L2-only controls on OASIS, 100 epochs each.
#
# Merged from previous phase6_solo_oasis.sh + phase7_solo_controls_oasis.sh
# (now archived). Single runner with three groups:
#
#   1. SOLO (native loss per backbone, Phase 6 design):
#      - LKU-Net   (Jia et al., WBIR 2022)       — pure CNN, MSE + LKU smoothloss
#      - EffMorph  (Bin Aziz et al., WACV 2025)  — axial-plane transformer
#      - MambaMorph (Guo et al., 2024)           — SSM, NCC + Grad3d on velocity
#      - VMambaMorph (Wang et al., 2024)         — 4-direction Mamba cross-scan
#
#   2. CTCF L2-ONLY controls (CTCF training protocol with cascade disabled,
#      Phase 7 design): clean cascade-vs-L2 comparison.
#
#   3. NATIVE LKU-32 reference (Phase 7 design): LKU-Net at start_channel=32
#      with native MSE+smoothloss, the original LKU-Net OASIS configuration.
#
# Usage:
#   conda activate ctcf
#   bash tools/phase6_solo_controls_oasis.sh
#
# Override defaults:
#   MAX_EPOCH=100 GPU=0 PATHS_PROFILE=--2 bash tools/phase6_solo_controls_oasis.sh
#
# Skip groups:
#   SKIP_SOLO_*    — skip Phase 6 solo runs per backbone
#   SKIP_CTRL_*    — skip Phase 7 L2-only CTCF-protocol controls
#   SKIP_NATIVE_LKU32 — skip the LKU-32 native reference

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

# --- skip flags ----------------------------------------------------------
SKIP_SOLO_LKU="${SKIP_SOLO_LKU:-0}"
SKIP_SOLO_EFFM="${SKIP_SOLO_EFFM:-0}"
SKIP_SOLO_MAMBA="${SKIP_SOLO_MAMBA:-0}"
SKIP_SOLO_VMAMBA="${SKIP_SOLO_VMAMBA:-0}"

SKIP_CTRL_LKU8="${SKIP_CTRL_LKU8:-0}"
SKIP_CTRL_LKU32="${SKIP_CTRL_LKU32:-0}"
SKIP_CTRL_MAMBA="${SKIP_CTRL_MAMBA:-0}"
SKIP_CTRL_VMAMBA="${SKIP_CTRL_VMAMBA:-0}"

SKIP_NATIVE_LKU32="${SKIP_NATIVE_LKU32:-0}"

COMMON="--ds OASIS ${PATHS_PROFILE} --gpu ${GPU} --max_epoch ${MAX_EPOCH} --use_tb 1 --save_ckpt 1"
CTCF_COMMON="${COMMON} --w_ncc 1.0 --w_reg 1.0 --w_icon 0.05 --w_jac 0.005"

run() {
    local exp_name="$1"; shift
    echo "==================================================================="
    echo "> ${exp_name}"
    echo "==================================================================="
    "${PYBIN}" -m "$@" --exp "${exp_name}" ${COMMON}
}

run_ctcf() {
    local exp_name="$1"; shift
    echo "==================================================================="
    echo "> ${exp_name}"
    echo "==================================================================="
    "${PYBIN}" -m experiments.train_CTCF "$@" --exp "${exp_name}" ${CTCF_COMMON}
}

# ============================================================
# Group 1 — Solo Phase 6 (native loss)
# ============================================================

if [ "${SKIP_SOLO_LKU}" != "1" ]; then
    run "P6_LKU8_OASIS" experiments.train_LKUNet --config LKU-8 --sim mse --w_reg 0.01
    run "P6_LKU4_OASIS" experiments.train_LKUNet --config LKU-4 --sim mse --w_reg 0.01
fi

if [ "${SKIP_SOLO_EFFM}" != "1" ]; then
    run "P6_EFFM_2x3_2_HIRES_OASIS" experiments.train_EfficientMorph --config EfficientMorph_2x3_2_hires
    run "P6_EFFM_2x3_2_OASIS"       experiments.train_EfficientMorph --config EfficientMorph_2x3_2
fi

if [ "${SKIP_SOLO_MAMBA}" != "1" ]; then
    run "P6_MAMBA_DIFFEO_OASIS" experiments.train_MambaMorph --config MambaMorph --diffeo 1
fi

if [ "${SKIP_SOLO_VMAMBA}" != "1" ]; then
    run "P6_VMAMBA_OASIS" experiments.train_VMambaMorph --config VMambaMorph
fi

# ============================================================
# Group 2 — L2-only controls under CTCF protocol (Phase 7)
# ============================================================

if [ "${SKIP_CTRL_LKU8}" != "1" ]; then
    run_ctcf "P7_CTRL_LKU8_L2ONLY_OASIS" --config CTCF-LKU8-solo
fi

if [ "${SKIP_CTRL_LKU32}" != "1" ]; then
    run_ctcf "P7_CTRL_LKU32_L2ONLY_OASIS" --config CTCF-LKU32-solo
fi

if [ "${SKIP_CTRL_MAMBA}" != "1" ]; then
    run_ctcf "P7_CTRL_MAMBA_L2ONLY_OASIS" --config CTCF-Mamba-solo
fi

if [ "${SKIP_CTRL_VMAMBA}" != "1" ]; then
    run_ctcf "P7_CTRL_VMAMBA_L2ONLY_OASIS" --config CTCF-VMamba-solo
fi

# ============================================================
# Group 3 — Native LKU-32 reference (Phase 7)
# ============================================================

if [ "${SKIP_NATIVE_LKU32}" != "1" ]; then
    run "P7_NATIVE_LKU32_OASIS" experiments.train_LKUNet --config LKU-32 --sim mse --w_reg 0.01
fi

echo "==================================================================="
echo "Phase 6+7 solo/controls runs complete."
echo "Results in logs/{P6_*,P7_*}/logfile.log and results/{P6_*,P7_*}/ckpt/best.pth"
echo "==================================================================="
