#!/usr/bin/env bash
# Phase 7 solo/control runs on OASIS.
#
# Purpose:
#   1. Add native unsupervised LKU-32 solo, because the original LKU-Net OASIS
#      width was start_channel=32 but used supervised Dice loss.
#   2. Train L2-only controls under the shared CTCF loss for clean
#      cascade-vs-L2 comparisons in Paper 2.
#
# Usage:
#   conda activate ctcf
#   bash tools/phase7_solo_controls_oasis.sh
#
# Override defaults:
#   MAX_EPOCH=100 GPU=0 PATHS_PROFILE=--2 bash tools/phase7_solo_controls_oasis.sh
#
# Skip groups:
#   SKIP_NATIVE_LKU32=1 SKIP_MAMBA=1 SKIP_VMAMBA=1 bash tools/phase7_solo_controls_oasis.sh

set -e

GPU="${GPU:-0}"
MAX_EPOCH="${MAX_EPOCH:-100}"
PATHS_PROFILE="${PATHS_PROFILE:---2}"
PYBIN="${PYBIN:-python}"

SKIP_NATIVE_LKU32="${SKIP_NATIVE_LKU32:-0}"
SKIP_LKU8="${SKIP_LKU8:-0}"
SKIP_LKU32="${SKIP_LKU32:-0}"
SKIP_MAMBA="${SKIP_MAMBA:-0}"
SKIP_VMAMBA="${SKIP_VMAMBA:-0}"

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

if [ "${SKIP_LKU8}" != "1" ]; then
    run_ctcf "P7_CTRL_LKU8_L2ONLY_OASIS" --config CTCF-LKU8-solo
fi

if [ "${SKIP_LKU32}" != "1" ]; then
    run_ctcf "P7_CTRL_LKU32_L2ONLY_OASIS" --config CTCF-LKU32-solo
fi

if [ "${SKIP_MAMBA}" != "1" ]; then
    run_ctcf "P7_CTRL_MAMBA_L2ONLY_OASIS" --config CTCF-Mamba-solo
fi

if [ "${SKIP_VMAMBA}" != "1" ]; then
    run_ctcf "P7_CTRL_VMAMBA_L2ONLY_OASIS" --config CTCF-VMamba-solo
fi

if [ "${SKIP_NATIVE_LKU32}" != "1" ]; then
    run "P7_NATIVE_LKU32_OASIS" experiments.train_LKUNet --config LKU-32 --sim mse --w_reg 0.01
fi

echo "==================================================================="
echo "Phase 7 solo/control runs complete."
echo "Results in logs/P7_*/logfile.log and experiments/P7_*/ckpt/best.pth"
echo "==================================================================="
